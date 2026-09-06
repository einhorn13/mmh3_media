from __future__ import annotations

from .node_support import (
    CATEGORY,
    MMH3,
    MMH3ResourceError,
    Path,
    _packet,
    _safe_output_prefix,
    build_fastvae_preview,
    compare_packets,
    export_packet,
    folder_paths,
    io,
    ui,
    json,
    preview_is_fresh,
)
from .preview import materialize_preview_image, preview_representation, resource_summary
from .resource_model import CORE_RESOURCE_ROLES


def _typed_resource(packet, kind: str, role: str, order: int | None, resource_id: str):
    if resource_id:
        resource = packet.get_by_id(resource_id)
        if resource is not None and resource["kind"] != kind:
            raise MMH3ResourceError(f"Resource {resource_id} is kind={resource['kind']!r}, expected {kind!r}")
        return resource
    if kind != "json" and role == "auxiliary" and order is None:
        primary = packet.get_primary(kind)
        if primary is not None:
            return primary
    matches = [r for r in packet.resources()
               if r["kind"] == kind and r["role"] == role and r.get("order") == order]
    if len(matches) > 1:
        raise MMH3ResourceError(f"Multiple {kind} resources match {role}@{order}; select a resource_id")
    return matches[0] if matches else None

class MMH3Compare(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Compare",
            display_name="MMH3 Compare",
            category=CATEGORY,
            description="Semantic packet comparison. Preview cache is ignored by default.",
            inputs=[
                MMH3.Input("packet_a"),
                MMH3.Input("packet_b"),
                io.Boolean.Input("include_preview_cache", default=False, advanced=True),
            ],
            outputs=[io.Boolean.Output("equal"), io.String.Output("summary"), io.String.Output("diff_json")],
        )

    @classmethod
    def execute(cls, packet_a, packet_b, include_preview_cache: bool = False) -> io.NodeOutput:
        result = compare_packets(_packet(packet_a), _packet(packet_b), include_preview=bool(include_preview_cache))
        return io.NodeOutput(bool(result["equal"]), result["summary"], json.dumps(result, ensure_ascii=False, indent=2))


class MMH3Export(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Export",
            display_name="MMH3 Export",
            category=CATEGORY,
            description="Export packet.json and resources into a normal directory tree for interoperability/debugging.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("directory", default="mmh3_export/MMH3"),
                io.Boolean.Input("include_preview_cache", default=True, advanced=True),
            ],
            outputs=[io.String.Output("path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, packet, directory: str, include_preview_cache: bool = True) -> io.NodeOutput:
        rel = _safe_output_prefix(directory)
        root = Path(folder_paths.get_output_directory()).resolve()
        dest = (root / rel).resolve()
        try:
            dest.relative_to(root)
        except ValueError as e:
            raise MMH3ResourceError("Export directory escapes ComfyUI output directory") from e
        return io.NodeOutput(export_packet(_packet(packet), dest, include_preview=bool(include_preview_cache)))


class MMH3Preview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Preview",
            display_name="MMH3 Preview",
            category=CATEGORY,
            description=(
                "Explicitly reuse or generate a disposable representation of the primary H3 latent. "
                "Representations live outside resources[] and never become primary/resource semantics."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Vae.Input("fast_vae", optional=True, tooltip="Used only for explicit contact-sheet generation when no fresh representation is reused."),
                io.Combo.Input("mode", options=["if_missing_or_stale", "refresh", "cache_only"], default="if_missing_or_stale"),
                io.Int.Input("frames", default=8, min=1, max=16),
                io.Int.Input("max_size", default=768, min=128, max=2048, step=64),
            ],
            outputs=[MMH3.Output("packet"), io.Image.Output("preview"), io.Boolean.Output("regenerated")],
        )

    @classmethod
    def execute(cls, packet, mode: str, frames: int, max_size: int, fast_vae=None) -> io.NodeOutput:
        packet = _packet(packet)
        existing = preview_representation(packet)
        fresh = preview_is_fresh(packet)
        if existing is not None and mode != "refresh" and (mode == "cache_only" or fresh):
            image = materialize_preview_image(packet)
            target = existing["target"]
            summary = resource_summary(packet, target) if target != "$packet" else str(packet.manifest.get("name") or "MMH3 packet")
            return io.NodeOutput(packet, image, False, ui=ui.PreviewText(summary))
        if mode == "cache_only":
            raise MMH3ResourceError("No fresh image representation is stored in this packet")
        out, image = build_fastvae_preview(packet, fast_vae, frames=int(frames), max_size=int(max_size))
        target = out.primary("latent").resource_id
        return io.NodeOutput(out, image, True, ui=ui.PreviewText(resource_summary(out, target)))


class _MMH3GetBase(io.ComfyNode):
    KIND: str = ""
    OUTPUT_TYPE = None
    NODE_ID = ""
    DISPLAY_NAME = ""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=cls.NODE_ID, display_name=cls.DISPLAY_NAME, category=CATEGORY,
            description=f"Materialize one canonical {cls.KIND.upper()} resource by ID or generic role/order. Empty auxiliary/null selection resolves the explicit primary binding first.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("role", options=list(CORE_RESOURCE_ROLES), default="auxiliary"),
                io.Int.Input("order", default=-1, min=-1, max=9999, tooltip="-1 selects resources whose order is null."),
                io.String.Input("resource_id", default="", advanced=True, tooltip="Stable selector; overrides role/order when set."),
                io.Combo.Input("missing", options=["error", "none"], default="error", advanced=True),
            ],
            outputs=[
                cls.OUTPUT_TYPE.Output("resource"), io.Boolean.Output("exists"),
                io.String.Output("resource_id"), io.String.Output("descriptor_json"),
            ],
        )

    @classmethod
    def execute(cls, packet, role: str, order: int, resource_id: str, missing: str) -> io.NodeOutput:
        packet = _packet(packet)
        resource_id = resource_id.strip()
        requested_order = None if int(order) < 0 else int(order)
        res = _typed_resource(packet, cls.KIND, role, requested_order, resource_id)
        if res is None:
            if missing == "none":
                return io.NodeOutput(None, False, "", "{}")
            label = resource_id.strip() or f"{role}@{None if int(order) < 0 else int(order)}"
            raise MMH3ResourceError(f"Resource {label} does not exist")
        payload = packet.ref(res["id"]).materialize()
        return io.NodeOutput(payload, True, res["id"], json.dumps(res, ensure_ascii=False, indent=2))


class MMH3GetLatent(_MMH3GetBase):
    KIND = "latent"; OUTPUT_TYPE = io.Latent; NODE_ID = "MMH3GetLatent"; DISPLAY_NAME = "MMH3 Get Latent"


class MMH3GetImage(_MMH3GetBase):
    KIND = "image"; OUTPUT_TYPE = io.Image; NODE_ID = "MMH3GetImage"; DISPLAY_NAME = "MMH3 Get Image"


class MMH3GetVideo(_MMH3GetBase):
    KIND = "video"; OUTPUT_TYPE = io.Video; NODE_ID = "MMH3GetVideo"; DISPLAY_NAME = "MMH3 Get Video"


class MMH3GetAudio(_MMH3GetBase):
    KIND = "audio"; OUTPUT_TYPE = io.Audio; NODE_ID = "MMH3GetAudio"; DISPLAY_NAME = "MMH3 Get Audio"


class MMH3GetMask(_MMH3GetBase):
    KIND = "mask"; OUTPUT_TYPE = io.Mask; NODE_ID = "MMH3GetMask"; DISPLAY_NAME = "MMH3 Get Mask"


class MMH3GetJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3GetJSON", display_name="MMH3 Get JSON", category=CATEGORY,
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("role", options=list(CORE_RESOURCE_ROLES), default="auxiliary"),
                io.Int.Input("order", default=-1, min=-1, max=9999),
                io.String.Input("resource_id", default="", advanced=True),
                io.Combo.Input("missing", options=["error", "none"], default="error", advanced=True),
            ],
            outputs=[io.String.Output("json"), io.Boolean.Output("exists"), io.String.Output("resource_id"), io.String.Output("descriptor_json")],
        )

    @classmethod
    def execute(cls, packet, role: str, order: int, resource_id: str, missing: str) -> io.NodeOutput:
        packet = _packet(packet)
        res = _typed_resource(packet, "json", role, None if int(order) < 0 else int(order), resource_id.strip())
        if res is None:
            if missing == "none":
                return io.NodeOutput("", False, "", "{}")
            raise MMH3ResourceError(f"JSON resource {resource_id or role} does not exist")
        payload = packet.ref(res["id"]).materialize()
        return io.NodeOutput(json.dumps(payload, ensure_ascii=False, indent=2), True, res["id"], json.dumps(res, ensure_ascii=False, indent=2))
