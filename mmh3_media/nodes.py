from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import folder_paths  # type: ignore
from aiohttp import web  # type: ignore
from server import PromptServer  # type: ignore
from comfy_api.latest import ComfyExtension, io  # type: ignore
from typing_extensions import override

from .archive import get_resource_payload, load_archive, save_archive
from .constants import H3_LATENT_ORIGINS, ORDERED_ROLES
from .core import MMH3Media
from .errors import MMH3ResourceError
from .h3 import (
    concat_h3_av_latent,
    h3_metadata_set_origin,
    h3_metadata_with_context,
    h3_pair_compatibility,
    split_h3_av_latent,
    validate_h3_av_latent,
)
from .inspection import inspect_packet
from .compare import compare_packets
from .export import export_packet
from .preview import build_fastvae_preview, preview_is_fresh
from .util import deep_copy_json
from .serializers import infer_kind

MMH3 = io.Custom("MMH3_MEDIA")
CATEGORY = "MiniMax H3/MMH3 Media"

_ROLE_OPTIONS = [
    "h3_av_latent",
    "video",
    "audio",
    "first_frame",
    "last_frame",
    "picture_ref",
    "video_ref",
    "audio_ref",
    "preview",
    "mask",
    "continuation_context",
    "custom",
]


def _packet(value: Any) -> MMH3Media:
    if not isinstance(value, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    return value


def _parse_object(text: str, label: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise MMH3ResourceError(f"{label} is not valid JSON: {e}") from e
    if not isinstance(value, dict):
        raise MMH3ResourceError(f"{label} must be a JSON object")
    return value


def _resolve_role(role: str, custom_role: str = "") -> str:
    if role == "custom" and custom_role.strip():
        return custom_role.strip()
    return role


def _iter_mmh3_files(root: str, prefix: str) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    out: list[str] = []
    try:
        for p in base.rglob("*.mmh3"):
            if p.is_file():
                out.append(f"{prefix}::{p.relative_to(base).as_posix()}")
    except OSError:
        pass
    return out


def _load_options() -> list[str]:
    values = _iter_mmh3_files(folder_paths.get_input_directory(), "input")
    values.extend(_iter_mmh3_files(folder_paths.get_output_directory(), "output"))
    values.sort(key=str.casefold)
    return values or ["(none)"]


@PromptServer.instance.routes.get("/mmh3_media/files")
async def _mmh3_media_files_route(_request):
    # Remote COMBO source: rescans on refresh, so newly saved packets appear without a ComfyUI restart.
    return web.json_response(_load_options())


def _resolve_load_path(file: str, path_override: str) -> Path:
    if path_override and path_override.strip():
        return Path(path_override).expanduser().resolve()
    if not file or file == "(none)":
        raise MMH3ResourceError("No .mmh3 file selected")
    try:
        area, rel = file.split("::", 1)
    except ValueError as e:
        raise MMH3ResourceError(f"Invalid MMH3 file selector: {file!r}") from e
    root = {
        "input": Path(folder_paths.get_input_directory()),
        "output": Path(folder_paths.get_output_directory()),
    }.get(area)
    if root is None:
        raise MMH3ResourceError(f"Unknown MMH3 file area {area!r}")
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as e:
        raise MMH3ResourceError("MMH3 file selector escapes its ComfyUI directory") from e
    return candidate


def _safe_output_prefix(prefix: str) -> Path:
    raw = (prefix or "mmh3/MMH3").strip().replace("\\", "/")
    p = Path(raw)
    if p.is_absolute() or any(part in ("", ".", "..") or ":" in part for part in p.parts):
        raise MMH3ResourceError("filename_prefix must be a safe portable relative path without '..' or ':'")
    if p.suffix.lower() == ".mmh3":
        p = p.with_suffix("")
    return p


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    for i in range(1, 100000):
        candidate = path.with_name(f"{stem}_{i:05d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise MMH3ResourceError(f"Could not allocate a unique filename near {path}")


def _resolve_save_path(packet: MMH3Media, filename_prefix: str, target: str, overwrite: bool) -> Path:
    if target == "source (in-place)":
        if not packet.source_archive:
            raise MMH3ResourceError("In-place save requires a packet loaded/saved from a .mmh3 file")
        return Path(packet.source_archive).resolve()
    rel = _safe_output_prefix(filename_prefix)
    path = (Path(folder_paths.get_output_directory()) / rel).with_suffix(".mmh3").resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    try:
        path.relative_to(output_root)
    except ValueError as e:
        raise MMH3ResourceError("Save path escapes the ComfyUI output directory") from e
    if path.exists() and not overwrite:
        path = _unique_path(path)
    return path


class MMH3Create(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Create",
            display_name="MMH3 Create",
            category=CATEGORY,
            description="Create an in-memory MMH3 media packet. Nothing is written to disk.",
            inputs=[
                io.String.Input("name", default="untitled"),
                io.String.Input("notes", default="", multiline=True, tooltip="Free-form creator/user notes stored directly in packet.json."),
                io.String.Input("task", default="", advanced=True),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=False, advanced=True),
                io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, advanced=True),
                io.Combo.Input(
                    "latent_origin",
                    options=list(H3_LATENT_ORIGINS),
                    default="unknown",
                    advanced=True,
                    tooltip="Provenance of the optional H3 latent. Use sampler_output only for a direct H3 sampler result; tensor shape alone cannot prove provenance.",
                ),
                io.Latent.Input("latent", optional=True),
                io.Video.Input("video", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[MMH3.Output("packet")],
        )

    @classmethod
    def execute(
        cls,
        name: str,
        notes: str,
        task: str,
        prompt: str,
        seed: int,
        latent_origin: str = "unknown",
        latent=None,
        video=None,
        audio=None,
        first_frame=None,
        last_frame=None,
    ) -> io.NodeOutput:
        generation: dict[str, Any] = {}
        if task.strip():
            generation["task"] = task.strip()
        if prompt:
            generation["prompt"] = prompt
        if seed >= 0:
            generation["seed"] = int(seed)
        packet = MMH3Media.create(name=name.strip() or "untitled", generation=generation)
        if notes:
            packet = packet.edit_metadata(notes=notes)

        if latent is not None:
            info = validate_h3_av_latent(latent)
            metadata = {"h3": h3_metadata_with_context(info, origin=latent_origin)}
            packet = packet.put(
                latent, kind="latent", role="h3_av_latent", metadata=metadata, record_history=False
            )
            gen = dict(packet.manifest.get("generation", {}))
            gen.update(
                {
                    "width": info.width,
                    "height": info.height,
                    "frames": info.frames,
                    "fps": info.fps,
                    "audio_sample_rate": info.audio_sample_rate,
                    "audio_latent_rate": info.audio_latent_rate,
                }
            )
            # Preserve the clean COW contract without exposing a separate internal mutator.
            manifest = deep_copy_json(packet.manifest)
            manifest["generation"] = gen
            packet = MMH3Media(manifest, packet.source_archive, packet.payloads, packet.origins, packet.verify_mode, True)
        if video is not None:
            packet = packet.put(video, kind="video", role="video", record_history=False)
        if audio is not None:
            packet = packet.put(audio, kind="audio", role="audio", record_history=False)
        if first_frame is not None:
            packet = packet.put(first_frame, kind="image", role="first_frame", record_history=False)
        if last_frame is not None:
            packet = packet.put(last_frame, kind="image", role="last_frame", record_history=False)
        return io.NodeOutput(packet)


class MMH3Load(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Load",
            display_name="MMH3 Load",
            category=CATEGORY,
            description="Load packet.json and the ZIP directory only. Heavy media/latents remain lazy.",
            inputs=[
                io.Combo.Input(
                    "file",
                    options=_load_options(),
                    remote=io.RemoteOptions(
                        route="/mmh3_media/files",
                        refresh_button=False,
                        control_after_refresh="first",
                    ),
                    tooltip="Use the prominent refresh button at the bottom of this node to rescan ComfyUI input/output for .mmh3 files.",
                ),
                io.Combo.Input("verify", options=["on_access", "manifest", "full"], default="on_access", advanced=True),
                io.String.Input(
                    "path_override",
                    default="",
                    advanced=True,
                    tooltip="Optional absolute/local path for automation. The file selector is ignored when set.",
                ),
            ],
            outputs=[MMH3.Output("packet")],
        )

    @classmethod
    def execute(cls, file: str, verify: str, path_override: str) -> io.NodeOutput:
        path = _resolve_load_path(file, path_override)
        return io.NodeOutput(load_archive(path, verify=verify))

    @classmethod
    def fingerprint_inputs(cls, file: str, verify: str, path_override: str):
        try:
            path = _resolve_load_path(file, path_override)
            stat = path.stat()
            return (str(path), stat.st_mtime_ns, stat.st_size, verify)
        except Exception:
            return (file, path_override, verify, "missing")


class MMH3Save(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Save",
            display_name="MMH3 Save",
            category=CATEGORY,
            description="Atomically serialize a self-contained .mmh3 archive. Existing media are byte-copied when possible.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("filename_prefix", default="mmh3/MMH3"),
                io.Combo.Input(
                    "target",
                    options=["output", "source (in-place)"],
                    default="output",
                    advanced=True,
                ),
                io.Boolean.Input("overwrite", default=False, advanced=True),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, packet, filename_prefix: str, target: str, overwrite: bool) -> io.NodeOutput:
        packet = _packet(packet)
        path = _resolve_save_path(packet, filename_prefix, target, overwrite)
        saved, saved_path = save_archive(packet, path)
        return io.NodeOutput(saved, saved_path)


class MMH3Put(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Put",
            display_name="MMH3 Put",
            category=CATEGORY,
            description="Add or replace one typed media resource. Replace/upsert preserves resource_id.",
            inputs=[
                MMH3.Input("packet"),
                io.MultiType.Input("resource", types=[io.Latent, io.Image, io.Video, io.Audio, io.Mask]),
                io.Combo.Input("role", options=_ROLE_OPTIONS, default="custom"),
                io.Int.Input("slot", default=0, min=0, max=9999),
                io.Combo.Input("mode", options=["upsert", "add", "replace"], default="upsert"),
                io.String.Input("custom_role", default="", advanced=True),
                io.String.Input("metadata_json", default="", multiline=True, advanced=True),
                io.Combo.Input(
                    "h3_latent_origin",
                    options=["preserve", *H3_LATENT_ORIGINS],
                    default="preserve",
                    advanced=True,
                    tooltip="Only for role=h3_av_latent. preserve keeps existing provenance on replace; a new latent becomes unknown.",
                ),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("resource_id")],
        )

    @classmethod
    def execute(
        cls,
        packet,
        resource,
        role: str,
        slot: int,
        mode: str,
        custom_role: str,
        metadata_json: str,
        h3_latent_origin: str = "preserve",
    ) -> io.NodeOutput:
        packet = _packet(packet)
        resolved_role = _resolve_role(role, custom_role)
        kind = infer_kind(resource)
        metadata = _parse_object(metadata_json, "metadata_json")
        if resolved_role == "h3_av_latent":
            info = validate_h3_av_latent(resource)
            metadata = dict(metadata)
            existing = packet.get_by_role_slot(resolved_role, int(slot))
            existing_h3: dict[str, Any] = {}
            if existing is not None and isinstance(existing.get("metadata"), dict):
                maybe_h3 = existing["metadata"].get("h3")
                if isinstance(maybe_h3, dict):
                    existing_h3.update(maybe_h3)
            supplied_h3 = metadata.get("h3")
            if isinstance(supplied_h3, dict):
                existing_h3.update(supplied_h3)
            origin = None if h3_latent_origin == "preserve" else h3_latent_origin
            metadata["h3"] = h3_metadata_with_context(info, existing_h3=existing_h3, origin=origin)
        elif h3_latent_origin != "preserve":
            raise MMH3ResourceError("h3_latent_origin is only meaningful for role=h3_av_latent")
        out, resource_id = packet.put_with_id(
            resource, kind=kind, role=resolved_role, slot=int(slot), mode=mode, metadata=metadata
        )
        return io.NodeOutput(out, resource_id)


class MMH3Remove(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Remove",
            display_name="MMH3 Remove",
            category=CATEGORY,
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("role", options=_ROLE_OPTIONS, default="custom"),
                io.Int.Input("slot", default=0, min=0, max=9999),
                io.String.Input("resource_id", default="", advanced=True, tooltip="If set, overrides role/slot."),
                io.String.Input("custom_role", default="", advanced=True),
                io.Combo.Input("missing", options=["error", "ignore"], default="error", advanced=True),
            ],
            outputs=[MMH3.Output("packet")],
        )

    @classmethod
    def execute(cls, packet, role: str, slot: int, resource_id: str, custom_role: str, missing: str) -> io.NodeOutput:
        packet = _packet(packet)
        role = _resolve_role(role, custom_role)
        return io.NodeOutput(packet.remove(role=role, slot=int(slot), resource_id=resource_id.strip(), missing=missing))


class MMH3Move(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = [x for x in ["picture_ref", "video_ref", "audio_ref", "mask", "continuation_context"] if x in ORDERED_ROLES]
        return io.Schema(
            node_id="MMH3Move",
            display_name="MMH3 Move",
            category=CATEGORY,
            description="Reorder an ordered role. Slots are compact and resource_id remains stable.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("role", options=options, default="picture_ref"),
                io.Int.Input("from_slot", default=0, min=0, max=9999),
                io.Int.Input("to_slot", default=0, min=0, max=9999),
            ],
            outputs=[MMH3.Output("packet")],
        )

    @classmethod
    def execute(cls, packet, role: str, from_slot: int, to_slot: int) -> io.NodeOutput:
        return io.NodeOutput(_packet(packet).move(role=role, from_slot=int(from_slot), to_slot=int(to_slot)))


class MMH3Metadata(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Metadata",
            display_name="MMH3 Metadata",
            category=CATEGORY,
            description="Edit packet metadata only. Resource descriptors and history are protected from JSON patching.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("name", default="", tooltip="Blank keeps the current value."),
                io.String.Input("task", default="", advanced=True),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=False, advanced=True),
                io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, advanced=True),
                io.String.Input("tags", default="", advanced=True, tooltip="Comma-separated; blank keeps current tags."),
                io.Combo.Input("notes_action", options=["keep", "set", "clear"], default="keep"),
                io.String.Input("notes", default="", multiline=True, tooltip="Creator/user notes. Applied only when notes_action=set."),
                io.String.Input(
                    "custom_json_merge_patch",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip="RFC 7396-style merge patch for non-reserved packet metadata. null removes a field. Applied after the dedicated fields above, so patch values win on conflicts.",
                ),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("notes")],
        )

    @classmethod
    def execute(cls, packet, name: str, task: str, prompt: str, seed: int, tags: str, notes_action: str, notes: str, custom_json_merge_patch: str) -> io.NodeOutput:
        patch = _parse_object(custom_json_merge_patch, "custom_json_merge_patch") if custom_json_merge_patch.strip() else None
        out = _packet(packet).edit_metadata(
            name=name,
            task=task,
            prompt=prompt,
            seed=int(seed),
            tags=tags,
            notes=(None if notes_action == "keep" else ("" if notes_action == "clear" else notes)),
            merge_patch_json=patch,
        )
        return io.NodeOutput(out, out.manifest.get("notes", ""))


class MMH3Inspect(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Inspect",
            display_name="MMH3 Inspect",
            category=CATEGORY,
            description="Manifest-only inspection. Loaded heavy resources are not decoded or materialized.",
            inputs=[MMH3.Input("packet")],
            outputs=[
                io.String.Output("summary"),
                io.String.Output("info_json"),
                io.String.Output("notes"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("frames"),
                io.Float.Output("fps"),
                io.Float.Output("duration"),
                io.Boolean.Output("has_latent"),
                io.String.Output("prompt", tooltip="Stored generation prompt. Empty string when not recorded."),
                io.String.Output("task", tooltip="Stored generation task, for example fl2va/ref2va. Empty string when not recorded."),
                io.Int.Output("seed", tooltip="Stored generation seed. Returns 0 when not recorded; inspect info_json when presence vs seed=0 matters."),
                io.String.Output("name", tooltip="Packet name."),
            ],
        )

    @classmethod
    def execute(cls, packet) -> io.NodeOutput:
        info = inspect_packet(_packet(packet))
        geo = info["geometry"]
        generation = info.get("generation", {})
        seed = generation.get("seed")
        try:
            seed_out = int(seed) if seed is not None else 0
        except (TypeError, ValueError):
            seed_out = 0
        return io.NodeOutput(
            info["summary"],
            json.dumps(info, ensure_ascii=False, indent=2),
            str(info.get("notes", "")),
            int(geo["width"] or 0),
            int(geo["height"] or 0),
            int(geo["frames"] or 0),
            float(geo["fps"] or 0.0),
            float(geo["duration"] or 0.0),
            bool(info["has"]["latent"]),
            str(generation.get("prompt") or ""),
            str(generation.get("task") or ""),
            seed_out,
            str(info.get("name") or ""),
        )


class MMH3H3AVSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3AVSeparate",
            display_name="MMH3 H3 AV Separate",
            category=CATEGORY,
            description="Strictly split a MiniMax H3 joint AV latent into video and audio branches. Preserves auxiliary LATENT fields and split noise masks without copying tensor storage.",
            inputs=[io.Latent.Input("av_latent")],
            outputs=[
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, av_latent) -> io.NodeOutput:
        video_latent, audio_latent = split_h3_av_latent(av_latent)
        return io.NodeOutput(video_latent, audio_latent)


class MMH3H3AVCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3AVCombine",
            display_name="MMH3 H3 AV Combine",
            category=CATEGORY,
            description="Combine separated MiniMax H3 video/audio latents into one joint AV latent. Rejects batch/layout/duration mismatches instead of silently trimming or padding audio.",
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        return io.NodeOutput(concat_h3_av_latent(video_latent, audio_latent))


class MMH3H3Provenance(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3Provenance",
            display_name="MMH3 H3 Provenance",
            category=CATEGORY,
            description="Declare how the stored H3 AV latent was produced without loading or rewriting the latent payload.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input(
                    "origin",
                    options=list(H3_LATENT_ORIGINS),
                    default="sampler_output",
                    tooltip="Use sampler_output only when you know this is the direct output of an H3 sampler. This changes metadata only.",
                ),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, packet, origin: str) -> io.NodeOutput:
        packet = _packet(packet)
        res = packet.get_by_role_slot("h3_av_latent", 0)
        if res is None:
            raise MMH3ResourceError("Packet has no h3_av_latent")
        metadata = deep_copy_json(res.get("metadata", {}))
        h3 = metadata.get("h3")
        metadata["h3"] = h3_metadata_set_origin(h3, origin)
        out = packet.replace_resource_metadata(res["id"], metadata)
        ctx = metadata["h3"]["latent_context"]
        status = f"origin={origin}; continuation={ctx['continuation']['status']}; seam={ctx['seam_source']['status']}"
        return io.NodeOutput(out, status)


class MMH3H3Compatibility(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3Compatibility",
            display_name="MMH3 H3 Compatibility",
            category=CATEGORY,
            description="Manifest-only structural compatibility check for two H3 AV packets before a future seam/stitch adapter. Never approves naive latent concatenation.",
            inputs=[MMH3.Input("packet_a"), MMH3.Input("packet_b")],
            outputs=[
                io.Boolean.Output("seam_compatible"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(cls, packet_a, packet_b) -> io.NodeOutput:
        a = inspect_packet(_packet(packet_a)).get("h3_latent_context")
        b = inspect_packet(_packet(packet_b)).get("h3_latent_context")
        result = h3_pair_compatibility(a, b)
        if result["compatible"]:
            summary = "H3 seam source compatibility: OK. Naive latent concatenation remains unsafe."
        else:
            summary = "H3 seam source compatibility: NO — " + "; ".join(result["reasons"])
        if result.get("warnings"):
            summary += " Warnings: " + " | ".join(result["warnings"])
        return io.NodeOutput(bool(result["compatible"]), summary, json.dumps(result, ensure_ascii=False, indent=2))



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
            description="Preview is disposable cache. Reuse it when fresh or regenerate a contact sheet from H3 video latent through FastVAE.",
            inputs=[
                MMH3.Input("packet"),
                io.Vae.Input("fast_vae", optional=True, tooltip="Fast/Tiny H3-compatible VAE used only when preview must be generated."),
                io.Combo.Input("mode", options=["if_missing_or_stale", "refresh", "cache_only"], default="if_missing_or_stale"),
                io.Int.Input("frames", default=8, min=1, max=16),
                io.Int.Input("max_size", default=768, min=128, max=2048, step=64),
            ],
            outputs=[MMH3.Output("packet"), io.Image.Output("preview"), io.Boolean.Output("regenerated")],
        )

    @classmethod
    def execute(cls, packet, mode: str, frames: int, max_size: int, fast_vae=None) -> io.NodeOutput:
        packet = _packet(packet)
        existing = packet.get_by_role_slot("preview", 0)
        fresh = preview_is_fresh(packet)
        if existing is not None and mode != "refresh" and (mode == "cache_only" or fresh):
            image = get_resource_payload(packet, existing)
            return io.NodeOutput(packet, image, False)
        if mode == "cache_only":
            raise MMH3ResourceError("No usable preview cache is stored in this packet")
        out, image = build_fastvae_preview(packet, fast_vae, frames=int(frames), max_size=int(max_size))
        return io.NodeOutput(out, image, True)


class _MMH3GetBase(io.ComfyNode):
    KIND: str = ""
    OUTPUT_TYPE = None
    ROLES: list[str] = ["custom"]
    NODE_ID = ""
    DISPLAY_NAME = ""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category=CATEGORY,
            description=f"Extract one {cls.KIND.upper()} resource lazily and return the native ComfyUI type.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("role", options=cls.ROLES, default=cls.ROLES[0]),
                io.Int.Input("slot", default=0, min=0, max=9999),
                io.String.Input("resource_id", default="", advanced=True, tooltip="Stable selector; overrides role/slot when set."),
                io.String.Input("custom_role", default="", advanced=True),
                io.Combo.Input("missing", options=["error", "none"], default="error", advanced=True, tooltip="With none, the typed resource output is None when absent; use the exists output to guard downstream execution."),
            ],
            outputs=[
                cls.OUTPUT_TYPE.Output("resource"),
                io.Boolean.Output("exists"),
                io.String.Output("resource_id"),
                io.String.Output("metadata_json"),
            ],
        )

    @classmethod
    def execute(cls, packet, role: str, slot: int, resource_id: str, custom_role: str, missing: str) -> io.NodeOutput:
        packet = _packet(packet)
        role = _resolve_role(role, custom_role)
        res = packet.select(role=role, slot=int(slot), resource_id=resource_id.strip())
        if res is None:
            if missing == "none":
                return io.NodeOutput(None, False, "", "{}")
            label = resource_id.strip() or f"{role}[{slot}]"
            raise MMH3ResourceError(f"Resource {label} does not exist")
        if res["kind"] != cls.KIND:
            raise MMH3ResourceError(
                f"Selected resource {res['id']} is kind={res['kind']!r}, but {cls.DISPLAY_NAME} requires {cls.KIND!r}"
            )
        payload = get_resource_payload(packet, res)
        return io.NodeOutput(payload, True, res["id"], json.dumps(res.get("metadata", {}), ensure_ascii=False, indent=2))


class MMH3GetLatent(_MMH3GetBase):
    KIND = "latent"
    OUTPUT_TYPE = io.Latent
    ROLES = ["h3_av_latent", "continuation_context", "custom"]
    NODE_ID = "MMH3GetLatent"
    DISPLAY_NAME = "MMH3 Get Latent"


class MMH3GetImage(_MMH3GetBase):
    KIND = "image"
    OUTPUT_TYPE = io.Image
    ROLES = ["first_frame", "last_frame", "picture_ref", "preview", "custom"]
    NODE_ID = "MMH3GetImage"
    DISPLAY_NAME = "MMH3 Get Image"


class MMH3GetVideo(_MMH3GetBase):
    KIND = "video"
    OUTPUT_TYPE = io.Video
    ROLES = ["video", "video_ref", "custom"]
    NODE_ID = "MMH3GetVideo"
    DISPLAY_NAME = "MMH3 Get Video"


class MMH3GetAudio(_MMH3GetBase):
    KIND = "audio"
    OUTPUT_TYPE = io.Audio
    ROLES = ["audio", "audio_ref", "custom"]
    NODE_ID = "MMH3GetAudio"
    DISPLAY_NAME = "MMH3 Get Audio"


class MMH3GetMask(_MMH3GetBase):
    KIND = "mask"
    OUTPUT_TYPE = io.Mask
    ROLES = ["mask", "continuation_context", "custom"]
    NODE_ID = "MMH3GetMask"
    DISPLAY_NAME = "MMH3 Get Mask"


class MMH3GetJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3GetJSON",
            display_name="MMH3 Get JSON",
            category=CATEGORY,
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("role", options=["continuation_context", "custom"], default="custom"),
                io.Int.Input("slot", default=0, min=0, max=9999),
                io.String.Input("resource_id", default="", advanced=True),
                io.String.Input("custom_role", default="", advanced=True),
                io.Combo.Input("missing", options=["error", "none"], default="error", advanced=True, tooltip="With none, returns an empty JSON string and exists=false when absent."),
            ],
            outputs=[
                io.String.Output("json"),
                io.Boolean.Output("exists"),
                io.String.Output("resource_id"),
                io.String.Output("metadata_json"),
            ],
        )

    @classmethod
    def execute(cls, packet, role: str, slot: int, resource_id: str, custom_role: str, missing: str) -> io.NodeOutput:
        packet = _packet(packet)
        role = _resolve_role(role, custom_role)
        res = packet.select(role=role, slot=int(slot), resource_id=resource_id.strip())
        if res is None:
            if missing == "none":
                return io.NodeOutput("", False, "", "{}")
            raise MMH3ResourceError(f"JSON resource {resource_id or f'{role}[{slot}]'} does not exist")
        if res["kind"] != "json":
            raise MMH3ResourceError(f"Selected resource is kind={res['kind']!r}, expected 'json'")
        payload = get_resource_payload(packet, res)
        return io.NodeOutput(
            json.dumps(payload, ensure_ascii=False, indent=2),
            True,
            res["id"],
            json.dumps(res.get("metadata", {}), ensure_ascii=False, indent=2),
        )


class MMH3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MMH3Create,
            MMH3Load,
            MMH3Save,
            MMH3Put,
            MMH3Remove,
            MMH3Move,
            MMH3Metadata,
            MMH3Inspect,
            MMH3Export,
            MMH3Compare,
            MMH3Preview,
            MMH3H3AVSeparate,
            MMH3H3AVCombine,
            MMH3H3Provenance,
            MMH3H3Compatibility,
            MMH3GetLatent,
            MMH3GetImage,
            MMH3GetVideo,
            MMH3GetAudio,
            MMH3GetMask,
            MMH3GetJSON,
        ]
