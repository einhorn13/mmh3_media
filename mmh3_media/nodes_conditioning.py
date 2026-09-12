from __future__ import annotations

from .generation_contract import H3_CONTINUATION_FAMILIES, resolve_h3_continuation_family

from .node_support import (
    ADD_GUIDE_NODE_ID,
    CATEGORY,
    H3_LATENT_ORIGINS,
    H3_MODEL_FAMILIES,
    INTENTS,
    MMH3,
    MMH3ResourceError,
    MODES,
    POLICIES,
    PREFLIGHT_OPERATIONS,
    REFERENCE_PRESETS,
    REFERENCE_PURPOSES,
    TILE_GUIDER_NODE_ID,
    _packet,
    _parse_object,
    build_h3_expansion,
    build_reference_cache_spec,
    compare_reference_cache_latents,
    configure_reference,
    folder_paths,
    get_generation_loras,
    hashlib,
    inspect_packet,
    io,
    json,
    lookup_reference_cache,
    materialize_reference_cache,
    materialize_video_reference,
    pack_h3_result,
    preflight_packet,
    put_reference_cache,
    require_native_h3_contract,
    resolve_packet,
    resolve_reference_set,
    torch,
    ui,
)
from .reference_sizing import native_reference_image_size, reference_image_sizing_from_manifest, reference_resize_dimensions


class MMH3H3ReferenceImageResize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3ReferenceImageResize",
            display_name="H3 Reference Image Resize",
            category=CATEGORY,
            description="Internal aspect-preserving short-edge downscale used by Ref2VA custom sizing.",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("short_edge", min=32, max=2048, step=32),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(cls, image, short_edge: int) -> io.NodeOutput:
        height, width = int(image.shape[1]), int(image.shape[2])
        target_width, target_height = reference_resize_dimensions(width, height, int(short_edge))
        if (target_width, target_height) == (width, height):
            return io.NodeOutput(image)
        channels_first = image.movedim(-1, 1)
        resized = torch.nn.functional.interpolate(
            channels_first,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).movedim(1, -1)
        return io.NodeOutput(resized)

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
                io.String.Output("aspect_ratio", tooltip="Reduced display aspect ratio, for example 16:9. Empty when geometry is unknown."),
                io.Float.Output("aspect_ratio_value", tooltip="Width divided by height. Returns 0 when geometry is unknown."),
                io.Int.Output("frames"),
                io.Float.Output("fps"),
                io.Float.Output("duration"),
                io.Boolean.Output("has_latent"),
                io.String.Output("prompt", tooltip="Stored generation prompt. Empty string when not recorded."),
                io.String.Output("task", tooltip="Stored H3 generation mode, for example fl2va or ref2va. Empty when not recorded."),
                io.Int.Output("seed", tooltip="Stored generation seed. Zero remains a valid seed."),
                io.Boolean.Output("seed_recorded", tooltip="Distinguishes an absent seed from the valid seed value 0."),
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
            str(geo.get("aspect_ratio") or ""),
            float(geo.get("aspect_ratio_value") or 0.0),
            int(geo["frames"] or 0),
            float(geo["fps"] or 0.0),
            float(geo["duration"] or 0.0),
            bool(info["has"]["latent"]),
            str(generation.get("prompt") or ""),
            str(generation.get("task") or ""),
            seed_out,
            seed is not None,
            str(info.get("name") or ""),
        )


class MMH3Preflight(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Preflight",
            display_name="MMH3 Preflight",
            category=CATEGORY,
            description="F15 fail-closed preflight for packet geometry, H3 provenance, LoRAs and required runtime nodes. Does not load models.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("operation", options=list(PREFLIGHT_OPERATIONS), default="generate"),
                io.Combo.Input("unknown_loras", options=["error", "allow"], default="error", advanced=True),
                io.Boolean.Input("verify_runtime_nodes", default=True, advanced=True),
                io.Boolean.Input("verify_lora_files", default=True, advanced=True),
                io.String.Input("required_nodes_json", default="[]", multiline=True, advanced=True),
                io.Combo.Input("generation_mode", options=list(MODES), default="auto", optional=True, advanced=True),
                io.Combo.Input("model_family", options=list(H3_MODEL_FAMILIES), default="auto", optional=True, advanced=True),
                io.String.Input("checkpoint_name", default="", optional=True, advanced=True),
                io.Boolean.Input("allow_experimental_model_family", default=False, optional=True, advanced=True),
            ],
            outputs=[io.Boolean.Output("ready"), io.String.Output("summary"), io.String.Output("info_json")],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        operation: str,
        unknown_loras: str,
        verify_runtime_nodes: bool,
        verify_lora_files: bool,
        required_nodes_json: str,
        generation_mode: str = "auto",
        model_family: str = "auto",
        checkpoint_name: str = "",
        allow_experimental_model_family: bool = False,
    ) -> io.NodeOutput:
        try:
            required_nodes = json.loads(required_nodes_json or "[]")
        except json.JSONDecodeError as exc:
            raise MMH3ResourceError(f"required_nodes_json is not valid JSON: {exc}") from exc
        if not isinstance(required_nodes, list) or not all(isinstance(item, str) for item in required_nodes):
            raise MMH3ResourceError("required_nodes_json must be a JSON array of node IDs")

        runtime_nodes = None
        runtime_contracts = None
        if verify_runtime_nodes:
            import nodes as comfy_nodes  # type: ignore
            mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
            runtime_nodes = tuple(mappings.keys())
            runtime_contracts = {}
            add_guide_class = mappings.get(ADD_GUIDE_NODE_ID)
            if add_guide_class is not None:
                try:
                    runtime_contracts[ADD_GUIDE_NODE_ID] = add_guide_class.GET_NODE_INFO_V1()
                except Exception:
                    # Pure preflight reports a fail-closed schema-inspection diagnostic.
                    pass
            tile_guider_class = mappings.get(TILE_GUIDER_NODE_ID)
            if tile_guider_class is not None:
                try:
                    runtime_contracts[TILE_GUIDER_NODE_ID] = {
                        "input": tile_guider_class.INPUT_TYPES(),
                        "output": list(tile_guider_class.RETURN_TYPES),
                    }
                except Exception:
                    # Pure preflight reports a fail-closed schema-inspection diagnostic.
                    pass

        packet = _packet(packet)
        inventory = None
        if verify_lora_files and operation == "high_sigma_refine":
            inventory = {}
            recorded = get_generation_loras(packet) or ()
            for name in folder_paths.get_filename_list("loras"):
                normalized = str(name).replace("\\", "/")
                full = folder_paths.get_full_path("loras", name)
                digest = None
                wanted = next((entry.get("sha256") for entry in recorded if entry["name"] == normalized), None)
                if full and wanted:
                    hasher = hashlib.sha256()
                    with open(full, "rb") as handle:
                        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            hasher.update(chunk)
                    digest = hasher.hexdigest()
                inventory[normalized] = digest

        report = preflight_packet(
            packet,
            operation=operation,
            runtime_nodes=runtime_nodes,
            runtime_contracts=runtime_contracts,
            required_nodes=required_nodes,
            lora_inventory=inventory,
            allow_unknown_loras=unknown_loras == "allow",
            generation_mode=generation_mode,
            model_family=model_family,
            checkpoint_name=checkpoint_name,
            allow_experimental_model_family=allow_experimental_model_family,
        )
        return io.NodeOutput(report.ready, report.summary(), json.dumps(report.to_dict(), ensure_ascii=False, indent=2), ui=ui.PreviewText(report.summary()))


class MMH3ReferenceConfigure(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ReferenceConfigure",
            display_name="MMH3 Reference Settings",
            category=CATEGORY,
            description="Set how one image, video or audio reference is used by H3. Payload data is not loaded or copied.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input(
                    "resource_id",
                    tooltip="Stable resource ID. Connect resource_id from the MMH3 Put node that added this reference.",
                ),
                io.Combo.Input(
                    "inclusion",
                    options=["keep", "include", "exclude"],
                    default="keep",
                    tooltip="keep leaves the current state; include enables this reference; exclude keeps it in the packet but omits it from H3 conditioning.",
                ),
                io.Int.Input(
                    "order",
                    default=-1,
                    min=-1,
                    max=9999,
                    tooltip="Presentation order among references. -1 keeps the current order from MMH3 Put.",
                ),
                io.String.Input(
                    "purposes_json",
                    display_name="purposes (JSON)",
                    default="",
                    multiline=False,
                    advanced=True,
                    tooltip=f"How H3 should classify this reference. Blank keeps the current value. JSON array using: {', '.join(REFERENCE_PURPOSES)}.",
                ),
                io.Combo.Input(
                    "binding_action",
                    display_name="soundtrack binding",
                    options=["keep", "set", "clear"],
                    default="keep",
                    advanced=True,
                    tooltip="Audio only: keep the current video binding, set a binding, or clear it.",
                ),
                io.String.Input(
                    "video_resource_id",
                    display_name="bind to video resource ID",
                    default="",
                    advanced=True,
                    tooltip="For soundtrack binding=set, enter the stable ID of an existing video reference.",
                ),
            ],
            outputs=[
                MMH3.Output("packet", tooltip="Packet with the updated reference settings."),
                io.String.Output("resource_id", tooltip="Stable ID of the configured reference."),
                io.String.Output("summary", tooltip="Short result suitable for a text preview."),
                io.String.Output("info_json", tooltip="Resolved reference report after this change."),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        resource_id: str,
        inclusion: str,
        order: int,
        purposes_json: str,
        binding_action: str,
        video_resource_id: str,
    ) -> io.NodeOutput:
        purposes = None
        if purposes_json.strip():
            try:
                purposes = json.loads(purposes_json)
            except json.JSONDecodeError as e:
                raise MMH3ResourceError(f"purposes_json is not valid JSON: {e}") from e
            if not isinstance(purposes, list):
                raise MMH3ResourceError("purposes_json must be a JSON array")
        result = configure_reference(
            _packet(packet),
            resource_id,
            inclusion=inclusion,
            purposes=purposes,
            order=None if int(order) < 0 else int(order),
            binding_action=binding_action,
            video_resource_id=video_resource_id,
        )
        report = result.report
        summary = (
            f"Reference {result.resource_id}: configured · active={len(report.resources)} "
            f"disabled={len(report.disabled_resources)} · ready={report.ready}"
        )
        return io.NodeOutput(result.packet, result.resource_id, summary, result.info_json())


class MMH3ReferenceReport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ReferenceReport",
            display_name="MMH3 Reference Report",
            category=CATEGORY,
            description="Manifest-only F13 preflight: exact H3 presentation order, prompt tags, soundtrack bindings, filters and native limits.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("preset", options=list(REFERENCE_PRESETS), default="all"),
                io.Int.Input("target_width_override", default=0, min=0, max=16384, advanced=True),
                io.Int.Input("target_height_override", default=0, min=0, max=16384, advanced=True),
                io.Int.Input("target_frames_override", default=0, min=0, max=3600, advanced=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Boolean.Output("ready"),
                io.Int.Output("active_count"),
                io.String.Output("prompt_tags_json"),
                io.String.Output("binding_map_json"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
                io.String.Output("cost_json"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        preset: str,
        target_width_override: int = 0,
        target_height_override: int = 0,
        target_frames_override: int = 0,
        ref_image_size: str = "match",
    ) -> io.NodeOutput:
        packet = _packet(packet)
        generation = packet.manifest.get("generation", {})
        target_width = int(target_width_override or generation.get("width") or 0)
        target_height = int(target_height_override or generation.get("height") or 0)
        target_frames = int(target_frames_override or generation.get("frames") or 0)
        report = resolve_reference_set(
            packet,
            preset=preset,
            target_width=target_width or None,
            target_height=target_height or None,
            target_frames=target_frames or None,
            ref_image_size=ref_image_size,
        )
        tags = [item.prompt_tag for item in report.presentation]
        bindings = {
            item.resource.resource_id: item.paired_video_resource_id
            for item in report.presentation
            if item.paired_video_resource_id
        }
        state = "READY" if report.ready else "BLOCKED"
        lines = [
            f"{state} · references preset={report.preset} · active={len(report.resources)} "
            f"disabled={len(report.disabled_resources)} filtered={len(report.filtered_resources)}"
        ]
        lines.extend(
            f"{index + 1}. {item.prompt_tag} ← {item.resource.kind} order={item.resource.order} ({item.resource.resource_id})"
            for index, item in enumerate(report.presentation)
        )
        lines.extend(f"{item.severity.upper()} [{item.code}] {item.message}" for item in report.diagnostics)
        if report.cost is not None:
            lines.append(report.cost.summary())
            lines.extend(
                f"{item.get('severity', 'info').upper()} [{item.get('code', 'reference_cost')}] {item.get('message', '')}"
                for item in report.cost.diagnostics
            )
        summary = "\n".join(lines)
        return io.NodeOutput(
            packet,
            report.ready,
            len(report.resources),
            json.dumps(tags, ensure_ascii=False),
            json.dumps(bindings, ensure_ascii=False, indent=2),
            summary,
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            json.dumps(None if report.cost is None else report.cost.to_dict(), ensure_ascii=False, indent=2),
            ui=ui.PreviewText(summary),
        )


class MMH3ReferenceCacheStore(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ReferenceCacheStore",
            display_name="MMH3 Reference Cache Store",
            category=CATEGORY,
            description="Store an exact native H3 VAE reference latent with a strict source/geometry/VAE/core fingerprint. Qwen presentation is intentionally not cached.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("source_resource_id"),
                io.Latent.Input("reference_latent"),
                io.String.Input("vae_fingerprint", tooltip="Required content/config fingerprint of the H3 video VAE, not only a display filename."),
                io.Int.Input("target_width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("target_height", default=768, min=32, max=16384, step=32),
                io.Int.Input("target_frames", default=124, min=5, max=3600),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.String.Output("cache_resource_id"),
                io.String.Output("cache_key"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        source_resource_id: str,
        reference_latent,
        vae_fingerprint: str,
        target_width: int,
        target_height: int,
        target_frames: int,
        ref_image_size: str,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        contract = require_native_h3_contract("ref2va")
        spec = build_reference_cache_spec(
            packet,
            source_resource_id,
            target_width=int(target_width),
            target_height=int(target_height),
            target_frames=int(target_frames),
            ref_image_size=ref_image_size,
            vae_fingerprint=vae_fingerprint,
            native_contract_fingerprint=contract.fingerprint,
        )
        out, resource_id = put_reference_cache(packet, spec, reference_latent)
        info = {"stored": True, "cache_resource_id": resource_id, "spec": spec.to_dict()}
        return io.NodeOutput(out, resource_id, spec.key, json.dumps(info, ensure_ascii=False, indent=2))


class MMH3ReferenceCacheGet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ReferenceCacheGet",
            display_name="MMH3 Reference Cache Get",
            category=CATEGORY,
            description="Resolve an exact VAE reference-block cache. Any source, resize, target, VAE or native-contract drift is a visible miss.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("source_resource_id"),
                io.String.Input("vae_fingerprint"),
                io.Int.Input("target_width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("target_height", default=768, min=32, max=16384, step=32),
                io.Int.Input("target_frames", default=124, min=5, max=3600),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Combo.Input("missing", options=["error", "none"], default="error", advanced=True),
            ],
            outputs=[
                io.Latent.Output("reference_latent"),
                io.Boolean.Output("hit"),
                io.String.Output("cache_resource_id"),
                io.String.Output("cache_key"),
                io.String.Output("info_json"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        source_resource_id: str,
        vae_fingerprint: str,
        target_width: int,
        target_height: int,
        target_frames: int,
        ref_image_size: str,
        missing: str,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        contract = require_native_h3_contract("ref2va")
        spec = build_reference_cache_spec(
            packet,
            source_resource_id,
            target_width=int(target_width),
            target_height=int(target_height),
            target_frames=int(target_frames),
            ref_image_size=ref_image_size,
            vae_fingerprint=vae_fingerprint,
            native_contract_fingerprint=contract.fingerprint,
        )
        lookup = lookup_reference_cache(packet, spec)
        if not lookup.hit:
            if missing == "error":
                stale = f"; stale candidates: {', '.join(lookup.stale_resource_ids)}" if lookup.stale_resource_ids else ""
                raise MMH3ResourceError(f"Reference cache miss for key {spec.key}{stale}")
            info_json = json.dumps(lookup.to_dict(), ensure_ascii=False, indent=2)
            return io.NodeOutput(None, False, "", spec.key, info_json, ui=ui.PreviewText(f"CACHE MISS · {spec.key}"))
        latent = materialize_reference_cache(packet, lookup)
        info_json = json.dumps(lookup.to_dict(), ensure_ascii=False, indent=2)
        return io.NodeOutput(
            latent,
            True,
            str(lookup.descriptor["id"]),
            spec.key,
            info_json,
            ui=ui.PreviewText(f"CACHE HIT · {spec.key}\nresource={lookup.descriptor['id']}"),
        )


class MMH3ReferenceCacheVerify(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ReferenceCacheVerify",
            display_name="MMH3 Reference Cache Verify",
            category=CATEGORY,
            description="Exact tensor equivalence check for an original H3 VAE reference latent and a cache round-trip.",
            inputs=[io.Latent.Input("original"), io.Latent.Input("cached")],
            outputs=[
                io.Boolean.Output("exact"),
                io.Float.Output("max_abs_error"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, original, cached) -> io.NodeOutput:
        result = compare_reference_cache_latents(original, cached)
        summary = (
            f"{'EXACT' if result['exact'] else 'MISMATCH'} · shape={result['original_shape']} "
            f"dtype={result['original_dtype']} · max_abs_error={result['max_abs_error']}"
        )
        return io.NodeOutput(
            bool(result["exact"]),
            float(result["max_abs_error"] or 0.0),
            summary,
            json.dumps(result, ensure_ascii=False, indent=2),
            ui=ui.PreviewText(summary),
        )


class MMH3ResolveReport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ResolveReport",
            display_name="MMH3 Resolve Report",
            category=CATEGORY,
            description="Manifest-only F00 preflight. Resolves mode, overrides and resource bindings without loading media payloads.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("intent", options=list(INTENTS), default="condition/generate"),
                io.Combo.Input("mode", options=list(MODES), default="auto"),
                io.Combo.Input("policy", options=list(POLICIES), default="auto"),
                io.String.Input("prompt_override", default="", multiline=True, dynamic_prompts=False, advanced=True),
                io.Int.Input("seed_override", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, advanced=True),
                io.Int.Input("width_override", default=0, min=0, max=16384, advanced=True),
                io.Int.Input("height_override", default=0, min=0, max=16384, advanced=True),
                io.Int.Input("frames_override", default=0, min=0, max=100000, advanced=True),
                io.Float.Input("fps_override", default=0.0, min=0.0, max=1000.0, advanced=True),
                io.Combo.Input("reference_preset", options=list(REFERENCE_PRESETS), default="all", advanced=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Boolean.Output("ready"),
                io.String.Output("mode"),
                io.String.Output("prompt"),
                io.Int.Output("seed"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("frames"),
                io.Float.Output("fps"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        intent: str,
        mode: str,
        policy: str,
        prompt_override: str,
        seed_override: int,
        width_override: int,
        height_override: int,
        frames_override: int,
        fps_override: float,
        reference_preset: str = "all",
        ref_image_size: str = "match",
    ) -> io.NodeOutput:
        packet = _packet(packet)
        result = resolve_packet(
            packet,
            intent=intent,
            mode=mode,
            policy=policy,
            reference_preset=reference_preset,
            ref_image_size=ref_image_size,
            overrides={
                "prompt": prompt_override,
                "seed": seed_override,
                "width": width_override,
                "height": height_override,
                "frames": frames_override,
                "fps": fps_override,
            },
        )
        summary = result.summary()
        return io.NodeOutput(
            packet,
            result.ready,
            result.mode,
            str(result.value("prompt") or ""),
            int(result.value("seed") or 0),
            int(result.value("width") or 0),
            int(result.value("height") or 0),
            int(result.value("frames") or 0),
            float(result.value("fps") or 0.0),
            summary,
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            ui=ui.PreviewText(summary),
        )


class MMH3H3VideoReference(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3VideoReference",
            display_name="MMH3 H3 Video Reference",
            category=CATEGORY,
            description="Lazy canonical reference/video adapter for native H3 Ref2VA. Returns an IMAGE frame batch and an explicitly selected soundtrack.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("resource_id"),
                io.String.Input("paired_audio_resource_id", default="", advanced=True),
                io.Boolean.Input("include_embedded_audio", default=False, advanced=True),
                io.Float.Input("target_fps", default=24.0, min=1.0, max=120.0, advanced=True),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        resource_id: str,
        paired_audio_resource_id: str,
        include_embedded_audio: bool,
        target_fps: float,
    ) -> io.NodeOutput:
        result = materialize_video_reference(
            _packet(packet),
            resource_id.strip(),
            paired_audio_resource_id=paired_audio_resource_id.strip(),
            include_embedded_audio=bool(include_embedded_audio),
            target_fps=float(target_fps),
        )
        return io.NodeOutput(result.frames, result.audio, result.info_json())


class MMH3H3AutoCondition(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3AutoCondition",
            display_name="Prepare H3 Generation",
            category=CATEGORY,
            description="Resolve the packet task and prepare native H3 conditioning and latent inputs.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input(
                    "prompt_override",
                    display_name="Prompt override",
                    default="",
                    multiline=True,
                    tooltip="Leave blank to use the prompt stored in the MMH3 packet.",
                ),
                io.Int.Input(
                    "seed_override",
                    display_name="Seed override",
                    default=-1,
                    min=-1,
                    max=0xFFFFFFFFFFFFFFFF,
                    advanced=True,
                    tooltip="-1 uses the seed stored in the packet.",
                ),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae", display_name="Video VAE"),
                io.Vae.Input("audio_vae", display_name="Audio VAE", optional=True),
                io.Int.Input("width_override", display_name="Width", default=0, min=0, max=16384, force_input=True),
                io.Int.Input("height_override", display_name="Height", default=0, min=0, max=16384, force_input=True),
                io.Int.Input("frames_override", display_name="Frames", default=0, min=0, max=3600, force_input=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("latent"),
                io.Int.Output("seed"),
                MMH3.Output("packet"),
                io.String.Output("mode"),
                io.String.Output("status"),
                io.String.Output("info_json"),
                io.Combo.Output("task_family"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        prompt_override: str,
        seed_override: int,
        clip,
        video_vae,
        width_override: int,
        height_override: int,
        frames_override: int,
        audio_vae=None,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        from .scheduled_references import validate_scene_compilation
        validate_scene_compilation(packet, prompt_override or packet.manifest.get("generation", {}).get("prompt", ""))
        reference_size_mode, reference_short_edge = reference_image_sizing_from_manifest(packet.manifest)
        native_ref_image_size = native_reference_image_size(reference_size_mode)
        resolved = resolve_packet(
            packet,
            intent="condition/generate",
            mode="auto",
            policy="auto",
            reference_preset="all",
            ref_image_size=native_ref_image_size,
            overrides={
                "prompt": prompt_override,
                "seed": seed_override,
                "width": width_override,
                "height": height_override,
                "frames": frames_override,
            },
        )
        if not resolved.ready:
            raise MMH3ResourceError(resolved.summary())
        width = int(resolved.value("width") or 0)
        height = int(resolved.value("height") or 0)
        frames = int(resolved.value("frames") or 0)
        fps = float(resolved.value("fps") or 0.0)
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise MMH3ResourceError("H3 width and height must be positive multiples of 32")
        if frames < 5 or (frames - 5) % 17:
            raise MMH3ResourceError("H3 frame count must use the 17n+5 grid")
        if fps != 24.0:
            raise MMH3ResourceError("H3 generation requires 24 FPS")
        expansion = build_h3_expansion(
            packet,
            resolved,
            clip=clip,
            video_vae=video_vae,
            audio_vae=audio_vae,
            ref_image_size=native_ref_image_size,
            ref_image_short_edge=reference_short_edge if reference_size_mode == "custom" else None,
        )
        info = resolved.to_dict()
        info["reference_image_sizing"] = {
            "mode": reference_size_mode,
            "short_edge": reference_short_edge,
            "native_mode": native_ref_image_size,
        }
        task_family = "ref2va" if resolved.mode == "ref2va" else "fl2va"
        info["task_family"] = task_family
        info["conditioning_mode"] = resolved.mode
        info["native_h3_contract_fingerprint"] = expansion.contract_fingerprint
        status = resolved.summary() + f"\nNative H3 contract: {expansion.contract_fingerprint[:12]}"
        return io.NodeOutput(
            expansion.positive,
            expansion.latent,
            int(resolved.value("seed") or 0),
            packet,
            resolved.mode,
            status,
            json.dumps(info, ensure_ascii=False, indent=2),
            task_family,
            expand=expansion.graph,
        )


class MMH3H3ContinuationCondition(io.ComfyNode):
    """Prepare future conditioning independently from the family that produced the source latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3ContinuationCondition",
            display_name="Prepare H3 Continuation",
            category=CATEGORY,
            description=(
                "Prepare the unknown future of a direct H3 continuation. The protected joint-AV prefix owns "
                "continuity; choose FL2VA or Ref2VA independently for future conditioning. Auto uses Ref2VA "
                "when active packet references exist, otherwise FL2VA."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.String.Input(
                    "handover_info_json",
                    display_name="Handover info",
                    multiline=True,
                    force_input=True,
                    tooltip="Connect info_json from MMH3 H3 Continuation Handover so source-lineage proof is preserved.",
                ),
                io.Combo.Input(
                    "task_family",
                    display_name="Future task family",
                    options=list(H3_CONTINUATION_FAMILIES),
                    default="auto",
                    tooltip=(
                        "FL2VA ignores packet references/keyframes for future conditioning. Ref2VA consumes active "
                        "packet references. This does not change the family that produced the source latent."
                    ),
                ),
                io.String.Input(
                    "prompt_override",
                    display_name="Prompt override",
                    default="",
                    multiline=True,
                    tooltip="Leave blank to use the prompt stored in the MMH3 packet.",
                ),
                io.Int.Input(
                    "seed_override",
                    display_name="Seed override",
                    default=-1,
                    min=-1,
                    max=0xFFFFFFFFFFFFFFFF,
                    advanced=True,
                    tooltip="-1 uses the seed stored in the packet.",
                ),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae", display_name="Video VAE"),
                io.Vae.Input("audio_vae", display_name="Audio VAE", optional=True),
                io.Int.Input("width_override", display_name="Width", default=0, min=0, max=16384, force_input=True),
                io.Int.Input("height_override", display_name="Height", default=0, min=0, max=16384, force_input=True),
                io.Int.Input("frames_override", display_name="Frames", default=0, min=0, max=3600, force_input=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Int.Output("seed"),
                MMH3.Output("packet"),
                io.String.Output("mode"),
                io.String.Output("status"),
                io.String.Output("info_json"),
                io.Combo.Output("task_family"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        handover_info_json: str,
        task_family: str,
        prompt_override: str,
        seed_override: int,
        clip,
        video_vae,
        width_override: int,
        height_override: int,
        frames_override: int,
        audio_vae=None,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        reference_size_mode, reference_short_edge = reference_image_sizing_from_manifest(packet.manifest)
        native_ref_image_size = native_reference_image_size(reference_size_mode)
        handover_info = _parse_object(handover_info_json, "handover_info_json")
        from .scheduled_references import validate_scene_compilation
        validate_scene_compilation(packet, prompt_override or packet.manifest.get("generation", {}).get("prompt", ""))
        if handover_info.get("contract") != "minimax_h3_joint_av_continuation_v1":
            raise MMH3ResourceError(
                "Prepare H3 Continuation requires info_json from MMH3 H3 Continuation Handover"
            )

        reference_set = resolve_reference_set(packet, preset="all")
        family, conditioning_mode = resolve_h3_continuation_family(
            task_family,
            has_references=bool(reference_set.resources),
        )
        resolved = resolve_packet(
            packet,
            intent="condition/generate",
            mode=conditioning_mode,
            policy="auto",
            reference_preset="all",
            ref_image_size=native_ref_image_size,
            overrides={
                "prompt": prompt_override,
                "seed": seed_override,
                "width": width_override,
                "height": height_override,
                "frames": frames_override,
            },
        )
        if not resolved.ready:
            raise MMH3ResourceError(resolved.summary())
        width = int(resolved.value("width") or 0)
        height = int(resolved.value("height") or 0)
        frames = int(resolved.value("frames") or 0)
        fps = float(resolved.value("fps") or 0.0)
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise MMH3ResourceError("H3 width and height must be positive multiples of 32")
        if frames < 5 or (frames - 5) % 17:
            raise MMH3ResourceError("H3 frame count must use the 17n+5 grid")
        if fps != 24.0:
            raise MMH3ResourceError("H3 continuation requires 24 FPS")

        expansion = build_h3_expansion(
            packet,
            resolved,
            clip=clip,
            video_vae=video_vae,
            audio_vae=audio_vae,
            ref_image_size=native_ref_image_size,
            ref_image_short_edge=reference_short_edge if reference_size_mode == "custom" else None,
        )
        conditioning_info = resolved.to_dict()
        conditioning_info["reference_image_sizing"] = {
            "mode": reference_size_mode,
            "short_edge": reference_short_edge,
            "native_mode": native_ref_image_size,
        }
        conditioning_info["task_family"] = family
        conditioning_info["native_conditioning_mode"] = conditioning_mode
        conditioning_info["native_h3_contract_fingerprint"] = expansion.contract_fingerprint

        # Keep the resolver fields at the process-info root so Pack H3 Result can
        # preserve prompt/seed and selected input-resource provenance exactly as it
        # does for ordinary Auto Condition. Handover metadata wins on overlapping
        # contract fields and remains directly readable by Latent Stitch.
        process_info = dict(conditioning_info)
        process_info.update(handover_info)
        process_info["task_family"] = family
        process_info["conditioning_mode"] = conditioning_mode
        process_info["conditioning"] = conditioning_info
        status = (
            f"{handover_info.get('summary') or 'READY · F02 continuation'}\n"
            f"Future conditioning: {family} ({conditioning_mode}) · refs={len(reference_set.resources)} · "
            f"native={expansion.contract_fingerprint[:12]}"
        )
        return io.NodeOutput(
            expansion.positive,
            int(resolved.value("seed") or 0),
            packet,
            conditioning_mode,
            status,
            json.dumps(process_info, ensure_ascii=False, indent=2),
            family,
            expand=expansion.graph,
        )


class MMH3PackH3Result(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3PackH3Result",
            display_name="MMH3 Pack H3 Result",
            category=CATEGORY,
            description="Atomically pack standard H3 process outputs into a new packet and record one process-level provenance entry.",
            inputs=[
                MMH3.Input("packet"),
                io.Latent.Input("latent", optional=True),
                io.Video.Input("video", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.String.Input("operation", default="generate"),
                io.String.Input("mode", default="", advanced=True),
                io.String.Input("status", default="", multiline=True, advanced=True),
                io.String.Input("process_info_json", default="", multiline=True, advanced=True),
                io.String.Input(
                    "generation_settings_json",
                    default="",
                    multiline=True,
                    optional=True,
                    advanced=True,
                ),
                io.String.Input(
                    "sampling_profile_json",
                    default="",
                    multiline=True,
                    optional=True,
                    advanced=True,
                ),
                io.String.Input(
                    "optimization_profile_json",
                    default="",
                    multiline=True,
                    optional=True,
                    advanced=True,
                ),
                io.String.Input("applied_loras_json", default="", multiline=True, optional=True, advanced=True),
                io.String.Input("control_process_info_json", default="", multiline=True, optional=True, advanced=True),
                io.Combo.Input(
                    "latent_origin",
                    options=list(H3_LATENT_ORIGINS),
                    default="sampler_output",
                    advanced=True,
                ),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.String.Output("summary"),
                io.String.Output("result_info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        operation: str,
        mode: str,
        status: str,
        process_info_json: str,
        latent_origin: str,
        latent=None,
        video=None,
        audio=None,
        first_frame=None,
        last_frame=None,
        generation_settings_json: str = "",
        sampling_profile_json: str = "",
        optimization_profile_json: str = "",
        applied_loras_json: str = "",
        control_process_info_json: str = "",
    ) -> io.NodeOutput:
        applied_loras = None
        if applied_loras_json.strip():
            try:
                applied_loras = json.loads(applied_loras_json)
            except json.JSONDecodeError as exc:
                raise MMH3ResourceError(f"applied_loras_json is not valid JSON: {exc}") from exc
            if applied_loras is not None and not isinstance(applied_loras, list):
                raise MMH3ResourceError("applied_loras_json must be an ordered JSON array or null")
        control_process_info = None
        if control_process_info_json.strip():
            control_process_info = _parse_object(
                control_process_info_json, "control_process_info_json"
            )
        process_info = _parse_object(process_info_json, "process_info_json")
        if generation_settings_json.strip():
            if "generation_settings" in process_info:
                raise MMH3ResourceError(
                    "process_info.generation_settings is reserved; pass generation_settings_json separately"
                )
            process_info["generation_settings"] = _parse_object(
                generation_settings_json, "generation_settings_json"
            )
        if sampling_profile_json.strip():
            if "sampling" in process_info:
                raise MMH3ResourceError(
                    "process_info.sampling is reserved; pass sampling_profile_json separately"
                )
            process_info["sampling"] = _parse_object(
                sampling_profile_json, "sampling_profile_json"
            )
        if optimization_profile_json.strip():
            if "optimization" in process_info:
                raise MMH3ResourceError(
                    "process_info.optimization is reserved; pass optimization_profile_json separately"
                )
            process_info["optimization"] = _parse_object(
                optimization_profile_json, "optimization_profile_json"
            )
        result = pack_h3_result(
            _packet(packet),
            latent=latent,
            video=video,
            audio=audio,
            first_frame=first_frame,
            last_frame=last_frame,
            operation=operation,
            mode=mode,
            status=status,
            process_info=process_info,
            latent_origin=latent_origin,
            applied_loras=applied_loras,
            control_process_info=control_process_info,
        )
        info_json = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        return io.NodeOutput(result.packet, result.summary(), info_json)
