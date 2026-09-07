from __future__ import annotations

from typing import Any

from .control_contract import (
    H3_CONTROL_ALGORITHMS,
    H3_CONTROL_ALGORITHM_CONTEXTS,
    H3_CONTROL_AUTO_PREFERENCES,
    H3_CONTROL_KIND_OPTIONS,
    H3_CONTROL_TEMPORAL_POLICIES,
    build_control_configuration,
    get_control_configuration,
    set_control_configuration,
    resolve_control_algorithm,
)
from .control_preflight import (
    CONTROLNET_LOADER_NODE_ID,
    H3_CONTROL_APPLY_NODE_ID,
    preflight_h3_control,
)
from .control_provider import (
    build_control_apply_plan,
    build_control_pass_through_info,
    build_h3_control_expansion,
    h3_fun_control_video_required,
    materialize_control_video,
    validate_masked_edit_inputs,
)
from .h3_fun_model_patch import (
    CURRENT_BACKEND,
    CURRENT_PROVIDER,
    apply_current_h3_fun_patch,
    available_h3_fun_checkpoints,
    load_current_h3_fun_patch,
    current_patch_preflight_info,
)
from .node_support import CATEGORY, MMH3, MMH3ResourceError, _packet, _parse_object, folder_paths, hashlib, io, json, ui
from .util import json_dumps_canonical


_CONCRETE_ALGORITHMS = tuple(item for item in H3_CONTROL_ALGORITHMS if item != "auto")


def _runtime_node_info(node_class: Any) -> dict[str, Any]:
    getter = getattr(node_class, "GET_NODE_INFO_V1", None)
    if callable(getter):
        value = getter()
        if isinstance(value, dict):
            return value
    input_types = getattr(node_class, "INPUT_TYPES", None)
    return_types = getattr(node_class, "RETURN_TYPES", None)
    if callable(input_types) and isinstance(return_types, (list, tuple)):
        return {"input": input_types(), "output": list(return_types)}
    raise MMH3ResourceError("Runtime node does not expose an inspectable schema")


class MMH3ControlConfigure(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ControlConfigure",
            display_name="MMH3 Control Configure",
            category=CATEGORY,
            description=(
                "Manifest-only F16 algorithm/control configuration. Records explicit requested/effective "
                "selection, temporal/audio/mask policies and stable packet resource IDs without loading media."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("requested_algorithm", options=list(H3_CONTROL_ALGORITHMS), default="native_reference"),
                io.Combo.Input(
                    "effective_algorithm",
                    options=["same_as_requested", *_CONCRETE_ALGORITHMS],
                    default="same_as_requested",
                    tooltip="For explicit algorithms keep same_as_requested. auto requires a concrete choice and selection reason.",
                ),
                io.Combo.Input(
                    "algorithm_context",
                    options=list(H3_CONTROL_ALGORITHM_CONTEXTS),
                    default="generation",
                    advanced=True,
                ),
                io.Combo.Input(
                    "auto_preference",
                    options=list(H3_CONTROL_AUTO_PREFERENCES),
                    default="stable_native_first",
                    advanced=True,
                ),
                io.String.Input(
                    "auto_capabilities_json",
                    default="{}",
                    multiline=True,
                    advanced=True,
                    tooltip="Required only when requested_algorithm=auto and effective_algorithm=same_as_requested.",
                ),
                io.Combo.Input("control_kind", options=list(H3_CONTROL_KIND_OPTIONS), default="none"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Combo.Input("temporal_policy", options=list(H3_CONTROL_TEMPORAL_POLICIES), default="strict"),
                io.String.Input("control_video_resource_id", default="", optional=True),
                io.String.Input("inpaint_source_resource_id", default="", optional=True),
                io.String.Input("mask_resource_id", default="", optional=True),
                io.String.Input("selection_reason", default="", optional=True, advanced=True),
                io.Boolean.Input("reference_composition", default=False, advanced=True),
                io.Combo.Input(
                    "composition_order",
                    options=["reference_then_control", "control_then_reference"],
                    default="reference_then_control",
                    advanced=True,
                ),
                io.String.Input("preprocessor_name", default="", optional=True, advanced=True),
                io.String.Input("preprocessor_version", default="", optional=True, advanced=True),
                io.String.Input("preprocessor_settings_json", default="{}", multiline=True, advanced=True),
                io.String.Input("checkpoint_sha256", default="", optional=True, advanced=True),
                io.String.Input("capability_fingerprint", default="", optional=True, advanced=True),
                io.String.Input("quantization", default="", optional=True, advanced=True),
                io.String.Input("dtype", default="", optional=True, advanced=True),
                io.String.Input("base_family", default="", optional=True, advanced=True),
                io.String.Input("adaln_form", default="", optional=True, advanced=True),
                io.String.Input("vae_fingerprint", default="", optional=True, advanced=True),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("summary"), io.String.Output("control_json")],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        requested_algorithm: str,
        effective_algorithm: str,
        algorithm_context: str,
        auto_preference: str,
        auto_capabilities_json: str,
        control_kind: str,
        strength: float,
        start_percent: float,
        end_percent: float,
        temporal_policy: str,
        control_video_resource_id: str = "",
        inpaint_source_resource_id: str = "",
        mask_resource_id: str = "",
        selection_reason: str = "",
        reference_composition: bool = False,
        composition_order: str = "reference_then_control",
        preprocessor_name: str = "",
        preprocessor_version: str = "",
        preprocessor_settings_json: str = "{}",
        checkpoint_sha256: str = "",
        capability_fingerprint: str = "",
        quantization: str = "",
        dtype: str = "",
        base_family: str = "",
        adaln_form: str = "",
        vae_fingerprint: str = "",
    ) -> io.NodeOutput:
        settings = _parse_object(preprocessor_settings_json, "preprocessor_settings_json")
        effective = "" if effective_algorithm == "same_as_requested" else effective_algorithm
        if requested_algorithm == "auto" and not effective:
            resolution = resolve_control_algorithm(
                requested_algorithm="auto",
                context=algorithm_context,
                availability=_parse_object(auto_capabilities_json, "auto_capabilities_json"),
                preference=auto_preference,
            )
            effective = resolution.effective_algorithm
            selection_reason = resolution.reason
        config = build_control_configuration(
            requested_algorithm=requested_algorithm,
            effective_algorithm=effective,
            control_kind=control_kind,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            temporal_policy=temporal_policy,
            control_video_resource_id=control_video_resource_id,
            inpaint_source_resource_id=inpaint_source_resource_id,
            mask_resource_id=mask_resource_id,
            selection_reason=selection_reason,
            reference_composition=reference_composition,
            composition_order=composition_order,
            preprocessor_name=preprocessor_name,
            preprocessor_version=preprocessor_version,
            preprocessor_settings=settings,
            checkpoint_sha256=checkpoint_sha256,
            capability_fingerprint=capability_fingerprint,
            quantization=quantization,
            dtype=dtype,
            base_family=base_family,
            adaln_form=adaln_form,
            vae_fingerprint=vae_fingerprint,
        )
        out = set_control_configuration(_packet(packet), config)
        summary = (
            f"F16 control configured · requested={config.requested_algorithm} "
            f"effective={config.effective_algorithm} kind={config.control_kind} "
            f"strength={config.strength:g} temporal={config.temporal_policy}"
        )
        return io.NodeOutput(out, summary, json.dumps(config.to_dict(), ensure_ascii=False, indent=2))


class MMH3ControlPreflight(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ControlPreflight",
            display_name="MMH3 Control Preflight",
            category=CATEGORY,
            description=(
                "F16 fail-closed capability/schema/checkpoint/base/VAE/temporal preflight. "
                "Inspects node schemas but never loads a model or media payload."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Boolean.Input("verify_runtime_nodes", default=True),
                io.String.Input("loader_node_id", default=CONTROLNET_LOADER_NODE_ID, advanced=True),
                io.String.Input("apply_node_id", default=H3_CONTROL_APPLY_NODE_ID, advanced=True),
                io.String.Input("checkpoint_info_json", default="{}", multiline=True),
                io.String.Input("checkpoint_name", default="", optional=True),
                io.Boolean.Input("verify_checkpoint_file", default=True),
                io.String.Input("base_info_json", default="{}", multiline=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("target_frames", default=124, min=5, max=3600, step=17),
                io.Int.Input("control_frames", default=124, min=1, max=3600),
            ],
            outputs=[io.Boolean.Output("ready"), io.String.Output("summary"), io.String.Output("info_json"), io.String.Output("capability_fingerprint")],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        verify_runtime_nodes: bool,
        loader_node_id: str,
        apply_node_id: str,
        checkpoint_info_json: str,
        checkpoint_name: str,
        verify_checkpoint_file: bool,
        base_info_json: str,
        width: int,
        height: int,
        target_frames: int,
        control_frames: int,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        config = get_control_configuration(packet)
        if config is None:
            raise MMH3ResourceError("Packet has no F16 control configuration; run MMH3 Control Configure first")
        checkpoint = _parse_object(checkpoint_info_json, "checkpoint_info_json")
        base = _parse_object(base_info_json, "base_info_json")
        if verify_checkpoint_file and config.effective_algorithm in (
            "fun_controlnet_union_bf16",
            "fun_controlnet_union_int8_convrot",
        ):
            selected_name = checkpoint_name.strip()
            if not selected_name:
                raise MMH3ResourceError("ControlNet preflight requires checkpoint_name when file verification is enabled")
            full_path = folder_paths.get_full_path("controlnet", selected_name)
            if not full_path:
                raise MMH3ResourceError(f"ControlNet checkpoint {selected_name!r} was not found")
            hasher = hashlib.sha256()
            with open(full_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            actual_sha256 = hasher.hexdigest()
            declared_sha256 = str(checkpoint.get("sha256") or "").strip().lower()
            if declared_sha256 and declared_sha256 != actual_sha256:
                raise MMH3ResourceError(
                    f"Declared ControlNet checkpoint SHA-256 does not match {selected_name!r}"
                )
            checkpoint["name"] = selected_name.replace("\\", "/")
            checkpoint["sha256"] = actual_sha256
            checkpoint["sha256_verified"] = True

        runtime_nodes = None
        runtime_contracts = None
        if verify_runtime_nodes:
            import nodes as comfy_nodes  # type: ignore

            mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
            runtime_nodes = tuple(mappings.keys())
            runtime_contracts = {}
            for node_id in (loader_node_id.strip(), apply_node_id.strip()):
                node_class = mappings.get(node_id)
                if node_class is None:
                    continue
                try:
                    runtime_contracts[node_id] = _runtime_node_info(node_class)
                except Exception:
                    # The pure preflight emits a fail-closed schema diagnostic.
                    pass

        report = preflight_h3_control(
            config,
            runtime_nodes=runtime_nodes,
            runtime_contracts=runtime_contracts,
            loader_node_id=loader_node_id.strip(),
            apply_node_id=apply_node_id.strip(),
            checkpoint=checkpoint,
            base=base,
            media={
                "width": int(width),
                "height": int(height),
                "target_frames": int(target_frames),
                "control_frames": int(control_frames),
            },
        )
        summary = report.summary()
        return io.NodeOutput(
            report.ready,
            summary,
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            report.capability_fingerprint,
            ui=ui.PreviewText(summary),
        )


class MMH3MaskedEditCondition(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3MaskedEditCondition",
            display_name="MMH3 Masked Edit Condition",
            category=CATEGORY,
            description=(
                "Strict typed source + white=regenerate mask assembly for F16 model-level video inpaint. "
                "Requires exact source/mask timeline and geometry; no resize or hold-last occurs here."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Image.Input("source_video"),
                io.Mask.Input("mask"),
                io.Image.Input("control_video", optional=True),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Image.Output("source_video"),
                io.Mask.Output("mask"),
                io.Image.Output("control_video"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(cls, packet, source_video, mask, control_video=None) -> io.NodeOutput:
        packet = _packet(packet)
        config = get_control_configuration(packet)
        if config is None or config.control_kind != "inpaint":
            raise MMH3ResourceError("MMH3 Masked Edit Condition requires packet control_kind='inpaint'")
        info = validate_masked_edit_inputs(source_video, mask, control_video)
        info["control"] = config.to_dict()
        return io.NodeOutput(
            packet,
            source_video,
            mask,
            control_video,
            json.dumps(info, ensure_ascii=False, indent=2),
        )


class MMH3ControlVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ControlVideo",
            display_name="MMH3 Control Video",
            category=CATEGORY,
            description=(
                "Lazy packet VIDEO to upstream IMAGE-timeline bridge by canonical H3 control usage. "
                "FPS conversion is explicit; spatial resize and control timeline alignment are deferred."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("usage", options=["control_video", "inpaint_source"], default="control_video"),
                io.String.Input("resource_id", default="", optional=True),
                io.Float.Input("target_fps", default=24.0, min=1.0, max=120.0, advanced=True),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.String.Output("resource_id"),
                io.String.Output("info_json"),
                io.Int.Output("frame_count"),
                io.Int.Output("width"),
                io.Int.Output("height"),
            ],
        )

    @classmethod
    def execute(cls, packet, usage: str, resource_id: str, target_fps: float) -> io.NodeOutput:
        result = materialize_control_video(
            _packet(packet),
            usage=usage,
            resource_id=resource_id,
            target_fps=target_fps,
        )
        return io.NodeOutput(
            result.frames,
            result.resource_id,
            result.info_json(),
            int(result.frames.shape[0]),
            int(result.frames.shape[2]),
            int(result.frames.shape[1]),
        )


class MMH3H3ControlApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3ControlApply",
            display_name="MMH3 H3 Control Apply",
            category=CATEGORY,
            description=(
                "Packet-native F16 expansion to the exact upstream MiniMax H3 Fun ControlNet apply node. "
                "Active control requires a READY preflight; strength=0 is a model-free pass-through."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Conditioning.Input("positive"),
                io.ControlNet.Input("control_net", optional=True),
                io.Vae.Input("vae", optional=True),
                io.String.Input("preflight_info_json", default="{}", multiline=True, optional=True),
                io.Image.Input("control_video", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Image.Input("source_video", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                MMH3.Output("packet"),
                io.String.Output("summary"),
                io.String.Output("process_info_json"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        positive,
        control_net=None,
        vae=None,
        preflight_info_json: str = "{}",
        control_video=None,
        mask=None,
        source_video=None,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        config = get_control_configuration(packet)
        if config is None:
            raise MMH3ResourceError("Packet has no F16 control configuration")
        if config.strength == 0.0:
            info = build_control_pass_through_info(config)
            summary = f"F16 control pass-through · algorithm={config.effective_algorithm} strength=0"
            return io.NodeOutput(
                positive,
                packet,
                summary,
                json.dumps(info, ensure_ascii=False, indent=2),
            )
        if control_net is None or vae is None:
            raise MMH3ResourceError("Active F16 control requires both control_net and vae")
        preflight_info = _parse_object(preflight_info_json, "preflight_info_json")
        plan = build_control_apply_plan(config, preflight_info)
        expansion = build_h3_control_expansion(
            plan,
            positive=positive,
            control_net=control_net,
            vae=vae,
            control_video=control_video,
            mask=mask,
            source_video=source_video,
        )
        info = plan.to_dict()
        summary = (
            f"F16 control {'pass-through' if plan.pass_through else 'applied'} · "
            f"algorithm={plan.adapter.algorithm} kind={plan.config.control_kind} "
            f"capability={plan.capability_fingerprint[:12]}"
        )
        return io.NodeOutput(
            expansion.positive,
            packet,
            summary,
            json.dumps(info, ensure_ascii=False, indent=2),
            expand=expansion.graph,
        )


class MMH3H3FunControl(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        checkpoints = available_h3_fun_checkpoints(folder_paths) or ["minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"]
        return io.Schema(
            node_id="MMH3H3FunControl",
            display_name="MMH3 H3 Fun Control",
            category=CATEGORY,
            description=(
                "High-level F16 owner for the current upstream MiniMax H3 Fun MODEL_PATCH contract (PR #15975). "
                "Loads the control patch, verifies live checkpoint/base compatibility, aligns strict control inputs, "
                "patches MODEL, and emits canonical MMH3 control provenance. Checkpoints are resolved from "
                "models/model_patches first, then models/controlnet."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Combo.Input("control_checkpoint", options=checkpoints),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("target_frames", default=124, min=5, max=3600, step=17),
                io.Image.Input(
                    "control_video", optional=True, lazy=True,
                    tooltip=(
                        "Lazy structural control input. For inpaint this is an optional supplemental Pose stream; "
                        "it is requested only when control_video_resource_id is configured."
                    ),
                ),
                io.Mask.Input("mask", optional=True),
                io.Image.Input("source_video", optional=True),
            ],
            outputs=[
                io.Model.Output("model"),
                MMH3.Output("packet"),
                io.String.Output("summary"),
                io.String.Output("process_info_json"),
            ],
        )

    @classmethod
    def check_lazy_status(
        cls, packet, model=None, vae=None, control_checkpoint: str = "", width: int = 1344,
        height: int = 768, target_frames: int = 124, control_video=None, mask=None, source_video=None,
    ) -> list[str]:
        """Request the expensive control-video branch only when the configured contract needs it."""
        config = get_control_configuration(_packet(packet))
        if config is None:
            return []
        if h3_fun_control_video_required(config) and control_video is None:
            return ["control_video"]
        return []

    @classmethod
    def execute(
        cls, packet, model, vae, control_checkpoint: str, width: int, height: int, target_frames: int,
        control_video=None, mask=None, source_video=None,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        config = get_control_configuration(packet)
        if config is None:
            raise MMH3ResourceError("Packet has no F16 control configuration; run MMH3 Control Configure first")
        if config.effective_algorithm not in ("fun_controlnet_union_bf16", "fun_controlnet_union_int8_convrot"):
            raise MMH3ResourceError(
                f"MMH3 H3 Fun Control requires a Fun Control algorithm, got {config.effective_algorithm!r}"
            )
        if int(width) % 32 or int(height) % 32:
            raise MMH3ResourceError("H3 Fun control width and height must be exact multiples of 32")
        if int(target_frames) < 5 or (int(target_frames) - 5) % 17:
            raise MMH3ResourceError("H3 Fun control target_frames must follow the 17n+5 grid")

        if config.strength == 0.0:
            info = build_control_pass_through_info(config)
            info["backend"] = CURRENT_BACKEND
            return io.NodeOutput(
                model, packet,
                f"F16 control pass-through · algorithm={config.effective_algorithm} strength=0",
                json.dumps(info, ensure_ascii=False, indent=2),
            )

        # Reuse the packet's temporal policy and strict geometry checks before the runtime patch owns any model state.
        from .control_provider import H3ControlApplyPlan, control_provider_for_algorithm, prepare_control_inputs
        adapter = control_provider_for_algorithm(config.effective_algorithm)
        provisional = H3ControlApplyPlan(
            adapter=adapter, config=config, capability_fingerprint="",
            target_width=int(width), target_height=int(height), target_frames=int(target_frames),
            pass_through=False, capability={},
        )
        control_video, mask, source_video = prepare_control_inputs(
            provisional, control_video=control_video, mask=mask, source_video=source_video
        )

        loaded = load_current_h3_fun_patch(folder_paths, control_checkpoint, model)
        expected_quant = "bf16" if config.effective_algorithm.endswith("_bf16") else "int8_convrot"
        if loaded.quantization != expected_quant:
            raise MMH3ResourceError(
                f"Selected algorithm requires {expected_quant}, but checkpoint runtime detected {loaded.quantization}"
            )
        if config.control_kind == "inpaint" and loaded.control_in_dim != 49:
            raise MMH3ResourceError(
                f"H3 Fun inpaint requires control_in_dim=49; checkpoint exposes {loaded.control_in_dim}"
            )

        patched = apply_current_h3_fun_patch(
            model, loaded, vae,
            control_video=control_video, mask=mask, source_video=source_video,
            strength=config.strength, start_percent=config.start_percent, end_percent=config.end_percent,
        )

        preflight = current_patch_preflight_info(
            config, loaded, width=int(width), height=int(height), frames=int(target_frames)
        )
        plan = build_control_apply_plan(config, preflight)
        info = plan.to_dict()
        info["backend"] = CURRENT_BACKEND
        info["checkpoint"] = loaded.filename
        info["checkpoint_source_folder"] = loaded.source_folder
        summary = (
            f"F16 H3 Fun MODEL_PATCH applied · kind={config.control_kind} strength={config.strength:g} "
            f"checkpoint={loaded.filename} capability={plan.capability_fingerprint[:12]}"
        )
        return io.NodeOutput(patched, packet, summary, json.dumps(info, ensure_ascii=False, indent=2))
