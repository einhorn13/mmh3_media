from __future__ import annotations

from typing import Any

from .node_support import (
    CATEGORY,
    MMH3,
    MMH3ResourceError,
    TILE_BACKEND_OVERLAP_MODES,
    TILE_BLEND_MODES,
    TILE_CONTEXT_SOURCES,
    TILE_GUIDER_NODE_ID,
    TILE_OVERLAP_MODES,
    TILE_TRAVERSALS,
    UPSCALE_GEOMETRY_MODES,
    _packet,
    _parse_object,
    build_latent_stitch_upscale_target,
    build_latent_upscale_process_report,
    build_latent_upscale_refine_sampling,
    build_latent_upscale_refine_target,
    concat_h3_av_latent,
    deep_copy_json,
    finalize_external_tile_video,
    folder_paths,
    h3_expected_audio_t,
    io,
    json,
    nested_parts,
    plan_spatial_tiles,
    prepare_decoded_packet_latent_upscale,
    prepare_packet_latent_upscale,
    run_native_h3_tile_refine,
    tile_backend_fingerprint_from_preflight,
    torch,
    validate_h3_av_latent,
)
from .control_contract import get_control_configuration
from .control_provider import (
    build_control_apply_plan,
    build_control_pass_through_info,
    prepare_control_inputs,
)
from .h3_fun_model_patch import (
    CURRENT_BACKEND,
    apply_current_h3_fun_patch,
    available_h3_fun_checkpoints,
    current_patch_preflight_info,
    load_current_h3_fun_patch,
)
from .control_tiles import (
    apply_control_to_conditioning,
    build_tiled_control_process_info,
    clone_guider_with_conditioning,
    crop_control_inputs_for_tile,
)
from .upscaler_adapter import UPSCALER_NODE, build_upscaler_inputs, resolve_upscaler_api


class MMH3H3LearnedUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3LearnedUpscale",
            display_name="MMH3 H3 Learned Upscale",
            category=CATEGORY,
            description=(
                "Stable MMH3 adapter for legacy and Plus MinimaxH3LatentUpscaler3D releases. "
                "Upscales video only to the exact prepared dimensions."
            ),
            inputs=[
                io.Latent.Input("video_latent"),
                io.String.Input("model_name"),
                io.Int.Input("target_width", min=32, max=8192, step=32),
                io.Int.Input("target_height", min=32, max=8192, step=32),
                io.Int.Input("align", default=32, min=16, max=512, step=16, advanced=True),
                io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="bf16"),
                io.Boolean.Input(
                    "offload_after_upscale",
                    default=True,
                    advanced=True,
                    tooltip="Free learned-upscaler VRAM before H3 refinement; disable for faster repeated runs.",
                ),
                io.Boolean.Input(
                    "legacy_temporal_chunking",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Legacy low-VRAM fallback only. Full-sequence inference is recommended because temporal "
                        "chunking changes Conv3D/GroupNorm context and can create boundary differences."
                    ),
                ),
            ],
            outputs=[io.Latent.Output("video_latent")],
            enable_expand=True,
        )

    @classmethod
    def execute(
        cls,
        video_latent,
        model_name: str,
        target_width: int,
        target_height: int,
        align: int,
        device: str,
        precision: str,
        offload_after_upscale: bool,
        legacy_temporal_chunking: bool,
    ) -> io.NodeOutput:
        import nodes
        from comfy_execution.graph_utils import GraphBuilder

        backend = nodes.NODE_CLASS_MAPPINGS.get(UPSCALER_NODE)
        if backend is None:
            raise MMH3ResourceError("Install Minimax H3 Latent Upscaler (3D) to use F07")
        api = resolve_upscaler_api(backend)
        graph = GraphBuilder()
        upscaler = graph.node(
            UPSCALER_NODE,
            **build_upscaler_inputs(
                api,
                latent=video_latent,
                model_name=model_name,
                target_width=target_width,
                target_height=target_height,
                align=align,
                device=device,
                precision=precision,
                offload_after_upscale=offload_after_upscale,
                legacy_temporal_chunking=legacy_temporal_chunking,
            ),
        )
        return io.NodeOutput(upscaler.out(0), expand=graph.finalize())

class MMH3H3LatentUpscalePrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3LatentUpscalePrepare",
            display_name="MMH3 H3 Latent Upscale Prepare",
            category=CATEGORY,
            description=(
                "F07 packet/provenance/geometry boundary for a replaceable learned video-latent upscaler. "
                "Returns the original audio latent unchanged and exact aligned target dimensions."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("geometry_mode", options=list(UPSCALE_GEOMETRY_MODES), default="scale"),
                io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05),
                io.Int.Input("target_width", default=0, min=0, max=4096, step=32, advanced=True),
                io.Int.Input("target_height", default=0, min=0, max=4096, step=32, advanced=True),
                io.Float.Input("target_megapixels", default=0.0, min=0.0, max=8.0, step=0.1, advanced=True),
                io.Int.Input("align", default=32, min=32, max=512, step=32, advanced=True),
                io.Boolean.Input("enable_chunking", default=True, advanced=True),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
                io.Int.Output("target_width"),
                io.Int.Output("target_height"),
                io.Int.Output("frames"),
                io.String.Output("source_resource_id"),
                io.String.Output("summary"),
                io.String.Output("process_info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        geometry_mode: str,
        scale: float,
        target_width: int,
        target_height: int,
        target_megapixels: float,
        align: int,
        enable_chunking: bool,
    ) -> io.NodeOutput:
        prepared = prepare_packet_latent_upscale(
            _packet(packet),
            geometry_mode=geometry_mode,
            scale=float(scale),
            target_width=int(target_width),
            target_height=int(target_height),
            target_megapixels=float(target_megapixels),
            align=int(align),
            enable_chunking=bool(enable_chunking),
        )
        return io.NodeOutput(
            prepared.packet,
            prepared.video_latent,
            prepared.audio_latent,
            prepared.plan.target_width,
            prepared.plan.target_height,
            prepared.plan.frames,
            prepared.source_resource_id,
            prepared.plan.summary(),
            json.dumps(prepared.plan.to_dict(), ensure_ascii=False, indent=2),
        )


class MMH3H3LatentUpscaleTarget(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3LatentUpscaleTarget",
            display_name="MMH3 H3 Latent Upscale Refine Target",
            category=CATEGORY,
            description=(
                "Build a strict joint H3 refine target from upscaled video and source audio. "
                "Native nested masks denoise video and lock audio."
            ),
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("summary")],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        target = build_latent_upscale_refine_target(video_latent, audio_latent)
        return io.NodeOutput(
            target,
            "READY · F07 video-only high-sigma target · video_mask=1 · audio_mask=0",
        )


class MMH3H3DecodedUpscalePrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3DecodedUpscalePrepare",
            display_name="MMH3 H3 Decoded Upscale Prepare",
            category=CATEGORY,
            description=(
                "F07 decoded-video entry path. Normalizes primary packet AV to the H3 grid, VAE-encodes it, "
                "preserves the audio latent and returns aligned learned-upscale geometry with vae_encoded provenance."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Combo.Input("missing_audio_policy", options=["error", "silence"], default="error"),
                io.Combo.Input("geometry_mode", options=list(UPSCALE_GEOMETRY_MODES), default="target_megapixels"),
                io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05),
                io.Int.Input("target_width", default=0, min=0, max=4096, step=32, advanced=True),
                io.Int.Input("target_height", default=0, min=0, max=4096, step=32, advanced=True),
                io.Float.Input("target_megapixels", default=2.1, min=0.1, max=8.0, step=0.1),
                io.Int.Input("align", default=32, min=32, max=512, step=32, advanced=True),
                io.Boolean.Input("enable_chunking", default=True, advanced=True),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
                io.Int.Output("target_width"),
                io.Int.Output("target_height"),
                io.Int.Output("frames"),
                io.String.Output("source_resource_id"),
                io.String.Output("summary"),
                io.String.Output("process_info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        video_vae,
        audio_vae,
        missing_audio_policy: str,
        geometry_mode: str,
        scale: float,
        target_width: int,
        target_height: int,
        target_megapixels: float,
        align: int,
        enable_chunking: bool,
    ) -> io.NodeOutput:
        prepared = prepare_decoded_packet_latent_upscale(
            _packet(packet),
            video_vae=video_vae,
            audio_vae=audio_vae,
            missing_audio_policy=missing_audio_policy,
            geometry_mode=geometry_mode,
            scale=float(scale),
            target_width=int(target_width),
            target_height=int(target_height),
            target_megapixels=float(target_megapixels),
            align=int(align),
            enable_chunking=bool(enable_chunking),
        )
        report = prepared.plan.to_dict()
        report["source_adapter"] = deep_copy_json(prepared.source_adapter)
        return io.NodeOutput(
            prepared.packet,
            prepared.video_latent,
            prepared.audio_latent,
            prepared.plan.target_width,
            prepared.plan.target_height,
            prepared.plan.frames,
            prepared.source_resource_id,
            prepared.plan.summary() + " · source=decoded/vae_encoded",
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class MMH3H3UpscaleRefineSampling(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3UpscaleRefineSampling",
            display_name="MMH3 H3 Upscale Refine Sampling",
            category=CATEGORY,
            description=(
                "Derive the HR second-pass sampler from the source MMH3 packet. Inherits task family, "
                "sampler/scheduler and AV shifts. Steps inherit the source unless explicitly overridden. "
                "Choose a regenerated low-noise grid or an exact recorded tail with H3 Refine Scheduler."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Float.Input(
                    "denoise_override",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.025,
                    advanced=True,
                    tooltip="0 = source-aware auto. The full 0–1 domain is available; 0.05–0.50 is recommended because stronger refinement may replace source structure.",
                ),
                io.Int.Input("steps_override", default=0, min=0, max=10000, optional=True,
                    tooltip="0 = inherit count in regenerated_tail, or select a denoise fraction in source_tail. Positive = actual refine steps. Turbo overrides are experimental."),
                io.Combo.Input("schedule_mode", options=["regenerated_tail", "source_tail"], default="regenerated_tail", optional=True,
                    tooltip="source_tail reuses recorded sigmas exactly; 0 steps selects denoise fraction of recorded intervals. Older packets require regenerated_tail."),
            ],
            outputs=[
                io.String.Output("refine_sampling_json"),
                io.Combo.Output("task_family"),
                io.Int.Output("steps"),
                io.Float.Output("video_shift"),
                io.Float.Output("audio_shift"),
                io.Combo.Output("sampler_name"),
                io.Combo.Output("scheduler"),
                io.Float.Output("denoise"),
                io.String.Output("summary"),
            ],
        )

    @classmethod
    def execute(cls, packet, denoise_override: float, steps_override: int = 0, schedule_mode: str = "regenerated_tail") -> io.NodeOutput:
        recipe = build_latent_upscale_refine_sampling(
            _packet(packet), denoise_override=denoise_override, steps_override=steps_override, schedule_mode=schedule_mode
        )
        return io.NodeOutput(
            json.dumps(recipe.info, ensure_ascii=False, indent=2),
            recipe.task_family,
            recipe.steps,
            recipe.video_shift,
            recipe.audio_shift,
            recipe.sampler,
            recipe.scheduler,
            recipe.denoise,
            recipe.summary(),
        )


class MMH3H3LatentUpscaleReport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3LatentUpscaleReport",
            display_name="MMH3 H3 Latent Upscale Report",
            category=CATEGORY,
            description=(
                "Merge F07 geometry, source-LoRA and process-LoRA facts into the atomic pack diagnostics "
                "and exact applied model stack, with optional validated spatial tile plan/run reports. "
                "Does not load models or mutate a packet."
            ),
            inputs=[
                io.String.Input("geometry_report_json", multiline=True),
                io.String.Input("source_lora_report_json", multiline=True),
                io.String.Input("process_loras_json", default="[]", multiline=True),
                io.String.Input("tile_plan_json", default="", multiline=True, advanced=True),
                io.String.Input("tile_run_report_json", default="", multiline=True, advanced=True),
                io.String.Input("native_tile_adapter_json", default="", multiline=True, advanced=True),
                io.String.Input("external_tile_report_json", default="", multiline=True, advanced=True),
                io.String.Input("execution_profile_json", default="", multiline=True, advanced=True),
                io.String.Input("upscaler_model", default="minimax_h3_latent_upscaler_3d_bf16.safetensors"),
                io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp32", "fp16", "bf16"], default="bf16"),
                io.Combo.Input("sigma_profile", options=["3_steps", "4_steps", "5_steps", "fast_res2m_4step", "source_aware"], default="3_steps"),
                io.String.Input("refine_sampling_json", default="", multiline=True, optional=True, advanced=True),
                io.String.Input("preflight_json", default="", multiline=True, optional=True, advanced=True,
                    tooltip="When connected, the checked settings replace the legacy upscaler model/device/precision widgets."),
                io.String.Input("audio_delivery_json", default="", multiline=True, optional=True, advanced=True),
            ],
            outputs=[
                io.String.Output("process_info_json"),
                io.String.Output("applied_loras_json"),
                io.String.Output("summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        geometry_report_json: str,
        source_lora_report_json: str,
        process_loras_json: str,
        tile_plan_json: str,
        tile_run_report_json: str,
        native_tile_adapter_json: str,
        external_tile_report_json: str,
        upscaler_model: str,
        device: str,
        precision: str,
        sigma_profile: str,
        refine_sampling_json: str = "",
        execution_profile_json: str = "",
        preflight_json: str = "",
        audio_delivery_json: str = "",
    ) -> io.NodeOutput:
        preflight = _parse_object(preflight_json, "preflight") if preflight_json.strip() else None
        if preflight is not None:
            if preflight.get("contract") != "mmh3_f07_preflight_v1" or preflight.get("ready") is not True:
                raise MMH3ResourceError("F07 report requires a successful preflight")
            if preflight["geometry"] != _parse_object(geometry_report_json, "geometry"):
                raise MMH3ResourceError("F07 report geometry disagrees with preflight")
            settings = preflight["settings"]
            upscaler_model, device, precision = settings["upscaler_model"], settings["device"], settings["precision"]
        try:
            process_loras = json.loads(process_loras_json or "[]")
        except json.JSONDecodeError as exc:
            raise MMH3ResourceError(f"process_loras_json is not valid JSON: {exc}") from exc
        if not isinstance(process_loras, list):
            raise MMH3ResourceError("process_loras_json must be an ordered JSON array")
        tile_plan = _parse_object(tile_plan_json, "tile_plan_json") if str(tile_plan_json or "").strip() else None
        tile_run_report = (
            _parse_object(tile_run_report_json, "tile_run_report_json")
            if str(tile_run_report_json or "").strip()
            else None
        )
        external_tile_report = (
            _parse_object(external_tile_report_json, "external_tile_report_json")
            if str(external_tile_report_json or "").strip()
            else None
        )
        native_tile_adapter = (
            _parse_object(native_tile_adapter_json, "native_tile_adapter_json")
            if str(native_tile_adapter_json or "").strip()
            else None
        )
        execution_profile = (
            _parse_object(execution_profile_json, "execution_profile_json")
            if str(execution_profile_json or "").strip()
            else None
        )
        refine_sampling = (
            _parse_object(refine_sampling_json, "refine_sampling_json")
            if str(refine_sampling_json or "").strip()
            else None
        )
        if preflight is not None and (refine_sampling is None or refine_sampling.get("refine") != preflight["refine"]):
            raise MMH3ResourceError("F07 executed sampling recipe disagrees with preflight")
        report = build_latent_upscale_process_report(
            _parse_object(geometry_report_json, "geometry_report_json"),
            _parse_object(source_lora_report_json, "source_lora_report_json"),
            process_loras=process_loras,
            upscaler_model=upscaler_model,
            device=device,
            precision=precision,
            sigma_profile=sigma_profile,
            execution_profile=execution_profile,
            refine_sampling=refine_sampling,
            tile_plan=tile_plan,
            tile_run_report=tile_run_report,
            native_tile_adapter_report=native_tile_adapter,
            external_tile_report=external_tile_report,
        )
        info = report.to_dict()
        if preflight is not None:
            info["preflight"] = preflight
        if audio_delivery_json.strip():
            info["audio_delivery"] = _parse_object(audio_delivery_json, "audio delivery")
            if preflight is not None and info["audio_delivery"]["policy"] != settings["audio_policy"]:
                raise MMH3ResourceError("F07 audio delivery disagrees with preflight")
        return io.NodeOutput(
            json.dumps(info, ensure_ascii=False, indent=2),
            json.dumps(list(report.applied_loras), ensure_ascii=False, indent=2),
            report.summary(),
        )


class MMH3H3ExternalTileFinalize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3ExternalTileFinalize",
            display_name="MMH3 H3 External Tile Finalize",
            category=CATEGORY,
            description=(
                "Encode an IMAGE-only external tile result through the supplied VideoVAE, recombine exact source "
                "audio and emit derived provenance. Does not sample or decode."
            ),
            inputs=[
                io.Image.Input("video"),
                io.Vae.Input("video_vae"),
                io.Latent.Input("source_audio_latent"),
                io.String.Input("preflight_info_json", multiline=True),
                io.Combo.Input("overlap_mode", options=list(TILE_BACKEND_OVERLAP_MODES), default="Context Only Overlap"),
                io.String.Input("backend_settings_json", default="{}", multiline=True, advanced=True),
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.String.Output("finalize_report_json"),
                io.String.Output("summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        video_vae,
        source_audio_latent,
        preflight_info_json: str,
        overlap_mode: str,
        backend_settings_json: str,
    ) -> io.NodeOutput:
        settings = _parse_object(backend_settings_json or "{}", "backend_settings_json")
        preflight_info = _parse_object(preflight_info_json, "preflight_info_json")
        backend_schema_fingerprint = tile_backend_fingerprint_from_preflight(preflight_info)
        result = finalize_external_tile_video(
            video,
            source_audio_latent,
            lambda pixels: {"samples": video_vae.encode(pixels)},
            backend_node_id=TILE_GUIDER_NODE_ID,
            backend_schema_fingerprint=backend_schema_fingerprint,
            overlap_mode=overlap_mode,
            batch_size=1,
            settings=settings,
        )
        return io.NodeOutput(
            result.latent,
            json.dumps(result.report, ensure_ascii=False, indent=2),
            result.summary(),
        )


class MMH3H3NativeTileRefine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3NativeTileRefine",
            display_name="MMH3 H3 Native Tile Refine",
            category=CATEGORY,
            description=(
                "Sequentially refine a decoded H3 video in spatial tiles using the supplied Guider/Sampler/Sigmas. "
                "Every tile carries the complete timeline, video uses an all-one noise mask, source audio uses an "
                "all-zero mask and sampled audio is discarded. The final composited video is VAE-encoded once and "
                "recombined with the exact source audio latent. Optional F16 inputs crop full-canvas control to each "
                "tile sample_rect before sampling; full-canvas control is never sent to a cropped latent."
            ),
            inputs=[
                io.Image.Input("video"),
                io.Custom("GUIDER").Input("guider"),
                io.Custom("SAMPLER").Input("sampler"),
                io.Custom("SIGMAS").Input("sigmas"),
                io.Vae.Input("video_vae"),
                io.Latent.Input("source_audio_latent"),
                io.Int.Input("frames", default=5, min=5, max=999999),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("tile_width", default=640, min=32, max=8192, step=32),
                io.Int.Input("tile_height", default=384, min=32, max=8192, step=32),
                io.Int.Input("overlap", default=64, min=0, max=4096, step=32),
                io.Int.Input("context_padding", default=64, min=0, max=4096, step=32),
                io.Combo.Input("traversal", options=list(TILE_TRAVERSALS), default="snake"),
                io.Combo.Input("overlap_mode", options=list(TILE_OVERLAP_MODES), default="context_only"),
                io.Combo.Input("blend_mode", options=list(TILE_BLEND_MODES), default="hard"),
                io.Combo.Input("context_source", options=list(TILE_CONTEXT_SOURCES), default="composited"),
                MMH3.Input("packet", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Combo.Input(
                    "control_checkpoint",
                    options=["", *available_h3_fun_checkpoints(folder_paths)],
                    default="",
                    optional=True,
                    tooltip="Current #15975 MODEL_PATCH checkpoint. Preferred over legacy control_net inputs.",
                ),
                io.ControlNet.Input("control_net", optional=True),
                io.Vae.Input("control_vae", optional=True),
                io.String.Input(
                    "control_preflight_info_json",
                    default="{}",
                    multiline=True,
                    optional=True,
                ),
                io.Image.Input("control_video", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Image.Input("source_video", optional=True),
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.Image.Output("video"),
                io.String.Output("tile_plan_json"),
                io.String.Output("tile_run_report_json"),
                io.String.Output("native_adapter_report_json"),
                io.String.Output("summary"),
                io.String.Output("control_process_info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        guider,
        sampler,
        sigmas,
        video_vae,
        source_audio_latent,
        frames: int,
        seed: int,
        tile_width: int,
        tile_height: int,
        overlap: int,
        context_padding: int,
        traversal: str,
        overlap_mode: str,
        blend_mode: str,
        context_source: str,
        packet=None,
        positive=None,
        control_checkpoint: str = "",
        control_net=None,
        control_vae=None,
        control_preflight_info_json: str = "{}",
        control_video=None,
        mask=None,
        source_video=None,
    ) -> io.NodeOutput:
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            raise MMH3ResourceError("MMH3 native tile refine expects a [T,H,W,C] IMAGE tensor")
        decoded_frames, height, width, channels = (int(value) for value in video.shape)
        if channels != 3:
            raise MMH3ResourceError("MMH3 native tile refine requires RGB video")
        frames = int(frames)
        if frames < 5 or (frames - 5) % 17:
            raise MMH3ResourceError("MMH3 native tile refine frames must use the H3 17k+5 grid")
        if decoded_frames < frames:
            raise MMH3ResourceError("MMH3 native tile refine decoded input has fewer frames than requested")
        if not isinstance(source_audio_latent, dict) or not isinstance(source_audio_latent.get("samples"), torch.Tensor):
            raise MMH3ResourceError("MMH3 native tile refine requires a separated source audio LATENT")
        if int(source_audio_latent["samples"].shape[-1]) != h3_expected_audio_t(frames):
            raise MMH3ResourceError("MMH3 native tile refine frames do not match source audio duration")
        video = video[:frames].detach()
        plan = plan_spatial_tiles(
            target_width=width,
            target_height=height,
            frames=frames,
            tile_width=int(tile_width),
            tile_height=int(tile_height),
            overlap=int(overlap),
            context_padding=int(context_padding),
            align=32,
            traversal=traversal,
            overlap_mode=overlap_mode,
            blend_mode=blend_mode,
            context_source=context_source,
            element_bytes=video.element_size(),
        )

        control_plan = None
        current_control_patch = None
        prepared_control = (None, None, None)
        control_process_info: dict[str, Any] | None = None
        supplied_control_inputs = any(
            value is not None
            for value in (positive, control_net, control_vae, control_video, mask, source_video)
        ) or bool(str(control_checkpoint or "").strip()) or str(control_preflight_info_json or "{}").strip() not in ("", "{}")
        if packet is None:
            if supplied_control_inputs:
                raise MMH3ResourceError(
                    "Tile-aware F16 inputs require the configured MMH3 packet"
                )
        else:
            packet = _packet(packet)
            config = get_control_configuration(packet)
            if config is None:
                raise MMH3ResourceError("Packet has no F16 control configuration")
            if config.strength == 0.0:
                control_process_info = build_control_pass_through_info(config)
            else:
                current_control_patch = None
                selected_checkpoint = str(control_checkpoint or "").strip()
                if selected_checkpoint:
                    current_control_patch = load_current_h3_fun_patch(
                        folder_paths, selected_checkpoint, guider.model_patcher
                    )
                    expected_quant = (
                        "bf16" if config.effective_algorithm.endswith("_bf16") else "int8_convrot"
                    )
                    if current_control_patch.quantization != expected_quant:
                        raise MMH3ResourceError(
                            f"Selected tile control algorithm requires {expected_quant}, but checkpoint runtime detected {current_control_patch.quantization}"
                        )
                    preflight_info = current_patch_preflight_info(
                        config, current_control_patch, width=width, height=height, frames=frames
                    )
                else:
                    # Backward-compatible #15860 path for old saved workflows. New examples use MODEL_PATCH.
                    if positive is None or control_net is None:
                        raise MMH3ResourceError(
                            "Active tile-aware F16 control requires control_checkpoint for current MODEL_PATCH, "
                            "or positive + control_net + preflight for the legacy #15860 path"
                        )
                    preflight_info = _parse_object(
                        control_preflight_info_json,
                        "control_preflight_info_json",
                    )
                control_plan = build_control_apply_plan(config, preflight_info)
                prepared_control = prepare_control_inputs(
                    control_plan,
                    control_video=control_video,
                    mask=mask,
                    source_video=source_video,
                )
                control_process_info = build_tiled_control_process_info(control_plan, plan)
                if current_control_patch is not None:
                    control_process_info["backend"] = CURRENT_BACKEND
                    control_process_info["checkpoint"] = current_control_patch.filename

        import comfy.model_management  # type: ignore
        import comfy.sample  # type: ignore
        import comfy.sampler_helpers  # type: ignore
        import comfy.utils  # type: ignore
        import latent_preview  # type: ignore

        progress = comfy.utils.ProgressBar(len(plan.tiles))

        def sample_joint(target, tile):
            tile_guider = guider
            if control_plan is not None:
                tile_inputs = crop_control_inputs_for_tile(
                    plan,
                    tile,
                    control_video=prepared_control[0],
                    mask=prepared_control[1],
                    source_video=prepared_control[2],
                )
                if current_control_patch is not None:
                    import copy
                    tile_guider = copy.copy(guider)
                    tile_guider.model_patcher = apply_current_h3_fun_patch(
                        guider.model_patcher,
                        current_control_patch,
                        control_vae if control_vae is not None else video_vae,
                        control_video=tile_inputs.control_video,
                        mask=tile_inputs.mask,
                        source_video=tile_inputs.source_video,
                        strength=control_plan.config.strength,
                        start_percent=control_plan.config.start_percent,
                        end_percent=control_plan.config.end_percent,
                    )
                else:
                    tile_positive = apply_control_to_conditioning(
                        positive,
                        control_net,
                        control_vae if control_vae is not None else video_vae,
                        control_plan,
                        tile_inputs,
                    )
                    tile_guider = clone_guider_with_conditioning(
                        guider,
                        tile_positive,
                        convert_conditioning=comfy.sampler_helpers.convert_cond,
                    )
            latent_image = target["samples"]
            noise = comfy.sample.prepare_noise(latent_image, int(seed), target.get("batch_index"))
            callback = latent_preview.prepare_callback(tile_guider.model_patcher, sigmas.shape[-1] - 1)
            samples = tile_guider.sample(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask=target.get("noise_mask"),
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=int(seed),
            )
            samples = samples.to(comfy.model_management.intermediate_device())
            out = target.copy()
            out["samples"] = samples
            progress.update(1)
            return out

        result = run_native_h3_tile_refine(
            video,
            source_audio_latent,
            plan,
            encode_video=lambda pixels: {"samples": video_vae.encode(pixels)},
            sample_joint=sample_joint,
            decode_video=lambda latent: video_vae.decode(latent["samples"]),
        )
        final_video_latent = {"samples": video_vae.encode(result.video.detach().clone())}
        final_latent = concat_h3_av_latent(final_video_latent, source_audio_latent)
        final_info = validate_h3_av_latent(final_latent, strict_audio_length=True)
        if (final_info.width, final_info.height, final_info.frames) != (width, height, frames):
            raise MMH3ResourceError("MMH3 native tile final VideoVAE output changed target geometry")
        if nested_parts(final_latent["samples"])[1] is not source_audio_latent["samples"]:
            raise MMH3ResourceError("MMH3 native tile finalization lost exact source-audio identity")
        summary = result.summary() + " · final=derived/exact-audio"
        if control_plan is not None:
            summary += f" · control={control_plan.config.control_kind}/per-tile"
        elif control_process_info is not None:
            summary += " · control=pass-through"
        return io.NodeOutput(
            final_latent,
            result.video,
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
            json.dumps(result.spatial_run.report, ensure_ascii=False, indent=2),
            json.dumps(result.report, ensure_ascii=False, indent=2),
            summary,
            (
                json.dumps(control_process_info, ensure_ascii=False, indent=2)
                if control_process_info is not None
                else ""
            ),
        )


class MMH3H3LatentStitchUpscaleTarget(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3LatentStitchUpscaleTarget",
            display_name="MMH3 H3 Latent Stitch Upscale Target",
            category=CATEGORY,
            description=(
                "F05 segment-upscale continuation target. Uses the learned-upscaled current video as the HR initializer, "
                "optionally overwrites its prefix with the exact joint-AV tail of the previous HR sampler output, and "
                "emits provenance that Latent Stitch can validate. Leave previous_high_packet disconnected for segment 1."
            ),
            inputs=[
                MMH3.Input("source_packet"),
                io.Latent.Input("upscaled_video_latent"),
                io.Latent.Input("source_audio_latent"),
                MMH3.Input("previous_high_packet", optional=True),
                io.String.Input("upscale_process_info_json", default="", multiline=True, optional=True, advanced=True),
                io.String.Input("conditioning_info_json", default="", multiline=True, optional=True, advanced=True),
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.String.Output("operation"),
                io.String.Output("mode"),
                io.String.Output("status"),
                io.String.Output("process_info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        source_packet,
        upscaled_video_latent,
        source_audio_latent,
        previous_high_packet=None,
        upscale_process_info_json: str = "",
        conditioning_info_json: str = "",
    ) -> io.NodeOutput:
        upscale_process_info = (
            _parse_object(upscale_process_info_json, "upscale_process_info_json")
            if str(upscale_process_info_json or "").strip()
            else None
        )
        conditioning_info = (
            _parse_object(conditioning_info_json, "conditioning_info_json")
            if str(conditioning_info_json or "").strip()
            else None
        )
        result = build_latent_stitch_upscale_target(
            _packet(source_packet),
            upscaled_video_latent,
            source_audio_latent,
            previous_high_packet=None if previous_high_packet is None else _packet(previous_high_packet),
            upscale_process_info=upscale_process_info,
            conditioning_info=conditioning_info,
        )
        return io.NodeOutput(
            result.latent,
            result.operation,
            result.mode,
            result.summary,
            json.dumps(result.process_info, ensure_ascii=False, indent=2),
        )
