from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from .archive import get_resource_payload
from .constants import AUDIO_SAMPLE_RATE, DIT_SPATIAL_PATCH, FPS, LATENT_SPATIAL_DIVISOR
from .continuation import h3_audio_t_from_exact_frames, h3_video_t_from_frames, is_exact_h3_av_handover_boundary
from .core import MMH3Media
from .errors import MMH3ResourceError
from .decoded_continuation import _default_audio_resampler, _normalize_canvas, _resample_frames, _stereo_waveform
from .h3_contract import h3_latent_contract_from_resource
from .h3 import H3LatentInfo, concat_h3_av_latent, make_nested_tensor, nested_parts, split_h3_av_latent, validate_h3_av_latent
from .lora_provenance import normalize_generation_loras
from .spatial_tiles import SpatialTilePlan, SpatialTileRunResult, validate_masked_spatial_tile_run_report, validate_spatial_tile_plan_report, validate_spatial_tile_run_report
from .tile_backend import TileBackendFinalizeResult, validate_external_tile_finalize_report
from .util import deep_copy_json


UPSCALE_GEOMETRY_MODES = ("scale", "target_dimensions", "target_megapixels")


@dataclass(frozen=True)
class LatentUpscalePlan:
    geometry_mode: str
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    frames: int
    requested_scale: float | None
    requested_megapixels: float | None
    actual_scale_x: float
    actual_scale_y: float
    align: int
    enable_chunking: bool
    warnings: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"READY · F07 latent upscale · {self.source_width}x{self.source_height} → "
            f"{self.target_width}x{self.target_height} · "
            f"scale={self.actual_scale_x:.4f}x{self.actual_scale_y:.4f} · "
            f"audio=preserve · chunking={'on' if self.enable_chunking else 'off'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "contract": "mmh3_f07_latent_upscale_prepare_v1",
            "geometry_mode": self.geometry_mode,
            "source": {
                "width": self.source_width,
                "height": self.source_height,
                "frames": self.frames,
            },
            "target": {
                "width": self.target_width,
                "height": self.target_height,
                "frames": self.frames,
            },
            "requested_scale": self.requested_scale,
            "requested_megapixels": self.requested_megapixels,
            "actual_scale": {"x": self.actual_scale_x, "y": self.actual_scale_y},
            "align": self.align,
            "enable_chunking": self.enable_chunking,
            "audio_policy": "preserve_exact_latent_stream",
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PreparedLatentUpscale:
    packet: MMH3Media
    video_latent: dict[str, Any]
    audio_latent: dict[str, Any]
    plan: LatentUpscalePlan
    source_resource_id: str
    source_adapter: dict[str, Any] | None = None


@dataclass(frozen=True)
class LatentUpscaleProcessReport:
    info: dict[str, Any]
    applied_loras: tuple[dict[str, Any], ...]

    def summary(self) -> str:
        target = self.info["geometry"]["target"]
        tile = self.info["tile_policy"]
        if not tile["enabled"]:
            tile_summary = "tiles=full-frame"
        elif tile["mode"] == "spatial_tiles":
            tile_summary = f"tiles={len(tile['plan']['tiles'])}"
        else:
            tile_summary = "tiles=external-backend"
        return (
            f"READY · F07 upscale/refine report · {target['width']}x{target['height']} · "
            f"source_loras={len(self.info['source_lora_reapply']['applied'])} · "
            f"process_loras={len(self.info['process_loras'])} · {tile_summary} · "
            f"sigmas={self.info['refine']['sigma_profile']}"
        )

    def to_dict(self) -> dict[str, Any]:
        return deep_copy_json(self.info)


def _aligned_nearest(value: float, align: int) -> int:
    # Mirrors the external learned upscaler's pixel-space alignment exactly.
    return max(align, int(round(float(value) / align)) * align)


def plan_latent_upscale_geometry(
    info: H3LatentInfo,
    *,
    geometry_mode: str = "scale",
    scale: float = 2.0,
    target_width: int = 0,
    target_height: int = 0,
    target_megapixels: float = 0.0,
    align: int = 32,
    enable_chunking: bool = True,
) -> LatentUpscalePlan:
    if geometry_mode not in UPSCALE_GEOMETRY_MODES:
        raise MMH3ResourceError(f"Unsupported F07 geometry mode {geometry_mode!r}")
    align = int(align)
    required_align = LATENT_SPATIAL_DIVISOR * DIT_SPATIAL_PATCH
    if align < required_align or align % required_align:
        raise MMH3ResourceError(f"F07 align must be a multiple of {required_align}")
    if info.frames is None:
        raise MMH3ResourceError("F07 requires a source latent on the stock H3 temporal grid")
    if info.batch != 1:
        raise MMH3ResourceError("F07 packet-native upscale currently requires batch=1")
    source_width, source_height = int(info.width), int(info.height)
    requested_scale: float | None = None
    requested_megapixels: float | None = None
    warnings: list[str] = []

    if geometry_mode == "scale":
        requested_scale = float(scale)
        if not math.isfinite(requested_scale) or requested_scale < 1.0 or requested_scale > 4.0:
            raise MMH3ResourceError("F07 scale must be finite and within 1–4x")
        width = _aligned_nearest(source_width * requested_scale, align)
        height = _aligned_nearest(source_height * requested_scale, align)
    elif geometry_mode == "target_megapixels":
        requested_megapixels = float(target_megapixels)
        if not math.isfinite(requested_megapixels) or requested_megapixels < 0.1 or requested_megapixels > 8.0:
            raise MMH3ResourceError("F07 target_megapixels must be finite and within 0.1–8.0")
        # The current learned upscaler defines 1 MP as 1024x1024.
        target_area = requested_megapixels * 1024.0 * 1024.0
        aspect = source_width / source_height
        width = _aligned_nearest(math.sqrt(target_area * aspect), align)
        height = _aligned_nearest(math.sqrt(target_area / aspect), align)
    else:
        width, height = int(target_width), int(target_height)
        if width <= 0 or height <= 0:
            raise MMH3ResourceError("F07 target_dimensions requires positive target_width and target_height")
        if width > 4096 or height > 4096:
            raise MMH3ResourceError("F07 target_dimensions follows the external upscaler limit of 4096 pixels per axis")
        if width % align or height % align:
            raise MMH3ResourceError(f"F07 target dimensions must be divisible by align={align}")
        source_aspect = source_width / source_height
        target_aspect = width / height
        if abs(target_aspect / source_aspect - 1.0) > 0.02:
            warnings.append("Explicit target dimensions change source aspect ratio by more than 2%.")

    scale_x, scale_y = width / source_width, height / source_height
    if scale_x < 1.0 or scale_y < 1.0:
        raise MMH3ResourceError("F07 latent upscale cannot downscale either spatial axis")
    if scale_x > 4.0 or scale_y > 4.0:
        raise MMH3ResourceError("F07 target exceeds the supported 4x limit on a spatial axis")
    return LatentUpscalePlan(
        geometry_mode,
        source_width,
        source_height,
        width,
        height,
        int(info.frames),
        requested_scale,
        requested_megapixels,
        scale_x,
        scale_y,
        align,
        bool(enable_chunking),
        tuple(warnings),
    )


def prepare_packet_latent_upscale(
    packet: MMH3Media,
    *,
    geometry_mode: str = "scale",
    scale: float = 2.0,
    target_width: int = 0,
    target_height: int = 0,
    target_megapixels: float = 0.0,
    align: int = 32,
    enable_chunking: bool = True,
) -> PreparedLatentUpscale:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    descriptor = packet.get_primary("latent")
    if descriptor is None or descriptor.get("kind") != "latent":
        raise MMH3ResourceError("F07 requires a primary H3 latent resource")
    origin = h3_latent_contract_from_resource(descriptor).get("origin", "unknown")
    if origin != "sampler_output":
        raise MMH3ResourceError(f"F07 high-sigma refine requires sampler_output provenance; got {origin!r}")
    latent = get_resource_payload(packet, descriptor)
    info = validate_h3_av_latent(latent, strict_audio_length=True)
    plan = plan_latent_upscale_geometry(
        info,
        geometry_mode=geometry_mode,
        scale=scale,
        target_width=target_width,
        target_height=target_height,
        target_megapixels=target_megapixels,
        align=align,
        enable_chunking=enable_chunking,
    )
    # SamplerCustomAdvanced forwards its input mask into the completed output.
    # It belongs to the previous sampling stage; the HR target builds its own
    # protection mask from continuation lineage. Keep the source packet intact.
    video_latent, audio_latent = split_h3_av_latent({"samples": latent["samples"]})
    return PreparedLatentUpscale(packet, video_latent, audio_latent, plan, descriptor["id"])


def prepare_decoded_packet_latent_upscale(
    packet: MMH3Media,
    *,
    video_vae: Any,
    audio_vae: Any,
    missing_audio_policy: str = "error",
    geometry_mode: str = "target_megapixels",
    scale: float = 2.0,
    target_width: int = 0,
    target_height: int = 0,
    target_megapixels: float = 2.1,
    align: int = 32,
    enable_chunking: bool = True,
) -> PreparedLatentUpscale:
    """Encode primary decoded AV as an F07 source before learned latent upscale.

    The decoded timeline is normalized to 24 FPS and trimmed to the largest
    complete H3 17n+5 grid. Audio is stereo/32 kHz and duration-locked before
    VAE encoding. This is intentionally a lossy ``vae_encoded`` entry path.
    """
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    if missing_audio_policy not in {"error", "silence"}:
        raise MMH3ResourceError("missing_audio_policy must be 'error' or 'silence'")
    video_resource = packet.get_primary("video")
    if video_resource is None or video_resource.get("kind") != "video":
        raise MMH3ResourceError("F07 decoded upscale requires a primary decoded video")
    video = get_resource_payload(packet, video_resource)
    if not hasattr(video, "get_components"):
        raise MMH3ResourceError("Decoded video does not expose get_components()")
    components = video.get_components()
    frames = getattr(components, "images", None)
    try:
        source_fps = float(getattr(components, "frame_rate", None))
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError("Decoded video has an invalid frame rate") from exc
    normalized = _resample_frames(frames, source_fps, float(FPS))
    available_frames = int(normalized.shape[0])
    if available_frames < 5:
        raise MMH3ResourceError("F07 decoded upscale requires at least 5 frames at 24 FPS")
    h3_frames = 5 + 17 * ((available_frames - 5) // 17)
    source_h, source_w = int(normalized.shape[1]), int(normalized.shape[2])
    canvas_width = max(32, (source_w // 32) * 32)
    canvas_height = max(32, (source_h // 32) * 32)
    normalized_frames = _normalize_canvas(normalized[:h3_frames], canvas_width, canvas_height)

    audio_resource = packet.get_primary("audio")
    selected_audio = get_resource_payload(packet, audio_resource) if audio_resource is not None else getattr(components, "audio", None)
    required_samples = h3_frames * AUDIO_SAMPLE_RATE // FPS
    used_silence = False
    source_audio_rate: int | None = None
    if selected_audio is None:
        if missing_audio_policy == "error":
            raise MMH3ResourceError("F07 decoded upscale requires audio or explicit silence policy")
        waveform = normalized_frames.new_zeros((1, 2, required_samples))
        used_silence = True
    else:
        waveform, source_audio_rate = _stereo_waveform(selected_audio)
        if source_audio_rate != AUDIO_SAMPLE_RATE:
            waveform = _default_audio_resampler(waveform, source_audio_rate, AUDIO_SAMPLE_RATE)
        if int(waveform.shape[-1]) < required_samples:
            if missing_audio_policy == "error":
                raise MMH3ResourceError("Decoded audio is too short for the normalized F07 timeline")
            waveform = torch.nn.functional.pad(waveform, (0, required_samples - int(waveform.shape[-1])))
            used_silence = True
        waveform = waveform[..., :required_samples]

    audio_vae_rate = int(getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE))
    if audio_vae_rate != AUDIO_SAMPLE_RATE:
        raise MMH3ResourceError(f"F07 decoded upscale requires a 32 kHz audio VAE; got {audio_vae_rate}")
    video_latent = {"samples": video_vae.encode(normalized_frames)}
    audio_latent = {"samples": audio_vae.encode(waveform.movedim(1, -1))}
    joint = concat_h3_av_latent(video_latent, audio_latent)
    info = validate_h3_av_latent(joint, strict_audio_length=True)
    plan = plan_latent_upscale_geometry(
        info,
        geometry_mode=geometry_mode,
        scale=scale,
        target_width=target_width,
        target_height=target_height,
        target_megapixels=target_megapixels,
        align=align,
        enable_chunking=enable_chunking,
    )
    warnings = list(plan.warnings)
    if h3_frames != available_frames:
        warnings.append(f"Decoded timeline trimmed from {available_frames} to {h3_frames} frames for the H3 17n+5 grid.")
        plan = replace(plan, warnings=tuple(warnings))
    source_adapter = {
        "version": 1,
        "contract": "mmh3_f07_decoded_upscale_source_v1",
        "latent_origin": "vae_encoded",
        "source_resource_ids": {
            "video": video_resource["id"],
            **({"audio": audio_resource["id"]} if audio_resource is not None else {}),
        },
        "source_fps": source_fps,
        "normalized_fps": FPS,
        "source_frames": available_frames,
        "frames": h3_frames,
        "canvas": {"width": canvas_width, "height": canvas_height},
        "audio_source": "packet_audio" if audio_resource is not None else ("embedded_audio" if selected_audio is not None else "silence"),
        "source_audio_sample_rate": source_audio_rate,
        "audio_samples": required_samples,
        "used_silence": used_silence,
    }
    return PreparedLatentUpscale(packet, video_latent, audio_latent, plan, video_resource["id"], source_adapter)


def build_latent_upscale_process_report(
    geometry_report: Mapping[str, Any],
    source_lora_report: Mapping[str, Any],
    *,
    process_loras: Sequence[Mapping[str, Any]] = (),
    upscaler_model: str,
    device: str,
    precision: str,
    sigma_profile: str,
    execution_profile: Mapping[str, Any] | None = None,
    refine_sampling: Mapping[str, Any] | None = None,
    tile_plan: SpatialTilePlan | Mapping[str, Any] | None = None,
    tile_run_report: SpatialTileRunResult | Mapping[str, Any] | None = None,
    native_tile_adapter_report: Mapping[str, Any] | None = None,
    external_tile_report: TileBackendFinalizeResult | Mapping[str, Any] | None = None,
) -> LatentUpscaleProcessReport:
    geometry = deep_copy_json(dict(geometry_report))
    if geometry.get("contract") != "mmh3_f07_latent_upscale_prepare_v1":
        raise MMH3ResourceError("F07 process report requires the versioned prepare geometry contract")
    lora_report = deep_copy_json(dict(source_lora_report))
    if lora_report.get("version") != 1 or lora_report.get("ready") is not True:
        raise MMH3ResourceError("F07 process report requires a READY source LoRA reapply report v1")
    source_applied_raw = lora_report.get("applied")
    if not isinstance(source_applied_raw, list):
        raise MMH3ResourceError("F07 source LoRA report is missing its applied array")
    source_applied = normalize_generation_loras(
        [{key: value for key, value in entry.items() if key != "source_index"} for entry in source_applied_raw]
    )
    normalized_process = normalize_generation_loras(process_loras)
    upscaler_model = str(upscaler_model or "").strip()
    if not upscaler_model:
        raise MMH3ResourceError("F07 process report requires the learned upscaler model name")
    if device not in {"cuda", "rocm", "cpu"}:
        raise MMH3ResourceError(f"Unsupported F07 upscaler device {device!r}")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise MMH3ResourceError(f"Unsupported F07 upscaler precision {precision!r}")
    if sigma_profile not in {"3_steps", "4_steps", "5_steps", "fast_res2m_4step", "source_aware"}:
        raise MMH3ResourceError(f"Unsupported F07 sigma profile {sigma_profile!r}")
    normalized_execution_profile: dict[str, Any] | None = None
    if execution_profile is not None:
        normalized_execution_profile = deep_copy_json(dict(execution_profile))
        if (
            normalized_execution_profile.get("version") != 1
            or normalized_execution_profile.get("contract") != "mmh3_f07_execution_profile_v1"
        ):
            raise MMH3ResourceError("F07 execution profile requires mmh3_f07_execution_profile_v1")
    normalized_refine_sampling: dict[str, Any] | None = None
    if refine_sampling is not None:
        normalized_refine_sampling = deep_copy_json(dict(refine_sampling))
        if (
            normalized_refine_sampling.get("version") != 1
            or normalized_refine_sampling.get("contract") != "mmh3_f05_upscale_refine_sampling_v1"
        ):
            raise MMH3ResourceError("F05/F07 source-aware refine sampling requires mmh3_f05_upscale_refine_sampling_v1")
    if sigma_profile == "source_aware" and normalized_refine_sampling is None:
        raise MMH3ResourceError("F07 sigma_profile=source_aware requires refine_sampling provenance")
    if external_tile_report is not None:
        if tile_plan is not None or tile_run_report is not None or native_tile_adapter_report is not None:
            raise MMH3ResourceError("F07 external tile finalize and shared tile-run contracts cannot be mixed")
        raw_external_report = (
            external_tile_report.report
            if isinstance(external_tile_report, TileBackendFinalizeResult)
            else external_tile_report
        )
        target = geometry["target"]
        tile_policy = {
            "enabled": True,
            "mode": "external_decoded_backend",
            "finalize": validate_external_tile_finalize_report(
                raw_external_report,
                target_width=target["width"],
                target_height=target["height"],
                frames=target["frames"],
            ),
            "peak_vram_reduction_claimed": False,
        }
    elif tile_plan is None:
        if tile_run_report is not None or native_tile_adapter_report is not None:
            raise MMH3ResourceError("F07 tile run report requires its owning spatial tile plan")
        tile_policy: dict[str, Any] = {
            "enabled": False,
            "mode": "full_frame",
            "peak_vram_reduction_claimed": False,
        }
    else:
        raw_tile_plan = tile_plan.to_dict() if isinstance(tile_plan, SpatialTilePlan) else tile_plan
        target = geometry["target"]
        tile_policy = {
            "enabled": True,
            "mode": "spatial_tiles",
            "plan": validate_spatial_tile_plan_report(
                raw_tile_plan,
                target_width=target["width"],
                target_height=target["height"],
                frames=target["frames"],
            ),
            "peak_vram_reduction_claimed": False,
        }
        if tile_run_report is not None:
            raw_run_report = tile_run_report.report if isinstance(tile_run_report, SpatialTileRunResult) else tile_run_report
            if raw_run_report.get("contract") == "mmh3_spatial_masked_tile_run_v1":
                tile_policy["run"] = validate_masked_spatial_tile_run_report(raw_run_report, plan=tile_policy["plan"])
            else:
                tile_policy["run"] = validate_spatial_tile_run_report(raw_run_report, plan=tile_policy["plan"])
        if native_tile_adapter_report is not None:
            if "run" not in tile_policy:
                raise MMH3ResourceError("F07 native tile adapter report requires a completed tile run")
            adapter = deep_copy_json(dict(native_tile_adapter_report))
            expected_sampler = {
                "batch_size": 1,
                "timeline_per_tile": "complete",
                "video_noise_mask": 1,
                "audio_noise_mask": 0,
                "sampled_audio_action": "discard",
            }
            observed = adapter.get("observed")
            if (
                adapter.get("version") != 1
                or adapter.get("contract") != "mmh3_h3_native_tile_refine_v1"
                or adapter.get("operation") != "native_h3_spatial_tile_refine"
                or adapter.get("plan") != tile_policy["plan"]
                or adapter.get("spatial_run") != tile_policy["run"]
                or adapter.get("sampler") != expected_sampler
                or adapter.get("final_audio_owner")
                != "source_audio_latent_recombined_after_full_video_vae_encode"
                or not isinstance(observed, dict)
                or observed.get("tiles_sampled") != len(tile_policy["plan"]["tiles"])
                or observed.get("source_audio_identity_preserved_in_each_target") is not True
                or observed.get("peak_vram_measured") is not False
            ):
                raise MMH3ResourceError("F07 native tile adapter report does not match canonical H3 sampler ownership")
            tile_policy["native_adapter"] = adapter
    # Process LoRAs are loaded onto the base refine model first; recorded source
    # LoRAs are then reapplied by the expandable adapter in original source order.
    actual_stack = normalized_process + source_applied
    refine_info: dict[str, Any] = {"sigma_profile": sigma_profile}
    if normalized_execution_profile is not None:
        refine_info["execution_profile"] = normalized_execution_profile
    if normalized_refine_sampling is not None:
        refine_info["source_aware_sampling"] = normalized_refine_sampling
    info = {
        "version": 1,
        "contract": "mmh3_f07_latent_upscale_refine_v1",
        "operation": "latent_upscale_refine",
        "geometry": geometry,
        "upscaler": {
            "model": upscaler_model,
            "device": device,
            "precision": precision,
            "enable_chunking": bool(geometry.get("enable_chunking", False)),
        },
        "source_lora_reapply": lora_report,
        "process_loras": deep_copy_json(normalized_process),
        "actual_model_loras": deep_copy_json(actual_stack),
        "refine": refine_info,
        "tile_policy": tile_policy,
        "audio_policy": "preserve_exact_source_latent_after_joint_refine",
    }
    return LatentUpscaleProcessReport(info, tuple(actual_stack))


def build_latent_upscale_refine_target(
    video_latent: dict[str, Any],
    audio_latent: dict[str, Any],
) -> dict[str, Any]:
    """Create a video-only high-sigma target with native per-stream AV masks.

    Video receives denoise mask 1; source audio receives mask 0 and is also
    recombined again after sampling by the canonical workflow.
    """
    joint = concat_h3_av_latent(video_latent, audio_latent)
    video = video_latent["samples"]
    audio = audio_latent["samples"]
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise MMH3ResourceError("F07 refine target requires separated tensor video/audio latents")
    output = dict(joint)
    output["noise_mask"] = make_nested_tensor(
        [torch.ones_like(video), torch.zeros_like(audio)]
    )
    return output


@dataclass(frozen=True)
class LatentStitchUpscaleTarget:
    """Prepared high-resolution refine target for an F05 stitch-upscale chain."""

    latent: dict[str, Any]
    process_info: dict[str, Any]
    operation: str
    mode: str
    summary: str


@dataclass(frozen=True)
class LatentUpscaleRefineSampling:
    """Source-aware low-sigma refine recipe for packet-native H3 upscale."""

    task_family: str
    steps: int
    video_shift: float
    audio_shift: float
    sampler: str
    scheduler: str
    denoise: float
    info: dict[str, Any]

    def summary(self) -> str:
        return (
            f"READY · F05 source-aware refine · {self.task_family} · "
            f"{self.sampler}/{self.scheduler} · steps={self.steps} · denoise={self.denoise:.3f}"
        )


def _packet_task_family(packet: MMH3Media) -> str:
    ext = packet.manifest.get("extensions", {})
    mmh3 = ext.get("mmh3_media") if isinstance(ext, Mapping) else None
    process = mmh3.get("last_process") if isinstance(mmh3, Mapping) else None
    process = process if isinstance(process, Mapping) else {}
    info = process.get("info") if isinstance(process.get("info"), Mapping) else {}
    family = str(info.get("task_family") or "")
    if family in {"fl2va", "ref2va"}:
        return family
    mode = str(process.get("mode") or packet.manifest.get("generation", {}).get("task") or "")
    if mode == "ref2va":
        return "ref2va"
    if mode in {"t2va", "i2va", "l2va", "fl2va"}:
        return "fl2va"
    return "unknown"


def build_latent_upscale_refine_sampling(
    packet: MMH3Media,
    *,
    denoise_override: float = 0.0,
) -> LatentUpscaleRefineSampling:
    """Derive a low-sigma HR refine schedule from the source MMH3 generation contract.

    The upscale second pass must not silently replace the source trajectory family with a
    generic high-sigma Euler recipe.  We inherit task family, sampler/scheduler, step grid
    and AV shifts recorded by the source packet, then truncate that *same* trajectory to a
    conservative tail using ``BasicScheduler.denoise``.  This keeps Turbo/custom sources on
    their own sampling grid while reducing context drift from excessive re-noising.
    """
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("F05 refine sampling expects an MMH3_MEDIA packet")
    family = _packet_task_family(packet)
    if family not in {"fl2va", "ref2va"}:
        raise MMH3ResourceError("F05 refine sampling cannot resolve FL2VA/Ref2VA task family from the source packet")
    process, info = _last_process_info(packet)
    raw_sampling = info.get("sampling") if isinstance(info.get("sampling"), Mapping) else None
    warnings: list[str] = []
    inherited = raw_sampling is not None

    if raw_sampling is None:
        # Legacy/base packets may predate sampling-profile persistence.  Falling back to the
        # stock base recipe is only safe when no known trajectory-distilled adapter is present.
        loras = packet.manifest.get("generation", {}).get("loras", [])
        names = [str(item.get("name") or "").lower() for item in loras if isinstance(item, Mapping)]
        trajectory_tokens = ("turbo", "pdd", "minimax-h3-acc", "minimax_h3_acc", "fasth3", "fast_h3")
        if any(any(token in name for token in trajectory_tokens) for name in names):
            raise MMH3ResourceError(
                "F05 upscale source uses a trajectory-specific acceleration LoRA but has no recorded sampling profile; "
                "repack/regenerate the segment with sampling provenance instead of guessing a refine schedule"
            )
        raw_sampling = {
            "version": 2,
            "contract": "mmh3_h3_sampling_preset_v2",
            "profile": "standard (20 steps)",
            "task_family": family,
            "steps": 20,
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "sigma_preset": "scheduler_generated",
            "runtime_validated": True,
        }
        warnings.append("Source packet lacked sampling provenance; stock H3 20-step sampling was inferred.")

    sampling = deep_copy_json(dict(raw_sampling))
    contract = str(sampling.get("contract") or "")
    if contract != "mmh3_h3_sampling_preset_v2":
        raise MMH3ResourceError(
            f"F05 upscale does not know how to preserve source sampling contract {contract!r}; "
            "trajectory-coupled methods require an explicit compatible refine adapter"
        )
    recorded_family = str(sampling.get("task_family") or family)
    if recorded_family != family:
        raise MMH3ResourceError(
            f"F05 upscale sampling/task-family mismatch: packet={family}, sampling={recorded_family}"
        )
    sigma_preset = str(sampling.get("sigma_preset") or "scheduler_generated")
    if sigma_preset != "scheduler_generated":
        raise MMH3ResourceError(
            f"F05 upscale requires a scheduler-generated source trajectory; got sigma_preset={sigma_preset!r}"
        )
    try:
        steps = int(sampling["steps"])
        video_shift = float(sampling["video_shift"])
        audio_shift = float(sampling["audio_shift"])
        sampler = str(sampling["sampler"])
        scheduler = str(sampling["scheduler"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MMH3ResourceError("F05 source sampling provenance is incomplete") from exc
    if steps < 1 or not sampler or not scheduler:
        raise MMH3ResourceError("F05 source sampling provenance contains invalid steps/sampler/scheduler")

    profile = str(sampling.get("profile") or "custom").lower()
    from .fasth3 import FASTH3_PROFILE
    if profile == FASTH3_PROFILE:
        raise MMH3ResourceError("FastH3 dense has no validated low-sigma refine recipe; use decoded video upscale")
    if denoise_override and float(denoise_override) > 0:
        denoise = float(denoise_override)
    elif "turbo (4" in profile:
        # Keep two trained trajectory intervals rather than restarting near sigma~1.
        denoise = 0.50
    elif "turbo (8" in profile:
        # Keep roughly the final three intervals.
        denoise = 0.375
    else:
        # Base/custom: approximately the final quarter of the source trajectory.
        denoise = 0.25
    if not (0.05 <= denoise <= 0.50):
        raise MMH3ResourceError("F05 refine denoise must stay within 0.05–0.50 to protect upscale context")

    refine = {
        "version": 1,
        "contract": "mmh3_f05_upscale_refine_sampling_v1",
        "task_family": family,
        "conditioning_mode": str(process.get("mode") or ""),
        "source_sampling_inherited": inherited,
        "source_sampling": sampling,
        "refine": {
            "steps": steps,
            "video_shift": video_shift,
            "audio_shift": audio_shift,
            "sampler": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
            "policy": "truncate_source_scheduler_tail",
            "max_denoise": 0.50,
        },
        "warnings": warnings,
    }
    return LatentUpscaleRefineSampling(
        family, steps, video_shift, audio_shift, sampler, scheduler, denoise, refine
    )


def _last_process_info(packet: MMH3Media) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    ext = packet.manifest.get("extensions", {})
    mmh3 = ext.get("mmh3_media") if isinstance(ext, Mapping) else None
    process = mmh3.get("last_process") if isinstance(mmh3, Mapping) else None
    process = process if isinstance(process, Mapping) else {}
    info = process.get("info") if isinstance(process.get("info"), Mapping) else {}
    return process, info


def _source_upscale_identity(packet: MMH3Media) -> tuple[str, str, str]:
    resource = packet.get_primary("latent")
    if resource is None or resource.get("kind") != "latent":
        raise MMH3ResourceError("F05 stitch-upscale source packet requires a primary H3 latent")
    contract = h3_latent_contract_from_resource(resource)
    if contract.get("origin") != "sampler_output":
        raise MMH3ResourceError(
            "F05 stitch-upscale requires low-resolution source segments with sampler_output provenance"
        )
    revision = str((resource.get("content") or {}).get("revision") or "")
    if not revision:
        raise MMH3ResourceError("F05 stitch-upscale source latent is missing immutable resource revision")
    return packet.manifest["id"], resource["id"], revision


def build_latent_stitch_upscale_target(
    source_packet: MMH3Media,
    upscaled_video_latent: dict[str, Any],
    source_audio_latent: dict[str, Any],
    *,
    previous_high_packet: MMH3Media | None = None,
    upscale_process_info: Mapping[str, Any] | None = None,
    conditioning_info: Mapping[str, Any] | None = None,
) -> LatentStitchUpscaleTarget:
    """Build one member of a segment-upscale -> HR-continuation -> latent-stitch chain.

    The learned upscaled video latent initializes the full current segment.  For segment 2+
    the exact joint-AV tail from the previous *high-resolution sampler output* overwrites the
    current prefix.  Video outside the prefix is denoised; audio is preserved exactly after
    the prefix replacement.  This creates a true HR continuation lineage rather than a set of
    independently refined clips that only happen to share a low-resolution ancestor.
    """
    if not isinstance(source_packet, MMH3Media):
        raise MMH3ResourceError("F05 stitch-upscale expects source_packet to be MMH3_MEDIA")
    source_packet_id, source_resource_id, source_revision = _source_upscale_identity(source_packet)
    source_resource = source_packet.get_primary("latent")
    assert source_resource is not None
    source_contract = h3_latent_contract_from_resource(source_resource)
    source_frames = source_contract["timeline"].get("frames")
    if not isinstance(source_frames, int):
        raise MMH3ResourceError("F05 stitch-upscale source must be on the stock H3 temporal grid")

    video = upscaled_video_latent.get("samples")
    audio = source_audio_latent.get("samples")
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise MMH3ResourceError("F05 stitch-upscale requires tensor video/audio latents")
    if video.ndim != 5 or audio.ndim != 4:
        raise MMH3ResourceError("F05 stitch-upscale received invalid separated H3 latent ranks")

    # Validate the target joint latent before adding masks. Audio length also proves the
    # current segment duration while the upscaled video proves the target spatial geometry.
    current_joint = concat_h3_av_latent(upscaled_video_latent, source_audio_latent)
    current_info = validate_h3_av_latent(current_joint, strict_audio_length=True)
    if current_info.frames != source_frames:
        raise MMH3ResourceError(
            f"F05 stitch-upscale temporal mismatch: source={source_frames}f, upscaled target={current_info.frames}f"
        )
    if current_info.batch != 1:
        raise MMH3ResourceError("F05 stitch-upscale currently requires batch=1")

    task_family = _packet_task_family(source_packet)
    process, source_process_info = _last_process_info(source_packet)
    normalized_upscale_process: dict[str, Any] | None = None
    if upscale_process_info is not None:
        normalized_upscale_process = deep_copy_json(dict(upscale_process_info))
        if normalized_upscale_process.get("contract") != "mmh3_f07_latent_upscale_refine_v1":
            raise MMH3ResourceError("F05 stitch-upscale requires an F07 latent-upscale refine report when supplied")
    normalized_conditioning: dict[str, Any] | None = None
    if conditioning_info is not None:
        normalized_conditioning = deep_copy_json(dict(conditioning_info))
        conditioning_family = str(normalized_conditioning.get("task_family") or "")
        if conditioning_family and conditioning_family != task_family:
            raise MMH3ResourceError(
                f"F05 upscale conditioning/task-family mismatch: source={task_family}, conditioning={conditioning_family}"
            )

    upscale_info: dict[str, Any] = {
        "version": 1,
        "contract": "mmh3_f05_latent_stitch_upscale_v1",
        "strategy": "segment_upscale_then_high_res_continuation_then_latent_stitch",
        "upscale_source_packet_id": source_packet_id,
        "upscale_source_latent_resource_id": source_resource_id,
        "upscale_source_latent_revision": source_revision,
        "source_canvas": deep_copy_json(source_contract["canvas"]),
        "target_canvas": {"width": current_info.width, "height": current_info.height},
        "frames": source_frames,
        "task_family": task_family,
        "video_initializer": "learned_upscaled_low_res_segment",
        "audio_policy": "preserve_low_res_segment_audio_except_exact_hr_handover_prefix",
        **({"upscale_refine": normalized_upscale_process} if normalized_upscale_process is not None else {}),
        **({"conditioning": normalized_conditioning} if normalized_conditioning is not None else {}),
    }

    target_video = video.clone()
    target_audio = audio.clone()
    video_mask = torch.ones_like(target_video)
    audio_mask = torch.zeros_like(target_audio)

    if previous_high_packet is None:
        info = {
            **upscale_info,
            "segment_role": "first",
            "task_family": task_family,
            "conditioning_mode": str(process.get("mode") or ""),
            "video_mask_policy": "denoise_full_upscaled_video",
            "audio_mask_policy": "preserve_exact_source_audio",
        }
        latent = {
            "samples": make_nested_tensor([target_video, target_audio]),
            "noise_mask": make_nested_tensor([video_mask, audio_mask]),
        }
        validate_h3_av_latent(latent, strict_audio_length=True)
        return LatentStitchUpscaleTarget(
            latent=latent,
            process_info=info,
            operation="latent_upscale_refine",
            mode=task_family if task_family != "unknown" else str(process.get("mode") or ""),
            summary=(
                f"READY · F05 stitch-upscale first · {current_info.width}x{current_info.height} · "
                f"{source_frames}f · family={task_family}"
            ),
        )

    if not isinstance(previous_high_packet, MMH3Media):
        raise MMH3ResourceError("previous_high_packet must be MMH3_MEDIA when connected")
    previous_resource = previous_high_packet.get_primary("latent")
    if previous_resource is None or previous_resource.get("kind") != "latent":
        raise MMH3ResourceError("Previous high-resolution packet has no primary H3 latent")
    previous_contract = h3_latent_contract_from_resource(previous_resource)
    if previous_contract.get("origin") != "sampler_output":
        raise MMH3ResourceError(
            "F05 stitch-upscale previous high-resolution segment must be a direct sampler_output"
        )
    previous_revision = str((previous_resource.get("content") or {}).get("revision") or "")
    if not previous_revision:
        raise MMH3ResourceError("Previous high-resolution latent is missing immutable resource revision")
    previous_latent = get_resource_payload(previous_high_packet, previous_resource)
    previous_info = validate_h3_av_latent(previous_latent, strict_audio_length=True)
    if previous_info.batch != 1:
        raise MMH3ResourceError("Previous high-resolution segment must have batch=1")
    if (previous_info.width, previous_info.height) != (current_info.width, current_info.height):
        raise MMH3ResourceError(
            "F05 stitch-upscale requires one high-resolution canvas across the chain: "
            f"previous={previous_info.width}x{previous_info.height}, current={current_info.width}x{current_info.height}"
        )

    # The high-resolution chain must mirror the already-proven low-resolution continuation
    # chain.  This prevents accidentally using an unrelated low-res B with high-res A'.
    if process.get("operation") != "continuation":
        raise MMH3ResourceError(
            "F05 stitch-upscale segment 2+ requires the current low-resolution packet to be a continuation"
        )
    low_parent_packet_id = str(source_process_info.get("source_packet_id") or "")
    low_video_handover = source_process_info.get("video_handover_frames")
    low_audio_handover = source_process_info.get("audio_handover_frames")
    try:
        handover_frames = int(low_video_handover)
        audio_handover_frames = int(low_audio_handover)
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError("Low-resolution continuation is missing valid AV handover metadata") from exc
    if handover_frames != audio_handover_frames or not is_exact_h3_av_handover_boundary(handover_frames):
        raise MMH3ResourceError(
            "F05 stitch-upscale requires the low-resolution chain to use one exact shared AV handover boundary"
        )

    previous_process, previous_process_info = _last_process_info(previous_high_packet)
    previous_upscale_source_id = str(previous_process_info.get("upscale_source_packet_id") or "")
    if not previous_upscale_source_id:
        nested_upscale = previous_process_info.get("upscale")
        if isinstance(nested_upscale, Mapping):
            previous_upscale_source_id = str(nested_upscale.get("upscale_source_packet_id") or "")
    if not previous_upscale_source_id:
        raise MMH3ResourceError(
            "Previous high-resolution packet lacks F05 stitch-upscale source identity; rebuild it with the current workflow"
        )
    low_parent_identity = (low_parent_packet_id,
                           str(source_process_info.get("source_latent_resource_id") or ""),
                           str(source_process_info.get("source_latent_revision") or ""))
    previous_source_identity = (previous_upscale_source_id,
                                str(previous_process_info.get("upscale_source_latent_resource_id") or ""),
                                str(previous_process_info.get("upscale_source_latent_revision") or ""))
    if not all(low_parent_identity) or low_parent_identity != previous_source_identity:
        raise MMH3ResourceError(
            "F05 stitch-upscale low/high lineage mismatch: current low-res segment does not continue "
            "the low-res source used for the previous high-res segment"
        )

    if handover_frames > source_frames or handover_frames > int(previous_info.frames or 0):
        raise MMH3ResourceError("F05 stitch-upscale handover exceeds current or previous segment duration")
    video_handover_t = h3_video_t_from_frames(handover_frames)
    audio_handover_t = h3_audio_t_from_exact_frames(handover_frames, label="handover_frames")
    previous_video, previous_audio = nested_parts(previous_latent["samples"])
    if video_handover_t > target_video.shape[2] or video_handover_t > previous_video.shape[2]:
        raise MMH3ResourceError("F05 stitch-upscale video handover exceeds latent timeline")
    if audio_handover_t > target_audio.shape[-1] or audio_handover_t > previous_audio.shape[-1]:
        raise MMH3ResourceError("F05 stitch-upscale audio handover exceeds latent timeline")

    target_video[:, :, :video_handover_t] = previous_video[:, :, -video_handover_t:]
    target_audio[..., :audio_handover_t] = previous_audio[..., -audio_handover_t:]
    video_mask[:, :, :video_handover_t] = 0.0

    previous_family = _packet_task_family(previous_high_packet)
    continuation_info = {
        "ready": True,
        "contract": "minimax_h3_joint_av_continuation_v1",
        "source_frames": int(previous_info.frames or 0),
        "target_frames": source_frames,
        "video_handover_frames": handover_frames,
        "audio_handover_frames": handover_frames,
        "audio_feather_frames": 0,
        "source_video_t": int(previous_info.video_shape[2]),
        "source_audio_t": int(previous_info.audio_shape[-1]),
        "target_video_t": int(current_info.video_shape[2]),
        "target_audio_t": int(current_info.audio_shape[-1]),
        "copied_video_t": video_handover_t,
        "copied_audio_t": audio_handover_t,
        "feather_audio_t": 0,
        "noise_mask_semantics": "0=preserve, 1=denoise",
        "naive_latent_concat_safe": False,
        "source_packet_id": previous_high_packet.manifest["id"],
        "source_latent_resource_id": previous_resource["id"],
        "source_latent_revision": previous_revision,
        "source_process_operation": str(previous_process.get("operation") or ""),
        "source_process_mode": str(previous_process.get("mode") or ""),
        "source_task_family": previous_family,
        "task_family": task_family,
        "conditioning_mode": str(process.get("mode") or ""),
        "upscale": {
            **upscale_info,
            "segment_role": "continuation",
            "low_res_parent_packet_id": low_parent_packet_id,
            "low_res_handover_frames": handover_frames,
            "previous_high_upscale_source_packet_id": previous_upscale_source_id,
            "video_mask_policy": "protect_hr_prefix_denoise_upscaled_future",
            "audio_mask_policy": "preserve_exact_audio_after_hr_prefix_replacement",
        },
        # Duplicate the current source identity at top-level so the next HR segment can
        # validate low/high lineage without needing to understand nested process variants.
        "upscale_source_packet_id": source_packet_id,
        "upscale_source_latent_resource_id": source_resource_id,
        "upscale_source_latent_revision": source_revision,
    }
    latent = {
        "samples": make_nested_tensor([target_video, target_audio]),
        "noise_mask": make_nested_tensor([video_mask, audio_mask]),
    }
    validate_h3_av_latent(latent, strict_audio_length=True)
    return LatentStitchUpscaleTarget(
        latent=latent,
        process_info=continuation_info,
        operation="continuation",
        mode=task_family if task_family != "unknown" else str(process.get("mode") or ""),
        summary=(
            f"READY · F05 stitch-upscale continuation · {current_info.width}x{current_info.height} · "
            f"{source_frames}f · handover={handover_frames}f · {previous_family}->{task_family}"
        ),
    )
