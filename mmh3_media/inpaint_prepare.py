from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .errors import MMH3ResourceError
from .generation_settings import adapt_h3_canvas


MASK_MODE_AUTO = "Auto Object — SAM 3.1"
MASK_MODE_MANUAL = "Manual Mask"
MASK_MODE_EXISTING = "Existing MMH3 Mask"
MASK_MODES = (MASK_MODE_AUTO, MASK_MODE_MANUAL, MASK_MODE_EXISTING)
TARGET_FPS = 24.0
AUTO_MASK_DILATION_PX = 8


@dataclass(frozen=True)
class PreparedInpaintInputs:
    source_video: torch.Tensor
    mask: torch.Tensor
    width: int
    height: int
    frames: int
    source_fps: float
    source_frames: int
    source_width: int
    source_height: int
    resampled_frames: int
    duration_seconds: float
    padded_frames: int
    trimmed_frames: int
    mask_mode: str

    def summary(self) -> str:
        timeline_note = ""
        if self.padded_frames:
            timeline_note = f" · padded={self.padded_frames}"
        elif self.trimmed_frames:
            timeline_note = f" · trimmed={self.trimmed_frames}"
        return (
            f"READY · inpaint · {self.width}x{self.height} · {self.frames}f @ 24fps "
            f"· mask={self.mask_mode}{timeline_note}"
        )


def legal_h3_frame_count(frame_count_24fps: int) -> int:
    """Return the nearest legal MiniMax H3 pixel-frame count (17n+5)."""
    frames = int(frame_count_24fps)
    if frames <= 0:
        raise MMH3ResourceError("Inpaint source video must contain at least one frame")
    if frames <= 5:
        return 5
    n = max(0, int(round((frames - 5) / 17.0)))
    return 5 + 17 * n


def _validate_source_frames(source_frames: torch.Tensor, source_fps: float) -> tuple[int, int, int, float]:
    if not isinstance(source_frames, torch.Tensor) or source_frames.ndim != 4:
        raise MMH3ResourceError("Inpaint source frames must be IMAGE [T,H,W,C]")
    if source_frames.shape[0] < 1 or source_frames.shape[1] < 1 or source_frames.shape[2] < 1:
        raise MMH3ResourceError("Inpaint source video must have a non-empty timeline and geometry")
    if source_frames.shape[-1] not in (3, 4):
        raise MMH3ResourceError("Inpaint source video must have 3 or 4 image channels")
    try:
        fps = float(source_fps)
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError("Inpaint source fps must be numeric") from exc
    if fps <= 0 or not math.isfinite(fps):
        raise MMH3ResourceError("Inpaint source fps must be a positive finite value")
    return int(source_frames.shape[0]), int(source_frames.shape[1]), int(source_frames.shape[2]), fps


def _validate_mask(mask: torch.Tensor, source_height: int, source_width: int) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise MMH3ResourceError("Inpaint mask must be a MASK tensor")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] < 1:
        raise MMH3ResourceError("Inpaint mask must be [T,H,W] or [H,W]")
    if int(mask.shape[1]) != source_height or int(mask.shape[2]) != source_width:
        raise MMH3ResourceError(
            f"Inpaint mask geometry {int(mask.shape[2])}x{int(mask.shape[1])} does not match "
            f"source video {source_width}x{source_height}"
        )
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0)


def _temporal_resample_indexes(frame_count: int, source_fps: float, target_fps: float = TARGET_FPS) -> torch.Tensor:
    target_count = max(1, int(round(frame_count * target_fps / source_fps)))
    positions = torch.arange(target_count, dtype=torch.float64)
    indexes = torch.floor(positions * source_fps / target_fps).to(torch.long)
    return indexes.clamp_(0, frame_count - 1)


def _match_mask_timeline(
    mask: torch.Tensor,
    *,
    original_count: int,
    resampled_count: int,
    target_count: int,
    indexes: torch.Tensor,
) -> torch.Tensor:
    count = int(mask.shape[0])
    if count == 1:
        return mask[:1].repeat(target_count, 1, 1)
    if count == original_count:
        mask = mask.index_select(0, indexes.to(mask.device))
        count = int(mask.shape[0])
    if count == resampled_count:
        if target_count < count:
            return mask[:target_count]
        if target_count > count:
            return torch.cat((mask, mask[-1:].repeat(target_count - count, 1, 1)), dim=0)
        return mask
    if count == target_count:
        return mask
    raise MMH3ResourceError(
        "Inpaint mask timeline must be one static frame, match the source-video frame count, "
        "match the 24fps-resampled timeline, or already match the legal H3 target timeline"
    )


def _fit_spatial(
    value: torch.Tensor,
    *,
    target_width: int,
    target_height: int,
    is_mask: bool,
) -> torch.Tensor:
    """Aspect-preserving cover resize followed by center crop to the exact H3 canvas."""
    if is_mask:
        tensor = value.unsqueeze(1)
        source_height, source_width = int(value.shape[1]), int(value.shape[2])
    else:
        tensor = value[..., :3].movedim(-1, 1)
        source_height, source_width = int(value.shape[1]), int(value.shape[2])

    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, int(math.ceil(source_width * scale)))
    resized_height = max(target_height, int(math.ceil(source_height * scale)))
    mode = "nearest" if is_mask else "bilinear"
    kwargs = {} if is_mask else {"align_corners": False}
    # Resize in bounded temporal chunks. Video tensors can contain hundreds of
    # frames; a single interpolation call would create a large transient buffer.
    resized_parts = []
    for start in range(0, int(tensor.shape[0]), 16):
        part = F.interpolate(
            tensor[start : start + 16],
            size=(resized_height, resized_width),
            mode=mode,
            **kwargs,
        )
        top = max(0, (resized_height - target_height) // 2)
        left = max(0, (resized_width - target_width) // 2)
        resized_parts.append(part[..., top : top + target_height, left : left + target_width])
    tensor = torch.cat(resized_parts, dim=0)
    if is_mask:
        return tensor[:, 0].clamp(0.0, 1.0)
    return tensor.movedim(1, -1).clamp(0.0, 1.0)


def _auto_mask_cleanup(mask: torch.Tensor) -> torch.Tensor:
    """Conservative fixed inpaint padding for SAM masks; not exposed as a user knob."""
    radius = AUTO_MASK_DILATION_PX
    if radius <= 0:
        return mask
    x = mask.unsqueeze(1)
    x = F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return x[:, 0].clamp(0.0, 1.0)


def prepare_inpaint_inputs(
    source_frames: torch.Tensor,
    source_fps: float,
    mask: torch.Tensor,
    *,
    mask_mode: str,
) -> PreparedInpaintInputs:
    """Conform source video and selected temporal mask to the strict H3-Fun inpaint contract.

    Both streams receive the same temporal and spatial transform. The source is first
    resampled to H3's 24fps path, then the nearest legal 17n+5 timeline is selected.
    A SAM auto-mask receives a small fixed dilation after geometry conformance so the
    regenerate region covers segmentation-boundary remnants. Manual/existing masks are
    otherwise preserved apart from required geometry/timeline conformance.
    """
    if mask_mode not in MASK_MODES:
        raise MMH3ResourceError(f"Unsupported inpaint mask mode {mask_mode!r}; expected one of {MASK_MODES}")
    original_count, source_height, source_width, fps = _validate_source_frames(source_frames, source_fps)
    mask = _validate_mask(mask, source_height, source_width)

    indexes = _temporal_resample_indexes(original_count, fps)
    indexes_device = indexes.to(source_frames.device)
    video_24 = source_frames.index_select(0, indexes_device)
    resampled_count = int(video_24.shape[0])
    target_count = legal_h3_frame_count(resampled_count)

    if target_count < resampled_count:
        video_target = video_24[:target_count]
        trimmed = resampled_count - target_count
        padded = 0
    elif target_count > resampled_count:
        pad = video_24[-1:].repeat(target_count - resampled_count, 1, 1, 1)
        video_target = torch.cat((video_24, pad), dim=0)
        trimmed = 0
        padded = target_count - resampled_count
    else:
        video_target = video_24
        trimmed = padded = 0

    mask_target = _match_mask_timeline(
        mask,
        original_count=original_count,
        resampled_count=resampled_count,
        target_count=target_count,
        indexes=indexes,
    )

    target_width, target_height = adapt_h3_canvas(source_width, source_height)
    video_target = _fit_spatial(
        video_target, target_width=target_width, target_height=target_height, is_mask=False
    )
    mask_target = _fit_spatial(
        mask_target, target_width=target_width, target_height=target_height, is_mask=True
    )
    if mask_mode == MASK_MODE_AUTO:
        mask_target = _auto_mask_cleanup(mask_target)

    return PreparedInpaintInputs(
        source_video=video_target,
        mask=mask_target,
        width=target_width,
        height=target_height,
        frames=target_count,
        source_fps=fps,
        source_frames=original_count,
        source_width=source_width,
        source_height=source_height,
        resampled_frames=resampled_count,
        duration_seconds=target_count / TARGET_FPS,
        padded_frames=padded,
        trimmed_frames=trimmed,
        mask_mode=mask_mode,
    )


def build_inpaint_generation_settings(prepared: PreparedInpaintInputs) -> dict:
    return {
        "contract": "mmh3_h3_generation_settings_v2",
        "geometry_mode": "auto_from_source",
        "timeline_mode": "auto_from_source",
        "audio_policy": "generate",
        "effective_overrides": {
            "width": prepared.width,
            "height": prepared.height,
            "frames": prepared.frames,
        },
        "source_geometry": {
            "width": prepared.source_width,
            "height": prepared.source_height,
            "frames": prepared.source_frames,
            "fps": prepared.source_fps,
        },
        "inpaint": {
            "mask_mode": prepared.mask_mode,
            "mask_policy": "white_regenerate",
            "auto_mask_padding_px": AUTO_MASK_DILATION_PX if prepared.mask_mode == MASK_MODE_AUTO else 0,
        },
    }


def build_inpaint_packet(source_video_payload, prepared: PreparedInpaintInputs, *, prompt: str, seed: int):
    """Build the canonical MMH3/F16 state around already-conformed source media."""
    from .control_contract import build_control_configuration, set_control_configuration
    from .core import MMH3Media
    from .h3_resource_semantics import make_control_resource_contract
    from .resource_model import descriptor_from_media_metadata

    prompt = str(prompt or "").strip()
    if not prompt:
        raise MMH3ResourceError("Masked inpaint requires a generation prompt")
    generation = {
        "task": "t2va",
        "prompt": prompt,
        "seed": int(seed),
        "width": prepared.width,
        "height": prepared.height,
        "frames": prepared.frames,
        "fps": TARGET_FPS,
    }
    packet = MMH3Media.create(name="F16 F10 masked video inpaint", generation=generation)
    video_descriptor = descriptor_from_media_metadata(
        "video",
        {
            "dimensions": [prepared.width, prepared.height],
            "frame_count": prepared.frames,
            "fps": TARGET_FPS,
            "duration": prepared.duration_seconds,
            "shape": list(prepared.source_video.shape),
        },
    )
    packet, source_id = packet.put_with_id(
        source_video_payload,
        kind="video",
        role="context",
        name="Inpaint source video",
        tags=["inpaint-source"],
        descriptor=video_descriptor,
        extensions={"minimax_h3": {"control": make_control_resource_contract(usage="inpaint_source")}},
        record_history=False,
    )
    mask_descriptor = {
        "shape": {"width": prepared.width, "height": prepared.height, "frames": prepared.frames},
        "tensor": {"shape": list(prepared.mask.shape), "dtype": str(prepared.mask.dtype)},
        "timing": {"fps": TARGET_FPS, "duration": prepared.duration_seconds},
    }
    packet, mask_id = packet.put_with_id(
        prepared.mask,
        kind="mask",
        role="control",
        name="Inpaint regenerate mask",
        tags=["inpaint-mask", "white-regenerate"],
        descriptor=mask_descriptor,
        extensions={"minimax_h3": {"control": make_control_resource_contract(usage="mask")}},
        record_history=False,
    )
    packet = set_control_configuration(
        packet,
        build_control_configuration(
            requested_algorithm="fun_controlnet_union_int8_convrot",
            control_kind="inpaint",
            strength=1.0,
            start_percent=0.0,
            end_percent=1.0,
            temporal_policy="strict",
            inpaint_source_resource_id=source_id,
            mask_resource_id=mask_id,
        ),
    )
    return packet, source_id, mask_id, build_inpaint_generation_settings(prepared)


__all__ = [
    "AUTO_MASK_DILATION_PX",
    "MASK_MODE_AUTO",
    "MASK_MODE_EXISTING",
    "MASK_MODE_MANUAL",
    "MASK_MODES",
    "PreparedInpaintInputs",
    "TARGET_FPS",
    "build_inpaint_generation_settings",
    "build_inpaint_packet",
    "legal_h3_frame_count",
    "prepare_inpaint_inputs",
]
