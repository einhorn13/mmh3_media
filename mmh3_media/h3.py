from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .constants import (
    AUDIO_CHANNELS,
    AUDIO_LATENT_FPS,
    AUDIO_SAMPLE_RATE,
    AUDIO_STEREO_CHANNELS,
    DIT_SPATIAL_PATCH,
    FPS,
    H3_LATENT_ORIGINS,
    LATENT_SPATIAL_DIVISOR,
    VIDEO_CHANNELS,
)
from .errors import MMH3ResourceError


@dataclass(frozen=True)
class H3LatentInfo:
    video_shape: tuple[int, ...]
    audio_shape: tuple[int, ...]
    width: int
    height: int
    frames: int | None
    fps: int
    audio_sample_rate: int
    audio_latent_rate: int
    batch: int
    warnings: tuple[str, ...] = ()


def is_nested_tensor(value: Any) -> bool:
    return bool(getattr(value, "is_nested", False)) and (
        hasattr(value, "unbind") or hasattr(value, "tensors")
    )


def nested_parts(value: Any) -> list[torch.Tensor]:
    if not is_nested_tensor(value):
        raise MMH3ResourceError("Expected a ComfyUI NestedTensor")
    if hasattr(value, "unbind"):
        parts = value.unbind()
    else:
        parts = value.tensors
    return list(parts)


def make_nested_tensor(parts: list[torch.Tensor]):
    try:
        import comfy.nested_tensor  # type: ignore

        return comfy.nested_tensor.NestedTensor(tuple(parts))
    except Exception:
        # Keeps core/tests usable outside ComfyUI; ComfyUI runtime will take the branch above.
        class _PortableNestedTensor:
            is_nested = True

            def __init__(self, tensors):
                self.tensors = list(tensors)

            def unbind(self):
                return self.tensors

            def to(self, *args, **kwargs):
                return _PortableNestedTensor([t.to(*args, **kwargs) for t in self.tensors])

            def cpu(self):
                return self.to(device="cpu")

        return _PortableNestedTensor(parts)


def h3_frame_count_from_video_t(video_t: int) -> int | None:
    # Stock ComfyUI: T=2 for 5 frames, then +5 latent positions per +17 pixel frames.
    if video_t < 2 or (video_t - 2) % 5 != 0:
        return None
    return 5 + ((video_t - 2) // 5) * 17


def h3_expected_audio_t(frames: int) -> int:
    return round((frames / FPS) * AUDIO_LATENT_FPS)


def validate_h3_av_latent(latent: dict, *, strict_audio_length: bool = True) -> H3LatentInfo:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise MMH3ResourceError("H3 AV latent must be a LATENT dict containing 'samples'")
    samples = latent["samples"]
    if not is_nested_tensor(samples):
        raise MMH3ResourceError(
            "H3 AV latent must contain joint video+audio NestedTensor samples; a plain latent tensor is not enough"
        )
    parts = nested_parts(samples)
    if len(parts) != 2:
        raise MMH3ResourceError(f"H3 AV latent must have exactly 2 streams (video, audio); got {len(parts)}")
    video, audio = parts
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise MMH3ResourceError("H3 AV latent streams must be torch tensors")
    if video.ndim != 5:
        raise MMH3ResourceError(f"H3 video latent must be [B,24,T,H,W], got {tuple(video.shape)}")
    if video.shape[1] != VIDEO_CHANNELS:
        raise MMH3ResourceError(f"H3 video latent must have 24 channels, got {video.shape[1]}")
    if audio.ndim != 4:
        raise MMH3ResourceError(f"H3 audio latent must be [B,32,2,T40], got {tuple(audio.shape)}")
    if audio.shape[1] != AUDIO_CHANNELS or audio.shape[2] != AUDIO_STEREO_CHANNELS:
        raise MMH3ResourceError(
            f"H3 audio latent must have shape [B,32,2,T40], got {tuple(audio.shape)}"
        )
    if video.shape[0] != audio.shape[0]:
        raise MMH3ResourceError(
            f"H3 video/audio batch mismatch: video B={video.shape[0]}, audio B={audio.shape[0]}"
        )
    if not video.is_floating_point() or not audio.is_floating_point():
        raise MMH3ResourceError("H3 video/audio latent streams must use floating dtypes")
    if video.shape[-1] % DIT_SPATIAL_PATCH or video.shape[-2] % DIT_SPATIAL_PATCH:
        raise MMH3ResourceError(
            "H3 video latent spatial grid must be divisible by the DiT 2x2 patch; "
            f"got HxW={video.shape[-2]}x{video.shape[-1]}"
        )

    frames = h3_frame_count_from_video_t(int(video.shape[2]))
    warnings: list[str] = []
    if frames is None:
        warnings.append(
            f"video latent T={video.shape[2]} is off the stock H3 causal 17k+5 temporal grid; pixel frame count is unknown"
        )
    else:
        expected_audio = h3_expected_audio_t(frames)
        if int(audio.shape[-1]) != expected_audio:
            message = (
                f"H3 AV duration mismatch: video T={video.shape[2]} implies {frames} frames / "
                f"audio T40={expected_audio}, but audio latent has T40={audio.shape[-1]}"
            )
            if strict_audio_length:
                raise MMH3ResourceError(message)
            warnings.append(message)

    return H3LatentInfo(
        video_shape=tuple(int(x) for x in video.shape),
        audio_shape=tuple(int(x) for x in audio.shape),
        width=int(video.shape[-1]) * LATENT_SPATIAL_DIVISOR,
        height=int(video.shape[-2]) * LATENT_SPATIAL_DIVISOR,
        frames=frames,
        fps=FPS,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_latent_rate=AUDIO_LATENT_FPS,
        batch=int(video.shape[0]),
        warnings=tuple(warnings),
    )



def split_h3_av_latent(av_latent: dict) -> tuple[dict, dict]:
    """Split a validated MiniMax H3 joint AV LATENT without copying tensor storage.

    Non-stream LATENT fields are shallow-copied to both outputs. Nested noise masks,
    when present, are split in the same order as samples.
    """
    validate_h3_av_latent(av_latent, strict_audio_length=False)
    streams = nested_parts(av_latent["samples"])

    video_latent = av_latent.copy()
    audio_latent = av_latent.copy()
    video_latent["samples"] = streams[0]
    audio_latent["samples"] = streams[1]

    if "noise_mask" in av_latent and av_latent["noise_mask"] is not None:
        mask = av_latent["noise_mask"]
        if not is_nested_tensor(mask):
            raise MMH3ResourceError("H3 AV latent noise_mask must be a 2-stream NestedTensor when present")
        masks = nested_parts(mask)
        if len(masks) != 2:
            raise MMH3ResourceError(f"H3 AV latent noise_mask must have exactly 2 streams; got {len(masks)}")
        video_latent["noise_mask"] = masks[0]
        audio_latent["noise_mask"] = masks[1]

    return video_latent, audio_latent


def concat_h3_av_latent(video_latent: dict, audio_latent: dict) -> dict:
    """Combine plain H3 video/audio LATENTs into a strict joint AV latent.

    Unlike the generic LTXV helper this function does not trim/pad audio or silently
    replace a stream inside an already-joint latent. Duration/batch/layout mismatches
    are errors. Video-side auxiliary LATENT fields win on key conflicts because the
    video branch is commonly transformed (e.g. spatial upscaling) while audio is a
    bypass from an earlier split.
    """
    if not isinstance(video_latent, dict) or "samples" not in video_latent:
        raise MMH3ResourceError("H3 video latent must be a LATENT dict containing 'samples'")
    if not isinstance(audio_latent, dict) or "samples" not in audio_latent:
        raise MMH3ResourceError("H3 audio latent must be a LATENT dict containing 'samples'")

    video = video_latent["samples"]
    audio = audio_latent["samples"]
    if is_nested_tensor(video) or is_nested_tensor(audio):
        raise MMH3ResourceError("MMH3 H3 AV Combine expects separated video/audio latents, not an already-joint AV latent")
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != VIDEO_CHANNELS:
        shape = tuple(video.shape) if isinstance(video, torch.Tensor) else type(video).__name__
        raise MMH3ResourceError(f"H3 video latent must be [B,24,T,H,W], got {shape}")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 4 or audio.shape[1] != AUDIO_CHANNELS or audio.shape[2] != AUDIO_STEREO_CHANNELS:
        shape = tuple(audio.shape) if isinstance(audio, torch.Tensor) else type(audio).__name__
        raise MMH3ResourceError(f"H3 audio latent must be [B,32,2,T40], got {shape}")
    if not video.is_floating_point() or not audio.is_floating_point():
        raise MMH3ResourceError("H3 video/audio latent streams must use floating dtypes")
    if video.shape[0] != audio.shape[0]:
        raise MMH3ResourceError(f"H3 video/audio batch mismatch: video B={video.shape[0]}, audio B={audio.shape[0]}")
    if video.shape[-1] % DIT_SPATIAL_PATCH or video.shape[-2] % DIT_SPATIAL_PATCH:
        raise MMH3ResourceError(
            "H3 video latent spatial grid must be divisible by the DiT 2x2 patch; "
            f"got HxW={video.shape[-2]}x{video.shape[-1]}"
        )

    # Reuse the canonical validator for the temporal-grid and duration contract.
    output = dict(audio_latent)
    output.update(video_latent)
    output["samples"] = make_nested_tensor([video, audio])

    video_mask = video_latent.get("noise_mask")
    audio_mask = audio_latent.get("noise_mask")
    if video_mask is not None or audio_mask is not None:
        if video_mask is not None and is_nested_tensor(video_mask):
            raise MMH3ResourceError("Separated video noise_mask must not be a NestedTensor")
        if audio_mask is not None and is_nested_tensor(audio_mask):
            raise MMH3ResourceError("Separated audio noise_mask must not be a NestedTensor")
        if video_mask is None:
            video_mask = torch.ones_like(video)
        if audio_mask is None:
            audio_mask = torch.ones_like(audio)
        output["noise_mask"] = make_nested_tensor([video_mask, audio_mask])
    else:
        output.pop("noise_mask", None)

    validate_h3_av_latent(output, strict_audio_length=True)
    return output


def validate_h3_latent_origin(origin: str) -> str:
    if origin not in H3_LATENT_ORIGINS:
        raise MMH3ResourceError(
            f"H3 latent origin must be one of {H3_LATENT_ORIGINS}, got {origin!r}"
        )
    return origin
