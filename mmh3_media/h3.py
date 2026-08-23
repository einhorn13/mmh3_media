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
    H3_LATENT_CONTEXT_VERSION,
    H3_LATENT_ORIGINS,
    H3_TEMPORAL_GRID,
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

def h3_info_metadata(info: H3LatentInfo) -> dict:
    return {
        "video_shape": list(info.video_shape),
        "audio_shape": list(info.audio_shape),
        "width": info.width,
        "height": info.height,
        "frames": info.frames,
        "fps": info.fps,
        "audio_sample_rate": info.audio_sample_rate,
        "audio_latent_rate": info.audio_latent_rate,
        "batch": info.batch,
        **({"warnings": list(info.warnings)} if info.warnings else {}),
    }


def validate_h3_latent_origin(origin: str) -> str:
    if origin not in H3_LATENT_ORIGINS:
        raise MMH3ResourceError(
            f"H3 latent origin must be one of {H3_LATENT_ORIGINS}, got {origin!r}"
        )
    return origin


def h3_latent_context(info: H3LatentInfo, *, origin: str = "unknown") -> dict:
    """Build a self-contained, non-algorithmic descriptor for downstream H3 adapters.

    The descriptor deliberately distinguishes tensor-layout validity from provenance.
    A valid H3 AV tensor does not prove it is a direct sampler output.
    """
    origin = validate_h3_latent_origin(origin)
    on_stock_grid = info.frames is not None
    batch_ok = info.batch == 1

    if origin == "sampler_output" and on_stock_grid and batch_ok:
        continuation_status = "eligible"
        continuation_reason = "Direct H3 sampler output on the stock temporal grid."
    elif origin in {"vae_encoded", "derived"}:
        continuation_status = "ineligible"
        continuation_reason = (
            "Latent provenance is not a direct H3 sampler output; a continuation adapter may still "
            "support reconstructed/derived input through a different path."
        )
    else:
        continuation_status = "unknown"
        continuation_reason = "Tensor layout is valid, but direct-sampler provenance was not declared."

    if not on_stock_grid:
        continuation_status = "ineligible"
        continuation_reason = "Video latent is off the stock H3 causal temporal grid."
    elif not batch_ok:
        continuation_status = "ineligible"
        continuation_reason = "Canonical continuation handoff currently requires batch=1."

    seam_status = "eligible" if on_stock_grid and batch_ok else "ineligible"
    seam_reason = (
        "Valid H3 AV source for an H3-aware seam/stitch adapter; this does not make naive latent concatenation safe."
        if seam_status == "eligible"
        else "Source is off-grid or batched; canonical H3 seam adapters should reject or normalize it explicitly."
    )

    return {
        "version": H3_LATENT_CONTEXT_VERSION,
        "model_family": "minimax_h3",
        "layout": "joint_av",
        "origin": origin,
        "temporal_grid": H3_TEMPORAL_GRID if on_stock_grid else "unknown",
        "canvas": {"width": info.width, "height": info.height},
        "frames": info.frames,
        "fps": info.fps,
        "video_latent_t": info.video_shape[2],
        "audio_latent_t": info.audio_shape[-1],
        "audio_sample_rate": info.audio_sample_rate,
        "audio_latent_rate": info.audio_latent_rate,
        "batch": info.batch,
        "continuation": {"status": continuation_status, "reason": continuation_reason},
        "seam_source": {"status": seam_status, "reason": seam_reason},
        "naive_latent_concat_safe": False,
    }


def h3_metadata_with_context(
    info: H3LatentInfo,
    *,
    existing_h3: dict | None = None,
    origin: str | None = None,
) -> dict:
    """Merge fresh shape facts with a durable context descriptor.

    Existing unknown extension fields under metadata.h3 are preserved. If origin is omitted,
    an existing context origin is retained; otherwise it falls back to ``unknown``.
    """
    existing = dict(existing_h3 or {})
    existing_context = existing.get("latent_context")
    existing_origin = None
    if isinstance(existing_context, dict):
        candidate = existing_context.get("origin")
        if candidate in H3_LATENT_ORIGINS:
            existing_origin = candidate
    resolved_origin = origin if origin is not None else (existing_origin or "unknown")
    fresh = h3_info_metadata(info)
    existing.update(fresh)
    existing["latent_context"] = h3_latent_context(info, origin=resolved_origin)
    return existing



def h3_context_from_metadata(h3: dict | None) -> dict | None:
    """Return the schema-v1 latent_context when present."""
    if not isinstance(h3, dict):
        return None
    existing = h3.get("latent_context")
    return dict(existing) if isinstance(existing, dict) else None


def h3_metadata_set_origin(h3: dict, origin: str) -> dict:
    """Rebuild standardized H3 metadata with an explicit provenance declaration."""
    if not isinstance(h3, dict):
        raise MMH3ResourceError("H3 metadata is missing or invalid")
    try:
        video_shape = tuple(int(x) for x in h3["video_shape"])
        audio_shape = tuple(int(x) for x in h3["audio_shape"])
        info = H3LatentInfo(
            video_shape=video_shape,
            audio_shape=audio_shape,
            width=int(h3["width"]),
            height=int(h3["height"]),
            frames=int(h3["frames"]) if h3.get("frames") is not None else None,
            fps=int(h3.get("fps", FPS)),
            audio_sample_rate=int(h3.get("audio_sample_rate", AUDIO_SAMPLE_RATE)),
            audio_latent_rate=int(h3.get("audio_latent_rate", AUDIO_LATENT_FPS)),
            batch=int(h3.get("batch", video_shape[0])),
            warnings=tuple(str(x) for x in h3.get("warnings", []) if isinstance(x, str)),
        )
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise MMH3ResourceError("H3 metadata lacks enough shape/timing information to set provenance") from e
    return h3_metadata_with_context(info, existing_h3=h3, origin=origin)


def h3_pair_compatibility(a: dict, b: dict) -> dict:
    """Compare two saved H3 context descriptors without loading latent tensors.

    This is compatibility for a future H3-aware seam adapter, never approval for ``torch.cat``.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {
            "compatible": False,
            "naive_latent_concat_safe": False,
            "reasons": ["Both packets need H3 latent_context metadata."],
            "warnings": [],
        }

    required_equal = (
        ("model_family", "model family"),
        ("layout", "latent layout"),
        ("fps", "video fps"),
        ("audio_sample_rate", "audio sample rate"),
        ("audio_latent_rate", "audio latent rate"),
        ("batch", "batch size"),
    )
    for key, label in required_equal:
        if a.get(key) != b.get(key):
            reasons.append(f"Different {label}: {a.get(key)!r} vs {b.get(key)!r}")

    ac, bc = a.get("canvas"), b.get("canvas")
    if not isinstance(ac, dict) or not isinstance(bc, dict) or ac.get("width") != bc.get("width") or ac.get("height") != bc.get("height"):
        reasons.append(f"Different canvas: {ac!r} vs {bc!r}")
    if a.get("temporal_grid") != H3_TEMPORAL_GRID or b.get("temporal_grid") != H3_TEMPORAL_GRID:
        reasons.append("Both sources must be on the stock H3 causal temporal grid.")

    for label, ctx in (("A", a), ("B", b)):
        origin = ctx.get("origin", "unknown")
        if origin != "sampler_output":
            warnings.append(
                f"{label} origin is {origin!r}; seam compatibility is structural, not proof of direct-sampler provenance."
            )

    return {
        "compatible": not reasons,
        "naive_latent_concat_safe": False,
        "reasons": reasons,
        "warnings": warnings,
    }
