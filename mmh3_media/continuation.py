from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .constants import AUDIO_LATENT_FPS, FPS
from .errors import MMH3ResourceError
from .h3 import h3_expected_audio_t, make_nested_tensor, nested_parts, validate_h3_av_latent


@dataclass(frozen=True)
class H3ContinuationPlan:
    source_frames: int
    target_frames: int
    video_handover_frames: int
    audio_handover_frames: int
    audio_feather_frames: int
    source_video_t: int
    source_audio_t: int
    target_video_t: int
    target_audio_t: int
    copied_video_t: int
    copied_audio_t: int
    feather_audio_t: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": True,
            "contract": "minimax_h3_joint_av_continuation_v1",
            "source_frames": self.source_frames,
            "target_frames": self.target_frames,
            "video_handover_frames": self.video_handover_frames,
            "audio_handover_frames": self.audio_handover_frames,
            "audio_feather_frames": self.audio_feather_frames,
            "source_video_t": self.source_video_t,
            "source_audio_t": self.source_audio_t,
            "target_video_t": self.target_video_t,
            "target_audio_t": self.target_audio_t,
            "copied_video_t": self.copied_video_t,
            "copied_audio_t": self.copied_audio_t,
            "feather_audio_t": self.feather_audio_t,
            "noise_mask_semantics": "0=preserve, 1=denoise",
            "naive_latent_concat_safe": False,
            "timing": {
                "generated_seconds": self.target_frames / FPS,
                "context_seconds": self.video_handover_frames / FPS,
                "new_seconds": (self.target_frames - self.video_handover_frames) / FPS,
                "new_frames": self.target_frames - self.video_handover_frames,
                "prompt_to_output_offset_seconds": -self.video_handover_frames / FPS,
            },
        }

    def summary(self) -> str:
        feather = f", audio feather {self.audio_feather_frames}f" if self.audio_feather_frames else ""
        return (
            f"READY · F02 continuation · target={self.target_frames}f · "
            f"video prefix={self.video_handover_frames}f · audio prefix={self.audio_handover_frames}f{feather} · "
            f"{self.target_frames / FPS:.2f}s generated / {self.video_handover_frames / FPS:.2f}s context / "
            f"{(self.target_frames - self.video_handover_frames) / FPS:.2f}s new"
        )


@dataclass(frozen=True)
class H3ContinuationResult:
    latent: dict
    plan: H3ContinuationPlan


def build_h3_target_from_prefix(
    prefix_latent: dict,
    *,
    target_frames: int = 124,
    audio_feather_frames: int = 0,
) -> H3ContinuationResult:
    """Place an already aligned H3 joint AV prefix at the start of a fresh target."""
    prefix_info = validate_h3_av_latent(prefix_latent, strict_audio_length=True)
    if prefix_info.frames is None:
        raise MMH3ResourceError("H3 continuation prefix must be on the stock temporal grid")
    if prefix_info.batch != 1:
        raise MMH3ResourceError(f"H3 continuation prefix currently requires batch=1; got {prefix_info.batch}")

    target_frames = int(target_frames)
    target_video_t = h3_video_t_from_frames(target_frames)
    target_audio_t = h3_expected_audio_t(target_frames)
    if prefix_info.frames > target_frames:
        raise MMH3ResourceError(
            f"Continuation prefix duration {prefix_info.frames} exceeds target duration {target_frames}"
        )

    audio_feather_frames = int(audio_feather_frames)
    if audio_feather_frames < 0:
        raise MMH3ResourceError("audio_feather_frames cannot be negative")
    feather_audio_t = 0
    if audio_feather_frames:
        feather_audio_t = h3_audio_t_from_exact_frames(
            audio_feather_frames, label="audio_feather_frames"
        )
        if feather_audio_t > prefix_info.audio_shape[-1]:
            raise MMH3ResourceError("audio_feather_frames cannot exceed the prefix duration")

    prefix_video, prefix_audio = nested_parts(prefix_latent["samples"])
    copied_video_t = int(prefix_video.shape[2])
    copied_audio_t = int(prefix_audio.shape[-1])
    target_video = prefix_video.new_zeros(
        (1, prefix_video.shape[1], target_video_t, prefix_video.shape[-2], prefix_video.shape[-1])
    )
    target_audio = prefix_audio.new_zeros(
        (1, prefix_audio.shape[1], prefix_audio.shape[2], target_audio_t)
    )
    target_video[:, :, :copied_video_t] = prefix_video
    target_audio[..., :copied_audio_t] = prefix_audio

    video_mask = torch.ones_like(target_video)
    audio_mask = torch.ones_like(target_audio)
    video_mask[:, :, :copied_video_t] = 0.0
    audio_mask[..., :copied_audio_t] = 0.0
    if feather_audio_t:
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(
                0.0,
                math.pi,
                feather_audio_t + 2,
                dtype=audio_mask.dtype,
                device=audio_mask.device,
            )[1:-1]
        )
        start = copied_audio_t - feather_audio_t
        audio_mask[..., start:copied_audio_t] = ramp

    target = {
        "samples": make_nested_tensor([target_video, target_audio]),
        "noise_mask": make_nested_tensor([video_mask, audio_mask]),
    }
    target_info = validate_h3_av_latent(target, strict_audio_length=True)
    plan = H3ContinuationPlan(
        source_frames=prefix_info.frames,
        target_frames=target_frames,
        video_handover_frames=prefix_info.frames,
        audio_handover_frames=prefix_info.frames,
        audio_feather_frames=audio_feather_frames,
        source_video_t=prefix_info.video_shape[2],
        source_audio_t=prefix_info.audio_shape[-1],
        target_video_t=target_info.video_shape[2],
        target_audio_t=target_info.audio_shape[-1],
        copied_video_t=copied_video_t,
        copied_audio_t=copied_audio_t,
        feather_audio_t=feather_audio_t,
    )
    return H3ContinuationResult(target, plan)


def h3_video_t_from_frames(frames: int) -> int:
    frames = int(frames)
    if frames < 5 or (frames - 5) % 17:
        raise MMH3ResourceError(f"H3 frame count must be on the 17k+5 grid; got {frames}")
    return 2 + 5 * ((frames - 5) // 17)


def h3_audio_t_from_exact_frames(frames: int, *, label: str = "frame count") -> int:
    frames = int(frames)
    numerator = frames * AUDIO_LATENT_FPS
    if frames < 1 or numerator % FPS:
        raise MMH3ResourceError(
            f"{label} must map to an exact 40 Hz audio-latent boundary at 24 FPS; got {frames} frames"
        )
    return numerator // FPS


def is_exact_h3_av_handover_boundary(frames: int) -> bool:
    frames = int(frames)
    return frames >= 5 and (frames - 5) % 17 == 0 and (frames * AUDIO_LATENT_FPS) % FPS == 0


def exact_h3_av_handover_boundaries(max_frames: int) -> tuple[int, ...]:
    return tuple(
        frames
        for frames in range(5, int(max_frames) + 1, 17)
        if is_exact_h3_av_handover_boundary(frames)
    )


def build_h3_continuation_handover(
    source_latent: dict,
    *,
    source_origin: str,
    target_frames: int = 124,
    video_handover_frames: int = 39,
    audio_handover_frames: int | None = None,
    audio_feather_frames: int = 0,
    target_width: int = 0,
    target_height: int = 0,
) -> H3ContinuationResult:
    """Build a fresh joint AV target with a copied source tail and per-stream denoise masks.

    This is a temporal handover, never concatenation: copied source values occupy the
    target prefix and the rest of the target is initialized to zero for sampling.
    """
    if source_origin != "sampler_output":
        raise MMH3ResourceError(
            f"F02 continuation requires source_origin='sampler_output'; got {source_origin!r}"
        )
    source_info = validate_h3_av_latent(source_latent, strict_audio_length=True)
    if ((target_width and int(target_width) != source_info.width)
            or (target_height and int(target_height) != source_info.height)):
        raise MMH3ResourceError(
            f"Direct latent continuation requires source resolution {source_info.width}x{source_info.height}; "
            "choose Source in Video Settings. Use decoded continuation or latent upscale to change resolution."
        )
    if source_info.frames is None:
        raise MMH3ResourceError("F02 continuation requires a source on the stock H3 temporal grid")
    if source_info.batch != 1:
        raise MMH3ResourceError(f"F02 continuation currently requires batch=1; got {source_info.batch}")

    target_frames = int(target_frames)
    target_video_t = h3_video_t_from_frames(target_frames)
    target_audio_t = h3_expected_audio_t(target_frames)
    video_handover_frames = int(video_handover_frames)
    if not is_exact_h3_av_handover_boundary(video_handover_frames):
        raise MMH3ResourceError(
            "video_handover_frames must be an exact H3 AV boundary (39, 90, 141, 192, ...); "
            f"got {video_handover_frames}"
        )
    audio_handover_frames = (
        video_handover_frames if audio_handover_frames is None else int(audio_handover_frames)
    )
    audio_handover_t = h3_audio_t_from_exact_frames(
        audio_handover_frames, label="audio_handover_frames"
    )
    audio_feather_frames = int(audio_feather_frames)
    if audio_feather_frames < 0:
        raise MMH3ResourceError("audio_feather_frames cannot be negative")
    feather_audio_t = 0
    if audio_feather_frames:
        feather_audio_t = h3_audio_t_from_exact_frames(
            audio_feather_frames, label="audio_feather_frames"
        )
        if feather_audio_t > audio_handover_t:
            raise MMH3ResourceError("audio_feather_frames cannot exceed audio_handover_frames")

    for label, requested in (
        ("video_handover_frames", video_handover_frames),
        ("audio_handover_frames", audio_handover_frames),
    ):
        if requested > source_info.frames:
            raise MMH3ResourceError(f"{label}={requested} exceeds source duration {source_info.frames}")
        if requested > target_frames:
            raise MMH3ResourceError(f"{label}={requested} exceeds target duration {target_frames}")

    video_handover_t = h3_video_t_from_frames(video_handover_frames)
    source_video, source_audio = nested_parts(source_latent["samples"])
    if audio_handover_t > int(source_audio.shape[-1]):
        raise MMH3ResourceError("Requested audio handover exceeds the source audio latent")

    target_video = source_video.new_zeros(
        (1, source_video.shape[1], target_video_t, source_video.shape[-2], source_video.shape[-1])
    )
    target_audio = source_audio.new_zeros(
        (1, source_audio.shape[1], source_audio.shape[2], target_audio_t)
    )
    target_video[:, :, :video_handover_t] = source_video[:, :, -video_handover_t:]
    target_audio[..., :audio_handover_t] = source_audio[..., -audio_handover_t:]

    video_mask = torch.ones_like(target_video)
    audio_mask = torch.ones_like(target_audio)
    video_mask[:, :, :video_handover_t] = 0.0
    audio_mask[..., :audio_handover_t] = 0.0
    if feather_audio_t:
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(
                0.0,
                math.pi,
                feather_audio_t + 2,
                dtype=audio_mask.dtype,
                device=audio_mask.device,
            )[1:-1]
        )
        start = audio_handover_t - feather_audio_t
        audio_mask[..., start:audio_handover_t] = ramp

    target = {
        "samples": make_nested_tensor([target_video, target_audio]),
        "noise_mask": make_nested_tensor([video_mask, audio_mask]),
    }
    target_info = validate_h3_av_latent(target, strict_audio_length=True)
    plan = H3ContinuationPlan(
        source_frames=source_info.frames,
        target_frames=target_frames,
        video_handover_frames=video_handover_frames,
        audio_handover_frames=audio_handover_frames,
        audio_feather_frames=audio_feather_frames,
        source_video_t=source_info.video_shape[2],
        source_audio_t=source_info.audio_shape[-1],
        target_video_t=target_info.video_shape[2],
        target_audio_t=target_info.audio_shape[-1],
        copied_video_t=video_handover_t,
        copied_audio_t=audio_handover_t,
        feather_audio_t=feather_audio_t,
    )
    return H3ContinuationResult(target, plan)
