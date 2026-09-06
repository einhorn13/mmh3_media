from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .constants import AUDIO_SAMPLE_RATE, FPS
from .continuation import (
    H3ContinuationResult,
    build_h3_target_from_prefix,
    is_exact_h3_av_handover_boundary,
)
from .errors import MMH3ResourceError
from .h3 import make_nested_tensor, validate_h3_av_latent
from .h3_references import _resample_frames


AudioResampler = Callable[[torch.Tensor, int, int], torch.Tensor]


@dataclass(frozen=True)
class DecodedContinuationPlan:
    source_fps: float
    source_frames: int
    normalized_frames: int
    prefix_frames: int
    width: int
    height: int
    source_audio_sample_rate: int | None
    audio_source: str
    audio_samples: int
    missing_audio_policy: str
    used_silence: bool
    round_trip_warning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": True,
            "contract": "minimax_h3_decoded_continuation_v1",
            "source_fps": self.source_fps,
            "source_frames": self.source_frames,
            "normalized_fps": FPS,
            "normalized_frames": self.normalized_frames,
            "prefix_frames": self.prefix_frames,
            "width": self.width,
            "height": self.height,
            "source_audio_sample_rate": self.source_audio_sample_rate,
            "audio_source": self.audio_source,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "audio_samples": self.audio_samples,
            "missing_audio_policy": self.missing_audio_policy,
            "used_silence": self.used_silence,
            "latent_origin": "vae_encoded",
            "round_trip_warning": self.round_trip_warning,
            "naive_latent_concat_safe": False,
        }

    def info_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def summary(self) -> str:
        audio = "silence" if self.used_silence else "source audio"
        return (
            f"READY · F03 decoded continuation · {self.prefix_frames}f prefix · "
            f"{self.width}x{self.height} · {audio} · VAE round trip"
        )


@dataclass(frozen=True)
class PreparedDecodedPrefix:
    frames: torch.Tensor
    waveform: torch.Tensor
    plan: DecodedContinuationPlan


@dataclass(frozen=True)
class DecodedContinuationResult:
    latent: dict
    plan: DecodedContinuationPlan
    continuation: H3ContinuationResult


def _default_audio_resampler(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - ComfyUI provides torchaudio
        raise MMH3ResourceError("Audio resampling requires torchaudio in the ComfyUI runtime") from exc
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def _normalize_canvas(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise MMH3ResourceError(f"F03 target canvas must be positive multiples of 32; got {width}x{height}")
    source_h, source_w = int(frames.shape[1]), int(frames.shape[2])
    if source_h == height and source_w == width:
        return frames
    scale = max(width / source_w, height / source_h)
    resized_w = max(width, int(round(source_w * scale)))
    resized_h = max(height, int(round(source_h * scale)))
    nchw = frames.movedim(-1, 1)
    resized = F.interpolate(nchw, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    top = (resized_h - height) // 2
    left = (resized_w - width) // 2
    return resized[:, :, top : top + height, left : left + width].movedim(1, -1)


def _stereo_waveform(audio: Any) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise MMH3ResourceError("Decoded audio must be a ComfyUI AUDIO dictionary")
    waveform = audio.get("waveform")
    try:
        sample_rate = int(audio.get("sample_rate"))
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError("Decoded audio has no valid sample_rate") from exc
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1:
        raise MMH3ResourceError(
            f"Decoded audio waveform must be [1,C,L], got {getattr(waveform, 'shape', None)}"
        )
    if sample_rate <= 0 or waveform.shape[1] < 1:
        raise MMH3ResourceError("Decoded audio sample rate/channels are invalid")
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2, :]
    return waveform, sample_rate


def prepare_decoded_prefix(
    video: Any,
    *,
    audio: Any = None,
    prefix_frames: int = 39,
    width: int = 0,
    height: int = 0,
    missing_audio_policy: str = "error",
    audio_resampler: AudioResampler | None = None,
) -> PreparedDecodedPrefix:
    if missing_audio_policy not in {"error", "silence"}:
        raise MMH3ResourceError("missing_audio_policy must be 'error' or 'silence'")
    prefix_frames = int(prefix_frames)
    if not is_exact_h3_av_handover_boundary(prefix_frames):
        raise MMH3ResourceError("prefix_frames must be an exact H3 AV boundary (39, 90, 141, ...)")
    if not hasattr(video, "get_components"):
        raise MMH3ResourceError("Decoded video does not expose get_components()")
    components = video.get_components()
    frames = getattr(components, "images", None)
    frame_rate = getattr(components, "frame_rate", None)
    try:
        source_fps = float(frame_rate)
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError(f"Decoded video has invalid frame rate {frame_rate!r}") from exc
    normalized = _resample_frames(frames, source_fps, float(FPS))
    if normalized.shape[0] < prefix_frames:
        raise MMH3ResourceError(
            f"Decoded video has only {normalized.shape[0]} frames at {FPS} FPS; {prefix_frames} required"
        )

    source_h, source_w = int(normalized.shape[1]), int(normalized.shape[2])
    target_width = int(width) if int(width) > 0 else (source_w // 32) * 32
    target_height = int(height) if int(height) > 0 else (source_h // 32) * 32
    prefix = _normalize_canvas(normalized[-prefix_frames:], target_width, target_height)

    embedded_audio = getattr(components, "audio", None)
    selected_audio = audio if audio is not None else embedded_audio
    audio_source = "packet_audio" if audio is not None else ("embedded_audio" if embedded_audio is not None else "silence")
    required_samples = prefix_frames * AUDIO_SAMPLE_RATE // FPS
    source_audio_rate: int | None = None
    used_silence = False
    if selected_audio is None:
        if missing_audio_policy == "error":
            raise MMH3ResourceError(
                "Decoded video has no audio; choose missing_audio_policy='silence' explicitly to continue"
            )
        waveform = prefix.new_zeros((1, 2, required_samples))
        used_silence = True
    else:
        waveform, source_audio_rate = _stereo_waveform(selected_audio)
        if source_audio_rate != AUDIO_SAMPLE_RATE:
            resampler = audio_resampler or _default_audio_resampler
            waveform = resampler(waveform, source_audio_rate, AUDIO_SAMPLE_RATE)
        video_end_sample = round(int(normalized.shape[0]) * AUDIO_SAMPLE_RATE / FPS)
        available_end = min(video_end_sample, int(waveform.shape[-1]))
        start = available_end - required_samples
        if start < 0:
            if missing_audio_policy == "error":
                raise MMH3ResourceError(
                    f"Decoded audio is too short for the {prefix_frames}-frame synchronized tail"
                )
            pad = waveform.new_zeros((1, 2, -start))
            waveform = torch.cat((pad, waveform[..., :available_end]), dim=-1)
            used_silence = True
        else:
            waveform = waveform[..., start:available_end]

    warning = (
        "Decoded pixels/PCM are VAE-encoded before continuation; the seam is not a lossless "
        "direct-sampler latent handover."
    )
    plan = DecodedContinuationPlan(
        source_fps=source_fps,
        source_frames=int(frames.shape[0]),
        normalized_frames=int(normalized.shape[0]),
        prefix_frames=prefix_frames,
        width=target_width,
        height=target_height,
        source_audio_sample_rate=source_audio_rate,
        audio_source=audio_source,
        audio_samples=required_samples,
        missing_audio_policy=missing_audio_policy,
        used_silence=used_silence,
        round_trip_warning=warning,
    )
    return PreparedDecodedPrefix(prefix, waveform, plan)


def encode_decoded_prefix(
    prepared: PreparedDecodedPrefix,
    *,
    video_vae: Any,
    audio_vae: Any,
    target_frames: int = 124,
    audio_feather_frames: int = 0,
) -> DecodedContinuationResult:
    audio_vae_rate = int(getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE))
    if audio_vae_rate != AUDIO_SAMPLE_RATE:
        raise MMH3ResourceError(
            f"F03 requires the MiniMax H3 32 kHz audio VAE; got audio_sample_rate={audio_vae_rate}"
        )
    video_samples = video_vae.encode(prepared.frames)
    audio_samples = audio_vae.encode(prepared.waveform.movedim(1, -1))
    prefix_latent = {"samples": make_nested_tensor([video_samples, audio_samples])}
    validate_h3_av_latent(prefix_latent, strict_audio_length=True)
    continuation = build_h3_target_from_prefix(
        prefix_latent,
        target_frames=int(target_frames),
        audio_feather_frames=int(audio_feather_frames),
    )
    return DecodedContinuationResult(continuation.latent, prepared.plan, continuation)
