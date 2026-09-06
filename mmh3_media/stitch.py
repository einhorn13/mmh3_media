from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .archive import get_resource_payload
from .constants import AUDIO_SAMPLE_RATE, FPS
from .core import MMH3Media
from .errors import MMH3ResourceError
from .resource_model import resource_facts


@dataclass(frozen=True)
class DecodedSegment:
    frames: torch.Tensor
    waveform: torch.Tensor
    packet_id: str
    video_resource_id: str
    audio_resource_id: str


@dataclass(frozen=True)
class StitchCompatibility:
    ready: bool
    facts: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "contract": "mmh3_decoded_stitch_compatibility_v1",
            "segments": list(self.facts),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StitchPlan:
    video_mode: str
    audio_mode: str
    overlap_frames: int
    output_frames: int
    output_audio_samples: int
    segments: tuple[dict[str, Any], ...]
    seam_strategy: str = "manual"
    color_match_mode: str = "off"
    color_match_strength: float = 1.0
    color_match_decay_frames: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": True,
            "contract": "mmh3_decoded_av_stitch_v2",
            "fps": FPS,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "video_mode": self.video_mode,
            "audio_mode": self.audio_mode,
            "overlap_frames": self.overlap_frames,
            "output_frames": self.output_frames,
            "output_audio_samples": self.output_audio_samples,
            "duration_seconds": self.output_frames / FPS,
            "segments": list(self.segments),
            "seam_strategy": self.seam_strategy,
            "color_match_mode": self.color_match_mode,
            "color_match_strength": self.color_match_strength,
            "color_match_decay_frames": self.color_match_decay_frames,
            "audio_boundary_policy": "absolute_round(frame_index*sample_rate/fps)",
            "latent_concat_used": False,
            "auto_seam_policy": {
                "timeline": "fixed_overlap_no_duration_shift",
                "video_methods": ["crossfade", "ownership_cut"],
                "selection": "scene_and_motion_aware" if self.seam_strategy == "auto_seamless" else "manual",
                "color_match_scope": "seam_local_with_decay",
            },
        }

    def summary(self) -> str:
        return (
            f"READY · F05 decoded stitch · {len(self.segments)} segments · "
            f"{self.output_frames}f / {self.output_audio_samples} samples · {self.video_mode}"
        )


@dataclass(frozen=True)
class StitchResult:
    frames: torch.Tensor
    audio: dict[str, Any]
    plan: StitchPlan


def resolve_stitch_transition(transition: str, transition_frames: int, frame_counts: Sequence[int]) -> tuple[str, str, int]:
    """Auto = 0.25 s, shortened for small clips; never remove continuation context."""
    if transition == "Cut":
        return "cut", "cut", 0
    if transition not in {"Crossfade", "Auto Seamless"}:
        raise MMH3ResourceError(f"Unknown stitch transition {transition!r}")
    shortest = min(frame_counts)
    requested = int(transition_frames)
    if requested < 0:
        raise MMH3ResourceError("Transition frames must be 0 (automatic) or positive")
    overlap = requested or min(round(FPS * 0.25), shortest // 2)
    if overlap < 1 or overlap >= shortest:
        raise MMH3ResourceError("Crossfade needs at least two frames per segment and a transition shorter than every segment")
    return "crossfade", "half_cosine", overlap


def pcm_boundary(frame_index: int) -> int:
    frame_index = int(frame_index)
    if frame_index < 0:
        raise MMH3ResourceError("PCM frame boundary cannot be negative")
    return round(frame_index * AUDIO_SAMPLE_RATE / FPS)


def _metadata_number(metadata: dict[str, Any], key: str, cast):
    try:
        value = cast(metadata.get(key))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def inspect_stitch_packet(
    packet: MMH3Media,
    *,
    index: int = 0,
    expected_dimensions: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Inspect one decoded segment without materializing its video or audio payload."""
    reasons: list[str] = []
    if not isinstance(packet, MMH3Media):
        return {}, (f"Segment {index} is not an MMH3_MEDIA packet.",)
    video = packet.get_primary("video")
    audio = packet.get_primary("audio")
    if video is None or video.get("kind") != "video":
        reasons.append(f"Segment {index} has no primary decoded video.")
    if audio is None or audio.get("kind") != "audio":
        reasons.append(f"Segment {index} has no primary decoded audio.")
    video_meta = resource_facts(video) if isinstance(video, dict) else {}
    audio_meta = resource_facts(audio) if isinstance(audio, dict) else {}
    width, height = video_meta.get("width"), video_meta.get("height")
    dimensions = (int(width), int(height)) if isinstance(width, int) and isinstance(height, int) else None
    fps = _metadata_number(video_meta, "fps", float)
    if fps is None:
        fps = _metadata_number(packet.manifest.get("generation", {}), "fps", float)
    frame_count = _metadata_number(video_meta, "frames", int)
    sample_rate = _metadata_number(audio_meta, "sample_rate", int)
    channels = _metadata_number(audio_meta, "channels", int)
    samples = _metadata_number(audio_meta, "samples", int)
    if dimensions is None or dimensions[0] <= 0 or dimensions[1] <= 0:
        reasons.append(f"Segment {index} cannot prove decoded video dimensions from its canonical descriptor.")
    elif expected_dimensions is not None and dimensions != expected_dimensions:
        reasons.append(
            f"Segment {index} canvas {dimensions[0]}x{dimensions[1]} differs from "
            f"{expected_dimensions[0]}x{expected_dimensions[1]}."
        )
    if fps != float(FPS):
        reasons.append(f"Segment {index} must prove {FPS} FPS; got {fps!r}.")
    if frame_count is None:
        reasons.append(f"Segment {index} has no decoded frame_count metadata.")
    if sample_rate != AUDIO_SAMPLE_RATE:
        reasons.append(f"Segment {index} must prove {AUDIO_SAMPLE_RATE} Hz audio; got {sample_rate!r}.")
    if channels != 2:
        reasons.append(f"Segment {index} must prove stereo audio; got {channels!r} channel(s).")
    if samples is None:
        reasons.append(f"Segment {index} has no decoded audio sample count metadata.")
    elif frame_count is not None:
        delta = samples - pcm_boundary(frame_count)
        if abs(delta) > math.ceil(AUDIO_SAMPLE_RATE / FPS):
            reasons.append(
                f"Segment {index} AV duration mismatch is {delta} samples; "
                "maximum safe conform/trim is one video frame."
            )
    return {
        "index": index,
        "packet_id": packet.manifest["id"],
        "video_resource_id": video.get("id") if isinstance(video, dict) else None,
        "audio_resource_id": audio.get("id") if isinstance(audio, dict) else None,
        "dimensions": list(dimensions) if dimensions is not None else None,
        "fps": fps,
        "frame_count": frame_count,
        "audio_sample_rate": sample_rate,
        "audio_channels": channels,
        "audio_samples": samples,
    }, tuple(reasons)


def inspect_stitch_packets(packets: Sequence[MMH3Media]) -> StitchCompatibility:
    if len(packets) < 2:
        return StitchCompatibility(False, (), ("F05 stitch requires at least two ordered packets.",))
    reasons: list[str] = []
    facts: list[dict[str, Any]] = []
    expected_dimensions: tuple[int, int] | None = None
    for index, packet in enumerate(packets):
        fact, packet_reasons = inspect_stitch_packet(
            packet,
            index=index,
            expected_dimensions=expected_dimensions,
        )
        reasons.extend(packet_reasons)
        if fact:
            facts.append(fact)
            dimensions = fact.get("dimensions")
            if expected_dimensions is None and isinstance(dimensions, list) and len(dimensions) == 2:
                expected_dimensions = (int(dimensions[0]), int(dimensions[1]))
    return StitchCompatibility(not reasons, tuple(facts), tuple(reasons))


def materialize_decoded_segment(packet: MMH3Media) -> DecodedSegment:
    video_descriptor = packet.get_primary("video")
    audio_descriptor = packet.get_primary("audio")
    if video_descriptor is None or audio_descriptor is None:
        raise MMH3ResourceError("F05 requires primary decoded video and audio resources")
    video = get_resource_payload(packet, video_descriptor)
    audio = get_resource_payload(packet, audio_descriptor)
    if not hasattr(video, "get_components"):
        raise MMH3ResourceError("Decoded video does not expose get_components()")
    components = video.get_components()
    frames = getattr(components, "images", None)
    try:
        frame_rate = float(getattr(components, "frame_rate", None))
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError("Decoded video has invalid frame rate") from exc
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[0] < 1:
        raise MMH3ResourceError(f"Decoded video frames must be IMAGE [T,H,W,C], got {getattr(frames, 'shape', None)}")
    if frame_rate != float(FPS):
        raise MMH3ResourceError(f"F05 requires decoded video at {FPS} FPS; got {frame_rate}")
    if not isinstance(audio, dict):
        raise MMH3ResourceError("Decoded audio must be a ComfyUI AUDIO dictionary")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if (
        not isinstance(waveform, torch.Tensor)
        or waveform.ndim != 3
        or waveform.shape[0] != 1
        or waveform.shape[1] != 2
    ):
        raise MMH3ResourceError(
            f"F05 decoded audio must be stereo [1,2,L], got {getattr(waveform, 'shape', None)}"
        )
    if int(sample_rate) != AUDIO_SAMPLE_RATE:
        raise MMH3ResourceError(f"F05 requires decoded audio at {AUDIO_SAMPLE_RATE} Hz")
    return DecodedSegment(
        frames,
        waveform,
        packet.manifest["id"],
        video_descriptor["id"],
        audio_descriptor["id"],
    )


def _normalize_audio_length(
    waveform: torch.Tensor,
    target_samples: int,
    *,
    tolerance_samples: int,
) -> tuple[torch.Tensor, str, int]:
    actual = int(waveform.shape[-1])
    delta = target_samples - actual
    if abs(delta) > tolerance_samples:
        raise MMH3ResourceError(
            f"Decoded audio differs from its absolute PCM boundary by {delta} samples; "
            f"safe limit is {tolerance_samples}"
        )
    if delta == 0:
        return waveform, "none", 0
    if delta < 0:
        return waveform[..., :target_samples], "trim", delta
    if actual < 2:
        raise MMH3ResourceError("Audio undershoot cannot be time-conformed from fewer than two samples")
    conformed = F.interpolate(waveform, size=target_samples, mode="linear", align_corners=False)
    return conformed, "time_conform", delta


def stitch_decoded_segments(
    segments: Sequence[DecodedSegment],
    *,
    video_mode: str = "cut",
    audio_mode: str = "cut",
    overlap_frames: int = 0,
    tolerance_samples: int | None = None,
) -> StitchResult:
    if len(segments) < 2:
        raise MMH3ResourceError("F05 stitch requires at least two decoded segments")
    if (video_mode, audio_mode) not in {("cut", "cut"), ("crossfade", "half_cosine")}:
        raise MMH3ResourceError(
            "Synchronized F05 modes are cut+cut or crossfade+half_cosine; mixed duration semantics are unsafe"
        )
    overlap_frames = int(overlap_frames)
    if video_mode == "cut" and overlap_frames != 0:
        raise MMH3ResourceError("cut mode requires overlap_frames=0")
    if video_mode == "crossfade" and overlap_frames < 1:
        raise MMH3ResourceError("crossfade mode requires overlap_frames>=1")
    tolerance = (
        math.ceil(AUDIO_SAMPLE_RATE / FPS)
        if tolerance_samples is None
        else int(tolerance_samples)
    )
    if tolerance < 0:
        raise MMH3ResourceError("tolerance_samples cannot be negative")

    first = segments[0]
    if not isinstance(first.frames, torch.Tensor) or not isinstance(first.waveform, torch.Tensor):
        raise MMH3ResourceError("Decoded segment tensors are invalid")
    canvas = tuple(first.frames.shape[1:])
    output_frames = first.frames
    first_target = pcm_boundary(int(first.frames.shape[0]))
    output_audio, adjustment, delta = _normalize_audio_length(
        first.waveform, first_target, tolerance_samples=tolerance
    )
    segment_map: list[dict[str, Any]] = [
        {
            "index": 0,
            "packet_id": first.packet_id,
            "video_resource_id": first.video_resource_id,
            "audio_resource_id": first.audio_resource_id,
            "input_frames": int(first.frames.shape[0]),
            "input_audio_samples": int(first.waveform.shape[-1]),
            "output_frame_start": 0,
            "output_frame_end": int(first.frames.shape[0]),
            "output_sample_start": 0,
            "output_sample_end": first_target,
            "audio_adjustment": adjustment,
            "audio_adjustment_samples": delta,
        }
    ]
    total_frames = int(first.frames.shape[0])

    for index, segment in enumerate(segments[1:], start=1):
        if tuple(segment.frames.shape[1:]) != canvas:
            raise MMH3ResourceError(
                f"Segment {index} frame canvas/channels {tuple(segment.frames.shape[1:])} differs from {canvas}"
            )
        input_frames = int(segment.frames.shape[0])
        if video_mode == "crossfade" and overlap_frames >= min(total_frames, input_frames):
            raise MMH3ResourceError("overlap_frames must be shorter than both sides of every seam")
        frame_start = total_frames - overlap_frames
        frame_end = frame_start + input_frames
        sample_start = pcm_boundary(frame_start)
        sample_end = pcm_boundary(frame_end)
        target_segment_samples = sample_end - sample_start
        normalized_audio, adjustment, delta = _normalize_audio_length(
            segment.waveform,
            target_segment_samples,
            tolerance_samples=tolerance,
        )

        if video_mode == "cut":
            output_frames = torch.cat((output_frames, segment.frames), dim=0)
            output_audio = torch.cat((output_audio, normalized_audio), dim=-1)
        else:
            ramp = 0.5 - 0.5 * torch.cos(
                torch.linspace(
                    0.0,
                    math.pi,
                    overlap_frames + 2,
                    dtype=output_frames.dtype,
                    device=output_frames.device,
                )[1:-1]
            )
            blended_frames = (
                output_frames[-overlap_frames:] * (1.0 - ramp[:, None, None, None])
                + segment.frames[:overlap_frames] * ramp[:, None, None, None]
            )
            output_frames = torch.cat(
                (output_frames[:-overlap_frames], blended_frames, segment.frames[overlap_frames:]),
                dim=0,
            )
            overlap_samples = int(output_audio.shape[-1]) - sample_start
            if overlap_samples < 1 or overlap_samples >= normalized_audio.shape[-1]:
                raise MMH3ResourceError("Audio overlap derived from absolute PCM boundaries is invalid")
            audio_ramp = 0.5 - 0.5 * torch.cos(
                torch.linspace(
                    0.0,
                    math.pi,
                    overlap_samples + 2,
                    dtype=output_audio.dtype,
                    device=output_audio.device,
                )[1:-1]
            )
            blended_audio = (
                output_audio[..., -overlap_samples:] * (1.0 - audio_ramp)
                + normalized_audio[..., :overlap_samples] * audio_ramp
            )
            output_audio = torch.cat(
                (output_audio[..., :-overlap_samples], blended_audio, normalized_audio[..., overlap_samples:]),
                dim=-1,
            )
        total_frames = frame_end
        segment_map.append(
            {
                "index": index,
                "packet_id": segment.packet_id,
                "video_resource_id": segment.video_resource_id,
                "audio_resource_id": segment.audio_resource_id,
                "input_frames": input_frames,
                "input_audio_samples": int(segment.waveform.shape[-1]),
                "output_frame_start": frame_start,
                "output_frame_end": frame_end,
                "output_sample_start": sample_start,
                "output_sample_end": sample_end,
                "audio_adjustment": adjustment,
                "audio_adjustment_samples": delta,
            }
        )

    expected_samples = pcm_boundary(total_frames)
    if int(output_audio.shape[-1]) != expected_samples:
        raise MMH3ResourceError(
            f"Internal F05 PCM drift: produced {output_audio.shape[-1]}, expected {expected_samples}"
        )
    plan = StitchPlan(
        video_mode,
        audio_mode,
        overlap_frames,
        total_frames,
        expected_samples,
        tuple(segment_map),
    )
    return StitchResult(
        output_frames,
        {"waveform": output_audio, "sample_rate": AUDIO_SAMPLE_RATE},
        plan,
    )
