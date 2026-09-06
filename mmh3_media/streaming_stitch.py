from __future__ import annotations

import io
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from .archive import get_resource_payload
from .constants import AUDIO_SAMPLE_RATE, FPS
from .core import MMH3Media
from .errors import MMH3ResourceError
from .stitch import StitchPlan, _normalize_audio_length, pcm_boundary


@dataclass(frozen=True)
class StreamingSegment:
    video: Any
    frame_count: int
    waveform: torch.Tensor
    packet_id: str
    video_resource_id: str
    audio_resource_id: str


@dataclass(frozen=True)
class StreamingStitchResult:
    video: "StreamingStitchedVideo"
    audio: dict[str, Any]
    plan: StitchPlan


def _half_cosine(length: int) -> np.ndarray:
    positions = np.arange(1, length + 1, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(np.pi * positions / (length + 1))


def _sample_rgb(frame: np.ndarray, max_side: int = 96) -> np.ndarray:
    """Cheap deterministic spatial sampling for seam analysis without full-frame copies."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise MMH3ResourceError(f"Expected RGB24 frame for seam analysis; got {frame.shape}")
    height, width = frame.shape[:2]
    stride = max(1, math.ceil(max(height, width) / max_side))
    return frame[::stride, ::stride, :].astype(np.float32, copy=False)


def _rgb_stats(frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        raise MMH3ResourceError("Color matching requires at least one seam frame")
    data = np.concatenate([_sample_rgb(frame).reshape(-1, 3) for frame in frames], axis=0)
    # Percentile clipping makes a bright specular highlight or a black letterbox much
    # less likely to drive the whole correction.
    low = np.percentile(data, 2.0, axis=0)
    high = np.percentile(data, 98.0, axis=0)
    clipped = np.clip(data, low, high)
    return clipped.mean(axis=0), clipped.std(axis=0)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _frame_delta(left: np.ndarray, right: np.ndarray) -> float:
    a = _luma(_sample_rgb(left))
    b = _luma(_sample_rgb(right))
    if a.shape != b.shape:
        raise MMH3ResourceError(f"Seam analysis canvas mismatch: {left.shape} vs {right.shape}")
    return float(np.mean(np.abs(a - b)) / 255.0)


def _normalized_structure_delta(left: np.ndarray, right: np.ndarray) -> float:
    """Brightness/color-insensitive structural distance used for seam ownership search."""
    a = _luma(_sample_rgb(left)).astype(np.float32)
    b = _luma(_sample_rgb(right)).astype(np.float32)
    if a.shape != b.shape:
        raise MMH3ResourceError(f"Seam analysis canvas mismatch: {left.shape} vs {right.shape}")
    a = (a - float(a.mean())) / max(float(a.std()), 4.0)
    b = (b - float(b.mean())) / max(float(b.std()), 4.0)
    return float(np.mean(np.abs(a - b)) / 4.0)


def _motion_energy(frames: Sequence[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    return float(np.mean([_normalized_structure_delta(a, b) for a, b in zip(frames[:-1], frames[1:])]))


def _edge_histogram(frame: np.ndarray, bins: int = 12) -> np.ndarray:
    gray = _luma(_sample_rgb(frame))
    dx = np.diff(gray, axis=1, prepend=gray[:, :1])
    dy = np.diff(gray, axis=0, prepend=gray[:1, :])
    magnitude = np.hypot(dx, dy)
    scale = max(float(np.percentile(magnitude, 95.0)), 1.0)
    hist, _ = np.histogram(np.clip(magnitude / scale, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    return hist / max(float(hist.sum()), 1.0)


def _scene_similarity(left: np.ndarray, right: np.ndarray) -> float:
    # Gradient-distribution similarity is deliberately insensitive to an exposure jump.
    a = _edge_histogram(left)
    b = _edge_histogram(right)
    edge_similarity = float(np.minimum(a, b).sum())
    structure = _normalized_structure_delta(left, right)
    structure_similarity = max(0.0, 1.0 - structure)
    return 0.55 * edge_similarity + 0.45 * structure_similarity


def _color_transform(
    left: Sequence[np.ndarray], right: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    left_mean, left_std = _rgb_stats(left)
    right_mean, right_std = _rgb_stats(right)
    safe_std = np.maximum(right_std, 1.0)
    gain = np.clip(left_std / safe_std, 0.75, 1.33)
    bias = np.clip(left_mean - right_mean * gain, -32.0, 32.0)

    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    left_luma = float(np.dot(left_mean, weights))
    right_luma = float(np.dot(right_mean, weights))
    exposure_jump = abs(left_luma - right_luma) / 255.0
    left_chroma = left_mean - left_luma
    right_chroma = right_mean - right_luma
    white_balance_jump = float(np.max(np.abs(left_chroma - right_chroma)) / 255.0)
    left_contrast = float(np.dot(left_std, weights))
    right_contrast = float(np.dot(right_std, weights))
    contrast_jump = abs(left_contrast - right_contrast) / max(left_contrast, right_contrast, 16.0)
    score = max(exposure_jump, white_balance_jump, min(contrast_jump, 1.0))
    return gain, bias, score, {
        "exposure_jump": exposure_jump,
        "white_balance_jump": white_balance_jump,
        "contrast_jump": contrast_jump,
    }


def _apply_color_transform(frame: np.ndarray, gain: np.ndarray, bias: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return frame
    data = frame.astype(np.float32)
    corrected = data * gain.reshape(1, 1, 3) + bias.reshape(1, 1, 3)
    mixed = data * (1.0 - alpha) + corrected * alpha
    return np.rint(mixed).clip(0, 255).astype(np.uint8)


def _auto_seam_decision(left: Sequence[np.ndarray], right: Sequence[np.ndarray]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise MMH3ResourceError("Auto Seamless requires equal non-empty seam probes")
    boundary_delta = _frame_delta(left[-1], right[0])
    boundary_structure = _normalized_structure_delta(left[-1], right[0])
    left_motion = _motion_energy(left)
    right_motion = _motion_energy(right)
    reference_motion = max((left_motion + right_motion) * 0.5, 0.012)
    motion_risk = boundary_structure / reference_motion
    scene_similarity = _scene_similarity(left[-1], right[0])

    best_split = max(1, len(left) // 2)
    best_score = float("inf")
    if len(left) > 1:
        for split in range(1, len(left)):
            score = _normalized_structure_delta(left[split - 1], right[split])
            if score < best_score:
                best_score = score
                best_split = split
    else:
        best_score = _normalized_structure_delta(left[0], right[0])

    scene_change = scene_similarity < 0.42 and boundary_delta > 0.12
    # Crossfade is useful when frames are genuinely close or when the boundary is no
    # harsher than the motion already present inside the clips. Otherwise blending
    # moving edges creates visible double images, so use an ownership cut inside the
    # same overlap window (duration stays identical).
    use_cut = scene_change or (motion_risk > 2.8 and boundary_delta > 0.055)
    return {
        "method": "ownership_cut" if use_cut else "crossfade",
        "split": int(best_split),
        "boundary_delta": float(boundary_delta),
        "boundary_structure": float(boundary_structure),
        "left_motion": float(left_motion),
        "right_motion": float(right_motion),
        "motion_risk": float(motion_risk),
        "scene_similarity": float(scene_similarity),
        "scene_change": bool(scene_change),
        "ownership_score": float(best_score),
    }

def _emit_except_tail(
    iterator: Iterator[np.ndarray],
    overlap_frames: int,
    stats: dict[str, int],
) -> Iterator[np.ndarray]:
    tail: deque[np.ndarray] = deque()
    count = 0
    for frame in iterator:
        count += 1
        tail.append(frame)
        stats["max_buffered_frames"] = max(stats.get("max_buffered_frames", 0), len(tail))
        if len(tail) > overlap_frames:
            yield tail.popleft()
    return list(tail), count


def iter_stitched_rgb_frames(
    frame_iterators: Sequence[Iterable[np.ndarray]],
    frame_counts: Sequence[int],
    *,
    video_mode: str,
    overlap_frames: int,
    color_match_mode: str = "off",
    color_match_strength: float = 1.0,
    color_match_decay_frames: int = 24,
    seam_strategy: str = "manual",
    stats: dict[str, Any] | None = None,
) -> Iterator[np.ndarray]:
    if len(frame_iterators) != len(frame_counts) or len(frame_iterators) < 2:
        raise MMH3ResourceError("Streaming stitch requires matching iterators/counts for at least two segments")
    stats = stats if stats is not None else {}
    stats.setdefault("max_buffered_frames", 0)
    stats.setdefault("color_match_seams", 0)
    stats.setdefault("color_match_max_score", 0.0)
    stats.setdefault("auto_crossfade_seams", 0)
    stats.setdefault("auto_ownership_cut_seams", 0)
    stats.setdefault("auto_scene_change_seams", 0)
    stats.setdefault("auto_motion_risk_max", 0.0)
    stats.setdefault("seam_decisions", [])
    if seam_strategy not in {"manual", "auto_seamless"}:
        raise MMH3ResourceError(f"Unknown seam_strategy {seam_strategy!r}")
    if color_match_mode not in {"off", "auto", "always"}:
        raise MMH3ResourceError(f"Unknown color_match_mode {color_match_mode!r}")
    color_match_strength = float(color_match_strength)
    if not 0.0 <= color_match_strength <= 1.0:
        raise MMH3ResourceError("color_match_strength must be between 0 and 1")
    color_match_decay_frames = max(0, int(color_match_decay_frames))
    if video_mode == "cut":
        if overlap_frames != 0:
            raise MMH3ResourceError("Streaming cut requires overlap_frames=0")
        for index, (frames, expected) in enumerate(zip(frame_iterators, frame_counts)):
            count = 0
            for frame in frames:
                count += 1
                yield frame
            if count != int(expected):
                raise MMH3ResourceError(
                    f"Streaming segment {index} decoded {count} frames; manifest promised {expected}"
                )
        return
    if video_mode != "crossfade" or overlap_frames < 1:
        raise MMH3ResourceError("Streaming video mode must be cut or crossfade with a positive overlap")

    first_iterator = iter(frame_iterators[0])
    first_tail, first_count = yield from _emit_except_tail(first_iterator, overlap_frames, stats)
    if first_count != int(frame_counts[0]):
        raise MMH3ResourceError(
            f"Streaming segment 0 decoded {first_count} frames; manifest promised {frame_counts[0]}"
        )
    if len(first_tail) != overlap_frames:
        raise MMH3ResourceError("First streaming segment is shorter than the requested overlap")
    previous_tail: deque[np.ndarray] = deque(first_tail)

    for index in range(1, len(frame_iterators)):
        iterator = iter(frame_iterators[index])
        head: deque[np.ndarray] = deque()
        for _ in range(overlap_frames):
            try:
                head.append(next(iterator))
            except StopIteration as exc:
                raise MMH3ResourceError(
                    f"Streaming segment {index} is shorter than the requested overlap"
                ) from exc
        stats["max_buffered_frames"] = max(
            stats.get("max_buffered_frames", 0), len(previous_tail) + len(head)
        )
        ramp = _half_cosine(overlap_frames)
        left_probe = list(previous_tail)
        right_probe = list(head)
        decision = {
            "method": "crossfade",
            "split": max(1, overlap_frames // 2),
            "scene_change": False,
            "motion_risk": 0.0,
            "scene_similarity": 1.0,
            "boundary_delta": 0.0,
        }
        if seam_strategy == "auto_seamless":
            decision = _auto_seam_decision(left_probe, right_probe)
            stats["auto_motion_risk_max"] = max(
                float(stats.get("auto_motion_risk_max", 0.0)), float(decision["motion_risk"])
            )
            if decision["method"] == "ownership_cut":
                stats["auto_ownership_cut_seams"] = int(stats.get("auto_ownership_cut_seams", 0)) + 1
            else:
                stats["auto_crossfade_seams"] = int(stats.get("auto_crossfade_seams", 0)) + 1
            if decision["scene_change"]:
                stats["auto_scene_change_seams"] = int(stats.get("auto_scene_change_seams", 0)) + 1

        gain = np.ones(3, dtype=np.float32)
        bias = np.zeros(3, dtype=np.float32)
        score = 0.0
        color_parts = {"exposure_jump": 0.0, "white_balance_jump": 0.0, "contrast_jump": 0.0}
        apply_match = False
        if color_match_mode != "off":
            gain, bias, score, color_parts = _color_transform(left_probe, right_probe)
            stats["color_match_max_score"] = max(float(stats.get("color_match_max_score", 0.0)), score)
            # Auto matching is suppressed for a detected scene change: different scenes
            # should not be forced into the same palette. "Always" remains an explicit override.
            apply_match = color_match_mode == "always" or (
                color_match_mode == "auto" and not bool(decision["scene_change"]) and score >= 0.055
            )
            if apply_match:
                stats["color_match_seams"] = int(stats.get("color_match_seams", 0)) + 1

        seam_record = dict(decision)
        seam_record.update(color_parts)
        seam_record["color_match_score"] = float(score)
        seam_record["color_match_applied"] = bool(apply_match)
        stats["seam_decisions"].append(seam_record)

        corrected_index = 0
        def correction_alpha() -> float:
            nonlocal corrected_index
            if not apply_match:
                corrected_index += 1
                return 0.0
            if color_match_decay_frames <= 0:
                alpha = color_match_strength if corrected_index < overlap_frames else 0.0
            else:
                alpha = color_match_strength * max(0.0, 1.0 - corrected_index / color_match_decay_frames)
            corrected_index += 1
            return alpha

        corrected_head = [
            _apply_color_transform(frame, gain, bias, correction_alpha()) for frame in right_probe
        ]

        def corrected_right_body() -> Iterator[np.ndarray]:
            for frame in iterator:
                yield _apply_color_transform(frame, gain, bias, correction_alpha())

        def seam_frames() -> Iterator[np.ndarray]:
            if decision["method"] == "ownership_cut":
                split = int(decision["split"])
                for offset, (left, right) in enumerate(zip(left_probe, corrected_head)):
                    if left.shape != right.shape:
                        raise MMH3ResourceError(
                            f"Streaming seam canvas mismatch: {left.shape} vs {right.shape}"
                        )
                    yield left if offset < split else right
                previous_tail.clear()
                return
            offset = 0
            while previous_tail:
                left = previous_tail.popleft()
                right = corrected_head[offset]
                if left.shape != right.shape:
                    raise MMH3ResourceError(
                        f"Streaming seam canvas mismatch: {left.shape} vs {right.shape}"
                    )
                yield np.rint(
                    left.astype(np.float32) * (1.0 - ramp[offset])
                    + right.astype(np.float32) * ramp[offset]
                ).clip(0, 255).astype(np.uint8)
                offset += 1

        # Retain the tail of the assembled output, not merely the unblended body of
        # the current source. This also covers neighboring overlaps that intersect
        # inside a short intermediate segment (for example 10 frames, overlap 6).
        assembled = chain(seam_frames(), corrected_right_body())
        if index == len(frame_iterators) - 1:
            count = 0
            for frame in assembled:
                count += 1
                yield frame
        else:
            tail, count = yield from _emit_except_tail(assembled, overlap_frames, stats)
            if len(tail) != overlap_frames:
                raise MMH3ResourceError(f"Streaming segment {index} has no complete tail overlap")
            previous_tail = deque(tail)
        if count != int(frame_counts[index]):
            raise MMH3ResourceError(
                f"Streaming segment {index} decoded {count} frames; manifest promised {frame_counts[index]}"
            )


def materialize_streaming_segment(packet: MMH3Media, fact: dict[str, Any]) -> StreamingSegment:
    video_descriptor = packet.get_primary("video")
    audio_descriptor = packet.get_primary("audio")
    if video_descriptor is None or audio_descriptor is None:
        raise MMH3ResourceError("Streaming F05 requires explicit primary video and audio")
    video = get_resource_payload(packet, video_descriptor)
    audio = get_resource_payload(packet, audio_descriptor)
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[:2] != (1, 2):
        raise MMH3ResourceError("Streaming F05 audio must be [1,2,L]")
    if int(audio.get("sample_rate")) != AUDIO_SAMPLE_RATE:
        raise MMH3ResourceError(f"Streaming F05 requires {AUDIO_SAMPLE_RATE} Hz audio")
    frame_count = int(fact.get("frame_count") or 0)
    if frame_count < 1:
        raise MMH3ResourceError("Streaming F05 requires manifest frame_count")
    return StreamingSegment(
        video,
        frame_count,
        waveform,
        packet.manifest["id"],
        video_descriptor["id"],
        audio_descriptor["id"],
    )


def _streaming_audio_plan(
    segments: Sequence[StreamingSegment],
    *,
    video_mode: str,
    audio_mode: str,
    overlap_frames: int,
) -> tuple[dict[str, Any], StitchPlan]:
    if (video_mode, audio_mode) not in {("cut", "cut"), ("crossfade", "half_cosine")}:
        raise MMH3ResourceError("Streaming F05 requires synchronized cut+cut or crossfade+half_cosine")
    if video_mode == "cut" and overlap_frames != 0:
        raise MMH3ResourceError("Streaming cut requires overlap_frames=0")
    if video_mode == "crossfade" and overlap_frames < 1:
        raise MMH3ResourceError("Streaming crossfade requires overlap_frames>=1")
    tolerance = math.ceil(AUDIO_SAMPLE_RATE / FPS)
    first = segments[0]
    total_frames = first.frame_count
    first_end = pcm_boundary(total_frames)
    output_audio, adjustment, delta = _normalize_audio_length(
        first.waveform, first_end, tolerance_samples=tolerance
    )
    segment_map: list[dict[str, Any]] = [
        {
            "index": 0,
            "packet_id": first.packet_id,
            "video_resource_id": first.video_resource_id,
            "audio_resource_id": first.audio_resource_id,
            "input_frames": first.frame_count,
            "input_audio_samples": int(first.waveform.shape[-1]),
            "output_frame_start": 0,
            "output_frame_end": total_frames,
            "output_sample_start": 0,
            "output_sample_end": first_end,
            "audio_adjustment": adjustment,
            "audio_adjustment_samples": delta,
        }
    ]
    for index, segment in enumerate(segments[1:], start=1):
        if video_mode == "crossfade" and overlap_frames >= min(total_frames, segment.frame_count):
            raise MMH3ResourceError("Streaming overlap must be shorter than both sides of every seam")
        frame_start = total_frames - overlap_frames
        frame_end = frame_start + segment.frame_count
        sample_start = pcm_boundary(frame_start)
        sample_end = pcm_boundary(frame_end)
        normalized, adjustment, delta = _normalize_audio_length(
            segment.waveform,
            sample_end - sample_start,
            tolerance_samples=tolerance,
        )
        if video_mode == "cut":
            output_audio = torch.cat((output_audio, normalized), dim=-1)
        else:
            overlap_samples = int(output_audio.shape[-1]) - sample_start
            ramp = 0.5 - 0.5 * torch.cos(
                torch.linspace(
                    0.0,
                    math.pi,
                    overlap_samples + 2,
                    dtype=output_audio.dtype,
                    device=output_audio.device,
                )[1:-1]
            )
            blended = (
                output_audio[..., -overlap_samples:] * (1.0 - ramp)
                + normalized[..., :overlap_samples] * ramp
            )
            output_audio = torch.cat(
                (output_audio[..., :-overlap_samples], blended, normalized[..., overlap_samples:]),
                dim=-1,
            )
        total_frames = frame_end
        segment_map.append(
            {
                "index": index,
                "packet_id": segment.packet_id,
                "video_resource_id": segment.video_resource_id,
                "audio_resource_id": segment.audio_resource_id,
                "input_frames": segment.frame_count,
                "input_audio_samples": int(segment.waveform.shape[-1]),
                "output_frame_start": frame_start,
                "output_frame_end": frame_end,
                "output_sample_start": sample_start,
                "output_sample_end": sample_end,
                "audio_adjustment": adjustment,
                "audio_adjustment_samples": delta,
            }
        )
    expected = pcm_boundary(total_frames)
    if int(output_audio.shape[-1]) != expected:
        raise MMH3ResourceError("Streaming F05 produced internal PCM drift")
    plan = StitchPlan(
        video_mode,
        audio_mode,
        overlap_frames,
        total_frames,
        expected,
        tuple(segment_map),
    )
    return {"waveform": output_audio, "sample_rate": AUDIO_SAMPLE_RATE}, plan


def build_streaming_stitch(
    segments: Sequence[StreamingSegment],
    *,
    dimensions: tuple[int, int],
    video_mode: str,
    audio_mode: str,
    overlap_frames: int,
    seam_strategy: str = "manual",
    color_match_mode: str = "off",
    color_match_strength: float = 1.0,
    color_match_decay_frames: int = 24,
) -> StreamingStitchResult:
    if len(segments) < 2:
        raise MMH3ResourceError("Streaming F05 requires at least two segments")
    audio, plan = _streaming_audio_plan(
        segments,
        video_mode=video_mode,
        audio_mode=audio_mode,
        overlap_frames=int(overlap_frames),
    )
    plan = StitchPlan(
        plan.video_mode, plan.audio_mode, plan.overlap_frames, plan.output_frames,
        plan.output_audio_samples, plan.segments, seam_strategy=seam_strategy,
        color_match_mode=color_match_mode, color_match_strength=float(color_match_strength),
        color_match_decay_frames=int(color_match_decay_frames),
    )
    video = StreamingStitchedVideo(
        tuple(segment.video for segment in segments),
        tuple(segment.frame_count for segment in segments),
        dimensions,
        audio,
        plan,
    )
    return StreamingStitchResult(video, audio, plan)


class StreamingStitchedVideo:
    _mmh3_skip_components_metadata = True

    def __init__(
        self,
        sources: tuple[Any, ...],
        frame_counts: tuple[int, ...],
        dimensions: tuple[int, int],
        audio: dict[str, Any] | None,
        plan: StitchPlan,
    ):
        self.sources = sources
        self.frame_counts = frame_counts
        self.dimensions = (int(dimensions[0]), int(dimensions[1]))
        self.audio = audio
        self.plan = plan
        self.last_save_stats: dict[str, int] = {}

    def get_dimensions(self) -> tuple[int, int]:
        return self.dimensions

    def get_duration(self) -> float:
        return self.plan.output_frames / FPS

    def get_frame_count(self) -> int:
        return self.plan.output_frames

    def get_frame_rate(self) -> Fraction:
        return Fraction(FPS)

    def get_bit_depth(self) -> int:
        return 8

    def get_color_space(self) -> str:
        return "sRGB"

    @staticmethod
    def _source_frames(source: Any) -> Iterator[np.ndarray]:
        try:
            import av
        except ImportError as exc:  # pragma: no cover - supplied by ComfyUI
            raise MMH3ResourceError("Streaming video encode requires PyAV") from exc
        if not hasattr(source, "get_stream_source"):
            raise MMH3ResourceError("Streaming source video does not expose get_stream_source()")
        stream_source = source.get_stream_source()
        if isinstance(stream_source, io.BytesIO):
            stream_source.seek(0)
        with av.open(stream_source, mode="r") as container:
            if not container.streams.video:
                raise MMH3ResourceError("Streaming source contains no video stream")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")

    def _iter_frames(self, stats: dict[str, int]) -> Iterator[np.ndarray]:
        if len(self.sources) == 1:
            count = 0
            for frame in self._source_frames(self.sources[0]):
                count += 1
                yield frame
            if count != int(self.frame_counts[0]):
                raise MMH3ResourceError(
                    f"Streaming segment 0 decoded {count} frames; manifest promised {self.frame_counts[0]}"
                )
            return
        yield from iter_stitched_rgb_frames(
            [self._source_frames(source) for source in self.sources],
            self.frame_counts,
            video_mode=self.plan.video_mode,
            overlap_frames=self.plan.overlap_frames,
            color_match_mode=self.plan.color_match_mode,
            color_match_strength=self.plan.color_match_strength,
            color_match_decay_frames=self.plan.color_match_decay_frames,
            seam_strategy=self.plan.seam_strategy,
            stats=stats,
        )

    def get_components(self):
        frames = list(self._iter_frames({}))
        images = torch.from_numpy(np.stack(frames)).to(dtype=torch.float32).div_(255.0)
        from comfy_api.latest import Types  # type: ignore

        return Types.VideoComponents(images=images, audio=self.audio, frame_rate=Fraction(FPS))

    def get_stream_source(self):
        buffer = io.BytesIO()
        self.save_to(buffer)
        buffer.seek(0)
        return buffer

    def as_trimmed(self, start_time=None, duration=None, strict_duration=True):
        from comfy_api.latest import InputImpl  # type: ignore

        video = InputImpl.VideoFromFile(self.get_stream_source())
        return video.as_trimmed(start_time, duration, strict_duration)

    def save_to(
        self,
        path,
        format="auto",
        codec="auto",
        metadata=None,
        bit_depth=None,
        crf=None,
        color_space=None,
    ):
        try:
            import av
        except ImportError as exc:  # pragma: no cover - supplied by ComfyUI
            raise MMH3ResourceError("Streaming video encode requires PyAV") from exc
        format_value = getattr(format, "value", format)
        codec_value = getattr(codec, "value", codec)
        suffix = Path(os.fspath(path)).suffix.lower() if isinstance(path, (str, os.PathLike)) else ".mp4"
        if format_value == "auto":
            format_value = {".mkv": "matroska", ".webm": "webm"}.get(suffix, "mp4")
        else:
            format_value = {"mkv": "matroska"}.get(str(format_value), str(format_value))
        if codec_value == "auto":
            codec_value = "libsvtav1" if format_value == "webm" else "h264"
        elif codec_value == "av1":
            codec_value = "libsvtav1"
        open_kwargs: dict[str, Any] = {"mode": "w", "format": format_value}
        if format_value == "mp4":
            open_kwargs["options"] = {
                "movflags": "use_metadata_tags" if not isinstance(path, (str, os.PathLike)) else "use_metadata_tags+faststart"
            }
        width, height = self.dimensions
        if width % 2 or height % 2:
            raise MMH3ResourceError(f"Streaming H.264 output requires even dimensions; got {width}x{height}")
        stats: dict[str, Any] = {"max_buffered_frames": 0, "encoded_frames": 0}
        with av.open(path, **open_kwargs) as output:
            if metadata:
                for key, value in metadata.items():
                    output.metadata[key] = value if isinstance(value, str) else json.dumps(value)
            video_stream = output.add_stream(codec_value, rate=Fraction(FPS))
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            if crf is not None:
                video_stream.options = {"crf": str(crf)}
            audio_stream = None
            if self.audio is not None:
                audio_stream = output.add_stream(
                    "libopus" if format_value == "webm" else "aac",
                    rate=AUDIO_SAMPLE_RATE,
                    layout="stereo",
                )
            for index, image in enumerate(self._iter_frames(stats)):
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, FPS)
                for packet in video_stream.encode(frame):
                    output.mux(packet)
                stats["encoded_frames"] += 1
            for packet in video_stream.encode(None):
                output.mux(packet)

            if audio_stream is not None:
                waveform = self.audio["waveform"][0, :, : self.plan.output_audio_samples]
                audio_frame = av.AudioFrame.from_ndarray(
                    waveform.float().cpu().contiguous().numpy(),
                    format="fltp",
                    layout="stereo",
                )
                audio_frame.sample_rate = AUDIO_SAMPLE_RATE
                audio_frame.pts = 0
                for packet in audio_stream.encode(audio_frame):
                    output.mux(packet)
                for packet in audio_stream.encode(None):
                    output.mux(packet)
        if stats["encoded_frames"] != self.plan.output_frames:
            raise MMH3ResourceError(
                f"Streaming encoder wrote {stats['encoded_frames']} frames; expected {self.plan.output_frames}"
            )
        self.last_save_stats = stats
