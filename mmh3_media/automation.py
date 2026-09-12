from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import MMH3ResourceError


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
H3_AUDIO_TIMELINE_PLANNERS = frozenset({"h3_audio_timeline", "h3_audio_interactive"})


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def _job_id(material: dict[str, Any]) -> str:
    body = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def plan_batch_inputs(paths: Iterable[str], *, order: str = "natural") -> dict[str, Any]:
    if order not in {"natural", "provided"}:
        raise MMH3ResourceError("Batch order must be natural or provided")
    normalized = [str(Path(value).as_posix()) for value in paths if str(value).strip()]
    if not normalized:
        raise MMH3ResourceError("Batch input list is empty")
    if len(set(map(str.casefold, normalized))) != len(normalized):
        raise MMH3ResourceError("Batch input list contains duplicates")
    if order == "natural":
        normalized.sort(key=_natural_key)

    jobs = []
    for index, path in enumerate(normalized):
        suffix = Path(path).suffix.casefold()
        if suffix == ".mmh3":
            kind = "mmh3"
        elif suffix in VIDEO_SUFFIXES:
            kind = "video"
        else:
            raise MMH3ResourceError(f"Unsupported batch input extension: {path}")
        material = {"index": index, "path": path, "kind": kind}
        jobs.append({**material, "job_id": _job_id(material), "status": "planned"})
    return {"contract": "mmh3_batch_input_plan_v1", "order": order, "jobs": jobs}


@dataclass(frozen=True)
class ChunkPlan:
    source_id: str
    total_frames: int
    fps: Fraction
    audio_sample_rate: int
    chunks: tuple[dict[str, Any], ...]
    excluded_tail: dict[str, int] | None
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "mmh3_long_video_chunk_plan_v1",
            "source_id": self.source_id,
            "total_frames": self.total_frames,
            "fps": {"numerator": self.fps.numerator, "denominator": self.fps.denominator},
            "audio_sample_rate": self.audio_sample_rate,
            "settings": dict(self.settings),
            "chunks": [dict(chunk) for chunk in self.chunks],
            "excluded_tail": dict(self.excluded_tail) if self.excluded_tail else None,
        }

    def summary(self) -> str:
        dropped = "" if not self.excluded_tail else f" · dropped_tail={self.excluded_tail['frames']}f"
        return f"READY · long-video plan · chunks={len(self.chunks)}{dropped}"


@dataclass(frozen=True)
class ChunkExecutionSelection:
    job: dict[str, Any]
    fps: Fraction
    audio_sample_rate: int
    start_time_seconds: float
    duration_seconds: float

    def context(self) -> dict[str, Any]:
        read_start = int(self.job["read_start_frame"])
        write_start = int(self.job["write_start_frame"])
        return {
            "contract": "mmh3_video_chunk_context_v1",
            "job_id": self.job["job_id"],
            "index": int(self.job["index"]),
            "read_start_frame": read_start,
            "read_end_frame": int(self.job["read_end_frame"]),
            "write_start_frame": write_start,
            "write_end_frame": int(self.job["write_end_frame"]),
            "write_start_in_chunk": write_start - read_start,
            "write_end_in_chunk": int(self.job["write_end_frame"]) - read_start,
            "audio_read_start": int(self.job["audio_read_start"]),
            "audio_read_end": int(self.job["audio_read_end"]),
            "audio_write_start": int(self.job["audio_write_start"]),
            "audio_write_end": int(self.job["audio_write_end"]),
            "fps": {"numerator": self.fps.numerator, "denominator": self.fps.denominator},
            "audio_sample_rate": self.audio_sample_rate,
            "start_time_seconds": self.start_time_seconds,
            "duration_seconds": self.duration_seconds,
            "output_name": self.job["output_name"],
        }


def resolve_chunk_execution(
    plan: dict[str, Any], *, chunk_index: int = 0, job_id: str = ""
) -> ChunkExecutionSelection:
    if not isinstance(plan, dict) or plan.get("contract") != "mmh3_long_video_chunk_plan_v1":
        raise MMH3ResourceError("Expected an mmh3_long_video_chunk_plan_v1 plan")
    fps_value = plan.get("fps")
    if not isinstance(fps_value, dict):
        raise MMH3ResourceError("Chunk plan FPS is missing")
    try:
        fps = Fraction(int(fps_value["numerator"]), int(fps_value["denominator"]))
        sample_rate = int(plan["audio_sample_rate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise MMH3ResourceError("Chunk plan FPS/audio rate is invalid") from exc
    if fps <= 0 or sample_rate <= 0:
        raise MMH3ResourceError("Chunk plan FPS/audio rate must be positive")
    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise MMH3ResourceError("Chunk plan has no jobs")
    selected = None
    wanted_id = str(job_id or "").strip()
    if wanted_id:
        selected = next((item for item in chunks if isinstance(item, dict) and item.get("job_id") == wanted_id), None)
        if selected is None:
            raise MMH3ResourceError(f"Chunk job_id {wanted_id!r} does not exist")
    elif 0 <= int(chunk_index) < len(chunks):
        selected = chunks[int(chunk_index)]
    else:
        raise MMH3ResourceError(f"chunk_index={chunk_index} is outside 0..{len(chunks) - 1}")
    if not isinstance(selected, dict):
        raise MMH3ResourceError("Selected chunk job is invalid")
    required = {
        "job_id", "index", "read_start_frame", "read_end_frame", "write_start_frame",
        "write_end_frame", "audio_read_start", "audio_read_end", "audio_write_start",
        "audio_write_end", "output_name",
    }
    if not required.issubset(selected):
        raise MMH3ResourceError("Selected chunk job is missing required boundaries")
    read_start = int(selected["read_start_frame"])
    read_end = int(selected["read_end_frame"])
    if read_start < 0 or read_end <= read_start:
        raise MMH3ResourceError("Selected chunk read boundaries are invalid")
    return ChunkExecutionSelection(
        dict(selected),
        fps,
        sample_rate,
        float(Fraction(read_start, 1) / fps),
        float(Fraction(read_end - read_start, 1) / fps),
    )


def _frames(seconds: float, fps: Fraction, *, minimum: int = 0) -> int:
    duration = float(seconds)
    if not math.isfinite(duration) or duration < 0:
        raise MMH3ResourceError("Timeline duration/context must be finite and non-negative")
    value = int(Fraction(str(duration)) * fps + Fraction(1, 2))
    return max(minimum, value)


def _audio_boundary(frame: int, fps: Fraction, sample_rate: int) -> int:
    return int(Fraction(frame * sample_rate, 1) / fps + Fraction(1, 2))


def plan_long_video_chunks(
    *,
    source_id: str,
    total_frames: int,
    fps: float | Fraction,
    chunk_duration_seconds: float = 10.0,
    overlap_seconds: float = 1.0,
    boundary_mode: str = "fixed",
    scene_cuts: Sequence[int] = (),
    scene_search_seconds: float = 1.5,
    min_tail_seconds: float = 2.0,
    min_tail_policy: str = "merge_previous",
    audio_sample_rate: int = 32000,
) -> ChunkPlan:
    rate = fps if isinstance(fps, Fraction) else Fraction(str(float(fps))).limit_denominator(1001)
    if total_frames < 1 or rate <= 0 or audio_sample_rate < 1:
        raise MMH3ResourceError("Long-video source geometry/FPS/audio rate must be positive")
    if boundary_mode not in {"fixed", "scene_aware"}:
        raise MMH3ResourceError("boundary_mode must be fixed or scene_aware")
    if min_tail_policy not in {"merge_previous", "keep", "drop"}:
        raise MMH3ResourceError("min_tail_policy must be merge_previous, keep or drop")
    chunk_frames = _frames(chunk_duration_seconds, rate, minimum=1)
    overlap_frames = _frames(overlap_seconds, rate)
    search_frames = _frames(scene_search_seconds, rate)
    min_tail_frames = _frames(min_tail_seconds, rate, minimum=1)
    if overlap_frames * 2 >= chunk_frames:
        raise MMH3ResourceError("Chunk overlap/context must be shorter than half a chunk")

    cuts = sorted({int(value) for value in scene_cuts if 0 < int(value) < total_frames})
    boundaries = [0]
    while boundaries[-1] + chunk_frames < total_frames:
        nominal = boundaries[-1] + chunk_frames
        boundary = nominal
        if boundary_mode == "scene_aware":
            candidates = [cut for cut in cuts if abs(cut - nominal) <= search_frames and cut > boundaries[-1]]
            if candidates:
                boundary = min(candidates, key=lambda cut: (abs(cut - nominal), cut))
        if boundary <= boundaries[-1]:
            raise MMH3ResourceError("Chunk planner did not make forward progress")
        boundaries.append(boundary)
    boundaries.append(total_frames)

    excluded_tail = None
    tail_frames = boundaries[-1] - boundaries[-2]
    if len(boundaries) > 2 and tail_frames < min_tail_frames:
        if min_tail_policy == "merge_previous":
            boundaries.pop(-2)
        elif min_tail_policy == "drop":
            excluded_tail = {"start_frame": boundaries[-2], "end_frame": total_frames, "frames": tail_frames}
            boundaries.pop()

    chunks: list[dict[str, Any]] = []
    for index, (write_start, write_end) in enumerate(zip(boundaries, boundaries[1:])):
        read_start = max(0, write_start - overlap_frames)
        read_end = min(total_frames, write_end + overlap_frames)
        material = {
            "source_id": source_id,
            "index": index,
            "read": [read_start, read_end],
            "write": [write_start, write_end],
            "fps": [rate.numerator, rate.denominator],
        }
        chunks.append(
            {
                "job_id": _job_id(material),
                "index": index,
                "status": "planned",
                "read_start_frame": read_start,
                "read_end_frame": read_end,
                "write_start_frame": write_start,
                "write_end_frame": write_end,
                "context_before_frames": write_start - read_start,
                "context_after_frames": read_end - write_end,
                "audio_read_start": _audio_boundary(read_start, rate, audio_sample_rate),
                "audio_read_end": _audio_boundary(read_end, rate, audio_sample_rate),
                "audio_write_start": _audio_boundary(write_start, rate, audio_sample_rate),
                "audio_write_end": _audio_boundary(write_end, rate, audio_sample_rate),
                "output_name": f"chunk_{index:04d}_{write_start:08d}_{write_end:08d}",
            }
        )

    settings = {
        "chunk_duration_seconds": float(chunk_duration_seconds),
        "overlap_seconds": float(overlap_seconds),
        "boundary_mode": boundary_mode,
        "scene_search_seconds": float(scene_search_seconds),
        "min_tail_seconds": float(min_tail_seconds),
        "min_tail_policy": min_tail_policy,
    }
    return ChunkPlan(str(source_id), int(total_frames), rate, int(audio_sample_rate), tuple(chunks), excluded_tail, settings)


__all__ = [
    "ChunkExecutionSelection",
    "ChunkPlan",
    "VIDEO_SUFFIXES",
    "H3_AUDIO_TIMELINE_PLANNERS",
    "plan_batch_inputs",
    "plan_long_video_chunks",
    "resolve_chunk_execution",
    "plan_h3_audio_timeline_chunks",
    "plan_h3_audio_interactive_sequence",
    "append_h3_audio_interactive_try",
    "finalize_h3_audio_interactive_plan",
]


def _next_h3_legal_frames(value: int, *, minimum: int = 5, maximum: int | None = None) -> int:
    """Return the smallest H3 17k+5 frame count >= value."""
    wanted = max(int(minimum), int(value))
    if wanted <= 5:
        result = 5
    else:
        result = 5 + 17 * ((wanted - 5 + 16) // 17)
    if maximum is not None and result > int(maximum):
        raise MMH3ResourceError(f"Requested H3 generation window requires {result} frames, above maximum {maximum}")
    return result



def _build_h3_audio_try_chunk(
    *,
    source_id: str,
    index: int,
    total_frames: int,
    total_audio_samples: int,
    audio_sample_rate: int,
    write_start_frame: int,
    shot_duration_seconds: float,
    context_seconds: float,
    min_generation_frames: int,
    max_generation_frames: int,
) -> dict[str, Any]:
    """Build one ownership shot inside a legal H3 generation window.

    ``shot_duration_seconds`` describes the delivered/owned timeline, not the
    generation window. H3 context and the 17n+5 constraint are absorbed into a
    larger read window automatically.
    """
    rate = int(audio_sample_rate)
    frame_rate = Fraction(24, 1)
    start = int(write_start_frame)
    if start < 0 or start >= int(total_frames):
        raise MMH3ResourceError("Interactive H3 shot cursor is outside the master timeline")
    duration = float(shot_duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise MMH3ResourceError("Next H3 shot duration must be positive")
    requested_owned_frames = _frames(duration, frame_rate, minimum=1)
    write_end = min(int(total_frames), start + requested_owned_frames)
    owned_frames = write_end - start
    requested_context = _frames(float(context_seconds), frame_rate)
    if requested_context < 0:
        raise MMH3ResourceError("H3 shot context cannot be negative")
    if int(min_generation_frames) < 5 or int(max_generation_frames) < int(min_generation_frames):
        raise MMH3ResourceError("H3 generation frame bounds are invalid")
    minimum = _next_h3_legal_frames(int(min_generation_frames), minimum=5)
    maximum = 5 + 17 * max(0, (int(max_generation_frames) - 5) // 17)
    if minimum > maximum:
        raise MMH3ResourceError("H3 generation frame bounds are invalid")
    if owned_frames > maximum:
        raise MMH3ResourceError(
            f"Requested shot owns {owned_frames} frames, above the maximum legal H3 window {maximum}"
        )

    # Context is a preference, not ownership. Near the maximum H3 duration we
    # shrink it instead of rejecting an otherwise legal requested shot. Preserve
    # leading context first because it is the only context that can carry recent
    # timeline information into the owned region.
    context_before = min(requested_context, start)
    context_after_requested = requested_context
    overflow = max(0, context_before + owned_frames + context_after_requested - maximum)
    context_after_requested -= min(context_after_requested, overflow)
    overflow = max(0, context_before + owned_frames + context_after_requested - maximum)
    context_before -= min(context_before, overflow)
    read_start = start - context_before
    required_generation = context_before + owned_frames + context_after_requested
    generation_frames = _next_h3_legal_frames(
        max(minimum, required_generation), minimum=minimum, maximum=maximum
    )
    read_end = read_start + generation_frames
    audio_read_start = _audio_boundary(read_start, frame_rate, rate)
    audio_read_end = _audio_boundary(read_end, frame_rate, rate)
    audio_write_start = _audio_boundary(start, frame_rate, rate)
    audio_write_end = min(int(total_audio_samples), _audio_boundary(write_end, frame_rate, rate))
    material = {
        "source_id": str(source_id),
        "index": int(index),
        "read": [read_start, read_end],
        "write": [start, write_end],
        "fps": [24, 1],
        "kind": "h3_audio_interactive",
    }
    return {
        "job_id": _job_id(material),
        "index": int(index),
        "status": "planned",
        "read_start_frame": read_start,
        "read_end_frame": read_end,
        "write_start_frame": start,
        "write_end_frame": write_end,
        "context_before_frames": context_before,
        "context_after_frames": max(0, read_end - write_end),
        "requested_context_frames": requested_context,
        "audio_read_start": audio_read_start,
        "audio_read_end": audio_read_end,
        "audio_write_start": audio_write_start,
        "audio_write_end": audio_write_end,
        "source_audio_samples": int(total_audio_samples),
        "conditioning_pad_samples": max(0, audio_read_end - int(total_audio_samples)),
        "generation_frames": generation_frames,
        "requested_shot_duration_seconds": duration,
        "actual_shot_duration_seconds": owned_frames / 24.0,
        "output_name": f"chunk_{int(index):04d}_{start:08d}_{write_end:08d}",
    }


def plan_h3_audio_interactive_sequence(
    *,
    source_id: str,
    total_audio_samples: int,
    audio_sample_rate: int = 32000,
    first_shot_duration_seconds: float = 5.0,
    context_seconds: float = 0.5,
    min_generation_frames: int = 124,
    max_generation_frames: int = 362,
) -> ChunkPlan:
    """Start an F18 human-in-the-loop sequence with one configurable shot.

    Later shots are appended only after the previous shot is accepted. This keeps
    candidate choice and the master-audio cursor coupled without pre-planning the
    whole song at one fixed duration.
    """
    samples = int(total_audio_samples)
    rate = int(audio_sample_rate)
    if samples < 1 or rate != 32000:
        raise MMH3ResourceError("Interactive H3 audio timeline requires non-empty 32000 Hz master audio")
    total_frames = (samples * 24 + rate - 1) // rate
    first = _build_h3_audio_try_chunk(
        source_id=str(source_id), index=0, total_frames=total_frames,
        total_audio_samples=samples, audio_sample_rate=rate, write_start_frame=0,
        shot_duration_seconds=float(first_shot_duration_seconds), context_seconds=float(context_seconds),
        min_generation_frames=int(min_generation_frames), max_generation_frames=int(max_generation_frames),
    )
    finished = int(first["write_end_frame"]) >= int(total_frames)
    settings = {
        "planner": "h3_audio_interactive",
        "master_audio_samples": samples,
        "master_audio_duration_seconds": samples / rate,
        "default_context_seconds": float(context_seconds),
        "min_generation_frames": int(min_generation_frames),
        "max_generation_frames": int(max_generation_frames),
        "sequence_cursor_frame": int(first["write_end_frame"]),
        "sequence_cursor_audio_sample": int(first["audio_write_end"]),
        "finalized": bool(finished),
        "video_tail_seconds": total_frames / 24.0 - samples / rate,
    }
    return ChunkPlan(str(source_id), int(total_frames), Fraction(24, 1), rate, (first,), None, settings)


def append_h3_audio_interactive_try(
    plan: Mapping[str, Any],
    *,
    shot_duration_seconds: float,
    context_seconds: float | None = None,
) -> dict[str, Any]:
    """Append one next shot to an interactive F18 plan.

    The next shot always starts at the exact accepted timeline cursor. The caller
    owns lifecycle validation (for example, only appending after review/accept).
    """
    if not isinstance(plan, Mapping) or plan.get("contract") != "mmh3_long_video_chunk_plan_v1":
        raise MMH3ResourceError("Interactive append requires an mmh3_long_video_chunk_plan_v1 plan")
    settings = dict(plan.get("settings") or {})
    if settings.get("planner") != "h3_audio_interactive":
        raise MMH3ResourceError("Interactive append requires an h3_audio_interactive plan")
    if bool(settings.get("finalized")):
        raise MMH3ResourceError("Interactive H3 sequence is already finalized")
    chunks = [dict(item) for item in plan.get("chunks") or []]
    if not chunks:
        raise MMH3ResourceError("Interactive H3 sequence has no initial shot")
    total_frames = int(plan.get("total_frames") or 0)
    cursor = int(chunks[-1].get("write_end_frame") or 0)
    if cursor >= total_frames:
        raise MMH3ResourceError("Interactive H3 sequence already reaches the end of the master audio")
    ctx = float(settings.get("default_context_seconds") or 0.0) if context_seconds is None else float(context_seconds)
    job = _build_h3_audio_try_chunk(
        source_id=str(plan.get("source_id") or ""), index=len(chunks), total_frames=total_frames,
        total_audio_samples=int(settings.get("master_audio_samples") or 0),
        audio_sample_rate=int(plan.get("audio_sample_rate") or 0), write_start_frame=cursor,
        shot_duration_seconds=float(shot_duration_seconds), context_seconds=ctx,
        min_generation_frames=int(settings.get("min_generation_frames") or 124),
        max_generation_frames=int(settings.get("max_generation_frames") or 362),
    )
    chunks.append(job)
    settings["sequence_cursor_frame"] = int(job["write_end_frame"])
    settings["sequence_cursor_audio_sample"] = int(job["audio_write_end"])
    settings["last_requested_shot_duration_seconds"] = float(shot_duration_seconds)
    settings["last_context_seconds"] = ctx
    if int(job["write_end_frame"]) >= total_frames:
        settings["finalized"] = True
    updated = dict(plan)
    updated["chunks"] = chunks
    updated["settings"] = settings
    updated["excluded_tail"] = None
    return updated


def finalize_h3_audio_interactive_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze an interactive sequence at its current cursor for assembly.

    If the song has not been fully covered, the untouched tail is explicitly marked
    excluded rather than silently omitted.
    """
    if not isinstance(plan, Mapping) or plan.get("contract") != "mmh3_long_video_chunk_plan_v1":
        raise MMH3ResourceError("Interactive finalize requires an mmh3_long_video_chunk_plan_v1 plan")
    settings = dict(plan.get("settings") or {})
    if settings.get("planner") != "h3_audio_interactive":
        raise MMH3ResourceError("Interactive finalize requires an h3_audio_interactive plan")
    chunks = [dict(item) for item in plan.get("chunks") or []]
    if not chunks:
        raise MMH3ResourceError("Interactive H3 sequence has no shots")
    cursor = int(chunks[-1]["write_end_frame"])
    total_frames = int(plan["total_frames"])
    settings["sequence_cursor_frame"] = cursor
    settings["sequence_cursor_audio_sample"] = int(chunks[-1]["audio_write_end"])
    settings["finalized"] = True
    updated = dict(plan)
    updated["chunks"] = chunks
    updated["settings"] = settings
    updated["excluded_tail"] = (
        {"start_frame": cursor, "end_frame": total_frames, "frames": total_frames - cursor}
        if cursor < total_frames else None
    )
    return updated

def plan_h3_audio_timeline_chunks(
    *,
    source_id: str,
    total_audio_samples: int,
    audio_sample_rate: int = 32000,
    fps: int = 24,
    generation_duration_seconds: float = 8.0,
    context_seconds: float = 0.5,
    min_generation_frames: int = 124,
    max_generation_frames: int = 362,
) -> ChunkPlan:
    """Plan an audio-master H3 timeline with legal generation windows.

    The write timeline follows the immutable source audio. Each generation read window
    is a legal H3 17k+5 frame count. The last window may extend beyond the master
    timeline; runtime audio chunking pads only that conditioning tail with silence,
    while write ownership and final assembly remain bounded to the original master.
    """
    samples = int(total_audio_samples)
    rate = int(audio_sample_rate)
    frame_rate = int(fps)
    if samples < 1 or rate != 32000 or float(fps) != 24:
        raise MMH3ResourceError("H3 audio timeline requires non-empty 32000 Hz audio at 24 FPS")
    if not math.isfinite(float(generation_duration_seconds)) or float(generation_duration_seconds) <= 0:
        raise MMH3ResourceError("H3 generation duration must be finite and positive")
    if int(min_generation_frames) < 5 or int(max_generation_frames) < int(min_generation_frames):
        raise MMH3ResourceError("H3 generation frame bounds are invalid")
    min_frames = _next_h3_legal_frames(int(min_generation_frames), minimum=5)
    max_frames = 5 + 17 * max(0, (int(max_generation_frames) - 5) // 17)
    if min_frames > max_frames:
        raise MMH3ResourceError("H3 generation frame bounds are invalid")
    requested = _frames(generation_duration_seconds, Fraction(frame_rate, 1), minimum=min_frames)
    generation_frames = _next_h3_legal_frames(requested, minimum=min_frames, maximum=max_frames)
    context_frames = _frames(context_seconds, Fraction(frame_rate, 1))
    if context_frames < 0 or context_frames * 2 >= generation_frames - 5:
        raise MMH3ResourceError("H3 audio timeline context must be shorter than half a generation window")

    # Ceil so video never ends before the master audio. At most one video frame of
    # tail exceeds the exact audio duration; final audio remains the immutable master.
    total_frames = (samples * frame_rate + rate - 1) // rate
    chunks: list[dict[str, Any]] = []
    write_start = 0
    index = 0
    while write_start < total_frames:
        if index == 0:
            read_start = 0
            write_capacity = generation_frames - context_frames
        else:
            read_start = max(0, write_start - context_frames)
            write_capacity = generation_frames - 2 * context_frames
        if write_capacity < 1:
            raise MMH3ResourceError("H3 audio timeline has no writable frames after context")
        write_end = min(total_frames, write_start + write_capacity)
        read_end = read_start + generation_frames
        material = {
            "source_id": str(source_id),
            "index": index,
            "read": [read_start, read_end],
            "write": [write_start, write_end],
            "fps": [frame_rate, 1],
            "kind": "h3_audio_timeline",
        }
        audio_read_start = _audio_boundary(read_start, Fraction(frame_rate, 1), rate)
        audio_read_end = _audio_boundary(read_end, Fraction(frame_rate, 1), rate)
        audio_write_start = _audio_boundary(write_start, Fraction(frame_rate, 1), rate)
        audio_write_end = min(samples, _audio_boundary(write_end, Fraction(frame_rate, 1), rate))
        chunks.append({
            "job_id": _job_id(material),
            "index": index,
            "status": "planned",
            "read_start_frame": read_start,
            "read_end_frame": read_end,
            "write_start_frame": write_start,
            "write_end_frame": write_end,
            "context_before_frames": write_start - read_start,
            "context_after_frames": max(0, read_end - write_end),
            "audio_read_start": audio_read_start,
            "audio_read_end": audio_read_end,
            "audio_write_start": audio_write_start,
            "audio_write_end": audio_write_end,
            "source_audio_samples": samples,
            "conditioning_pad_samples": max(0, audio_read_end - samples),
            "generation_frames": generation_frames,
            "output_name": f"chunk_{index:04d}_{write_start:08d}_{write_end:08d}",
        })
        write_start = write_end
        index += 1

    settings = {
        "planner": "h3_audio_timeline",
        "generation_duration_seconds": float(generation_duration_seconds),
        "generation_frames": generation_frames,
        "context_seconds": float(context_seconds),
        "context_frames": context_frames,
        "master_audio_samples": samples,
        "master_audio_duration_seconds": samples / rate,
        "video_tail_seconds": total_frames / frame_rate - samples / rate,
    }
    return ChunkPlan(str(source_id), int(total_frames), Fraction(frame_rate, 1), rate, tuple(chunks), None, settings)
