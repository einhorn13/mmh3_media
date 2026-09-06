from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import MMH3ResourceError


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


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
    value = int(Fraction(str(float(seconds))) * fps + Fraction(1, 2))
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
    "plan_batch_inputs",
    "plan_long_video_chunks",
    "resolve_chunk_execution",
]
