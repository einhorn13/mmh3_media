from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .archive import load_archive
from .core import MMH3Media
from .errors import MMH3Error, MMH3ResourceError
from .stitch import inspect_stitch_packet


BATCH_STITCH_CONTRACT = "mmh3_batch_stitch_preflight_v1"


@dataclass(frozen=True)
class BatchStitchPreparation:
    packets: tuple[MMH3Media, ...]
    report: dict[str, Any]

    def summary(self) -> str:
        counts = self.report["counts"]
        return (
            f"READY · batch stitch · accepted={counts['accepted']} · "
            f"skipped={counts['skipped']}"
        )


def _resolve_input_path(raw_path: str, search_roots: Sequence[str | Path]) -> Path:
    source = Path(raw_path).expanduser()
    if source.is_absolute():
        return source.resolve()
    candidates = [(Path(root).expanduser().resolve() / source).resolve() for root in search_roots]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) > 1:
        raise MMH3ResourceError(
            f"Relative batch path is ambiguous across search roots: {raw_path}"
        )
    if existing:
        return existing[0]
    return candidates[0] if candidates else source.resolve()


def _validate_modes(video_mode: str, audio_mode: str, overlap_frames: int) -> None:
    if (video_mode, audio_mode) not in {("cut", "cut"), ("crossfade", "half_cosine")}:
        raise MMH3ResourceError(
            "Batch stitch modes must be cut+cut or crossfade+half_cosine"
        )
    if video_mode == "cut" and int(overlap_frames) != 0:
        raise MMH3ResourceError("Batch cut mode requires overlap_frames=0")
    if video_mode == "crossfade" and int(overlap_frames) < 1:
        raise MMH3ResourceError("Batch crossfade mode requires overlap_frames>=1")


def inspect_batch_stitch_plan(
    plan: dict[str, Any],
    *,
    search_roots: Sequence[str | Path] = (),
    error_policy: str = "stop_on_error",
    video_mode: str = "cut",
    audio_mode: str = "cut",
    overlap_frames: int = 0,
) -> BatchStitchPreparation:
    """Resolve and validate arbitrary ordered MMH3 inputs without decoding media."""
    if error_policy not in {"stop_on_error", "skip_invalid"}:
        raise MMH3ResourceError("Batch stitch error_policy must be stop_on_error or skip_invalid")
    _validate_modes(video_mode, audio_mode, int(overlap_frames))
    if not isinstance(plan, dict) or plan.get("contract") != "mmh3_batch_input_plan_v1":
        raise MMH3ResourceError("Batch stitch requires mmh3_batch_input_plan_v1")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise MMH3ResourceError("Batch stitch plan has no jobs")

    packets: list[MMH3Media] = []
    items: list[dict[str, Any]] = []
    expected_dimensions: tuple[int, int] | None = None
    for position, job in enumerate(jobs):
        item = {
            "position": position,
            "index": job.get("index") if isinstance(job, dict) else None,
            "job_id": job.get("job_id") if isinstance(job, dict) else None,
            "input_path": job.get("path") if isinstance(job, dict) else None,
            "kind": job.get("kind") if isinstance(job, dict) else None,
            "status": "rejected",
            "reasons": [],
        }
        if not isinstance(job, dict) or job.get("index") != position:
            item["reasons"].append("job index is missing or does not match ordered position")
            items.append(item)
            continue
        raw_path = job.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            item["reasons"].append("job path is missing")
            items.append(item)
            continue
        if job.get("kind") == "video":
            item["status"] = "requires_import"
            item["reasons"].append(
                "raw video must first pass MMH3BatchNormalizeImport; direct batch stitch does not hide RGB decode or AV conform"
            )
            items.append(item)
            continue
        if job.get("kind") != "mmh3":
            item["reasons"].append(f"unsupported job kind {job.get('kind')!r}")
            items.append(item)
            continue
        try:
            resolved = _resolve_input_path(raw_path, search_roots)
            item["resolved_path"] = str(resolved)
            packet = load_archive(resolved, verify="on_access")
            fact, reasons = inspect_stitch_packet(
                packet,
                index=position,
                expected_dimensions=expected_dimensions,
            )
            item["facts"] = fact
            item["reasons"].extend(reasons)
            if not reasons:
                dimensions = fact.get("dimensions")
                if expected_dimensions is None and isinstance(dimensions, list):
                    expected_dimensions = (int(dimensions[0]), int(dimensions[1]))
                item["status"] = "accepted"
                item["accepted_index"] = len(packets)
                packets.append(packet)
        except (MMH3Error, OSError) as exc:
            item["reasons"].append(str(exc))
        items.append(item)

    rejected = [item for item in items if item["status"] != "accepted"]
    report = {
        "contract": BATCH_STITCH_CONTRACT,
        "ready": len(packets) >= 2 and (not rejected or error_policy == "skip_invalid"),
        "source_contract": plan["contract"],
        "order": plan.get("order"),
        "error_policy": error_policy,
        "settings": {
            "assembly_mode": "streaming",
            "video_mode": video_mode,
            "audio_mode": audio_mode,
            "overlap_frames": int(overlap_frames),
        },
        "counts": {
            "planned": len(items),
            "accepted": len(packets),
            "skipped": len(rejected),
        },
        "items": items,
    }
    return BatchStitchPreparation(tuple(packets), report)


def prepare_batch_stitch(
    plan: dict[str, Any],
    *,
    search_roots: Sequence[str | Path] = (),
    error_policy: str = "stop_on_error",
    video_mode: str = "cut",
    audio_mode: str = "cut",
    overlap_frames: int = 0,
) -> BatchStitchPreparation:
    """Resolve/validate arbitrary ordered MMH3 inputs and enforce execution readiness."""
    prepared = inspect_batch_stitch_plan(
        plan,
        search_roots=search_roots,
        error_policy=error_policy,
        video_mode=video_mode,
        audio_mode=audio_mode,
        overlap_frames=overlap_frames,
    )
    rejected = [item for item in prepared.report["items"] if item["status"] != "accepted"]
    if rejected and error_policy == "stop_on_error":
        details = "; ".join(
            f"item {item['position']}: {', '.join(item['reasons'])}" for item in rejected
        )
        raise MMH3ResourceError("Batch stitch stopped by per-item validation: " + details)
    if len(prepared.packets) < 2:
        raise MMH3ResourceError(
            f"Batch stitch requires at least two accepted MMH3 segments; got {len(prepared.packets)}"
        )
    return prepared

