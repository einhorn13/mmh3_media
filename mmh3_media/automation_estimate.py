from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping

from .errors import MMH3ResourceError
from .util import deep_copy_json


PREQUEUE_ESTIMATE_CONTRACT = "mmh3_prequeue_estimate_v1"


def _known(value: Any, *, unit: str = "") -> dict[str, Any]:
    result = {"status": "known", "value": value}
    if unit:
        result["unit"] = unit
    return result


def _unknown(reason: str, *, unit: str = "") -> dict[str, Any]:
    result = {"status": "unknown", "value": None, "reason": str(reason)}
    if unit:
        result["unit"] = unit
    return result


def _fps(plan: Mapping[str, Any]) -> Fraction:
    value = plan.get("fps")
    if not isinstance(value, Mapping):
        raise MMH3ResourceError("Long-video estimate requires plan FPS")
    try:
        rate = Fraction(int(value["numerator"]), int(value["denominator"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise MMH3ResourceError("Long-video estimate plan FPS is invalid") from exc
    if rate <= 0:
        raise MMH3ResourceError("Long-video estimate plan FPS must be positive")
    return rate


def _duration(frames: int, fps: Fraction) -> float:
    return float(Fraction(int(frames), 1) / fps)


def _metric_bundle(frames: int, pcm_samples: int, fps: Fraction) -> dict[str, Any]:
    return {
        "frames": _known(int(frames), unit="frames"),
        "duration_seconds": _known(_duration(int(frames), fps), unit="seconds"),
        "pcm_samples": _known(int(pcm_samples), unit="samples"),
    }


def _unknown_metrics(reason: str) -> dict[str, Any]:
    return {
        "frames": _unknown(reason, unit="frames"),
        "duration_seconds": _unknown(reason, unit="seconds"),
        "pcm_samples": _unknown(reason, unit="samples"),
    }


def _batch_estimate(
    plan: Mapping[str, Any],
    *,
    batch_preflight: Mapping[str, Any] | None,
    effective_settings: Mapping[str, Any],
    error_policy: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise MMH3ResourceError("Batch estimate requires planned jobs")
    counts: dict[str, Any] = {"jobs": len(jobs), "chunks": 0}
    no_overlap = {
        "frames": _known(0, unit="frames"),
        "pcm_samples": _known(0, unit="samples"),
        "extra_work_factor": _known(1.0, unit="ratio"),
        "note": "batch inputs are read once; stitch crossfade changes ownership/output length but is not read-context overlap",
    }
    if not isinstance(batch_preflight, Mapping):
        reason = "batch media metadata was not inspected; provide mmh3_batch_stitch_preflight_v1 for exact totals"
        counts.update({"accepted": _unknown(reason), "skipped": _unknown(reason)})
        return counts, {"input": _unknown_metrics(reason), "expected_output": _unknown_metrics(reason)}, no_overlap
    if batch_preflight.get("contract") != "mmh3_batch_stitch_preflight_v1":
        raise MMH3ResourceError("Batch estimate received an unsupported preflight contract")

    items = batch_preflight.get("items")
    if not isinstance(items, list):
        raise MMH3ResourceError("Batch estimate preflight items are malformed")
    accepted = [item for item in items if isinstance(item, Mapping) and item.get("status") == "accepted"]
    skipped = len(items) - len(accepted)
    counts.update({"accepted": len(accepted), "skipped": skipped})
    all_facts = [item.get("facts") for item in accepted]
    metrics_known = bool(accepted) and all(
        isinstance(fact, Mapping)
        and isinstance(fact.get("frame_count"), int)
        and isinstance(fact.get("audio_samples"), int)
        and isinstance(fact.get("fps"), (int, float))
        and float(fact.get("fps")) > 0
        for fact in all_facts
    )
    if not metrics_known:
        reason = "accepted batch items do not all prove frame_count/audio_samples/FPS metadata"
        input_metrics = _unknown_metrics(reason)
    else:
        frames = sum(int(fact["frame_count"]) for fact in all_facts)
        samples = sum(int(fact["audio_samples"]) for fact in all_facts)
        fps = Fraction(str(float(all_facts[0]["fps"]))).limit_denominator(1001)
        input_metrics = _metric_bundle(frames, samples, fps)

    policy_blocks_output = error_policy == "stop_on_error" and skipped > 0
    if policy_blocks_output:
        output_metrics = _unknown_metrics("selected stop_on_error policy blocks output because one or more batch items were rejected")
    elif len(accepted) < 2:
        output_metrics = _unknown_metrics("fewer than two accepted batch items cannot produce a stitched output")
    elif not metrics_known:
        output_metrics = _unknown_metrics("batch output cannot be derived without accepted item frame/audio metadata")
    else:
        video_mode = str(effective_settings.get("video_mode") or batch_preflight.get("settings", {}).get("video_mode") or "")
        overlap = int(effective_settings.get("overlap_frames", batch_preflight.get("settings", {}).get("overlap_frames", 0)) or 0)
        total_input_frames = sum(int(fact["frame_count"]) for fact in all_facts)
        output_frames = total_input_frames - (overlap * (len(accepted) - 1) if video_mode == "crossfade" else 0)
        fps = Fraction(str(float(all_facts[0]["fps"]))).limit_denominator(1001)
        sample_rate = int(all_facts[0].get("audio_sample_rate") or 0)
        if sample_rate <= 0:
            output_metrics = _unknown_metrics("batch output PCM cannot be derived without a positive audio sample rate")
            output_metrics["frames"] = _known(output_frames, unit="frames")
            output_metrics["duration_seconds"] = _known(_duration(output_frames, fps), unit="seconds")
        else:
            output_samples = int(Fraction(output_frames * sample_rate, 1) / fps + Fraction(1, 2))
            output_metrics = _metric_bundle(output_frames, output_samples, fps)
    return counts, {"input": input_metrics, "expected_output": output_metrics}, no_overlap


def _long_video_estimate(plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise MMH3ResourceError("Long-video estimate requires chunk jobs")
    fps = _fps(plan)
    total_frames = int(plan.get("total_frames") or 0)
    sample_rate = int(plan.get("audio_sample_rate") or 0)
    if total_frames <= 0 or sample_rate <= 0:
        raise MMH3ResourceError("Long-video estimate requires positive source frames and audio sample rate")
    input_samples = int(Fraction(total_frames * sample_rate, 1) / fps + Fraction(1, 2))
    read_frames = sum(int(item["read_end_frame"]) - int(item["read_start_frame"]) for item in chunks)
    write_frames = sum(int(item["write_end_frame"]) - int(item["write_start_frame"]) for item in chunks)
    read_samples = sum(int(item["audio_read_end"]) - int(item["audio_read_start"]) for item in chunks)
    write_samples = sum(int(item["audio_write_end"]) - int(item["audio_write_start"]) for item in chunks)
    overhead_frames = read_frames - write_frames
    overhead_samples = read_samples - write_samples
    factor = float(Fraction(read_frames, write_frames)) if write_frames > 0 else None
    counts = {"jobs": len(chunks), "chunks": len(chunks)}
    metrics = {
        "input": _metric_bundle(total_frames, input_samples, fps),
        "expected_output": _metric_bundle(write_frames, write_samples, fps),
        "read_volume": _metric_bundle(read_frames, read_samples, fps),
    }
    overlap = {
        "frames": _known(overhead_frames, unit="frames"),
        "pcm_samples": _known(overhead_samples, unit="samples"),
        "extra_work_factor": _known(factor, unit="ratio") if factor is not None else _unknown("no owned output frames", unit="ratio"),
        "extra_work_percent": _known((factor - 1.0) * 100.0, unit="percent") if factor is not None else _unknown("no owned output frames", unit="percent"),
        "note": "read-frame work proxy only; actual runtime/VRAM requires a measured runtime profile",
    }
    return counts, metrics, overlap


def build_prequeue_estimate(
    plan: Mapping[str, Any],
    *,
    operation: str,
    effective_settings: Mapping[str, Any] | None = None,
    output_prefix: str = "",
    error_policy: str = "",
    batch_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise MMH3ResourceError("Pre-queue estimate requires a plan object")
    settings = deep_copy_json(dict(effective_settings or {}))
    contract = plan.get("contract")
    policy = str(error_policy or "").strip()
    if contract == "mmh3_long_video_chunk_plan_v1":
        counts, metrics, overlap = _long_video_estimate(plan)
        output_names = [str(item.get("output_name") or "") for item in plan["chunks"]]
    elif contract == "mmh3_batch_input_plan_v1":
        counts, metrics, overlap = _batch_estimate(
            plan,
            batch_preflight=batch_preflight,
            effective_settings=settings,
            error_policy=policy,
        )
        output_names = [str(item.get("output_name") or "") for item in plan.get("jobs", []) if isinstance(item, Mapping) and item.get("output_name")]
    else:
        raise MMH3ResourceError("Unsupported plan contract for pre-queue estimate")

    prefix = str(output_prefix or "").strip()
    result = {
        "contract": PREQUEUE_ESTIMATE_CONTRACT,
        "source_plan_contract": str(contract),
        "operation": str(operation or "unknown"),
        "counts": counts,
        "metrics": metrics,
        "read_overlap": overlap,
        "effective_settings": settings,
        "output_naming": {
            "prefix": _known(prefix) if prefix else _unknown("output prefix was not selected"),
            "planned_names": output_names,
        },
        "error_policy": _known(policy) if policy else _unknown("error policy was not selected"),
        "runtime_profile": {
            "time_estimate": _unknown("no measured runtime profile was supplied", unit="seconds"),
            "vram_estimate": _unknown("no measured runtime profile was supplied", unit="bytes"),
        },
    }
    return result


def _fmt_metric(value: Mapping[str, Any], suffix: str) -> str:
    if value.get("status") != "known":
        return "unknown"
    raw = value.get("value")
    if isinstance(raw, float):
        return f"{raw:.3f}{suffix}"
    return f"{raw}{suffix}"


def prequeue_summary(report: Mapping[str, Any]) -> str:
    if report.get("contract") != PREQUEUE_ESTIMATE_CONTRACT:
        raise MMH3ResourceError("Unsupported pre-queue estimate contract")
    metrics = report["metrics"]
    input_metrics = metrics["input"]
    output = metrics["expected_output"]
    overlap = report["read_overlap"]
    factor = overlap["extra_work_factor"]
    jobs = report["counts"].get("jobs", "unknown")
    prefix = report["output_naming"]["prefix"]
    policy = report["error_policy"]
    return (
        f"READY · pre-queue · jobs={jobs} · operation={report['operation']} · "
        f"input={_fmt_metric(input_metrics['frames'], 'f')}/{_fmt_metric(input_metrics['duration_seconds'], 's')}/"
        f"{_fmt_metric(input_metrics['pcm_samples'], ' samples')} · "
        f"output={_fmt_metric(output['frames'], 'f')}/{_fmt_metric(output['duration_seconds'], 's')}/"
        f"{_fmt_metric(output['pcm_samples'], ' samples')} · overlap={_fmt_metric(overlap['frames'], 'f')} · "
        f"work={_fmt_metric(factor, 'x')} · prefix={prefix.get('value') if prefix.get('status') == 'known' else 'unknown'} · "
        f"error_policy={policy.get('value') if policy.get('status') == 'known' else 'unknown'}"
    )


__all__ = ["PREQUEUE_ESTIMATE_CONTRACT", "build_prequeue_estimate", "prequeue_summary"]
