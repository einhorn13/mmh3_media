from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .errors import MMH3ResourceError


PROOF_CONTRACT = "mmh3_h3_audio_sync_chunk_proof_v2"
SETTINGS_CONTRACT = "mmh3_long_video_audio_sync_settings_v2"
MODES = ("audio_driven", "lipsync")
SOURCE_MODES = ("auto", "video_reference", "image_reference")
AUDIO_OUTPUT_MODES = ("master_only", "master_plus_generated", "generated_only")


def _normalize_tracks(tracks: Sequence[Mapping[str, Any]], total_frames: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw in tracks:
        track_id = str(raw.get("track_id") or "").strip()
        if not track_id or track_id in ids:
            raise MMH3ResourceError("Audio-sync tracks require unique non-empty track_id values")
        start, end = int(raw.get("start_frame", -1)), int(raw.get("end_frame", -1))
        if start < 0 or end <= start or end > total_frames:
            raise MMH3ResourceError(f"Audio-sync track {track_id!r} has invalid frame bounds")
        ids.add(track_id)
        result.append({
            "track_id": track_id,
            "start_frame": start,
            "end_frame": end,
            "speaker": bool(raw.get("speaker", True)),
            "protected": bool(raw.get("protected", False)),
            "speaker_label": str(raw.get("speaker_label") or "").strip(),
        })
    return sorted(result, key=lambda item: (item["start_frame"], item["end_frame"], item["track_id"]))



def validate_audio_sync_source(packet: Any, settings: Mapping[str, Any]) -> None:
    """Bind conditioning to the same immutable audio that final assembly uses."""
    if settings.get("contract") != SETTINGS_CONTRACT:
        raise MMH3ResourceError("Unknown H3 audio-sync settings contract")
    if packet.manifest.get("id") != settings.get("source_id"):
        raise MMH3ResourceError("H3 audio-sync source packet changed")
    audio = packet.get_primary("audio")
    expected = settings.get("source_audio") or {}
    if audio is None or audio["id"] != expected.get("resource_id"):
        raise MMH3ResourceError("H3 audio-sync primary audio resource changed")
    content = audio.get("content") or {}
    revision = str(content.get("revision") or content.get("digest") or "unknown")
    if revision != str(expected.get("revision") or "unknown"):
        raise MMH3ResourceError("H3 audio-sync source audio revision changed")


def _settings_with_hash(settings: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(settings)
    value.pop("settings_sha256", None)
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    value["settings_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return value


def _validated_audio_delivery(
    *, audio_output_mode: str, master_gain_db: float, generated_gain_db: float, peak_limit: bool
) -> dict[str, Any]:
    output_mode = str(audio_output_mode or "master_only")
    if output_mode not in AUDIO_OUTPUT_MODES:
        raise MMH3ResourceError(f"Unsupported F18 audio output mode {output_mode!r}")
    master_gain = float(master_gain_db)
    generated_gain = float(generated_gain_db)
    if not (-120.0 <= master_gain <= 24.0 and -120.0 <= generated_gain <= 24.0):
        raise MMH3ResourceError("F18 audio gains must be between -120 dB and +24 dB")
    return {
        "master_audio": "immutable_source_owner",
        "audio_output_mode": output_mode,
        "master_gain_db": master_gain,
        "generated_gain_db": generated_gain,
        "peak_limit": bool(peak_limit),
        "generated_audio_semantics": "full_h3_generated_mix_not_fx_stem",
        "upscale": "timeline_preserving_first",
        "generative_refine_requires_lipsync_revalidation": True,
    }


def update_audio_delivery_policy(
    settings: Mapping[str, Any], *, audio_output_mode: str, master_gain_db: float,
    generated_gain_db: float, peak_limit: bool
) -> dict[str, Any]:
    """Change only final F18 delivery mixing; generation/conditioning proof is unchanged."""
    if settings.get("contract") != SETTINGS_CONTRACT:
        raise MMH3ResourceError("Unknown H3 audio-sync settings contract")
    updated = dict(settings)
    updated["delivery_policy"] = _validated_audio_delivery(
        audio_output_mode=audio_output_mode, master_gain_db=master_gain_db,
        generated_gain_db=generated_gain_db, peak_limit=peak_limit,
    )
    return _settings_with_hash(updated)


def _job_settings(settings: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    read_start, read_end = int(job["read_start_frame"]), int(job["read_end_frame"])
    write_start, write_end = int(job["write_start_frame"]), int(job["write_end_frame"])
    tracks = settings.get("tracks") or []
    visible = [item["track_id"] for item in tracks if item["start_frame"] < read_end and item["end_frame"] > read_start]
    speakers = [
        item["track_id"] for item in tracks
        if item["speaker"] and not item["protected"] and item["start_frame"] < write_end and item["end_frame"] > write_start
    ]
    return {
        "job_id": job["job_id"],
        "visible_track_ids": visible,
        "speaker_track_ids": speakers,
        "audio_ownership": [int(job["audio_write_start"]), int(job["audio_write_end"])],
    }


def extend_long_video_audio_sync_settings(settings: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    """Extend frozen F18 settings for one controlled interactive-plan append."""
    if settings.get("contract") != SETTINGS_CONTRACT:
        raise MMH3ResourceError("Unknown H3 audio-sync settings contract")
    updated = dict(settings)
    jobs = [dict(item) for item in settings.get("jobs") or []]
    if any(str(item.get("job_id") or "") == str(job.get("job_id") or "") for item in jobs):
        raise MMH3ResourceError("H3 audio-sync settings already contain this interactive job")
    jobs.append(_job_settings(settings, job))
    updated["jobs"] = jobs
    return _settings_with_hash(updated)

def build_long_video_audio_sync_settings(
    *,
    source_id: str,
    source_archive: str,
    source_audio_resource_id: str,
    source_audio_revision: str,
    chunk_plan: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
    mode: str,
    source_channels: int = 0,
    source_audio_samples: int = 0,
    source_mode: str = "auto",
    require_candidate_review: bool = False,
    audio_output_mode: str = "master_only",
    master_gain_db: float = 0.0,
    generated_gain_db: float = -18.0,
    peak_limit: bool = True,
) -> dict[str, Any]:
    if chunk_plan.get("contract") != "mmh3_long_video_chunk_plan_v1" or str(chunk_plan.get("source_id")) != str(source_id):
        raise MMH3ResourceError("Audio-sync settings require a matching long-video chunk plan")
    if mode not in MODES:
        raise MMH3ResourceError(f"Unsupported H3 audio-sync mode {mode!r}")
    delivery_policy = _validated_audio_delivery(
        audio_output_mode=audio_output_mode, master_gain_db=master_gain_db,
        generated_gain_db=generated_gain_db, peak_limit=peak_limit,
    )
    requested_source_mode = str(source_mode or "auto")
    if requested_source_mode not in SOURCE_MODES:
        raise MMH3ResourceError(f"Unsupported H3 audio-sync source_mode {requested_source_mode!r}")
    planner = str((chunk_plan.get("settings") or {}).get("planner") or "")
    audio_master_planners = {"h3_audio_timeline", "h3_audio_interactive"}
    inferred_source_mode = "image_reference" if planner in audio_master_planners else "video_reference"
    resolved_source_mode = inferred_source_mode if requested_source_mode == "auto" else requested_source_mode
    if planner in audio_master_planners and resolved_source_mode != "image_reference":
        raise MMH3ResourceError("H3 audio-master timeline requires source_mode=image_reference")
    if not source_audio_resource_id:
        raise MMH3ResourceError("H3 audio sync requires a primary source audio resource")
    if not str(source_archive or "").strip():
        raise MMH3ResourceError("H3 audio sync requires a saved source MMH3 archive")
    channels = int(source_channels or 0)
    if channels not in {0, 1, 2}:
        raise MMH3ResourceError("H3 audio sync supports mono or stereo source audio")
    samples = int(source_audio_samples or 0)
    planned_samples = int((chunk_plan.get("settings") or {}).get("master_audio_samples") or 0)
    if samples and planned_samples and samples != planned_samples:
        raise MMH3ResourceError("H3 audio-master plan no longer matches the immutable source audio length")

    normalized = _normalize_tracks(tracks, int(chunk_plan["total_frames"]))
    if mode == "lipsync" and not normalized:
        normalized = [{
            "track_id": "primary_speaker",
            "start_frame": 0,
            "end_frame": int(chunk_plan["total_frames"]),
            "speaker": True,
            "protected": False,
            "speaker_label": "",
        }]
    if mode == "lipsync" and not any(item["speaker"] and not item["protected"] for item in normalized):
        raise MMH3ResourceError("Lipsync mode requires at least one unprotected speaker track")

    jobs = []
    for chunk in chunk_plan["chunks"]:
        read_start, read_end = int(chunk["read_start_frame"]), int(chunk["read_end_frame"])
        write_start, write_end = int(chunk["write_start_frame"]), int(chunk["write_end_frame"])
        visible = [item["track_id"] for item in normalized if item["start_frame"] < read_end and item["end_frame"] > read_start]
        speakers = [item["track_id"] for item in normalized if item["speaker"] and not item["protected"] and item["start_frame"] < write_end and item["end_frame"] > write_start]
        jobs.append({
            "job_id": chunk["job_id"],
            "visible_track_ids": visible,
            "speaker_track_ids": speakers,
            "audio_ownership": [int(chunk["audio_write_start"]), int(chunk["audio_write_end"])],
        })

    settings = {
        "contract": SETTINGS_CONTRACT,
        "source_id": str(source_id),
        "source_archive": str(source_archive),
        "source_audio": {
            "resource_id": str(source_audio_resource_id),
            "revision": str(source_audio_revision or "unknown"),
            "channels": channels,
            "samples": samples,
        },
        "mode": mode,
        "source_mode": resolved_source_mode,
        "review_policy": "candidate_required" if require_candidate_review else "direct_commit_allowed",
        "tracks": normalized,
        "jobs": jobs,
        "h3": {
            "task_family": "ref2va",
            "audio_sample_rate": 32000,
            "audio_driven": {
                "reference_usage": "standalone_audio",
                "timeline_guide": False,
                "intent": "transfer beat, tempo, accents and musical energy; do not copy speech identity unless explicitly prompted",
            },
            "lipsync": {
                "reference_usage": "paired_video_soundtrack" if resolved_source_mode == "video_reference" else "standalone_audio_with_image_identity",
                "timeline_guide": True,
                "guide_node": "MiniMaxH3AddGuide",
                "guide_frame_idx": 0,
                "intent": "follow synchronized source speech or singing timing and visible mouth articulation",
            },
        },
        "audio_policy": "immutable_source_timeline",
        "stereo_policy": "preserve_source_channels",
        "generated_audio_capture": True,
        "multi_speaker_note": "H3 accepts stereo natively, but independent channel-to-face speaker binding is not guaranteed.",
        "delivery_policy": delivery_policy,
    }
    return _settings_with_hash(settings)


def build_h3_audio_sync_chunk_proof(settings: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    if settings.get("contract") != SETTINGS_CONTRACT:
        raise MMH3ResourceError("Unknown H3 audio-sync settings contract")
    expected = next((item for item in settings.get("jobs", []) if item.get("job_id") == job.get("job_id")), None)
    if expected is None:
        raise MMH3ResourceError("H3 audio-sync job is not present in frozen settings")
    mode = str(settings.get("mode") or "")
    h3_mode = settings.get("h3", {}).get(mode, {}) if isinstance(settings.get("h3"), Mapping) else {}
    return {
        "contract": PROOF_CONTRACT,
        "job_id": str(job.get("job_id") or ""),
        "mode": mode,
        "source_mode": str(settings.get("source_mode") or "video_reference"),
        "review_policy": str(settings.get("review_policy") or "direct_commit_allowed"),
        "engine": "minimax_h3_native",
        "task_family": "ref2va",
        "audio_ownership": list(expected["audio_ownership"]),
        "speaker_track_ids": list(expected["speaker_track_ids"]),
        "source_audio_resource_id": settings["source_audio"]["resource_id"],
        "source_audio_revision": settings["source_audio"]["revision"],
        "source_channels": int(settings["source_audio"].get("channels") or 0),
        "reference_usage": h3_mode.get("reference_usage"),
        "timeline_guide": bool(h3_mode.get("timeline_guide")),
        "guide_node": h3_mode.get("guide_node") if h3_mode.get("timeline_guide") else None,
        "guide_frame_idx": h3_mode.get("guide_frame_idx") if h3_mode.get("timeline_guide") else None,
        "audio_policy": settings.get("audio_policy"),
        "artifact_audio_role": "generated_h3_audio" if settings.get("generated_audio_capture") else None,
    }


def validate_lipsync_chunk_proof(settings: Mapping[str, Any], job: Mapping[str, Any], process_info: Mapping[str, Any]) -> None:
    if settings.get("contract") != SETTINGS_CONTRACT:
        raise MMH3ResourceError("Unknown H3 audio-sync settings contract")
    proof = process_info.get("lipsync") or process_info.get("audio_sync")
    if not isinstance(proof, Mapping) or proof.get("contract") != PROOF_CONTRACT:
        raise MMH3ResourceError("H3 audio-sync artifact has no valid chunk proof")
    expected = build_h3_audio_sync_chunk_proof(settings, job)
    for key in (
        "job_id", "mode", "source_mode", "review_policy", "engine", "task_family", "audio_ownership", "speaker_track_ids",
        "source_audio_resource_id", "source_audio_revision", "source_channels", "reference_usage",
        "timeline_guide", "guide_node", "guide_frame_idx", "audio_policy",
    ):
        # Older v2 ledgers/proofs predate these additive fields. Only those
        # ledgers may omit them; new review/source policies remain mandatory.
        if key in {"source_mode", "review_policy"} and key not in settings and key not in proof:
            continue
        if proof.get(key) != expected.get(key):
            raise MMH3ResourceError(f"H3 audio-sync proof field {key!r} differs from frozen settings")
    if settings.get("generated_audio_capture") and proof.get("artifact_audio_role") != expected.get("artifact_audio_role"):
        raise MMH3ResourceError("H3 audio-sync proof field 'artifact_audio_role' differs from frozen settings")


__all__ = [
    "MODES", "SOURCE_MODES", "AUDIO_OUTPUT_MODES", "PROOF_CONTRACT", "SETTINGS_CONTRACT",
    "build_h3_audio_sync_chunk_proof", "build_long_video_audio_sync_settings",
    "extend_long_video_audio_sync_settings", "update_audio_delivery_policy", "validate_lipsync_chunk_proof",
]
