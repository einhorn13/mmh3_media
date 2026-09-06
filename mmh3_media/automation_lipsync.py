from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .errors import MMH3ResourceError


PROOF_CONTRACT = "mmh3_h3_audio_sync_chunk_proof_v2"
SETTINGS_CONTRACT = "mmh3_long_video_audio_sync_settings_v2"
MODES = ("audio_driven", "lipsync")


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
) -> dict[str, Any]:
    if chunk_plan.get("contract") != "mmh3_long_video_chunk_plan_v1" or str(chunk_plan.get("source_id")) != str(source_id):
        raise MMH3ResourceError("Audio-sync settings require a matching long-video chunk plan")
    if mode not in MODES:
        raise MMH3ResourceError(f"Unsupported H3 audio-sync mode {mode!r}")
    if not source_audio_resource_id:
        raise MMH3ResourceError("H3 audio sync requires a primary source audio resource")
    if not str(source_archive or "").strip():
        raise MMH3ResourceError("H3 audio sync requires a saved source MMH3 archive")
    channels = int(source_channels or 0)
    if channels not in {0, 1, 2}:
        raise MMH3ResourceError("H3 audio sync supports mono or stereo source audio")

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
        },
        "mode": mode,
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
                "reference_usage": "paired_video_soundtrack",
                "timeline_guide": True,
                "guide_node": "MiniMaxH3AddGuide",
                "guide_frame_idx": 0,
                "intent": "follow synchronized source speech or singing timing and visible mouth articulation",
            },
        },
        "audio_policy": "immutable_source_owner",
        "stereo_policy": "preserve_source_channels",
        "multi_speaker_note": "H3 accepts stereo natively, but independent channel-to-face speaker binding is not guaranteed.",
    }
    body = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    settings["settings_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return settings


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
    }


def validate_lipsync_chunk_proof(settings: Mapping[str, Any], job: Mapping[str, Any], process_info: Mapping[str, Any]) -> None:
    if settings.get("contract") != SETTINGS_CONTRACT:
        raise MMH3ResourceError("Unknown H3 audio-sync settings contract")
    proof = process_info.get("lipsync") or process_info.get("audio_sync")
    if not isinstance(proof, Mapping) or proof.get("contract") != PROOF_CONTRACT:
        raise MMH3ResourceError("H3 audio-sync artifact has no valid chunk proof")
    expected = build_h3_audio_sync_chunk_proof(settings, job)
    for key in (
        "job_id", "mode", "engine", "task_family", "audio_ownership", "speaker_track_ids",
        "source_audio_resource_id", "source_audio_revision", "source_channels", "reference_usage",
        "timeline_guide", "guide_node", "guide_frame_idx", "audio_policy",
    ):
        if proof.get(key) != expected.get(key):
            raise MMH3ResourceError(f"H3 audio-sync proof field {key!r} differs from frozen settings")


__all__ = [
    "MODES", "PROOF_CONTRACT", "SETTINGS_CONTRACT",
    "build_h3_audio_sync_chunk_proof", "build_long_video_audio_sync_settings",
    "validate_lipsync_chunk_proof",
]
