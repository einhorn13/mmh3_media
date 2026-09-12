from __future__ import annotations

import hashlib
import json
import os
import tempfile
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .errors import MMH3ResourceError
from .util import deep_copy_json


EXECUTION_CONTRACT = "mmh3_resumable_execution_v1"
EXECUTION_SUMMARY_CONTRACT = "mmh3_execution_summary_v1"
DEFAULT_LEASE_TIMEOUT_SECONDS = 86400.0
JOB_STATES = {"pending", "running", "review", "completed", "failed", "cancelled"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _plan_jobs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = plan.get("contract")
    key = "chunks" if contract == "mmh3_long_video_chunk_plan_v1" else "jobs"
    if contract not in {"mmh3_long_video_chunk_plan_v1", "mmh3_batch_input_plan_v1"}:
        raise MMH3ResourceError("Unsupported automation plan contract")
    jobs = plan.get(key)
    if not isinstance(jobs, list) or not jobs:
        raise MMH3ResourceError("Automation plan contains no jobs")
    ids = [str(job.get("job_id") or "") for job in jobs if isinstance(job, Mapping)]
    if len(ids) != len(jobs) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise MMH3ResourceError("Automation plan job IDs must be present and unique")
    return [deep_copy_json(dict(job)) for job in jobs]


def create_execution_ledger(
    plan: Mapping[str, Any],
    *,
    operation: str,
    effective_settings: Mapping[str, Any] | None = None,
    error_policy: str = "",
) -> dict[str, Any]:
    jobs = _plan_jobs(plan)
    selected_error_policy = str(error_policy or "")
    if selected_error_policy not in {"", "stop_on_error", "continue_on_error"}:
        raise MMH3ResourceError("Automation error_policy must be stop_on_error or continue_on_error")
    normalized_plan = deep_copy_json(dict(plan))
    settings = deep_copy_json(dict(effective_settings or {}))
    ledger_jobs = [
        {
            "job_id": job["job_id"],
            "state": "pending",
            "attempts": 0,
            "job": job,
            "artifact": None,
            "error": None,
            "last_error": None,
            "lease_id": None,
            "lease_acquired_at": None,
            "lease_expires_at": None,
            "candidates": [],
            "selected_candidate_id": None,
        }
        for job in jobs
    ]
    return {
        "contract": EXECUTION_CONTRACT,
        "revision": 0,
        "operation": str(operation or "process"),
        "error_policy": selected_error_policy,
        "plan_contract": normalized_plan["contract"],
        "plan_sha256": _canonical_hash(normalized_plan),
        "effective_settings": settings,
        "effective_settings_sha256": _canonical_hash(settings),
        "plan": normalized_plan,
        "jobs": ledger_jobs,
        "status": "planned",
    }


def _validate_ledger(ledger: Mapping[str, Any]) -> None:
    if ledger.get("contract") != EXECUTION_CONTRACT:
        raise MMH3ResourceError("Unsupported execution ledger contract")
    if "error_policy" in ledger and not isinstance(ledger.get("error_policy"), str):
        raise MMH3ResourceError("Execution ledger error_policy must be a string")
    plan = ledger.get("plan")
    jobs = ledger.get("jobs")
    if not isinstance(plan, Mapping) or not isinstance(jobs, list):
        raise MMH3ResourceError("Execution ledger plan/jobs are malformed")
    if ledger.get("plan_sha256") != _canonical_hash(plan):
        raise MMH3ResourceError("Execution ledger plan fingerprint mismatch")
    settings = ledger.get("effective_settings")
    if not isinstance(settings, Mapping) or ledger.get("effective_settings_sha256") != _canonical_hash(settings):
        raise MMH3ResourceError("Execution ledger settings fingerprint mismatch")
    planned_jobs = _plan_jobs(plan)
    planned_by_id = {job["job_id"]: job for job in planned_jobs}
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, Mapping) or job.get("state") not in JOB_STATES:
            raise MMH3ResourceError("Execution ledger contains an invalid job state")
        job_id = str(job.get("job_id") or "")
        if job_id in seen or job_id not in planned_by_id or job.get("job") != planned_by_id[job_id]:
            raise MMH3ResourceError("Execution ledger job payload does not match its immutable plan")
        seen.add(job_id)
        attempts = job.get("attempts")
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
            raise MMH3ResourceError("Execution ledger contains an invalid attempt count")
        lease = job.get("lease_id")
        if job["state"] == "running" and (not isinstance(lease, str) or not lease):
            raise MMH3ResourceError("Running execution job is missing its lease")
        if job["state"] != "running" and lease is not None:
            raise MMH3ResourceError("Only a running execution job may retain a lease")
        candidates = job.get("candidates", [])
        if not isinstance(candidates, list):
            raise MMH3ResourceError("Execution ledger candidates must be an array")
        candidate_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise MMH3ResourceError("Execution ledger contains a malformed candidate")
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or candidate_id in candidate_ids:
                raise MMH3ResourceError("Execution ledger candidate IDs must be present and unique per job")
            if candidate.get("job_id") not in (None, job_id):
                raise MMH3ResourceError("Execution candidate belongs to another job")
            candidate_ids.add(candidate_id)
        selected = job.get("selected_candidate_id")
        if selected is not None and str(selected) not in candidate_ids:
            raise MMH3ResourceError("Selected execution candidate does not exist")
        if job["state"] == "review" and not candidates:
            raise MMH3ResourceError("Review execution job has no candidates")
    if seen != set(planned_by_id):
        raise MMH3ResourceError("Execution ledger jobs do not cover the immutable plan")


def _derive_status(jobs: list[dict[str, Any]]) -> str:
    states = {job["state"] for job in jobs}
    if states == {"completed"}:
        return "completed"
    if "running" in states:
        return "running"
    if "review" in states:
        return "needs_review"
    if "failed" in states:
        return "needs_retry"
    if states <= {"completed", "cancelled"} and "cancelled" in states:
        return "cancelled"
    return "planned"


def transition_execution_job(
    ledger: Mapping[str, Any],
    job_id: str,
    action: str,
    *,
    artifact: Mapping[str, Any] | None = None,
    error: str = "",
    lease_id: str = "",
    lease_timeout_seconds: float = DEFAULT_LEASE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    _validate_ledger(ledger)
    updated = deep_copy_json(dict(ledger))
    matches = [job for job in updated["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise MMH3ResourceError(f"Execution job {job_id!r} does not exist")
    job = matches[0]
    state = job["state"]
    allowed = {
        "start": {"pending", "failed"},
        "candidate": {"running"},
        "complete": {"running"},
        "accept_candidate": {"review", "pending", "failed", "cancelled"},
        "reroll": {"review"},
        "fail": {"running"},
        "cancel": {"pending", "running", "review", "failed"},
        "reset": {"failed", "cancelled"},
    }
    if action not in allowed or state not in allowed[action]:
        raise MMH3ResourceError(f"Invalid execution transition: {state} -> {action}")
    if action == "start":
        attempt = int(job["attempts"]) + 1
        lease = _canonical_hash(
            {
                "plan_sha256": updated["plan_sha256"],
                "settings_sha256": updated["effective_settings_sha256"],
                "job_id": job_id,
                "attempt": attempt,
            }
        )[:24]
        now = datetime.now(timezone.utc)
        timeout = max(1.0, float(lease_timeout_seconds))
        job.update({
            "state": "running",
            "attempts": attempt,
            "error": None,
            "lease_id": lease,
            "lease_acquired_at": now.isoformat().replace("+00:00", "Z"),
            "lease_expires_at": (now + timedelta(seconds=timeout)).isoformat().replace("+00:00", "Z"),
        })
    elif action == "candidate":
        if str(lease_id) != str(job.get("lease_id") or ""):
            raise MMH3ResourceError("Execution lease mismatch; refusing a stale candidate")
        if not isinstance(artifact, Mapping) or not artifact:
            raise MMH3ResourceError("Recording an execution candidate requires artifact metadata")
        candidate = deep_copy_json(dict(artifact))
        candidate_id = str(candidate.get("candidate_id") or _canonical_hash({
            "job_id": job_id,
            "attempt": int(job["attempts"]),
            "sha256": candidate.get("sha256"),
            "packet_path": candidate.get("packet_path"),
        })[:16])
        candidate["candidate_id"] = candidate_id
        candidate["job_id"] = job_id
        candidate["attempt"] = int(job["attempts"])
        candidates = list(job.get("candidates") or [])
        if any(str(item.get("candidate_id") or "") == candidate_id for item in candidates if isinstance(item, Mapping)):
            raise MMH3ResourceError("Execution candidate ID already exists")
        candidates.append(candidate)
        job.update({
            "state": "review",
            "candidates": candidates,
            "artifact": None,
            "error": None,
            "lease_id": None,
            "lease_acquired_at": None,
            "lease_expires_at": None,
        })
    elif action == "complete":
        if (updated.get("effective_settings") or {}).get("review_policy") == "candidate_required":
            raise MMH3ResourceError("Direct completion is blocked by candidate review policy")
        if str(lease_id) != str(job.get("lease_id") or ""):
            raise MMH3ResourceError("Execution lease mismatch; refusing a stale artifact commit")
        if not isinstance(artifact, Mapping) or not artifact:
            raise MMH3ResourceError("Completing an execution job requires artifact metadata")
        job.update({"state": "completed", "artifact": deep_copy_json(dict(artifact)), "error": None, "lease_id": None, "lease_acquired_at": None, "lease_expires_at": None})
    elif action == "accept_candidate":
        candidate_id = str((artifact or {}).get("candidate_id") or "")
        if not candidate_id:
            raise MMH3ResourceError("Accepting a candidate requires artifact_json.candidate_id")
        candidate = next((item for item in job.get("candidates", []) if str(item.get("candidate_id") or "") == candidate_id), None)
        if candidate is None:
            raise MMH3ResourceError(f"Execution candidate {candidate_id!r} does not exist")
        job.update({
            "state": "completed",
            "artifact": deep_copy_json(dict(candidate)),
            "selected_candidate_id": candidate_id,
            "error": None,
            "lease_id": None,
            "lease_acquired_at": None,
            "lease_expires_at": None,
        })
    elif action == "reroll":
        job.update({
            "state": "pending",
            "artifact": None,
            "selected_candidate_id": None,
            "error": None,
            "lease_id": None,
            "lease_acquired_at": None,
            "lease_expires_at": None,
        })
    elif action == "fail":
        if str(lease_id) != str(job.get("lease_id") or ""):
            raise MMH3ResourceError("Execution lease mismatch; refusing a stale failure")
        if not str(error).strip():
            raise MMH3ResourceError("Failing an execution job requires an error message")
        job.update({"state": "failed", "artifact": None, "error": str(error), "last_error": str(error), "lease_id": None, "lease_acquired_at": None, "lease_expires_at": None})
    elif action == "cancel":
        if state == "running" and str(lease_id) != str(job.get("lease_id") or ""):
            raise MMH3ResourceError("Execution lease mismatch; refusing a stale cancellation")
        job.update({"state": "cancelled", "error": str(error or "cancelled"), "last_error": str(error or "cancelled"), "lease_id": None, "lease_acquired_at": None, "lease_expires_at": None})
    else:
        job.update({"state": "pending", "artifact": None, "selected_candidate_id": None, "error": None, "lease_id": None, "lease_acquired_at": None, "lease_expires_at": None})
    updated["revision"] = int(updated.get("revision", 0)) + 1
    updated["status"] = _derive_status(updated["jobs"])
    return updated


def _job_entry(ledger: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    matches = [job for job in ledger["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise MMH3ResourceError(f"Execution job {job_id!r} does not exist")
    return matches[0]


def deterministic_artifact_prefix(ledger: Mapping[str, Any], job_id: str) -> str:
    _validate_ledger(ledger)
    entry = _job_entry(ledger, job_id)
    job = entry["job"]
    index = int(job.get("index", 0))
    output_name = str(job.get("output_name") or Path(str(job.get("path") or f"job_{index:04d}")).stem)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", output_name).strip("._") or f"job_{index:04d}"
    operation = re.sub(r"[^A-Za-z0-9._-]+", "_", str(ledger["operation"])).strip("._") or "process"
    return f"mmh3_automation/{operation}/{ledger['plan_sha256'][:12]}/{index:04d}_{safe_name}_{job_id}"


def acquire_next_execution_job(
    ledger: Mapping[str, Any], *, mode: str = "pending_and_failed", exclude_job_ids: tuple[str, ...] = (), lease_timeout_seconds: float = DEFAULT_LEASE_TIMEOUT_SECONDS
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    excluded = set(exclude_job_ids)
    candidates = tuple(job_id for job_id in select_resume_jobs(ledger, mode=mode) if job_id not in excluded)
    if not candidates:
        return deep_copy_json(dict(ledger)), None
    job_id = candidates[0]
    if _job_entry(ledger, job_id)["state"] == "cancelled":
        ledger = transition_execution_job(ledger, job_id, "reset")
    updated = transition_execution_job(ledger, job_id, "start", lease_timeout_seconds=lease_timeout_seconds)
    entry = _job_entry(updated, job_id)
    lease = {
        "contract": "mmh3_execution_job_lease_v1",
        "job_id": job_id,
        "attempt": int(entry["attempts"]),
        "lease_id": entry["lease_id"],
        "lease_acquired_at": entry.get("lease_acquired_at"),
        "lease_expires_at": entry.get("lease_expires_at"),
        "filename_prefix": (
            deterministic_artifact_prefix(updated, job_id) + f"_candidate_{int(entry['attempts']):02d}"
            if (updated.get("effective_settings") or {}).get("review_policy") == "candidate_required"
            else deterministic_artifact_prefix(updated, job_id)
        ),
        "job": deep_copy_json(dict(entry["job"])),
    }
    return updated, lease


def _validate_saved_execution_packet(ledger, entry, saved_packet) -> None:
    """Apply the same operation proof to direct commits and reviewed candidates."""
    if ledger.get("operation") == "long_video_upscale":
        from .automation_upscale import validate_upscale_chunk_artifact

        extensions = saved_packet.manifest.get("extensions", {})
        namespace = extensions.get("mmh3_media", {}) if isinstance(extensions, Mapping) else {}
        last_process = namespace.get("last_process", {}) if isinstance(namespace, Mapping) else {}
        process_info = last_process.get("info") if isinstance(last_process, Mapping) else None
        if not isinstance(process_info, Mapping):
            raise MMH3ResourceError("Upscale artifact has no MMH3 last_process.info proof")
        validate_upscale_chunk_artifact(ledger["effective_settings"], process_info)
    elif ledger.get("operation") in {"long_video_lipsync", "long_video_audio_driven"}:
        from .automation_lipsync import validate_lipsync_chunk_proof

        extensions = saved_packet.manifest.get("extensions", {})
        namespace = extensions.get("mmh3_media", {}) if isinstance(extensions, Mapping) else {}
        last_process = namespace.get("last_process", {}) if isinstance(namespace, Mapping) else {}
        process_info = last_process.get("info") if isinstance(last_process, Mapping) else None
        if not isinstance(process_info, Mapping):
            raise MMH3ResourceError("H3 audio-sync artifact has no MMH3 last_process.info proof")
        validate_lipsync_chunk_proof(ledger["effective_settings"], entry["job"], process_info)
        delivery = (ledger["effective_settings"].get("delivery_policy") or {})
        if str(delivery.get("audio_output_mode") or "master_only") != "master_only" and saved_packet.get_primary("audio") is None:
            raise MMH3ResourceError("Selected F18 audio output mode requires captured generated H3 audio")


def commit_execution_artifact(
    ledger: Mapping[str, Any],
    job_id: str,
    *,
    lease_id: str,
    packet_path: str | Path,
    packet_id: str = "",
) -> dict[str, Any]:
    _validate_ledger(ledger)
    entry = _job_entry(ledger, job_id)
    if (ledger.get("effective_settings") or {}).get("review_policy") == "candidate_required":
        raise MMH3ResourceError("Direct artifact commit is blocked by candidate review policy; use commit_candidate then accept_candidate")
    path = Path(packet_path).resolve()
    if path.suffix.casefold() != ".mmh3" or not path.is_file():
        raise MMH3ResourceError("Artifact commit requires an existing .mmh3 packet_path")
    from .archive import load_archive

    try:
        saved_packet = load_archive(path, verify="manifest")
    except Exception as exc:
        raise MMH3ResourceError(f"Artifact commit rejected an invalid MMH3 archive: {exc}") from exc
    actual_packet_id = str(saved_packet.manifest.get("id") or "")
    if packet_id and str(packet_id) != actual_packet_id:
        raise MMH3ResourceError("Artifact packet_id does not match the saved MMH3 archive")
    _validate_saved_execution_packet(ledger, entry, saved_packet)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    artifact: dict[str, Any] = {
        "contract": "mmh3_execution_artifact_v1",
        "job_id": job_id,
        "attempt": int(entry["attempts"]),
        "packet_path": str(path),
        "packet_id": actual_packet_id,
        "size": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }
    job = entry["job"]
    if ledger.get("plan_contract") == "mmh3_long_video_chunk_plan_v1":
        artifact["ownership"] = {
            key: int(job[key])
            for key in (
                "read_start_frame", "read_end_frame", "write_start_frame", "write_end_frame",
                "audio_read_start", "audio_read_end", "audio_write_start", "audio_write_end",
            )
        }
    return transition_execution_job(
        ledger, job_id, "complete", artifact=artifact, lease_id=lease_id
    )


def commit_execution_candidate(
    ledger: Mapping[str, Any],
    job_id: str,
    *,
    lease_id: str,
    packet_path: str | Path,
    packet_id: str = "",
    label: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Verify one generated MMH3 artifact and park the job in explicit human review."""
    _validate_ledger(ledger)
    entry = _job_entry(ledger, job_id)
    path = Path(packet_path).resolve()
    if path.suffix.casefold() != ".mmh3" or not path.is_file():
        raise MMH3ResourceError("Candidate commit requires an existing .mmh3 packet_path")
    from .archive import load_archive
    saved_packet = load_archive(path, verify="manifest")
    actual_packet_id = str(saved_packet.manifest.get("id") or "")
    if packet_id and str(packet_id) != actual_packet_id:
        raise MMH3ResourceError("Candidate packet_id does not match the saved MMH3 archive")
    _validate_saved_execution_packet(ledger, entry, saved_packet)
    artifact: dict[str, Any] = {
        "contract": "mmh3_execution_candidate_v1",
        "job_id": job_id,
        "attempt": int(entry["attempts"]),
        "packet_path": str(path),
        "packet_id": actual_packet_id,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "label": str(label or "").strip(),
        "notes": str(notes or "").strip(),
    }
    job = entry["job"]
    if ledger.get("plan_contract") == "mmh3_long_video_chunk_plan_v1":
        artifact["ownership"] = {
            key: int(job[key]) for key in (
                "read_start_frame", "read_end_frame", "write_start_frame", "write_end_frame",
                "audio_read_start", "audio_read_end", "audio_write_start", "audio_write_end",
            )
        }
    return transition_execution_job(ledger, job_id, "candidate", artifact=artifact, lease_id=lease_id)


def accept_execution_candidate(ledger: Mapping[str, Any], job_id: str, candidate_id: str) -> dict[str, Any]:
    return transition_execution_job(
        ledger,
        job_id,
        "accept_candidate",
        artifact={"candidate_id": str(candidate_id or "").strip()},
    )



def append_interactive_audio_try(
    ledger: Mapping[str, Any],
    *,
    shot_duration_seconds: float,
    context_seconds: float | None = None,
) -> dict[str, Any]:
    """Append one next F18 shot after every existing shot has been accepted."""
    _validate_ledger(ledger)
    if ledger.get("plan_contract") != "mmh3_long_video_chunk_plan_v1":
        raise MMH3ResourceError("Interactive F18 append requires a long-video chunk plan")
    plan_settings = (ledger.get("plan") or {}).get("settings") or {}
    if plan_settings.get("planner") != "h3_audio_interactive":
        raise MMH3ResourceError("Interactive F18 append requires an h3_audio_interactive plan")
    unfinished = [job for job in ledger["jobs"] if job["state"] != "completed"]
    if unfinished:
        states = ", ".join(sorted({str(job["state"]) for job in unfinished}))
        raise MMH3ResourceError(f"Accept or resolve the current F18 shot before planning the next try ({states})")
    from .automation import append_h3_audio_interactive_try
    from .automation_lipsync import extend_long_video_audio_sync_settings

    updated = deep_copy_json(dict(ledger))
    old_count = len(updated["plan"]["chunks"])
    new_plan = append_h3_audio_interactive_try(
        updated["plan"], shot_duration_seconds=float(shot_duration_seconds), context_seconds=context_seconds
    )
    if len(new_plan["chunks"]) != old_count + 1:
        raise MMH3ResourceError("Interactive F18 planner did not append exactly one shot")
    new_job = deep_copy_json(dict(new_plan["chunks"][-1]))
    settings = updated.get("effective_settings") or {}
    if settings.get("contract") == "mmh3_long_video_audio_sync_settings_v2":
        settings = extend_long_video_audio_sync_settings(settings, new_job)
    updated["plan"] = new_plan
    updated["plan_sha256"] = _canonical_hash(new_plan)
    updated["effective_settings"] = deep_copy_json(dict(settings))
    updated["effective_settings_sha256"] = _canonical_hash(updated["effective_settings"])
    updated["jobs"].append({
        "job_id": new_job["job_id"],
        "state": "pending",
        "attempts": 0,
        "job": new_job,
        "artifact": None,
        "error": None,
        "last_error": None,
        "lease_id": None,
        "lease_acquired_at": None,
        "lease_expires_at": None,
        "candidates": [],
        "selected_candidate_id": None,
    })
    updated["revision"] = int(updated.get("revision", 0)) + 1
    updated["status"] = _derive_status(updated["jobs"])
    _validate_ledger(updated)
    return updated


def finalize_interactive_audio_sequence(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze an accepted interactive F18 prefix so bounded assembly may run."""
    _validate_ledger(ledger)
    plan_settings = (ledger.get("plan") or {}).get("settings") or {}
    if plan_settings.get("planner") != "h3_audio_interactive":
        raise MMH3ResourceError("Interactive F18 finalize requires an h3_audio_interactive plan")
    if any(job["state"] != "completed" for job in ledger["jobs"]):
        raise MMH3ResourceError("Accept or resolve every planned F18 shot before finalizing the sequence")
    from .automation import finalize_h3_audio_interactive_plan

    updated = deep_copy_json(dict(ledger))
    new_plan = finalize_h3_audio_interactive_plan(updated["plan"])
    updated["plan"] = new_plan
    updated["plan_sha256"] = _canonical_hash(new_plan)
    updated["revision"] = int(updated.get("revision", 0)) + 1
    updated["status"] = _derive_status(updated["jobs"])
    _validate_ledger(updated)
    return updated

def set_execution_audio_delivery_policy(
    ledger: Mapping[str, Any], *, audio_output_mode: str, master_gain_db: float = 0.0,
    generated_gain_db: float = -18.0, peak_limit: bool = True,
) -> dict[str, Any]:
    """Update only F18 final-audio delivery without invalidating generated candidates."""
    _validate_ledger(ledger)
    if any(job["state"] == "running" for job in ledger["jobs"]):
        raise MMH3ResourceError("Cannot change F18 delivery audio while a generation job is running")
    settings = ledger.get("effective_settings") or {}
    if settings.get("contract") != "mmh3_long_video_audio_sync_settings_v2":
        raise MMH3ResourceError("Audio delivery policy requires F18 audio-sync settings")
    from .automation_lipsync import update_audio_delivery_policy

    updated = deep_copy_json(dict(ledger))
    new_settings = update_audio_delivery_policy(
        settings, audio_output_mode=audio_output_mode, master_gain_db=master_gain_db,
        generated_gain_db=generated_gain_db, peak_limit=peak_limit,
    )
    updated["effective_settings"] = new_settings
    updated["effective_settings_sha256"] = _canonical_hash(new_settings)
    updated["revision"] = int(updated.get("revision", 0)) + 1
    _validate_ledger(updated)
    return updated


def select_resume_jobs(ledger: Mapping[str, Any], *, mode: str = "pending_and_failed") -> tuple[str, ...]:
    _validate_ledger(ledger)
    states_by_mode = {
        "pending_and_failed": {"pending", "failed"},
        "failed_only": {"failed"},
        "include_cancelled": {"pending", "failed", "cancelled"},
    }
    try:
        states = states_by_mode[mode]
    except KeyError as exc:
        raise MMH3ResourceError(f"Unknown resume mode {mode!r}") from exc
    return tuple(job["job_id"] for job in ledger["jobs"] if job["state"] in states)


def build_chunk_assembly_map(ledger: Mapping[str, Any]) -> dict[str, Any]:
    _validate_ledger(ledger)
    if ledger.get("plan_contract") != "mmh3_long_video_chunk_plan_v1":
        raise MMH3ResourceError("Assembly map requires a long-video chunk plan")
    plan_settings = (ledger.get("plan") or {}).get("settings") or {}
    if plan_settings.get("planner") == "h3_audio_interactive" and not bool(plan_settings.get("finalized")):
        raise MMH3ResourceError("Interactive F18 assembly is blocked until the accepted sequence is finalized")
    if any(job["state"] != "completed" for job in ledger["jobs"]):
        raise MMH3ResourceError("Assembly is blocked until every planned chunk is completed")
    jobs = sorted(ledger["jobs"], key=lambda item: int(item["job"]["index"]))
    expected_frame = 0
    expected_audio = 0
    segments = []
    for item in jobs:
        job = item["job"]
        if int(job["write_start_frame"]) != expected_frame or int(job["audio_write_start"]) != expected_audio:
            raise MMH3ResourceError("Chunk write/audio ownership is not contiguous")
        artifact = item.get("artifact")
        if not isinstance(artifact, Mapping) or not artifact.get("packet_path"):
            raise MMH3ResourceError("Completed chunk is missing artifact.packet_path")
        if artifact.get("job_id") not in (None, item["job_id"]):
            raise MMH3ResourceError("Completed chunk artifact belongs to another job")
        ownership = artifact.get("ownership")
        if ownership is not None:
            expected_ownership = {
                key: int(job[key])
                for key in (
                    "read_start_frame", "read_end_frame", "write_start_frame", "write_end_frame",
                    "audio_read_start", "audio_read_end", "audio_write_start", "audio_write_end",
                )
            }
            if ownership != expected_ownership:
                raise MMH3ResourceError("Completed chunk artifact ownership does not match the plan")
        expected_frame = int(job["write_end_frame"])
        expected_audio = int(job["audio_write_end"])
        segments.append(
            {
                "job_id": item["job_id"],
                "packet_path": str(artifact["packet_path"]),
                "packet_id": str(artifact.get("packet_id") or ""),
                "size": artifact.get("size"),
                "sha256": artifact.get("sha256"),
                "write_start_frame": int(job["write_start_frame"]),
                "write_end_frame": expected_frame,
                "audio_write_start": int(job["audio_write_start"]),
                "audio_write_end": expected_audio,
                "video_trim_start_frame": int(job["write_start_frame"]) - int(job["read_start_frame"]),
                "video_trim_end_frame": int(job["write_end_frame"]) - int(job["read_start_frame"]),
                "audio_trim_start": int(job["audio_write_start"]) - int(job["audio_read_start"]),
                "audio_trim_end": int(job["audio_write_end"]) - int(job["audio_read_start"]),
            }
        )
    excluded = ledger["plan"].get("excluded_tail")
    total_frames = int(ledger["plan"]["total_frames"])
    expected_end = int(excluded["start_frame"]) if excluded else total_frames
    if expected_frame != expected_end:
        raise MMH3ResourceError("Chunk assembly does not cover the declared output timeline")

    effective_settings = ledger["effective_settings"]
    mode = None
    if effective_settings.get("contract") == "mmh3_long_video_audio_sync_settings_v2":
        mode = str(effective_settings.get("mode") or "")
        if mode not in {"audio_driven", "lipsync"}:
            raise MMH3ResourceError("Audio-sync assembly settings contain an unsupported mode")
        if ledger.get("operation") != f"long_video_{mode}":
            raise MMH3ResourceError("Audio-sync assembly operation does not match the frozen settings mode")

    return {
        "contract": "mmh3_chunk_assembly_map_v1",
        "plan_sha256": ledger["plan_sha256"],
        "operation": ledger["operation"],
        "mode": mode,
        "effective_settings": deep_copy_json(effective_settings),
        "effective_settings_sha256": ledger["effective_settings_sha256"],
        "segments": segments,
        "output_frames": expected_frame,
        "output_audio_samples": expected_audio,
        "excluded_tail": deep_copy_json(excluded),
    }



def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _interval_diagnostics(intervals: list[tuple[int, int, str]], expected_end: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gaps: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    cursor = 0
    previous_id = ""
    for start, end, job_id in sorted(intervals, key=lambda item: (item[0], item[1], item[2])):
        if start > cursor:
            gaps.append({"start": cursor, "end": start})
        elif start < cursor:
            overlaps.append({"start": start, "end": min(cursor, end), "job_id": job_id, "previous_job_id": previous_id})
        if end > cursor:
            cursor = end
            previous_id = job_id
    if cursor < expected_end:
        gaps.append({"start": cursor, "end": expected_end})
    return gaps, overlaps


def build_execution_summary(
    ledger: Mapping[str, Any], *, now: datetime | None = None, verify_artifacts: bool = True
) -> dict[str, Any]:
    _validate_ledger(ledger)
    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    grouped: dict[str, list[dict[str, Any]]] = {state: [] for state in JOB_STATES}
    issues: list[dict[str, Any]] = []
    completed_frame_intervals: list[tuple[int, int, str]] = []
    completed_audio_intervals: list[tuple[int, int, str]] = []
    plan_frame_intervals: list[tuple[int, int, str]] = []
    plan_audio_intervals: list[tuple[int, int, str]] = []

    for entry in ledger["jobs"]:
        state = str(entry["state"])
        artifact = entry.get("artifact") if isinstance(entry.get("artifact"), Mapping) else None
        artifact_report = None
        if artifact is not None:
            path_text = str(artifact.get("packet_path") or "")
            path = Path(path_text) if path_text else None
            exists = bool(path and path.is_file())
            actual_size = path.stat().st_size if exists else None
            actual_sha = _sha256_file(path) if exists and verify_artifacts else None
            expected_size = artifact.get("size")
            expected_sha = str(artifact.get("sha256") or "") or None
            size_matches = None if not exists or expected_size is None else actual_size == int(expected_size)
            sha_matches = None if not exists or not expected_sha or actual_sha is None else actual_sha == expected_sha
            valid = exists and size_matches is not False and sha_matches is not False
            artifact_report = {
                "packet_path": path_text,
                "packet_id": str(artifact.get("packet_id") or ""),
                "expected_size": expected_size,
                "expected_sha256": expected_sha,
                "exists": exists,
                "verified_size": actual_size,
                "verified_sha256": actual_sha,
                "size_matches": size_matches,
                "sha256_matches": sha_matches,
                "valid": bool(valid),
            }
            if not valid:
                code = "artifact_missing" if not exists else "artifact_corrupt"
                issues.append({"code": code, "severity": "error", "job_id": entry["job_id"], "message": f"Artifact verification failed for {entry['job_id']}"})
        elif state == "completed":
            issues.append({"code": "artifact_missing_metadata", "severity": "error", "job_id": entry["job_id"], "message": "Completed job has no artifact metadata"})

        lease = None
        if state == "running":
            acquired = _parse_utc(entry.get("lease_acquired_at"))
            expires = _parse_utc(entry.get("lease_expires_at"))
            stale = bool(expires and instant >= expires)
            age_unknown = acquired is None or expires is None
            lease = {
                "lease_id": str(entry.get("lease_id") or ""),
                "acquired_at": entry.get("lease_acquired_at"),
                "expires_at": entry.get("lease_expires_at"),
                "stale": stale,
                "age_unknown": age_unknown,
            }
            if stale:
                issues.append({"code": "stale_lease", "severity": "error", "job_id": entry["job_id"], "message": "Running job lease has expired"})
            elif age_unknown:
                issues.append({"code": "lease_age_unknown", "severity": "warning", "job_id": entry["job_id"], "message": "Legacy running lease has no expiry metadata"})

        candidate_reports = []
        for candidate in entry.get("candidates", []):
            path_text = str(candidate.get("packet_path") or "")
            path = Path(path_text) if path_text else None
            exists = bool(path and path.is_file())
            candidate_reports.append({
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "attempt": int(candidate.get("attempt") or 0),
                "label": str(candidate.get("label") or ""),
                "notes": str(candidate.get("notes") or ""),
                "packet_path": path_text,
                "packet_id": str(candidate.get("packet_id") or ""),
                "exists": exists,
                "selected": str(candidate.get("candidate_id") or "") == str(entry.get("selected_candidate_id") or ""),
            })
        job_report = {
            "job_id": entry["job_id"],
            "attempts": int(entry["attempts"]),
            "last_error": entry.get("last_error", entry.get("error")),
            "artifact": artifact_report,
            "lease": lease,
            "candidates": candidate_reports,
            "selected_candidate_id": entry.get("selected_candidate_id"),
        }
        grouped[state].append(job_report)

        job = entry["job"]
        if ledger.get("plan_contract") == "mmh3_long_video_chunk_plan_v1":
            frame_interval = (int(job["write_start_frame"]), int(job["write_end_frame"]), entry["job_id"])
            audio_interval = (int(job["audio_write_start"]), int(job["audio_write_end"]), entry["job_id"])
            plan_frame_intervals.append(frame_interval)
            plan_audio_intervals.append(audio_interval)
            if state == "completed" and artifact_report and artifact_report["valid"]:
                completed_frame_intervals.append(frame_interval)
                completed_audio_intervals.append(audio_interval)

    total = len(ledger["jobs"])
    completed = len(grouped["completed"])
    aggregates: dict[str, Any] = {
        "jobs_total": total,
        "jobs_completed": completed,
        "success_rate": (completed / total) if total else 0.0,
        "frames_processed": None,
        "pcm_samples_processed": None,
        "skipped_frame_ranges": [],
        "skipped_pcm_ranges": [],
    }
    assembly_reasons: list[str] = []
    if ledger.get("plan_contract") == "mmh3_long_video_chunk_plan_v1":
        excluded = ledger["plan"].get("excluded_tail")
        plan_settings = ledger["plan"].get("settings") or {}
        interactive_open = plan_settings.get("planner") == "h3_audio_interactive" and not bool(plan_settings.get("finalized"))
        if interactive_open:
            expected_frame_end = int(plan_settings.get("sequence_cursor_frame") or 0)
        else:
            expected_frame_end = int(excluded["start_frame"]) if isinstance(excluded, Mapping) else int(ledger["plan"]["total_frames"])
        fps_value = ledger["plan"].get("fps", {})
        try:
            fps_num = int(fps_value["numerator"])
            fps_den = int(fps_value["denominator"])
            sample_rate = int(ledger["plan"]["audio_sample_rate"])
            master_samples = int(plan_settings.get("master_audio_samples") or 0)
            if plan_settings.get("planner") == "h3_audio_interactive":
                expected_audio_end = int(plan_settings.get("sequence_cursor_audio_sample") or 0)
            else:
                expected_audio_end = master_samples or int((expected_frame_end * sample_rate * fps_den / fps_num) + 0.5)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            expected_audio_end = max((end for _, end, _ in plan_audio_intervals), default=0)
        plan_frame_gaps, plan_frame_overlaps = _interval_diagnostics(plan_frame_intervals, expected_frame_end)
        plan_audio_gaps, plan_audio_overlaps = _interval_diagnostics(plan_audio_intervals, expected_audio_end)
        completed_frame_gaps, _ = _interval_diagnostics(completed_frame_intervals, expected_frame_end)
        completed_audio_gaps, _ = _interval_diagnostics(completed_audio_intervals, expected_audio_end)
        aggregates.update({
            "frames_processed": sum(end - start for start, end, _ in completed_frame_intervals),
            "pcm_samples_processed": sum(end - start for start, end, _ in completed_audio_intervals),
            "skipped_frame_ranges": completed_frame_gaps,
            "skipped_pcm_ranges": completed_audio_gaps,
            "excluded_tail": deep_copy_json(excluded),
        })
        for code, values, dimension in (("plan_gap", plan_frame_gaps, "frames"), ("plan_overlap", plan_frame_overlaps, "frames"), ("plan_gap", plan_audio_gaps, "pcm"), ("plan_overlap", plan_audio_overlaps, "pcm")):
            for value in values:
                issues.append({"code": code, "severity": "error", "dimension": dimension, "range": value, "message": f"{dimension} ownership {code.replace('plan_', '')}"})
        if plan_frame_gaps or plan_audio_gaps:
            assembly_reasons.append("gaps")
        if plan_frame_overlaps or plan_audio_overlaps:
            assembly_reasons.append("overlaps")
        if interactive_open:
            assembly_reasons.append("interactive_not_finalized")

    counts = {state: len(grouped[state]) for state in ("completed", "review", "failed", "cancelled", "pending", "running")}
    if any(counts[state] for state in ("review", "failed", "cancelled", "pending", "running")):
        assembly_reasons.append("jobs_not_completed")
    if any(issue["code"].startswith("artifact_") for issue in issues):
        assembly_reasons.append("artifact_verification_failed")
    if any(issue["code"] in {"stale_lease", "lease_age_unknown"} for issue in issues):
        assembly_reasons.append("lease_requires_inspection")

    ready = not assembly_reasons and ledger.get("plan_contract") == "mmh3_long_video_chunk_plan_v1"
    if ready:
        try:
            build_chunk_assembly_map(ledger)
        except MMH3ResourceError as exc:
            ready = False
            assembly_reasons.append("assembly_contract_error")
            issues.append({"code": "assembly_impossible", "severity": "error", "message": str(exc)})
    if ledger.get("plan_contract") != "mmh3_long_video_chunk_plan_v1":
        assembly_reasons.append("assembly_not_applicable")

    hard_inspect = any(issue["code"] in {"artifact_missing", "artifact_corrupt", "artifact_missing_metadata", "plan_gap", "plan_overlap", "stale_lease", "lease_age_unknown", "assembly_impossible"} for issue in issues)
    if ready:
        recommendation = "assemble"
    elif counts["review"]:
        recommendation = "review_candidates"
    elif "interactive_not_finalized" in assembly_reasons and not any(counts[state] for state in ("failed", "cancelled", "pending", "running")):
        recommendation = "append_next_or_finalize"
    elif hard_inspect or counts["cancelled"] or counts["running"]:
        recommendation = "inspect"
    elif counts["failed"]:
        recommendation = "retry_failed_only"
    elif counts["pending"]:
        recommendation = "resume_pending"
    else:
        recommendation = "inspect"

    return {
        "contract": EXECUTION_SUMMARY_CONTRACT,
        "generated_at": instant.isoformat().replace("+00:00", "Z"),
        "ledger": {
            "contract": ledger["contract"],
            "revision": int(ledger.get("revision", 0)),
            "status": ledger.get("status"),
            "operation": ledger.get("operation"),
            "plan_contract": ledger.get("plan_contract"),
            "plan_sha256": ledger.get("plan_sha256"),
        },
        "counts": counts,
        "jobs": {state: grouped[state] for state in ("completed", "review", "failed", "cancelled", "pending", "running")},
        "aggregates": aggregates,
        "issues": issues,
        "assembly": {"ready": bool(ready), "reasons": list(dict.fromkeys(assembly_reasons))},
        "recommendation": recommendation,
    }


def save_execution_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    if summary.get("contract") != EXECUTION_SUMMARY_CONTRACT:
        raise MMH3ResourceError("Unsupported execution summary contract")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return target


def save_execution_ledger(ledger: Mapping[str, Any], path: str | Path) -> Path:
    _validate_ledger(ledger)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(ledger, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return target


def load_execution_ledger(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MMH3ResourceError("Execution ledger file must contain a JSON object")
    _validate_ledger(value)
    return value


__all__ = [
    "EXECUTION_CONTRACT",
    "EXECUTION_SUMMARY_CONTRACT",
    "DEFAULT_LEASE_TIMEOUT_SECONDS",
    "JOB_STATES",
    "accept_execution_candidate",
    "acquire_next_execution_job",
    "append_interactive_audio_try",
    "build_chunk_assembly_map",
    "build_execution_summary",
    "commit_execution_artifact",
    "commit_execution_candidate",
    "create_execution_ledger",
    "deterministic_artifact_prefix",
    "finalize_interactive_audio_sequence",
    "load_execution_ledger",
    "save_execution_ledger",
    "save_execution_summary",
    "set_execution_audio_delivery_policy",
    "select_resume_jobs",
    "transition_execution_job",
]
