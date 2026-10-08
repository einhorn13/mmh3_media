from __future__ import annotations

import json
import hashlib
import time
import urllib.error
import urllib.request
import uuid
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from .automation_execution import (
    acquire_next_execution_job,
    build_execution_summary,
    commit_execution_artifact,
    commit_execution_candidate,
    save_execution_ledger,
    load_execution_ledger,
    transition_execution_job,
)
from .errors import MMH3ResourceError
from .util import deep_copy_json


class RemoteJobFailed(MMH3ResourceError):
    """History proves the remote execution is terminal."""


class RunnerCancelled(MMH3ResourceError):
    """The owned prompt has left the remote queue after targeted cancellation."""


class RemoteStateUncertain(MMH3ResourceError):
    """A dispatched request must not become retryable after a local failure."""


class QueueHTTPError(MMH3ResourceError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


PLACEHOLDERS = {
    "__MMH3_PLAN_JSON__": "plan_json",
    "__MMH3_JOB_ID__": "job_id",
    "__MMH3_LEASE_ID__": "lease_id",
    "__MMH3_FILENAME_PREFIX__": "filename_prefix",
    "__MMH3_EFFECTIVE_SETTINGS_JSON__": "effective_settings_json",
    "__MMH3_ATTEMPT_SEED__": "attempt_seed",
}


def materialize_job_workflow(template: Mapping[str, Any], ledger: Mapping[str, Any], lease: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "plan_json": json.dumps(ledger["plan"], ensure_ascii=False),
        "job_id": str(lease["job_id"]),
        "lease_id": str(lease["lease_id"]),
        "filename_prefix": str(lease["filename_prefix"]),
        "effective_settings_json": json.dumps(ledger.get("effective_settings", {}), ensure_ascii=False),
        "attempt_seed": int.from_bytes(hashlib.sha256(
            f"{lease['job_id']}:{int(lease.get('attempt', 1))}".encode("utf-8")
        ).digest()[:8], "big"),
    }
    replaced: set[str] = set()

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str) and value in PLACEHOLDERS:
            replaced.add(value)
            return values[PLACEHOLDERS[value]]
        if isinstance(value, str):
            match = re.fullmatch(r"__MMH3_SETTING_([A-Za-z0-9_.-]+)__", value)
            if match:
                selected: Any = ledger.get("effective_settings", {})
                for part in match.group(1).split("."):
                    if not isinstance(selected, Mapping) or part not in selected:
                        raise MMH3ResourceError(f"Ledger effective settings have no {match.group(1)!r}")
                    selected = selected[part]
                return deep_copy_json(selected)
        return value

    result = replace(deep_copy_json(dict(template)))
    missing = [placeholder for placeholder in ("__MMH3_PLAN_JSON__", "__MMH3_JOB_ID__", "__MMH3_FILENAME_PREFIX__") if placeholder not in replaced]
    if missing:
        raise MMH3ResourceError("Workflow template is missing required placeholders: " + ", ".join(missing))
    return result


def materialize_assembly_workflow(template: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    if ledger.get("status") != "completed":
        raise MMH3ResourceError("Final assembly requires a completed execution ledger")
    ledger_json = json.dumps(ledger, ensure_ascii=False)
    replaced = False

    def replace(value: Any) -> Any:
        nonlocal replaced
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if value == "__MMH3_LEDGER_JSON__":
            replaced = True
            return ledger_json
        return value

    result = replace(deep_copy_json(dict(template)))
    if not replaced:
        raise MMH3ResourceError("Assembly workflow template is missing __MMH3_LEDGER_JSON__")
    return result


def run_final_assembly(
    ledger: Mapping[str, Any],
    workflow_template: Mapping[str, Any],
    *,
    client: Any,
    save_node_id: str = "",
    poll_seconds: float = 1.0,
    timeout_seconds: float = 86400.0,
    on_event: Callable[[str], None] = print,
) -> str:
    workflow = materialize_assembly_workflow(workflow_template, ledger)
    on_event("QUEUE final_assembly")
    prompt_id = client.queue(workflow)
    history = client.wait(prompt_id, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
    packet_path = find_saved_mmh3(history, save_node_id)
    on_event(f"DONE final_assembly -> {packet_path}")
    return packet_path


def find_saved_mmh3(history_entry: Mapping[str, Any], save_node_id: str = "") -> str:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, Mapping):
        raise MMH3ResourceError("ComfyUI history has no outputs")
    roots = [outputs.get(save_node_id)] if save_node_id else list(outputs.values())
    matches: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.casefold().endswith(".mmh3"):
            matches.append(value)

    for root in roots:
        if root is not None:
            visit(root)
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise MMH3ResourceError(f"Expected exactly one saved .mmh3 output; found {len(unique)}")
    return unique[0]


def stamp_submission_workflow(workflow, receipt):
    """Persist attempt ownership inside saved packets without changing media state."""
    workflow = deep_copy_json(workflow)
    for node_id, node in list(workflow.items()):
        if not isinstance(node, dict) or node.get("class_type") != "MMH3Save":
            continue
        packet = node.get("inputs", {}).get("packet")
        if not isinstance(packet, list) or len(packet) != 2:
            raise MMH3ResourceError("Automation Save must receive a linked packet")
        receipt_id = "_mmh3_receipt_" + str(node_id)
        if receipt_id in workflow:
            raise MMH3ResourceError("Reserved automation receipt node ID already exists")
        workflow[receipt_id] = {"class_type": "MMH3Metadata", "inputs": {
            "packet": packet, "name": "", "tags_action": "keep", "tags": "",
            "notes_action": "keep", "notes": "",
            "custom_json_merge_patch": json.dumps({"extensions": {"mmh3_media": {"execution_receipt": receipt}}})}}
        node["inputs"]["packet"] = [receipt_id, 0]
    return workflow


def recover_saved_artifact(output_root, submission):
    from .archive import load_archive
    root = Path(output_root).resolve()
    prefix = (root / submission["filename_prefix"]).resolve()
    if not prefix.is_relative_to(root):
        raise MMH3ResourceError("Recovery prefix is outside the selected output root")
    matches = []
    for index, path in enumerate(prefix.parent.glob(prefix.name + "*.mmh3")):
        if index >= 100:
            raise MMH3ResourceError("Too many recovery artifacts; inspect this attempt manually")
        if path.resolve() != path:
            continue
        packet = load_archive(path, verify="full")
        receipt = packet.manifest.get("extensions", {}).get("mmh3_media", {}).get("execution_receipt")
        if receipt == submission["receipt"]:
            matches.append(str(path))
    if len(matches) > 1:
        raise MMH3ResourceError("Multiple saved artifacts own this attempt; select the intended result manually")
    return matches[0] if matches else None


class ComfyQueueClient:
    def __init__(self, server: str, *, timeout: float = 30.0):
        self.server = server.rstrip("/")
        self.timeout = float(timeout)
        self.client_id = uuid.uuid4().hex

    def _json(self, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.server + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise QueueHTTPError(exc.code, f"ComfyUI HTTP {exc.code} for {path}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise MMH3ResourceError(f"ComfyUI request failed for {path}: {exc}") from exc

    def queue(self, workflow: Mapping[str, Any]) -> str:
        response = self._json("/prompt", {"prompt": workflow, "client_id": self.client_id})
        prompt_id = response.get("prompt_id") if isinstance(response, Mapping) else None
        if not prompt_id:
            raise MMH3ResourceError(f"ComfyUI rejected prompt: {response}")
        return str(prompt_id)

    def queue_owned(self, workflow, submission):
        try:
            response = self._json("/prompt", {"prompt": workflow, "client_id": self.client_id,
                "prompt_id": submission["prompt_id"], "extra_data": {"mmh3_submission": submission["token"]}})
        except QueueHTTPError as exc:
            if 400 <= exc.status < 500 and exc.status not in {408, 429}:
                raise RemoteJobFailed(str(exc)) from exc
            raise
        if not isinstance(response, Mapping) or not response.get("prompt_id"):
            raise RemoteJobFailed(f"ComfyUI rejected prompt: {response}")
        return str(response["prompt_id"])

    def locate_submission(self, submission):
        if submission.get("server") != self.server:
            raise MMH3ResourceError("Checkpoint belongs to another ComfyUI server")
        # Queue tuples and history both retain extra_data. The intent token survives
        # a lost POST response even on hosts that assign their own prompt IDs.
        queue = self._json("/queue")
        items = [("running", item) for item in queue.get("queue_running", [])]
        items += [("pending", item) for item in queue.get("queue_pending", [])]
        history = self._json("/history")
        items += [("history", entry.get("prompt")) for entry in history.values() if isinstance(entry, Mapping)]
        matches = [(state, str(item[1])) for state, item in items
                   if isinstance(item, (list, tuple)) and len(item) > 3
                   and isinstance(item[3], Mapping) and item[3].get("mmh3_submission") == submission["token"]]
        ids = {pid for _, pid in matches}
        if len(ids) > 1:
            raise MMH3ResourceError("Multiple remote prompts match this attempt; refusing automatic retry")
        if matches:
            state, pid = matches[0]
            return state, pid
        return "missing", submission["prompt_id"]

    def wait_owned(self, submission, *, poll_seconds, timeout_seconds, control=None):
        deadline = time.monotonic() + timeout_seconds
        cancelling = bool(submission.get("cancel_requested"))
        prompt_id = submission["prompt_id"]
        while time.monotonic() < deadline:
            response = self._json(f"/history/{prompt_id}")
            entry = response.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                # Completed output wins a cancellation race and is preserved.
                if status.get("status_str") == "error" or status.get("completed") is False:
                    if cancelling:
                        raise RunnerCancelled("Owned prompt cancelled")
                    raise RemoteJobFailed(f"ComfyUI job failed: {status}")
                return entry
            if control and control() == "cancel":
                cancelling = True
            if cancelling:
                # Never use global /interrupt: old hosts could cancel another project.
                self._json(f"/api/jobs/{prompt_id}/cancel", {})
            queue = self._json("/queue")
            present = any(isinstance(item, (list, tuple)) and len(item) > 1 and str(item[1]) == prompt_id
                          for key in ("queue_running", "queue_pending") for item in queue.get(key, []))
            if not present:
                # Recheck history after the queue snapshot to cover completion races.
                response = self._json(f"/history/{prompt_id}")
                if prompt_id in response:
                    continue
                if cancelling:
                    raise RunnerCancelled("Owned prompt removed from queue")
                raise MMH3ResourceError("Prompt is absent from queue/history; keep checkpoint for reconciliation")
            time.sleep(max(0.1, poll_seconds))
        raise MMH3ResourceError(f"Remote state retained after {timeout_seconds:g}s timeout; resume this checkpoint")

    def wait(self, prompt_id: str, *, poll_seconds: float, timeout_seconds: float) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            response = self._json(f"/history/{prompt_id}")
            if isinstance(response, Mapping) and prompt_id in response:
                entry = response[prompt_id]
                status = entry.get("status", {}) if isinstance(entry, Mapping) else {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    raise MMH3ResourceError(f"ComfyUI job failed: {status}")
                return entry
            time.sleep(max(0.1, poll_seconds))
        raise MMH3ResourceError(f"ComfyUI job timed out after {timeout_seconds:g}s")


def run_execution_ledger(
    ledger: Mapping[str, Any],
    workflow_template: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    client: Any,
    save_node_id: str = "",
    resume_mode: str = "pending_and_failed",
    continue_on_error: bool = False,
    poll_seconds: float = 1.0,
    timeout_seconds: float = 86400.0,
    on_event: Callable[[str], None] = print,
    control: Callable[[], str] | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    from .project_actions import _publication_lock
    path = Path(checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Two runner processes must not submit the same immutable attempt concurrently.
    with _publication_lock(path.with_name(path.name + ".runner.lock")):
        if path.exists():
            saved = load_execution_ledger(path)
            if any(saved.get(key) != ledger.get(key) for key in ("plan_sha256", "effective_settings_sha256", "operation")):
                raise MMH3ResourceError("Checkpoint belongs to another plan/settings; use a separate checkpoint")
            if saved["revision"] >= ledger["revision"]:
                ledger = saved
        return _run_execution_ledger(ledger, workflow_template, checkpoint_path=path, client=client,
            save_node_id=save_node_id, resume_mode=resume_mode, continue_on_error=continue_on_error,
            poll_seconds=poll_seconds, timeout_seconds=timeout_seconds, on_event=on_event, control=control,
            output_root=output_root)


def _run_execution_ledger(ledger, workflow_template, *, checkpoint_path, client, save_node_id,
                          resume_mode, continue_on_error, poll_seconds, timeout_seconds, on_event, control, output_root):
    current = deep_copy_json(dict(ledger))
    attempted: set[str] = set()

    def record(job_id, submission):
        entry = next(job for job in current["jobs"] if job["job_id"] == job_id)
        entry["submission"] = deep_copy_json(submission)
        current["revision"] += 1
        save_execution_ledger(current, checkpoint_path)

    while True:
        running = [job for job in current["jobs"] if job["state"] == "running"]
        recovering = bool(running)
        if len(running) > 1:
            raise MMH3ResourceError("Multiple running jobs require reconciliation before sequential execution")
        if recovering:
            entry = running[0]
            submission = entry.get("submission")
            if not submission:
                raise MMH3ResourceError("Legacy running job has no queue receipt; inspect ComfyUI before resetting it")
            lease = {"job_id": entry["job_id"], "lease_id": entry["lease_id"], "attempt": entry["attempts"]}
        else:
            if control and control() in {"pause", "cancel"}:
                on_event("PAUSED before next segment")
                return current
            current, lease = acquire_next_execution_job(
                current, mode=resume_mode, exclude_job_ids=tuple(attempted), lease_timeout_seconds=timeout_seconds
            )
        if lease is None:
            return current
        job_id = lease["job_id"]
        attempted.add(job_id)
        owned_client = callable(getattr(client, "queue_owned", None))
        dispatched = recovering
        if not recovering:
            try:
                workflow = materialize_job_workflow(workflow_template, current, lease)
                submission = {"contract": "mmh3_queue_submission_v1", "prompt_id": str(uuid.uuid4()),
                    "token": uuid.uuid4().hex, "server": getattr(client, "server", ""),
                    "lease_id": lease["lease_id"], "attempt": lease["attempt"], "filename_prefix": lease["filename_prefix"]}
                submission["receipt"] = {"token": submission["token"], "job_id": job_id, "lease_id": lease["lease_id"],
                    "attempt": lease["attempt"], "plan_sha256": current["plan_sha256"],
                    "settings_sha256": current["effective_settings_sha256"]}
                workflow = stamp_submission_workflow(workflow, submission["receipt"])
                submission["workflow_sha256"] = hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest()
            except Exception as exc:
                current = transition_execution_job(current, job_id, "fail", error=str(exc), lease_id=lease["lease_id"])
                save_execution_ledger(current, checkpoint_path)
                if continue_on_error:
                    continue
                return current
            record(job_id, submission)  # Durable intent precedes POST.
        try:
            if recovering and owned_client and submission.get("server") != client.server:
                raise MMH3ResourceError("Checkpoint belongs to another ComfyUI server")
            if recovering and not submission.get("artifact_path") and owned_client:
                state, prompt_id = client.locate_submission(submission)
                if state == "missing":
                    artifact = (recover_saved_artifact(output_root, submission)
                                if output_root and submission.get("receipt") else None)
                    if not artifact:
                        on_event(f"RECONCILE {job_id}: remote queue/history has no receipt; no duplicate was submitted")
                        return current
                    submission["artifact_path"] = artifact
                else:
                    submission["prompt_id"] = prompt_id
                record(job_id, submission)
                on_event(f"RESUME {job_id} prompt={prompt_id}")
            elif not recovering:
                on_event(f"QUEUE {job_id} attempt={lease['attempt']}")
                dispatched = True  # A lost response is not evidence of rejection.
                submission["prompt_id"] = (client.queue_owned(workflow, submission) if owned_client else client.queue(workflow))
                # Persist outside failure handling: a disk error must keep a running
                # intent, never mark an already queued prompt as retryable.
                try:
                    record(job_id, submission)
                except Exception:
                    # Disk persistence failed after dispatch, even with a legacy
                    # client. Keep the durable pre-POST intent recoverable.
                    on_event(f"RECONCILE {job_id}: queue receipt could not be persisted")
                    raise RemoteStateUncertain("Queue receipt persistence failed after dispatch")
            if submission.get("artifact_path"):
                packet_path = submission["artifact_path"]
            else:
                def check_control():
                    action = control() if control else "run"
                    if action == "cancel" and not submission.get("cancel_requested"):
                        submission["cancel_requested"] = True
                        record(job_id, submission)
                    return action
                history = (client.wait_owned(submission, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds,
                    control=check_control) if owned_client else
                    client.wait(submission["prompt_id"], poll_seconds=poll_seconds, timeout_seconds=timeout_seconds))
                packet_path = find_saved_mmh3(history, save_node_id)
                submission["artifact_path"] = packet_path
                record(job_id, submission)
            if (current.get("effective_settings") or {}).get("review_policy") == "candidate_required":
                current = commit_execution_candidate(
                    current, job_id, lease_id=lease["lease_id"], packet_path=packet_path
                )
            else:
                current = commit_execution_artifact(
                    current, job_id, lease_id=lease["lease_id"], packet_path=packet_path
                )
        except (Exception, KeyboardInterrupt) as exc:
            if isinstance(exc, (KeyboardInterrupt, RemoteStateUncertain)) or (owned_client and dispatched
                    and not isinstance(exc, (RemoteJobFailed, RunnerCancelled))):
                # Network errors, timeout, disk failure and process interruption are
                # uncertainty, not remote failure. Never blindly queue another attempt.
                on_event(f"RECONCILE {job_id}: {exc}; running receipt retained")
                if isinstance(exc, KeyboardInterrupt):
                    raise
                return current
            current = transition_execution_job(
                current, job_id, "cancel" if isinstance(exc, RunnerCancelled) else "fail",
                error=f"{type(exc).__name__}: {exc}", lease_id=lease["lease_id"]
            )
            save_execution_ledger(current, checkpoint_path)
            on_event(f"FAILED {job_id}: {exc}")
            if not continue_on_error or isinstance(exc, (KeyboardInterrupt, RunnerCancelled)):
                if isinstance(exc, KeyboardInterrupt):
                    raise
                return current
        else:
            # The artifact is committed. Persistence/reporting failures must propagate
            # without attempting an invalid completed -> fail transition.
            save_execution_ledger(current, checkpoint_path)
            on_event(f"DONE {job_id} -> {packet_path}")


def format_execution_summary(summary: Mapping[str, Any]) -> str:
    counts = summary.get("counts", {})
    aggregates = summary.get("aggregates", {})
    assembly = summary.get("assembly", {})
    success = float(aggregates.get("success_rate") or 0.0) * 100.0
    parts = [
        f"completed={counts.get('completed', 0)}",
        f"review={counts.get('review', 0)}",
        f"failed={counts.get('failed', 0)}",
        f"cancelled={counts.get('cancelled', 0)}",
        f"pending={counts.get('pending', 0)}",
        f"running={counts.get('running', 0)}",
        f"success={success:.1f}%",
    ]
    if aggregates.get("frames_processed") is not None:
        parts.append(f"frames={aggregates['frames_processed']}")
    if aggregates.get("pcm_samples_processed") is not None:
        parts.append(f"pcm={aggregates['pcm_samples_processed']}")
    parts.extend([
        f"issues={len(summary.get('issues', []))}",
        f"assembly={'ready' if assembly.get('ready') else 'blocked'}",
        f"next={summary.get('recommendation', 'inspect')}",
    ])
    return "SUMMARY " + " ".join(parts)


def summarize_execution_ledger(ledger: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    summary = build_execution_summary(ledger)
    return summary, format_execution_summary(summary)


__all__ = [
    "ComfyQueueClient",
    "PLACEHOLDERS",
    "find_saved_mmh3",
    "format_execution_summary",
    "materialize_assembly_workflow",
    "materialize_job_workflow",
    "run_execution_ledger",
    "run_final_assembly",
    "summarize_execution_ledger",
]
