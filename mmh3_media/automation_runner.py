from __future__ import annotations

import json
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
    save_execution_ledger,
    transition_execution_job,
)
from .errors import MMH3ResourceError
from .util import deep_copy_json


PLACEHOLDERS = {
    "__MMH3_PLAN_JSON__": "plan_json",
    "__MMH3_JOB_ID__": "job_id",
    "__MMH3_LEASE_ID__": "lease_id",
    "__MMH3_FILENAME_PREFIX__": "filename_prefix",
    "__MMH3_EFFECTIVE_SETTINGS_JSON__": "effective_settings_json",
}


def materialize_job_workflow(template: Mapping[str, Any], ledger: Mapping[str, Any], lease: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "plan_json": json.dumps(ledger["plan"], ensure_ascii=False),
        "job_id": str(lease["job_id"]),
        "lease_id": str(lease["lease_id"]),
        "filename_prefix": str(lease["filename_prefix"]),
        "effective_settings_json": json.dumps(ledger.get("effective_settings", {}), ensure_ascii=False),
    }

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str) and value in PLACEHOLDERS:
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
    missing = [placeholder for placeholder in ("__MMH3_PLAN_JSON__", "__MMH3_JOB_ID__", "__MMH3_FILENAME_PREFIX__") if placeholder not in json.dumps(template)]
    if missing:
        raise MMH3ResourceError("Workflow template is missing required placeholders: " + ", ".join(missing))
    return result


def materialize_assembly_workflow(template: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    if ledger.get("status") != "completed":
        raise MMH3ResourceError("Final assembly requires a completed execution ledger")
    ledger_json = json.dumps(ledger, ensure_ascii=False)

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if value == "__MMH3_LEDGER_JSON__":
            return ledger_json
        return value

    serialized = json.dumps(template)
    if "__MMH3_LEDGER_JSON__" not in serialized:
        raise MMH3ResourceError("Assembly workflow template is missing __MMH3_LEDGER_JSON__")
    return replace(deep_copy_json(dict(template)))


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
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise MMH3ResourceError(f"ComfyUI request failed for {path}: {exc}") from exc

    def queue(self, workflow: Mapping[str, Any]) -> str:
        response = self._json("/prompt", {"prompt": workflow, "client_id": self.client_id})
        prompt_id = response.get("prompt_id") if isinstance(response, Mapping) else None
        if not prompt_id:
            raise MMH3ResourceError(f"ComfyUI rejected prompt: {response}")
        return str(prompt_id)

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
) -> dict[str, Any]:
    current = deep_copy_json(dict(ledger))
    attempted: set[str] = set()
    while True:
        current, lease = acquire_next_execution_job(
            current, mode=resume_mode, exclude_job_ids=tuple(attempted), lease_timeout_seconds=timeout_seconds
        )
        if lease is None:
            return current
        save_execution_ledger(current, checkpoint_path)
        job_id = lease["job_id"]
        attempted.add(job_id)
        try:
            workflow = materialize_job_workflow(workflow_template, current, lease)
            on_event(f"QUEUE {job_id} attempt={lease['attempt']}")
            prompt_id = client.queue(workflow)
            history = client.wait(prompt_id, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
            packet_path = find_saved_mmh3(history, save_node_id)
            current = commit_execution_artifact(
                current, job_id, lease_id=lease["lease_id"], packet_path=packet_path
            )
            save_execution_ledger(current, checkpoint_path)
            on_event(f"DONE {job_id} -> {packet_path}")
        except (Exception, KeyboardInterrupt) as exc:
            current = transition_execution_job(
                current, job_id, "fail", error=f"{type(exc).__name__}: {exc}", lease_id=lease["lease_id"]
            )
            save_execution_ledger(current, checkpoint_path)
            on_event(f"FAILED {job_id}: {exc}")
            if not continue_on_error or isinstance(exc, KeyboardInterrupt):
                if isinstance(exc, KeyboardInterrupt):
                    raise
                return current


def format_execution_summary(summary: Mapping[str, Any]) -> str:
    counts = summary.get("counts", {})
    aggregates = summary.get("aggregates", {})
    assembly = summary.get("assembly", {})
    success = float(aggregates.get("success_rate") or 0.0) * 100.0
    parts = [
        f"completed={counts.get('completed', 0)}",
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
