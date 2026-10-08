"""Atomic control requests shared by a running CLI and a second terminal."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from .errors import MMH3ResourceError

# Windows briefly denies access while the other terminal replaces the file.
_SHARING_RETRIES = 20
_SHARING_DELAY = 0.05


def _retry_sharing(operation):
    for attempt in range(_SHARING_RETRIES):
        try:
            return operation()
        except PermissionError:
            if attempt == _SHARING_RETRIES - 1:
                raise
            time.sleep(_SHARING_DELAY)


def read_runner_control(path):
    path = Path(path)
    try:
        text = _retry_sharing(lambda: path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "run"
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise MMH3ResourceError("Runner control file is not valid JSON") from exc
    action = value.get("action") if isinstance(value, dict) else None
    if action not in {"run", "pause", "cancel"}:
        raise MMH3ResourceError("Runner control must request run, pause or cancel")
    return action


def write_runner_control(path, action):
    if action not in {"run", "pause", "cancel"}:
        raise MMH3ResourceError("Unknown runner control action")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".control_", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"action": action}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _retry_sharing(lambda: os.replace(temporary, target))
    finally:
        Path(temporary).unlink(missing_ok=True)
