from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

from .errors import MMH3FormatError

_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def deep_copy_json(value: Any) -> Any:
    return copy.deepcopy(value)


def json_dumps_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False


def sha256_file(path: str | os.PathLike, chunk_size: int = 8 * 1024 * 1024) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def copy_and_hash(src: BinaryIO, dst: BinaryIO, chunk_size: int = 8 * 1024 * 1024) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break
        dst.write(chunk)
        h.update(chunk)
        size += len(chunk)
    return h.hexdigest(), size


def safe_member_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise MMH3FormatError("Resource path must be a non-empty string")
    p = PurePosixPath(path.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or any(part in ("", ".") for part in p.parts):
        raise MMH3FormatError(f"Unsafe archive member path: {path!r}")
    # Windows drive/UNC-like prefixes are not valid ZIP logical paths either.
    if ":" in p.parts[0]:
        raise MMH3FormatError(f"Unsafe archive member path: {path!r}")
    return p.as_posix()


def safe_filename_component(text: str, fallback: str = "resource") -> str:
    out = _ID_SAFE.sub("_", (text or "").strip()).strip("._")
    return out[:120] or fallback


_CACHE_SESSION_ROOT: Path | None = None
_CACHE_STALE_SECONDS = 7 * 24 * 60 * 60


def _cache_base_root() -> Path:
    env = os.getenv("MMH3_MEDIA_CACHE")
    if env:
        return Path(env)
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_temp_directory()) / "mmh3_media_cache"
    except Exception:
        return Path(tempfile.gettempdir()) / "mmh3_media_cache"


def _cleanup_stale_cache_sessions(base: Path) -> None:
    # Never prune a fresh session directory: VideoFromFile may retain a path and read it later.
    # Only abandoned/old process caches are eligible here.
    now = datetime.now(timezone.utc).timestamp()
    try:
        entries = list(base.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir() or not entry.name.startswith("session-"):
            continue
        try:
            age = now - entry.stat().st_mtime
            if age >= _CACHE_STALE_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def get_cache_root() -> Path:
    global _CACHE_SESSION_ROOT
    if _CACHE_SESSION_ROOT is not None:
        return _CACHE_SESSION_ROOT
    base = _cache_base_root()
    base.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_cache_sessions(base)
    session = base / f"session-{os.getpid()}-{uuid.uuid4().hex[:10]}"
    session.mkdir(parents=True, exist_ok=True)
    _CACHE_SESSION_ROOT = session

    def _cleanup_current_session() -> None:
        shutil.rmtree(session, ignore_errors=True)

    atexit.register(_cleanup_current_session)
    return session


def merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7396 JSON Merge Patch."""
    if not isinstance(patch, dict):
        return deep_copy_json(patch)
    if not isinstance(target, dict):
        target = {}
    else:
        target = deep_copy_json(target)
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict):
            target[key] = merge_patch(target.get(key), value)
        else:
            target[key] = deep_copy_json(value)
    return target


def iter_files(root: Path, suffix: str) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob(f"*{suffix}") if p.is_file())
