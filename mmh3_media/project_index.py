"""Small atomic review index; accepted .mmh3 files remain portable snapshots."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .archive import _durable_replace, load_archive
from .errors import MMH3ResourceError
from .project_review import build_project_review_state
from .util import deep_copy_json

CONTRACT = "mmh3_project_index_v1"
FIELDS = ("project_candidates", "project_segment_sources")


def index_path(archive):
    path = Path(archive).with_suffix(".index.json")
    if path.resolve() != path:
        raise MMH3ResourceError("Project index must stay inside its managed directory")
    return path


def _anchor(packet):
    for key in FIELDS:
        packet = packet.set_extension_value("mmh3_media", key, None)
    return build_project_review_state(packet).state_digest


def load_project_archive(path, *, verify="manifest"):
    packet = load_archive(path, verify=verify)
    sidecar = index_path(path)
    if not sidecar.exists():
        return packet  # Existing projects migrate on their first review edit.
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise MMH3ResourceError("Project index cannot be read; preserve it before recovery") from exc
    if (not isinstance(value, dict) or value.get("contract") != CONTRACT
            or not isinstance(value.get("anchor"), str) or not re.fullmatch(r"[0-9a-f]{64}", value["anchor"])):
        raise MMH3ResourceError("Unsupported project review index")
    # Acceptance atomically publishes an archive containing the latest index state.
    # A crash before removing/updating the old index cannot undo that publication.
    if value.get("anchor") != _anchor(packet):
        return packet
    fields = value.get("fields")
    if not isinstance(fields, dict) or set(fields) != set(FIELDS):
        raise MMH3ResourceError("Malformed project index fields")
    records = fields["project_candidates"]
    refs = fields["project_segment_sources"]
    if (not isinstance(records, list) or any(not isinstance(r, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(r.get("id", "")))
            or r.get("status") not in {"review", "rejected", "accepted", "trashed"}
            or type(r.get("selected")) is not bool for r in records)
            or len({r["id"] for r in records}) != len(records)
            or not isinstance(refs, list)
            or any(not isinstance(r, str) or not re.fullmatch(r"[0-9a-f]{64}", r) for r in refs)):
        raise MMH3ResourceError("Malformed project review index")
    for key in FIELDS:
        packet = packet.set_extension_value("mmh3_media", key, fields[key])
    return packet


def save_project_index(path, packet):
    """Caller holds the same publication lock as archive acceptance."""
    target = index_path(path)
    namespace = packet.manifest.get("extensions", {}).get("mmh3_media", {})
    value = {"contract": CONTRACT, "anchor": _anchor(packet),
             "fields": {key: deep_copy_json(namespace.get(key) or []) for key in FIELDS}}
    fd, temporary = tempfile.mkstemp(prefix=".review_", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _durable_replace(Path(temporary), target)
    finally:
        Path(temporary).unlink(missing_ok=True)
