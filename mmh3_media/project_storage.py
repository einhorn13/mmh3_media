"""Conservative storage inventory and reversible candidate retirement.

No archive members, accepted revisions, snapshots or unknown files are deleted.
Whole-file moves keep recovery possible even if metadata publication is interrupted.
"""
from __future__ import annotations

import hashlib
import os

from .archive import load_archive, save_archive
from .errors import MMH3ResourceError
from .project_actions import _managed_directory, _require_digest, project_archive_lock
from .project_review import build_project_review_state
from .util import json_dumps_canonical, sha256_file


def storage_report(manager, project_id):
    directory = manager.directory(project_id)
    current = manager.current(project_id)
    packet = load_archive(current, verify="manifest")
    records = manager._records(packet)
    namespace = packet.manifest["extensions"]["mmh3_media"]
    protected = {item["id"] for item in records if item.get("selected") or item["status"] == "accepted"}
    for receipt in namespace.get("project_publications", []):
        protected.add(receipt.get("request", {}).get("candidate_sha256"))
    files = []
    totals = {"current": 0, "candidates": 0, "snapshots": 0, "trash": 0, "other": 0}
    # Do not follow junctions or symlinks, including directories.
    pending = [directory]
    while pending:
        parent = pending.pop()
        for path in sorted(parent.iterdir()):
            if path.resolve() != path or path.is_symlink():
                raise MMH3ResourceError("Storage inventory contains an unsafe link; no cleanup is permitted")
            if path.is_dir():
                pending.append(path)
                continue
            if not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            if relative.startswith(".mmh3_review/") and path.name == "publication.lock":
                continue  # OS coordination, not project data; its first byte may be locked.
            category = ("current" if path == current else "candidates" if path.parent == directory / "candidates"
                        else "trash" if path.parent == directory / "trash"
                        else "snapshots" if relative.startswith(".mmh3_review/") and path.suffix == ".mmh3" else "other")
            digest, size = sha256_file(path)
            totals[category] += size
            files.append({"file": relative, "sha256": digest, "bytes": size, "category": category})
            if category == "snapshots":
                saved = load_archive(path, verify="manifest")
                protected.update(item["id"] for item in manager._records(saved))
    by_name = {item["file"]: item for item in files}
    candidates = []
    for record in records:
        cid = record["id"]
        active = by_name.get(f"candidates/{cid}.mmh3")
        trash = by_name.get(f"trash/{cid}.mmh3")
        if active and trash:
            raise MMH3ResourceError("Candidate exists in both active storage and trash; resolve duplicate before cleanup")
        artifact = active or trash
        valid = artifact is not None and artifact["sha256"] == cid
        candidates.append({"id": cid, "name": record["name"], "bytes": artifact["bytes"] if artifact else 0,
            "in_trash": bool(trash), "protected": cid in protected,
            "can_trash": bool(active and valid and record["status"] == "rejected" and cid not in protected),
            "can_restore": bool(trash and valid), "integrity_ok": valid})
    state_digest = build_project_review_state(packet).state_digest
    digest = hashlib.sha256(json_dumps_canonical({"state": state_digest, "files": files}).encode()).hexdigest()
    return {"state_digest": state_digest, "storage_digest": digest, "bytes": totals, "candidates": candidates,
            "reclaimable_bytes": 0, "policy": "Trash is reversible and does not free disk space. Permanent deletion is unavailable. Snapshots and referenced candidates are protected."}


def move_candidate(manager, project_id, candidate_id, expected, expected_storage, *, restore=False):
    current = manager.current(project_id)
    with project_archive_lock(current):
        report = storage_report(manager, project_id)
        _require_digest(report["state_digest"], expected)
        _require_digest(report["storage_digest"], expected_storage, "Storage inventory")
        item = next((item for item in report["candidates"] if item["id"] == candidate_id), None)
        if item is None or not item["can_restore" if restore else "can_trash"]:
            raise MMH3ResourceError("Candidate cannot be restored" if restore else "Only unreferenced rejected candidates can be moved to trash")
        directory = manager.directory(project_id)
        source = directory / ("trash" if restore else "candidates") / (candidate_id + ".mmh3")
        destination = _managed_directory(directory, "candidates" if restore else "trash") / source.name
        if destination.exists() or destination.is_symlink():
            raise MMH3ResourceError("Storage destination already exists")
        packet = load_archive(current, verify="full")
        records = manager._records(packet)
        record = next(record for record in records if record["id"] == candidate_id)
        record.update(status="rejected" if restore else "trashed", selected=False)
        # A crash here leaves a complete archive in exactly one location. Lookup and
        # inventory inspect both locations so Restore remains available after restart.
        os.rename(source, destination)
        packet = packet.set_extension_value("mmh3_media", "project_candidates", records)
        save_archive(packet, current)
    return manager.state(project_id)
