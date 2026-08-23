from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

from .constants import (
    FORMAT_NAME,
    ORDERED_ROLES,
    SCHEMA_VERSION,
    SINGLETON_ROLES,
)
from .errors import MMH3ResourceError
from .schema import validate_role_kind, validate_verify_mode
from .validation import require_valid_manifest
from .util import deep_copy_json, merge_patch, utc_now_iso


def _resource_id() -> str:
    return "res_" + uuid.uuid4().hex


def _packet_id() -> str:
    return "mmh3_" + uuid.uuid4().hex


def _placeholder_path(kind: str, role: str, slot: int, rid: str) -> str:
    # Writer canonicalizes paths at save time. This placeholder is always safe and unique.
    return f"pending/{role}_{slot:03d}_{rid}.{kind}"


@dataclass(frozen=True)
class MMH3Media:
    manifest: dict[str, Any]
    source_archive: str | None = None
    payloads: dict[str, Any] = field(default_factory=dict, repr=False)
    origins: dict[str, str] = field(default_factory=dict, repr=False)
    verify_mode: str = "on_access"
    dirty: bool = True

    def __post_init__(self):
        validate_verify_mode(self.verify_mode)
        require_valid_manifest(self.manifest)

    @staticmethod
    def create(name: str = "", generation: dict[str, Any] | None = None) -> "MMH3Media":
        now = utc_now_iso()
        manifest = {
            "format": FORMAT_NAME,
            "schema_version": SCHEMA_VERSION,
            "id": _packet_id(),
            "name": name or "untitled",
            "created_at": now,
            "updated_at": now,
            "generation": deep_copy_json(generation or {}),
            "notes": "",
            "resources": [],
            "history": [],
            "extensions": {},
        }
        return MMH3Media(manifest=manifest, dirty=True)

    def _copy(self, *, dirty: bool = True) -> tuple[dict, dict, dict]:
        return copy.deepcopy(self.manifest), dict(self.payloads), dict(self.origins)

    def resources(self) -> list[dict[str, Any]]:
        return self.manifest["resources"]

    def get_by_id(self, resource_id: str) -> dict[str, Any] | None:
        for res in self.resources():
            if res["id"] == resource_id:
                return res
        return None

    def get_by_role_slot(self, role: str, slot: int = 0) -> dict[str, Any] | None:
        for res in self.resources():
            if res["role"] == role and res["slot"] == slot:
                return res
        return None

    def select(self, *, role: str | None = None, slot: int = 0, resource_id: str = "") -> dict[str, Any] | None:
        if resource_id:
            return self.get_by_id(resource_id)
        if role is None:
            raise MMH3ResourceError("Either role or resource_id is required")
        return self.get_by_role_slot(role, slot)

    def _append_history(self, manifest: dict, op: str, **fields: Any) -> None:
        entry = {"op": op, "timestamp": utc_now_iso()}
        entry.update(deep_copy_json(fields))
        manifest.setdefault("history", []).append(entry)

    def _normalize_slots(self, manifest: dict, role: str) -> None:
        if role not in ORDERED_ROLES:
            return
        same = sorted((r for r in manifest["resources"] if r["role"] == role), key=lambda r: r["slot"])
        for i, res in enumerate(same):
            res["slot"] = i

    def put(
        self,
        payload: Any,
        *,
        kind: str,
        role: str,
        slot: int = 0,
        mode: str = "upsert",
        metadata: dict[str, Any] | None = None,
        record_history: bool = True,
    ) -> "MMH3Media":
        packet, _ = self.put_with_id(
            payload, kind=kind, role=role, slot=slot, mode=mode, metadata=metadata, record_history=record_history
        )
        return packet

    def put_with_id(
        self,
        payload: Any,
        *,
        kind: str,
        role: str,
        slot: int = 0,
        mode: str = "upsert",
        metadata: dict[str, Any] | None = None,
        record_history: bool = True,
    ) -> tuple["MMH3Media", str]:
        validate_role_kind(role, kind)
        from .serializers import validate_payload_contract
        validate_payload_contract(payload, kind, role)
        if mode not in ("upsert", "add", "replace"):
            raise MMH3ResourceError("Put mode must be upsert, add, or replace")
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
            raise MMH3ResourceError("slot must be a non-negative integer")
        if role in SINGLETON_ROLES:
            slot = 0

        manifest, payloads, origins = self._copy()
        same = sorted((r for r in manifest["resources"] if r["role"] == role), key=lambda r: r["slot"])
        existing = next((r for r in same if r["slot"] == slot), None)

        if mode == "add":
            if role in SINGLETON_ROLES and same:
                raise MMH3ResourceError(f"Role {role!r} is singleton; use replace/upsert")
            if role in ORDERED_ROLES:
                insert_at = min(slot, len(same))
                for r in same:
                    if r["slot"] >= insert_at:
                        r["slot"] += 1
                slot = insert_at
                existing = None
            elif existing is not None:
                raise MMH3ResourceError(f"Resource {role}[{slot}] already exists; use replace/upsert")
        elif mode == "replace" and existing is None:
            raise MMH3ResourceError(f"Resource {role}[{slot}] does not exist; replace requires an existing resource")
        elif mode == "upsert" and existing is None and role in ORDERED_ROLES:
            slot = min(slot, len(same))
            for r in same:
                if r["slot"] >= slot:
                    r["slot"] += 1

        if existing is None:
            rid = _resource_id()
            res = {
                "id": rid,
                "kind": kind,
                "role": role,
                "slot": slot,
                "path": _placeholder_path(kind, role, slot, rid),
                "metadata": deep_copy_json(metadata or {}),
            }
            manifest["resources"].append(res)
            action = "add"
        else:
            rid = existing["id"]
            existing["kind"] = kind
            existing["metadata"] = deep_copy_json(metadata if metadata is not None else existing.get("metadata", {}))
            existing.pop("sha256", None)
            existing.pop("size", None)
            existing.pop("serializer", None)
            existing.pop("media_type", None)
            res = existing
            action = "replace"
            origins.pop(rid, None)

        payloads[rid] = payload
        self._normalize_slots(manifest, role)
        manifest["updated_at"] = utc_now_iso()
        if record_history:
            self._append_history(manifest, "resource_put", action=action, resource_id=rid, role=role, slot=res["slot"], kind=kind)
        require_valid_manifest(manifest)
        return MMH3Media(manifest, self.source_archive, payloads, origins, self.verify_mode, True), rid

    def remove(
        self,
        *,
        role: str | None = None,
        slot: int = 0,
        resource_id: str = "",
        missing: str = "error",
        record_history: bool = True,
    ) -> "MMH3Media":
        target = self.select(role=role, slot=slot, resource_id=resource_id)
        if target is None:
            if missing == "ignore":
                return self
            label = resource_id or f"{role}[{slot}]"
            raise MMH3ResourceError(f"Resource {label} does not exist")
        manifest, payloads, origins = self._copy()
        rid = target["id"]
        removed = next(r for r in manifest["resources"] if r["id"] == rid)
        manifest["resources"] = [r for r in manifest["resources"] if r["id"] != rid]
        payloads.pop(rid, None)
        origins.pop(rid, None)
        self._normalize_slots(manifest, removed["role"])
        manifest["updated_at"] = utc_now_iso()
        if record_history:
            self._append_history(
                manifest,
                "resource_remove",
                resource_id=rid,
                role=removed["role"],
                slot=removed["slot"],
                kind=removed["kind"],
            )
        require_valid_manifest(manifest)
        return MMH3Media(manifest, self.source_archive, payloads, origins, self.verify_mode, True)

    def move(self, *, role: str, from_slot: int, to_slot: int, record_history: bool = True) -> "MMH3Media":
        if role not in ORDERED_ROLES:
            raise MMH3ResourceError(
                f"Role {role!r} is not an ordered collection. Move is intended for refs/masks/continuation contexts."
            )
        manifest, payloads, origins = self._copy()
        same = sorted((r for r in manifest["resources"] if r["role"] == role), key=lambda r: r["slot"])
        if from_slot < 0 or from_slot >= len(same):
            raise MMH3ResourceError(f"{role}[{from_slot}] does not exist")
        to_slot = max(0, min(to_slot, len(same) - 1))
        item = same.pop(from_slot)
        same.insert(to_slot, item)
        for i, res in enumerate(same):
            res["slot"] = i
        manifest["updated_at"] = utc_now_iso()
        if record_history and from_slot != to_slot:
            self._append_history(manifest, "resource_move", resource_id=item["id"], role=role, from_slot=from_slot, to_slot=to_slot)
        require_valid_manifest(manifest)
        return MMH3Media(manifest, self.source_archive, payloads, origins, self.verify_mode, True)

    def replace_resource_metadata(
        self,
        resource_id: str,
        metadata: dict[str, Any],
        *,
        record_history: bool = True,
    ) -> "MMH3Media":
        if not isinstance(metadata, dict):
            raise MMH3ResourceError("Resource metadata must be an object")
        target = self.get_by_id(resource_id)
        if target is None:
            raise MMH3ResourceError(f"Resource {resource_id!r} does not exist")
        manifest, payloads, origins = self._copy()
        res = next(r for r in manifest["resources"] if r["id"] == resource_id)
        res["metadata"] = deep_copy_json(metadata)
        manifest["updated_at"] = utc_now_iso()
        if record_history:
            self._append_history(
                manifest,
                "resource_metadata_edit",
                resource_id=resource_id,
                role=res["role"],
                slot=res["slot"],
                kind=res["kind"],
            )
        require_valid_manifest(manifest)
        return MMH3Media(manifest, self.source_archive, payloads, origins, self.verify_mode, True)

    def edit_metadata(
        self,
        *,
        name: str = "",
        task: str = "",
        prompt: str = "",
        seed: int = -1,
        tags: str = "",
        notes: str | None = None,
        merge_patch_json: dict[str, Any] | None = None,
        record_history: bool = False,
    ) -> "MMH3Media":
        manifest, payloads, origins = self._copy()
        if name:
            manifest["name"] = name
        gen = manifest.setdefault("generation", {})
        if task:
            gen["task"] = task
        if prompt:
            gen["prompt"] = prompt
        if seed >= 0:
            gen["seed"] = int(seed)
        if tags:
            manifest["tags"] = [x.strip() for x in tags.split(",") if x.strip()]
        if notes is not None:
            manifest["notes"] = str(notes)
        if merge_patch_json:
            if not isinstance(merge_patch_json, dict):
                raise MMH3ResourceError("custom JSON merge patch must be an object")
            # Metadata editing must never be able to rewrite resource ownership/history or core identity.
            # Make this a hard contract instead of silently restoring reserved fields after the patch: silent
            # restoration would make a malformed automation look as if it succeeded.
            reserved = {"format", "schema_version", "id", "created_at", "resources", "history"}
            touched = sorted(reserved.intersection(merge_patch_json))
            if touched:
                raise MMH3ResourceError(
                    "custom JSON patch cannot modify reserved packet fields: " + ", ".join(touched)
                )
            manifest = merge_patch(manifest, merge_patch_json)
            manifest["format"] = FORMAT_NAME
            manifest["schema_version"] = SCHEMA_VERSION
            manifest["id"] = self.manifest["id"]
            manifest.setdefault("extensions", {})
            manifest.setdefault("generation", {})
        manifest["updated_at"] = utc_now_iso()
        if record_history:
            self._append_history(manifest, "metadata_edit")
        require_valid_manifest(manifest)
        return MMH3Media(manifest, self.source_archive, payloads, origins, self.verify_mode, True)

    def with_saved_state(self, *, manifest: dict, source_archive: str) -> "MMH3Media":
        origins = {r["id"]: r["path"] for r in manifest["resources"]}
        return MMH3Media(
            manifest=deep_copy_json(manifest),
            source_archive=source_archive,
            payloads={},
            origins=origins,
            verify_mode=self.verify_mode,
            dirty=False,
        )
