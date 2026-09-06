from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .constants import FORMAT_NAME, SCHEMA_VERSION
from .errors import MMH3ResourceError
from .schema import validate_role_kind, validate_verify_mode
from .resource_model import make_resource_descriptor, normalize_tags, validate_resource_descriptor
from .validation import require_valid_manifest
from .util import deep_copy_json, is_jsonable, merge_patch, utc_now_iso


def _resource_id() -> str:
    return "res_" + uuid.uuid4().hex


def _packet_id() -> str:
    return "mmh3_" + uuid.uuid4().hex


def _placeholder_path(kind: str, rid: str) -> str:
    # Writer canonicalizes the suffix at save time; IDs keep paths unique and stable.
    return f"pending/{kind}/{rid}.bin"


@dataclass(frozen=True)
class MMH3Media:
    manifest: dict[str, Any]
    source_archive: str | None = None
    payloads: dict[str, Any] = field(default_factory=dict, repr=False)
    origins: dict[str, str] = field(default_factory=dict, repr=False)
    verify_mode: str = "on_access"
    dirty: bool = True
    representation_payloads: dict[str, Any] = field(default_factory=dict, repr=False)
    representation_origins: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        validate_verify_mode(self.verify_mode)
        require_valid_manifest(self.manifest)
        self._validate_primary_bindings()
        from .representations import normalize_representation_index
        normalize_representation_index(self.manifest.get("representations", {}), packet=self)


    @staticmethod
    def _primary_kinds() -> tuple[str, ...]:
        return ("video", "audio", "image", "latent", "mask")

    def _validate_primary_bindings(self) -> None:
        bindings = self.manifest.get("primary")
        if not isinstance(bindings, Mapping):
            raise MMH3ResourceError("primary bindings must be an object")
        for kind, resource_id in bindings.items():
            if kind not in self._primary_kinds():
                raise MMH3ResourceError(f"Unsupported primary kind {kind!r}")
            if not isinstance(resource_id, str) or not resource_id:
                raise MMH3ResourceError(f"Primary binding for {kind!r} must be a resource id")
            resource = self.get_by_id(resource_id)
            if resource is None:
                raise MMH3ResourceError(f"Primary {kind!r} references missing resource {resource_id!r}")
            descriptor = self.ref(resource_id).descriptor
            if descriptor["kind"] != kind:
                raise MMH3ResourceError(
                    f"Primary {kind!r} must reference kind {kind!r}, got {descriptor['kind']!r}"
                )

    def _spawn(
        self,
        manifest: dict[str, Any],
        payloads: dict[str, Any],
        origins: dict[str, str],
        *,
        dirty: bool = True,
        source_archive: str | None = None,
        representation_payloads: dict[str, Any] | None = None,
        representation_origins: dict[str, str] | None = None,
    ) -> "MMH3Media":
        return MMH3Media(
            manifest,
            self.source_archive if source_archive is None else source_archive,
            payloads,
            origins,
            self.verify_mode,
            dirty,
            dict(self.representation_payloads if representation_payloads is None else representation_payloads),
            dict(self.representation_origins if representation_origins is None else representation_origins),
        )

    @property
    def representation_records(self):
        """Return a detached read-only view of representation records by target."""
        from .representations import normalize_representation_index

        normalized = normalize_representation_index(self.manifest.get("representations", {}), packet=self)
        return MappingProxyType({
            target: tuple(deep_copy_json(record) for record in records)
            for target, records in normalized.items()
        })

    def get_representation(self, representation_id: str) -> dict[str, Any] | None:
        if not isinstance(representation_id, str) or not representation_id:
            raise MMH3ResourceError("representation_id must be a non-empty string")
        for records in self.representation_records.values():
            for record in records:
                if record["id"] == representation_id:
                    return deep_copy_json(record)
        return None

    def representations_for(self, target: str, *, kind: str | None = None, fresh_only: bool = False):
        from .representations import list_representations

        if target != "$packet" and self.get_by_id(target) is None:
            raise MMH3ResourceError(f"Representation target {target!r} does not exist")
        return list_representations(self, target, kind=kind, fresh_only=fresh_only)

    def add_representation(
        self,
        record: dict[str, Any],
        payload: Any = None,
        *,
        replace_same_kind: bool = True,
    ) -> "MMH3Media":
        """Return a snapshot with one derived representation attached.

        Representation state is deliberately outside ``resources[]`` and does
        not affect resource identity or primary bindings.
        """
        from .representations import representation_is_fresh, validate_representation_record

        record = deep_copy_json(record)
        validate_representation_record(record, packet=self)
        if not representation_is_fresh(self, record):
            raise MMH3ResourceError("Cannot attach a representation that is already stale for its target")
        manifest, payloads, origins = self._copy()
        index = manifest.setdefault("representations", {})
        existing = list(index.get(record["target"], []))
        removed_ids: set[str] = set()
        if replace_same_kind:
            kept = []
            for item in existing:
                if item.get("kind") == record["kind"]:
                    removed_ids.add(str(item.get("id")))
                else:
                    kept.append(item)
            existing = kept
        if any(item.get("id") == record["id"] for records in index.values() for item in records):
            raise MMH3ResourceError(f"Representation {record['id']!r} already exists")
        existing.append(record)
        index[record["target"]] = existing
        rep_payloads = dict(self.representation_payloads)
        rep_origins = dict(self.representation_origins)
        for rid in removed_ids:
            rep_payloads.pop(rid, None)
            rep_origins.pop(rid, None)
        if payload is not None:
            rep_payloads[record["id"]] = payload
            rep_origins.pop(record["id"], None)
        manifest["updated_at"] = utc_now_iso()
        return self._spawn(
            manifest, payloads, origins,
            representation_payloads=rep_payloads,
            representation_origins=rep_origins,
        )

    def put_representation(
        self,
        *,
        target: str,
        kind: str,
        payload: Any = None,
        media_type: str | None = None,
        generator: str,
        metadata: Mapping[str, Any] | None = None,
        replace_same_kind: bool = True,
    ) -> tuple["MMH3Media", str]:
        from .representations import make_representation_record, representation_source_identity

        revision, digest = representation_source_identity(self, target)
        record = make_representation_record(
            target=target, kind=kind, source_revision=revision, source_digest=digest,
            media_type=media_type, path=None, generator=generator, metadata=metadata,
        )
        return self.add_representation(record, payload, replace_same_kind=replace_same_kind), record["id"]

    def remove_representation(self, representation_id: str, *, missing: str = "error") -> "MMH3Media":
        found = self.get_representation(representation_id)
        if found is None:
            if missing == "ignore":
                return self
            raise MMH3ResourceError(f"Representation {representation_id!r} does not exist")
        manifest, payloads, origins = self._copy()
        index = manifest.setdefault("representations", {})
        target = found["target"]
        remaining = [item for item in index.get(target, []) if item.get("id") != representation_id]
        if remaining:
            index[target] = remaining
        else:
            index.pop(target, None)
        rep_payloads = dict(self.representation_payloads)
        rep_origins = dict(self.representation_origins)
        rep_payloads.pop(representation_id, None)
        rep_origins.pop(representation_id, None)
        manifest["updated_at"] = utc_now_iso()
        return self._spawn(
            manifest, payloads, origins,
            representation_payloads=rep_payloads,
            representation_origins=rep_origins,
        )

    def clear_representations(self, target: str) -> "MMH3Media":
        if target not in self.manifest.get("representations", {}):
            return self
        manifest, payloads, origins = self._copy()
        removed = manifest.setdefault("representations", {}).pop(target, [])
        rep_payloads = dict(self.representation_payloads)
        rep_origins = dict(self.representation_origins)
        for record in removed:
            rid = str(record.get("id") or "")
            rep_payloads.pop(rid, None)
            rep_origins.pop(rid, None)
        manifest["updated_at"] = utc_now_iso()
        return self._spawn(
            manifest, payloads, origins,
            representation_payloads=rep_payloads,
            representation_origins=rep_origins,
        )

    @property
    def primary_bindings(self):
        """Read-only view of canonical packet-level primary bindings."""
        return MappingProxyType(dict(self.manifest.get("primary", {})))

    def primary(self, kind: str):
        """Return the primary resource handle for *kind*, or None when unset."""
        if kind not in self._primary_kinds():
            raise MMH3ResourceError(f"Unsupported primary kind {kind!r}")
        resource_id = self.primary_bindings.get(kind)
        return self.ref(resource_id) if resource_id is not None else None

    def get_primary(self, kind: str) -> dict[str, Any] | None:
        """Return the canonical primary descriptor for *kind* without payload I/O."""
        ref = self.primary(kind)
        return None if ref is None else ref.descriptor

    def set_primary(self, kind: str, resource_id: str | None) -> "MMH3Media":
        """Return a new packet snapshot with an explicit primary binding update."""
        if kind not in self._primary_kinds():
            raise MMH3ResourceError(f"Unsupported primary kind {kind!r}")
        bindings = dict(self.manifest.get("primary", {}))
        if resource_id is None:
            bindings.pop(kind, None)
        else:
            resource = self.get_by_id(resource_id)
            if resource is None:
                raise MMH3ResourceError(f"Resource {resource_id!r} does not exist")
            descriptor = self.ref(resource_id).descriptor
            if descriptor["kind"] != kind:
                raise MMH3ResourceError(
                    f"Cannot set primary {kind!r} to {descriptor['kind']!r} resource {resource_id!r}"
                )
            bindings[kind] = resource_id
        if bindings == self.manifest.get("primary", {}):
            return self
        manifest, payloads, origins = self._copy()
        manifest["primary"] = bindings
        manifest["updated_at"] = utc_now_iso()
        require_valid_manifest(manifest)
        return self._spawn(manifest, payloads, origins)

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
            "primary": {},
            "resources": [],
            "representations": {},
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

    def get_resource(self, resource_id: str) -> dict[str, Any] | None:
        """Return a detached resource descriptor without materializing payload data."""
        resource = self.get_by_id(resource_id)
        return deep_copy_json(resource) if resource is not None else None

    def ref(self, resource_id: str):
        """Return a lazy handle bound to this immutable packet snapshot."""
        from .resource_ref import MMH3ResourceRef

        return MMH3ResourceRef(self, resource_id)

    def rebind_ref(self, resource_ref):
        """Explicitly rebind a resource handle to this newer packet snapshot."""
        from .resource_ref import MMH3ResourceRef

        if not isinstance(resource_ref, MMH3ResourceRef):
            raise MMH3ResourceError("Expected an MMH3_RESOURCE handle")
        return resource_ref.rebind(self)

    def select_resources(
        self, *, kind: str | None = None, role: str | None = None
    ):
        """Select lazy resource handles without reading any payloads."""
        return tuple(
            self.ref(resource["id"])
            for resource in self.resources()
            if (kind is None or resource["kind"] == kind)
            and (role is None or resource["role"] == role)
        )

    def resource_descriptors(self) -> list[dict[str, Any]]:
        """Return canonical v0.3 logical descriptors without materializing payloads."""
        return [deep_copy_json(resource) for resource in self.manifest["resources"]]

    def filter_resource_refs(
        self,
        *,
        kind: str | None = None,
        role: str | None = None,
        tags: tuple[str, ...] | list[str] | None = None,
    ):
        """Filter via canonical v0.3 fields; never reads payload data."""
        wanted_tags = {str(tag).strip().lower().replace("_", "-") for tag in (tags or ()) if str(tag).strip()}
        refs = []
        for raw in self.manifest["resources"]:
            ref = self.ref(raw["id"])
            descriptor = ref.descriptor
            if kind is not None and descriptor["kind"] != kind:
                continue
            if role is not None and descriptor["role"] != role:
                continue
            if wanted_tags and not wanted_tags.issubset(set(descriptor["tags"])):
                continue
            refs.append(ref)
        refs.sort(key=lambda ref: (ref.order is None, ref.order if ref.order is not None else 0, ref.resource_id))
        return tuple(refs)

    def get_by_role_order(self, role: str, order: int | None = None) -> dict[str, Any] | None:
        for res in self.resources():
            if res["role"] == role and res.get("order") == order:
                return res
        return None

    def select(self, *, role: str | None = None, order: int | None = None, resource_id: str = "") -> dict[str, Any] | None:
        if resource_id:
            return self.get_by_id(resource_id)
        if role is None:
            raise MMH3ResourceError("Either role or resource_id is required")
        return self.get_by_role_order(role, order)

    def _append_history(self, manifest: dict, op: str, **fields: Any) -> None:
        entry = {"op": op, "timestamp": utc_now_iso()}
        entry.update(deep_copy_json(fields))
        manifest.setdefault("history", []).append(entry)

    def record_operation(self, op: str, **fields: Any) -> "MMH3Media":
        if not isinstance(op, str) or not op.strip():
            raise MMH3ResourceError("History operation must be a non-empty string")
        if not is_jsonable(fields):
            raise MMH3ResourceError("History operation fields must be JSON serializable")
        manifest, payloads, origins = self._copy()
        self._append_history(manifest, op.strip(), **fields)
        manifest["updated_at"] = utc_now_iso()
        require_valid_manifest(manifest)
        return self._spawn(manifest, payloads, origins)

    def set_extension_value(self, namespace: str, key: str, value: Any) -> "MMH3Media":
        if not isinstance(namespace, str) or not namespace.strip():
            raise MMH3ResourceError("Extension namespace must be a non-empty string")
        if not isinstance(key, str) or not key.strip():
            raise MMH3ResourceError("Extension key must be a non-empty string")
        if not is_jsonable(value):
            raise MMH3ResourceError("Extension value must be JSON serializable")
        manifest, payloads, origins = self._copy()
        extensions = manifest.setdefault("extensions", {})
        existing = extensions.get(namespace)
        if existing is None:
            existing = extensions[namespace] = {}
        if not isinstance(existing, dict):
            raise MMH3ResourceError(f"Extension namespace {namespace!r} is not an object")
        existing[key.strip()] = deep_copy_json(value)
        manifest["updated_at"] = utc_now_iso()
        require_valid_manifest(manifest)
        return self._spawn(manifest, payloads, origins)

    @staticmethod
    def _ordered(resources: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        return sorted(
            (r for r in resources if r["role"] == role),
            key=lambda r: (r.get("order") is None, r.get("order") if r.get("order") is not None else 0, r["id"]),
        )

    def put(
        self, payload: Any, *, kind: str, role: str = "auxiliary", order: int | None = None,
        mode: str = "add", name: str = "", tags=None, descriptor: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None, extensions: dict[str, Any] | None = None,
        resource_id: str = "", record_history: bool = True,
    ) -> "MMH3Media":
        packet, _ = self.put_with_id(
            payload, kind=kind, role=role, order=order, mode=mode, name=name, tags=tags,
            descriptor=descriptor, provenance=provenance, extensions=extensions,
            resource_id=resource_id, record_history=record_history,
        )
        return packet

    def put_with_id(
        self, payload: Any, *, kind: str, role: str = "auxiliary", order: int | None = None,
        mode: str = "add", name: str = "", tags=None, descriptor: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None, extensions: dict[str, Any] | None = None,
        resource_id: str = "", record_history: bool = True,
    ) -> tuple["MMH3Media", str]:
        validate_role_kind(role, kind)
        from .serializers import validate_payload_contract
        validate_payload_contract(payload, kind, descriptor=descriptor, extensions=extensions)
        if mode not in ("upsert", "add", "replace"):
            raise MMH3ResourceError("Put mode must be upsert, add, or replace")
        if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 0):
            raise MMH3ResourceError("order must be null or a non-negative integer")
        manifest, payloads, origins = self._copy()
        existing = None
        if resource_id:
            existing = next((r for r in manifest["resources"] if r["id"] == resource_id), None)
        elif mode in ("replace", "upsert"):
            existing = next((r for r in manifest["resources"] if r["role"] == role and r.get("order") == order and r["kind"] == kind), None)
        if mode == "replace" and existing is None:
            raise MMH3ResourceError("replace requires an existing resource_id or matching role/kind/order")
        if mode == "add" and resource_id and existing is not None:
            raise MMH3ResourceError(f"Resource {resource_id!r} already exists")
        rid = existing["id"] if existing is not None else (resource_id.strip() or _resource_id())
        revision = _resource_id()
        if existing is None:
            res = make_resource_descriptor(
                resource_id=rid, kind=kind, role=role, name=name, order=order, tags=tags,
                path=_placeholder_path(kind, rid), descriptor=descriptor, provenance=provenance,
                revision=revision, extensions=extensions,
            )
            manifest["resources"].append(res)
            action = "add"
        else:
            preserved = deep_copy_json(existing)
            res = make_resource_descriptor(
                resource_id=rid, kind=kind, role=role, name=name or str(preserved.get("name", "")),
                order=order, tags=tags if tags is not None else preserved.get("tags", []),
                path=preserved.get("path") or _placeholder_path(kind, rid),
                media_type=None, serializer=None,
                descriptor=descriptor if descriptor is not None else preserved.get("descriptor", {}),
                provenance=provenance if provenance is not None else preserved.get("provenance", {}),
                revision=revision, extensions=extensions if extensions is not None else preserved.get("extensions", {}),
            )
            idx = next(i for i, r in enumerate(manifest["resources"]) if r["id"] == rid)
            manifest["resources"][idx] = res
            action = "replace"
            origins.pop(rid, None)
        payloads[rid] = payload
        manifest["updated_at"] = utc_now_iso()
        if record_history:
            self._append_history(manifest, "resource_put", action=action, resource_id=rid, role=role, order=order, kind=kind)
        require_valid_manifest(manifest)
        return self._spawn(manifest, payloads, origins), rid

    def put_primary(
        self, payload: Any, *, kind: str, name: str = "", tags=None,
        descriptor: dict[str, Any] | None = None, provenance: dict[str, Any] | None = None,
        extensions: dict[str, Any] | None = None, resource_id: str = "",
        record_history: bool = True,
    ) -> tuple["MMH3Media", str]:
        """Store a canonical primary resource and bind it atomically."""
        if kind not in self._primary_kinds():
            raise MMH3ResourceError(
                "Primary resources may only be " + ", ".join(self._primary_kinds())
            )
        current = self.manifest.get("primary", {}).get(kind)
        target_id = resource_id.strip() or (str(current) if current else "")
        packet, rid = self.put_with_id(
            payload, kind=kind, role="auxiliary", order=None,
            mode="replace" if target_id else "add", resource_id=target_id,
            name=name, tags=tags, descriptor=descriptor, provenance=provenance, extensions=extensions,
            record_history=record_history,
        )
        return packet.set_primary(kind, rid), rid

    def remove(
        self, *, role: str | None = None, order: int | None = None, resource_id: str = "",
        missing: str = "error", record_history: bool = True,
    ) -> "MMH3Media":
        target = self.select(role=role, order=order, resource_id=resource_id)
        if target is None:
            if missing == "ignore":
                return self
            raise MMH3ResourceError(f"Resource {resource_id or f'{role}@{order}'} does not exist")
        manifest, payloads, origins = self._copy()
        rid = target["id"]
        removed = next(r for r in manifest["resources"] if r["id"] == rid)
        manifest["resources"] = [r for r in manifest["resources"] if r["id"] != rid]
        payloads.pop(rid, None); origins.pop(rid, None)
        manifest["primary"] = {k: v for k, v in manifest.get("primary", {}).items() if v != rid}
        removed_representations = manifest.setdefault("representations", {}).pop(rid, [])
        rep_payloads = dict(self.representation_payloads); rep_origins = dict(self.representation_origins)
        for record in removed_representations:
            rep_id = str(record.get("id") or ""); rep_payloads.pop(rep_id, None); rep_origins.pop(rep_id, None)
        manifest["updated_at"] = utc_now_iso()
        if record_history:
            self._append_history(manifest, "resource_remove", resource_id=rid, role=removed["role"], order=removed.get("order"), kind=removed["kind"])
        require_valid_manifest(manifest)
        return self._spawn(manifest, payloads, origins, representation_payloads=rep_payloads, representation_origins=rep_origins)

    def move(self, *, role: str, from_order: int, to_order: int, record_history: bool = True) -> "MMH3Media":
        if role not in {"reference", "control", "context", "intermediate", "auxiliary"}:
            raise MMH3ResourceError(f"Unsupported v0.3 resource role {role!r}")
        manifest, payloads, origins = self._copy()
        same = self._ordered(manifest["resources"], role)
        if from_order < 0 or from_order >= len(same):
            raise MMH3ResourceError(f"{role} order {from_order} does not exist")
        to_order = max(0, min(to_order, len(same) - 1))
        item = same.pop(from_order); same.insert(to_order, item)
        for i, res in enumerate(same): res["order"] = i
        manifest["updated_at"] = utc_now_iso()
        if record_history and from_order != to_order:
            self._append_history(manifest, "resource_move", resource_id=item["id"], role=role, from_order=from_order, to_order=to_order)
        require_valid_manifest(manifest)
        return self._spawn(manifest, payloads, origins)

    def update_resource(
        self, resource_id: str, *, role: str | None = None, order: int | None | object = ...,
        name: str | None = None, tags=None, descriptor: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None, extensions: dict[str, Any] | None = None,
        record_history: bool = True,
    ) -> "MMH3Media":
        target = self.get_by_id(resource_id)
        if target is None: raise MMH3ResourceError(f"Resource {resource_id!r} does not exist")
        manifest, payloads, origins = self._copy()
        res = next(r for r in manifest["resources"] if r["id"] == resource_id)
        if role is not None: validate_role_kind(role, res["kind"]); res["role"] = role
        if order is not ...:
            if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 0): raise MMH3ResourceError("order must be null or non-negative")
            res["order"] = order
        if name is not None: res["name"] = str(name)
        if tags is not None: res["tags"] = normalize_tags(tags)
        if descriptor is not None: res["descriptor"] = deep_copy_json(descriptor)
        if provenance is not None: res["provenance"] = deep_copy_json(provenance)
        if extensions is not None: res["extensions"] = deep_copy_json(extensions)
        validate_resource_descriptor(res)
        manifest["updated_at"] = utc_now_iso()
        if record_history: self._append_history(manifest, "resource_update", resource_id=resource_id, role=res["role"], order=res.get("order"), kind=res["kind"])
        require_valid_manifest(manifest)
        return self._spawn(manifest, payloads, origins)

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
        return self._spawn(manifest, payloads, origins)

    def with_saved_state(self, *, manifest: dict, source_archive: str) -> "MMH3Media":
        origins = {r["id"]: r["path"] for r in manifest["resources"]}
        representation_origins = {
            record["id"]: record["path"]
            for records in manifest.get("representations", {}).values()
            for record in records
            if isinstance(record, dict) and isinstance(record.get("path"), str) and record.get("path")
        }
        return MMH3Media(
            manifest=deep_copy_json(manifest),
            source_archive=source_archive,
            payloads={},
            origins=origins,
            verify_mode=self.verify_mode,
            dirty=False,
            representation_payloads={},
            representation_origins=representation_origins,
        )
