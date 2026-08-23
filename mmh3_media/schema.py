from __future__ import annotations

import re
from typing import Any

from .constants import (
    FORMAT_NAME,
    ORDERED_ROLES,
    ROLE_KIND_COMPAT,
    SCHEMA_VERSION,
    SINGLETON_ROLES,
    SUPPORTED_KINDS,
    VERIFY_MODES,
)
from .errors import MMH3FormatError, MMH3ResourceError
from .util import safe_member_path

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_role_kind(role: str, kind: str) -> None:
    if not isinstance(role, str) or not role.strip():
        raise MMH3ResourceError("Resource role must be a non-empty string")
    if kind not in SUPPORTED_KINDS:
        raise MMH3ResourceError(f"Unsupported resource kind {kind!r}")
    allowed = ROLE_KIND_COMPAT.get(role)
    if allowed is not None and kind not in allowed:
        raise MMH3ResourceError(
            f"Role {role!r} does not accept kind {kind!r}; expected one of {sorted(allowed)}"
        )


def validate_manifest(manifest: Any, *, allow_unknown_kinds: bool = False) -> None:
    if not isinstance(manifest, dict):
        raise MMH3FormatError("packet.json root must be a JSON object")
    if manifest.get("format") != FORMAT_NAME:
        raise MMH3FormatError(f"Not an {FORMAT_NAME} manifest")
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        if isinstance(version, int) and version > SCHEMA_VERSION:
            raise MMH3FormatError(
                f"MMH3_MEDIA schema_version={version} is newer than this plugin supports ({SCHEMA_VERSION})"
            )
        raise MMH3FormatError(f"Unsupported MMH3_MEDIA schema_version={version!r}")
    if not isinstance(manifest.get("id"), str) or not manifest["id"]:
        raise MMH3FormatError("Manifest 'id' must be a non-empty string")
    if "name" in manifest and not isinstance(manifest["name"], str):
        raise MMH3FormatError("Manifest 'name' must be a string")
    for required in ("generation", "notes", "resources", "history", "extensions"):
        if required not in manifest:
            raise MMH3FormatError(f"Manifest is missing required field {required!r}")
    if not isinstance(manifest["generation"], dict):
        raise MMH3FormatError("Manifest 'generation' must be an object")
    if not isinstance(manifest["notes"], str):
        raise MMH3FormatError("Manifest 'notes' must be a string")
    resources = manifest["resources"]
    if not isinstance(resources, list):
        raise MMH3FormatError("Manifest 'resources' must be an array")
    if not isinstance(manifest["history"], list):
        raise MMH3FormatError("Manifest 'history' must be an array")
    if not isinstance(manifest["extensions"], dict):
        raise MMH3FormatError("Manifest 'extensions' must be an object")

    ids: set[str] = set()
    selectors: set[tuple[str, int]] = set()
    paths: set[str] = set()
    for i, res in enumerate(resources):
        if not isinstance(res, dict):
            raise MMH3FormatError(f"resources[{i}] must be an object")
        rid = res.get("id")
        kind = res.get("kind")
        role = res.get("role")
        slot = res.get("slot")
        path = res.get("path")
        if not isinstance(rid, str) or not rid:
            raise MMH3FormatError(f"resources[{i}].id must be a non-empty string")
        if rid in ids:
            raise MMH3FormatError(f"Duplicate resource id {rid!r}")
        ids.add(rid)
        if not isinstance(kind, str) or not kind:
            raise MMH3FormatError(f"Resource {rid}: kind must be a non-empty string")
        if not allow_unknown_kinds and kind not in SUPPORTED_KINDS:
            raise MMH3FormatError(f"Resource {rid}: unsupported kind {kind!r}")
        if not isinstance(role, str) or not role:
            raise MMH3FormatError(f"Resource {rid}: role must be a non-empty string")
        if kind in SUPPORTED_KINDS:
            allowed = ROLE_KIND_COMPAT.get(role)
            if allowed is not None and kind not in allowed:
                raise MMH3FormatError(
                    f"Resource {rid}: role {role!r} is incompatible with kind {kind!r}; expected {sorted(allowed)}"
                )
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
            raise MMH3FormatError(f"Resource {rid}: slot must be a non-negative integer")
        selector = (role, slot)
        if selector in selectors:
            raise MMH3FormatError(f"Duplicate role+slot selector {role}[{slot}]")
        selectors.add(selector)
        if not isinstance(path, str):
            raise MMH3FormatError(f"Resource {rid}: path must be a string")
        path = safe_member_path(path)
        if path in paths:
            raise MMH3FormatError(f"Duplicate archive member path {path!r}")
        paths.add(path)
        if "metadata" not in res:
            raise MMH3FormatError(f"Resource {rid}: missing required metadata object")
        if not isinstance(res["metadata"], dict):
            raise MMH3FormatError(f"Resource {rid}: metadata must be an object")
        sha = res.get("sha256")
        if role == "preview" and sha is not None:
            raise MMH3FormatError(f"Resource {rid}: preview cache must not carry sha256")
        if sha is not None and (not isinstance(sha, str) or not _SHA256_RE.match(sha)):
            raise MMH3FormatError(f"Resource {rid}: invalid sha256")
        size = res.get("size")
        if size is not None and (not isinstance(size, int) or size < 0):
            raise MMH3FormatError(f"Resource {rid}: invalid size")

    for role in SINGLETON_ROLES:
        slots = sorted(r["slot"] for r in resources if r["role"] == role)
        if any(slot != 0 for slot in slots):
            raise MMH3FormatError(f"Singleton role {role!r} must use slot 0")
        if len(slots) > 1:
            raise MMH3FormatError(f"Singleton role {role!r} appears more than once")
    for role in ORDERED_ROLES:
        slots = sorted(r["slot"] for r in resources if r["role"] == role)
        if slots and slots != list(range(len(slots))):
            raise MMH3FormatError(
                f"Ordered role {role!r} must use dense slots 0..{len(slots)-1}; got {slots}"
            )


def validate_verify_mode(mode: str) -> None:
    if mode not in VERIFY_MODES:
        raise MMH3FormatError(f"verify must be one of {VERIFY_MODES}, got {mode!r}")
