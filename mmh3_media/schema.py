from __future__ import annotations

import re
from typing import Any

from .constants import FORMAT_NAME, REFERENCE_PURPOSES, SCHEMA_VERSION, VERIFY_MODES
from .errors import MMH3FormatError, MMH3ResourceError
from .lora_provenance import normalize_generation_loras
from .util import safe_member_path

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_role_kind(role: str, kind: str) -> None:
    from .resource_model import CORE_RESOURCE_KINDS, CORE_RESOURCE_ROLES
    if role not in CORE_RESOURCE_ROLES:
        raise MMH3ResourceError(f"Unsupported v0.3 resource role {role!r}")
    if kind not in CORE_RESOURCE_KINDS:
        raise MMH3ResourceError(f"Unsupported v0.3 resource kind {kind!r}")


def validate_manifest(manifest: Any, *, allow_unknown_kinds: bool = False) -> None:
    validate_schema2_manifest(manifest, allow_unknown_kinds=allow_unknown_kinds)


def validate_verify_mode(mode: str) -> None:
    if mode not in VERIFY_MODES:
        raise MMH3FormatError(f"verify must be one of {VERIFY_MODES}, got {mode!r}")


def validate_schema2_manifest(manifest: Any, *, allow_unknown_kinds: bool = False) -> None:
    """Validate the canonical MMH3 v0.3 / schema-2 manifest contract."""
    from .resource_model import CORE_RESOURCE_KINDS, validate_resource_descriptor

    if not isinstance(manifest, dict):
        raise MMH3FormatError("packet.json root must be a JSON object")
    if manifest.get("format") != FORMAT_NAME:
        raise MMH3FormatError(f"Not an {FORMAT_NAME} manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise MMH3FormatError(
            f"MMH3 v0.3 archives require schema_version={SCHEMA_VERSION}; got {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("id"), str) or not manifest["id"]:
        raise MMH3FormatError("Manifest 'id' must be a non-empty string")
    if "name" in manifest and not isinstance(manifest["name"], str):
        raise MMH3FormatError("Manifest 'name' must be a string")
    for required in (
        "generation", "notes", "primary", "resources", "representations", "history", "extensions"
    ):
        if required not in manifest:
            raise MMH3FormatError(f"Manifest is missing required field {required!r}")
    if not isinstance(manifest["generation"], dict):
        raise MMH3FormatError("Manifest 'generation' must be an object")
    if "loras" in manifest["generation"]:
        try:
            normalize_generation_loras(manifest["generation"]["loras"])
        except MMH3ResourceError as exc:
            raise MMH3FormatError(str(exc)) from exc
    if not isinstance(manifest["notes"], str):
        raise MMH3FormatError("Manifest 'notes' must be a string")
    if not isinstance(manifest["primary"], dict):
        raise MMH3FormatError("Manifest 'primary' must be an object")
    if not isinstance(manifest["resources"], list):
        raise MMH3FormatError("Manifest 'resources' must be an array")
    if not isinstance(manifest["representations"], dict):
        raise MMH3FormatError("Manifest 'representations' must be an object")
    if not isinstance(manifest["history"], list):
        raise MMH3FormatError("Manifest 'history' must be an array")
    if not isinstance(manifest["extensions"], dict):
        raise MMH3FormatError("Manifest 'extensions' must be an object")

    resources_by_id: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for index, resource in enumerate(manifest["resources"]):
        if not isinstance(resource, dict):
            raise MMH3FormatError(f"resources[{index}] must be an object")
        try:
            validate_resource_descriptor(resource)
        except MMH3ResourceError as exc:
            if allow_unknown_kinds and "Unsupported v0.3 resource kind" in str(exc):
                pass
            else:
                raise MMH3FormatError(f"resources[{index}]: {exc}") from exc
        rid = resource.get("id")
        if rid in resources_by_id:
            raise MMH3FormatError(f"Duplicate resource id {rid!r}")
        resources_by_id[str(rid)] = resource
        path = resource.get("path")
        if not isinstance(path, str) or not path:
            raise MMH3FormatError(f"Resource {rid}: archive path must be a non-empty string")
        path = safe_member_path(path)
        if path in paths:
            raise MMH3FormatError(f"Duplicate archive member path {path!r}")
        paths.add(path)
        if not allow_unknown_kinds and resource.get("kind") not in CORE_RESOURCE_KINDS:
            raise MMH3FormatError(f"Resource {rid}: unsupported kind {resource.get('kind')!r}")
        content = resource.get("content", {})
        digest = content.get("digest") if isinstance(content, dict) else None
        if digest is not None:
            if not isinstance(digest, str) or not digest.startswith("sha256:") or not _SHA256_RE.match(digest[7:]):
                raise MMH3FormatError(f"Resource {rid}: invalid content.digest")

    allowed_primary = {"video", "audio", "image", "latent", "mask"}
    for kind, rid in manifest["primary"].items():
        if kind not in allowed_primary:
            raise MMH3FormatError(f"Unsupported primary kind {kind!r}")
        if not isinstance(rid, str) or not rid:
            raise MMH3FormatError(f"Primary binding for {kind!r} must be a resource id")
        target = resources_by_id.get(rid)
        if target is None:
            raise MMH3FormatError(f"Primary {kind!r} references missing resource {rid!r}")
        if target.get("kind") != kind:
            raise MMH3FormatError(
                f"Primary {kind!r} must reference kind {kind!r}, got {target.get('kind')!r}"
            )


def validate_archive_manifest(manifest: Any) -> None:
    """Validate the only v0.3 archive target: schema 2."""
    validate_schema2_manifest(manifest, allow_unknown_kinds=False)
