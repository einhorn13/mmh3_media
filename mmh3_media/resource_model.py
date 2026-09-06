from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .errors import MMH3ResourceError
from .util import deep_copy_json

CORE_RESOURCE_ROLES = ("reference", "control", "context", "intermediate", "auxiliary")
CORE_RESOURCE_KINDS = ("latent", "image", "video", "audio", "mask", "json")
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def normalize_tags(tags: Iterable[str] | None) -> list[str]:
    if tags is None:
        return []
    normalized: set[str] = set()
    for value in tags:
        if not isinstance(value, str):
            raise MMH3ResourceError("Resource tags must be strings")
        tag = value.strip().lower().replace("_", "-")
        if not tag:
            continue
        if not _TAG_RE.match(tag):
            raise MMH3ResourceError(f"Invalid resource tag {value!r}")
        normalized.add(tag)
    return sorted(normalized)


def validate_resource_descriptor(resource: Any) -> None:
    if not isinstance(resource, dict):
        raise MMH3ResourceError("Resource descriptor must be an object")
    required = (
        "id", "kind", "name", "role", "order", "tags", "path", "media_type",
        "serializer", "descriptor", "provenance", "content", "extensions",
    )
    missing = [key for key in required if key not in resource]
    if missing:
        raise MMH3ResourceError(f"Resource descriptor missing fields: {', '.join(missing)}")
    if "slot" in resource:
        raise MMH3ResourceError("v0.3 resource descriptors must not contain legacy 'slot'")
    rid = resource["id"]
    if not isinstance(rid, str) or not rid.strip():
        raise MMH3ResourceError("Resource id must be a non-empty string")
    kind = resource["kind"]
    if kind not in CORE_RESOURCE_KINDS:
        raise MMH3ResourceError(f"Unsupported v0.3 resource kind {kind!r}")
    role = resource["role"]
    if role not in CORE_RESOURCE_ROLES:
        raise MMH3ResourceError(f"Unsupported v0.3 resource role {role!r}")
    if not isinstance(resource["name"], str):
        raise MMH3ResourceError("Resource name must be a string")
    order = resource["order"]
    if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 0):
        raise MMH3ResourceError("Resource order must be null or a non-negative integer")
    tags = resource["tags"]
    if not isinstance(tags, list) or tags != normalize_tags(tags):
        raise MMH3ResourceError("Resource tags must be a normalized sorted list")
    for key in ("path", "media_type", "serializer"):
        if resource[key] is not None and not isinstance(resource[key], str):
            raise MMH3ResourceError(f"Resource {key} must be a string or null")
    for key in ("descriptor", "provenance", "content", "extensions"):
        if not isinstance(resource[key], dict):
            raise MMH3ResourceError(f"Resource {key} must be an object")
    content = resource["content"]
    revision = content.get("revision")
    if not isinstance(revision, str) or not revision:
        raise MMH3ResourceError("Resource content.revision must be a non-empty string")
    digest = content.get("digest")
    if digest is not None and (not isinstance(digest, str) or not digest.startswith("sha256:")):
        raise MMH3ResourceError("Resource content.digest must be null or sha256:<hex>")
    size = content.get("size")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
        raise MMH3ResourceError("Resource content.size must be null or a non-negative integer")


def make_resource_descriptor(
    *,
    resource_id: str,
    kind: str,
    role: str = "auxiliary",
    name: str = "",
    order: int | None = None,
    tags: Iterable[str] | None = None,
    path: str | None = None,
    media_type: str | None = None,
    serializer: str | None = None,
    descriptor: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    revision: str,
    digest: str | None = None,
    size: int | None = None,
    extensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "id": resource_id,
        "kind": kind,
        "name": name,
        "role": role,
        "order": order,
        "tags": normalize_tags(tags),
        "path": path,
        "media_type": media_type,
        "serializer": serializer,
        "descriptor": deep_copy_json(descriptor or {}),
        "provenance": deep_copy_json(provenance or {}),
        "content": {"revision": revision, "digest": digest, "size": size},
        "extensions": deep_copy_json(extensions or {}),
    }
    validate_resource_descriptor(result)
    return result


def descriptor_from_media_metadata(kind: str, metadata: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    shape: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    tensor: dict[str, Any] = {}

    dims = metadata.get("dimensions")
    if isinstance(dims, (list, tuple)) and len(dims) >= 2:
        shape["width"], shape["height"] = int(dims[0]), int(dims[1])
    raw_shape = metadata.get("shape")
    if isinstance(raw_shape, (list, tuple)):
        tensor["shape"] = [int(value) for value in raw_shape]
    dtype = metadata.get("dtype")
    if isinstance(dtype, str) and dtype:
        tensor["dtype"] = dtype

    if kind == "video":
        if isinstance(metadata.get("frame_count"), int):
            shape["frames"] = int(metadata["frame_count"])
        for key in ("fps", "duration"):
            if isinstance(metadata.get(key), (int, float)):
                timing[key] = float(metadata[key])
    elif kind == "audio":
        for key in ("channels", "samples"):
            if isinstance(metadata.get(key), int):
                shape[key] = int(metadata[key])
        if isinstance(metadata.get("sample_rate"), int):
            timing["sample_rate"] = int(metadata["sample_rate"])
        if isinstance(metadata.get("duration"), (int, float)):
            timing["duration"] = float(metadata["duration"])

    if shape:
        observed["shape"] = shape
    if timing:
        observed["timing"] = timing
    if tensor:
        observed["tensor"] = tensor
    return observed




def resource_facts(resource: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten canonical observed/serialization descriptor facts for runtime consumers."""
    descriptor = resource.get("descriptor")
    if not isinstance(descriptor, Mapping):
        return {}
    out: dict[str, Any] = {}
    for section in ("shape", "timing", "tensor", "serialization"):
        value = descriptor.get(section)
        if isinstance(value, Mapping):
            out.update(deep_copy_json(dict(value)))
    return out
