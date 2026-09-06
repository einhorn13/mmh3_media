from __future__ import annotations

from typing import Any, Mapping, Sequence

from .errors import MMH3ResourceError
from .util import deep_copy_json

H3_REFERENCE_CONTRACT_VERSION = 1
H3_CONTEXT_CONTRACT_VERSION = 1
H3_CONTROL_RESOURCE_CONTRACT_VERSION = 1

_REFERENCE_KINDS = ("image", "video", "audio")
_CONTEXT_USAGES = ("first_frame", "last_frame")
_CONTROL_USAGES = ("control_video", "inpaint_source", "mask")
_CONTROL_EXPECTED = {
    "control_video": ("control", "video"),
    "inpaint_source": ("context", "video"),
    "mask": ("control", "mask"),
}


def _h3_extension(descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = descriptor.get("extensions")
    if not isinstance(extensions, Mapping):
        return {}
    value = extensions.get("minimax_h3")
    return value if isinstance(value, Mapping) else {}


def make_reference_contract(
    *,
    kind: str,
    enabled: bool = True,
    purposes: Sequence[str] = ("unknown",),
    video_resource_id: str | None = None,
) -> dict[str, Any]:
    if kind not in _REFERENCE_KINDS:
        raise MMH3ResourceError(f"H3 reference kind must be one of {_REFERENCE_KINDS}; got {kind!r}")
    if not isinstance(enabled, bool):
        raise MMH3ResourceError("H3 reference enabled must be boolean")
    if (
        isinstance(purposes, (str, bytes))
        or not purposes
        or not all(isinstance(item, str) and item for item in purposes)
    ):
        raise MMH3ResourceError("H3 reference purposes must be a non-empty string array")
    normalized: list[str] = []
    for purpose in purposes:
        if purpose not in normalized:
            normalized.append(purpose)
    value: dict[str, Any] = {
        "contract_version": H3_REFERENCE_CONTRACT_VERSION,
        "enabled": enabled,
        "purposes": normalized,
    }
    if video_resource_id is not None:
        video_resource_id = str(video_resource_id).strip()
        if kind != "audio":
            raise MMH3ResourceError("Only an audio H3 reference may bind to a video reference")
        if not video_resource_id:
            raise MMH3ResourceError("H3 reference video binding ID must not be empty")
        value["binding"] = {"video_resource_id": video_resource_id}
    return value


def make_context_contract(*, usage: str) -> dict[str, Any]:
    if usage not in _CONTEXT_USAGES:
        raise MMH3ResourceError(f"Unsupported H3 context usage {usage!r}")
    return {"contract_version": H3_CONTEXT_CONTRACT_VERSION, "usage": usage}


def make_control_resource_contract(*, usage: str) -> dict[str, Any]:
    if usage not in _CONTROL_USAGES:
        raise MMH3ResourceError(f"Unsupported H3 control resource usage {usage!r}")
    return {"contract_version": H3_CONTROL_RESOURCE_CONTRACT_VERSION, "usage": usage}


def make_cache_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MMH3ResourceError("H3 reference cache contract must be an object")
    if not isinstance(value.get("key"), str) or not value["key"]:
        raise MMH3ResourceError("H3 reference cache contract has no cache key")
    source = value.get("source")
    if (
        not isinstance(source, Mapping)
        or not isinstance(source.get("resource_id"), str)
        or not source["resource_id"]
    ):
        raise MMH3ResourceError("H3 reference cache contract has invalid source binding")
    return deep_copy_json(dict(value))


def reference_contract(descriptor: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if descriptor.get("role") != "reference" or descriptor.get("kind") not in _REFERENCE_KINDS:
        return None
    value = _h3_extension(descriptor).get("reference")
    if not isinstance(value, Mapping):
        raise MMH3ResourceError(f"H3 reference {descriptor.get('id')!r} has no reference contract")
    if value.get("contract_version") != H3_REFERENCE_CONTRACT_VERSION:
        raise MMH3ResourceError(f"H3 reference {descriptor.get('id')!r} has unsupported reference contract version")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise MMH3ResourceError(f"H3 reference {descriptor.get('id')!r} enabled must be boolean")
    purposes = value.get("purposes")
    if not isinstance(purposes, list) or not purposes or not all(isinstance(item, str) and item for item in purposes):
        raise MMH3ResourceError(f"H3 reference {descriptor.get('id')!r} purposes must be a non-empty string array")
    binding = value.get("binding")
    if binding is not None:
        if descriptor.get("kind") != "audio":
            raise MMH3ResourceError(f"Only audio H3 reference {descriptor.get('id')!r} may carry a video binding")
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("video_resource_id"), str)
            or not binding["video_resource_id"]
        ):
            raise MMH3ResourceError(f"H3 reference {descriptor.get('id')!r} has invalid video binding")
    return value


def reference_enabled(descriptor: Mapping[str, Any]) -> bool:
    value = reference_contract(descriptor)
    return False if value is None else bool(value["enabled"])


def reference_purposes(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    value = reference_contract(descriptor)
    return () if value is None else tuple(value["purposes"])


def reference_video_binding(descriptor: Mapping[str, Any]) -> str | None:
    value = reference_contract(descriptor)
    if value is None or value.get("binding") is None:
        return None
    return str(value["binding"]["video_resource_id"])


def context_usage(descriptor: Mapping[str, Any]) -> str | None:
    if descriptor.get("role") != "context":
        return None
    value = _h3_extension(descriptor).get("context")
    if not isinstance(value, Mapping):
        return None
    if value.get("contract_version") != H3_CONTEXT_CONTRACT_VERSION:
        raise MMH3ResourceError(f"H3 context {descriptor.get('id')!r} has unsupported context contract version")
    usage = value.get("usage")
    if usage not in _CONTEXT_USAGES:
        raise MMH3ResourceError(f"H3 context {descriptor.get('id')!r} has unsupported usage {usage!r}")
    if descriptor.get("kind") != "image":
        raise MMH3ResourceError(f"H3 context {descriptor.get('id')!r} usage {usage!r} requires kind='image'")
    return str(usage)


def control_usage(descriptor: Mapping[str, Any]) -> str | None:
    if descriptor.get("role") not in ("control", "context"):
        return None
    value = _h3_extension(descriptor).get("control")
    if not isinstance(value, Mapping):
        return None
    if value.get("contract_version") != H3_CONTROL_RESOURCE_CONTRACT_VERSION:
        raise MMH3ResourceError(f"H3 control resource {descriptor.get('id')!r} has unsupported control contract version")
    usage = value.get("usage")
    if usage not in _CONTROL_USAGES:
        raise MMH3ResourceError(f"H3 control resource {descriptor.get('id')!r} has unsupported usage {usage!r}")
    expected_role, expected_kind = _CONTROL_EXPECTED[str(usage)]
    if descriptor.get("role") != expected_role or descriptor.get("kind") != expected_kind:
        raise MMH3ResourceError(
            f"H3 control resource {descriptor.get('id')!r} usage {usage!r} requires {expected_role}/{expected_kind}, "
            f"got {descriptor.get('role')}/{descriptor.get('kind')}"
        )
    return str(usage)


def cache_contract(descriptor: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if descriptor.get("role") != "intermediate":
        return None
    value = _h3_extension(descriptor).get("cache")
    if value is None:
        return None
    if descriptor.get("kind") != "latent":
        raise MMH3ResourceError(f"H3 reference cache {descriptor.get('id')!r} must be intermediate/latent")
    try:
        return make_cache_contract(value)
    except MMH3ResourceError as exc:
        raise MMH3ResourceError(f"H3 reference cache {descriptor.get('id')!r}: {exc}") from exc


def find_context_resource(packet, usage: str) -> dict[str, Any] | None:
    """Return the first canonical H3 context resource for *usage* without payload I/O."""
    make_context_contract(usage=usage)  # validates the requested usage
    for ref in packet.filter_resource_refs(role="context"):
        descriptor = ref.descriptor
        if context_usage(descriptor) == usage:
            return descriptor
    return None


def find_control_resource(packet, usage: str) -> dict[str, Any] | None:
    """Return the first canonical H3 control/context resource for *usage*."""
    make_control_resource_contract(usage=usage)  # validates the requested usage
    for role in ("control", "context"):
        for ref in packet.filter_resource_refs(role=role):
            descriptor = ref.descriptor
            if control_usage(descriptor) == usage:
                return descriptor
    return None
