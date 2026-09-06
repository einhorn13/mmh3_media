from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .core import MMH3Media
from .errors import MMH3ResourceError
from .constants import REFERENCE_PURPOSES
from .resolution import ResolvedReferenceSet, resolve_reference_set
from .h3_resource_semantics import make_reference_contract, reference_contract


@dataclass(frozen=True)
class ReferenceConfigurationResult:
    packet: MMH3Media
    resource_id: str
    report: ResolvedReferenceSet

    def info_json(self) -> str:
        return json.dumps(self.report.to_dict(), ensure_ascii=False, indent=2)


def _normalize_purposes(purposes: Sequence[str]) -> list[str]:
    if isinstance(purposes, (str, bytes)):
        raise MMH3ResourceError("Reference purposes must be an array of purpose names")
    result: list[str] = []
    for purpose in purposes:
        if purpose not in REFERENCE_PURPOSES:
            raise MMH3ResourceError(f"Unsupported reference purpose {purpose!r}; expected one of {REFERENCE_PURPOSES}")
        if purpose not in result:
            result.append(purpose)
    if not result:
        raise MMH3ResourceError("Reference purposes must not be empty; use ['unknown'] when not classified")
    return result


def configure_reference(
    packet: MMH3Media,
    resource_id: str,
    *,
    inclusion: str = "keep",
    purposes: Sequence[str] | None = None,
    order: int | None = None,
    binding_action: str = "keep",
    video_resource_id: str = "",
) -> ReferenceConfigurationResult:
    """Configure one reference by stable ID without materializing its payload."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    resource_id = str(resource_id or "").strip()
    descriptor = packet.get_by_id(resource_id)
    if descriptor is None:
        raise MMH3ResourceError(f"Resource {resource_id!r} does not exist")
    if inclusion not in ("keep", "include", "exclude"):
        raise MMH3ResourceError("inclusion must be keep, include, or exclude")
    if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 0):
        raise MMH3ResourceError("order must be a non-negative integer when provided")
    if binding_action not in ("keep", "set", "clear"):
        raise MMH3ResourceError("binding_action must be keep, set, or clear")

    out = packet
    initial = out.ref(resource_id).descriptor
    if initial["kind"] not in ("image", "video", "audio"):
        raise MMH3ResourceError(
            f"Resource {resource_id!r} has kind {initial['kind']!r}; H3 references require image, video, or audio"
        )
    target_order = initial.get("order") if order is None else order
    if target_order is None:
        target_order = 0
    extensions = dict(initial.get("extensions") or {})
    h3 = dict(extensions.get("minimax_h3") or {})
    if initial.get("role") != "reference" or "reference" not in h3:
        h3["reference"] = make_reference_contract(kind=initial["kind"])
    extensions["minimax_h3"] = h3
    out = out.update_resource(
        resource_id, role="reference", order=int(target_order), extensions=extensions, record_history=False
    )
    canonical = out.ref(resource_id).descriptor

    current_contract = reference_contract(canonical)
    if current_contract is None:
        raise MMH3ResourceError(f"Resource {resource_id!r} has no canonical H3 reference contract")
    enabled = bool(current_contract["enabled"])
    selected_purposes = list(current_contract["purposes"])
    bound_video_id = None
    binding = current_contract.get("binding")
    if isinstance(binding, dict):
        bound_video_id = binding.get("video_resource_id")
    if inclusion != "keep":
        enabled = inclusion == "include"
    if purposes is not None:
        selected_purposes = _normalize_purposes(purposes)

    if binding_action != "keep":
        if canonical["kind"] != "audio":
            raise MMH3ResourceError("Only an audio reference can bind to a video reference soundtrack")
        if binding_action == "clear":
            bound_video_id = None
        else:
            video_resource_id = str(video_resource_id or "").strip()
            video = out.get_by_id(video_resource_id)
            if video is None:
                raise MMH3ResourceError(
                    f"Soundtrack binding target {video_resource_id!r} must be an existing video reference"
                )
            video_canonical = out.ref(video_resource_id).descriptor
            if video_canonical.get("role") != "reference" or video_canonical.get("kind") != "video":
                raise MMH3ResourceError(
                    f"Soundtrack binding target {video_resource_id!r} must be an existing reference/video resource"
                )
            bound_video_id = video_resource_id

    contract = make_reference_contract(
        kind=canonical["kind"],
        enabled=enabled,
        purposes=selected_purposes,
        video_resource_id=bound_video_id,
    )
    extensions = dict(canonical.get("extensions") or {})
    h3 = dict(extensions.get("minimax_h3") or {})
    h3["reference"] = contract
    extensions["minimax_h3"] = h3
    out = out.update_resource(resource_id, extensions=extensions, record_history=False)
    configured_descriptor = out.ref(resource_id).descriptor
    out = out.record_operation(
        "reference_configure",
        resource_id=resource_id,
        role=configured_descriptor["role"],
        kind=configured_descriptor["kind"],
        order=configured_descriptor["order"],
        inclusion=inclusion,
        purposes=None if purposes is None else list(purposes),
        binding_action=binding_action,
        video_resource_id=str(video_resource_id or "") if binding_action == "set" else "",
    )
    return ReferenceConfigurationResult(out, resource_id, resolve_reference_set(out))
