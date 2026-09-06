from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .core import MMH3Media
from .errors import MMH3ResourceError
from .util import deep_copy_json, json_dumps_canonical
from .h3_resource_semantics import control_usage


H3_CONTROL_CONTRACT = "mmh3_h3_control_provider_v1"
H3_CONTROL_ALGORITHMS = (
    "native_reference",
    "native_full_frame_refine",
    "native_tiled_refine",
    "external_tile_backend",
    "fun_controlnet_union_bf16",
    "fun_controlnet_union_int8_convrot",
    "auto",
)
H3_CONTROLNET_ALGORITHMS = (
    "fun_controlnet_union_bf16",
    "fun_controlnet_union_int8_convrot",
)
H3_CONTROL_KINDS = ("canny", "depth", "hed", "mlsd", "pose", "inpaint")
H3_CONTROL_KIND_OPTIONS = ("none", *H3_CONTROL_KINDS)
H3_CONTROL_RESOURCE_USAGES = ("control_video", "inpaint_source", "mask")
# Temporary alias for downstream UI code; these are H3 usages, not core resource roles.
H3_CONTROL_RESOURCE_ROLES = H3_CONTROL_RESOURCE_USAGES
H3_CONTROL_TEMPORAL_POLICIES = ("strict", "trim", "pad_hold_last")
H3_CONTROL_ALGORITHM_CONTEXTS = ("generation", "full_frame_refine", "tiled_refine")
H3_CONTROL_AUTO_PREFERENCES = ("stable_native_first", "int8_first", "bf16_first")
H3_CONTROL_CONTEXT_ALGORITHMS = {
    "generation": (
        "native_reference",
        "fun_controlnet_union_int8_convrot",
        "fun_controlnet_union_bf16",
    ),
    "full_frame_refine": (
        "native_full_frame_refine",
        "fun_controlnet_union_int8_convrot",
        "fun_controlnet_union_bf16",
    ),
    "tiled_refine": (
        "native_tiled_refine",
        "external_tile_backend",
        "fun_controlnet_union_int8_convrot",
        "fun_controlnet_union_bf16",
    ),
}
H3_CONTROL_MASK_POLICY = "white_regenerate"
H3_CONTROL_AUDIO_POLICY = "uncontrolled_preserve_current_owner"
H3_CONTROL_PROVIDER_ID = "comfy_minimax_h3_fun_controlnet"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ControlDiagnostic:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class ControlPreprocessor:
    name: str = ""
    version: str = ""
    settings_json: str = "{}"

    def settings(self) -> dict[str, Any]:
        value = json.loads(self.settings_json)
        if not isinstance(value, dict):
            raise MMH3ResourceError("Control preprocessor settings must be a JSON object")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "settings": self.settings()}


@dataclass(frozen=True)
class ControlProviderProvenance:
    provider_id: str = H3_CONTROL_PROVIDER_ID
    capability_fingerprint: str = ""
    checkpoint_sha256: str = ""
    quantization: str = ""
    dtype: str = ""
    base_family: str = ""
    adaln_form: str = ""
    vae_fingerprint: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.provider_id,
            "capability_fingerprint": self.capability_fingerprint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "base_family": self.base_family,
            "adaln_form": self.adaln_form,
            "vae_fingerprint": self.vae_fingerprint,
        }


@dataclass(frozen=True)
class ControlAlgorithmResolution:
    requested_algorithm: str
    effective_algorithm: str
    context: str
    preference: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "requested_algorithm": self.requested_algorithm,
            "effective_algorithm": self.effective_algorithm,
            "context": self.context,
            "preference": self.preference,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class H3ControlConfiguration:
    requested_algorithm: str
    effective_algorithm: str
    control_kind: str
    strength: float
    start_percent: float
    end_percent: float
    temporal_policy: str
    control_video_resource_id: str = ""
    inpaint_source_resource_id: str = ""
    mask_resource_id: str = ""
    selection_reason: str = ""
    reference_composition: bool = False
    composition_order: str = "reference_then_control"
    preprocessor: ControlPreprocessor = ControlPreprocessor()
    provider: ControlProviderProvenance = ControlProviderProvenance()
    contract: str = H3_CONTROL_CONTRACT
    mask_policy: str = H3_CONTROL_MASK_POLICY
    audio_policy: str = H3_CONTROL_AUDIO_POLICY

    @property
    def controlnet_active(self) -> bool:
        return self.effective_algorithm in H3_CONTROLNET_ALGORITHMS and self.strength > 0.0

    @property
    def source_resource_ids(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (
                self.control_video_resource_id,
                self.inpaint_source_resource_id,
                self.mask_resource_id,
            )
            if value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "requested_algorithm": self.requested_algorithm,
            "effective_algorithm": self.effective_algorithm,
            "selection_reason": self.selection_reason,
            "type": None if self.control_kind == "none" else self.control_kind,
            "strength": self.strength,
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
            "temporal_policy": self.temporal_policy,
            "mask_policy": self.mask_policy,
            "audio_policy": self.audio_policy,
            "source_resource_ids": list(self.source_resource_ids),
            "resources": {
                "control_video": self.control_video_resource_id or None,
                "inpaint_source": self.inpaint_source_resource_id or None,
                "mask": self.mask_resource_id or None,
            },
            "reference_composition": {
                "enabled": self.reference_composition,
                "order": self.composition_order,
            },
            "preprocessor": self.preprocessor.to_dict(),
            "provider": self.provider.to_dict(),
        }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _canonical_settings(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "{}"
    if not isinstance(value, Mapping):
        raise MMH3ResourceError("Control preprocessor settings must be a JSON object")
    try:
        return json_dumps_canonical(deep_copy_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError(f"Control preprocessor settings are not JSON-safe: {exc}") from exc


def resolve_control_algorithm(
    *,
    requested_algorithm: str,
    context: str,
    availability: Mapping[str, Any],
    preference: str = "stable_native_first",
) -> ControlAlgorithmResolution:
    requested = _clean_text(requested_algorithm).lower()
    context = _clean_text(context).lower()
    preference = _clean_text(preference).lower()
    if requested not in H3_CONTROL_ALGORITHMS:
        raise MMH3ResourceError(f"Unsupported requested H3 control algorithm {requested!r}")
    if context not in H3_CONTROL_ALGORITHM_CONTEXTS:
        raise MMH3ResourceError(
            f"Unsupported control algorithm context {context!r}; expected one of {H3_CONTROL_ALGORITHM_CONTEXTS}"
        )
    if preference not in H3_CONTROL_AUTO_PREFERENCES:
        raise MMH3ResourceError(
            f"Unsupported auto preference {preference!r}; expected one of {H3_CONTROL_AUTO_PREFERENCES}"
        )
    if not isinstance(availability, Mapping):
        raise MMH3ResourceError("Control algorithm availability must be an object")

    allowed = H3_CONTROL_CONTEXT_ALGORITHMS[context]

    def state(algorithm: str) -> tuple[bool, str]:
        raw = availability.get(algorithm)
        if isinstance(raw, bool):
            return raw, "available" if raw else "unavailable"
        if isinstance(raw, Mapping):
            available = raw.get("available")
            if not isinstance(available, bool):
                raise MMH3ResourceError(
                    f"Availability for {algorithm!r} must contain boolean 'available'"
                )
            return available, _clean_text(raw.get("reason")) or ("available" if available else "unavailable")
        raise MMH3ResourceError(
            f"Availability snapshot is missing explicit state for candidate {algorithm!r}"
        )

    if requested != "auto":
        if requested not in allowed:
            raise MMH3ResourceError(
                f"Algorithm {requested!r} is not valid for control context {context!r}"
            )
        available, detail = state(requested)
        if not available:
            raise MMH3ResourceError(
                f"Explicitly requested algorithm {requested!r} is unavailable: {detail}"
            )
        return ControlAlgorithmResolution(requested, requested, context, preference, f"explicit:{detail}")

    native = tuple(item for item in allowed if item not in H3_CONTROLNET_ALGORITHMS)
    int8 = ("fun_controlnet_union_int8_convrot",) if "fun_controlnet_union_int8_convrot" in allowed else ()
    bf16 = ("fun_controlnet_union_bf16",) if "fun_controlnet_union_bf16" in allowed else ()
    order = {
        "stable_native_first": (*native, *int8, *bf16),
        "int8_first": (*int8, *bf16, *native),
        "bf16_first": (*bf16, *int8, *native),
    }[preference]
    rejected: list[str] = []
    for candidate in order:
        available, detail = state(candidate)
        if available:
            reason = (
                f"auto:{preference}:{context}:selected={candidate}:{detail};"
                f"rejected={','.join(rejected) or 'none'}"
            )
            return ControlAlgorithmResolution("auto", candidate, context, preference, reason)
        rejected.append(f"{candidate}({detail})")
    raise MMH3ResourceError(
        f"No available algorithm for auto context {context!r}; checked: {', '.join(rejected)}"
    )


def build_control_configuration(
    *,
    requested_algorithm: str,
    effective_algorithm: str = "",
    control_kind: str = "none",
    strength: float = 1.0,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    temporal_policy: str = "strict",
    control_video_resource_id: str = "",
    inpaint_source_resource_id: str = "",
    mask_resource_id: str = "",
    selection_reason: str = "",
    reference_composition: bool = False,
    composition_order: str = "reference_then_control",
    preprocessor_name: str = "",
    preprocessor_version: str = "",
    preprocessor_settings: Mapping[str, Any] | None = None,
    provider_id: str = H3_CONTROL_PROVIDER_ID,
    capability_fingerprint: str = "",
    checkpoint_sha256: str = "",
    quantization: str = "",
    dtype: str = "",
    base_family: str = "",
    adaln_form: str = "",
    vae_fingerprint: str = "",
) -> H3ControlConfiguration:
    requested = _clean_text(requested_algorithm).lower()
    effective = _clean_text(effective_algorithm).lower() or requested
    kind = _clean_text(control_kind).lower() or "none"
    temporal = _clean_text(temporal_policy).lower()
    reason = _clean_text(selection_reason)

    if requested not in H3_CONTROL_ALGORITHMS:
        raise MMH3ResourceError(
            f"Unsupported requested H3 control algorithm {requested!r}; expected one of {H3_CONTROL_ALGORITHMS}"
        )
    if effective not in H3_CONTROL_ALGORITHMS or effective == "auto":
        raise MMH3ResourceError("effective_algorithm must be one concrete H3 algorithm, never 'auto'")
    if requested != "auto" and effective != requested:
        raise MMH3ResourceError(
            f"Explicitly requested algorithm {requested!r} cannot silently become {effective!r}"
        )
    if requested == "auto" and not reason:
        raise MMH3ResourceError("The opt-in 'auto' algorithm requires a non-empty selection_reason")
    if kind not in H3_CONTROL_KIND_OPTIONS:
        raise MMH3ResourceError(f"Unsupported control kind {kind!r}; expected one of {H3_CONTROL_KIND_OPTIONS}")
    if temporal not in H3_CONTROL_TEMPORAL_POLICIES:
        raise MMH3ResourceError(
            f"Unsupported temporal policy {temporal!r}; expected one of {H3_CONTROL_TEMPORAL_POLICIES}"
        )

    try:
        strength_value = float(strength)
        start_value = float(start_percent)
        end_value = float(end_percent)
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError("Control strength and window values must be numeric") from exc
    if not all(math.isfinite(value) for value in (strength_value, start_value, end_value)):
        raise MMH3ResourceError("Control strength and window values must be finite")
    if not 0.0 <= strength_value <= 10.0:
        raise MMH3ResourceError("Control strength must be in the upstream range 0..10")
    if not 0.0 <= start_value <= end_value <= 1.0:
        raise MMH3ResourceError("Control window must satisfy 0 <= start_percent <= end_percent <= 1")
    if strength_value > 0.0 and start_value == end_value:
        raise MMH3ResourceError("An active control requires a non-empty start/end window")

    is_controlnet = effective in H3_CONTROLNET_ALGORITHMS
    if is_controlnet and kind == "none" and strength_value > 0.0:
        raise MMH3ResourceError("An active ControlNet algorithm requires an explicit control kind")
    if not is_controlnet and kind != "none":
        raise MMH3ResourceError("Native/external algorithms cannot claim a ControlNet control kind")
    if kind == "inpaint" and strength_value > 0.0:
        if not _clean_text(mask_resource_id) or not _clean_text(inpaint_source_resource_id):
            raise MMH3ResourceError("Active inpaint control requires both mask and inpaint_source resource IDs")
    if kind != "inpaint" and (_clean_text(mask_resource_id) or _clean_text(inpaint_source_resource_id)):
        raise MMH3ResourceError("mask/inpaint_source resources are only legal for control kind 'inpaint'")
    if is_controlnet and kind != "inpaint" and strength_value > 0.0 and not _clean_text(control_video_resource_id):
        raise MMH3ResourceError("Active non-inpaint ControlNet requires a control_video resource ID")
    if checkpoint_sha256 and not _SHA256_RE.fullmatch(_clean_text(checkpoint_sha256).lower()):
        raise MMH3ResourceError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    if capability_fingerprint and not _SHA256_RE.fullmatch(_clean_text(capability_fingerprint).lower()):
        raise MMH3ResourceError("capability_fingerprint must be a lowercase SHA-256 digest")

    return H3ControlConfiguration(
        requested_algorithm=requested,
        effective_algorithm=effective,
        control_kind=kind,
        strength=strength_value,
        start_percent=start_value,
        end_percent=end_value,
        temporal_policy=temporal,
        control_video_resource_id=_clean_text(control_video_resource_id),
        inpaint_source_resource_id=_clean_text(inpaint_source_resource_id),
        mask_resource_id=_clean_text(mask_resource_id),
        selection_reason=reason,
        reference_composition=bool(reference_composition),
        composition_order=_clean_text(composition_order) or "reference_then_control",
        preprocessor=ControlPreprocessor(
            _clean_text(preprocessor_name),
            _clean_text(preprocessor_version),
            _canonical_settings(preprocessor_settings),
        ),
        provider=ControlProviderProvenance(
            _clean_text(provider_id) or H3_CONTROL_PROVIDER_ID,
            _clean_text(capability_fingerprint).lower(),
            _clean_text(checkpoint_sha256).lower(),
            _clean_text(quantization).lower(),
            _clean_text(dtype).lower(),
            _clean_text(base_family).lower(),
            _clean_text(adaln_form).lower(),
            _clean_text(vae_fingerprint).lower(),
        ),
    )


def control_configuration_from_dict(value: Mapping[str, Any]) -> H3ControlConfiguration:
    if not isinstance(value, Mapping):
        raise MMH3ResourceError("H3 control configuration must be an object")
    if value.get("contract") != H3_CONTROL_CONTRACT:
        raise MMH3ResourceError(f"Unsupported H3 control contract {value.get('contract')!r}")
    resources = value.get("resources")
    resources = resources if isinstance(resources, Mapping) else {}
    preprocessor = value.get("preprocessor")
    preprocessor = preprocessor if isinstance(preprocessor, Mapping) else {}
    provider = value.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    composition = value.get("reference_composition")
    composition = composition if isinstance(composition, Mapping) else {}
    return build_control_configuration(
        requested_algorithm=value.get("requested_algorithm", ""),
        effective_algorithm=value.get("effective_algorithm", ""),
        control_kind=value.get("type") or "none",
        strength=value.get("strength", 1.0),
        start_percent=value.get("start_percent", 0.0),
        end_percent=value.get("end_percent", 1.0),
        temporal_policy=value.get("temporal_policy", "strict"),
        control_video_resource_id=resources.get("control_video") or "",
        inpaint_source_resource_id=resources.get("inpaint_source") or "",
        mask_resource_id=resources.get("mask") or "",
        selection_reason=value.get("selection_reason", ""),
        reference_composition=composition.get("enabled", False),
        composition_order=composition.get("order", "reference_then_control"),
        preprocessor_name=preprocessor.get("name", ""),
        preprocessor_version=preprocessor.get("version", ""),
        preprocessor_settings=preprocessor.get("settings", {}),
        provider_id=provider.get("id", H3_CONTROL_PROVIDER_ID),
        capability_fingerprint=provider.get("capability_fingerprint", ""),
        checkpoint_sha256=provider.get("checkpoint_sha256", ""),
        quantization=provider.get("quantization", ""),
        dtype=provider.get("dtype", ""),
        base_family=provider.get("base_family", ""),
        adaln_form=provider.get("adaln_form", ""),
        vae_fingerprint=provider.get("vae_fingerprint", ""),
    )


def _validate_packet_resources(packet: MMH3Media, config: H3ControlConfiguration) -> None:
    expected = {
        "control_video": config.control_video_resource_id,
        "inpaint_source": config.inpaint_source_resource_id,
        "mask": config.mask_resource_id,
    }
    for role, resource_id in expected.items():
        if not resource_id:
            continue
        resource = packet.get_by_id(resource_id)
        if resource is None:
            raise MMH3ResourceError(f"Control resource {resource_id!r} does not exist in the packet")
        descriptor = packet.ref(resource_id).descriptor
        usage = control_usage(descriptor)
        if usage != role:
            raise MMH3ResourceError(
                f"Control resource {resource_id!r} has generic role {descriptor.get('role')!r} and H3 usage {usage!r}; expected usage {role!r}"
            )


def set_control_configuration(packet: MMH3Media, config: H3ControlConfiguration) -> MMH3Media:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    if not isinstance(config, H3ControlConfiguration):
        raise MMH3ResourceError("Expected an immutable H3ControlConfiguration")
    _validate_packet_resources(packet, config)
    return packet.set_extension_value("mmh3_media", "control", config.to_dict()).record_operation(
        "h3_control_configure",
        requested_algorithm=config.requested_algorithm,
        effective_algorithm=config.effective_algorithm,
        control_kind=config.control_kind,
        source_resource_ids=list(config.source_resource_ids),
    )


def get_control_configuration(packet: MMH3Media) -> H3ControlConfiguration | None:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    namespace = packet.manifest.get("extensions", {}).get("mmh3_media")
    if namespace is None:
        return None
    if not isinstance(namespace, Mapping):
        raise MMH3ResourceError("extensions.mmh3_media must be an object")
    value = namespace.get("control")
    if value is None:
        return None
    config = control_configuration_from_dict(value)
    _validate_packet_resources(packet, config)
    return config
