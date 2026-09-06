from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .control_contract import (
    H3_CONTROLNET_ALGORITHMS,
    H3_CONTROL_AUDIO_POLICY,
    H3_CONTROL_KINDS,
    H3_CONTROL_MASK_POLICY,
    H3ControlConfiguration,
    ControlDiagnostic,
)
from .errors import MMH3ResourceError
from .util import deep_copy_json, json_dumps_canonical


CONTROLNET_LOADER_NODE_ID = "ControlNetLoader"
H3_CONTROL_APPLY_NODE_ID = "MiniMaxH3FunControlNetApply"
H3_CONTROL_GEOMETRY_DIVISOR = 32
H3_CONTROL_INPAINT_INPUT_DIM = 49
H3_CONTROL_TEMPORAL_GRID = "17n+5"
H3_CONTROL_INJECTION_LAYERS = (0, 10, 20, 30, 40)
H3_CONTROL_QUANTIZATIONS = ("bf16", "int8_convrot")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RuntimeNodeContract:
    node_id: str
    fingerprint: str
    required_inputs: tuple[tuple[str, str], ...]
    optional_inputs: tuple[tuple[str, str], ...]
    output_types: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "fingerprint": self.fingerprint,
            "required_inputs": dict(self.required_inputs),
            "optional_inputs": dict(self.optional_inputs),
            "output_types": list(self.output_types),
        }


@dataclass(frozen=True)
class H3ControlCapabilityReport:
    diagnostics: tuple[ControlDiagnostic, ...]
    facts: dict[str, Any]
    loader_contract: RuntimeNodeContract | None = None
    apply_contract: RuntimeNodeContract | None = None

    @property
    def ready(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    @property
    def capability_fingerprint(self) -> str:
        value = self.facts.get("capability_fingerprint")
        return str(value or "")

    def summary(self) -> str:
        head = "READY" if self.ready else "BLOCKED"
        errors = sum(item.severity == "error" for item in self.diagnostics)
        warnings = sum(item.severity == "warning" for item in self.diagnostics)
        lines = [f"{head} · F16 control · errors={errors} warnings={warnings}"]
        lines.extend(f"{item.severity.upper()} [{item.code}] {item.message}" for item in self.diagnostics)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "facts": deep_copy_json(self.facts),
            "loader_contract": self.loader_contract.to_dict() if self.loader_contract else None,
            "apply_contract": self.apply_contract.to_dict() if self.apply_contract else None,
        }


def _input_type(spec: Any) -> str | None:
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    first = spec[0]
    if isinstance(first, str):
        return first
    if isinstance(first, (list, tuple)):
        return "COMBO"
    return None


def _schema_parts(info: Mapping[str, Any], node_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[str, ...]]:
    if not isinstance(info, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {node_id} is not an object")
    inputs = info.get("input")
    if not isinstance(inputs, Mapping):
        inputs = info.get("inputs")
    if not isinstance(inputs, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {node_id} has no input map")
    required = inputs.get("required")
    optional = inputs.get("optional")
    if not isinstance(required, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {node_id} has no required-input map")
    optional = optional if isinstance(optional, Mapping) else {}
    outputs = info.get("output")
    if not isinstance(outputs, (list, tuple)):
        outputs = info.get("outputs")
    if not isinstance(outputs, (list, tuple)):
        raise MMH3ResourceError(f"Runtime contract for {node_id} has no output type list")
    return required, optional, tuple(str(item) for item in outputs)


def _require_types(actual: Mapping[str, Any], expected: Mapping[str, str], node_id: str, group: str) -> None:
    missing = sorted(set(expected).difference(actual))
    if missing:
        raise MMH3ResourceError(f"Runtime node {node_id} is missing {group} inputs: {', '.join(missing)}")
    for name, expected_type in expected.items():
        actual_type = _input_type(actual[name])
        if actual_type != expected_type:
            raise MMH3ResourceError(
                f"Runtime node {node_id} input {name!r} changed type: expected {expected_type}, got {actual_type!r}"
            )


def _node_contract(node_id: str, info: Mapping[str, Any], required_types: Mapping[str, str], optional_types: Mapping[str, str], outputs: tuple[str, ...]) -> RuntimeNodeContract:
    required, optional, actual_outputs = _schema_parts(info, node_id)
    _require_types(required, required_types, node_id, "required")
    _require_types(optional, optional_types, node_id, "optional")
    if actual_outputs != outputs:
        raise MMH3ResourceError(
            f"Runtime node {node_id} output contract changed: expected {outputs}, got {actual_outputs}"
        )
    fingerprint = hashlib.sha256(json_dumps_canonical(dict(info)).encode("utf-8")).hexdigest()
    return RuntimeNodeContract(
        node_id,
        fingerprint,
        tuple((name, _input_type(spec) or "") for name, spec in required.items()),
        tuple((name, _input_type(spec) or "") for name, spec in optional.items()),
        actual_outputs,
    )


def parse_controlnet_loader_contract(
    info: Mapping[str, Any], *, node_id: str = CONTROLNET_LOADER_NODE_ID
) -> RuntimeNodeContract:
    return _node_contract(node_id, info, {"control_net_name": "COMBO"}, {}, ("CONTROL_NET",))


def parse_h3_control_apply_contract(
    info: Mapping[str, Any], *, node_id: str = H3_CONTROL_APPLY_NODE_ID
) -> RuntimeNodeContract:
    return _node_contract(
        node_id,
        info,
        {
            "positive": "CONDITIONING",
            "control_net": "CONTROL_NET",
            "vae": "VAE",
            "strength": "FLOAT",
            "start_percent": "FLOAT",
            "end_percent": "FLOAT",
        },
        {"control_video": "IMAGE", "mask": "MASK", "source_video": "IMAGE"},
        ("CONDITIONING",),
    )


def _add(items: list[ControlDiagnostic], severity: str, code: str, message: str) -> None:
    items.append(ControlDiagnostic(severity, code, message))


def _text(mapping: Mapping[str, Any], key: str) -> str:
    return str(mapping.get(key) or "").strip().lower()


def _unresolved_placeholder(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized.startswith(("replace_with_", "<replace_", "${"))


def _positive_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _valid_h3_frames(frames: int) -> bool:
    return frames >= 5 and (frames - 5) % 17 == 0


def preflight_h3_control(
    config: H3ControlConfiguration,
    *,
    runtime_nodes: Sequence[str] | None = None,
    runtime_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    loader_node_id: str = CONTROLNET_LOADER_NODE_ID,
    apply_node_id: str = H3_CONTROL_APPLY_NODE_ID,
    checkpoint: Mapping[str, Any] | None = None,
    base: Mapping[str, Any] | None = None,
    media: Mapping[str, Any] | None = None,
) -> H3ControlCapabilityReport:
    if not isinstance(config, H3ControlConfiguration):
        raise MMH3ResourceError("Expected an immutable H3ControlConfiguration")
    diagnostics: list[ControlDiagnostic] = []
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    base = base if isinstance(base, Mapping) else {}
    media = media if isinstance(media, Mapping) else {}
    loader_contract = apply_contract = None

    facts: dict[str, Any] = {
        "requested_algorithm": config.requested_algorithm,
        "effective_algorithm": config.effective_algorithm,
        "selection_reason": config.selection_reason,
        "control_kind": config.control_kind,
        "accepted_control_kinds": list(H3_CONTROL_KINDS),
        "temporal_policy": config.temporal_policy,
        "temporal_grid": H3_CONTROL_TEMPORAL_GRID,
        "geometry_divisor": H3_CONTROL_GEOMETRY_DIVISOR,
        "mask_policy": H3_CONTROL_MASK_POLICY,
        "audio_policy": H3_CONTROL_AUDIO_POLICY,
        "audio_rows_controlled": False,
        "control_in_dim": H3_CONTROL_INPAINT_INPUT_DIM,
        "injection_layers": list(H3_CONTROL_INJECTION_LAYERS),
        "strength": config.strength,
        "start_percent": config.start_percent,
        "end_percent": config.end_percent,
        "neutral_semantics": "strength=0 or no control input is pass-through",
        "chaining": "previous conditioning control is preserved and linked",
        "memory": {
            "offload_capability": "provider_owned_runtime_probe_required",
            "vram_claim": None,
        },
    }

    if config.effective_algorithm not in H3_CONTROLNET_ALGORITHMS:
        _add(diagnostics, "info", "controlnet_not_selected", "A preserved native/external algorithm is selected; ControlNet capability checks are not applicable.")
        facts["capability_fingerprint"] = ""
        return H3ControlCapabilityReport(tuple(diagnostics), facts)

    if config.control_kind != "inpaint" or config.control_video_resource_id:
        if not config.preprocessor.name or not config.preprocessor.version:
            _add(
                diagnostics,
                "error",
                "control_preprocessor_provenance_missing",
                "Structural control requires exact external preprocessor name and version provenance.",
            )
        elif _unresolved_placeholder(config.preprocessor.name) or _unresolved_placeholder(config.preprocessor.version):
            _add(
                diagnostics,
                "error",
                "control_preprocessor_provenance_unresolved",
                "Structural control preprocessor provenance still contains a template placeholder.",
            )

    if runtime_nodes is None:
        _add(diagnostics, "error", "runtime_inventory_missing", "ControlNet preflight requires an explicit runtime node inventory.")
    else:
        available = set(runtime_nodes)
        for node_id in (loader_node_id, apply_node_id):
            if node_id not in available:
                _add(diagnostics, "error", "required_node_missing", f"Required runtime node {node_id!r} is not registered.")

    if runtime_contracts is None:
        _add(diagnostics, "error", "runtime_schema_inventory_missing", "Exact loader/apply schemas were not supplied.")
    else:
        loader_info = runtime_contracts.get(loader_node_id)
        apply_info = runtime_contracts.get(apply_node_id)
        if loader_info is None:
            _add(diagnostics, "error", "loader_schema_missing", f"Exact schema for {loader_node_id!r} was not supplied.")
        else:
            try:
                loader_contract = parse_controlnet_loader_contract(loader_info, node_id=loader_node_id)
            except MMH3ResourceError as exc:
                _add(diagnostics, "error", "loader_schema_incompatible", str(exc))
        if apply_info is None:
            _add(diagnostics, "error", "apply_schema_missing", f"Exact schema for {apply_node_id!r} was not supplied.")
        else:
            try:
                apply_contract = parse_h3_control_apply_contract(apply_info, node_id=apply_node_id)
            except MMH3ResourceError as exc:
                _add(diagnostics, "error", "apply_schema_incompatible", str(exc))

    expected_quant = "bf16" if config.effective_algorithm.endswith("_bf16") else "int8_convrot"
    actual_quant = _text(checkpoint, "quantization") or config.provider.quantization
    checkpoint_hash = _text(checkpoint, "sha256") or config.provider.checkpoint_sha256
    checkpoint_dtype = _text(checkpoint, "dtype") or config.provider.dtype
    checkpoint_family = _text(checkpoint, "base_family") or config.provider.base_family
    checkpoint_adaln = _text(checkpoint, "adaln_form") or config.provider.adaln_form
    control_in_dim = _positive_int(checkpoint, "control_in_dim")
    facts["checkpoint"] = {
        "sha256": checkpoint_hash,
        "sha256_verified": checkpoint.get("sha256_verified") is True,
        "quantization": actual_quant,
        "dtype": checkpoint_dtype,
        "base_family": checkpoint_family,
        "adaln_form": checkpoint_adaln,
        "control_in_dim": control_in_dim,
    }
    if not _SHA256_RE.fullmatch(checkpoint_hash):
        _add(diagnostics, "error", "checkpoint_hash_unproven", "A lowercase SHA-256 for the selected ControlNet checkpoint is required.")
    elif checkpoint.get("sha256_verified") is not True:
        _add(diagnostics, "error", "checkpoint_hash_unverified", "ControlNet checkpoint SHA-256 must be verified from the selected file, not only declared.")
    if actual_quant != expected_quant:
        _add(diagnostics, "error", "checkpoint_quantization_mismatch", f"Algorithm {config.effective_algorithm!r} requires quantization {expected_quant!r}; got {actual_quant or 'unknown'!r}.")
    if expected_quant == "bf16" and checkpoint_dtype not in ("bf16", "bfloat16"):
        _add(diagnostics, "error", "checkpoint_dtype_mismatch", f"BF16 provider requires bf16/bfloat16 checkpoint dtype; got {checkpoint_dtype or 'unknown'!r}.")
    if config.control_kind == "inpaint" and control_in_dim != H3_CONTROL_INPAINT_INPUT_DIM:
        _add(diagnostics, "error", "inpaint_input_width_mismatch", f"Inpaint requires control_in_dim={H3_CONTROL_INPAINT_INPUT_DIM}; got {control_in_dim!r}.")

    base_family = _text(base, "family")
    base_adaln = _text(base, "adaln_form")
    base_vae = _text(base, "vae_fingerprint")
    control_vae = _text(checkpoint, "vae_fingerprint") or config.provider.vae_fingerprint
    facts["base"] = {
        "family": base_family,
        "adaln_form": base_adaln,
        "vae_fingerprint": base_vae,
        "cfg": base.get("cfg"),
        "guidance_distilled": base.get("guidance_distilled"),
    }
    if not checkpoint_family or not base_family:
        _add(diagnostics, "error", "base_family_unproven", "Both base and ControlNet checkpoint family must be explicit.")
    elif checkpoint_family != base_family:
        _add(diagnostics, "error", "base_family_mismatch", f"ControlNet family {checkpoint_family!r} does not match base family {base_family!r}.")
    if not checkpoint_adaln or not base_adaln:
        _add(diagnostics, "error", "adaln_form_unproven", "Both base and ControlNet AdaLN form must be explicit.")
    elif checkpoint_adaln != base_adaln:
        _add(diagnostics, "error", "adaln_form_mismatch", f"ControlNet AdaLN form {checkpoint_adaln!r} does not match base form {base_adaln!r}.")
    if not control_vae or not base_vae:
        _add(diagnostics, "error", "vae_fingerprint_unproven", "Exact base/control VAE fingerprints are required.")
    elif _unresolved_placeholder(control_vae) or _unresolved_placeholder(base_vae):
        _add(diagnostics, "error", "vae_fingerprint_unresolved", "Base/control VAE fingerprint still contains a template placeholder.")
    elif control_vae != base_vae:
        _add(diagnostics, "error", "vae_fingerprint_mismatch", "Control inputs and the base conditioning use different VAE fingerprints.")

    if base.get("guidance_distilled") is True:
        try:
            cfg = float(base.get("cfg"))
        except (TypeError, ValueError):
            cfg = float("nan")
        if cfg != 1.0:
            _add(diagnostics, "error", "guidance_distilled_cfg", f"Guidance-distilled H3 requires CFG=1; got {base.get('cfg')!r}.")

    width = _positive_int(media, "width")
    height = _positive_int(media, "height")
    target_frames = _positive_int(media, "target_frames")
    control_frames = _positive_int(media, "control_frames")
    facts["media"] = {
        "width": width,
        "height": height,
        "target_frames": target_frames,
        "control_frames": control_frames,
    }
    if width is None or height is None or width % H3_CONTROL_GEOMETRY_DIVISOR or height % H3_CONTROL_GEOMETRY_DIVISOR:
        _add(diagnostics, "error", "control_geometry_invalid", "Control geometry must provide positive width/height divisible by 32.")
    if target_frames is None or not _valid_h3_frames(target_frames):
        _add(diagnostics, "error", "target_temporal_grid_invalid", "Target frames must lie on the legal H3 grid 17n+5.")
    if config.controlnet_active:
        if control_frames is None:
            _add(diagnostics, "error", "control_timeline_unknown", "Active control requires an explicit control frame count.")
        elif target_frames is not None and control_frames != target_frames:
            if config.temporal_policy == "strict":
                _add(diagnostics, "error", "control_timeline_mismatch", f"strict policy requires {target_frames} control frames; got {control_frames}.")
            elif config.temporal_policy == "trim" and control_frames < target_frames:
                _add(diagnostics, "error", "control_timeline_too_short_for_trim", "trim policy cannot extend a short control timeline.")
            elif config.temporal_policy == "pad_hold_last" and control_frames > target_frames:
                _add(diagnostics, "error", "control_timeline_too_long_for_pad", "pad_hold_last policy cannot shorten a long control timeline.")
            else:
                _add(diagnostics, "warning", "control_timeline_alignment", f"Control timeline {control_frames} will be aligned to {target_frames} using explicit policy {config.temporal_policy!r}.")

    if config.control_kind == "inpaint":
        if not config.mask_resource_id or not config.inpaint_source_resource_id:
            _add(diagnostics, "error", "inpaint_resources_missing", "Inpaint requires source_video and mask; white mask pixels are regenerated.")
    elif config.inpaint_source_resource_id or config.mask_resource_id:
        _add(diagnostics, "error", "inpaint_resources_unexpected", "source_video/mask may only be used by inpaint control.")

    loader_fp = loader_contract.fingerprint if loader_contract else ""
    apply_fp = apply_contract.fingerprint if apply_contract else ""
    capability_material = {
        "provider": config.provider.provider_id,
        "loader_node_id": loader_node_id,
        "loader_schema_fingerprint": loader_fp,
        "apply_node_id": apply_node_id,
        "apply_schema_fingerprint": apply_fp,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_sha256_verified": checkpoint.get("sha256_verified") is True,
        "quantization": actual_quant,
        "dtype": checkpoint_dtype,
        "base_family": base_family,
        "adaln_form": base_adaln,
        "vae_fingerprint": base_vae,
        "control_in_dim": control_in_dim,
        "accepted_control_kinds": list(H3_CONTROL_KINDS),
        "geometry_divisor": H3_CONTROL_GEOMETRY_DIVISOR,
        "temporal_grid": H3_CONTROL_TEMPORAL_GRID,
        "audio_rows_controlled": False,
    }
    facts["capability_fingerprint"] = hashlib.sha256(
        json_dumps_canonical(capability_material).encode("utf-8")
    ).hexdigest()
    facts["capability"] = capability_material
    if config.provider.capability_fingerprint and config.provider.capability_fingerprint != facts["capability_fingerprint"]:
        _add(diagnostics, "error", "capability_fingerprint_mismatch", "Recorded provider capability fingerprint does not match the current exact schemas/checkpoint/base contract.")
    if not any(item.severity == "error" for item in diagnostics):
        _add(diagnostics, "info", "control_capability_ready", "Selected H3 ControlNet provider passed schema, checkpoint, base, VAE, geometry, temporal and audio-owner checks.")
    return H3ControlCapabilityReport(tuple(diagnostics), facts, loader_contract, apply_contract)
