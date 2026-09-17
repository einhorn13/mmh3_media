from __future__ import annotations

from dataclasses import dataclass, replace
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import MMH3ResourceError
from .util import deep_copy_json
from .sampling_presets import (
    SAMPLING_PRESET_CONTRACT,
    VDN_DMD_PROFILE,
    VDN_STAGE_B_PROFILE,
    build_sampling_preset,
)

VDN_NODE_ID = "ApplyVDNH3"
VDN_SUPPORTED_TASK_FAMILIES = ("fl2va", "ref2va")
VDN_LORA_MODES = ("merge", "bypass")
VDN_BRANCH_WEIGHT_MODES = ("auto", "stream", "cache_gpu")
VDN_RETAIN_BUFFER_MODES = ("auto", "on", "off")
VDN_ATTENTION_BACKENDS = ("grouped", "flex")
VDN_AUTO_CHECKPOINT = "auto"
VDN_DEFAULT_CHECKPOINT = "auto"
VDN_KNOWN_DMD_ALIASES = (
    "vdn-minimax-h3-int8-convrot-comfyui",
    "stage-dmd-step-250-int8_convrot_comfyui",
    "stage-dmd-step-250",
)
VDN_RUNTIME_REQUIRED_INPUTS = {
    "model",
    "vdn_checkpoint",
    "apply_turbo_adapter",
    "strength",
    "lora_mode",
    "branch_weights",
    "retain_buffers",
    "verbose",
    "attention_backend",
}


def _warn_vdn(message: str) -> None:
    """Emit a visible warning for an experimental but technically runnable VDN setup."""
    warnings.warn(f"MMH3 VDN warning: {message}", RuntimeWarning, stacklevel=3)


def _vdn_model_roots() -> tuple[Path, ...]:
    """Return configured VDN model roots without requiring the external node.

    ComfyUI-VDN-H3 normally registers a ``vdn`` model category, but MMH3 may be
    imported before that custom node depending on startup order.  Fall back to
    ``models/vdn`` so the UI can still expose installed stage directories.
    """
    roots: list[Path] = []
    try:
        import folder_paths  # type: ignore

        registered = getattr(folder_paths, "folder_names_and_paths", {}).get("vdn")
        if isinstance(registered, (tuple, list)) and registered:
            candidates = registered[0]
            if isinstance(candidates, (str, Path)):
                candidates = [candidates]
            if isinstance(candidates, (tuple, list, set)):
                roots.extend(Path(item) for item in candidates if item)
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            roots.append(Path(models_dir) / "vdn")
    except Exception:
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return tuple(unique)


def discover_vdn_checkpoints() -> tuple[str, ...]:
    """Discover stage directory names for the VDN checkpoint combo.

    Runtime validation still uses ApplyVDNH3.INPUT_TYPES() as the authority; this
    filesystem scan is only for a useful dropdown during schema construction.
    """
    found: set[str] = set()
    for root in _vdn_model_roots():
        try:
            for child in root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    found.add(child.name)
        except OSError:
            continue
    return tuple(sorted(found, key=str.casefold))


def vdn_checkpoint_options() -> list[str]:
    return [VDN_AUTO_CHECKPOINT, *discover_vdn_checkpoints()]


def _runtime_choices(schema: Mapping[str, Any], key: str) -> tuple[str, ...]:
    fields = {**(schema.get("required", {}) or {}), **(schema.get("optional", {}) or {})}
    spec = fields.get(key)
    choices = spec[0] if isinstance(spec, (tuple, list)) and spec else None
    if isinstance(choices, (tuple, list)):
        return tuple(str(choice) for choice in choices)
    return ()


def _vdn_checkpoint_family(name: str) -> str | None:
    lowered = str(name).strip().lower().replace("_", "-")
    if lowered in {item.lower().replace("_", "-") for item in VDN_KNOWN_DMD_ALIASES}:
        return "dmd"
    if "stage-dmd" in lowered or "-dmd-" in lowered:
        return "dmd"
    if "stage-b" in lowered or lowered.startswith("stageb-"):
        return "stage_b"
    return None


def _checkpoint_preference(name: str, family: str) -> tuple[int, str]:
    lowered = str(name).lower()
    score = 0
    if family == "dmd":
        if name == "vdn-minimax-h3-int8-convrot-comfyui":
            score += 100
        if "int8" in lowered and "convrot" in lowered:
            score += 50
        if name == "stage-dmd-step-250":
            score += 20
    elif family == "stage_b":
        if name == "stage-b-step-2000":
            score += 100
        if "int8" in lowered and "convrot" in lowered:
            score += 20
    return (-score, lowered)


def resolve_vdn_checkpoint(
    settings: "VDNOptimizationSettings",
    available: Sequence[str],
) -> str:
    """Resolve ``auto`` (and the legacy hard-coded DMD default) safely.

    Auto never picks a stage from the wrong trained trajectory.  Unknown custom
    stage names remain selectable explicitly but are not guessed by auto.
    """
    choices = tuple(dict.fromkeys(str(item) for item in available if str(item).strip()))
    requested = str(settings.checkpoint or "").strip()
    family = "dmd" if settings.apply_turbo_adapter else "stage_b"

    if requested and requested != VDN_AUTO_CHECKPOINT and requested in choices:
        return requested

    legacy_dmd_default = requested == "stage-dmd-step-250" and settings.apply_turbo_adapter
    if requested not in {"", VDN_AUTO_CHECKPOINT} and not legacy_dmd_default:
        raise MMH3ResourceError(
            f"Installed ApplyVDNH3 does not support vdn_checkpoint={requested!r}; "
            f"available: {list(choices)!r}"
        )

    compatible = [name for name in choices if _vdn_checkpoint_family(name) == family]
    if not compatible:
        label = "DMD 8-step" if family == "dmd" else "Stage-B 50-step"
        raise MMH3ResourceError(
            f"VDN checkpoint='auto' could not find a compatible {label} stage. "
            f"Available ApplyVDNH3 checkpoints: {list(choices)!r}. "
            "Install the matching stage under ComfyUI/models/vdn or select a compatible checkpoint explicitly."
        )
    compatible.sort(key=lambda name: _checkpoint_preference(name, family))
    return compatible[0]


@dataclass(frozen=True)
class VDNOptimizationSettings:
    task_family: str
    checkpoint: str
    apply_turbo_adapter: bool
    strength: float
    lora_mode: str
    branch_weights: str
    retain_buffers: str
    attention_backend: str
    verbose: bool

    @property
    def recommended_steps(self) -> int:
        return 8 if self.apply_turbo_adapter else 50

    @property
    def required_sampling_profile(self) -> str:
        return VDN_DMD_PROFILE if self.apply_turbo_adapter else VDN_STAGE_B_PROFILE

    @property
    def required_trajectory(self) -> str:
        return "vdn_dmd8" if self.apply_turbo_adapter else "vdn_stage_b50"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": "Saganaki22/ComfyUI-VDN-H3",
            "runtime_node": VDN_NODE_ID,
            "task_family": self.task_family,
            "checkpoint": self.checkpoint,
            "apply_turbo_adapter": self.apply_turbo_adapter,
            "recommended_steps": self.recommended_steps,
            "required_sampling_profile": self.required_sampling_profile,
            "required_trajectory": self.required_trajectory,
            "strength": self.strength,
            "lora_mode": self.lora_mode,
            "branch_weights": self.branch_weights,
            "retain_buffers": self.retain_buffers,
            "attention_backend": self.attention_backend,
            "verbose": self.verbose,
            "compatibility": {
                "sol_attn": False,
                "h3_sla": False,
                "task_families": list(VDN_SUPPORTED_TASK_FAMILIES),
            },
        }


def build_vdn_settings(
    *,
    task_family: str,
    checkpoint: str = VDN_DEFAULT_CHECKPOINT,
    apply_turbo_adapter: bool = True,
    strength: float = 1.0,
    lora_mode: str = "merge",
    branch_weights: str = "auto",
    retain_buffers: str = "auto",
    attention_backend: str = "grouped",
    verbose: bool = False,
) -> VDNOptimizationSettings:
    family = str(task_family or "").strip().lower()
    if family not in VDN_SUPPORTED_TASK_FAMILIES:
        raise MMH3ResourceError(
            f"VDN-H3 supports task_family {VDN_SUPPORTED_TASK_FAMILIES}; got {family!r}"
        )
    checkpoint = str(checkpoint or "").strip()
    if not checkpoint:
        raise MMH3ResourceError("VDN-H3 requires a checkpoint directory name under ComfyUI/models/vdn")
    if not 0.0 <= float(strength) <= 2.0:
        raise MMH3ResourceError("VDN-H3 strength must be between 0.0 and 2.0")
    if lora_mode not in VDN_LORA_MODES:
        raise MMH3ResourceError(f"Unsupported VDN-H3 lora_mode {lora_mode!r}")
    if branch_weights not in VDN_BRANCH_WEIGHT_MODES:
        raise MMH3ResourceError(f"Unsupported VDN-H3 branch_weights mode {branch_weights!r}")
    if retain_buffers not in VDN_RETAIN_BUFFER_MODES:
        raise MMH3ResourceError(f"Unsupported VDN-H3 retain_buffers mode {retain_buffers!r}")
    if attention_backend not in VDN_ATTENTION_BACKENDS:
        raise MMH3ResourceError(f"Unsupported VDN-H3 attention backend {attention_backend!r}")
    if apply_turbo_adapter and lora_mode != "merge":
        _warn_vdn(
            "VDN-H3 8-step/DMD is trained for lora_mode='merge'; "
            f"requested lora_mode={lora_mode!r}. Running unchanged as an experiment."
        )
    return VDNOptimizationSettings(
        family,
        checkpoint,
        bool(apply_turbo_adapter),
        float(strength),
        lora_mode,
        branch_weights,
        retain_buffers,
        attention_backend,
        bool(verbose),
    )



def validate_vdn_sampling_profile(
    settings: VDNOptimizationSettings,
    sampling_profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate VDN sampling and warn for off-training combinations.

    Resource/API failures remain hard errors, but sampling choices are user-owned.
    Known VDN recipe mismatches are warnings and are passed through unchanged so
    experimental trajectories (for example 6-step DMD) can reach ApplyVDNH3.
    """
    if not isinstance(sampling_profile, Mapping):
        raise MMH3ResourceError(
            "VDN-H3 requires a sampling contract from H3 Sampling. "
            f"Select {settings.required_sampling_profile!r} before H3 Optimizations."
        )
    profile = deep_copy_json(dict(sampling_profile))
    if str(profile.get("contract") or "") != SAMPLING_PRESET_CONTRACT:
        raise MMH3ResourceError("VDN-H3 received an unsupported sampling contract")

    if str(profile.get("task_family") or "") != settings.task_family:
        _warn_vdn(
            "sampling/model task-family mismatch: "
            f"sampling={profile.get('task_family')!r}, VDN={settings.task_family!r}. "
            "Running unchanged as an experiment."
        )

    actual_profile = str(profile.get("profile") or "")
    actual_trajectory = str(profile.get("trajectory") or "")
    try:
        actual_steps = int(profile.get("steps"))
    except (TypeError, ValueError):
        actual_steps = -1

    if (
        actual_profile != settings.required_sampling_profile
        or actual_trajectory != settings.required_trajectory
        or actual_steps != settings.recommended_steps
    ):
        _warn_vdn(
            "trajectory mismatch: selected stage is trained for "
            f"{settings.required_sampling_profile} / {settings.required_trajectory} / "
            f"{settings.recommended_steps} steps; got profile={actual_profile!r}, "
            f"trajectory={actual_trajectory!r}, steps={actual_steps}. "
            "Running the requested sampling trajectory unchanged."
        )

    if profile.get("recommended_lora") or profile.get("adapter"):
        _warn_vdn(
            "sampling carries an ordinary H3 acceleration/Turbo adapter while VDN normally owns "
            "its released DMD adapter. The requested profile is preserved; another optimization "
            "contract may still reject physically incompatible stacking."
        )

    expected = build_sampling_preset(
        profile=settings.required_sampling_profile, task_family=settings.task_family
    ).to_dict()
    for key in ("steps", "sampler", "scheduler", "video_shift", "audio_shift", "sigma_preset", "vdn_required"):
        if profile.get(key) != expected[key]:
            _warn_vdn(
                f"sampling contract mismatch for {key}: trained/recommended {expected[key]!r}, "
                f"requested {profile.get(key)!r}. Running unchanged as an experiment."
            )
    return profile


def validate_vdn_runtime_node(node_class: Any) -> None:
    if node_class is None:
        raise MMH3ResourceError(
            "VDN-H3 runtime is missing; install/update Saganaki22/ComfyUI-VDN-H3 "
            "and restart ComfyUI"
        )
    input_types = getattr(node_class, "INPUT_TYPES", None)
    if not callable(input_types):
        raise MMH3ResourceError("ApplyVDNH3 has no callable INPUT_TYPES contract")
    schema = input_types()
    required = set((schema or {}).get("required", {})) | set((schema or {}).get("optional", {}))
    missing = sorted(VDN_RUNTIME_REQUIRED_INPUTS - required)
    if missing:
        raise MMH3ResourceError(
            "Installed ApplyVDNH3 schema is incompatible; missing inputs: " + ", ".join(missing)
        )
    if tuple(getattr(node_class, "RETURN_TYPES", ())) != ("MODEL",):
        raise MMH3ResourceError("Installed ApplyVDNH3 must return exactly one MODEL")
    function_name = str(getattr(node_class, "FUNCTION", ""))
    if not function_name or not callable(getattr(node_class, function_name, None)):
        raise MMH3ResourceError("Installed ApplyVDNH3 has an invalid FUNCTION contract")


def apply_external_vdn(model: Any, node_class: Any, settings: VDNOptimizationSettings, sampling_profile: Mapping[str, Any] | None = None):
    sampling = validate_vdn_sampling_profile(settings, sampling_profile)
    from .optimization_contract import validate_optimization_application, record_optimization
    validate_optimization_application(model, "vdn_h3", sampling_profile=sampling)
    validate_vdn_runtime_node(node_class)
    schema = node_class.INPUT_TYPES()
    inputs = {**schema.get("required", {}), **schema.get("optional", {})}
    runtime_checkpoints = _runtime_choices(schema, "vdn_checkpoint")
    requested_checkpoint = settings.checkpoint
    if runtime_checkpoints:
        resolved_checkpoint = resolve_vdn_checkpoint(settings, runtime_checkpoints)
    elif requested_checkpoint == VDN_AUTO_CHECKPOINT:
        raise MMH3ResourceError(
            "VDN checkpoint='auto' requires ApplyVDNH3 to expose its checkpoint dropdown; "
            "update Saganaki22/ComfyUI-VDN-H3 and restart ComfyUI"
        )
    else:
        # Older/test runtimes may use an unconstrained string contract. Explicit
        # selections remain compatible, while auto deliberately requires discovery.
        resolved_checkpoint = requested_checkpoint
    resolved_settings = replace(settings, checkpoint=resolved_checkpoint)
    for key, value in (
        ("lora_mode", resolved_settings.lora_mode),
        ("branch_weights", resolved_settings.branch_weights),
        ("retain_buffers", resolved_settings.retain_buffers),
        ("attention_backend", resolved_settings.attention_backend),
    ):
        spec = inputs[key]
        choices = spec[0] if isinstance(spec, (tuple, list)) and spec else None
        if isinstance(choices, (tuple, list)) and value not in choices:
            raise MMH3ResourceError(
                f"Installed ApplyVDNH3 does not support {key}={value!r}; available: {choices!r}"
            )
    # Keep checkpoint/trajectory selection independent of the user's LoRA
    # policy: replacing the DMD adapter must not switch to a Stage-B checkpoint.
    lora_selection = sampling.get("turbo_loras_override") or {}
    runtime_turbo_adapter = resolved_settings.apply_turbo_adapter
    if lora_selection.get("mode") in {"custom", "disabled"}:
        runtime_turbo_adapter = False
    instance = node_class()
    function_name = str(getattr(node_class, "FUNCTION"))
    result = getattr(instance, function_name)(
        model=model,
        vdn_checkpoint=resolved_settings.checkpoint,
        apply_turbo_adapter=runtime_turbo_adapter,
        strength=resolved_settings.strength,
        lora_mode=resolved_settings.lora_mode,
        branch_weights=resolved_settings.branch_weights,
        retain_buffers=resolved_settings.retain_buffers,
        attention_backend=resolved_settings.attention_backend,
        verbose=resolved_settings.verbose,
    )
    if not isinstance(result, (tuple, list)) or len(result) != 1:
        raise MMH3ResourceError("ApplyVDNH3 returned an unexpected result shape")
    record_optimization(result[0], "vdn_h3")
    profile = {
        "contract": "mmh3_vdn_h3_v1",
        "enabled": True,
        "vdn": deep_copy_json(resolved_settings.to_dict()),
        "sampling": sampling,
    }
    profile["vdn"]["requested_checkpoint"] = requested_checkpoint
    profile["vdn"]["apply_turbo_adapter"] = runtime_turbo_adapter
    if lora_selection:
        profile["vdn"]["turbo_loras_override"] = deep_copy_json(lora_selection)
    checkpoint_status = resolved_checkpoint
    if requested_checkpoint != resolved_checkpoint:
        checkpoint_status = f"{requested_checkpoint} -> {resolved_checkpoint}"
    status = (
        f"READY · VDN-H3 · {resolved_settings.task_family} · {checkpoint_status} · "
        f"recommended_steps={resolved_settings.recommended_steps}"
    )
    return result[0], profile, status


__all__ = [
    "VDN_ATTENTION_BACKENDS",
    "VDN_AUTO_CHECKPOINT",
    "VDN_BRANCH_WEIGHT_MODES",
    "VDN_DEFAULT_CHECKPOINT",
    "VDN_LORA_MODES",
    "VDN_NODE_ID",
    "VDN_RETAIN_BUFFER_MODES",
    "VDN_SUPPORTED_TASK_FAMILIES",
    "VDNOptimizationSettings",
    "apply_external_vdn",
    "build_vdn_settings",
    "discover_vdn_checkpoints",
    "resolve_vdn_checkpoint",
    "vdn_checkpoint_options",
    "validate_vdn_runtime_node",
    "validate_vdn_sampling_profile",
]
