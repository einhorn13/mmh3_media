from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
VDN_BRANCH_WEIGHT_MODES = ("auto", "stream", "cache_gpu", "resident")
VDN_RETAIN_BUFFER_MODES = ("auto", "on", "off")
VDN_ATTENTION_BACKENDS = ("grouped", "flex")
VDN_DEFAULT_CHECKPOINT = "stage-dmd-step-250"
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
        raise MMH3ResourceError(
            "VDN-H3 8-step/DMD mode requires lora_mode='merge'; bypass is intentionally refused"
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
    """Require the sampling trajectory that the selected VDN stage was trained for.

    VDN DMD is not a generic attention patch that can be dropped onto ordinary H3 Turbo
    or a 20-step base trajectory.  The caller must provide a machine-readable H3 sampling
    profile and it must match the VDN stage exactly.
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
        raise MMH3ResourceError(
            "VDN-H3 sampling/model task-family mismatch: "
            f"sampling={profile.get('task_family')!r}, VDN={settings.task_family!r}"
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
        raise MMH3ResourceError(
            "VDN-H3 trajectory mismatch: "
            f"selected stage requires {settings.required_sampling_profile} / "
            f"{settings.required_trajectory} / {settings.recommended_steps} steps; "
            f"got profile={actual_profile!r}, trajectory={actual_trajectory!r}, steps={actual_steps}. "
            "Do not stack ordinary H3 Turbo/Standard sampling with VDN."
        )
    if profile.get("recommended_lora") or profile.get("adapter"):
        raise MMH3ResourceError(
            "VDN-H3 sampling must not carry an ordinary H3 acceleration/Turbo adapter; "
            "the released VDN DMD adapter is owned by ApplyVDNH3"
        )
    expected = build_sampling_preset(
        profile=settings.required_sampling_profile, task_family=settings.task_family
    ).to_dict()
    for key in ("steps", "sampler", "scheduler", "video_shift", "audio_shift", "sigma_preset", "vdn_required"):
        if profile.get(key) != expected[key]:
            raise MMH3ResourceError(
                f"VDN-H3 sampling contract mismatch for {key}: "
                f"expected {expected[key]!r}, got {profile.get(key)!r}"
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
    for key, value in (
        ("vdn_checkpoint", settings.checkpoint),
        ("lora_mode", settings.lora_mode),
        ("branch_weights", settings.branch_weights),
        ("retain_buffers", settings.retain_buffers),
        ("attention_backend", settings.attention_backend),
    ):
        spec = inputs[key]
        choices = spec[0] if isinstance(spec, (tuple, list)) and spec else None
        if isinstance(choices, (tuple, list)) and value not in choices:
            raise MMH3ResourceError(
                f"Installed ApplyVDNH3 does not support {key}={value!r}; available: {choices!r}"
            )
    instance = node_class()
    function_name = str(getattr(node_class, "FUNCTION"))
    result = getattr(instance, function_name)(
        model=model,
        vdn_checkpoint=settings.checkpoint,
        apply_turbo_adapter=settings.apply_turbo_adapter,
        strength=settings.strength,
        lora_mode=settings.lora_mode,
        branch_weights=settings.branch_weights,
        retain_buffers=settings.retain_buffers,
        attention_backend=settings.attention_backend,
        verbose=settings.verbose,
    )
    if not isinstance(result, (tuple, list)) or len(result) != 1:
        raise MMH3ResourceError("ApplyVDNH3 returned an unexpected result shape")
    record_optimization(result[0], "vdn_h3")
    profile = {
        "contract": "mmh3_vdn_h3_v1",
        "enabled": True,
        "vdn": deep_copy_json(settings.to_dict()),
        "sampling": sampling,
    }
    status = (
        f"READY · VDN-H3 · {settings.task_family} · {settings.checkpoint} · "
        f"recommended_steps={settings.recommended_steps}"
    )
    return result[0], profile, status


__all__ = [
    "VDN_ATTENTION_BACKENDS",
    "VDN_BRANCH_WEIGHT_MODES",
    "VDN_DEFAULT_CHECKPOINT",
    "VDN_LORA_MODES",
    "VDN_NODE_ID",
    "VDN_RETAIN_BUFFER_MODES",
    "VDN_SUPPORTED_TASK_FAMILIES",
    "VDNOptimizationSettings",
    "apply_external_vdn",
    "build_vdn_settings",
    "validate_vdn_runtime_node",
    "validate_vdn_sampling_profile",
]
