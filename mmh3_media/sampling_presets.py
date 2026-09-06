from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import MMH3ResourceError
from .util import deep_copy_json
from .fasth3 import FASTH3_PROFILE


STANDARD_PROFILE = "standard (20 steps)"
TURBO_4_PROFILE = "turbo (4 steps)"
TURBO_8_PROFILE = "turbo (8 steps)"
CUSTOM_PROFILE = "custom"
SAMPLING_PRESET_CONTRACT = "mmh3_h3_sampling_preset_v2"
SAMPLING_PROFILES = (STANDARD_PROFILE, TURBO_4_PROFILE, TURBO_8_PROFILE, FASTH3_PROFILE, CUSTOM_PROFILE)
PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    FASTH3_PROFILE: {
        "steps": 6, "video_shift": 12.0, "audio_shift": 3.0,
        "sampler": "res_multistep", "scheduler": "simple",
        "sigma_preset": "scheduler_generated", "runtime_validated": False,
    },
    STANDARD_PROFILE: {
        "steps": 20,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "sigma_preset": "scheduler_generated",
        "runtime_validated": True,
    },
}
TURBO_RECIPES: dict[str, dict[str, Any]] = {
    TURBO_4_PROFILE: {
        "supported_task_families": ["fl2va", "ref2va"],
        "steps": 4,
        "video_shift": 6.0,
        "audio_shift": 3.0,
        "sampler": "euler",
        "scheduler": "simple",
        "sigma_preset": "scheduler_generated",
        "runtime_validated": True,
        "recommended_lora": "minimax\\turbo\\minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
        "task_overrides": {
            "ref2va": {
                "video_shift": 12.0,
                "recommended_lora": "minimax\\minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            },
        },
    },
    TURBO_8_PROFILE: {
        "supported_task_families": ["fl2va"],
        "steps": 8,
        "video_shift": 6.0,
        "audio_shift": 3.0,
        "sampler": "euler",
        "scheduler": "simple",
        "sigma_preset": "scheduler_generated",
        "runtime_validated": True,
        "recommended_lora": "minimax\\turbo\\minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    },
}


@dataclass(frozen=True)
class SamplingPreset:
    profile: str
    task_family: str
    steps: int
    video_shift: float
    audio_shift: float
    sampler: str
    scheduler: str
    sigma_preset: str
    recommended_lora: str | None
    runtime_validated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "contract": SAMPLING_PRESET_CONTRACT,
            "profile": self.profile,
            "task_family": self.task_family,
            "steps": self.steps,
            "video_shift": self.video_shift,
            "audio_shift": self.audio_shift,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "sigma_preset": self.sigma_preset,
            "recommended_lora": self.recommended_lora,
            "runtime_validated": self.runtime_validated,
        }

    def summary(self) -> str:
        return (
            f"READY · H3 sampling · {self.profile} · {self.steps} steps · "
            f"{self.sampler}/{self.scheduler}"
        )


def build_sampling_preset(
    *,
    profile: str,
    task_family: str,
    custom_steps: int = 20,
    custom_video_shift: float = 12.0,
    custom_audio_shift: float = 3.0,
    custom_sampler: str = "res_multistep",
    custom_scheduler: str = "simple",
) -> SamplingPreset:
    if task_family not in {"fl2va", "ref2va"}:
        raise MMH3ResourceError("H3 sampling task family must be fl2va or ref2va")
    if profile not in SAMPLING_PROFILES:
        raise MMH3ResourceError(f"Unknown H3 sampling preset {profile!r}")

    if profile == CUSTOM_PROFILE:
        if int(custom_steps) < 1:
            raise MMH3ResourceError("Custom H3 sampling steps must be positive")
        recipe = {
            "steps": int(custom_steps),
            "video_shift": float(custom_video_shift),
            "audio_shift": float(custom_audio_shift),
            "sampler": str(custom_sampler),
            "scheduler": str(custom_scheduler),
            "sigma_preset": "scheduler_generated",
            "runtime_validated": False,
        }
    else:
        recipe = deep_copy_json(PROFILE_PRESETS.get(profile) or TURBO_RECIPES[profile])
        supported = recipe.pop("supported_task_families", [task_family])
        overrides = recipe.pop("task_overrides", {})
        if task_family not in supported:
            raise MMH3ResourceError(
                f"Sampling preset {profile!r} does not support task family {task_family!r}"
            )
        recipe.update(deep_copy_json(overrides.get(task_family, {})))

    return SamplingPreset(
        profile=profile,
        task_family=task_family,
        steps=int(recipe["steps"]),
        video_shift=float(recipe["video_shift"]),
        audio_shift=float(recipe["audio_shift"]),
        sampler=str(recipe["sampler"]),
        scheduler=str(recipe["scheduler"]),
        sigma_preset=str(recipe.get("sigma_preset", "scheduler_generated")),
        recommended_lora=(str(recipe["recommended_lora"]) if recipe.get("recommended_lora") else None),
        runtime_validated=bool(recipe.get("runtime_validated", False)),
    )


__all__ = [
    "CUSTOM_PROFILE",
    "PROFILE_PRESETS",
    "SAMPLING_PRESET_CONTRACT",
    "SAMPLING_PROFILES",
    "STANDARD_PROFILE",
    "SamplingPreset",
    "TURBO_4_PROFILE",
    "TURBO_8_PROFILE",
    "TURBO_RECIPES",
    "build_sampling_preset",
]
