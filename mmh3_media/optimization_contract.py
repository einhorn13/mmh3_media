"""Admission and ownership of H3 acceleration on a Comfy ModelPatcher."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

from .errors import MMH3ResourceError

STATE_KEY = "mmh3_optimization_state"
CONTRACT = "mmh3_optimization_state_v1"
SPARSE_MODES = {"vsa_native", "sol_native", "sol_attn", "sla_native", "h3_sla", "h3_sla_sol_attn"}
ATTENTION_MODES = (
    "inherit", "pytorch", "comfy_kitchen", "sage_attention_kj", "sol_attn",
    "vsa_native", "sol_native", "sla_native", "h3_sla", "h3_sla_sol_attn", "vdn_h3",
)
FP16_ACCUMULATION_MODES = ("inherit", "enabled", "disabled")


def _validate_modes(attention, fp16):
    if attention not in ATTENTION_MODES or fp16 not in FP16_ACCUMULATION_MODES:
        raise MMH3ResourceError("Unsupported MODEL optimization mode")


def _attachment(model, key):
    return getattr(model, "get_attachment", lambda _key: None)(key)


def resolve_sampling_profile(model, explicit=None):
    attached = _attachment(model, "mmh3_sampling_profile")
    if explicit is not None and not isinstance(explicit, Mapping):
        raise MMH3ResourceError("sampling_profile_json must contain a JSON object")
    if explicit is not None and attached is not None and explicit != attached:
        raise MMH3ResourceError("Explicit sampling profile conflicts with the MODEL sampling attachment")
    return explicit if explicit is not None else attached


def optimization_state(model):
    state = getattr(model, "model_options", {}).get(STATE_KEY)
    if state is None:
        return {"contract": CONTRACT, "attention": "inherit", "fp16_accumulation": "inherit"}
    if (not isinstance(state, dict) or state.get("contract") != CONTRACT
            or state.get("attention") not in ATTENTION_MODES
            or state.get("fp16_accumulation") not in FP16_ACCUMULATION_MODES):
        raise MMH3ResourceError("Unsupported MODEL optimization ownership contract")
    return state


def validate_optimization_application(model, attention="inherit", fp16="inherit", sampling_profile=None):
    """Reject repeats and unknown attention owners before any patch is installed.

    Precision-only and attention-only nodes may compose once each. The explicit
    legacy SLA -> Sol plan is one strategy, not permission for arbitrary stacking.
    """
    _validate_modes(attention, fp16)
    if attention == "inherit" and fp16 == "inherit":
        return
    state = optimization_state(model)
    if fp16 != "inherit" and state["fp16_accumulation"] != "inherit":
        raise MMH3ResourceError("MODEL already has FP16 accumulation optimization; duplicate application refused")
    if attention == "inherit":
        return
    if state["attention"] != "inherit":
        raise MMH3ResourceError("MODEL already has attention/architecture optimization; duplicate or incompatible application refused. Use the base MODEL branch.")
    profile = resolve_sampling_profile(model, sampling_profile)
    adapter = _attachment(model, "mmh3_sampling_adapter")
    from .fasth3 import FASTH3_PROFILE
    if (adapter == FASTH3_PROFILE or (profile and profile.get("profile") == FASTH3_PROFILE)) and attention in SPARSE_MODES | {"vdn_h3"}:
        raise MMH3ResourceError("FastH3 dense cannot be combined with sparse attention or VDN-H3")
    if attention == "vdn_h3" and (adapter or (profile and (profile.get("adapter") or profile.get("recommended_lora")))):
        raise MMH3ResourceError("VDN-H3 cannot be stacked with an ordinary H3/FastH3 acceleration adapter")
    if profile and profile.get("vdn_required") and attention != "vdn_h3":
        raise MMH3ResourceError("VDN-H3 sampling preset requires VDN-H3 in H3 Optimizations")
    if attention == "vsa_native":
        from .vsa_optimization import validate_vsa_sampling
        validate_vsa_sampling(profile, adapter)
    options = getattr(model, "model_options", {}).get("transformer_options", {})
    objects = getattr(model, "object_patches", {})
    if (options.get("optimized_attention_override") or options.get("patches_replace")
            or any("attn" in key or key.endswith(".forward") for key in objects)):
        raise MMH3ResourceError("MODEL already has an external attention/block patch; duplicate or incompatible application refused")


def record_optimization(model, attention="inherit", fp16="inherit"):
    """Record only on a successfully patched output; caller owns this clone."""
    _validate_modes(attention, fp16)
    if not isinstance(getattr(model, "model_options", None), dict):
        raise MMH3ResourceError("Optimization output is not a ModelPatcher with model_options")
    state = deepcopy(optimization_state(model))
    if attention != "inherit":
        state["attention"] = attention
    if fp16 != "inherit":
        state["fp16_accumulation"] = fp16
    model.model_options[STATE_KEY] = state
    return model


def require_unoptimized_sampling_model(model):
    state = optimization_state(model)
    options = getattr(model, "model_options", {}).get("transformer_options", {})
    if (state["attention"] != "inherit" or state["fp16_accumulation"] != "inherit"
            or _attachment(model, "mmh3_sampling_profile") or _attachment(model, "mmh3_sampling_adapter")
            or options.get("optimized_attention_override") or options.get("patches_replace")):
        raise MMH3ResourceError("H3 Sampling requires an unpatched base MODEL; sampling/optimizations already applied")
