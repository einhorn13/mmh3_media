from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Mapping

from .errors import MMH3ResourceError


LOGGER = logging.getLogger(__name__)


H3_SLA_NODE_ID = "H3SLAAttention"
H3_SLA_SOURCE_REPOSITORY = "https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes"
H3_SLA_SOURCE_COMMIT = "62adf9f85add5e2f44eb789a2568fe221e21a002"
H3_SLA_PROFILE_CONTRACT = "mmh3_f07_execution_profile_v1"
H3_SLA_INPUTS = (
    "model",
    "sparsity_ratio",
    "block_size",
    "min_seq_len",
    "dense_last_steps",
    "protect_audio",
    "enabled",
    "dense_steps",
    "dense_backend",
    "disable_fp16_accum",
    "stabilize_motion",
)
H3_SLA_OPTIONAL_INPUTS = ("reference_protection",)
H3_SLA_BLOCK_SIZES = ("64", "128")
H3_SLA_DENSE_BACKENDS = (
    "pytorch",
    "comfy_kitchen",
    "sage:auto",
    "sage:qk_int8_pv_fp16_cuda",
    "sage:qk_int8_pv_fp16_triton",
    "sage:qk_int8_pv_fp8_cuda",
    "sage:qk_int8_pv_fp8_cuda++",
    "auto",
)
_DENSE_STEPS_RE = re.compile(r"^\s*(?:\d+(?:\s*-\s*\d+)?)(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*\s*$")


def normalize_h3_sla_settings(
    *,
    sparsity_ratio: float = 0.90,
    block_size: str = "64",
    min_seq_len: int = 4096,
    dense_last_steps: int = 1,
    protect_audio: bool = True,
    enabled: bool = True,
    dense_steps: str = "0",
    dense_backend: str = "comfy_kitchen",
    disable_fp16_accum: bool = True,
    stabilize_motion: bool = True,
) -> dict[str, Any]:
    ratio = float(sparsity_ratio)
    if not 0.0 <= ratio <= 0.95:
        raise MMH3ResourceError("H3 SLA sparsity_ratio must be between 0.0 and 0.95")
    block = str(block_size)
    if block not in H3_SLA_BLOCK_SIZES:
        raise MMH3ResourceError(f"H3 SLA block_size must be one of {H3_SLA_BLOCK_SIZES}")
    minimum = int(min_seq_len)
    if not 0 <= minimum <= 1_000_000:
        raise MMH3ResourceError("H3 SLA min_seq_len must be between 0 and 1000000")
    last = int(dense_last_steps)
    if not 0 <= last <= 8:
        raise MMH3ResourceError("H3 SLA dense_last_steps must be between 0 and 8")
    step_spec = str(dense_steps).strip()
    if step_spec and not _DENSE_STEPS_RE.fullmatch(step_spec):
        raise MMH3ResourceError("H3 SLA dense_steps must be comma-separated indices or ranges")
    backend = str(dense_backend)
    if backend not in H3_SLA_DENSE_BACKENDS:
        raise MMH3ResourceError(f"H3 SLA dense_backend must be one of {H3_SLA_DENSE_BACKENDS}")
    return {
        "sparsity_ratio": ratio,
        "block_size": block,
        "min_seq_len": minimum,
        "dense_last_steps": last,
        "protect_audio": bool(protect_audio),
        "enabled": bool(enabled),
        "dense_steps": step_spec,
        "dense_backend": backend,
        "disable_fp16_accum": bool(disable_fp16_accum),
        "stabilize_motion": bool(stabilize_motion),
    }


def require_h3_sla_node_contract(node_class: type[Any]) -> tuple[str, ...]:
    try:
        schema = node_class.define_schema()
        node_id = schema.node_id
        input_ids = tuple(item.id for item in schema.inputs)
    except Exception as exc:
        raise MMH3ResourceError("Could not inspect external H3SLAAttention schema") from exc
    supported_inputs = (H3_SLA_INPUTS, H3_SLA_INPUTS + H3_SLA_OPTIONAL_INPUTS)
    if node_id != H3_SLA_NODE_ID or input_ids not in supported_inputs:
        raise MMH3ResourceError(
            "Unsupported H3SLAAttention schema; MMH3 requires the validated core inputs "
            f"{H3_SLA_INPUTS} with only the optional suffix {H3_SLA_OPTIONAL_INPUTS}, "
            f"got node_id={node_id!r}, inputs={input_ids!r}"
        )
    return input_ids


def _sla_wrappers(model: Any) -> tuple[Any, ...]:
    getter = getattr(model, "get_wrappers", None)
    if not callable(getter):
        raise MMH3ResourceError("H3 SLA requires a ComfyUI ModelPatcher with get_wrappers()")
    return tuple(getter("diffusion_model", "h3_sla_state"))


def _extract_model(output: Any) -> Any:
    result = getattr(output, "result", output)
    if isinstance(result, (tuple, list)) and result:
        return result[0]
    try:
        return output[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise MMH3ResourceError("H3SLAAttention returned no MODEL output") from exc


def _external_sla_state(override: Any) -> dict[str, Any]:
    try:
        state = inspect.getclosurevars(override).nonlocals.get("state")
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError("H3 SLA override has no inspectable v1.3.6 runtime state") from exc
    required = {"calls", "dense", "failed", "fp16_saved"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise MMH3ResourceError("H3 SLA override does not expose the expected v1.3.6 runtime state")
    return state


def _restore_external_matmul_flags(state: dict[str, Any]) -> None:
    saved = state.get("fp16_saved")
    if not isinstance(saved, tuple) or len(saved) != 2:
        return
    import torch

    backend = torch.backends.cuda.matmul
    original_fp16, original_bf16 = saved
    if original_fp16 is not None:
        backend.allow_fp16_reduced_precision_reduction = original_fp16
    if original_bf16 is not None:
        backend.allow_bf16_reduced_precision_reduction = original_bf16
    state["fp16_saved"] = None


def build_h3_sla_execution_profile(
    settings: Mapping[str, Any], *, patch_materialized: bool
) -> dict[str, Any]:
    return {
        "version": 1,
        "contract": H3_SLA_PROFILE_CONTRACT,
        "name": "plaguekind_h3_sla_v1_3_6",
        "source": {
            "repository": H3_SLA_SOURCE_REPOSITORY,
            "commit": H3_SLA_SOURCE_COMMIT,
            "node_id": H3_SLA_NODE_ID,
        },
        "optimization": {
            "kind": "h3_block_sparse_attention",
            "settings": dict(settings),
            "patch_materialized": bool(patch_materialized),
            "external_failure_policy": "mmh3_fail_closed_admission",
            "kernel_failure_policy": "mmh3_fail_closed_runtime_exception",
        },
        "runtime_validated": False,
        "risks": {
            "global_cuda_matmul_flags": bool(settings["disable_fp16_accum"]),
        },
    }


def apply_external_h3_sla(
    model: Any,
    node_class: type[Any],
    **raw_settings: Any,
) -> tuple[Any, dict[str, Any], str]:
    settings = normalize_h3_sla_settings(**raw_settings)
    if not settings["enabled"]:
        profile = build_h3_sla_execution_profile(settings, patch_materialized=False)
        return model, profile, "BYPASS · H3 SLA disabled · dense baseline"

    from .optimization_contract import validate_optimization_application, record_optimization
    fp16 = "disabled" if settings["disable_fp16_accum"] else "inherit"
    validate_optimization_application(model, "h3_sla", fp16)

    input_ids = require_h3_sla_node_contract(node_class)
    if _sla_wrappers(model):
        raise MMH3ResourceError("MODEL already has an h3_sla_state wrapper; refusing a duplicate SLA patch")

    external_settings = dict(settings)
    if "reference_protection" in input_ids:
        # Preserve the pre-v1.4 MMH3 behavior until reference-aware SLA becomes
        # an explicit, separately validated optimization option.
        external_settings["reference_protection"] = False
    output = node_class.execute(model=model, **external_settings)
    patched = _extract_model(output)
    wrappers = _sla_wrappers(patched)
    transformer_options = getattr(patched, "model_options", {}).get("transformer_options", {})
    override = transformer_options.get("optimized_attention_override")
    if len(wrappers) != 1 or not callable(override):
        raise MMH3ResourceError(
            "H3SLAAttention did not materialize its sparse-attention patch; "
            "the external node likely failed open, so MMH3 stopped the run"
        )
    runtime_state = _external_sla_state(override)

    observed = False

    def admission_probe(func, q, k, v, heads, *args, **kwargs):
        nonlocal observed
        if not observed:
            observed = True
            LOGGER.info(
                "H3 SLA first attention call: q_shape=%s q_dtype=%s heads=%s "
                "mask=%s skip_reshape=%s skip_output_reshape=%s dense_step=%s",
                tuple(getattr(q, "shape", ())),
                getattr(q, "dtype", None),
                heads,
                kwargs.get("mask"),
                kwargs.get("skip_reshape", False),
                kwargs.get("skip_output_reshape", False),
                (kwargs.get("transformer_options") or {}).get("_h3sla_dense"),
            )
        result = override(func, q, k, v, heads, *args, **kwargs)
        failure = runtime_state.get("failed")
        if failure is not None:
            _restore_external_matmul_flags(runtime_state)
            raise MMH3ResourceError(
                "External H3 SLA sparse kernel failed after admission; MMH3 refused its dense fallback: "
                + str(failure)
            )
        return result

    transformer_options["optimized_attention_override"] = admission_probe

    record_optimization(patched, "h3_sla", fp16)
    profile = build_h3_sla_execution_profile(settings, patch_materialized=True)
    return patched, profile, "READY · H3 SLA v1.3.6 hooks materialized · kernel execution pending"


__all__ = [
    "H3_SLA_BLOCK_SIZES",
    "H3_SLA_DENSE_BACKENDS",
    "H3_SLA_INPUTS",
    "H3_SLA_OPTIONAL_INPUTS",
    "H3_SLA_NODE_ID",
    "H3_SLA_PROFILE_CONTRACT",
    "H3_SLA_SOURCE_COMMIT",
    "H3_SLA_SOURCE_REPOSITORY",
    "apply_external_h3_sla",
    "build_h3_sla_execution_profile",
    "normalize_h3_sla_settings",
    "require_h3_sla_node_contract",
]
