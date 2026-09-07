from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence

from .errors import MMH3ResourceError
from .util import deep_copy_json
from .vdn_optimization import build_vdn_settings, validate_vdn_sampling_profile


MODEL_OPTIMIZATION_CONTRACT = "mmh3_h3_model_optimizations_v1"
ATTENTION_MODES = (
    "inherit",
    "pytorch",
    "comfy_kitchen",
    "sage_attention_kj",
    "sol_attn",
    "sol_native",
    "h3_sla",
    "h3_sla_sol_attn",
    "vdn_h3",
)
FP16_ACCUMULATION_MODES = ("inherit", "enabled", "disabled")
ATTENTION_NODE_IDS = {
    "pytorch": "ModelAttentionBackend",
    "comfy_kitchen": "ModelAttentionBackend",
    "sage_attention_kj": "PatchSageAttentionKJ",
    "sol_attn": "SolAttnPatch",
    "sol_native": "BlockSparseAttention",
    "h3_sla": "MMH3H3SLAApply",
    "vdn_h3": "MMH3H3VDNApply",
}
FP16_PATCH_NODE_ID = "MMH3H3FP16AccumulationPatch"


@dataclass(frozen=True)
class ModelOptimizationPlan:
    enabled: bool
    attention_mode: str
    fp16_accumulation: str
    settings: dict[str, Any]

    @property
    def attention_stages(self) -> tuple[str, ...]:
        if not self.enabled or self.attention_mode == "inherit":
            return ()
        if self.attention_mode == "h3_sla_sol_attn":
            return ("h3_sla", "sol_attn")
        return (self.attention_mode,)

    @property
    def required_nodes(self) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        nodes: list[str] = []
        if self.fp16_accumulation != "inherit":
            nodes.append(FP16_PATCH_NODE_ID)
        nodes.extend(ATTENTION_NODE_IDS[stage] for stage in self.attention_stages)
        return tuple(nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "contract": MODEL_OPTIMIZATION_CONTRACT,
            "enabled": self.enabled,
            "attention": {
                "mode": self.attention_mode if self.enabled else "inherit",
                "settings": deep_copy_json(self.settings.get("attention", {})) if self.enabled else {},
            },
            "torch": {
                "fp16_accumulation": self.fp16_accumulation if self.enabled else "inherit",
            },
        }

    def summary(self) -> str:
        if not self.enabled:
            return "BYPASS · H3 model optimizations"
        return (
            "READY · H3 model optimizations · "
            f"attention={self.attention_mode} · fp16_accumulation={self.fp16_accumulation}"
        )


@dataclass(frozen=True)
class ModelOptimizationExpansion:
    model: Any
    profile: dict[str, Any]
    status: str
    graph: dict[str, Any]


def build_model_optimization_plan(
    *,
    enabled: bool,
    attention_mode: str,
    fp16_accumulation: str,
    sage_mode: str = "auto",
    sage_allow_compile: bool = False,
    sol_tau: float = 1.0,
    sol_start_percent: float = 0.2,
    sol_end_percent: float = 0.9,
    sol_min_tokens: int = 4096,
    sol_int8_qk: bool = True,
    sol_sink_conditioning: str = "exact_kv_and_rows",
    sol_dense_blocks: str = "",
    sol_extra_tokens: int = 256,
    sla_sparsity_ratio: float = 0.90,
    sla_block_size: str = "64",
    sla_min_seq_len: int = 4096,
    sla_dense_last_steps: int = 1,
    sla_protect_audio: bool = True,
    sla_dense_backend: str = "comfy_kitchen",
    vdn_task_family: str = "fl2va",
    vdn_checkpoint: str = "stage-dmd-step-250",
    vdn_apply_turbo_adapter: bool = True,
    vdn_strength: float = 1.0,
    vdn_lora_mode: str = "merge",
    vdn_branch_weights: str = "auto",
    vdn_retain_buffers: str = "auto",
    vdn_attention_backend: str = "grouped",
    vdn_verbose: bool = False,
    sampling_profile: Mapping[str, Any] | None = None,
) -> ModelOptimizationPlan:
    if not enabled:
        return ModelOptimizationPlan(False, "inherit", "inherit", {})
    if attention_mode not in ATTENTION_MODES:
        raise MMH3ResourceError(f"Unknown attention mode {attention_mode!r}")
    if fp16_accumulation not in FP16_ACCUMULATION_MODES:
        raise MMH3ResourceError(f"Unknown FP16 accumulation mode {fp16_accumulation!r}")

    attention: dict[str, Any] = {}
    if attention_mode == "sage_attention_kj":
        attention = {"sage_mode": str(sage_mode), "allow_compile": bool(sage_allow_compile)}
    if attention_mode in {"sol_attn", "sol_native", "h3_sla_sol_attn"}:
        if not 0 <= sol_start_percent <= sol_end_percent <= 1:
            raise MMH3ResourceError("Sol-Attn requires 0 <= start <= end <= 1")
        if sol_sink_conditioning not in {"exact_kv", "exact_kv_and_rows", "off"}:
            raise MMH3ResourceError("Unsupported Sol-Attn conditioning protection")
        sol_settings = {
            "tau": float(sol_tau),
            "start_percent": float(sol_start_percent),
            "end_percent": float(sol_end_percent),
            "min_tokens": int(sol_min_tokens),
            "int8_qk": bool(sol_int8_qk),
            "sink_conditioning": sol_sink_conditioning,
            "dense_blocks": str(sol_dense_blocks),
        }
        attention = {"sol_attn": sol_settings} if attention_mode == "h3_sla_sol_attn" else sol_settings
        if attention_mode == "sol_native":
            if int(sol_extra_tokens) not in {0, 64, 128, 192, 256}:
                raise MMH3ResourceError("Native Sol extra tokens must be 0, 64, 128, 192 or 256")
            if any(part.strip() and not re.fullmatch(r"\s*\d+\s*(?:-\s*\d+\s*)?", part)
                   for part in str(sol_dense_blocks).split(",")):
                raise MMH3ResourceError("Native Sol requires non-negative dense block indices")
            attention.pop("int8_qk")
            attention["extra_tokens"] = int(sol_extra_tokens)
    if attention_mode == "vdn_h3":
        vdn = build_vdn_settings(
            task_family=vdn_task_family,
            checkpoint=vdn_checkpoint,
            apply_turbo_adapter=vdn_apply_turbo_adapter,
            strength=vdn_strength,
            lora_mode=vdn_lora_mode,
            branch_weights=vdn_branch_weights,
            retain_buffers=vdn_retain_buffers,
            attention_backend=vdn_attention_backend,
            verbose=vdn_verbose,
        )
        validated_sampling = validate_vdn_sampling_profile(vdn, sampling_profile)
        attention = vdn.to_dict()
        attention["sampling"] = validated_sampling
    if attention_mode in {"h3_sla", "h3_sla_sol_attn"}:
        sla_settings = {
            "sparsity_ratio": float(sla_sparsity_ratio),
            "block_size": str(sla_block_size),
            "min_seq_len": int(sla_min_seq_len),
            "dense_last_steps": int(sla_dense_last_steps),
            "protect_audio": bool(sla_protect_audio),
            "dense_backend": str(sla_dense_backend),
        }
        if attention_mode == "h3_sla_sol_attn":
            attention["h3_sla"] = sla_settings
            attention["apply_order"] = ["h3_sla", "sol_attn"]
        else:
            attention = sla_settings
    return ModelOptimizationPlan(
        True,
        attention_mode,
        fp16_accumulation,
        {"attention": attention},
    )


def build_model_optimization_expansion(
    plan: ModelOptimizationPlan,
    *,
    model: Any,
    runtime_node_ids: Sequence[str] | None = None,
    graph_builder_factory: Callable[[], Any] | None = None,
) -> ModelOptimizationExpansion:
    profile = plan.to_dict()
    if not plan.enabled:
        return ModelOptimizationExpansion(model, profile, plan.summary(), {})

    if runtime_node_ids is not None:
        missing = [node_id for node_id in plan.required_nodes if node_id not in set(runtime_node_ids)]
        if missing:
            hint = (
                ". Sol Attention requires ComfyUI-SolAttn_triton (Patch Sol-Attn / SolAttnPatch). "
                "If installed, check its startup import errors and restart ComfyUI after fixing them."
                if "SolAttnPatch" in missing else ""
            )
            raise MMH3ResourceError("Missing selected optimization node(s): " + ", ".join(missing) + hint)

    if graph_builder_factory is None:
        from comfy_execution.graph_utils import GraphBuilder  # type: ignore

        graph_builder_factory = GraphBuilder
    graph = graph_builder_factory()
    current_model = model

    if plan.fp16_accumulation != "inherit":
        torch_patch = graph.node(
            FP16_PATCH_NODE_ID,
            model=current_model,
            enabled=plan.fp16_accumulation == "enabled",
        )
        current_model = torch_patch.out(0)

    attention = dict(plan.settings.get("attention", {}))
    for stage in plan.attention_stages:
        settings = attention[stage] if plan.attention_mode == "h3_sla_sol_attn" else attention
        if stage in {"pytorch", "comfy_kitchen"}:
            backend = graph.node(
                "ModelAttentionBackend",
                model=current_model,
                attention=(
                    "pytorch attention"
                    if stage == "pytorch"
                    else "comfy kitchen attention"
                ),
            )
            current_model = backend.out(0)
        elif stage == "sage_attention_kj":
            sage = graph.node(
                "PatchSageAttentionKJ",
                model=current_model,
                sage_attention=settings["sage_mode"],
                allow_compile=settings["allow_compile"],
            )
            current_model = sage.out(0)
        elif stage == "sol_native":
            native_settings = dict(settings)
            tau = native_settings.pop("tau")
            sol = graph.node(
                "BlockSparseAttention",
                model=current_model,
                selection="Sol-Attn (adaptive tau)",
                **{"selection.tau": tau},
                **native_settings,
                verbose=False,
            )
            current_model = sol.out(0)
        elif stage == "sol_attn":
            sol = graph.node(
                "SolAttnPatch",
                model=current_model,
                **settings,
                morton=False,
                morton_curve="2d_frame",
                verbose=False,
                use_tma=False,
            )
            current_model = sol.out(0)
        elif stage == "vdn_h3":
            vdn = graph.node(
                "MMH3H3VDNApply",
                model=current_model,
                task_family=settings["task_family"],
                vdn_checkpoint=settings["checkpoint"],
                apply_turbo_adapter=settings["apply_turbo_adapter"],
                strength=settings["strength"],
                lora_mode=settings["lora_mode"],
                branch_weights=settings["branch_weights"],
                retain_buffers=settings["retain_buffers"],
                attention_backend=settings["attention_backend"],
                verbose=settings["verbose"],
                sampling_profile_json=__import__("json").dumps(settings["sampling"], ensure_ascii=False),
            )
            current_model = vdn.out(0)
        elif stage == "h3_sla":
            sla = graph.node(
                "MMH3H3SLAApply",
                model=current_model,
                enabled=True,
                sparsity_ratio=settings["sparsity_ratio"],
                block_size=settings["block_size"],
                min_seq_len=settings["min_seq_len"],
                dense_last_steps=settings["dense_last_steps"],
                protect_audio=settings["protect_audio"],
                dense_steps="0",
                dense_backend=settings["dense_backend"],
                disable_fp16_accum=False,
                stabilize_motion=True,
            )
            current_model = sla.out(0)

    return ModelOptimizationExpansion(
        current_model,
        profile,
        plan.summary(),
        graph.finalize(),
    )


__all__ = [
    "ATTENTION_MODES",
    "ATTENTION_NODE_IDS",
    "FP16_ACCUMULATION_MODES",
    "FP16_PATCH_NODE_ID",
    "MODEL_OPTIMIZATION_CONTRACT",
    "ModelOptimizationExpansion",
    "ModelOptimizationPlan",
    "build_model_optimization_expansion",
    "build_model_optimization_plan",
]
