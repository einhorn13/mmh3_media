from __future__ import annotations

import json


from .node_support import CATEGORY, MMH3ResourceError, io
from .optimization_contract import (
    record_optimization, require_unoptimized_sampling_model,
    resolve_sampling_profile, validate_optimization_application,
)
from .model_optimizations import (
    build_model_optimization_expansion,
    build_model_optimization_plan,
)
from .sampling_presets import (
    SAMPLING_ADAPTER_ATTACHMENT,
    SAMPLING_PROFILE_ATTACHMENT,
    VDN_DMD_PROFILE,
    VDN_STAGE_B_PROFILE,
    build_sampling_preset,
)
from .vdn_optimization import (
    VDN_ATTENTION_BACKENDS,
    VDN_BRANCH_WEIGHT_MODES,
    VDN_DEFAULT_CHECKPOINT,
    VDN_LORA_MODES,
    VDN_NODE_ID,
    VDN_RETAIN_BUFFER_MODES,
    apply_external_vdn,
    build_vdn_settings,
    validate_vdn_sampling_profile,
)
from .sla_optimization import (
    H3_SLA_BLOCK_SIZES,
    H3_SLA_DENSE_BACKENDS,
    H3_SLA_NODE_ID,
    apply_external_h3_sla,
)



class MMH3H3VDNApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3VDNApply",
            display_name="MMH3 VDN-H3 Apply",
            category=f"{CATEGORY}/Internal",
            description=(
                "Fail-closed adapter for Saganaki22/ComfyUI-VDN-H3 ApplyVDNH3. "
                "Supports FL2VA and Ref2VA packed layouts; validates the external node contract before patching."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("task_family", options=["fl2va", "ref2va"], default="fl2va"),
                io.String.Input("vdn_checkpoint", default=VDN_DEFAULT_CHECKPOINT),
                io.Boolean.Input("apply_turbo_adapter", default=True),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Combo.Input("lora_mode", options=list(VDN_LORA_MODES), default="merge"),
                io.Combo.Input("branch_weights", options=list(VDN_BRANCH_WEIGHT_MODES), default="auto"),
                io.Combo.Input("retain_buffers", options=list(VDN_RETAIN_BUFFER_MODES), default="auto"),
                io.Combo.Input("attention_backend", options=list(VDN_ATTENTION_BACKENDS), default="grouped"),
                io.Boolean.Input("verbose", default=False),
                io.String.Input("sampling_profile_json", default="", multiline=True, advanced=True),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("execution_profile_json"),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(
        cls, model, task_family: str, vdn_checkpoint: str, apply_turbo_adapter: bool,
        strength: float, lora_mode: str, branch_weights: str, retain_buffers: str,
        attention_backend: str, verbose: bool, sampling_profile_json: str = "",
    ) -> io.NodeOutput:
        settings = build_vdn_settings(
            task_family=task_family, checkpoint=vdn_checkpoint,
            apply_turbo_adapter=apply_turbo_adapter, strength=strength,
            lora_mode=lora_mode, branch_weights=branch_weights,
            retain_buffers=retain_buffers, attention_backend=attention_backend, verbose=verbose,
        )
        try:
            sampling_profile = json.loads(sampling_profile_json) if sampling_profile_json.strip() else None
        except json.JSONDecodeError as exc:
            raise MMH3ResourceError(f"VDN-H3 sampling_profile_json is not valid JSON: {exc}") from exc
        validate_vdn_sampling_profile(settings, sampling_profile)
        try:
            import nodes as comfy_nodes  # type: ignore
        except ImportError as exc:
            raise MMH3ResourceError("ComfyUI runtime nodes registry is unavailable") from exc
        node_class = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(VDN_NODE_ID)
        patched, profile, status = apply_external_vdn(model, node_class, settings, sampling_profile)
        return io.NodeOutput(patched, json.dumps(profile, ensure_ascii=False, indent=2), status)


class MMH3H3SLAApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3SLAApply",
            display_name="MMH3 H3 SLA Apply (PlagueKind v1.3.6)",
            category=CATEGORY,
            description=(
                "Fail-closed F07 adapter for PlagueKind H3SLAAttention v1.3.6. "
                "Verifies the external schema and the materialized ModelPatcher hooks before sampling."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("sparsity_ratio", default=0.90, min=0.0, max=0.95, step=0.05),
                io.Combo.Input("block_size", options=list(H3_SLA_BLOCK_SIZES), default="64"),
                io.Int.Input("min_seq_len", default=4096, min=0, max=1_000_000, step=1024),
                io.Int.Input("dense_last_steps", default=1, min=0, max=8),
                io.Boolean.Input("protect_audio", default=True),
                io.Boolean.Input("enabled", default=True),
                io.String.Input("dense_steps", default="0"),
                io.Combo.Input("dense_backend", options=list(H3_SLA_DENSE_BACKENDS), default="comfy_kitchen"),
                io.Boolean.Input("disable_fp16_accum", default=True),
                io.Boolean.Input("stabilize_motion", default=True),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("execution_profile_json"),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        sparsity_ratio: float,
        block_size: str,
        min_seq_len: int,
        dense_last_steps: int,
        protect_audio: bool,
        enabled: bool,
        dense_steps: str,
        dense_backend: str,
        disable_fp16_accum: bool,
        stabilize_motion: bool,
    ) -> io.NodeOutput:
        if not enabled:
            patched, profile, status = apply_external_h3_sla(
                model,
                None,
                sparsity_ratio=sparsity_ratio,
                block_size=block_size,
                min_seq_len=min_seq_len,
                dense_last_steps=dense_last_steps,
                protect_audio=protect_audio,
                enabled=False,
                dense_steps=dense_steps,
                dense_backend=dense_backend,
                disable_fp16_accum=disable_fp16_accum,
                stabilize_motion=stabilize_motion,
            )
            return io.NodeOutput(patched, json.dumps(profile, ensure_ascii=False, indent=2), status)
        try:
            import nodes as comfy_nodes  # type: ignore
        except ImportError as exc:
            raise MMH3ResourceError("ComfyUI runtime nodes registry is unavailable") from exc
        node_class = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(H3_SLA_NODE_ID)
        if node_class is None:
            raise MMH3ResourceError(
                "H3SLAAttention is missing; install/update PlagueKind Nodes to v1.3.6"
            )
        patched, profile, status = apply_external_h3_sla(
            model,
            node_class,
            sparsity_ratio=sparsity_ratio,
            block_size=block_size,
            min_seq_len=min_seq_len,
            dense_last_steps=dense_last_steps,
            protect_audio=protect_audio,
            enabled=enabled,
            dense_steps=dense_steps,
            dense_backend=dense_backend,
            disable_fp16_accum=disable_fp16_accum,
            stabilize_motion=stabilize_motion,
        )
        return io.NodeOutput(patched, json.dumps(profile, ensure_ascii=False, indent=2), status)


class MMH3H3FP16AccumulationPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3FP16AccumulationPatch",
            display_name="MMH3 H3 FP16 Accumulation Patch",
            category=f"{CATEGORY}/Internal",
            description="Scope FP16 accumulation to one MODEL sampling lifecycle.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, enabled: bool) -> io.NodeOutput:
        import torch
        from comfy.patcher_extension import CallbacksMP  # type: ignore

        validate_optimization_application(model, fp16="enabled" if enabled else "disabled")
        backend = torch.backends.cuda.matmul
        if not hasattr(backend, "allow_fp16_accumulation"):
            raise MMH3ResourceError("FP16 accumulation requires a PyTorch build that exposes allow_fp16_accumulation")
        patched = model.clone()
        saved: list[bool] = []

        def before_run(*_args, **_kwargs):
            saved.append(bool(backend.allow_fp16_accumulation))
            backend.allow_fp16_accumulation = bool(enabled)

        def after_run(*_args, **_kwargs):
            if saved:
                backend.allow_fp16_accumulation = saved.pop()

        patched.add_callback(CallbacksMP.ON_PRE_RUN, before_run)
        patched.add_callback(CallbacksMP.ON_CLEANUP, after_run)
        record_optimization(patched, fp16="enabled" if enabled else "disabled")
        return io.NodeOutput(patched)


class MMH3H3OptimizationRecord(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3H3OptimizationRecord", category=f"{CATEGORY}/Internal",
            inputs=[io.Model.Input("model"), io.String.Input("attention_mode"),
                    io.String.Input("fp16_accumulation")],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, attention_mode: str, fp16_accumulation: str):
        patched = model.clone()
        record_optimization(patched, attention_mode, fp16_accumulation)
        return io.NodeOutput(patched)


class MMH3H3SamplingPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        import folder_paths
        return io.Schema(
            node_id="MMH3H3SamplingPreset",
            display_name="H3 Sampling",
            category=CATEGORY,
            description="Select a sampling recipe; matching Turbo/FastH3 loading and actual adapter provenance follow automatically. Connect the unpatched H3 base model.",
            inputs=[
                io.DynamicCombo.Input(
                    "profile",
                    display_name="Preset",
                    options=[
                        io.DynamicCombo.Option("Turbo - 4 steps", []),
                        io.DynamicCombo.Option("Standard - 20 steps", []),
                        io.DynamicCombo.Option("Turbo - 8 steps", []),
                        io.DynamicCombo.Option("VDN-H3 DMD - 8 steps (experimental)", []),
                        io.DynamicCombo.Option("VDN-H3 Stage-B - 50 steps (experimental)", []),
                        io.DynamicCombo.Option("FastH3 dense - 6 steps (experimental)", [
                            io.Combo.Input("lora_name", options=["(select converted dense LoRA)"] + folder_paths.get_filename_list("loras"),
                                           tooltip="Select lora_convert_h3 output from dense-datafree, adaln=drop. Never select a converted VSA student. The converter does not certify training provenance."),
                        ]),
                        io.DynamicCombo.Option(
                            "Custom",
                            [
                                io.Int.Input("steps", display_name="Steps", default=20, min=1, max=100),
                                io.Float.Input("video_shift", display_name="Video shift", default=12.0, min=0.0, max=30.0, step=0.1, advanced=True),
                                io.Float.Input("audio_shift", display_name="Audio shift", default=3.0, min=0.0, max=30.0, step=0.1, advanced=True),
                                io.Combo.Input("sampler", display_name="Sampler", options=["res_multistep", "euler"], default="res_multistep", advanced=True),
                                io.Combo.Input("scheduler", display_name="Scheduler", options=["simple", "normal"], default="simple", advanced=True),
                            ],
                        ),
                    ],
                ),
                io.Combo.Input("task_family", display_name="Task family", options=["fl2va", "ref2va"], default="fl2va"),
                io.Model.Input("model", tooltip="Unpatched H3 base. Standard/Custom pass it through; Turbo/FastH3 load their adapter. Connect MODEL to Optimizations and applied_loras_json to Pack."),
            ],
            outputs=[
                io.String.Output("sampling_profile_json"),
                io.Int.Output("steps"),
                io.Float.Output("video_shift"),
                io.Float.Output("audio_shift"),
                io.Combo.Output("sampler_name"),
                io.Combo.Output("scheduler"),
                io.Model.Output("model"),
                io.String.Output("applied_loras_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        profile: dict | str,
        task_family: str,
        model,
    ) -> io.NodeOutput:
        if model is None:
            raise MMH3ResourceError("H3 Sampling requires the unpatched base MODEL input")
        if getattr(model, "patches", None) or getattr(model, "object_patches", None):
            raise MMH3ResourceError("H3 Sampling requires an unpatched base; apply creative LoRAs after Sampling")
        require_unoptimized_sampling_model(model)
        selected = profile if isinstance(profile, dict) else {"profile": profile}
        label = str(selected.get("profile") or "Turbo - 4 steps")
        profile_names = {
            "Standard - 20 steps": "standard (20 steps)",
            "Turbo - 4 steps": "turbo (4 steps)",
            "Turbo - 8 steps": "turbo (8 steps)",
            "FastH3 dense - 6 steps (experimental)": "fasth3 dense experimental (6 steps)",
            "VDN-H3 DMD - 8 steps (experimental)": "vdn-h3 dmd (8 steps)",
            "VDN-H3 Stage-B - 50 steps (experimental)": "vdn-h3 stage-b (50 steps)",
            "Custom": "custom",
        }
        preset = build_sampling_preset(
            profile=profile_names.get(label, label),
            task_family=task_family,
            custom_steps=int(selected.get("steps", 20)),
            custom_video_shift=float(selected.get("video_shift", 12.0)),
            custom_audio_shift=float(selected.get("audio_shift", 3.0)),
            custom_sampler=str(selected.get("sampler", "res_multistep")),
            custom_scheduler=str(selected.get("scheduler", "simple")),
        )
        from .fasth3 import FASTH3_PROFILE, apply_fasth3
        applied_loras = []
        info = preset.to_dict()
        if preset.profile == FASTH3_PROFILE:
            import folder_paths
            name = str(selected.get("lora_name") or "")
            if name not in folder_paths.get_filename_list("loras"):
                raise MMH3ResourceError("Select an installed converted FastH3 dense LoRA")
            path = folder_paths.get_full_path("loras", name)
            if not path or not path.lower().endswith(".safetensors"):
                raise MMH3ResourceError("FastH3 requires a safetensors adapter")
            model, entry = apply_fasth3(model, path, name)
            applied_loras = [entry]
            info["adapter"] = entry
        elif preset.recommended_lora:
            from .fasth3 import apply_packaged_turbo
            model, entry = apply_packaged_turbo(model, preset.recommended_lora)
            applied_loras = [entry]
            info["adapter"] = entry

        # Sampling is a trajectory contract, so carry it with MODEL into H3 Optimizations.
        # Clone only the otherwise-unpatched base path to avoid mutating a shared loader output.
        if not applied_loras and callable(getattr(model, "clone", None)):
            model = model.clone()
        setter = getattr(model, "set_attachments", None)
        if callable(setter):
            setter("mmh3_sampling_profile", json.loads(json.dumps(info)))
            if preset.recommended_lora:
                setter("mmh3_sampling_adapter", preset.profile)
        return io.NodeOutput(
            json.dumps(info, ensure_ascii=False, indent=2),
            preset.steps,
            preset.video_shift,
            preset.audio_shift,
            preset.sampler,
            preset.scheduler,
            model,
            json.dumps(applied_loras, ensure_ascii=False),
        )


def _sol_attention_inputs(native=False):
    return [
        io.Float.Input("sol_tau", display_name="Tau", default=1.0, min=0.0, max=4.0, step=0.05, advanced=True),
        io.Float.Input("sol_start_percent", display_name="Start fraction", default=0.2, min=0.0, max=1.0, step=0.01, advanced=True),
        io.Float.Input("sol_end_percent", display_name="End fraction", default=0.9, min=0.0, max=1.0, step=0.01, advanced=True),
        io.Int.Input("sol_min_tokens", display_name="Minimum tokens", default=4096, min=0, max=1048576, step=512, advanced=True),
        *([io.Int.Input("sol_extra_tokens", display_name="Extra exact tokens", default=256, min=0, max=256, step=64, advanced=True)] if native else [io.Boolean.Input("sol_int8_qk", display_name="INT8 QK", default=True, advanced=True)]),
        io.Combo.Input("sol_sink_conditioning", display_name="Conditioning protection", options=["exact_kv", "exact_kv_and_rows", "off"], default="exact_kv_and_rows", advanced=True),
        io.String.Input("sol_dense_blocks", display_name="Dense blocks", default="", tooltip=("Non-negative block indices, e.g. 0-2,47-49." if native else "Blocks excluded from Sol, e.g. 0-2,-1. In SLA → Sol these may use SLA instead of dense attention. Blank applies Sol to all blocks."), advanced=True),
    ]


def _vsa_attention_inputs():
    return [
        io.Float.Input("vsa_keep_percent", display_name="Keep video blocks (%)", default=10.0, min=0.5, max=95.0, step=0.5,
                       tooltip="Requires complete VSA model weights and their sampling recipe. No gate transplantation."),
        io.Float.Input("vsa_start_percent", display_name="Start fraction", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
        io.Float.Input("vsa_end_percent", display_name="End fraction", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True),
        io.Int.Input("vsa_min_tokens", display_name="Minimum tokens", default=0, min=0, max=1048576, step=512, advanced=True),
        io.String.Input("vsa_dense_blocks", display_name="Dense blocks", default="", advanced=True,
                        tooltip="Optional dense-only blocks. Changes the VSA execution recipe."),
        io.Boolean.Input("vsa_verbose", display_name="Log attention routing", default=True, advanced=True),
    ]


def _native_sla_attention_inputs():
    return [
        io.Float.Input("native_sla_keep_percent", display_name="Keep blocks (%)", default=15.0, min=0.5, max=95.0, step=0.5,
                       tooltip="Percentage kept exact, not sparsity. Core SLA-style routing; not identical to PlagueKind SLA."),
        *_sol_attention_inputs(native=True)[1:],
    ]


def _sla_attention_inputs():
    return [
        io.Float.Input("sla_sparsity_ratio", display_name="Sparsity", default=0.90, min=0.0, max=0.95, step=0.05, advanced=True),
        io.Combo.Input("sla_block_size", display_name="Block size", options=list(H3_SLA_BLOCK_SIZES), default="64", advanced=True),
        io.Int.Input("sla_min_seq_len", display_name="Minimum sequence length", default=4096, min=0, max=1_000_000, step=1024, advanced=True),
        io.Int.Input("sla_dense_last_steps", display_name="Dense final steps", default=1, min=0, max=8, advanced=True),
        io.Boolean.Input("sla_protect_audio", display_name="Protect audio", default=True, advanced=True),
        io.Combo.Input("sla_dense_backend", display_name="Dense backend", options=list(H3_SLA_DENSE_BACKENDS), default="comfy_kitchen", advanced=True),
    ]


def _vdn_attention_inputs(task_family: str):
    return [
        io.String.Input(
            "vdn_checkpoint", display_name="VDN checkpoint", default=VDN_DEFAULT_CHECKPOINT,
            tooltip="Directory under ComfyUI/models/vdn, e.g. stage-dmd-step-250 or an INT8 ConvRot conversion.",
        ),
        io.Boolean.Input(
            "vdn_apply_turbo_adapter", display_name="VDN 8-step turbo adapter", default=True,
            tooltip="ON: released DMD/turbo path; use 8 sampling steps. OFF: Stage-B path; use about 50 steps.",
        ),
        io.Float.Input("vdn_strength", display_name="VDN adapter strength", default=1.0, min=0.0, max=2.0, step=0.05, advanced=True),
        io.Combo.Input("vdn_lora_mode", display_name="VDN LoRA mode", options=list(VDN_LORA_MODES), default="merge", advanced=True),
        io.Combo.Input("vdn_branch_weights", display_name="VDN branch weights", options=list(VDN_BRANCH_WEIGHT_MODES), default="auto", advanced=True),
        io.Combo.Input("vdn_retain_buffers", display_name="VDN retain buffers", options=list(VDN_RETAIN_BUFFER_MODES), default="auto", advanced=True),
        io.Combo.Input("vdn_attention_backend", display_name="VDN local softmax", options=list(VDN_ATTENTION_BACKENDS), default="grouped", advanced=True),
        io.Boolean.Input("vdn_verbose", display_name="VDN verbose", default=False, advanced=True),
        io.String.Input("vdn_task_family", default=task_family, optional=True, advanced=True),
    ]


def h3_optimization_inputs(*, optional=False):
    """Shared attention and accumulation controls for generation and upscale."""
    return [
        io.DynamicCombo.Input(
            "attention",
            display_name="Attention",
            optional=optional,
            options=[
                io.DynamicCombo.Option("Default", []),
                io.DynamicCombo.Option("PyTorch", []),
                io.DynamicCombo.Option("Comfy Kitchen", []),
                io.DynamicCombo.Option("SageAttention (KJ)", [
                    io.Combo.Input("sage_mode", display_name="Implementation", options=["auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"], default="auto", advanced=True),
                    io.Boolean.Input("sage_allow_compile", display_name="Allow compile", default=False, advanced=True),
                ]),
                io.DynamicCombo.Option("Sol (ComfyUI)", _sol_attention_inputs(native=True)),
                io.DynamicCombo.Option("Sol (Kijai)", _sol_attention_inputs()),
                # Retain the serialized label for existing UI/API workflows.
                io.DynamicCombo.Option("Sol Attention", _sol_attention_inputs()),
                io.DynamicCombo.Option("VSA (ComfyUI)", _vsa_attention_inputs()),
                io.DynamicCombo.Option("SLA (ComfyUI)", _native_sla_attention_inputs()),
                io.DynamicCombo.Option("H3 SLA", _sla_attention_inputs()),
                io.DynamicCombo.Option("SLA → Sol (experimental)", _sla_attention_inputs() + _sol_attention_inputs()),
                io.DynamicCombo.Option("VDN-H3 · FL2VA (experimental)", _vdn_attention_inputs("fl2va")),
                io.DynamicCombo.Option("VDN-H3 · Ref2VA (experimental)", _vdn_attention_inputs("ref2va")),
            ],
        ),
        io.Combo.Input("fp16_accumulation", display_name="FP16 accumulation", options=["Default", "Enabled", "Disabled"], default="Default", optional=optional),
    ]


class MMH3H3ModelOptimizations(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3ModelOptimizations",
            display_name="H3 Optimizations",
            category=CATEGORY,
            description="Apply one H3 attention/architecture strategy plus optional scoped FP16 accumulation. VDN-H3 is an exclusive hybrid-attention architecture extension (FL2VA/Ref2VA) and requires the external ComfyUI-VDN-H3 runtime + VDN stage weights; do not stack it with Sol/SLA.",
            inputs=[
                io.Model.Input("model"),
                *h3_optimization_inputs(),
                io.String.Input("sampling_profile_json", default="", multiline=True, optional=True, advanced=True, tooltip="Optional explicit H3 Sampling contract. Normally carried on MODEL automatically; used by internal/refine graphs."),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("optimization_profile_json"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, model, attention: dict | str, fp16_accumulation: str, sampling_profile_json: str = "", **_unused) -> io.NodeOutput:
        selected = dict(attention) if isinstance(attention, dict) else {"attention": attention}
        attention_label = str(selected.get("attention") or "Default")
        attention_modes = {
            "Default": "inherit",
            "PyTorch": "pytorch",
            "Comfy Kitchen": "comfy_kitchen",
            "SageAttention (KJ)": "sage_attention_kj",
            "Sol Attention": "sol_attn",
            "Sol (Kijai)": "sol_attn",
            "Sol (ComfyUI)": "sol_native",
            "SLA (ComfyUI)": "sla_native",
            "VSA (ComfyUI)": "vsa_native",
            "H3 SLA": "h3_sla",
            "SLA → Sol (experimental)": "h3_sla_sol_attn",
            "VDN-H3 · FL2VA (experimental)": "vdn_h3",
            "VDN-H3 · Ref2VA (experimental)": "vdn_h3",
        }
        fp16_modes = {"Default": "inherit", "Enabled": "enabled", "Disabled": "disabled"}
        attention_mode = attention_modes.get(attention_label, attention_label)
        if attention_label == "VDN-H3 · FL2VA (experimental)":
            selected["vdn_task_family"] = "fl2va"
        elif attention_label == "VDN-H3 · Ref2VA (experimental)":
            selected["vdn_task_family"] = "ref2va"
        explicit_sampling = None
        if sampling_profile_json.strip():
            try:
                explicit_sampling = json.loads(sampling_profile_json)
            except json.JSONDecodeError as exc:
                raise MMH3ResourceError(f"sampling_profile_json is not valid JSON: {exc}") from exc
            if not isinstance(explicit_sampling, dict):
                raise MMH3ResourceError("sampling_profile_json must contain a JSON object")
        sampling_profile = resolve_sampling_profile(model, explicit_sampling)
        fp16_mode = fp16_modes.get(fp16_accumulation, fp16_accumulation)
        plan = build_model_optimization_plan(
            enabled=attention_mode != "inherit" or fp16_mode != "inherit",
            attention_mode=attention_mode,
            fp16_accumulation=fp16_mode,
            sampling_profile=sampling_profile,
            **{key: value for key, value in selected.items() if key != "attention"},
        )
        validate_optimization_application(model, plan.attention_mode, plan.fp16_accumulation, sampling_profile)
        runtime_nodes = None
        if plan.enabled:
            import nodes as comfy_nodes  # type: ignore

            runtime_nodes = tuple(getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).keys())
        expansion = build_model_optimization_expansion(
            plan,
            model=model,
            runtime_node_ids=runtime_nodes,
        )
        return io.NodeOutput(
            expansion.model,
            json.dumps(expansion.profile, ensure_ascii=False, indent=2),
            expand=expansion.graph,
        )
