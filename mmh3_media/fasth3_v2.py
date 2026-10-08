"""Complete FastH3 V2 checkpoint admission and the official native Comfy recipe."""
from pathlib import Path

from .errors import MMH3ResourceError

FASTH3_V2_PROFILE = "fasth3 v2 native (8 steps)"
FASTH3_V2_LABEL = "FastH3 V2 - 8 steps"
FASTH3_V2_SOURCE = "mmh3_fasth3_v2_checkpoint"
FASTH3_V2_WORKFLOW = "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_fastvideo_fasth3_t2v.json"
FASTH3_V2_VSA = dict(selection={"selection": "vsa", "keep_percent": 10.0},
    start_percent=0.2, end_percent=1.0, dense_blocks="", min_tokens=12288,
    extra_tokens=0, sink_conditioning="exact_kv_and_rows", verbose=False)


def is_fasth3_v2_checkpoint(name):
    basename = str(name).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return "fasth3_8step_v2_" in basename and basename.endswith(".safetensors")


def validate_fasth3_v2_header(path):
    """Inspect all gate weights before allocating the checkpoint's model tensors."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        prefixes = ("", "diffusion_model.")
        prefix = next((p for p in prefixes if p + "blocks.0.attn.to_gate_compress.weight" in keys), None)
        if prefix is None:
            raise MMH3ResourceError("FastH3 V2 requires the complete Comfy checkpoint with VSA gates, not a LoRA")
        gates = {k for k in keys if k.startswith(prefix + "blocks.") and k.endswith(".attn.to_gate_compress.weight")}
        expected = {f"{prefix}blocks.{i}.attn.to_gate_compress.weight" for i in range(50)}
        if gates != expected or any(tuple(handle.get_slice(k).get_shape()) != (7168, 5376) for k in expected):
            raise MMH3ResourceError("FastH3 V2 requires all 50 compatible VSA gate weights")
    return {"checkpoint": Path(path).name, "gate_blocks": 50, "runtime_validated": False}


def load_fasth3_v2(checkpoint):
    import folder_paths
    import comfy.sd
    if not is_fasth3_v2_checkpoint(checkpoint):
        raise MMH3ResourceError("Select a complete fasth3_8step_v2 Comfy safetensors checkpoint")
    path = folder_paths.get_full_path("diffusion_models", checkpoint)
    if not path:
        raise MMH3ResourceError("Selected FastH3 V2 checkpoint is not installed")
    evidence = validate_fasth3_v2_header(path)
    from comfy_extras import nodes_sparse_attention as sparse
    if sparse.BLOCK_SIZE != 64 or sparse.VSA_CUBE != (4, 4, 4):
        raise MMH3ResourceError("FastH3 V2 requires native ComfyUI 64-token cube VSA support")
    model = comfy.sd.load_diffusion_model(path, model_options={})
    from .vsa_optimization import validate_vsa_model
    checked = validate_vsa_model(model)
    if checked["gate_blocks"] != 50:
        raise MMH3ResourceError("Loaded FastH3 V2 model does not contain 50 VSA blocks")
    model.set_attachments(FASTH3_V2_SOURCE, dict(evidence, checkpoint=checkpoint.replace("\\", "/")))
    return model


def apply_fasth3_v2_recipe(model, profile):
    source = getattr(model, "get_attachment", lambda key: None)(FASTH3_V2_SOURCE)
    if not source:
        raise MMH3ResourceError("FastH3 V2 Sampling requires H3 FastH3 V2 Model (complete checkpoint, not base + LoRA)")
    from .vsa_optimization import validate_vsa_model
    from .optimization_contract import validate_optimization_application, record_optimization
    checked = validate_vsa_model(model)
    validate_optimization_application(model, "vsa_native", sampling_profile=profile)
    from comfy_extras.nodes_sparse_attention import BlockSparseAttention
    result = BlockSparseAttention.execute(model=model, **FASTH3_V2_VSA)
    patched = result.result[0]
    record_optimization(patched, "vsa_native")
    profile["checkpoint"] = dict(source)
    profile["vsa_model"] = checked
    profile["vsa_settings"] = dict(FASTH3_V2_VSA)
    profile["recipe_source"] = FASTH3_V2_WORKFLOW
    return patched
