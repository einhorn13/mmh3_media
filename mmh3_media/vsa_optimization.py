"""Admission for native ComfyUI VSA; no gate transplantation or weight loading."""
from __future__ import annotations

from .errors import MMH3ResourceError


def validate_vsa_model(model) -> dict:
    getter = getattr(model, "get_model_object", None)
    diffusion = getter("diffusion_model") if callable(getter) else None
    if diffusion is None or not any(cls.__name__ == "MiniMaxH3Model" for cls in type(diffusion).__mro__):
        raise MMH3ResourceError("Native VSA requires a loaded ComfyUI MiniMaxH3Model with VSA gates")
    blocks = getattr(diffusion, "blocks", ())
    if not blocks:
        raise MMH3ResourceError("Native VSA model has no H3 blocks")
    shapes = []
    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        gate = getattr(attn, "to_gate_compress", None)
        weight = getattr(gate, "weight", None)
        shape = tuple(getattr(weight, "shape", ()))
        hidden = getattr(getattr(attn, "qkv_proj", None), "in_features", None)
        heads = getattr(attn, "heads", 0)
        head_dim = getattr(attn, "head_dim", 0)
        if not callable(gate) or hidden is None or head_dim != 128 or heads <= 0 or shape != (heads * head_dim, hidden):
            raise MMH3ResourceError(
                f"Native VSA block {index} has missing/incompatible to_gate_compress weights; "
                "load a complete VSA checkpoint. Ref2VA gate transplantation is not supported."
            )
        shapes.append(list(shape))
    return {"gate_blocks": len(shapes), "gate_shapes": shapes, "runtime_validated": False}


def validate_vsa_sampling(profile, adapter=None):
    # Kernel selection cannot manufacture a trained sampling recipe. Allow a
    # caller-managed recipe or explicit Custom, but never silently reuse our
    # ordinary Standard/Turbo/FastH3-dense/VDN profiles.
    if adapter or (profile and (profile.get("adapter") or profile.get("recommended_lora"))):
        raise MMH3ResourceError("Native VSA requires VSA model weights, not an ordinary acceleration adapter")
    if profile and (profile.get("profile") != "custom" or profile.get("task_family") == "ref2va"):
        raise MMH3ResourceError("Native VSA requires a checkpoint-specific external or Custom sampling recipe; built-in dense/Turbo/VDN and Ref2VA recipes are unsupported")
