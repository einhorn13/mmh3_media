from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .errors import MMH3ResourceError


CURRENT_BACKEND = "comfy_h3_fun_model_patch_15975"
CURRENT_PROVIDER = "comfy_minimax_h3_fun_model_patch"


@dataclass(frozen=True)
class LoadedH3FunPatch:
    patcher: Any
    filename: str
    path: str
    source_folder: str
    sha256: str
    quantization: str
    dtype: str
    base_family: str
    adaln_form: str
    injection_layers: tuple[int, ...]
    control_in_dim: int

    def capability(self, *, vae_fingerprint: str = "runtime_object") -> dict[str, Any]:
        return {
            "provider": CURRENT_PROVIDER,
            "backend": CURRENT_BACKEND,
            "checkpoint_sha256": self.sha256,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "base_family": self.base_family,
            "adaln_form": self.adaln_form,
            "vae_fingerprint": vae_fingerprint,
            "control_in_dim": self.control_in_dim,
            "injection_layers": list(self.injection_layers),
            "source_folder": self.source_folder,
            "checkpoint": self.filename,
        }


def available_h3_fun_checkpoints(folder_paths_module: Any) -> list[str]:
    values: list[str] = []
    for folder in ("model_patches", "controlnet"):
        try:
            values.extend(folder_paths_module.get_filename_list(folder))
        except (KeyError, RuntimeError):
            pass
    return sorted(dict.fromkeys(values), key=str.casefold)


def resolve_h3_fun_checkpoint(folder_paths_module: Any, name: str) -> tuple[str, str]:
    selected = str(name or "").strip()
    if not selected:
        raise MMH3ResourceError("Select a MiniMax H3 Fun Control checkpoint")
    for folder in ("model_patches", "controlnet"):
        try:
            path = folder_paths_module.get_full_path(folder, selected)
        except (KeyError, RuntimeError):
            path = None
        if path:
            return str(path), folder
    raise MMH3ResourceError(
        f"MiniMax H3 Fun checkpoint {selected!r} was not found in models/model_patches or models/controlnet"
    )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_modules() -> tuple[Any, Any, Any]:
    try:
        from comfy.ldm.minimax import controlnet as h3_control  # type: ignore
        from comfy_extras import nodes_minimax_h3 as h3_nodes  # type: ignore
    except Exception as exc:
        raise MMH3ResourceError(
            "Current ComfyUI MiniMax H3 Fun MODEL_PATCH runtime is unavailable; update ComfyUI to a build containing PR #15975"
        ) from exc
    detector = getattr(h3_control, "is_minimax_h3_fun_state_dict", None)
    control_class = getattr(h3_control, "MiniMaxH3FunControl", None)
    patch_class = getattr(h3_nodes, "MiniMaxH3FunControlPatch", None)
    if not callable(detector) or control_class is None or patch_class is None:
        raise MMH3ResourceError(
            "Current ComfyUI does not expose MiniMaxH3FunControl/MiniMaxH3FunControlPatch; PR #15975-compatible core is required"
        )
    return h3_control, control_class, patch_class


def _adaln_dim(module: Any) -> int | None:
    table = getattr(module, "adaln_t_table", None)
    if table is not None and getattr(table, "ndim", 0) == 2:
        return int(table.shape[-1])
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        blocks = getattr(module, "control_blocks", None)
    if blocks is None or len(blocks) == 0:
        return None
    linear = getattr(getattr(blocks[0], "adaln_proj", None), "linear", None)
    weight = getattr(linear, "weight", None)
    if weight is None or getattr(weight, "ndim", 0) != 2:
        return None
    return int(weight.shape[1])


def _base_diffusion_model(model: Any) -> Any:
    direct = getattr(model, "get_model_object", None)
    if callable(direct):
        try:
            value = direct("diffusion_model")
            if value is not None:
                return value
        except Exception:
            pass
    return getattr(getattr(model, "model", None), "diffusion_model", None)


def load_current_h3_fun_patch(folder_paths_module: Any, checkpoint_name: str, model: Any) -> LoadedH3FunPatch:
    import comfy.model_management  # type: ignore
    import comfy.model_patcher  # type: ignore
    import comfy.ops  # type: ignore
    import comfy.utils  # type: ignore

    h3_control, control_class, _ = _official_modules()
    path, source_folder = resolve_h3_fun_checkpoint(folder_paths_module, checkpoint_name)
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    if not h3_control.is_minimax_h3_fun_state_dict(state_dict):
        raise MMH3ResourceError(f"Checkpoint {checkpoint_name!r} is not recognized as MiniMax H3 Fun Control")

    block_count = 0
    while f"control_blocks.{block_count}.after_proj.weight" in state_dict:
        block_count += 1
    if block_count <= 0:
        raise MMH3ResourceError("MiniMax H3 Fun checkpoint contains no control blocks")

    injection_layers = tuple(range(0, block_count * 10, 10))
    if metadata and metadata.get("control_blocks_places"):
        try:
            injection_layers = tuple(int(v) for v in json.loads(metadata["control_blocks_places"]))
        except Exception as exc:
            raise MMH3ResourceError("Invalid control_blocks_places metadata in H3 Fun checkpoint") from exc
    if len(injection_layers) != block_count:
        raise MMH3ResourceError("H3 Fun control_blocks_places count does not match checkpoint control blocks")

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    quant = comfy.utils.detect_layer_quantization(state_dict, "")
    if quant is not None:
        compute_dtype = torch.bfloat16
        operations = comfy.ops.mixed_precision_ops(quant, compute_dtype)
        quantization = "int8_convrot"
    else:
        compute_dtype = comfy.model_management.unet_dtype(
            model_params=-1,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=comfy.utils.weight_dtype(state_dict),
        )
        manual_cast = comfy.model_management.unet_manual_cast(
            compute_dtype, load_device, supported_dtypes=[torch.bfloat16, torch.float32]
        )
        operations = comfy.ops.pick_operations(compute_dtype, manual_cast)
        quantization = "bf16" if compute_dtype == torch.bfloat16 else str(compute_dtype).replace("torch.", "")

    qkv = state_dict.get("control_blocks.0.attn.qkv_proj.weight")
    q_norm = state_dict.get("control_blocks.0.attn.q_norm.weight")
    proj = state_dict.get("control_proj_in.weight")
    fc1 = state_dict.get("control_blocks.0.mlp.fc1.weight")
    if any(value is None for value in (qkv, q_norm, proj, fc1)):
        raise MMH3ResourceError("H3 Fun checkpoint is missing required current-core tensor keys")

    use_curves = bool(metadata and metadata.get("minimax_h3_fun_controlnet") == "adaln_basis")
    head_dim = int(q_norm.shape[0])
    control_in_dim = int(proj.shape[1] // 4)
    control = control_class(
        control_in_dim=control_in_dim,
        injection_layers=injection_layers,
        hidden_size=int(proj.shape[0]),
        num_attention_heads=int(qkv.shape[0] // (3 * head_dim)),
        attention_head_dim=head_dim,
        ffn_hidden_size=int(fc1.shape[0] // 2),
        time_embed_dim=8 if use_curves else 2688,
        use_adaln_curves=use_curves,
        operations=operations,
        device=offload_device,
        dtype=compute_dtype,
    )
    control.requires_grad_(False)
    patcher_class = getattr(comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher)
    patcher = patcher_class(control, load_device=load_device, offload_device=offload_device)
    control.load_state_dict(state_dict, assign=patcher.is_dynamic())

    base_dim = _adaln_dim(_base_diffusion_model(model))
    control_dim = _adaln_dim(control)
    if base_dim is not None and control_dim is not None and base_dim != control_dim:
        raise MMH3ResourceError(
            f"H3 Fun AdaLN mismatch: base consumes {base_dim} values but control consumes {control_dim}; pair pruned/curve checkpoints together or full-width checkpoints together"
        )

    return LoadedH3FunPatch(
        patcher=patcher,
        filename=str(checkpoint_name),
        path=str(Path(path)),
        source_folder=source_folder,
        sha256=sha256_file(path),
        quantization=quantization,
        dtype=str(compute_dtype).replace("torch.", ""),
        base_family="minimax_h3_pruned" if control_dim is not None and control_dim <= 16 else "minimax_h3",
        adaln_form="pruned_curve" if control_dim is not None and control_dim <= 16 else "single",
        injection_layers=injection_layers,
        control_in_dim=control_in_dim,
    )


def apply_current_h3_fun_patch(
    model: Any,
    loaded: LoadedH3FunPatch,
    vae: Any,
    *,
    control_video: torch.Tensor | None,
    mask: torch.Tensor | None,
    source_video: torch.Tensor | None,
    strength: float,
    start_percent: float,
    end_percent: float,
) -> Any:
    _, _, patch_class = _official_modules()
    if not 0.0 <= float(start_percent) <= float(end_percent) <= 1.0:
        raise MMH3ResourceError("Control start/end must satisfy 0 <= start <= end <= 1")
    if control_video is None and mask is None:
        raise MMH3ResourceError("H3 Fun control requires control_video, or mask + source_video for inpaint")
    if mask is not None and source_video is None:
        raise MMH3ResourceError("H3 Fun inpaint mask requires source_video")

    patched = model.clone()
    sampling = model.get_model_object("model_sampling")
    control_chw = None if control_video is None else control_video[..., :3].movedim(-1, 1)
    source_chw = None if source_video is None else source_video[..., :3].movedim(-1, 1)
    patch = patch_class(
        loaded.patcher,
        vae,
        control_chw,
        mask,
        source_chw,
        float(strength),
        float(sampling.percent_to_sigma(float(start_percent))),
        float(sampling.percent_to_sigma(float(end_percent))),
    )
    patch.register(patched)
    return patched


def current_patch_preflight_info(config: Any, loaded: LoadedH3FunPatch, *, width: int, height: int, frames: int) -> dict[str, Any]:
    """Build canonical proof material from the live MODEL_PATCH runtime.

    This intentionally replaces the old user-authored checkpoint/base JSON. The
    checkpoint hash, quantization and AdaLN form have already been derived from
    the selected file and live base model by ``load_current_h3_fun_patch``.
    """
    from .util import json_dumps_canonical

    capability = loaded.capability(
        vae_fingerprint=getattr(getattr(config, "provider", None), "vae_fingerprint", "") or "runtime_object"
    )
    provider_id = getattr(getattr(config, "provider", None), "provider_id", "")
    capability["provider"] = provider_id or CURRENT_PROVIDER
    fingerprint = hashlib.sha256(json_dumps_canonical(capability).encode("utf-8")).hexdigest()
    return {
        "ready": True,
        "facts": {
            "requested_algorithm": config.requested_algorithm,
            "effective_algorithm": config.effective_algorithm,
            "capability_fingerprint": fingerprint,
            "capability": capability,
            "checkpoint": {
                "sha256": loaded.sha256,
                "sha256_verified": True,
                "quantization": loaded.quantization,
                "dtype": loaded.dtype,
                "base_family": loaded.base_family,
                "adaln_form": loaded.adaln_form,
                "control_in_dim": loaded.control_in_dim,
            },
            "base": {
                "family": loaded.base_family,
                "adaln_form": loaded.adaln_form,
                "vae_fingerprint": capability["vae_fingerprint"],
            },
            "media": {"width": int(width), "height": int(height), "target_frames": int(frames)},
        },
    }
