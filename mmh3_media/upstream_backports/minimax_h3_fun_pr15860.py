from __future__ import annotations

"""Temporary backport of draft ComfyUI PR #15860.

SOURCE: https://github.com/Comfy-Org/ComfyUI/pull/15860
COMMIT: 6410a035e3c211055f86ee7445ee1fea85783f18 (kijai)
LICENSE: GPL-3.0, inherited from ComfyUI.

REMOVE THIS MODULE after an official ComfyUI release provides all three native
capabilities: MiniMaxH3ControlNet loading, MiniMaxH3Model control streams, and
MiniMaxH3FunControlNetApply. The installer is deliberately fail-closed and does
not mutate a core that already exposes the native implementation.
"""

import inspect
import logging
import textwrap
from dataclasses import dataclass
from functools import wraps
from typing import Any

from .minimax_h3_forward_patch import (
    BackportCompatibilityError,
    PR_COMMIT,
    PR_URL,
    patch_forward_source,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackportStatus:
    mode: str
    detail: str
    original_forward_sha256: str = ""


_STATUS: BackportStatus | None = None
_APPLY_CLASS: type[Any] | None = None


def _is_minimax_h3_fun_state_dict(state_dict: dict[str, Any]) -> bool:
    return (
        "control_proj_in.weight" in state_dict
        and "control_blocks.0.adaln_proj.linear.weight" in state_dict
        and (
            "control_blocks.0.attn.to_q.weight" in state_dict
            or "control_blocks.0.attn.qkv_proj.weight" in state_dict
        )
    )


def _install_forward_patch(model_module: Any) -> str:
    current = model_module.MiniMaxH3Model._forward
    if "control" in inspect.signature(current).parameters:
        return "native"

    source = textwrap.dedent(inspect.getsource(current))
    result = patch_forward_source(source)
    namespace = dict(vars(model_module))
    filename = inspect.getsourcefile(current) or "<MiniMaxH3Model._forward>"
    exec(compile(result.source, filename, "exec"), namespace)
    patched = namespace["_forward"]
    patched.__module__ = current.__module__
    patched.__qualname__ = current.__qualname__
    patched.__doc__ = current.__doc__
    setattr(patched, "__mmh3_pr15860_backport__", PR_COMMIT)
    model_module.MiniMaxH3Model._forward = patched
    return result.original_sha256


def _build_runtime_types(comfy_controlnet: Any, model_module: Any, io: Any):
    import torch
    import torch.nn as nn
    import comfy.latent_formats
    import comfy.ldm.common_dit
    import comfy.model_management
    import comfy.ops
    import comfy.utils

    # TEMPORARY UPSTREAM BACKPORT. Derived from ComfyUI PR #15860, commit above.
    class ControlDiTBlock(model_module.DiTBlock):
        def __init__(
            self,
            hidden,
            heads,
            head_dim,
            ffn,
            t_dim,
            eps,
            qk_eps,
            first_block=False,
            apply_silu=True,
            adaln_dtype=None,
            dtype=None,
            device=None,
            operations=None,
        ):
            super().__init__(
                hidden,
                heads,
                head_dim,
                ffn,
                t_dim,
                eps,
                qk_eps,
                apply_silu=apply_silu,
                adaln_dtype=adaln_dtype,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            if first_block:
                self.before_proj = operations.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)
            self.after_proj = operations.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)

    class MiniMaxH3FunControl(nn.Module):
        def __init__(
            self,
            control_in_dim=49,
            injection_layers=(0, 10, 20, 30, 40),
            hidden_size=5376,
            num_attention_heads=56,
            attention_head_dim=128,
            ffn_hidden_size=14336,
            time_embed_dim=2688,
            patch_size=(1, 2, 2),
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            use_adaln_curves=False,
            dtype=None,
            device=None,
            operations=None,
        ):
            super().__init__()
            self.dtype = dtype
            self.patch_size = tuple(patch_size)
            self.injection_layers = tuple(injection_layers)
            patch_dim = control_in_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
            self.control_proj_in = operations.Linear(
                patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device
            )
            self.control_blocks = nn.ModuleList(
                [
                    ControlDiTBlock(
                        hidden_size,
                        num_attention_heads,
                        attention_head_dim,
                        ffn_hidden_size,
                        time_embed_dim,
                        norm_eps,
                        qk_norm_eps,
                        first_block=(index == 0),
                        apply_silu=not use_adaln_curves,
                        adaln_dtype=torch.float32 if use_adaln_curves else dtype,
                        dtype=dtype,
                        device=device,
                        operations=operations,
                    )
                    for index in range(len(self.injection_layers))
                ]
            )

        def init_stream(self, h, control_latent, layout, t_emb):
            adaln_in = self.control_blocks[0].adaln_proj.linear.in_features
            if t_emb.shape[-1] != adaln_in:
                raise RuntimeError(
                    "MiniMax H3 controlnet adaln width {} does not match the base model's timestep "
                    "embedding width {}: the controlnet and base checkpoint use different adaln "
                    "forms (curve basis vs full), convert the controlnet to match the base model.".format(
                        adaln_in, t_emb.shape[-1]
                    )
                )
            patch_dim = self.control_proj_in.weight.shape[1]
            control_latent = comfy.ldm.common_dit.pad_to_patch_size(
                control_latent.to(torch.float32), self.patch_size
            )
            target_rows = model_module.patchify_video(control_latent, self.patch_size)
            if target_rows.shape[1] < patch_dim:
                target_rows = torch.nn.functional.pad(target_rows, (0, patch_dim - target_rows.shape[1]))
            img_update = layout.img_update.to(h.device)
            rows = torch.zeros(img_update.shape[0], patch_dim, dtype=torch.float32, device=h.device)
            rows[img_update] = target_rows
            c = h.clone()
            c[layout.img_pos.to(h.device)] = self.control_proj_in(rows).to(h.dtype)
            return self.control_blocks[0].before_proj(c).add_(h)

        def step(self, index, c, t_emb, mod_segments, rope_freqs, transformer_options=None):
            if transformer_options is None:
                transformer_options = {}
            block = self.control_blocks[index]
            c = model_module.DiTBlock.forward(
                block,
                c,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
            return c, block.after_proj(c)

    class MiniMaxH3ControlNet(comfy_controlnet.ControlNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.inpaint_mask = None
            self.inpaint_video = None

        def set_inpaint(self, mask, video):
            self.inpaint_mask = mask
            self.inpaint_video = video
            return self

        def _fit_frames(self, frames, pixel_t, width, height):
            idx = torch.arange(pixel_t).clamp(max=frames.shape[0] - 1)
            return comfy.utils.common_upscale(
                frames[idx], width, height, self.upscale_algorithm, "center"
            )

        @staticmethod
        def _fit_latent_t(latent, latent_t):
            if latent.shape[2] > latent_t:
                return latent[:, :, :latent_t]
            if latent.shape[2] < latent_t:
                return torch.nn.functional.pad(
                    latent, (0, 0, 0, 0, 0, latent_t - latent.shape[2])
                )
            return latent

        def _encode(self, frames):
            return self.vae.encode(frames.movedim(1, -1)).to(torch.float32)

        def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
            control_prev = None
            if self.previous_controlnet is not None:
                control_prev = self.previous_controlnet.get_control(
                    x_noisy, t, cond, batched_number, transformer_options
                )
            if self.timestep_range is not None:
                if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                    return control_prev

            latent_shapes = cond.get("latent_shapes", None)
            shape = tuple(latent_shapes[0]) if latent_shapes is not None else tuple(x_noisy.shape)
            latent_t, lat_h, lat_w = shape[2], shape[3], shape[4]
            if self.cond_hint is None or self.cond_hint.shape[2:] != (latent_t, lat_h, lat_w):
                if self.vae is None:
                    raise ValueError(
                        "This Controlnet needs a VAE; connect it through MiniMaxH3FunControlNetApply."
                    )
                self.cond_hint = None
                pixel_t = max((latent_t - 2) // 5, 0) * 17 + 5
                scale = self.vae.spacial_compression_encode()
                width, height = lat_w * scale, lat_h * scale
                channels = self.latent_format.latent_channels
                loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
                if self.cond_hint_original is not None:
                    hint = self._fit_latent_t(
                        self._encode(
                            self._fit_frames(self.cond_hint_original, pixel_t, width, height)
                        ),
                        latent_t,
                    )
                else:
                    hint = torch.zeros(1, channels, latent_t, lat_h, lat_w)

                if self.inpaint_mask is not None:
                    mask = (
                        self.inpaint_mask.reshape(
                            -1, 1, self.inpaint_mask.shape[-2], self.inpaint_mask.shape[-1]
                        )
                        > 0.5
                    ).to(torch.float32)
                    idx = torch.arange(pixel_t).clamp(max=mask.shape[0] - 1)
                    mask = comfy.utils.common_upscale(mask[idx], width, height, "bilinear", "center")
                    visibility = 1.0 - (mask > 0.5).to(torch.float32)
                    if self.inpaint_video is not None:
                        source = self._fit_frames(self.inpaint_video, pixel_t, width, height)
                    else:
                        source = torch.zeros(pixel_t, 3, height, width)
                    masked_latent = self._fit_latent_t(
                        self._encode(source * visibility.to(source.device)), latent_t
                    )
                    visibility_latent = torch.nn.functional.interpolate(
                        visibility.squeeze(1)[None, None],
                        size=(latent_t, lat_h, lat_w),
                        mode="trilinear",
                        align_corners=False,
                    )
                    hint = torch.cat(
                        [hint, visibility_latent.to(hint.device), masked_latent.to(hint.device)], dim=1
                    )
                comfy.model_management.load_models_gpu(loaded_models)
                self.cond_hint = hint
            self.cond_hint = self.cond_hint.to(device=x_noisy.device)
            out = {
                "minimax_fun": [
                    {"model": self.control_model, "latent": self.cond_hint, "strength": self.strength}
                ]
            }
            if control_prev is not None:
                for key, value in control_prev.items():
                    out[key] = value + out[key] if key == "minimax_fun" else value
            return out

        def copy(self):
            clone = MiniMaxH3ControlNet(
                None,
                global_average_pooling=self.global_average_pooling,
                load_device=self.load_device,
                manual_cast_dtype=self.manual_cast_dtype,
            )
            clone.control_model = self.control_model
            clone.control_model_wrapped = self.control_model_wrapped
            self.copy_to(clone)
            clone.inpaint_mask = self.inpaint_mask
            clone.inpaint_video = self.inpaint_video
            return clone

    class MiniMaxH3FunControlNetApply(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="MiniMaxH3FunControlNetApply",
                description=(
                    "Temporary GPL-3.0 backport of ComfyUI PR #15860. Apply a MiniMax H3 Fun "
                    "ControlNet with optional control video and/or inpainting mask."
                ),
                display_name="Apply MiniMax H3 Fun ControlNet (PR #15860 backport)",
                search_aliases=["minimax controlnet", "h3 controlnet", "video inpaint controlnet"],
                category="model/conditioning/controlnet",
                inputs=[
                    io.Conditioning.Input("positive"),
                    io.ControlNet.Input("control_net"),
                    io.Vae.Input("vae"),
                    io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                    io.Float.Input(
                        "start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True
                    ),
                    io.Float.Input(
                        "end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True
                    ),
                    io.Image.Input("control_video", optional=True),
                    io.Mask.Input("mask", optional=True, tooltip="1 marks regions to regenerate."),
                    io.Image.Input(
                        "source_video",
                        optional=True,
                        tooltip="Video behind the mask; read only when a mask is connected.",
                    ),
                ],
                outputs=[io.Conditioning.Output(display_name="positive")],
            )

        @classmethod
        def execute(
            cls,
            positive,
            control_net,
            vae,
            strength,
            start_percent,
            end_percent,
            control_video=None,
            mask=None,
            source_video=None,
        ):
            if strength == 0 or (control_video is None and mask is None):
                return io.NodeOutput(positive)
            if not isinstance(control_net, MiniMaxH3ControlNet):
                raise ValueError("this node needs a MiniMax H3 Fun ControlNet")
            control_hint = control_video.movedim(-1, 1) if control_video is not None else None
            source_hint = source_video.movedim(-1, 1) if source_video is not None else None
            cache = {}
            output = []
            for item in positive:
                metadata = item[1].copy()
                previous = metadata.get("control", None)
                if previous in cache:
                    control = cache[previous]
                else:
                    control = control_net.copy().set_cond_hint(
                        control_hint,
                        strength,
                        (start_percent, end_percent),
                        vae=vae,
                    )
                    if mask is not None:
                        control.set_inpaint(mask, source_hint)
                    control.set_previous_controlnet(previous)
                    cache[previous] = control
                metadata["control"] = control
                metadata["control_apply_to_uncond"] = True
                output.append([item[0], metadata])
            return io.NodeOutput(output)

    return ControlDiTBlock, MiniMaxH3FunControl, MiniMaxH3ControlNet, MiniMaxH3FunControlNetApply


def _install_loader_dispatch(comfy_controlnet: Any, loader: Any) -> None:
    current = comfy_controlnet.load_controlnet_state_dict
    if getattr(current, "__mmh3_pr15860_backport__", None) == PR_COMMIT:
        return

    @wraps(current)
    def dispatch(state_dict, model=None, model_options=None):
        if model_options is None:
            model_options = {}
        if _is_minimax_h3_fun_state_dict(state_dict):
            return loader(state_dict, model_options=model_options)
        return current(state_dict, model=model, model_options=model_options)

    setattr(dispatch, "__mmh3_pr15860_backport__", PR_COMMIT)
    setattr(dispatch, "__mmh3_original__", current)
    comfy_controlnet.load_controlnet_state_dict = dispatch


def ensure_minimax_h3_fun_backport(io: Any) -> tuple[BackportStatus, type[Any] | None]:
    global _STATUS, _APPLY_CLASS
    if _STATUS is not None:
        return _STATUS, _APPLY_CLASS

    import torch
    import comfy.controlnet as comfy_controlnet
    import comfy.latent_formats
    import comfy.ldm.minimax.model as model_module
    import comfy.model_management
    import comfy.ops
    import comfy.utils

    try:
        import comfy_extras.nodes_minimax_h3 as native_nodes
    except ImportError:
        native_nodes = None

    native_apply = getattr(native_nodes, "MiniMaxH3FunControlNetApply", None)
    native_control = getattr(comfy_controlnet, "MiniMaxH3ControlNet", None)
    native_forward = "control" in inspect.signature(model_module.MiniMaxH3Model._forward).parameters
    if native_apply is not None and native_control is not None and native_forward:
        _STATUS = BackportStatus("native", "Official ComfyUI MiniMax H3 Fun support is present")
        return _STATUS, None
    if any((native_apply is not None, native_control is not None, native_forward)):
        present = [
            name
            for name, value in (
                ("apply", native_apply is not None),
                ("controlnet", native_control is not None),
                ("model_forward", native_forward),
            )
            if value
        ]
        raise BackportCompatibilityError(
            "Partial native MiniMax H3 Fun implementation detected; refusing a mixed backport: "
            + ", ".join(present)
        )

    forward_sha = _install_forward_patch(model_module)
    _, MiniMaxH3FunControl, MiniMaxH3ControlNet, apply_class = _build_runtime_types(
        comfy_controlnet, model_module, io
    )

    def convert_minimax_h3_fun(state_dict):
        output = {}
        for key, value in state_dict.items():
            if key.endswith(".attn.to_q.weight"):
                base = key[: -len("to_q.weight")]
                output[base + "qkv_proj.weight"] = torch.cat(
                    [
                        state_dict[base + "to_q.weight"],
                        state_dict[base + "to_k.weight"],
                        state_dict[base + "to_v.weight"],
                    ],
                    dim=0,
                )
            elif key.endswith(".attn.to_k.weight") or key.endswith(".attn.to_v.weight"):
                continue
            elif key.endswith(".ff.net.0.proj.weight"):
                half = value.shape[0] // 2
                output[key.replace(".ff.net.0.proj.", ".mlp.fc1.")] = torch.cat(
                    [value[half:], value[:half]], dim=0
                )
            else:
                output[
                    key.replace(".attn.norm_q.", ".attn.q_norm.")
                    .replace(".attn.norm_k.", ".attn.k_norm.")
                    .replace(".attn.to_out.0.", ".attn.out_proj.")
                    .replace(".ff.net.2.", ".mlp.fc2.")
                ] = value
        return output

    def load_controlnet_minimax_h3(state_dict, model_options=None):
        if model_options is None:
            model_options = {}
        if "control_blocks.0.attn.to_q.weight" in state_dict:
            state_dict = convert_minimax_h3_fun(state_dict)
        load_device = comfy.model_management.get_torch_device()
        quantization = comfy.utils.detect_layer_quantization(state_dict, "")
        if quantization is not None:
            unet_dtype = model_options.get("dtype", torch.bfloat16)
        else:
            unet_dtype = model_options.get("dtype", comfy.utils.weight_dtype(state_dict))
        manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
        operations = model_options.get("custom_operations", None)
        if operations is None:
            operations = (
                comfy.ops.mixed_precision_ops(quantization, unet_dtype)
                if quantization is not None
                else comfy.ops.pick_operations(unet_dtype, manual_cast_dtype)
            )
        projection = state_dict["control_proj_in.weight"]
        num_blocks = 0
        while f"control_blocks.{num_blocks}.after_proj.weight" in state_dict:
            num_blocks += 1
        qkv = state_dict["control_blocks.0.attn.qkv_proj.weight"]
        head_dim = state_dict["control_blocks.0.attn.q_norm.weight"].shape[0]
        timestep_dim = state_dict["control_blocks.0.adaln_proj.linear.weight"].shape[1]
        hidden = projection.shape[0]
        model = MiniMaxH3FunControl(
            control_in_dim=projection.shape[1] // 4,
            injection_layers=tuple(range(0, num_blocks * 10, 10)),
            hidden_size=hidden,
            num_attention_heads=qkv.shape[0] // (3 * head_dim),
            attention_head_dim=head_dim,
            ffn_hidden_size=state_dict["control_blocks.0.mlp.fc1.weight"].shape[0] // 2,
            time_embed_dim=timestep_dim,
            use_adaln_curves=timestep_dim <= 16,
            operations=operations,
            device=comfy.model_management.unet_offload_device(),
            dtype=unet_dtype,
        )
        model = comfy_controlnet.controlnet_load_state_dict(model, state_dict)
        return MiniMaxH3ControlNet(
            model,
            compression_ratio=1,
            latent_format=comfy.latent_formats.MiniMaxH3Video(),
            load_device=load_device,
            manual_cast_dtype=manual_cast_dtype,
            extra_conds=[],
        )

    comfy_controlnet.MiniMaxH3ControlNet = MiniMaxH3ControlNet
    comfy_controlnet.convert_minimax_h3_fun = convert_minimax_h3_fun
    comfy_controlnet.load_controlnet_minimax_h3 = load_controlnet_minimax_h3
    _install_loader_dispatch(comfy_controlnet, load_controlnet_minimax_h3)

    _APPLY_CLASS = apply_class
    _STATUS = BackportStatus(
        "backport",
        f"Temporary ComfyUI PR #15860 backport active ({PR_COMMIT[:12]})",
        forward_sha,
    )
    LOGGER.warning("%s; remove after the official ComfyUI implementation ships", _STATUS.detail)
    return _STATUS, _APPLY_CLASS


__all__ = [
    "BackportStatus",
    "ensure_minimax_h3_fun_backport",
    "PR_COMMIT",
    "PR_URL",
]
