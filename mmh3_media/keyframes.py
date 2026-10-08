"""Explicit pixel-space preparation of H3 keyframes; never resizes latent state."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .errors import MMH3ResourceError


def prepare_keyframe(image, width, height, mode="crop", background=0.0):
    width, height = int(width), int(height)
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise MMH3ResourceError("Keyframe canvas must use positive multiples of 32")
    if mode not in {"crop", "contain", "stretch"}:
        raise MMH3ResourceError("Choose crop, contain or stretch")
    if not math.isfinite(float(background)) or not 0 <= background <= 1:
        raise MMH3ResourceError("Background must be within 0–1")
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3 or not image.is_floating_point():
        raise MMH3ResourceError("Keyframe requires a floating RGB IMAGE batch")
    if min(image.shape[:3]) < 1:
        raise MMH3ResourceError("Keyframe batch is empty")
    n, sh, sw, _ = image.shape
    factor = (max if mode == "crop" else min)(width / sw, height / sh)
    rw, rh = (width, height) if mode == "stretch" else (max(1, round(sw * factor)), max(1, round(sh * factor)))
    resized = F.interpolate(image.movedim(-1, 1).float(), size=(rh, rw), mode="bilinear", align_corners=False).movedim(1, -1).to(image.dtype)
    if mode == "crop":
        x, y = (rw - width) // 2, (rh - height) // 2
        result = resized[:, y:y + height, x:x + width]
    elif mode == "contain":
        result = image.new_full((n, height, width, 3), float(background))
        x, y = (width - rw) // 2, (height - rh) // 2
        result[:, y:y + rh, x:x + rw] = resized
    else:
        result = resized
    return result.contiguous(), {"mode": mode, "source": [sw, sh], "target": [width, height],
                                 "resized": [rw, rh], "background": float(background), "pixel_space": True}


def validate_refinement_mask(mask, frames=None):
    if (not isinstance(mask, torch.Tensor) or mask.ndim != 3 or min(mask.shape) < 1
            or (frames is not None and mask.shape[0] not in {1, frames})):
        raise MMH3ResourceError("Mask must be one frame or the complete video timeline")
    if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise MMH3ResourceError("Mask values must be finite and within 0–1")


def composite_refinement_mask(source, refined, mask):
    if (not isinstance(source, torch.Tensor) or not isinstance(refined, torch.Tensor)
            or source.ndim != 4 or source.shape != refined.shape or source.shape[-1] != 3
            or min(source.shape[:3]) < 1 or not source.is_floating_point() or not refined.is_floating_point()):
        raise MMH3ResourceError("Masked refinement requires matching RGB timelines")
    validate_refinement_mask(mask, source.shape[0])
    alpha = F.interpolate(mask[:, None].float(), size=source.shape[1:3], mode="nearest").movedim(1, -1).to(source)
    return source * (1 - alpha) + refined.to(source) * alpha
