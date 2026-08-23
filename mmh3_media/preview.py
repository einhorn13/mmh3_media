from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .archive import get_resource_payload
from .constants import PREVIEW_CACHE_VERSION
from .core import MMH3Media
from .errors import MMH3ResourceError
from .h3 import nested_parts, validate_h3_av_latent
from .util import utc_now_iso


def _to_image_batch(decoded: Any) -> torch.Tensor:
    if not isinstance(decoded, torch.Tensor):
        raise MMH3ResourceError("FastVAE decode did not return an IMAGE tensor")
    x = decoded.detach().to(device="cpu", dtype=torch.float32)
    if x.ndim == 5:
        # Common video layouts: [B,T,H,W,C] or [B,C,T,H,W].
        if x.shape[-1] in (1, 3, 4):
            x = x.reshape(-1, *x.shape[-3:])
        elif x.shape[1] in (1, 3, 4):
            x = x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[3], x.shape[4], x.shape[1])
    if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
        return x.clamp(0, 1)
    if x.ndim == 4 and x.shape[1] in (1, 3, 4):
        return x.permute(0, 2, 3, 1).clamp(0, 1)
    raise MMH3ResourceError(f"Unsupported FastVAE decode shape {tuple(x.shape)}")


def _contact_sheet(images: torch.Tensor, max_size: int) -> torch.Tensor:
    n, h, w, c = images.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        chw = images.permute(0, 3, 1, 2)
        images = F.interpolate(chw, size=(nh, nw), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        h, w = nh, nw
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    sheet = torch.zeros((rows * h, cols * w, c), dtype=images.dtype)
    for i in range(n):
        r, col = divmod(i, cols)
        sheet[r*h:(r+1)*h, col*w:(col+1)*w] = images[i]
    return sheet.unsqueeze(0)


def build_fastvae_preview(packet: MMH3Media, fast_vae: Any, *, frames: int = 8, max_size: int = 768) -> tuple[MMH3Media, torch.Tensor]:
    if fast_vae is None or not hasattr(fast_vae, "decode"):
        raise MMH3ResourceError("Preview regeneration requires a VAE/FastVAE object with decode()")
    res = packet.get_by_role_slot("h3_av_latent", 0)
    if res is None:
        raise MMH3ResourceError("FastVAE preview requires h3_av_latent")
    latent = get_resource_payload(packet, res)
    validate_h3_av_latent(latent, strict_audio_length=False)
    video = nested_parts(latent["samples"])[0]
    t = int(video.shape[2])
    count = max(1, min(int(frames), t))
    idx = torch.linspace(0, t - 1, count).round().long().unique()
    sampled = video[:, :, idx]
    decoded = fast_vae.decode(sampled)
    images = _to_image_batch(decoded)
    sheet = _contact_sheet(images, max(64, int(max_size)))
    source_hash = res.get("sha256")
    metadata = {
        "cache": {
            "type": "preview",
            "version": PREVIEW_CACHE_VERSION,
            "generated_at": utc_now_iso(),
            "source_resource_id": res["id"],
            "source_sha256": source_hash,
            "decoder": "fastvae",
            "requested_frames": int(frames),
            "decoded_frames": int(images.shape[0]),
            "max_size": int(max_size),
            "integrity": "excluded",
        }
    }
    out = packet.put(sheet, kind="image", role="preview", mode="upsert", metadata=metadata, record_history=False)
    return out, sheet


def preview_is_fresh(packet: MMH3Media) -> bool:
    preview = packet.get_by_role_slot("preview", 0)
    source = packet.get_by_role_slot("h3_av_latent", 0)
    if preview is None or source is None:
        return False
    cache = preview.get("metadata", {}).get("cache", {})
    if cache.get("type") != "preview" or cache.get("source_resource_id") != source.get("id"):
        return False
    expected = source.get("sha256")
    saved = cache.get("source_sha256")
    if expected and saved:
        return expected == saved
    # In-memory unsaved source has no stable hash; preview is only fresh if generated from this resource id.
    return source.get("id") == cache.get("source_resource_id")
