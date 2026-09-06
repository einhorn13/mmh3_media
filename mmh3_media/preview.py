from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .archive import get_representation_bytes
from .core import MMH3Media
from .errors import MMH3ResourceError
from .h3 import nested_parts, validate_h3_av_latent
from .representations import (
    PACKET_TARGET,
    best_representation,
    decode_image_representation,
    representation_is_fresh,
)


def _to_image_batch(decoded: Any) -> torch.Tensor:
    if not isinstance(decoded, torch.Tensor):
        raise MMH3ResourceError("FastVAE decode did not return an IMAGE tensor")
    x = decoded.detach().to(device="cpu", dtype=torch.float32)
    if x.ndim == 5:
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


def _primary_latent(packet: MMH3Media):
    ref = packet.primary("latent")
    if ref is None:
        raise MMH3ResourceError(
            "FastVAE preview requires an explicit primary latent binding; v0.3 does not infer primary from roles"
        )
    return ref


def build_fastvae_preview(
    packet: MMH3Media,
    fast_vae: Any,
    *,
    frames: int = 8,
    max_size: int = 768,
) -> tuple[MMH3Media, torch.Tensor]:
    """Generate an explicit H3 latent contact-sheet representation.

    The preview is attached to the primary latent through the representation
    index; it never creates a semantic ``resources[]`` entry.
    """
    if fast_vae is None or not hasattr(fast_vae, "decode"):
        raise MMH3ResourceError("Preview regeneration requires a VAE/FastVAE object with decode()")
    ref = _primary_latent(packet)
    latent = ref.materialize()
    validate_h3_av_latent(latent, strict_audio_length=False)
    video = nested_parts(latent["samples"])[0]
    t = int(video.shape[2])
    count = max(1, min(int(frames), t))
    idx = torch.linspace(0, t - 1, count).round().long().unique()
    sampled = video[:, :, idx]
    decoded = fast_vae.decode(sampled)
    images = _to_image_batch(decoded)
    sheet = _contact_sheet(images, max(64, int(max_size)))
    out, _repr_id = packet.put_representation(
        target=ref.resource_id,
        kind="contact_sheet",
        payload=sheet,
        media_type="image/png",
        generator="mmh3.preview.h3_fastvae_contact_sheet.v1",
        metadata={
            "requested_frames": int(frames),
            "decoded_frames": int(images.shape[0]),
            "max_size": int(max_size),
            "width": int(sheet.shape[2]),
            "height": int(sheet.shape[1]),
        },
        replace_same_kind=True,
    )
    return out, sheet


def preview_representation(packet: MMH3Media, *, target: str | None = None) -> dict[str, Any] | None:
    if target is None:
        latent = packet.primary("latent")
        target = latent.resource_id if latent is not None else PACKET_TARGET
    return best_representation(
        packet,
        target,
        kinds=("packet_overview", "thumbnail", "poster", "contact_sheet", "mask_overlay", "waveform"),
        fresh_only=True,
    )


def preview_is_fresh(packet: MMH3Media, *, target: str | None = None) -> bool:
    record = preview_representation(packet, target=target)
    return bool(record is not None and representation_is_fresh(packet, record))


def materialize_preview_image(packet: MMH3Media, *, target: str | None = None) -> torch.Tensor:
    record = preview_representation(packet, target=target)
    if record is None:
        raise MMH3ResourceError("No fresh image representation is available")
    media_type = str(record.get("media_type") or "")
    if not media_type.startswith("image/"):
        raise MMH3ResourceError(f"Selected representation is not an image ({media_type or 'unknown type'})")
    rep_id = record["id"]
    cached = packet.representation_payloads.get(rep_id)
    if isinstance(cached, torch.Tensor):
        return cached
    body = get_representation_bytes(packet, record)
    return decode_image_representation(body)


def _fmt_number(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(int(value))


def resource_summary(packet: MMH3Media, resource_id: str) -> str:
    ref = packet.ref(resource_id)
    descriptor = ref.descriptor
    kind = descriptor["kind"].upper()
    observed = descriptor.get("descriptor", {})
    shape = observed.get("shape", {}) if isinstance(observed.get("shape"), dict) else {}
    timing = observed.get("timing", {}) if isinstance(observed.get("timing"), dict) else {}
    tensor = observed.get("tensor", {}) if isinstance(observed.get("tensor"), dict) else {}
    parts = [kind]
    width, height = shape.get("width"), shape.get("height")
    if isinstance(width, int) and isinstance(height, int):
        parts.append(f"{width}×{height}")
    frames = shape.get("frames")
    if isinstance(frames, int):
        parts.append(f"{frames}f")
    fps = _fmt_number(timing.get("fps"))
    if fps:
        parts.append(f"{fps} fps")
    duration = _fmt_number(timing.get("duration"))
    if duration:
        parts.append(f"{duration} s")
    channels = shape.get("channels")
    if isinstance(channels, int):
        parts.append(f"{channels} ch")
    sample_rate = timing.get("sample_rate")
    if isinstance(sample_rate, int):
        parts.append(f"{sample_rate / 1000:g} kHz")
    if kind == "LATENT":
        tensor_shape = tensor.get("shape")
        if isinstance(tensor_shape, list):
            parts.append("×".join(str(x) for x in tensor_shape))
        h3 = descriptor.get("extensions", {}).get("minimax_h3")
        if isinstance(h3, dict):
            parts.append("H3")
    name = str(descriptor.get("name") or "").strip()
    prefix = f"{name} · " if name else ""
    return prefix + " · ".join(parts)


def preview_info(packet: MMH3Media, *, target: str) -> dict[str, Any]:
    """Descriptor-first preview card; does not materialize source payload data."""
    if target == PACKET_TARGET:
        primary = {
            kind: (packet.primary(kind).resource_id if packet.primary(kind) is not None else None)
            for kind in packet._primary_kinds()
        }
        summaries = []
        for kind in packet._primary_kinds():
            ref = packet.primary(kind)
            if ref is not None:
                summaries.append(resource_summary(packet, ref.resource_id))
        record = best_representation(packet, target, fresh_only=True)
        return {
            "target": target,
            "type": "packet",
            "name": packet.manifest.get("name") or "untitled",
            "summary": " | ".join(summaries) if summaries else "Empty packet",
            "primary": primary,
            "resource_count": len(packet.manifest.get("resources", [])),
            "representation": record,
        }

    ref = packet.ref(target)
    record = best_representation(packet, target, fresh_only=True)
    return {
        "target": target,
        "type": "resource",
        "summary": resource_summary(packet, target),
        "descriptor": ref.descriptor,
        "representation": record,
    }


__all__ = [
    "build_fastvae_preview",
    "preview_representation",
    "preview_is_fresh",
    "materialize_preview_image",
    "resource_summary",
    "preview_info",
]
