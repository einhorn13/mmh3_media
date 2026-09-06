"""Small media previews; never materialize lazy archive resources or decode latents."""
from __future__ import annotations

import io
import logging

import numpy as np
import torch
from PIL import Image

from .representations import best_representation
from .media_thumbnail import thumbnail_from_file, thumbnail_webp
from .errors import MMH3ResourceError


def cache_file_preview(packet, resource_id, source, kind):
    """Use the exact file being saved, including any trims/transforms already applied."""
    if kind not in {"image", "video"}:
        return packet
    preview_kind = "poster" if kind == "video" else "thumbnail"
    if best_representation(packet, resource_id, kinds=(preview_kind,), fresh_only=True):
        return packet
    try:
        body = thumbnail_from_file(source, kind)
        packet, _ = packet.put_representation(
            target=resource_id, kind=preview_kind, payload=body, media_type="image/webp",
            generator="mmh3.save.first_frame.v1", metadata={"sampled_frames": 1},
        )
    except (ValueError, RuntimeError, OSError, MMH3ResourceError) as exc:
        logging.getLogger(__name__).warning("Preview omitted for %s: %s", resource_id, exc)
    return packet


def _sheet(images: torch.Tensor) -> bytes:
    count = min(4, int(images.shape[0]))
    indices = np.linspace(0, int(images.shape[0]) - 1, count).round().astype(int)
    tiles = []
    for index in indices:
        frame = images[int(index)].detach().to(device="cpu", dtype=torch.float32)
        array = (frame.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        if array.shape[-1] == 1:
            array = array[..., 0]
        tile = Image.fromarray(array).convert("RGB")
        tile.thumbnail((320, 240), Image.Resampling.LANCZOS)
        tiles.append(tile)
    columns = min(2, count)
    sheet = Image.new("RGB", (320 * columns, 240 * ((count + columns - 1) // columns)), "#181818")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 320 + (320 - tile.width) // 2,
                          (index // columns) * 240 + (240 - tile.height) // 2))
    buffer = io.BytesIO()
    sheet.save(buffer, format="WEBP", quality=80)
    return buffer.getvalue()


def cache_materialized_previews(packet, *, result_ids=None):
    """Best-effort cache. Loaded/file-backed resources remain lazy and untouched."""
    result_ids = result_ids or {}
    for resource in packet.resources():
        rid, kind = resource["id"], resource["kind"]
        if kind not in {"image", "video"} or rid not in packet.payloads:
            continue
        existing = best_representation(packet, rid, kinds=("thumbnail", "poster", "contact_sheet"), fresh_only=True)
        if existing is not None:
            continue
        payload = packet.payloads[rid]
        # Only this native tensor-backed video type has a non-decoding accessor.
        try:
            if kind == "video":
                if type(payload).__name__ != "VideoFromComponents":
                    continue
                payload = payload.get_components().images
            if not isinstance(payload, torch.Tensor) or payload.ndim != 4 or not payload.shape[0]:
                continue
            array = (payload[0].detach().to(device="cpu", dtype=torch.float32).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            if array.shape[-1] == 1:
                array = array[..., 0]
            body = thumbnail_webp(Image.fromarray(array))
            repr_kind = "thumbnail" if kind == "image" else "poster"
            packet, _ = packet.put_representation(
                target=rid, kind=repr_kind, payload=body, media_type="image/webp",
                generator="mmh3.save.first_frame.v1", metadata={"sampled_frames": 1},
            )
            # Only pair outputs supplied together to Pack; never guess from primary bindings.
            if kind == "video" and result_ids.get("video") == rid and result_ids.get("latent"):
                packet, _ = packet.put_representation(
                    target=result_ids["latent"], kind="contact_sheet", payload=_sheet(payload), media_type="image/webp",
                    generator="mmh3.pack.decoded_result.v1",
                    metadata={"preview_source": "paired_decoded_result", "video_resource_id": rid},
                )
        except (ValueError, TypeError, RuntimeError, OSError, MMH3ResourceError) as exc:
            logging.getLogger(__name__).warning("Preview omitted for %s: %s", rid, exc)
    return packet
