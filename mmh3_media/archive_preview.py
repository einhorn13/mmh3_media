from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache
import zipfile

from .archive import get_representation_bytes, load_archive
from .errors import MMH3ResourceError
from .inspection import inspect_packet
from .representations import PACKET_TARGET, best_representation, representation_is_fresh
from .util import safe_member_path
from .media_thumbnail import thumbnail_from_file


MAX_LOAD_PREVIEW_BYTES = 16 * 1024 * 1024
PROMPT_SUMMARY_CHARS = 240
RASTER_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_THUMBNAIL_SOURCE_BYTES = 256 * 1024 * 1024


@lru_cache(maxsize=64)
def _source_thumbnail(path: str, mtime_ns: int, size: int, resource_id: str) -> bytes:
    # Cache key includes archive revision: overwrites must not retain an old poster.
    packet = load_archive(path, verify="manifest")
    resource = packet.get_by_id(resource_id)
    if resource is None or resource["kind"] not in {"image", "video"}:
        raise MMH3ResourceError("Automatic preview requires an image or video resource")
    member = safe_member_path(packet.origins.get(resource_id, resource["path"]))
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.file_size > MAX_THUMBNAIL_SOURCE_BYTES:
            raise MMH3ResourceError("Source exceeds the 256 MiB automatic preview limit")
        with archive.open(info) as stream:
            return thumbnail_from_file(stream, resource["kind"])


def _gallery(packet):
    from .preview import resource_summary

    primary = set(packet.primary_bindings.values())
    resources = sorted(packet.resources(), key=lambda r: (
        0 if r["id"] in primary and r["kind"] == "video" else
        1 if r["id"] in primary and r["kind"] == "latent" else
        2 if r.get("role") == "context" else 3,
        r.get("order") if r.get("order") is not None else 9999,
    ))
    items = []
    overview = best_representation(packet, PACKET_TARGET,
                                   kinds=("packet_overview", "thumbnail", "poster", "contact_sheet"), fresh_only=True)
    if overview is not None and overview.get("media_type") in RASTER_MEDIA_TYPES:
        items.append({"resource_id": PACKET_TARGET, "kind": "image", "role": "overview",
                      "name": "Archive overview", "summary": "Cached archive overview", "primary": False,
                      "representation_id": overview["id"], "preview_source": None})
    for resource in resources:
        if resource["kind"] not in {"image", "video", "latent", "audio", "mask"}:
            continue
        rid = resource["id"]
        usage = resource.get("extensions", {}).get("minimax_h3", {}).get("context", {}).get("usage")
        label = {"first_frame": "First frame", "last_frame": "Last frame"}.get(usage)
        candidates = [record for record in packet.representation_records.get(rid, [])
                      if record.get("media_type") in RASTER_MEDIA_TYPES and representation_is_fresh(packet, record)]
        preferred = "poster" if resource["kind"] == "video" else "thumbnail"
        record = next((r for r in reversed(candidates) if r["kind"] == preferred), candidates[-1] if candidates else None)
        items.append({
            "resource_id": rid, "kind": resource["kind"], "role": resource.get("role"),
            "name": resource.get("name") or label or resource["kind"].upper(),
            "summary": resource_summary(packet, rid), "primary": rid in primary,
            "representation_id": record["id"] if record else None,
            "can_preview_source": resource["kind"] in {"image", "video"},
            "preview_source": record.get("metadata", {}).get("preview_source") if record else None,
        })
    return items


def _summary_text(value: Any, limit: int = PROMPT_SUMMARY_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _archive_preview_record(packet):
    record = best_representation(
        packet,
        PACKET_TARGET,
        kinds=("packet_overview", "thumbnail", "poster", "contact_sheet"),
        fresh_only=True,
    )
    if record is not None and str(record.get("media_type") or "").startswith("image/"):
        return record
    candidates = []
    for target, records in packet.representation_records.items():
        if target == PACKET_TARGET:
            continue
        for candidate in records:
            if not representation_is_fresh(packet, candidate):
                continue
            if not str(candidate.get("media_type") or "").startswith("image/"):
                continue
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True)
    return candidates[0] if candidates else None


def build_archive_card(path: str | Path) -> dict[str, Any]:
    """Read manifest + representation metadata only; never decode source media/latents."""
    archive_path = Path(path).expanduser().resolve()
    packet = load_archive(archive_path, verify="manifest")
    inspected = inspect_packet(packet)
    preview = _archive_preview_record(packet)
    stat = archive_path.stat()
    generation = inspected.get("generation", {})
    geometry = inspected.get("geometry", {})
    has = dict(inspected.get("has", {}))
    has["preview"] = preview is not None

    preview_info: dict[str, Any] = {
        "available": preview is not None,
        "fresh": preview is not None,
        "status": "fresh" if preview is not None else "missing",
        "representation_id": preview.get("id") if preview is not None else None,
        "target": preview.get("target") if preview is not None else None,
        "kind": preview.get("kind") if preview is not None else None,
        "media_type": preview.get("media_type") if preview is not None else None,
        "size": preview.get("size") if preview is not None else None,
    }
    return {
        "format": inspected.get("format"),
        "schema_version": inspected.get("schema_version"),
        "id": inspected.get("id"),
        "name": inspected.get("name") or "untitled",
        "size_bytes": stat.st_size,
        "task": generation.get("task") or "—",
        "prompt_summary": _summary_text(generation.get("prompt")),
        "notes_summary": _summary_text(inspected.get("notes"), 160),
        "geometry": {
            "width": geometry.get("width"),
            "height": geometry.get("height"),
            "frames": geometry.get("frames"),
            "fps": geometry.get("fps"),
            "duration": geometry.get("duration"),
        },
        "has": has,
        "refs": inspected.get("refs", {}),
        "warnings": list(inspected.get("warnings", ()))[:3],
        "preview": preview_info,
        "gallery": _gallery(packet),
        "representation_count": sum(len(records) for records in packet.representation_records.values()),
        "revision": f"{stat.st_mtime_ns:x}-{stat.st_size:x}",
    }


def read_archive_preview(path: str | Path, representation_id: str | None = None,
                         resource_id: str | None = None) -> tuple[bytes, str, str]:
    """Use cached representation, or thumbnail one explicit image/video on CPU."""
    archive_path = Path(path).expanduser().resolve()
    if resource_id and not representation_id:
        stat = archive_path.stat()
        return _source_thumbnail(str(archive_path), stat.st_mtime_ns, stat.st_size, resource_id), "image/webp", resource_id
    packet = load_archive(archive_path, verify="manifest")
    record = packet.get_representation(representation_id) if representation_id else _archive_preview_record(packet)
    if record is None or not representation_is_fresh(packet, record):
        raise MMH3ResourceError("The selected MMH3 archive has no fresh image representation")
    media_type = str(record.get("media_type") or "").strip().lower()
    if media_type not in RASTER_MEDIA_TYPES:
        raise MMH3ResourceError(f"MMH3 representation is not an image ({media_type})")
    body = get_representation_bytes(packet, record)
    if len(body) > MAX_LOAD_PREVIEW_BYTES:
        raise MMH3ResourceError(
            f"MMH3 preview representation is too large for the load card ({len(body)} bytes)"
        )
    return body, media_type, str(record["id"])


__all__ = [
    "MAX_LOAD_PREVIEW_BYTES",
    "build_archive_card",
    "read_archive_preview",
]
