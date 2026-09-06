from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image

from .errors import MMH3ResourceError
from .util import deep_copy_json, safe_member_path, utc_now_iso

if TYPE_CHECKING:
    from .core import MMH3Media

PACKET_TARGET = "$packet"
REPRESENTATION_KINDS = (
    "thumbnail",
    "poster",
    "contact_sheet",
    "waveform",
    "audio_proxy",
    "video_proxy",
    "mask_overlay",
    "structured_summary",
    "packet_overview",
)
MAX_REPRESENTATION_BYTES = 32 * 1024 * 1024
_REPR_ID_RE = re.compile(r"^repr_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _representation_id() -> str:
    return "repr_" + uuid.uuid4().hex


def representation_source_identity(packet: "MMH3Media", target: str) -> tuple[str, str | None]:
    """Return a cheap content identity for representation freshness.

    Resource representations are bound to the resource content revision/digest.
    Packet-level representations are bound to a semantic packet snapshot token
    that deliberately excludes representation cache state and volatile paths.
    """
    if target == PACKET_TARGET:
        resource_descriptors = []
        for descriptor in packet.resource_descriptors():
            item = deep_copy_json(descriptor)
            item.pop("path", None)
            resource_descriptors.append(item)
        payload = {
            "packet_id": packet.manifest.get("id"),
            "name": packet.manifest.get("name"),
            "generation": packet.manifest.get("generation", {}),
            "notes": packet.manifest.get("notes", ""),
            "primary": dict(packet.primary_bindings),
            "resources": resource_descriptors,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        token = "packet:" + hashlib.sha256(encoded).hexdigest()
        return token, None

    ref = packet.ref(target)
    content = ref.descriptor.get("content", {})
    revision = str(content.get("revision") or "")
    if not revision:
        raise MMH3ResourceError(f"Resource {target!r} has no content revision")
    digest = content.get("digest")
    return revision, str(digest) if isinstance(digest, str) and digest else None


def validate_representation_record(record: Any, *, packet: "MMH3Media | None" = None) -> None:
    if not isinstance(record, dict):
        raise MMH3ResourceError("Representation record must be an object")
    required = (
        "id",
        "target",
        "kind",
        "media_type",
        "path",
        "source_revision",
        "source_digest",
        "created_at",
        "generator",
        "metadata",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise MMH3ResourceError(f"Representation record missing fields: {', '.join(missing)}")
    rid = record["id"]
    if not isinstance(rid, str) or not _REPR_ID_RE.match(rid):
        raise MMH3ResourceError("Representation id must use repr_<32 hex> form")
    target = record["target"]
    if not isinstance(target, str) or not target:
        raise MMH3ResourceError("Representation target must be a non-empty string")
    if packet is not None and target != PACKET_TARGET and packet.get_by_id(target) is None:
        raise MMH3ResourceError(f"Representation target {target!r} does not exist in packet")
    kind = record["kind"]
    if kind not in REPRESENTATION_KINDS:
        raise MMH3ResourceError(f"Unsupported representation kind {kind!r}")
    media_type = record["media_type"]
    if media_type is not None and (not isinstance(media_type, str) or "/" not in media_type):
        raise MMH3ResourceError("Representation media_type must be null or a MIME type")
    path = record["path"]
    if path is not None:
        if not isinstance(path, str) or not path:
            raise MMH3ResourceError("Representation path must be null or a non-empty string")
        safe_member_path(path)
    source_revision = record["source_revision"]
    if not isinstance(source_revision, str) or not source_revision:
        raise MMH3ResourceError("Representation source_revision must be a non-empty string")
    source_digest = record["source_digest"]
    if source_digest is not None:
        if not isinstance(source_digest, str) or not source_digest.startswith("sha256:"):
            raise MMH3ResourceError("Representation source_digest must be null or sha256:<hex>")
        raw_digest = source_digest.split(":", 1)[1]
        if not _SHA256_RE.match(raw_digest):
            raise MMH3ResourceError("Representation source_digest must contain a 64-character SHA-256")
    for key in ("created_at", "generator"):
        if not isinstance(record[key], str) or not record[key]:
            raise MMH3ResourceError(f"Representation {key} must be a non-empty string")
    if not isinstance(record["metadata"], dict):
        raise MMH3ResourceError("Representation metadata must be an object")
    size = record.get("size")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
        raise MMH3ResourceError("Representation size must be null or a non-negative integer")
    sha256 = record.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or not _SHA256_RE.match(sha256)):
        raise MMH3ResourceError("Representation sha256 must be null or 64 lowercase hex characters")
    if size is not None and size > MAX_REPRESENTATION_BYTES:
        raise MMH3ResourceError(
            f"Representation {rid} exceeds the {MAX_REPRESENTATION_BYTES}-byte safety limit"
        )


def make_representation_record(
    *,
    target: str,
    kind: str,
    source_revision: str,
    source_digest: str | None,
    media_type: str | None = None,
    path: str | None = None,
    generator: str,
    metadata: Mapping[str, Any] | None = None,
    representation_id: str | None = None,
    created_at: str | None = None,
    size: int | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": representation_id or _representation_id(),
        "target": target,
        "kind": kind,
        "media_type": media_type,
        "path": path,
        "source_revision": source_revision,
        "source_digest": source_digest,
        "created_at": created_at or utc_now_iso(),
        "generator": generator,
        "metadata": deep_copy_json(dict(metadata or {})),
    }
    if size is not None:
        record["size"] = int(size)
    if sha256 is not None:
        record["sha256"] = sha256
    validate_representation_record(record)
    return record


def normalize_representation_index(index: Any, *, packet: "MMH3Media | None" = None) -> dict[str, list[dict[str, Any]]]:
    if index is None:
        return {}
    if not isinstance(index, dict):
        raise MMH3ResourceError("Packet representations must be an object keyed by target id")
    valid_targets: set[str] | None = None
    if packet is not None:
        # Avoid packet.get_by_id() for every representation: on large manifests that turns
        # normalization into O(resources * representations) descriptor scans.
        valid_targets = {str(resource.get("id")) for resource in packet.manifest.get("resources", ())}
        valid_targets.add(PACKET_TARGET)
    out: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for target, records in index.items():
        if not isinstance(target, str) or not target:
            raise MMH3ResourceError("Representation index target keys must be non-empty strings")
        if not isinstance(records, list):
            raise MMH3ResourceError(f"Representations for {target!r} must be an array")
        normalized: list[dict[str, Any]] = []
        for raw in records:
            record = deep_copy_json(raw)
            if record.get("target") != target:
                raise MMH3ResourceError(
                    f"Representation {record.get('id')!r} target does not match index key {target!r}"
                )
            validate_representation_record(record)
            if valid_targets is not None and record["target"] not in valid_targets:
                raise MMH3ResourceError(f"Representation target {record['target']!r} does not exist in packet")
            if record["id"] in seen:
                raise MMH3ResourceError(f"Duplicate representation id {record['id']!r}")
            seen.add(record["id"])
            normalized.append(record)
        normalized.sort(key=lambda item: (item.get("created_at", ""), item["id"]))
        if normalized:
            out[target] = normalized
    return out


def representation_is_fresh(packet: "MMH3Media", record: Mapping[str, Any]) -> bool:
    try:
        validate_representation_record(dict(record), packet=packet)
        current_revision, current_digest = representation_source_identity(packet, str(record["target"]))
    except MMH3ResourceError:
        return False
    if str(record.get("source_revision")) != current_revision:
        return False
    saved_digest = record.get("source_digest")
    if saved_digest is not None and current_digest is not None and saved_digest != current_digest:
        return False
    return True


def list_representations(
    packet: "MMH3Media",
    target: str,
    *,
    kind: str | None = None,
    fresh_only: bool = False,
) -> tuple[dict[str, Any], ...]:
    records = packet.representation_records.get(target, ())
    out: list[dict[str, Any]] = []
    for record in records:
        detached = deep_copy_json(record)
        if kind is not None and detached["kind"] != kind:
            continue
        if fresh_only and not representation_is_fresh(packet, detached):
            continue
        out.append(detached)
    out.sort(key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True)
    return tuple(out)


def best_representation(
    packet: "MMH3Media",
    target: str,
    *,
    kinds: Iterable[str] | None = None,
    fresh_only: bool = True,
) -> dict[str, Any] | None:
    preferred = tuple(kinds or ())
    records = list_representations(packet, target, fresh_only=fresh_only)
    if not records:
        return None
    if preferred:
        rank = {kind: index for index, kind in enumerate(preferred)}
        ranked = [record for record in records if record["kind"] in rank]
        if ranked:
            ranked.sort(key=lambda item: (rank[item["kind"]], item.get("created_at", "")), reverse=False)
            best_rank = min(rank[item["kind"]] for item in ranked)
            candidates = [item for item in ranked if rank[item["kind"]] == best_rank]
            candidates.sort(key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True)
            return candidates[0]
    return records[0]


def canonical_representation_path(record: Mapping[str, Any]) -> str:
    media_type = str(record.get("media_type") or "").lower()
    suffix = {
        "image/png": ".png",
        "image/webp": ".webp",
        "image/jpeg": ".jpg",
        "audio/wav": ".wav",
        "video/mp4": ".mp4",
        "application/json": ".json",
    }.get(media_type, ".bin")
    return safe_member_path(f"representations/{record['id']}{suffix}")


def encode_representation_payload(payload: Any, record: Mapping[str, Any]) -> tuple[bytes, str, dict[str, Any]]:
    """Encode a bounded representation payload without touching source resources."""
    media_type = str(record.get("media_type") or "").strip().lower()
    metadata = deep_copy_json(record.get("metadata") or {})

    if isinstance(payload, torch.Tensor):
        if payload.ndim != 4 or payload.shape[0] != 1 or payload.shape[-1] not in (1, 3, 4):
            raise MMH3ResourceError(
                f"Image representation tensor must be [1,H,W,C], got {getattr(payload, 'shape', None)}"
            )
        arr = payload[0].detach().to(device="cpu", dtype=torch.float32).clamp(0, 1).numpy()
        arr = np.rint(arr * 255.0).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
            mode = "L"
        else:
            mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
        image = Image.fromarray(arr, mode=mode)
        buf = io.BytesIO()
        fmt = "WEBP" if media_type == "image/webp" else "PNG"
        image.save(buf, format=fmt, quality=85 if fmt == "WEBP" else None)
        body = buf.getvalue()
        actual_media = "image/webp" if fmt == "WEBP" else "image/png"
        metadata.setdefault("width", int(payload.shape[2]))
        metadata.setdefault("height", int(payload.shape[1]))
    elif isinstance(payload, (bytes, bytearray, memoryview)):
        body = bytes(payload)
        if not media_type:
            raise MMH3ResourceError("Byte representation payload requires media_type")
        actual_media = media_type
    elif isinstance(payload, (dict, list, str, int, float, bool)) or payload is None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        actual_media = media_type or "application/json"
    else:
        raise MMH3ResourceError(f"Unsupported representation payload type {type(payload).__name__}")

    if len(body) > MAX_REPRESENTATION_BYTES:
        raise MMH3ResourceError(
            f"Representation payload exceeds the {MAX_REPRESENTATION_BYTES}-byte safety limit"
        )
    return body, actual_media, metadata


def decode_image_representation(body: bytes) -> torch.Tensor:
    if len(body) > MAX_REPRESENTATION_BYTES:
        raise MMH3ResourceError("Representation payload exceeds preview safety limit")
    with Image.open(io.BytesIO(body)) as image:
        if image.mode not in ("L", "RGB", "RGBA"):
            image = image.convert("RGB")
        arr = np.asarray(image).copy()
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).to(dtype=torch.float32).div_(255.0).unsqueeze(0)


__all__ = [
    "PACKET_TARGET",
    "REPRESENTATION_KINDS",
    "MAX_REPRESENTATION_BYTES",
    "make_representation_record",
    "validate_representation_record",
    "normalize_representation_index",
    "representation_source_identity",
    "representation_is_fresh",
    "list_representations",
    "best_representation",
    "canonical_representation_path",
    "encode_representation_payload",
    "decode_image_representation",
]
