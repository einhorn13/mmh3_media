from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .core import MMH3Media
from .errors import MMH3FormatError, MMH3IntegrityError, MMH3ResourceError
from .schema import validate_verify_mode
from .validation import require_valid_manifest
from .serializers import (
    canonical_archive_path,
    deserialize_payload,
    serialize_payload,
    video_source_path,
)
from .util import copy_and_hash, get_cache_root, safe_member_path, sha256_file, utc_now_iso

PACKET_JSON = "packet.json"
MAX_PACKET_JSON = 16 * 1024 * 1024


def _zip_compression_for(res: dict[str, Any] | None) -> int:
    # PNG/video/safetensors/WAV are already compressed or binary; deflate wastes CPU and can expand memory pressure.
    return zipfile.ZIP_DEFLATED if res is None else zipfile.ZIP_STORED


def _read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    try:
        info = zf.getinfo(PACKET_JSON)
    except KeyError as e:
        raise MMH3FormatError("Archive is missing packet.json") from e
    if info.file_size > MAX_PACKET_JSON:
        raise MMH3FormatError(f"packet.json is unreasonably large ({info.file_size} bytes)")
    try:
        raw = zf.read(info)
        manifest = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise MMH3FormatError("packet.json is not valid UTF-8 JSON") from e
    require_valid_manifest(manifest)
    return manifest


def _validate_zip_members(zf: zipfile.ZipFile, manifest: dict[str, Any]) -> None:
    listed = zf.namelist()
    if len(listed) != len(set(listed)):
        raise MMH3FormatError("Archive contains duplicate member names")
    for info in zf.infolist():
        if info.is_dir():
            continue
        safe_member_path(info.filename)
    names = set(listed)
    for res in manifest["resources"]:
        path = safe_member_path(res["path"])
        if path not in names:
            raise MMH3FormatError(f"Resource {res['id']} points to missing archive member {path!r}")
        info = zf.getinfo(path)
        if info.is_dir():
            raise MMH3FormatError(f"Resource {res['id']} points to a directory, not a file")
        declared_size = res.get("size")
        if declared_size is not None and int(declared_size) != int(info.file_size):
            raise MMH3IntegrityError(
                f"Resource {res['id']} size mismatch: manifest={declared_size}, archive={info.file_size}"
            )


def _hash_member(zf: zipfile.ZipFile, path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with zf.open(path, "r") as src:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def load_archive(path: str | os.PathLike, verify: str = "on_access") -> MMH3Media:
    validate_verify_mode(verify)
    archive_path = Path(path).expanduser().resolve()
    if not archive_path.is_file():
        raise MMH3FormatError(f"MMH3 file does not exist: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise MMH3FormatError(f"Not a ZIP/MMH3 archive: {archive_path}")
    with zipfile.ZipFile(archive_path, "r", allowZip64=True) as zf:
        manifest = _read_manifest(zf)
        _validate_zip_members(zf, manifest)
        if verify == "full":
            for res in manifest["resources"]:
                if res.get("role") == "preview":
                    continue
                expected = res.get("sha256")
                actual, size = _hash_member(zf, res["path"])
                if expected is not None and actual != expected:
                    raise MMH3IntegrityError(
                        f"Checksum mismatch for {res['role']}[{res['slot']}] ({res['id']}): expected {expected}, got {actual}"
                    )
                if res.get("size") is not None and int(res["size"]) != size:
                    raise MMH3IntegrityError(f"Size mismatch for resource {res['id']}")
    origins = {res["id"]: res["path"] for res in manifest["resources"]}
    return MMH3Media(
        manifest=manifest,
        source_archive=str(archive_path),
        payloads={},
        origins=origins,
        verify_mode=verify,
        dirty=False,
    )


def _cache_path_for(packet: MMH3Media, res: dict[str, Any]) -> Path:
    suffix = Path(res["path"]).suffix or ".bin"
    token = res.get("sha256") or f"{packet.manifest['id']}_{res['id']}"
    return get_cache_root() / f"{token}{suffix}"


def materialize_resource_file(packet: MMH3Media, res: dict[str, Any]) -> Path:
    rid = res["id"]
    if rid in packet.payloads:
        raise MMH3ResourceError("In-memory resources are materialized by the serializer, not archive extraction")
    if not packet.source_archive:
        raise MMH3ResourceError(f"Resource {rid} has no in-memory payload and no source archive")
    origin = packet.origins.get(rid, res["path"])
    origin = safe_member_path(origin)
    target = _cache_path_for(packet, res)
    expected_size = res.get("size")
    if target.is_file() and (expected_size is None or target.stat().st_size == expected_size):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    h = hashlib.sha256()
    size = 0
    try:
        with zipfile.ZipFile(packet.source_archive, "r", allowZip64=True) as zf:
            try:
                src = zf.open(origin, "r")
            except KeyError as e:
                raise MMH3FormatError(f"Archive no longer contains resource member {origin!r}") from e
            with src, open(tmp, "wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
        if expected_size is not None and size != expected_size:
            raise MMH3IntegrityError(f"Resource {rid} size mismatch on access: expected {expected_size}, got {size}")
        expected_hash = res.get("sha256")
        if packet.verify_mode in ("on_access", "full") and expected_hash is not None and h.hexdigest() != expected_hash:
            raise MMH3IntegrityError(
                f"Resource {rid} checksum mismatch on access: expected {expected_hash}, got {h.hexdigest()}"
            )
        os.replace(tmp, target)
        return target
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def get_resource_payload(packet: MMH3Media, res: dict[str, Any]) -> Any:
    rid = res["id"]
    if rid in packet.payloads:
        payload = packet.payloads[rid]
        # Validate H3 now too; a mutable upstream tensor/dict may have changed since Put.
        if res["role"] == "h3_av_latent":
            from .h3 import validate_h3_av_latent

            validate_h3_av_latent(payload)
        return payload
    path = materialize_resource_file(packet, res)
    if res["kind"] == "video":
        # VideoFromFile may retain and lazily reopen the path, so keep it for the process session.
        return deserialize_payload(path, res)
    try:
        return deserialize_payload(path, res)
    finally:
        # Tensor/image/audio/json payloads are fully loaded by deserialize_payload; the extracted
        # member is no longer needed and must not accumulate in the temp cache.
        path.unlink(missing_ok=True)


def _staged_resource(
    packet: MMH3Media,
    res: dict[str, Any],
    stage_dir: Path,
) -> tuple[Path | None, Path | None, str | None, dict[str, Any] | None]:
    """Return (direct_source, staged_file, video_suffix, serializer_result)."""
    rid = res["id"]
    payload = packet.payloads[rid]
    if res["kind"] == "video":
        source, suffix = video_source_path(payload)
        if source is not None:
            return source, None, suffix, {
                "serializer": "source.video.bytecopy.v1",
                "media_type": None,
                "metadata_patch": {"source_byte_copy": True},
            }
    suffix = ".mp4" if res["kind"] == "video" else Path(canonical_archive_path(res)).suffix
    staged = stage_dir / f"{rid}{suffix}"
    result = serialize_payload(payload, res, staged)
    return None, staged, suffix if res["kind"] == "video" else None, result


def _merge_resource_metadata(res: dict[str, Any], patch: dict[str, Any]) -> None:
    md = copy.deepcopy(res.get("metadata", {}))
    md.update(copy.deepcopy(patch))
    res["metadata"] = md


def save_archive(packet: MMH3Media, path: str | os.PathLike) -> tuple[MMH3Media, str]:
    final_path = Path(path).expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.suffix.lower() != ".mmh3":
        final_path = final_path.with_suffix(".mmh3")
    tmp_path = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")
    stage_dir = Path(tempfile.mkdtemp(prefix="mmh3_stage_"))
    manifest = copy.deepcopy(packet.manifest)
    source_zf: zipfile.ZipFile | None = None
    try:
        if packet.source_archive:
            source_zf = zipfile.ZipFile(packet.source_archive, "r", allowZip64=True)

        # Stage changed resources first because serializer metadata/path may affect packet.json.
        staged: dict[str, tuple[Path | None, Path | None, str | None, dict[str, Any] | None]] = {}
        for res in manifest["resources"]:
            rid = res["id"]
            if rid in packet.payloads:
                staged[rid] = _staged_resource(packet, res, stage_dir)
                direct, staged_file, video_suffix, serializer_result = staged[rid]
                res["path"] = canonical_archive_path(res, video_suffix=video_suffix)
                if serializer_result:
                    if serializer_result.get("serializer"):
                        res["serializer"] = serializer_result["serializer"]
                    if serializer_result.get("media_type"):
                        res["media_type"] = serializer_result["media_type"]
                    _merge_resource_metadata(res, serializer_result.get("metadata_patch", {}))
            else:
                # Keep a readable canonical path after Move while remembering the old member through packet.origins.
                suffix = Path(packet.origins.get(rid, res["path"])).suffix if res["kind"] == "video" else None
                if res["kind"] in {"latent", "image", "video", "audio", "mask", "json"}:
                    res["path"] = canonical_archive_path(res, video_suffix=suffix or None)

        seen_paths: set[str] = set()
        for res in manifest["resources"]:
            res["path"] = safe_member_path(res["path"])
            if res["path"] in seen_paths:
                # Should be impossible with valid role+slot, but fail before writing ambiguous ZIPs.
                raise MMH3FormatError(f"Two resources resolve to the same archive path {res['path']!r}")
            seen_paths.add(res["path"])

        with zipfile.ZipFile(tmp_path, "w", allowZip64=True) as out_zf:
            for res in manifest["resources"]:
                rid = res["id"]
                dest_path = res["path"]
                h = hashlib.sha256()
                size = 0
                if rid in staged:
                    direct, staged_file, _, _ = staged[rid]
                    src_path = direct or staged_file
                    if src_path is None:
                        raise MMH3FormatError(f"Internal error: no staged source for {rid}")
                    with open(src_path, "rb") as src, out_zf.open(
                        dest_path, "w", force_zip64=True
                    ) as dst:
                        hhex, size = copy_and_hash(src, dst)
                    if res.get("role") == "preview":
                        res.pop("sha256", None)
                    else:
                        res["sha256"] = hhex
                    res["size"] = size
                else:
                    if source_zf is None:
                        raise MMH3FormatError(f"Resource {rid} has no payload and no source archive")
                    origin = safe_member_path(packet.origins.get(rid, res.get("path", "")))
                    try:
                        src = source_zf.open(origin, "r")
                    except KeyError as e:
                        raise MMH3FormatError(f"Source archive is missing unchanged resource {origin!r}") from e
                    with src, out_zf.open(dest_path, "w", force_zip64=True) as dst:
                        hhex, size = copy_and_hash(src, dst)
                    expected = res.get("sha256")
                    if res.get("role") != "preview":
                        if packet.verify_mode in ("on_access", "full") and expected is not None and hhex != expected:
                            raise MMH3IntegrityError(
                                f"Checksum mismatch while copying unchanged resource {rid}: expected {expected}, got {hhex}"
                            )
                        res["sha256"] = hhex
                    else:
                        res.pop("sha256", None)
                    res["size"] = size

            # Preview is cache: bind it to the final source checksum but never checksum the preview itself.
            source_res = next((r for r in manifest["resources"] if r.get("role") == "h3_av_latent"), None)
            preview_res = next((r for r in manifest["resources"] if r.get("role") == "preview"), None)
            if source_res is not None and preview_res is not None:
                cache = preview_res.setdefault("metadata", {}).get("cache")
                if isinstance(cache, dict) and cache.get("type") == "preview":
                    cache["source_resource_id"] = source_res["id"]
                    cache["source_sha256"] = source_res.get("sha256")
                    cache["integrity"] = "excluded"
                preview_res.pop("sha256", None)

            manifest["updated_at"] = utc_now_iso()
            require_valid_manifest(manifest)
            packet_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")
            if len(packet_bytes) > MAX_PACKET_JSON:
                raise MMH3FormatError("packet.json exceeds the 16 MiB safety limit")
            info = zipfile.ZipInfo(PACKET_JSON)
            info.compress_type = zipfile.ZIP_DEFLATED
            out_zf.writestr(info, packet_bytes)

        if source_zf is not None:
            source_zf.close()
            source_zf = None

        # Validate the completed temporary archive before the atomic replace.
        with zipfile.ZipFile(tmp_path, "r", allowZip64=True) as check:
            check_manifest = _read_manifest(check)
            _validate_zip_members(check, check_manifest)
        os.replace(tmp_path, final_path)
        saved = packet.with_saved_state(manifest=manifest, source_archive=str(final_path))
        return saved, str(final_path)
    finally:
        if source_zf is not None:
            source_zf.close()
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        shutil.rmtree(stage_dir, ignore_errors=True)
