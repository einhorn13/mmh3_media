from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .core import MMH3Media
from .errors import MMH3FormatError, MMH3IntegrityError, MMH3ResourceError
from .schema import validate_archive_manifest, validate_verify_mode
from .validation import require_valid_manifest
from .serializers import (
    canonical_v03_archive_path,
    deserialize_payload,
    serialize_payload,
    video_source_path,
)
from .util import copy_and_hash, get_cache_root, safe_member_path, sha256_file, utc_now_iso
from .representations import (
    MAX_REPRESENTATION_BYTES,
    PACKET_TARGET,
    canonical_representation_path,
    encode_representation_payload,
    validate_representation_record,
)

PACKET_JSON = "packet.json"
MAX_PACKET_JSON = 16 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)


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
    validate_archive_manifest(manifest)
    return manifest


def _iter_representation_records(manifest: dict[str, Any]):
    index = manifest.get("representations", {})
    if index is None:
        return
    if not isinstance(index, dict):
        raise MMH3FormatError("Manifest representations must be an object")
    for target, records in index.items():
        if not isinstance(records, list):
            raise MMH3FormatError(f"Representations for {target!r} must be an array")
        for record in records:
            try:
                validate_representation_record(record)
            except MMH3ResourceError as exc:
                raise MMH3FormatError(str(exc)) from exc
            if record.get("target") != target:
                raise MMH3FormatError(
                    f"Representation {record.get('id')!r} target does not match index key {target!r}"
                )
            yield record


def _resource_declared_size(res: dict[str, Any]) -> int | None:
    content = res.get("content")
    return content.get("size") if isinstance(content, dict) else None


def _resource_expected_sha256(res: dict[str, Any]) -> str | None:
    content = res.get("content")
    digest = content.get("digest") if isinstance(content, dict) else None
    return digest[7:] if isinstance(digest, str) and digest.startswith("sha256:") else None


def _resource_debug_label(res: dict[str, Any]) -> str:
    order = res.get("order")
    suffix = f" order={order}" if order is not None else ""
    return f"{res.get('role')}/{res.get('kind')}{suffix} ({res.get('id')})"


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
        declared_size = _resource_declared_size(res)
        if declared_size is not None and int(declared_size) != int(info.file_size):
            raise MMH3IntegrityError(
                f"Resource {res['id']} size mismatch: manifest={declared_size}, archive={info.file_size}"
            )
    resource_ids = {res["id"] for res in manifest["resources"]}
    resource_paths = {safe_member_path(res["path"]) for res in manifest["resources"]}
    representation_ids: set[str] = set()
    representation_paths: set[str] = set()
    for record in _iter_representation_records(manifest):
        rep_id = record["id"]
        if rep_id in representation_ids:
            raise MMH3FormatError(f"Duplicate representation id {rep_id!r}")
        representation_ids.add(rep_id)
        target = record["target"]
        if target != PACKET_TARGET and target not in resource_ids:
            raise MMH3FormatError(
                f"Representation {rep_id} targets missing resource {target!r}"
            )
        path = record.get("path")
        if path is None:
            continue
        path = safe_member_path(path)
        if path in representation_paths or path in resource_paths:
            raise MMH3FormatError(f"Duplicate representation archive path {path!r}")
        representation_paths.add(path)
        if path not in names:
            raise MMH3FormatError(
                f"Representation {record['id']} points to missing archive member {path!r}"
            )
        info = zf.getinfo(path)
        if info.is_dir():
            raise MMH3FormatError(f"Representation {record['id']} points to a directory")
        if info.file_size > MAX_REPRESENTATION_BYTES:
            raise MMH3FormatError(
                f"Representation {record['id']} exceeds preview safety limit ({info.file_size} bytes)"
            )
        declared_size = record.get("size")
        if declared_size is not None and int(declared_size) != int(info.file_size):
            raise MMH3IntegrityError(
                f"Representation {record['id']} size mismatch: manifest={declared_size}, archive={info.file_size}"
            )


def _hash_member(zf: zipfile.ZipFile, path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    try:
        with zf.open(path, "r") as src:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                size += len(chunk)
    except zipfile.BadZipFile as exc:
        raise MMH3IntegrityError(f"Corrupt ZIP member {path!r}: {exc}") from exc
    return h.hexdigest(), size


def load_archive(path: str | os.PathLike, verify: str = "on_access") -> MMH3Media:
    validate_verify_mode(verify)
    archive_path = Path(path).expanduser().resolve()
    if not archive_path.is_file():
        raise MMH3FormatError(f"MMH3 file does not exist: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise MMH3FormatError(f"Not a ZIP/MMH3 archive: {archive_path}")
    with zipfile.ZipFile(archive_path, "r", allowZip64=True) as zf:
        archive_manifest = _read_manifest(zf)
        _validate_zip_members(zf, archive_manifest)
        if verify == "full":
            for res in archive_manifest["resources"]:
                expected = _resource_expected_sha256(res)
                actual, size = _hash_member(zf, res["path"])
                if expected is not None and actual != expected:
                    raise MMH3IntegrityError(
                        f"Checksum mismatch for {_resource_debug_label(res)}: expected {expected}, got {actual}"
                    )
                declared_size = _resource_declared_size(res)
                if declared_size is not None and int(declared_size) != size:
                    raise MMH3IntegrityError(f"Size mismatch for resource {res['id']}")
            for record in _iter_representation_records(archive_manifest):
                if not record.get("path"):
                    continue
                actual, size = _hash_member(zf, record["path"])
                expected = record.get("sha256")
                if expected is not None and actual != expected:
                    raise MMH3IntegrityError(
                        f"Checksum mismatch for representation {record['id']}: expected {expected}, got {actual}"
                    )
                if record.get("size") is not None and int(record["size"]) != size:
                    raise MMH3IntegrityError(f"Size mismatch for representation {record['id']}")
    manifest = archive_manifest
    require_valid_manifest(manifest)
    origins = {res["id"]: res["path"] for res in manifest["resources"]}
    representation_origins = {
        record["id"]: record["path"]
        for record in _iter_representation_records(archive_manifest)
        if isinstance(record.get("path"), str) and record.get("path")
    }
    return MMH3Media(
        manifest=manifest,
        source_archive=str(archive_path),
        payloads={},
        origins=origins,
        verify_mode=verify,
        dirty=False,
        representation_origins=representation_origins,
    )


def _cache_path_for(packet: MMH3Media, res: dict[str, Any]) -> Path:
    suffix = Path(res["path"]).suffix or ".bin"
    token = _resource_expected_sha256(res) or f"{packet.manifest['id']}_{res['id']}"
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
    expected_size = _resource_declared_size(res)
    expected_hash = _resource_expected_sha256(res)
    if target.is_file() and (expected_size is None or target.stat().st_size == expected_size):
        if packet.verify_mode not in ("on_access", "full") or expected_hash is None:
            return target
        cached_hash, cached_size = sha256_file(target)
        if (expected_size is None or cached_size == expected_size) and cached_hash == expected_hash:
            return target
        # A same-sized stale/corrupt cache entry must never bypass on-access verification.
        # Remove it and recover from the archive below; archive bytes are verified while copying.
        target.unlink(missing_ok=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    h = hashlib.sha256()
    size = 0
    try:
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
        except zipfile.BadZipFile as exc:
            raise MMH3IntegrityError(f"Corrupt ZIP member {origin!r}: {exc}") from exc
        if expected_size is not None and size != expected_size:
            raise MMH3IntegrityError(f"Resource {rid} size mismatch on access: expected {expected_size}, got {size}")
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
        h3 = res.get("extensions", {}).get("minimax_h3", {}) if isinstance(res.get("extensions"), dict) else {}
        if res["kind"] == "latent" and isinstance(h3, dict) and isinstance(h3.get("latent"), dict):
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
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            # An external reader (e.g. on SMB) can still hold the cache file.
            # Session/stale-cache cleanup will retry; never mask a load error
            # or discard a successful payload because optional cleanup failed.
            _LOGGER.warning("Could not remove extracted MMH3 cache file %s; deferring cleanup: %s", path, exc)


def get_representation_bytes(packet: MMH3Media, record: dict[str, Any]) -> bytes:
    """Return one bounded representation payload without materializing its source."""
    validate_representation_record(record, packet=packet)
    rep_id = record["id"]
    if rep_id in packet.representation_payloads:
        body, _media_type, _metadata = encode_representation_payload(
            packet.representation_payloads[rep_id], record
        )
        return body
    if not packet.source_archive:
        raise MMH3ResourceError(f"Representation {rep_id} has no payload and no source archive")
    origin = packet.representation_origins.get(rep_id) or record.get("path")
    if not isinstance(origin, str) or not origin:
        raise MMH3ResourceError(f"Representation {rep_id} has no stored archive path")
    origin = safe_member_path(origin)
    with zipfile.ZipFile(packet.source_archive, "r", allowZip64=True) as zf:
        try:
            info = zf.getinfo(origin)
        except KeyError as exc:
            raise MMH3FormatError(f"Archive no longer contains representation {origin!r}") from exc
        if info.file_size > MAX_REPRESENTATION_BYTES:
            raise MMH3ResourceError(
                f"Representation {rep_id} exceeds preview safety limit ({info.file_size} bytes)"
            )
        expected_size = record.get("size")
        if expected_size is not None and int(expected_size) != int(info.file_size):
            raise MMH3IntegrityError(
                f"Representation {rep_id} size mismatch on access: expected {expected_size}, got {info.file_size}"
            )
        try:
            body = zf.read(info)
        except zipfile.BadZipFile as exc:
            raise MMH3IntegrityError(f"Corrupt representation ZIP member {origin!r}: {exc}") from exc
    expected = record.get("sha256")
    if packet.verify_mode in ("on_access", "full") and expected is not None:
        actual = hashlib.sha256(body).hexdigest()
        if actual != expected:
            raise MMH3IntegrityError(
                f"Representation {rep_id} checksum mismatch: expected {expected}, got {actual}"
            )
    return body


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
                "descriptor_patch": {"source_byte_copy": True},
            }
    suffix = ".mp4" if res["kind"] == "video" else Path(canonical_v03_archive_path(res)).suffix
    staged = stage_dir / f"{rid}{suffix}"
    result = serialize_payload(payload, res, staged)
    return None, staged, suffix if res["kind"] == "video" else None, result


def _merge_serialization_descriptor(res: dict[str, Any], patch: dict[str, Any]) -> None:
    descriptor = copy.deepcopy(res.get("descriptor", {}))
    storage = descriptor.get("serialization")
    if not isinstance(storage, dict):
        storage = {}
    storage.update(copy.deepcopy(patch))
    descriptor["serialization"] = storage
    res["descriptor"] = descriptor


def _durable_replace(tmp_path: Path, final_path: Path) -> None:
    """Persist a completed archive before and after the atomic rename when supported."""
    # Windows implements fsync via the writable-handle-only CRT _commit call.
    # The temporary archive is complete and owned by us, so opening it read/write
    # preserves the same durability contract on both Windows and POSIX.
    with tmp_path.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp_path, final_path)

    # On POSIX, syncing the parent directory makes the rename durable across a crash.
    # Some filesystems/platforms do not allow directory fsync; the archive is already
    # atomically replaced at that point, so treat that durability enhancement as best-effort.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(str(final_path.parent), flags)
    except OSError:
        return
    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
    finally:
        os.close(directory_fd)


def save_archive(packet: MMH3Media, path: str | os.PathLike) -> tuple[MMH3Media, str]:
    from .save_previews import cache_materialized_previews, cache_file_preview

    packet = cache_materialized_previews(packet)
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
                packet = cache_file_preview(packet, rid, direct or staged_file, res["kind"])
                res["path"] = canonical_v03_archive_path(res, video_suffix=video_suffix)
                if serializer_result:
                    if serializer_result.get("serializer"):
                        res["serializer"] = serializer_result["serializer"]
                    if serializer_result.get("media_type"):
                        res["media_type"] = serializer_result["media_type"]
                    _merge_serialization_descriptor(res, serializer_result.get("descriptor_patch", {}))
            else:
                # Keep a readable canonical path after Move while remembering the old member through packet.origins.
                suffix = Path(packet.origins.get(rid, res["path"])).suffix if res["kind"] == "video" else None
                if res["kind"] in {"latent", "image", "video", "audio", "mask", "json"}:
                    res["path"] = canonical_v03_archive_path(res, video_suffix=suffix or None)

        manifest["representations"] = copy.deepcopy(packet.manifest.get("representations", {}))
        seen_paths: set[str] = set()
        for res in manifest["resources"]:
            res["path"] = safe_member_path(res["path"])
            if res["path"] in seen_paths:
                # Should be impossible with stable resource IDs, but fail before writing ambiguous ZIPs.
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
                    res.setdefault("content", {})["digest"] = "sha256:" + hhex
                    res["content"]["size"] = size
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
                    expected = _resource_expected_sha256(res)
                    if packet.verify_mode in ("on_access", "full") and expected is not None and hhex != expected:
                        raise MMH3IntegrityError(
                            f"Checksum mismatch while copying unchanged resource {rid}: expected {expected}, got {hhex}"
                        )
                    res.setdefault("content", {})["digest"] = "sha256:" + hhex
                    res["content"]["size"] = size

            # Representations are derived cache artifacts outside resources[].  Bind their
            # freshness token to the final source digest when the source is a resource.
            resource_by_id = {res["id"]: res for res in manifest["resources"]}
            for record in _iter_representation_records(manifest):
                target = record["target"]
                if target in resource_by_id:
                    source = resource_by_id[target]
                    content = source.get("content", {}) if isinstance(source.get("content"), dict) else {}
                    if record.get("source_revision") == content.get("revision") and content.get("digest"):
                        record["source_digest"] = content["digest"]

                rep_id = record["id"]
                if rep_id in packet.representation_payloads:
                    body, media_type, metadata = encode_representation_payload(
                        packet.representation_payloads[rep_id], record
                    )
                    record["media_type"] = media_type
                    record["metadata"] = metadata
                    record["path"] = canonical_representation_path(record)
                    record["size"] = len(body)
                    record["sha256"] = hashlib.sha256(body).hexdigest()
                    out_zf.writestr(record["path"], body, compress_type=zipfile.ZIP_STORED)
                elif record.get("path"):
                    if source_zf is None:
                        raise MMH3FormatError(
                            f"Representation {rep_id} has no in-memory payload and no source archive"
                        )
                    origin = safe_member_path(
                        packet.representation_origins.get(rep_id, record["path"])
                    )
                    try:
                        src = source_zf.open(origin, "r")
                    except KeyError as exc:
                        raise MMH3FormatError(
                            f"Source archive is missing unchanged representation {origin!r}"
                        ) from exc
                    dest_path = safe_member_path(record["path"])
                    with src, out_zf.open(dest_path, "w", force_zip64=True) as dst:
                        hhex, size = copy_and_hash(src, dst)
                    if size > MAX_REPRESENTATION_BYTES:
                        raise MMH3FormatError(
                            f"Representation {rep_id} exceeds preview safety limit ({size} bytes)"
                        )
                    expected = record.get("sha256")
                    if packet.verify_mode in ("on_access", "full") and expected is not None and hhex != expected:
                        raise MMH3IntegrityError(
                            f"Checksum mismatch while copying representation {rep_id}: expected {expected}, got {hhex}"
                        )
                    record["sha256"] = hhex
                    record["size"] = size

            manifest["updated_at"] = utc_now_iso()
            require_valid_manifest(manifest)
            validate_archive_manifest(manifest)
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
        _durable_replace(tmp_path, final_path)
        saved = packet.with_saved_state(manifest=manifest, source_archive=str(final_path))
        return saved, str(final_path)
    finally:
        if source_zf is not None:
            source_zf.close()
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        shutil.rmtree(stage_dir, ignore_errors=True)
