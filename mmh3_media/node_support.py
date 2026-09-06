from __future__ import annotations

import json
import hashlib
import asyncio
import torch
from pathlib import Path
from fractions import Fraction
from typing import Any

import folder_paths  # type: ignore
from aiohttp import web  # type: ignore
from server import PromptServer  # type: ignore
from comfy_api.latest import ComfyExtension, InputImpl, Types, io, ui  # type: ignore
from typing_extensions import override

from .archive import get_representation_bytes, get_resource_payload, load_archive, save_archive
from .archive_preview import build_archive_card, read_archive_preview
from .constants import FPS, H3_LATENT_ORIGINS, REFERENCE_PRESETS, REFERENCE_PURPOSES
from .core import MMH3Media
from .errors import MMH3Error, MMH3ResourceError
from .h3 import (
    concat_h3_av_latent,
    h3_expected_audio_t,
    nested_parts,
    split_h3_av_latent,
    validate_h3_av_latent,
)
from .inspection import inspect_packet
from .compare import compare_packets
from .export import export_packet
from .preview import build_fastvae_preview, preview_info, preview_is_fresh
from .resolution import INTENTS, MODES, POLICIES, resolve_packet, resolve_reference_set
from .generation_contract import H3_MODEL_FAMILIES
from .continuation import build_h3_continuation_handover
from .decoded_continuation import encode_decoded_prefix, prepare_decoded_prefix
from .chain import commit_chain_segment, start_chain, validate_chain, validate_reroll_source
from .stitch import inspect_stitch_packets, materialize_decoded_segment, stitch_decoded_segments
from .streaming_stitch import build_streaming_stitch, materialize_streaming_segment
from .reference_management import configure_reference
from .comfy_h3_expansion import build_h3_expansion
from .h3_references import materialize_video_reference
from .lora_provenance import (
    clear_generation_loras,
    get_generation_loras,
    lora_provenance_summary,
    set_generation_loras,
)
from .lora_reapply import build_high_sigma_lora_plan, build_lora_reapply_expansion
from .latent_upscale import UPSCALE_GEOMETRY_MODES, build_latent_stitch_upscale_target, build_latent_upscale_process_report, build_latent_upscale_refine_sampling, build_latent_upscale_refine_target, prepare_decoded_packet_latent_upscale, prepare_packet_latent_upscale
from .h3_tile_refine import run_native_h3_tile_refine
from .spatial_tiles import TILE_BLEND_MODES, TILE_CONTEXT_SOURCES, TILE_OVERLAP_MODES, TILE_TRAVERSALS, plan_spatial_tiles
from .tile_backend import TILE_BACKEND_OVERLAP_MODES, finalize_external_tile_video, tile_backend_fingerprint_from_preflight
from .process_result import pack_h3_result, unpack_primary
from .util import deep_copy_json
from .serializers import infer_kind
from .media_metadata import describe_media_payload
from .reference_cache import (
    build_reference_cache_spec,
    compare_reference_cache_latents,
    lookup_reference_cache,
    materialize_reference_cache,
    put_reference_cache,
)
from .runtime_contract import ADD_GUIDE_NODE_ID, TILE_GUIDER_NODE_ID, require_native_h3_contract
from .preflight import PREFLIGHT_OPERATIONS, preflight_packet
from .representations import representation_is_fresh

MMH3 = io.Custom("MMH3_MEDIA")
CATEGORY = "MiniMax H3/MMH3 Media"

_ROLE_OPTIONS = ["auxiliary", "reference", "context", "control", "intermediate"]



def _packet(value: Any) -> MMH3Media:
    if not isinstance(value, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    return value


def _parse_object(text: str, label: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise MMH3ResourceError(f"{label} is not valid JSON: {e}") from e
    if not isinstance(value, dict):
        raise MMH3ResourceError(f"{label} must be a JSON object")
    return value



def _iter_mmh3_files(root: str, prefix: str) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    out: list[str] = []
    try:
        for p in base.rglob("*.mmh3"):
            if p.is_file():
                out.append(f"{prefix}::{p.relative_to(base).as_posix()}")
    except OSError:
        pass
    return out


def _iter_batch_media_files(root: str, prefix: str) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    suffixes = {".mmh3", ".mp4", ".mov", ".mkv", ".webm", ".avi"}
    out: list[str] = []
    try:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.casefold() in suffixes:
                out.append(f"{prefix}::{path.relative_to(base).as_posix()}")
    except OSError:
        pass
    return out


def _file_selector(path: str | Path) -> str | None:
    resolved = Path(path).resolve()
    for area, root in (("input", folder_paths.get_input_directory()), ("output", folder_paths.get_output_directory())):
        try:
            return f"{area}::{resolved.relative_to(Path(root).resolve()).as_posix()}"
        except ValueError:
            continue
    return None


def _load_options() -> list[str]:
    values = _iter_mmh3_files(folder_paths.get_input_directory(), "input")
    values.extend(_iter_mmh3_files(folder_paths.get_output_directory(), "output"))
    values.sort(key=str.casefold)
    # Keep the portable automation placeholder valid even after archives appear.
    # Canonical API workflows deliberately pair it with ``path_override``; if it
    # disappears from the COMBO choices, ComfyUI rejects the prompt before the
    # node can honor that override.
    return ["(none)", *values]


def _batch_file_options():
    values = _iter_batch_media_files(folder_paths.get_input_directory(), "input")
    values.extend(_iter_batch_media_files(folder_paths.get_output_directory(), "output"))
    return sorted(set(values), key=str.casefold)


@PromptServer.instance.routes.get("/mmh3_media/batch_files")
async def _mmh3_media_batch_files_route(_request):
    async with _preview_slots:
        values = await asyncio.to_thread(_batch_file_options)
    return web.json_response(values)


@PromptServer.instance.routes.get("/mmh3_media/file_info")
async def _mmh3_media_file_info_route(request):
    """Return a bounded manifest-only card for one safe input/output selector value."""
    try:
        selector = str(request.query.get("file") or "")
        path = _resolve_load_path(selector, "")
        async with _preview_slots:
            card = await asyncio.to_thread(build_archive_card, path)
        return web.json_response(card)
    except MMH3Error as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except OSError:
        return web.json_response({"error": "The selected MMH3 archive could not be read"}, status=400)


_preview_slots = asyncio.Semaphore(2)


@PromptServer.instance.routes.get("/mmh3_media/file_preview")
async def _mmh3_media_file_preview_route(request):
    """Serve cached images or CPU source thumbnails off the server's event loop."""
    try:
        selector = str(request.query.get("file") or "")
        path = _resolve_load_path(selector, "")
        async with _preview_slots:
            body, media_type, _resource_id = await asyncio.to_thread(
                read_archive_preview, path, request.query.get("representation_id"), request.query.get("resource_id")
            )
        return web.Response(
            body=body,
            content_type=media_type,
            headers={"Cache-Control": "private, max-age=60"},
        )
    except MMH3Error as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except (OSError, ValueError, KeyError):
        return web.json_response({"error": "The selected MMH3 preview could not be read"}, status=404)


@PromptServer.instance.routes.get("/mmh3_media/file_resource_info")
async def _mmh3_media_file_resource_info_route(request):
    """Return descriptor-first preview info for one archived resource, without payload I/O."""
    try:
        selector = str(request.query.get("file") or "")
        resource_id = str(request.query.get("resource_id") or "").strip()
        if not resource_id:
            raise MMH3ResourceError("resource_id is required")
        path = _resolve_load_path(selector, "")
        async with _preview_slots:
            info = await asyncio.to_thread(_archive_resource_info, path, resource_id)
        return web.json_response(info)
    except MMH3Error as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except OSError:
        return web.json_response({"error": "The selected MMH3 archive could not be read"}, status=400)


@PromptServer.instance.routes.get("/mmh3_media/file_representation")
async def _mmh3_media_file_representation_route(request):
    """Serve one explicitly selected fresh cached representation; never execute the graph."""
    try:
        selector = str(request.query.get("file") or "")
        representation_id = str(request.query.get("representation_id") or "").strip()
        if not representation_id:
            raise MMH3ResourceError("representation_id is required")
        path = _resolve_load_path(selector, "")
        async with _preview_slots:
            body, media_type = await asyncio.to_thread(_archive_representation, path, representation_id)
        return web.Response(
            body=body,
            content_type=media_type,
            headers={"Cache-Control": "private, max-age=60"},
        )
    except MMH3Error as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except OSError:
        return web.json_response({"error": "The selected MMH3 representation could not be read"}, status=404)


def _archive_resource_info(path, resource_id):
    return preview_info(load_archive(path, verify="manifest"), target=resource_id)


def _archive_representation(path, representation_id):
    packet = load_archive(path, verify="manifest")
    record = packet.get_representation(representation_id)
    if record is None or not representation_is_fresh(packet, record):
        raise MMH3ResourceError("Representation is missing or stale")
    return get_representation_bytes(packet, record), str(record.get("media_type") or "application/octet-stream")


def _resolve_load_path(file: str, path_override: str) -> Path:
    if path_override and path_override.strip():
        return Path(path_override).expanduser().resolve()
    if not file or file == "(none)":
        raise MMH3ResourceError("No .mmh3 file selected")
    try:
        area, rel = file.split("::", 1)
    except ValueError as e:
        raise MMH3ResourceError(f"Invalid MMH3 file selector: {file!r}") from e
    root = {
        "input": Path(folder_paths.get_input_directory()),
        "output": Path(folder_paths.get_output_directory()),
    }.get(area)
    if root is None:
        raise MMH3ResourceError(f"Unknown MMH3 file area {area!r}")
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as e:
        raise MMH3ResourceError("MMH3 file selector escapes its ComfyUI directory") from e
    return candidate


def _safe_output_prefix(prefix: str) -> Path:
    raw = (prefix or "mmh3/MMH3").strip().replace("\\", "/")
    p = Path(raw)
    if p.is_absolute() or any(part in ("", ".", "..") or ":" in part for part in p.parts):
        raise MMH3ResourceError("filename_prefix must be a safe portable relative path without '..' or ':'")
    if p.suffix.lower() == ".mmh3":
        p = p.with_suffix("")
    return p


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    for i in range(1, 100000):
        candidate = path.with_name(f"{stem}_{i:05d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise MMH3ResourceError(f"Could not allocate a unique filename near {path}")


def _resolve_save_path(packet: MMH3Media, filename_prefix: str, target: str, overwrite: bool) -> Path:
    if target == "source (in-place)":
        if not packet.source_archive:
            raise MMH3ResourceError("In-place save requires a packet loaded/saved from a .mmh3 file")
        if not overwrite:
            raise MMH3ResourceError("Enable overwrite to replace the source archive in place")
        return Path(packet.source_archive).resolve()
    rel = _safe_output_prefix(filename_prefix)
    path = (Path(folder_paths.get_output_directory()) / rel.parent / (rel.name + ".mmh3")).resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    try:
        path.relative_to(output_root)
    except ValueError as e:
        raise MMH3ResourceError("Save path escapes the ComfyUI output directory") from e
    if path.exists() and not overwrite:
        path = _unique_path(path)
    return path
