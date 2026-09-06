from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .core import MMH3Media
from .h3 import validate_h3_av_latent
from .h3_contract import h3_latent_contract_from_resource
from .h3_resource_semantics import find_context_resource
from .lora_provenance import get_generation_loras
from .resource_model import resource_facts


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _context_dimensions(packet: MMH3Media, usage: str) -> tuple[int | None, int | None]:
    resource = find_context_resource(packet, usage)
    if resource is None:
        return None, None
    facts = resource_facts(resource)
    width = _int_or_none(facts.get("width"))
    height = _int_or_none(facts.get("height"))
    if width is not None and height is not None:
        return width, height
    shape = facts.get("shape")
    if isinstance(shape, list) and len(shape) >= 3:
        return _int_or_none(shape[-2]), _int_or_none(shape[-3])
    return None, None


def _h3_contract(packet: MMH3Media, warnings: list[str]) -> dict[str, Any] | None:
    ref = packet.primary("latent")
    if ref is None:
        return None
    try:
        contract = h3_latent_contract_from_resource(ref.descriptor)
    except Exception as exc:
        warnings.append(f"H3 latent contract: {exc}")
        return None
    if ref.resource_id in packet.payloads:
        try:
            info = validate_h3_av_latent(packet.payloads[ref.resource_id], strict_audio_length=False)
            warnings.extend(info.warnings)
            canvas = contract["canvas"]
            timeline = contract["timeline"]
            if (info.width, info.height) != (canvas["width"], canvas["height"]):
                warnings.append("H3 latent payload geometry differs from its canonical contract")
            if info.frames is not None and info.frames != timeline.get("frames"):
                warnings.append("H3 latent payload frame count differs from its canonical contract")
        except Exception as exc:
            warnings.append(f"H3 AV latent validation: {exc}")
    return contract


def _contract_value(contract: Mapping[str, Any] | None, key: str) -> Any:
    if not contract:
        return None
    if key in ("width", "height"):
        return contract.get("canvas", {}).get(key)
    if key in ("frames", "fps"):
        return contract.get("timeline", {}).get(key)
    return None


def inspect_packet(packet: MMH3Media) -> dict[str, Any]:
    generation = packet.manifest.get("generation", {})
    notes = packet.manifest.get("notes", "")
    warnings: list[str] = []
    h3_contract = _h3_contract(packet, warnings)
    loras = get_generation_loras(packet)

    video = packet.get_primary("video")
    video_facts = resource_facts(video) if video else {}

    # Ref2VA workflows (notably F18 long-video lipsync/audio-driven) keep the
    # source clip as a reference resource rather than a primary decoded video.
    # When there is exactly one reference video, it is an unambiguous source
    # for auto geometry/timeline resolution. Multi-reference packets remain
    # fail-closed instead of silently picking an arbitrary reference.
    if not video_facts:
        reference_videos = [
            resource for resource in packet.resources()
            if resource.get("role") == "reference" and resource.get("kind") == "video"
        ]
        if len(reference_videos) == 1:
            video_facts = resource_facts(reference_videos[0])

    first_width, first_height = _context_dimensions(packet, "first_frame")
    last_width, last_height = _context_dimensions(packet, "last_frame")
    if None not in (first_width, first_height, last_width, last_height) and (first_width, first_height) != (last_width, last_height):
        warnings.append(
            f"first_frame geometry {first_width}x{first_height} differs from last_frame "
            f"{last_width}x{last_height}; first_frame defines auto geometry"
        )
    image_width = first_width if first_width is not None else last_width
    image_height = first_height if first_height is not None else last_height

    def choose_int(key: str, video_value: Any = None, image_value: Any = None) -> int | None:
        generated = _int_or_none(generation.get(key))
        latent = _int_or_none(_contract_value(h3_contract, key))
        observed = _int_or_none(video_value)
        if generated is not None and latent is not None and generated != latent:
            warnings.append(f"generation.{key}={generated} differs from H3 latent {key}={latent}")
        authoritative = latent if latent is not None else generated
        if authoritative is not None and observed is not None and authoritative != observed:
            warnings.append(f"packet {key}={authoritative} differs from primary video descriptor {key}={observed}")
        return authoritative if authoritative is not None else observed if observed is not None else _int_or_none(image_value)

    width = choose_int("width", video_facts.get("width"), image_width)
    height = choose_int("height", video_facts.get("height"), image_height)
    frames = choose_int("frames", video_facts.get("frames"))
    generation_fps = _float_or_none(generation.get("fps"))
    h3_fps = _float_or_none(_contract_value(h3_contract, "fps"))
    if generation_fps is not None and h3_fps is not None and generation_fps != h3_fps:
        warnings.append(f"generation.fps={generation_fps:g} differs from H3 latent fps={h3_fps:g}")
    fps = h3_fps if h3_fps is not None else generation_fps
    if fps is None:
        fps = _float_or_none(video_facts.get("fps"))
    duration = (frames / fps) if frames is not None and fps not in (None, 0) else _float_or_none(video_facts.get("duration"))
    aspect_ratio_value = (width / height) if width is not None and height not in (None, 0) else None
    aspect_ratio = None
    if width is not None and height not in (None, 0):
        divisor = math.gcd(width, height)
        aspect_ratio = f"{width // divisor}:{height // divisor}"

    resources = packet.resources()
    role_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for resource in resources:
        role_counts[resource["role"]] = role_counts.get(resource["role"], 0) + 1
        kind_counts[resource["kind"]] = kind_counts.get(resource["kind"], 0) + 1
    refs = {
        kind: sum(1 for resource in resources if resource.get("role") == "reference" and resource.get("kind") == kind)
        for kind in ("image", "video", "audio")
    }
    first = find_context_resource(packet, "first_frame")
    last = find_context_resource(packet, "last_frame")
    has = {
        "latent": packet.get_primary("latent") is not None,
        "video": packet.get_primary("video") is not None,
        "audio": packet.get_primary("audio") is not None,
        "first_frame": first is not None,
        "last_frame": last is not None,
    }
    history = packet.manifest.get("history", [])
    history_ops = [str(item.get("op", "?")) for item in history[-8:] if isinstance(item, dict)]

    name = packet.manifest.get("name") or "untitled"
    task = generation.get("task") or "—"
    geometry = "?×?"
    if width is not None and height is not None:
        geometry = f"{width}×{height}" + (f" ({aspect_ratio})" if aspect_ratio else "")
    temporal = ""
    if frames is not None:
        temporal = f" · {frames}f"
        if fps is not None:
            temporal += f" @ {fps:g} FPS"
        if duration is not None:
            temporal += f" / {duration:.2f}s"
    av_flags = " ".join(("L✓" if has["latent"] else "L–", "V✓" if has["video"] else "V–", "A✓" if has["audio"] else "A–"))
    lines = [
        str(name),
        f"{task} · {geometry}{temporal}",
        f"{av_flags} · Refs {refs['image']}/{refs['video']}/{refs['audio']}",
        "LoRAs: unknown" if loras is None else f"LoRAs: {len(loras)} recorded",
    ]
    if notes:
        lines.append("Notes: " + str(notes).replace("\n", " ")[:240])
    if h3_contract:
        lines.append(
            f"H3 contract v{h3_contract['contract_version']} · origin={h3_contract.get('origin', 'unknown')} · "
            f"geometry={h3_contract.get('binding', {}).get('geometry', 'unknown')} · "
            f"time={h3_contract.get('binding', {}).get('time', 'unknown')}"
        )
    if history_ops:
        lines.append("History: " + " → ".join(history_ops))
    if warnings:
        lines.append("Warnings: " + " | ".join(warnings[:3]))

    return {
        "format": packet.manifest.get("format"),
        "schema_version": packet.manifest.get("schema_version"),
        "id": packet.manifest.get("id"),
        "name": name,
        "source_archive": packet.source_archive,
        "dirty": packet.dirty,
        "verify_mode": packet.verify_mode,
        "generation": generation,
        "generation_loras": None if loras is None else list(loras),
        "generation_loras_known": loras is not None,
        "notes": notes,
        "geometry": {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "aspect_ratio_value": aspect_ratio_value,
            "frames": frames,
            "fps": fps,
            "duration": duration,
        },
        "has": has,
        "refs": refs,
        "role_counts": role_counts,
        "kind_counts": kind_counts,
        "h3_latent_contract": h3_contract,
        "history_ops": history_ops,
        "warnings": warnings,
        "summary": "\n".join(lines),
    }


def inspect_json(packet: MMH3Media) -> str:
    return json.dumps(inspect_packet(packet), ensure_ascii=False, indent=2)
