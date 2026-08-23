from __future__ import annotations

import json
from typing import Any

from .core import MMH3Media
from .h3 import h3_context_from_metadata, h3_frame_count_from_video_t, h3_expected_audio_t, h3_metadata_with_context, validate_h3_av_latent


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _latent_manifest_info(packet: MMH3Media, warnings: list[str]) -> dict[str, Any]:
    res = packet.get_by_role_slot("h3_av_latent", 0)
    if res is None:
        return {}
    if res["id"] in packet.payloads:
        try:
            info = validate_h3_av_latent(packet.payloads[res["id"]], strict_audio_length=False)
            warnings.extend(info.warnings)
            existing_h3 = None
            md = res.get("metadata", {})
            if isinstance(md, dict) and isinstance(md.get("h3"), dict):
                existing_h3 = md["h3"]
            return h3_metadata_with_context(info, existing_h3=existing_h3)
        except Exception as e:
            warnings.append(f"H3 AV latent validation: {e}")
            return {}

    # Loaded packets stay lazy: inspect the serializer metadata instead of touching safetensors.
    md = res.get("metadata", {})
    h3 = md.get("h3")
    if not isinstance(h3, dict):
        return {}
    out = dict(h3)
    video_shape = out.get("video_shape")
    audio_shape = out.get("audio_shape")
    if isinstance(video_shape, list) and len(video_shape) == 5:
        vt = _int_or_none(video_shape[2])
        if vt is not None:
            derived_frames = h3_frame_count_from_video_t(vt)
            declared_frames = _int_or_none(out.get("frames"))
            if derived_frames is None:
                warnings.append(f"Saved H3 video latent T={vt} is off the stock temporal grid")
            elif declared_frames is not None and derived_frames != declared_frames:
                warnings.append(
                    f"Saved H3 metadata says {declared_frames} frames, but video latent T={vt} derives {derived_frames}"
                )
            if isinstance(audio_shape, list) and len(audio_shape) == 4 and derived_frames is not None:
                at = _int_or_none(audio_shape[-1])
                expected = h3_expected_audio_t(derived_frames)
                if at is not None and at != expected:
                    warnings.append(
                        f"Saved H3 AV duration mismatch: video implies audio T40={expected}, stored audio T40={at}"
                    )
    return out


def inspect_packet(packet: MMH3Media) -> dict[str, Any]:
    generation = packet.manifest.get("generation", {})
    notes = packet.manifest.get("notes", "")
    warnings: list[str] = []
    h3 = _latent_manifest_info(packet, warnings)
    h3_context = h3_context_from_metadata(h3)

    def choose_int(key: str) -> int | None:
        g = _int_or_none(generation.get(key))
        h = _int_or_none(h3.get(key))
        if g is not None and h is not None and g != h:
            warnings.append(f"generation.{key}={g} differs from H3 latent {key}={h}")
        return g if g is not None else h

    width = choose_int("width")
    height = choose_int("height")
    frames = choose_int("frames")
    fps = _float_or_none(generation.get("fps"))
    if fps is None:
        fps = _float_or_none(h3.get("fps")) or 24.0
    duration = (frames / fps) if frames is not None and fps > 0 else None

    resources = packet.resources()
    role_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for res in resources:
        role_counts[res["role"]] = role_counts.get(res["role"], 0) + 1
        kind_counts[res["kind"]] = kind_counts.get(res["kind"], 0) + 1

    refs = {
        "picture": role_counts.get("picture_ref", 0),
        "video": role_counts.get("video_ref", 0),
        "audio": role_counts.get("audio_ref", 0),
    }
    has = {
        "latent": packet.get_by_role_slot("h3_av_latent", 0) is not None,
        "video": packet.get_by_role_slot("video", 0) is not None,
        "audio": packet.get_by_role_slot("audio", 0) is not None,
        "first_frame": packet.get_by_role_slot("first_frame", 0) is not None,
        "last_frame": packet.get_by_role_slot("last_frame", 0) is not None,
    }
    history = packet.manifest.get("history", [])
    history_ops = [str(x.get("op", "?")) for x in history[-8:] if isinstance(x, dict)]

    name = packet.manifest.get("name") or "untitled"
    task = generation.get("task") or "—"
    geometry = "?×?"
    if width is not None and height is not None:
        geometry = f"{width}×{height}"
    temporal = ""
    if frames is not None:
        temporal = f" · {frames}f"
        if duration is not None:
            temporal += f" / {duration:.2f}s"
    av_flags = " ".join(
        [
            "L✓" if has["latent"] else "L–",
            "V✓" if has["video"] else "V–",
            "A✓" if has["audio"] else "A–",
        ]
    )
    lines = [
        str(name),
        f"{task} · {geometry}{temporal}",
        f"{av_flags} · Refs {refs['picture']}/{refs['video']}/{refs['audio']}",
    ]
    if notes:
        lines.append("Notes: " + str(notes).replace("\n", " ")[:240])
    if h3_context:
        continuation = h3_context.get("continuation", {})
        seam = h3_context.get("seam_source", {})
        lines.append(
            f"H3 context: origin={h3_context.get('origin', 'unknown')} · continuation={continuation.get('status', 'unknown')} · seam={seam.get('status', 'unknown')}"
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
        "notes": notes,
        "geometry": {"width": width, "height": height, "frames": frames, "fps": fps, "duration": duration},
        "has": has,
        "refs": refs,
        "role_counts": role_counts,
        "kind_counts": kind_counts,
        "h3_latent": h3 or None,
        "h3_latent_context": h3_context,
        "history_ops": history_ops,
        "warnings": warnings,
        "summary": "\n".join(lines),
    }


def inspect_json(packet: MMH3Media) -> str:
    return json.dumps(inspect_packet(packet), ensure_ascii=False, indent=2)
