"""Resolve accepted revisions to saved artifacts and prepare exact AV ownership."""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from .archive import load_archive
from .errors import MMH3Error, MMH3ResourceError
from .project_actions import _managed_directory, _require_digest, project_archive_lock
from .project_review import build_project_review_state
from .stitch import inspect_stitch_packet, pcm_boundary
from .util import json_dumps_canonical, sha256_file


def accepted_sources(manager, project_id):
    current = manager.current(project_id)
    packet = load_archive(current, verify="manifest")
    namespace = packet.manifest["extensions"]["mmh3_media"]
    token = hashlib.sha256(os.path.normcase(str(current)).encode()).hexdigest()[:24]
    history = current.parent / ".mmh3_review" / token
    paths = [current]
    for receipt in namespace.get("project_publications", []):
        name = receipt.get("previous_snapshot", "")
        if not re.fullmatch(r"[0-9a-f]{64}\.mmh3", name):
            raise MMH3ResourceError("Invalid project snapshot reference")
        paths.append(history / name)
    for digest in namespace.get("project_segment_sources", []):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MMH3ResourceError("Invalid saved segment reference")
        paths.append(current.parent / "segments" / (digest + ".mmh3"))
    view = build_project_review_state(packet)
    sources = {}
    for path in dict.fromkeys(paths):
        if path.resolve() != path or not path.is_file():
            continue
        try:
            saved = build_project_review_state(load_archive(path, verify="manifest"))
            if not saved.head_state_matches or saved.chain_id != view.chain_id:
                continue
            head = next((s for s in saved.segments if s.segment_id == saved.head_segment_id), None)
            if head:
                key = (head.segment_id, head.active_revision, head.packet_state_fingerprint)
                sources.setdefault(key, path)
        except MMH3Error:
            continue
    return packet, view, sources


def project_timeline(manager, project_id):
    packet, view, sources = accepted_sources(manager, project_id)
    items = []
    for segment in sorted(view.segments, key=lambda s: s.index):
        if segment.status != "accepted":
            continue
        path = sources.get((segment.segment_id, segment.active_revision, segment.packet_state_fingerprint))
        items.append({"segment_id": segment.segment_id, "revision": segment.active_revision,
            "index": segment.index, "scene_id": segment.scene_id,
            "parent_segment_id": segment.parent_segment_id,
            "file": manager.selector(path) if path else None, "available": path is not None})
    return {"state_digest": view.state_digest, "segments": items}


def assembly_preview(manager, project_id, expected):
    packet, view, sources = accepted_sources(manager, project_id)
    _require_digest(view.state_digest, expected)
    items, errors, frames, dimensions = [], [], 0, None
    previous_id = ""
    for segment in sorted(view.segments, key=lambda s: s.index):
        if segment.status != "accepted":
            continue
        record = segment.to_dict()["record"]
        path = sources.get((segment.segment_id, segment.active_revision, segment.packet_state_fingerprint))
        if path is None:
            errors.append(f"Segment {segment.index + 1}: attach its saved accepted archive before assembly")
            previous_id = segment.segment_id
            continue
        source = load_archive(path, verify="manifest")
        fact, reasons = inspect_stitch_packet(source, index=segment.index, expected_dimensions=dimensions)
        errors.extend(reasons)
        if dimensions is None and fact.get("dimensions"):
            dimensions = tuple(fact["dimensions"])
        context = 0
        if items and record.get("handover") == "continuation":
            timing = (record.get("segment_plan") or {}).get("timing") or {}
            context = timing.get("context_frames")
            if type(context) is not int or context < 1:
                errors.append(f"Segment {segment.index + 1}: continuation context is not recorded; refusing to guess trim")
                context = 0
            if segment.parent_segment_id != previous_id:
                errors.append(f"Segment {segment.index + 1}: continuation parent is not the preceding accepted segment")
        count = fact.get("frame_count") or 0
        if count <= context:
            errors.append(f"Segment {segment.index + 1}: no new video frames after context")
        owned = max(0, count - context)
        audio_start = pcm_boundary(context)
        audio_end = audio_start + pcm_boundary(frames + owned) - pcm_boundary(frames)
        if audio_end > (fact.get("audio_samples") or 0):
            errors.append(f"Segment {segment.index + 1}: decoded audio is too short for exact assembly; resave with complete audio")
        digest, size = sha256_file(path)
        if re.fullmatch(r"[0-9a-f]{64}", path.stem) and digest != path.stem:
            errors.append(f"Segment {segment.index + 1}: saved artifact hash differs from its registered identity")
        items.append({"segment_id": segment.segment_id, "revision": segment.active_revision,
            "index": segment.index, "file": manager.selector(path), "sha256": digest, "size": size,
            "packet_id": source.manifest["id"], "video_trim_start_frame": context,
            "video_trim_end_frame": count, "audio_trim_start": audio_start, "audio_trim_end": audio_end,
            "output_start_frame": frames, "owned_frames": owned})
        frames += owned
        previous_id = segment.segment_id
    if not items:
        errors.append("No accepted saved segments to assemble")
    result = {"state_digest": expected, "ready": not errors, "errors": errors,
        "segments": items, "output_frames": frames, "output_audio_samples": pcm_boundary(frames),
        "duration_seconds": frames / 24, "policy": "Accepted revisions only. Repeated continuation context is removed from video and audio. Scene changes use cuts."}
    result["assembly_digest"] = hashlib.sha256(json_dumps_canonical(result).encode()).hexdigest()
    return result


def export_project(manager, project_id, expected, expected_assembly):
    from .automation_assembly import assemble_chunk_packets
    import json
    with project_archive_lock(manager.current(project_id)):
        preview = assembly_preview(manager, project_id, expected)
        _require_digest(preview["assembly_digest"], expected_assembly, "Assembly inputs")
        if not preview["ready"]:
            raise MMH3ResourceError("; ".join(preview["errors"]))
        base = _managed_directory(manager.directory(project_id), "exports")
        destination = base / expected_assembly
        if destination.exists():
            path = export_path(manager, project_id, expected_assembly)
            return {"export_id": expected_assembly, "filename": str(path.relative_to(manager.roots["output"])),
                    "duration_seconds": preview["duration_seconds"]}
        assembly_map = {"contract": "mmh3_chunk_assembly_map_v1", "output_frames": preview["output_frames"],
            "output_audio_samples": preview["output_audio_samples"],
            "segments": [{**item, "packet_path": str(manager.resolve(item["file"]))} for item in preview["segments"]]}
        result = assemble_chunk_packets(assembly_map)
        with tempfile.TemporaryDirectory(prefix=".export_", dir=base) as temporary:
            staged = Path(temporary)
            result.video.save_to(str(staged / "final.mp4"), format="mp4", codec="h264")
            digest, size = sha256_file(staged / "final.mp4")
            if size == 0:
                raise MMH3ResourceError("Video encoder produced an empty file")
            _require_digest(assembly_preview(manager, project_id, expected)["assembly_digest"], expected_assembly, "Assembly inputs")
            (staged / "export.json").write_text(json.dumps({"preview": preview, "video_sha256": digest}, ensure_ascii=False, indent=2), encoding="utf-8")
            os.rename(staged, destination)
        return {"export_id": expected_assembly, "filename": str((destination / "final.mp4").relative_to(manager.roots["output"])),
                "duration_seconds": preview["duration_seconds"]}


def export_path(manager, project_id, export_id):
    import json
    if not isinstance(export_id, str) or not re.fullmatch(r"[0-9a-f]{64}", export_id):
        raise MMH3ResourceError("Invalid export ID")
    directory = manager.directory(project_id) / "exports" / export_id
    path, receipt = directory / "final.mp4", directory / "export.json"
    if any(p.resolve() != p or not p.is_file() for p in (path, receipt)):
        raise MMH3ResourceError("Export is missing or unsafe")
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    _require_digest(sha256_file(path)[0], metadata.get("video_sha256"), "Export video")
    return path
