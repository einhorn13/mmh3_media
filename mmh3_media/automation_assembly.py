from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import torch

from .archive import get_resource_payload, load_archive
from .automation_adapters import trim_audio_samples
from .constants import AUDIO_SAMPLE_RATE, FPS
from .errors import MMH3ResourceError
from .process_result import pack_h3_result
from .stitch import StitchPlan
from .streaming_stitch import StreamingStitchedVideo
from .util import deep_copy_json
from .h3_resource_semantics import find_context_resource


@dataclass(frozen=True)
class ChunkAssemblyResult:
    packet: Any
    video: Any
    audio: dict[str, Any] | None
    report: dict[str, Any]


def load_immutable_lipsync_audio(settings: Mapping[str, Any], output_samples: int) -> dict[str, Any]:
    source_packet = load_archive(Path(str(settings["source_archive"])).resolve(), verify="manifest")
    if source_packet.manifest.get("id") != settings.get("source_id"):
        raise MMH3ResourceError("Lipsync immutable-audio source packet ID changed")
    expected_audio = settings.get("source_audio", {})
    expected_resource_id = expected_audio.get("resource_id")
    source_audio_desc = (
        source_packet.get_by_id(str(expected_resource_id))
        if isinstance(expected_resource_id, str) and expected_resource_id
        else None
    )
    if source_audio_desc is None or source_packet.ref(source_audio_desc["id"]).kind != "audio":
        raise MMH3ResourceError("Lipsync immutable source audio resource changed")
    content = source_audio_desc.get("content") if isinstance(source_audio_desc.get("content"), Mapping) else {}
    revision = str(content.get("revision") or content.get("digest") or "unknown")
    if revision != str(expected_audio.get("revision") or "unknown"):
        raise MMH3ResourceError("Lipsync immutable source audio revision changed")
    source_audio = get_resource_payload(source_packet, source_audio_desc)
    return trim_audio_samples(source_audio, 0, int(output_samples))


def _verify_artifact(segment: Mapping[str, Any]) -> Path:
    path = Path(str(segment.get("packet_path") or "")).resolve()
    expected_hash = str(segment.get("sha256") or "")
    expected_size = segment.get("size")
    if path.suffix.casefold() != ".mmh3" or not path.is_file():
        raise MMH3ResourceError(f"Assembly artifact is missing: {path}")
    if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
        raise MMH3ResourceError(f"Assembly artifact size changed: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    if not expected_hash or hasher.hexdigest() != expected_hash:
        raise MMH3ResourceError(f"Assembly artifact SHA-256 mismatch: {path}")
    return path


def assemble_chunk_packets(assembly_map: Mapping[str, Any]) -> ChunkAssemblyResult:
    if not isinstance(assembly_map, Mapping) or assembly_map.get("contract") != "mmh3_chunk_assembly_map_v1":
        raise MMH3ResourceError("Expected an mmh3_chunk_assembly_map_v1 map")
    segments_raw = assembly_map.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        raise MMH3ResourceError("Chunk assembly map has no segments")
    effective_settings = assembly_map.get("effective_settings")
    lipsync_settings = (
        effective_settings
        if isinstance(effective_settings, Mapping)
        and effective_settings.get("contract") == "mmh3_long_video_audio_sync_settings_v2"
        else None
    )

    videos: list[Any] = []
    frame_counts: list[int] = []
    audio_parts: list[torch.Tensor] = []
    packets = []
    dimensions: tuple[int, int] | None = None
    report_segments = []
    for index, raw in enumerate(segments_raw):
        if not isinstance(raw, Mapping):
            raise MMH3ResourceError("Chunk assembly segment is malformed")
        path = _verify_artifact(raw)
        packet = load_archive(path, verify="manifest")
        if raw.get("packet_id") and packet.manifest.get("id") != raw["packet_id"]:
            raise MMH3ResourceError(f"Assembly packet ID mismatch: {path}")
        video_desc = packet.get_primary("video")
        if video_desc is None:
            raise MMH3ResourceError(f"Assembly chunk {index} has no primary video")
        video = get_resource_payload(packet, video_desc)
        start_frame = int(raw["video_trim_start_frame"])
        end_frame = int(raw["video_trim_end_frame"])
        owned_frames = end_frame - start_frame
        if owned_frames < 1 or not hasattr(video, "as_trimmed"):
            raise MMH3ResourceError(f"Assembly chunk {index} has invalid video ownership/contract")
        fps = float(packet.manifest.get("generation", {}).get("fps") or FPS)
        if fps != float(FPS):
            raise MMH3ResourceError(f"Bounded assembly currently requires {FPS} FPS; got {fps}")
        owned_video = video.as_trimmed(start_frame / fps, owned_frames / fps, strict_duration=True)
        current_dimensions = tuple(int(value) for value in owned_video.get_dimensions())
        if dimensions is None:
            dimensions = current_dimensions
        elif current_dimensions != dimensions:
            raise MMH3ResourceError("Assembly chunk canvases do not match")
        videos.append(owned_video)
        frame_counts.append(owned_frames)

        audio_desc = packet.get_primary("audio")
        if lipsync_settings is None and audio_desc is not None:
            audio = get_resource_payload(packet, audio_desc)
            if int(audio.get("sample_rate", 0)) != AUDIO_SAMPLE_RATE:
                raise MMH3ResourceError(f"Assembly requires {AUDIO_SAMPLE_RATE} Hz audio")
            owned_audio = trim_audio_samples(audio, int(raw["audio_trim_start"]), int(raw["audio_trim_end"]))
            audio_parts.append(owned_audio["waveform"])
        elif lipsync_settings is None and audio_parts:
            raise MMH3ResourceError("Assembly cannot mix chunks with and without audio")
        packets.append(packet)
        report_segments.append(deep_copy_json(dict(raw)))

    output_audio = None
    if lipsync_settings is not None:
        expected_samples = int(assembly_map["output_audio_samples"])
        output_audio = load_immutable_lipsync_audio(lipsync_settings, expected_samples)
    elif audio_parts:
        if len(audio_parts) != len(packets):
            raise MMH3ResourceError("Assembly cannot mix chunks with and without audio")
        waveform = torch.cat(audio_parts, dim=-1)
        expected_samples = int(assembly_map["output_audio_samples"])
        if int(waveform.shape[-1]) != expected_samples:
            raise MMH3ResourceError("Assembled PCM length differs from exact ownership map")
        output_audio = {"waveform": waveform, "sample_rate": AUDIO_SAMPLE_RATE}

    total_frames = sum(frame_counts)
    if total_frames != int(assembly_map["output_frames"]):
        raise MMH3ResourceError("Assembled frame ownership differs from the declared output")
    plan = StitchPlan("cut", "cut", 0, total_frames, int(assembly_map["output_audio_samples"]), tuple(report_segments))
    output_video = StreamingStitchedVideo(tuple(videos), tuple(frame_counts), dimensions, output_audio, plan)
    report = deep_copy_json(dict(assembly_map))
    report.update({
        "assembly_mode": "bounded_streaming_ownership",
        "verified_artifacts": len(packets),
        "audio_source": "immutable_original_packet" if lipsync_settings is not None else "chunk_ownership",
    })
    base_packet = packets[-1]
    primary_latent = base_packet.get_primary("latent")
    if primary_latent is not None:
        base_packet = base_packet.remove(resource_id=primary_latent["id"], missing="ignore", record_history=False)
    for usage in ("first_frame", "last_frame"):
        context = find_context_resource(base_packet, usage)
        if context is not None:
            base_packet = base_packet.remove(resource_id=context["id"], missing="ignore", record_history=False)
    packed = pack_h3_result(
        base_packet,
        video=output_video,
        audio=output_audio,
        operation="automation_assembly",
        mode="cut+exact_pcm",
        status=f"READY · automation assembly · {len(packets)} chunks · {total_frames}f",
        process_info=report,
    )
    packet = packed.packet.edit_metadata(
        merge_patch_json={
            "generation": {
                "frames": total_frames,
                "fps": float(FPS),
                "audio_sample_rate": AUDIO_SAMPLE_RATE if output_audio is not None else None,
                "automation_chunk": None,
            }
        }
    ).set_extension_value("mmh3_media", "automation_assembly", report)
    return ChunkAssemblyResult(packet, output_video, output_audio, report)


__all__ = ["ChunkAssemblyResult", "assemble_chunk_packets", "load_immutable_lipsync_audio"]
