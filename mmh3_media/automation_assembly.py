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
    if int(source_audio.get("sample_rate", 0)) != AUDIO_SAMPLE_RATE:
        raise MMH3ResourceError(f"Immutable master audio must be {AUDIO_SAMPLE_RATE} Hz")
    waveform = _normalize_audio_tensor(source_audio.get("waveform"), name="Immutable master audio")
    expected_channels = int(expected_audio.get("channels") or 0)
    if expected_channels and int(waveform.shape[-2]) != expected_channels:
        raise MMH3ResourceError("Immutable master audio channel count changed")
    expected_source_samples = int(expected_audio.get("samples") or 0)
    if expected_source_samples and int(waveform.shape[-1]) != expected_source_samples:
        raise MMH3ResourceError("Immutable master audio sample count changed")
    normalized = {"waveform": waveform, "sample_rate": AUDIO_SAMPLE_RATE}
    return trim_audio_samples(normalized, 0, int(output_samples))



def _db_gain(db: float) -> float:
    value = float(db)
    if value <= -120.0:
        return 0.0
    return 10.0 ** (value / 20.0)


def _audio_channels(waveform: torch.Tensor) -> int:
    if waveform.ndim < 2:
        return 1
    return int(waveform.shape[-2])


def _normalize_audio_tensor(waveform: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(waveform, torch.Tensor) or waveform.ndim not in {2, 3}:
        raise MMH3ResourceError(f"{name} waveform must be [channels,samples] or [batch,channels,samples]")
    value = waveform.unsqueeze(0) if waveform.ndim == 2 else waveform
    if int(value.shape[0]) != 1:
        raise MMH3ResourceError(f"{name} waveform batch size must be 1; got {int(value.shape[0])}")
    if int(value.shape[-2]) not in {1, 2}:
        raise MMH3ResourceError(f"{name} waveform must be mono/stereo; got {int(value.shape[-2])} channels")
    if not torch.isfinite(value).all().item():
        raise MMH3ResourceError(f"{name} waveform contains NaN or Inf")
    return value


def _match_audio_channels(waveform: torch.Tensor, target_channels: int) -> torch.Tensor:
    value = _normalize_audio_tensor(waveform, name="Generated H3 audio")
    channels = _audio_channels(value)
    target = int(target_channels)
    if target not in {1, 2}:
        raise MMH3ResourceError(f"F18 output requires mono/stereo master audio; got {target}")
    if channels == target:
        return value
    if channels == 1 and target == 2:
        return value.repeat_interleave(2, dim=-2)
    if channels == 2 and target == 1:
        return value.mean(dim=-2, keepdim=True)
    raise MMH3ResourceError(f"Cannot map generated H3 audio channels {channels} -> {target}")


def _trim_generated_audio(audio: Mapping[str, Any], start: int, end: int) -> tuple[torch.Tensor, int]:
    if int(audio.get("sample_rate", 0)) != AUDIO_SAMPLE_RATE:
        raise MMH3ResourceError(f"Generated H3 audio must be {AUDIO_SAMPLE_RATE} Hz")
    waveform = _normalize_audio_tensor(audio.get("waveform"), name="Generated H3 audio")
    start_i, end_i = int(start), int(end)
    if start_i < 0 or end_i <= start_i:
        raise MMH3ResourceError("Generated H3 audio ownership is invalid")
    length = int(waveform.shape[-1])
    available_end = min(end_i, length)
    trimmed = waveform[..., start_i:available_end] if start_i < length else waveform[..., 0:0]
    missing = max(0, end_i - max(start_i, available_end))
    # H3 audio decode can differ by a small codec/grid tail. Larger gaps are a
    # broken candidate and must not be hidden under silence.
    max_padding = AUDIO_SAMPLE_RATE // 4
    if missing > max_padding:
        raise MMH3ResourceError(
            f"Generated H3 audio is {missing} samples short for owned timeline (>250 ms)"
        )
    if missing:
        shape = list(trimmed.shape)
        shape[-1] = missing
        trimmed = torch.cat([trimmed, torch.zeros(shape, dtype=trimmed.dtype, device=trimmed.device)], dim=-1)
    expected = end_i - start_i
    if int(trimmed.shape[-1]) != expected:
        raise MMH3ResourceError("Generated H3 audio trim did not match owned PCM length")
    return trimmed, missing


def _apply_peak_limit(waveform: torch.Tensor, enabled: bool) -> tuple[torch.Tensor, float]:
    if not enabled or waveform.numel() == 0:
        return waveform, 1.0
    if not torch.isfinite(waveform).all().item():
        raise MMH3ResourceError("F18 delivery audio contains NaN or Inf")
    peak = float(waveform.detach().abs().max().item())
    if peak <= 0.98 or peak <= 0.0:
        return waveform, 1.0
    scale = 0.98 / peak
    return waveform * scale, scale

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
    expected_samples = int(assembly_map.get("output_audio_samples") or 0)
    delivery = (lipsync_settings or {}).get("delivery_policy") or {}
    audio_output_mode = str(delivery.get("audio_output_mode") or "master_only")
    if audio_output_mode not in {"master_only", "master_plus_generated", "generated_only"}:
        raise MMH3ResourceError(f"Unsupported F18 audio output mode {audio_output_mode!r}")
    if audio_output_mode != "master_only" and not bool((lipsync_settings or {}).get("generated_audio_capture")):
        raise MMH3ResourceError("Selected F18 audio mode requires candidates that captured generated H3 audio")

    master_audio = load_immutable_lipsync_audio(lipsync_settings, expected_samples) if lipsync_settings is not None else None
    target_channels = _audio_channels(master_audio["waveform"]) if master_audio is not None else 0

    videos: list[Any] = []
    frame_counts: list[int] = []
    audio_parts: list[torch.Tensor] = []
    generated_parts: list[torch.Tensor] = []
    generated_padding_samples = 0
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
        elif lipsync_settings is not None and audio_output_mode != "master_only":
            if audio_desc is None:
                raise MMH3ResourceError(f"F18 candidate chunk {index} has no captured generated audio")
            generated = get_resource_payload(packet, audio_desc)
            owned_generated, padded = _trim_generated_audio(
                generated, int(raw["audio_trim_start"]), int(raw["audio_trim_end"])
            )
            generated_padding_samples += padded
            generated_parts.append(_match_audio_channels(owned_generated, target_channels))

        packets.append(packet)
        report_segments.append(deep_copy_json(dict(raw)))

    output_audio = None
    limiter_scale = 1.0
    if lipsync_settings is not None:
        assert master_audio is not None
        master_waveform = _normalize_audio_tensor(master_audio["waveform"], name="Immutable master audio")
        if int(master_waveform.shape[-1]) != expected_samples:
            raise MMH3ResourceError("Immutable master audio length differs from assembly ownership")
        master_gain = _db_gain(float(delivery.get("master_gain_db", 0.0)))
        generated_gain = _db_gain(float(delivery.get("generated_gain_db", -18.0)))
        if audio_output_mode == "master_only":
            waveform = master_waveform if master_gain == 1.0 else master_waveform * master_gain
        else:
            if len(generated_parts) != len(packets):
                raise MMH3ResourceError("Every selected F18 candidate must contain generated audio for this output mode")
            generated_parts = [part.to(device=master_waveform.device, dtype=master_waveform.dtype) for part in generated_parts]
            generated_waveform = torch.cat(generated_parts, dim=-1)
            if int(generated_waveform.shape[-1]) != expected_samples:
                raise MMH3ResourceError("Generated H3 PCM length differs from exact assembly ownership")
            if audio_output_mode == "generated_only":
                waveform = generated_waveform * generated_gain
            else:
                waveform = master_waveform * master_gain + generated_waveform * generated_gain
        if audio_output_mode != "master_only" or master_gain != 1.0:
            waveform, limiter_scale = _apply_peak_limit(waveform, bool(delivery.get("peak_limit", True)))
        output_audio = {"waveform": waveform, "sample_rate": AUDIO_SAMPLE_RATE}
    elif audio_parts:
        if len(audio_parts) != len(packets):
            raise MMH3ResourceError("Assembly cannot mix chunks with and without audio")
        waveform = torch.cat(audio_parts, dim=-1)
        if int(waveform.shape[-1]) != expected_samples:
            raise MMH3ResourceError("Assembled PCM length differs from exact ownership map")
        output_audio = {"waveform": waveform, "sample_rate": AUDIO_SAMPLE_RATE}

    total_frames = sum(frame_counts)
    if total_frames != int(assembly_map["output_frames"]):
        raise MMH3ResourceError("Assembled frame ownership differs from the declared output")
    plan = StitchPlan("cut", "cut", 0, total_frames, expected_samples, tuple(report_segments))
    output_video = StreamingStitchedVideo(tuple(videos), tuple(frame_counts), dimensions, output_audio, plan)
    report = deep_copy_json(dict(assembly_map))
    report.update({
        "assembly_mode": "bounded_streaming_ownership",
        "verified_artifacts": len(packets),
        "audio_source": (
            audio_output_mode if lipsync_settings is not None else "chunk_ownership"
        ),
        "audio_mix": {
            "mode": audio_output_mode if lipsync_settings is not None else "chunk_ownership",
            "master_gain_db": float(delivery.get("master_gain_db", 0.0)) if lipsync_settings is not None else None,
            "generated_gain_db": float(delivery.get("generated_gain_db", -18.0)) if lipsync_settings is not None else None,
            "peak_limit": bool(delivery.get("peak_limit", True)) if lipsync_settings is not None else None,
            "limiter_scale": limiter_scale,
            "generated_padding_samples": generated_padding_samples,
            "generated_audio_semantics": delivery.get("generated_audio_semantics") if lipsync_settings is not None else None,
        },
    })
    from .process_result import prepare_assembly_packet
    base_packet, report["source_controls"] = prepare_assembly_packet(packets)
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
