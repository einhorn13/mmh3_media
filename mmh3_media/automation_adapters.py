from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .automation import ChunkExecutionSelection


@dataclass(frozen=True)
class VideoChunkExpansion:
    packet: Any
    video: Any
    audio: Any
    context_json: str
    status: str
    graph: Any


def trim_audio_samples(audio: Any, start_sample: int, end_sample: int) -> dict[str, Any]:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Expected ComfyUI AUDIO with waveform and sample_rate")
    waveform = audio["waveform"]
    if not hasattr(waveform, "shape") or len(waveform.shape) < 1:
        raise ValueError("AUDIO waveform has no sample dimension")
    start = int(start_sample)
    end = int(end_sample)
    length = int(waveform.shape[-1])
    if start < 0 or end <= start:
        raise ValueError("Audio sample boundaries must satisfy 0 <= start < end")
    if end > length:
        raise ValueError(f"Audio sample boundary {end} exceeds source length {length}")
    return {"waveform": waveform[..., start:end], "sample_rate": int(audio["sample_rate"])}


def build_video_chunk_expansion(
    selection: ChunkExecutionSelection,
    *,
    packet: Any,
    has_audio: bool,
    graph_builder_factory: Callable[[], Any] | None = None,
) -> VideoChunkExpansion:
    if graph_builder_factory is None:
        from comfy_execution.graph_utils import GraphBuilder  # type: ignore

        graph_builder_factory = GraphBuilder
    graph = graph_builder_factory()
    context = selection.context()
    context_json = json.dumps(context, ensure_ascii=False, indent=2)

    source_video = graph.node(
        "MMH3GetVideo",
        packet=packet,
        role="auxiliary",
        order=-1,
        resource_id="",
        missing="error",
    )
    trimmed_video = graph.node(
        "Video Slice",
        video=source_video.out(0),
        start_time=selection.start_time_seconds,
        duration=selection.duration_seconds,
        strict_duration=True,
    )
    video_packet = graph.node(
        "MMH3Put",
        packet=packet,
        resource=trimmed_video.out(0),
        role="auxiliary",
        order=-1,
        mode="upsert",
        primary=True,
        resource_id="",
        name="",
        tags="automation_chunk",
        descriptor_json="{}",
        extensions_json=json.dumps({"mmh3_media": {"automation_chunk": context}}, ensure_ascii=False),
    )

    current_packet = video_packet.out(0)
    trimmed_audio: Any = None
    if has_audio:
        source_audio = graph.node(
            "MMH3GetAudio",
            packet=packet,
            role="auxiliary",
            order=-1,
            resource_id="",
            missing="error",
        )
        audio_trim = graph.node(
            "MMH3TrimAudioSamples",
            audio=source_audio.out(0),
            start_sample=int(selection.job["audio_read_start"]),
            end_sample=int(selection.job["audio_read_end"]),
        )
        trimmed_audio = audio_trim.out(0)
        audio_packet = graph.node(
            "MMH3Put",
            packet=current_packet,
            resource=trimmed_audio,
            role="auxiliary",
            order=-1,
            mode="upsert",
            primary=True,
            resource_id="",
            name="",
            tags="automation_chunk",
            descriptor_json="{}",
            extensions_json=json.dumps({"mmh3_media": {"automation_chunk": context}}, ensure_ascii=False),
        )
        current_packet = audio_packet.out(0)

    metadata = graph.node(
        "MMH3Metadata",
        packet=current_packet,
        name="",
        tags_action="keep",
        tags="",
        notes_action="keep",
        notes="",
        custom_json_merge_patch=json.dumps(
            {
                "generation": {
                    "frames": context["read_end_frame"] - context["read_start_frame"],
                    "fps": float(selection.fps),
                    "automation_chunk": context,
                }
            },
            ensure_ascii=False,
        ),
    )
    audio_label = "with audio" if has_audio else "video only"
    status = f"READY · chunk {context['index']} · {context['read_end_frame'] - context['read_start_frame']}f · {audio_label}"
    return VideoChunkExpansion(
        metadata.out(0), trimmed_video.out(0), trimmed_audio, context_json, status, graph.finalize()
    )


__all__ = ["VideoChunkExpansion", "build_video_chunk_expansion", "trim_audio_samples"]
