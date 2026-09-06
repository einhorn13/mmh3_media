from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import torch

from .archive import get_resource_payload
from .core import MMH3Media
from .errors import MMH3ResourceError
from .h3_resource_semantics import reference_contract


@dataclass(frozen=True)
class MaterializedVideoReference:
    frames: torch.Tensor
    audio: Any
    source_fps: float
    target_fps: float
    source_frames: int

    def info_json(self) -> str:
        return json.dumps(
            {
                "source_fps": self.source_fps,
                "target_fps": self.target_fps,
                "source_frames": self.source_frames,
                "output_frames": int(self.frames.shape[0]),
                "has_audio": self.audio is not None,
            },
            ensure_ascii=False,
            indent=2,
        )


def _resample_frames(frames: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise MMH3ResourceError(f"Reference video frames must be IMAGE [T,H,W,C], got {getattr(frames, 'shape', None)}")
    if frames.shape[0] < 1:
        raise MMH3ResourceError("Reference video contains no frames")
    if source_fps <= 0 or not math.isfinite(source_fps):
        raise MMH3ResourceError(f"Reference video has invalid frame rate {source_fps!r}")
    if target_fps <= 0 or not math.isfinite(target_fps):
        raise MMH3ResourceError(f"Target frame rate must be positive, got {target_fps!r}")
    if abs(source_fps - target_fps) < 1e-6:
        return frames
    target_count = max(1, int(round(frames.shape[0] * target_fps / source_fps)))
    positions = torch.arange(target_count, device=frames.device, dtype=torch.float64)
    indexes = torch.floor(positions * source_fps / target_fps).to(dtype=torch.long).clamp_(0, frames.shape[0] - 1)
    return frames.index_select(0, indexes)


def materialize_video_reference(
    packet: MMH3Media,
    resource_id: str,
    *,
    paired_audio_resource_id: str = "",
    include_embedded_audio: bool = False,
    target_fps: float = 24.0,
) -> MaterializedVideoReference:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    raw_descriptor = packet.get_by_id(resource_id)
    if raw_descriptor is None:
        raise MMH3ResourceError(f"Video reference resource {resource_id!r} does not exist")
    descriptor = packet.ref(resource_id).descriptor
    reference_contract(descriptor)
    if descriptor.get("kind") != "video":
        raise MMH3ResourceError(
            f"Resource {resource_id!r} is {descriptor.get('role')}/{descriptor.get('kind')}, expected reference/video"
        )
    video = get_resource_payload(packet, raw_descriptor)
    if not hasattr(video, "get_components"):
        raise MMH3ResourceError(f"Video reference {resource_id!r} does not expose get_components()")
    components = video.get_components()
    frames = getattr(components, "images", None)
    frame_rate = getattr(components, "frame_rate", None)
    try:
        source_fps = float(frame_rate)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise MMH3ResourceError(f"Video reference {resource_id!r} has invalid frame rate {frame_rate!r}") from exc
    output_frames = _resample_frames(frames, source_fps, float(target_fps))

    # The resolver owns reference ordering.  An embedded soundtrack has no
    # independent resource ID, so consuming it implicitly would make the
    # runtime graph disagree with the resolver report.  It is therefore an
    # explicit low-level adapter option; packet-native expansion only enables
    # audio through an audio reference -> video reference binding.
    audio = getattr(components, "audio", None) if include_embedded_audio else None
    if paired_audio_resource_id:
        raw_audio_descriptor = packet.get_by_id(paired_audio_resource_id)
        if raw_audio_descriptor is None:
            raise MMH3ResourceError(f"Paired audio resource {paired_audio_resource_id!r} does not exist")
        audio_descriptor = packet.ref(paired_audio_resource_id).descriptor
        reference_contract(audio_descriptor)
        if audio_descriptor.get("kind") != "audio":
            raise MMH3ResourceError(
                f"Paired resource {paired_audio_resource_id!r} is {audio_descriptor.get('role')}/{audio_descriptor.get('kind')}, expected reference/audio"
            )
        audio = get_resource_payload(packet, raw_audio_descriptor)

    return MaterializedVideoReference(output_frames, audio, source_fps, float(target_fps), int(frames.shape[0]))
