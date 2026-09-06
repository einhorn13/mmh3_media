from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import MMH3ResourceError


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


@dataclass(frozen=True)
class GenerationSettings:
    geometry_mode: str
    timeline_mode: str
    audio_policy: str
    width_override: int
    height_override: int
    frames_override: int
    source_geometry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "mmh3_h3_generation_settings_v2",
            "geometry_mode": self.geometry_mode,
            "timeline_mode": self.timeline_mode,
            "audio_policy": self.audio_policy,
            "effective_overrides": {
                "width": self.width_override,
                "height": self.height_override,
                "frames": self.frames_override,
            },
            "source_geometry": dict(self.source_geometry),
        }

    def summary(self) -> str:
        geometry = "auto" if not self.width_override else f"{self.width_override}x{self.height_override}"
        frames = "auto" if not self.frames_override else str(self.frames_override)
        return f"READY · generation settings · geometry={geometry} frames={frames} audio={self.audio_policy}"


def _legal_h3_frames(frames: int) -> bool:
    return frames >= 5 and (frames - 5) % 17 == 0


def _align_source_frames(frames: int) -> int:
    if frames < 5:
        raise MMH3ResourceError("Source duration is shorter than the minimum H3 timeline")
    return 5 + 17 * ((frames - 5) // 17)


def _align_canvas(value: float) -> int:
    return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def adapt_h3_canvas(source_width: int, source_height: int) -> tuple[int, int]:
    """Match ComfyUI's native MiniMax H3 image-canvas policy.

    Preserve the source aspect ratio, target a 768px short edge, cap the native
    canvas area at 768x1344, and align each edge to the H3 32px grid.
    """
    width, height = int(source_width), int(source_height)
    if width <= 0 or height <= 0:
        raise MMH3ResourceError("Source width/height must be positive")
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return _align_canvas(nominal_width), _align_canvas(nominal_height)


def build_generation_settings(
    source_geometry: Mapping[str, Any] | None,
    *,
    geometry_mode: str = "auto_from_source",
    custom_width: int = 768,
    custom_height: int = 768,
    timeline_mode: str = "auto_from_source",
    custom_frames: int = 73,
    audio_policy: str = "auto",
) -> GenerationSettings:
    if geometry_mode not in {"auto_from_source", "source", "custom"}:
        raise MMH3ResourceError(f"Unknown geometry_mode {geometry_mode!r}")
    if timeline_mode not in {"auto_from_source", "source_duration", "custom"}:
        raise MMH3ResourceError(f"Unknown timeline_mode {timeline_mode!r}")
    if audio_policy not in {"auto", "preserve", "generate", "silence", "error"}:
        raise MMH3ResourceError(f"Unknown audio_policy {audio_policy!r}")

    geometry = dict(source_geometry or {})
    width = height = 0
    if geometry_mode in {"auto_from_source", "source"}:
        try:
            source_width = int(geometry.get("width") or 0)
            source_height = int(geometry.get("height") or 0)
        except (TypeError, ValueError):
            source_width = source_height = 0
        if source_width > 0 and source_height > 0:
            if geometry_mode == "source":
                width, height = _align_canvas(source_width), _align_canvas(source_height)
            else:
                width, height = adapt_h3_canvas(source_width, source_height)
    else:
        width, height = int(custom_width), int(custom_height)
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise MMH3ResourceError("Custom H3 width/height must be >=32 and divisible by 32")

    frames = 0
    if timeline_mode == "source_duration":
        source_frames = geometry.get("frames")
        if source_frames in (None, 0):
            raise MMH3ResourceError("source_duration requires source frame metadata")
        frames = _align_source_frames(int(source_frames))
    elif timeline_mode == "custom":
        frames = int(custom_frames)
        if not _legal_h3_frames(frames):
            raise MMH3ResourceError("Custom H3 frames must use the 17n+5 grid")

    return GenerationSettings(
        geometry_mode,
        timeline_mode,
        audio_policy,
        width,
        height,
        frames,
        geometry,
    )


__all__ = [
    "BASE_SHORT_EDGE",
    "CANVAS_MULTIPLE",
    "GenerationSettings",
    "MAX_PIXELS",
    "adapt_h3_canvas",
    "build_generation_settings",
]
