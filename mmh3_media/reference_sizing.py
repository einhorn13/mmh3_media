from __future__ import annotations

from typing import Any, Mapping

from .errors import MMH3ResourceError


REFERENCE_IMAGE_SIZE_MODES = ("match", "max", "custom")
DEFAULT_REFERENCE_SHORT_EDGE = 768
MAX_REFERENCE_SHORT_EDGE = 2048


def normalize_reference_image_sizing(mode: str, short_edge: int) -> tuple[str, int]:
    mode = str(mode).strip().casefold()
    if mode not in REFERENCE_IMAGE_SIZE_MODES:
        raise MMH3ResourceError(f"Reference image sizing mode must be one of {', '.join(REFERENCE_IMAGE_SIZE_MODES)}")
    edge = int(short_edge)
    if edge < 32 or edge > MAX_REFERENCE_SHORT_EDGE or edge % 32:
        raise MMH3ResourceError("Custom reference short edge must be a multiple of 32 between 32 and 2048 pixels")
    return mode, edge


def reference_image_sizing_from_manifest(manifest: Mapping[str, Any]) -> tuple[str, int]:
    minimax = ((manifest.get("extensions") or {}).get("minimax_h3") or {})
    sizing = minimax.get("reference_image_sizing") or {}
    if not isinstance(sizing, Mapping):
        raise MMH3ResourceError("extensions.minimax_h3.reference_image_sizing must be an object")
    return normalize_reference_image_sizing(
        sizing.get("mode", "match"), sizing.get("short_edge", DEFAULT_REFERENCE_SHORT_EDGE)
    )


def native_reference_image_size(mode: str) -> str:
    if mode not in REFERENCE_IMAGE_SIZE_MODES:
        raise MMH3ResourceError(f"Unsupported reference image sizing mode {mode!r}")
    return "max" if mode == "custom" else mode


def reference_resize_dimensions(width: int, height: int, short_edge: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise MMH3ResourceError("Reference image dimensions must be positive")
    _, edge = normalize_reference_image_sizing("custom", short_edge)
    source_short = min(width, height)
    if source_short <= edge:
        return width, height
    scale = edge / source_short
    return (
        max(32, round(width * scale / 32) * 32),
        max(32, round(height * scale / 32) * 32),
    )

