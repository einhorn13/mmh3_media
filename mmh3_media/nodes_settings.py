from __future__ import annotations

import json
import math

from .generation_settings import build_generation_settings
from .node_support import CATEGORY, MMH3, _packet, inspect_packet, io
from .reference_sizing import (
    DEFAULT_REFERENCE_SHORT_EDGE,
    MAX_REFERENCE_SHORT_EDGE,
    REFERENCE_IMAGE_SIZE_MODES,
    normalize_reference_image_sizing,
)


_RATIOS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "21:9": (21, 9),
}


def _aligned_canvas(ratio: str, megapixels: float) -> tuple[int, int]:
    width_ratio, height_ratio = _RATIOS[ratio]
    total_pixels = float(megapixels) * 1024 * 1024
    scale = math.sqrt(total_pixels / (width_ratio * height_ratio))
    return (
        max(32, round(width_ratio * scale / 32) * 32),
        max(32, round(height_ratio * scale / 32) * 32),
    )


def _duration_frames(seconds: float) -> int:
    target = max(5.0, float(seconds) * 24.0)
    return max(5, 17 * max(0, round((target - 5) / 17)) + 5)


class MMH3H3GenerationSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        resolution_options = [io.DynamicCombo.Option("Source", [])]
        resolution_options.extend(
            io.DynamicCombo.Option(
                ratio,
                [
                    io.Float.Input(
                        "megapixels",
                        display_name="Resolution (MP)",
                        default=0.98,
                        min=0.1,
                        max=16.0,
                        step=0.05,
                        tooltip="Target pixel count; final width and height are aligned to 32 pixels.",
                    )
                ],
            )
            for ratio in _RATIOS
        )
        resolution_options.append(
            io.DynamicCombo.Option(
                "Custom",
                [
                    io.Int.Input("width", display_name="Width", default=1344, min=32, max=4096, step=32),
                    io.Int.Input("height", display_name="Height", default=768, min=32, max=4096, step=32),
                ],
            )
        )
        return io.Schema(
            node_id="MMH3H3GenerationSettings",
            display_name="H3 Video Settings",
            category=CATEGORY,
            description="Set output aspect ratio, resolution and video length with H3-safe alignment.",
            inputs=[
                MMH3.Input("packet"),
                io.DynamicCombo.Input(
                    "resolution",
                    display_name="Aspect ratio",
                    options=resolution_options,
                    tooltip="Source keeps the original dimensions (aligned to 32), without resizing to a base resolution. Required for direct latent continuation.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    display_name="Video length (seconds)",
                    default=3.0,
                    min=0.21,
                    max=150.0,
                    step=0.1,
                    tooltip="Rounded to the nearest legal H3 frame count at 24 FPS.",
                ),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("frames"),
                io.String.Output("settings_json", display_name="Settings JSON"),
            ],
        )

    @classmethod
    def execute(cls, packet, resolution: dict | str, duration_seconds: float) -> io.NodeOutput:
        value = _packet(packet)
        selected = resolution if isinstance(resolution, dict) else {"resolution": resolution}
        mode = str(selected.get("resolution") or "Source")
        frames = _duration_frames(duration_seconds)
        if mode == "Source":
            settings = build_generation_settings(
                inspect_packet(value).get("geometry"),
                geometry_mode="source",
                timeline_mode="custom",
                custom_frames=frames,
            )
        else:
            if mode == "Custom":
                width, height = int(selected.get("width", 1344)), int(selected.get("height", 768))
            else:
                width, height = _aligned_canvas(mode, float(selected.get("megapixels", 0.98)))
            settings = build_generation_settings(
                inspect_packet(value).get("geometry"),
                geometry_mode="custom",
                custom_width=width,
                custom_height=height,
                timeline_mode="custom",
                custom_frames=frames,
            )
        return io.NodeOutput(
            value,
            settings.width_override,
            settings.height_override,
            settings.frames_override,
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
        )


class MMH3H3ReferenceImageSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3ReferenceImageSettings",
            display_name="H3 Reference Image Size",
            category=CATEGORY,
            description="Choose native match/max sizing or cap reference images by their short edge before Ref2VA conditioning.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input(
                    "mode",
                    display_name="Sizing mode",
                    options=list(REFERENCE_IMAGE_SIZE_MODES),
                    default="match",
                    tooltip="match uses target pixel area; max uses native H3's 2048px short-edge cap; custom uses the pixel value below.",
                ),
                io.Int.Input(
                    "short_edge",
                    display_name="Custom short edge (px)",
                    default=DEFAULT_REFERENCE_SHORT_EDGE,
                    min=32,
                    max=MAX_REFERENCE_SHORT_EDGE,
                    step=32,
                    tooltip="Used only in custom mode. Preserves aspect ratio and never enlarges a smaller source image.",
                ),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("settings_json")],
        )

    @classmethod
    def execute(cls, packet, mode: str, short_edge: int) -> io.NodeOutput:
        value = _packet(packet)
        resolved_mode, resolved_edge = normalize_reference_image_sizing(mode, short_edge)
        settings = {"mode": resolved_mode, "short_edge": resolved_edge}
        out = value.edit_metadata(
            merge_patch_json={"extensions": {"minimax_h3": {"reference_image_sizing": settings}}}
        )
        return io.NodeOutput(out, json.dumps(settings, ensure_ascii=False, indent=2))


__all__ = [
    "MMH3H3GenerationSettings",
    "MMH3H3ReferenceImageSettings",
    "_aligned_canvas",
    "_duration_frames",
]
