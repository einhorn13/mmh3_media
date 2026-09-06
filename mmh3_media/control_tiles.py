from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .control_provider import H3ControlApplyPlan
from .errors import MMH3ResourceError
from .spatial_tiles import SpatialTile, SpatialTilePlan
from .util import deep_copy_json


H3_CONTROL_TILE_CONTRACT = "mmh3_h3_control_tile_adapter_v1"


@dataclass(frozen=True)
class H3TileControlInputs:
    control_video: torch.Tensor | None
    mask: torch.Tensor | None
    source_video: torch.Tensor | None
    tile: SpatialTile

    def to_dict(self) -> dict[str, Any]:
        rect = self.tile.sample_rect
        return {
            "tile_index": self.tile.index,
            "sample_rect": rect.to_dict(),
            "frames": _timeline(self.control_video, self.mask, self.source_video),
            "has_control_video": self.control_video is not None,
            "has_mask": self.mask is not None,
            "has_source_video": self.source_video is not None,
            "spatial_resize": False,
        }


def _timeline(*values: torch.Tensor | None) -> int:
    counts = {int(value.shape[0]) for value in values if value is not None}
    if not counts:
        return 0
    if len(counts) != 1:
        raise MMH3ResourceError("Tile control inputs have different timeline lengths")
    return counts.pop()


def _validate_full_canvas(
    value: torch.Tensor | None,
    *,
    label: str,
    plan: SpatialTilePlan,
    channels: bool,
) -> None:
    if value is None:
        return
    expected_ndim = 4 if channels else 3
    if not isinstance(value, torch.Tensor) or value.ndim != expected_ndim:
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise MMH3ResourceError(f"Tile {label} must be a full-canvas tensor with ndim={expected_ndim}; got {shape}")
    if int(value.shape[0]) != plan.frames:
        raise MMH3ResourceError(
            f"Tile {label} timeline must have {plan.frames} frames after alignment; got {value.shape[0]}"
        )
    if (int(value.shape[1]), int(value.shape[2])) != (plan.target_height, plan.target_width):
        raise MMH3ResourceError(
            f"Tile {label} must match full canvas {plan.target_width}x{plan.target_height}; "
            f"got {value.shape[2]}x{value.shape[1]}"
        )


def crop_control_inputs_for_tile(
    plan: SpatialTilePlan,
    tile: SpatialTile,
    *,
    control_video: torch.Tensor | None,
    mask: torch.Tensor | None,
    source_video: torch.Tensor | None,
) -> H3TileControlInputs:
    """Crop full-canvas F16 inputs to the exact sampler input rectangle.

    The complete timeline is retained for every tile. No spatial resize,
    temporal alignment or ownership decision is performed here.
    """
    if tile not in plan.tiles:
        raise MMH3ResourceError("Control tile does not belong to the supplied spatial plan")
    _validate_full_canvas(control_video, label="control_video", plan=plan, channels=True)
    _validate_full_canvas(mask, label="mask", plan=plan, channels=False)
    _validate_full_canvas(source_video, label="source_video", plan=plan, channels=True)
    if (mask is None) != (source_video is None):
        raise MMH3ResourceError("Tile inpaint requires source_video and mask together")
    _timeline(control_video, mask, source_video)
    rect = tile.sample_rect

    def crop(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value[:, rect.y : rect.bottom, rect.x : rect.right, ...]

    return H3TileControlInputs(crop(control_video), crop(mask), crop(source_video), tile)


def apply_control_to_conditioning(
    positive: Any,
    control_net: Any,
    vae: Any,
    plan: H3ControlApplyPlan,
    tile_inputs: H3TileControlInputs,
) -> Any:
    """Attach a fresh, tile-local ControlNet hint while preserving chains."""
    if plan.pass_through:
        return positive
    if control_net is None or vae is None:
        raise MMH3ResourceError("Active tile control requires both control_net and VAE")
    if not isinstance(positive, (list, tuple)):
        raise MMH3ResourceError("Tile control positive conditioning must be a sequence")
    hint = tile_inputs.control_video
    if hint is None and tile_inputs.mask is None:
        raise MMH3ResourceError("Active tile control has no tile-local control input")
    required = ("copy", "set_cond_hint")
    if any(not callable(getattr(control_net, name, None)) for name in required):
        raise MMH3ResourceError("Tile control_net does not expose the standard ControlNet copy/hint API")
    control_hint = hint.movedim(-1, 1) if hint is not None else None
    source_hint = (
        tile_inputs.source_video.movedim(-1, 1)
        if tile_inputs.source_video is not None
        else None
    )
    cache: dict[Any, Any] = {}
    output = []
    for item in positive:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
            raise MMH3ResourceError("Tile control conditioning item must be [embedding, metadata]")
        metadata = item[1].copy()
        previous = metadata.get("control")
        if previous in cache:
            control = cache[previous]
        else:
            control = control_net.copy().set_cond_hint(
                control_hint,
                plan.config.strength,
                (plan.config.start_percent, plan.config.end_percent),
                vae=vae,
            )
            if tile_inputs.mask is not None:
                set_inpaint = getattr(control, "set_inpaint", None)
                if not callable(set_inpaint):
                    raise MMH3ResourceError("Selected tile ControlNet does not support MiniMax H3 inpaint")
                set_inpaint(tile_inputs.mask, source_hint)
            set_previous = getattr(control, "set_previous_controlnet", None)
            if not callable(set_previous):
                raise MMH3ResourceError("Tile ControlNet does not support conditioning chains")
            set_previous(previous)
            cache[previous] = control
        metadata["control"] = control
        metadata["control_apply_to_uncond"] = True
        output.append([item[0], metadata])
    return output


def clone_guider_with_conditioning(
    guider: Any,
    positive: Any,
    *,
    convert_conditioning: Callable[[Any], list[dict[str, Any]]],
) -> Any:
    """Shallow-clone a Comfy guider and replace only its positive conditioning."""
    original = getattr(guider, "original_conds", None)
    if not isinstance(original, dict) or "positive" not in original:
        raise MMH3ResourceError("Tile control requires a guider with positive conditioning")
    converted = convert_conditioning(positive)
    if not isinstance(converted, list):
        raise MMH3ResourceError("Conditioning converter did not return a list")
    clone = copy.copy(guider)
    clone.original_conds = {
        key: [item.copy() for item in values]
        for key, values in original.items()
    }
    clone.original_conds["positive"] = converted
    return clone


def build_tiled_control_process_info(
    apply_plan: H3ControlApplyPlan,
    spatial_plan: SpatialTilePlan,
) -> dict[str, Any]:
    if (apply_plan.target_width, apply_plan.target_height, apply_plan.target_frames) != (
        spatial_plan.target_width,
        spatial_plan.target_height,
        spatial_plan.frames,
    ):
        raise MMH3ResourceError("Control preflight geometry does not match the spatial tile plan")
    info = apply_plan.to_dict()
    info["application_scope"] = "per_spatial_sample_tile"
    info["tile_adapter"] = {
        "contract": H3_CONTROL_TILE_CONTRACT,
        "version": 1,
        "spatial_plan": deep_copy_json(spatial_plan.to_dict()),
        "crop_basis": "sample_rect",
        "timeline_per_tile": "complete",
        "spatial_resize": False,
        "sampled_audio_controlled": False,
        "tiles": [
            {
                "tile_index": tile.index,
                "sample_rect": tile.sample_rect.to_dict(),
                "write_rect": tile.write_rect.to_dict(),
            }
            for tile in spatial_plan.tiles
        ],
    }
    return info
