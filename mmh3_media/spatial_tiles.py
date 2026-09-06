from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as torch_functional

from .errors import MMH3ResourceError
from .util import deep_copy_json


TILE_TRAVERSALS = ("row_major", "snake")
TILE_OVERLAP_MODES = ("context_only", "reprocess")
TILE_BLEND_MODES = ("hard", "linear", "half_cosine")
TILE_CONTEXT_SOURCES = ("original", "composited")
TILE_MASK_RESIZE_MODES = ("nearest", "bilinear")


@dataclass(frozen=True)
class TileRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersects(self, other: "TileRect") -> bool:
        return self.x < other.right and other.x < self.right and self.y < other.bottom and other.y < self.bottom

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class SpatialTile:
    index: int
    row: int
    column: int
    write_rect: TileRect
    sample_rect: TileRect
    overlaps_previous: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "row": self.row,
            "column": self.column,
            "write_rect": self.write_rect.to_dict(),
            "sample_rect": self.sample_rect.to_dict(),
            "overlaps_previous": list(self.overlaps_previous),
        }


@dataclass(frozen=True)
class SpatialTilePlan:
    target_width: int
    target_height: int
    frames: int
    tile_width: int
    tile_height: int
    overlap: int
    context_padding: int
    align: int
    traversal: str
    overlap_mode: str
    blend_mode: str
    context_source: str
    element_bytes: int
    tiles: tuple[SpatialTile, ...]

    def summary(self) -> str:
        return (
            f"READY · spatial tile plan · {len(self.tiles)} tiles · "
            f"{self.tile_width}x{self.tile_height} overlap={self.overlap} "
            f"padding={self.context_padding} · audio=external"
        )

    def to_dict(self) -> dict[str, Any]:
        target_area = self.target_width * self.target_height
        max_sample_area = max(tile.sample_rect.area for tile in self.tiles)
        total_sample_area = sum(tile.sample_rect.area for tile in self.tiles)
        total_write_area = sum(tile.write_rect.area for tile in self.tiles)
        channels = 3
        target_elements = target_area * self.frames * channels
        max_sample_elements = max_sample_area * self.frames * channels
        has_overlap = any(tile.overlaps_previous for tile in self.tiles)
        has_composited_context = len(self.tiles) > 1 and self.context_source == "composited" and self.context_padding > 0
        x_positions = sorted({tile.write_rect.x for tile in self.tiles})
        y_positions = sorted({tile.write_rect.y for tile in self.tiles})
        actual_x = [self.tile_width - (right - left) for left, right in zip(x_positions, x_positions[1:])]
        actual_y = [self.tile_height - (bottom - top) for top, bottom in zip(y_positions, y_positions[1:])]
        overlap_x = {"min": min(actual_x), "max": max(actual_x)} if actual_x else {"min": 0, "max": 0}
        overlap_y = {"min": min(actual_y), "max": max(actual_y)} if actual_y else {"min": 0, "max": 0}
        return {
            "version": 1,
            "contract": "mmh3_spatial_tile_plan_v1",
            "scope": "spatial_video_geometry_and_composite_policy",
            "target": {
                "width": self.target_width,
                "height": self.target_height,
                "frames": self.frames,
            },
            "tile": {
                "width": self.tile_width,
                "height": self.tile_height,
                "requested_overlap": self.overlap,
                "actual_overlap_x": overlap_x,
                "actual_overlap_y": overlap_y,
                "boundary_overlap_adjusted": any(value != self.overlap for value in (*actual_x, *actual_y)),
                "context_padding": self.context_padding,
                "align": self.align,
            },
            "traversal": self.traversal,
            "ownership": {
                "overlap_mode": self.overlap_mode,
                "prior_overlap_owner": "earlier_tile" if self.overlap_mode == "context_only" else "later_tile_after_blend",
                "blend_mode": self.blend_mode,
                "blend_owner": "pixel_compositor",
                "context_padding_owner": "sampler_input_only",
                "context_source": self.context_source,
                "sequential_required": has_overlap or has_composited_context,
                "traversal_affects_result": has_overlap or has_composited_context,
            },
            "av": {
                "video": "each spatial tile carries the complete frame timeline",
                "image_batch_semantics": "timeline_not_independent_tile_batch",
                "audio_action": "none",
                "sampled_audio_authoritative": False,
                "final_audio_owner": "consumer",
            },
            "sampler_boundary": {
                "owner": "consumer",
                "sampler_contract_included": False,
                "conditioning_contract_included": False,
                "vae_contract_included": False,
            },
            "memory_accounting": {
                "scope": "single_decoded_rgb_video_tensor_sizes_only",
                "element_bytes": self.element_bytes,
                "channels": channels,
                "target_spatial_pixels": target_area,
                "max_sample_spatial_pixels": max_sample_area,
                "total_sample_spatial_pixels": total_sample_area,
                "duplicate_write_spatial_pixels": total_write_area - target_area,
                "max_sample_to_target_area_ratio": max_sample_area / target_area,
                "target_video_tensor_bytes": target_elements * self.element_bytes,
                "max_sample_video_tensor_bytes": max_sample_elements * self.element_bytes,
                "peak_vram_estimated": False,
                "full_target_refine_peak_reduction_claimed": False,
                "excluded": [
                    "model_weights",
                    "conditioning",
                    "VAE_working_set",
                    "sampler_activations",
                    "joint_AV_latent",
                    "full_canvas_compositor_and_framework_cache",
                ],
            },
            "tiles": [tile.to_dict() for tile in self.tiles],
        }


@dataclass(frozen=True)
class SpatialTileRunResult:
    video: torch.Tensor
    report: dict[str, Any]

    def summary(self) -> str:
        return (
            f"READY · spatial tile run · {self.report['observed']['tiles_processed']} tiles · "
            f"context={self.report['context_source']} · audio=external"
        )


def _positions(total: int, tile: int, overlap: int) -> list[int]:
    if tile == total:
        return [0]
    stride = tile - overlap
    last = total - tile
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def _expanded(rect: TileRect, padding: int, width: int, height: int) -> TileRect:
    x = max(0, rect.x - padding)
    y = max(0, rect.y - padding)
    right = min(width, rect.right + padding)
    bottom = min(height, rect.bottom + padding)
    return TileRect(x, y, right - x, bottom - y)


def plan_spatial_tiles(
    *,
    target_width: int,
    target_height: int,
    frames: int,
    tile_width: int,
    tile_height: int,
    overlap: int = 64,
    context_padding: int = 64,
    align: int = 32,
    traversal: str = "snake",
    overlap_mode: str = "context_only",
    blend_mode: str = "hard",
    context_source: str = "composited",
    element_bytes: int = 4,
) -> SpatialTilePlan:
    values = {
        "target_width": int(target_width),
        "target_height": int(target_height),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
    }
    align = int(align)
    overlap = int(overlap)
    context_padding = int(context_padding)
    frames = int(frames)
    element_bytes = int(element_bytes)
    if align < 32 or align % 32:
        raise MMH3ResourceError("Spatial tile align must be a positive multiple of 32")
    if any(value <= 0 or value % align for value in values.values()):
        raise MMH3ResourceError(f"Spatial tile target and tile dimensions must be positive multiples of align={align}")
    if values["tile_width"] > values["target_width"] or values["tile_height"] > values["target_height"]:
        raise MMH3ResourceError("Spatial tile dimensions cannot exceed the target canvas")
    if overlap < 0 or overlap % align:
        raise MMH3ResourceError(f"Spatial tile overlap must be a non-negative multiple of align={align}")
    if overlap >= min(values["tile_width"], values["tile_height"]):
        raise MMH3ResourceError("Spatial tile overlap must be smaller than both tile dimensions")
    if context_padding < 0 or context_padding % align:
        raise MMH3ResourceError(f"Spatial tile context padding must be a non-negative multiple of align={align}")
    if frames < 5:
        raise MMH3ResourceError("H3 spatial tile planning requires at least 5 timeline frames")
    if traversal not in TILE_TRAVERSALS:
        raise MMH3ResourceError(f"Unsupported spatial tile traversal {traversal!r}")
    if overlap_mode not in TILE_OVERLAP_MODES:
        raise MMH3ResourceError(f"Unsupported spatial tile overlap mode {overlap_mode!r}")
    if blend_mode not in TILE_BLEND_MODES:
        raise MMH3ResourceError(f"Unsupported spatial tile blend mode {blend_mode!r}")
    if context_source not in TILE_CONTEXT_SOURCES:
        raise MMH3ResourceError(f"Unsupported spatial tile context source {context_source!r}")
    if element_bytes not in {1, 2, 4, 8}:
        raise MMH3ResourceError("Spatial tile element_bytes must be one of 1, 2, 4 or 8")

    xs = _positions(values["target_width"], values["tile_width"], overlap)
    ys = _positions(values["target_height"], values["tile_height"], overlap)
    actual_overlap = [values["tile_width"] - (right - left) for left, right in zip(xs, xs[1:])]
    actual_overlap.extend(values["tile_height"] - (bottom - top) for top, bottom in zip(ys, ys[1:]))
    has_actual_overlap = any(value > 0 for value in actual_overlap)
    if overlap_mode == "context_only" and blend_mode != "hard":
        raise MMH3ResourceError("Context-only overlap requires hard exclusive ownership")
    if not has_actual_overlap and blend_mode != "hard":
        raise MMH3ResourceError("Spatial tile blending requires an actual overlap; use blend_mode='hard'")
    if has_actual_overlap and overlap_mode == "reprocess" and blend_mode == "hard":
        raise MMH3ResourceError("Reprocessed overlap requires linear or half_cosine blending")
    coordinates: list[tuple[int, int, int, int]] = []
    for row, y in enumerate(ys):
        columns = list(enumerate(xs))
        if traversal == "snake" and row % 2:
            columns.reverse()
        coordinates.extend((row, column, x, y) for column, x in columns)

    tiles: list[SpatialTile] = []
    for index, (row, column, x, y) in enumerate(coordinates):
        write = TileRect(x, y, values["tile_width"], values["tile_height"])
        sample = _expanded(write, context_padding, values["target_width"], values["target_height"])
        previous = tuple(tile.index for tile in tiles if write.intersects(tile.write_rect))
        tiles.append(SpatialTile(index, row, column, write, sample, previous))
    return SpatialTilePlan(
        values["target_width"],
        values["target_height"],
        frames,
        values["tile_width"],
        values["tile_height"],
        overlap,
        context_padding,
        align,
        traversal,
        overlap_mode,
        blend_mode,
        context_source,
        element_bytes,
        tuple(tiles),
    )


def _blend_ramp(length: int, *, mode: str, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    position = torch.linspace(0.0, 1.0, length, device=device, dtype=torch.float32)
    if mode == "linear":
        return position
    if mode == "half_cosine":
        return 0.5 - 0.5 * torch.cos(torch.pi * position)
    return torch.ones(length, device=device, dtype=torch.float32)


def _composite_mask(plan: SpatialTilePlan, tile: SpatialTile, claimed: torch.Tensor) -> torch.Tensor:
    rect = tile.write_rect
    claimed_view = claimed[rect.y : rect.bottom, rect.x : rect.right]
    owner = ~claimed_view if plan.overlap_mode == "context_only" else torch.ones_like(claimed_view)
    if plan.blend_mode == "hard":
        return owner.to(dtype=torch.float32)
    alpha = owner.to(dtype=torch.float32)
    for previous_index in tile.overlaps_previous:
        previous = plan.tiles[previous_index].write_rect
        left = max(rect.x, previous.x)
        top = max(rect.y, previous.y)
        right = min(rect.right, previous.right)
        bottom = min(rect.bottom, previous.bottom)
        overlap_width = right - left
        overlap_height = bottom - top
        if overlap_width <= 0 or overlap_height <= 0:
            continue
        previous_center_x = previous.x + previous.width / 2.0
        current_center_x = rect.x + rect.width / 2.0
        previous_center_y = previous.y + previous.height / 2.0
        current_center_y = rect.y + rect.height / 2.0
        if previous_center_x < current_center_x:
            start = 0
            available = rect.width - start
            width = min(overlap_width, available)
            if width > 0:
                ramp = _blend_ramp(width, mode=plan.blend_mode, device=alpha.device)
                alpha[:, start : start + width] *= ramp.unsqueeze(0)
        elif previous_center_x > current_center_x:
            end = rect.width
            width = min(overlap_width, end)
            if width > 0:
                ramp = torch.flip(_blend_ramp(width, mode=plan.blend_mode, device=alpha.device), dims=(0,))
                alpha[:, end - width : end] *= ramp.unsqueeze(0)
        if previous_center_y < current_center_y:
            start = 0
            available = rect.height - start
            height = min(overlap_height, available)
            if height > 0:
                ramp = _blend_ramp(height, mode=plan.blend_mode, device=alpha.device)
                alpha[start : start + height, :] *= ramp.unsqueeze(1)
        elif previous_center_y > current_center_y:
            end = rect.height
            height = min(overlap_height, end)
            if height > 0:
                ramp = torch.flip(_blend_ramp(height, mode=plan.blend_mode, device=alpha.device), dims=(0,))
                alpha[end - height : end, :] *= ramp.unsqueeze(1)
    return alpha


def _tile_run_report_from_plan(plan_report: dict[str, Any]) -> dict[str, Any]:
    target = plan_report["target"]
    memory = plan_report["memory_accounting"]
    max_elements = memory["max_sample_spatial_pixels"] * target["frames"] * 3
    return {
        "version": 1,
        "contract": "mmh3_spatial_tile_run_v1",
        "operation": "spatial_video_tile_callback_composite",
        "plan": plan_report,
        "context_source": plan_report["ownership"]["context_source"],
        "adapter_boundary": {
            "owner": "consumer_callback",
            "sampler_contract_recorded": False,
            "conditioning_contract_recorded": False,
            "VAE_contract_recorded": False,
        },
        "av": {
            "video": "complete timeline per spatial sample rectangle",
            "audio_action": "none",
            "final_audio_owner": "consumer",
        },
        "observed": {
            "tiles_processed": len(plan_report["tiles"]),
            "source_video_mutated": False,
            "output_shape": [target["frames"], target["height"], target["width"], 3],
            "max_sample_tensor_elements": max_elements,
            "max_result_tensor_elements": max_elements,
            "peak_vram_measured": False,
        },
    }


def run_spatial_video_tiles(
    video: torch.Tensor,
    plan: SpatialTilePlan,
    process_tile: Callable[[torch.Tensor, SpatialTile], torch.Tensor],
) -> SpatialTileRunResult:
    """Run a sampler-agnostic spatial crop/callback/composite loop.

    ``process_tile`` receives a private clone shaped ``[T,H,W,C]`` for the
    tile's sample rectangle and must return a tensor with the same shape. The
    callback owns every sampler, conditioning and VAE decision. Audio is not
    accepted or produced by this boundary.
    """
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise MMH3ResourceError("Spatial tile runner expects a [T,H,W,C] video tensor")
    if not isinstance(plan, SpatialTilePlan):
        raise MMH3ResourceError("Spatial tile runner expects a validated SpatialTilePlan")
    if not callable(process_tile):
        raise MMH3ResourceError("Spatial tile runner requires a callable process_tile adapter")
    frames, height, width, channels = (int(value) for value in video.shape)
    if (width, height, frames) != (plan.target_width, plan.target_height, plan.frames):
        raise MMH3ResourceError("Spatial tile runner video shape does not match the tile plan target")
    if channels != 3:
        raise MMH3ResourceError("Spatial tile runner requires exactly three RGB channels")
    if not video.is_floating_point():
        raise MMH3ResourceError("Spatial tile runner requires a floating-point video tensor for blending")
    if video.element_size() != plan.element_bytes:
        raise MMH3ResourceError("Spatial tile runner tensor element size does not match the plan memory accounting")

    # Only canvas and per-tile inputs are writable; source remains a borrowed view.
    source = video.detach()
    canvas = source.clone()
    claimed = torch.zeros((height, width), device=video.device, dtype=torch.bool)
    max_sample_elements = 0
    max_result_elements = 0
    for tile in plan.tiles:
        sample = tile.sample_rect
        context = source if plan.context_source == "original" else canvas
        tile_input = context[:, sample.y : sample.bottom, sample.x : sample.right, :].clone()
        tile_result = process_tile(tile_input, tile)
        if not isinstance(tile_result, torch.Tensor):
            raise MMH3ResourceError(f"Spatial tile {tile.index} adapter did not return a tensor")
        if tuple(tile_result.shape) != tuple(tile_input.shape):
            raise MMH3ResourceError(f"Spatial tile {tile.index} adapter changed the sample tensor shape")
        if tile_result.device != video.device or tile_result.dtype != video.dtype:
            raise MMH3ResourceError(f"Spatial tile {tile.index} adapter changed tensor device or dtype")
        write = tile.write_rect
        offset_x = write.x - sample.x
        offset_y = write.y - sample.y
        write_result = tile_result[
            :,
            offset_y : offset_y + write.height,
            offset_x : offset_x + write.width,
            :,
        ]
        alpha = _composite_mask(plan, tile, claimed).to(dtype=video.dtype)
        current = canvas[:, write.y : write.bottom, write.x : write.right, :]
        alpha = alpha.unsqueeze(0).unsqueeze(-1)
        current.copy_(write_result * alpha + current * (1.0 - alpha))
        claimed[write.y : write.bottom, write.x : write.right] = True
        max_sample_elements = max(max_sample_elements, tile_input.numel())
        max_result_elements = max(max_result_elements, tile_result.numel())

    report = _tile_run_report_from_plan(plan.to_dict())
    if report["observed"]["max_sample_tensor_elements"] != max_sample_elements:
        raise MMH3ResourceError("Spatial tile runner observed sample accounting differs from its plan")
    if report["observed"]["max_result_tensor_elements"] != max_result_elements:
        raise MMH3ResourceError("Spatial tile runner observed result accounting differs from its plan")
    return SpatialTileRunResult(canvas, report)


def prepare_spatial_region_mask(
    region_mask: torch.Tensor,
    *,
    target_width: int,
    target_height: int,
    resize_mode: str = "nearest",
) -> torch.Tensor:
    """Normalize one spatial region mask to ``[H,W]`` without timeline batching."""
    if not isinstance(region_mask, torch.Tensor) or region_mask.ndim not in {2, 3}:
        raise MMH3ResourceError("Spatial region mask must be [H,W] or [1,H,W]")
    if region_mask.ndim == 3:
        if int(region_mask.shape[0]) != 1:
            raise MMH3ResourceError("Spatial region mask must have exactly one spatial plane, not a timeline batch")
        region_mask = region_mask[0]
    if resize_mode not in TILE_MASK_RESIZE_MODES:
        raise MMH3ResourceError(f"Unsupported spatial region mask resize mode {resize_mode!r}")
    if region_mask.dtype != torch.bool and not region_mask.is_floating_point():
        raise MMH3ResourceError("Spatial region mask must be boolean or floating point")
    mask = region_mask.detach().clone().to(dtype=torch.float32)
    if not torch.isfinite(mask).all():
        raise MMH3ResourceError("Spatial region mask contains non-finite values")
    if mask.numel() == 0 or float(mask.min()) < 0.0 or float(mask.max()) > 1.0:
        raise MMH3ResourceError("Spatial region mask values must be within [0,1]")
    target = (int(target_height), int(target_width))
    if target[0] <= 0 or target[1] <= 0:
        raise MMH3ResourceError("Spatial region mask target dimensions must be positive")
    if tuple(mask.shape) != target:
        options = {"size": target, "mode": resize_mode}
        if resize_mode == "bilinear":
            options["align_corners"] = False
        mask = torch_functional.interpolate(mask[None, None], **options)[0, 0]
    return mask


def run_masked_spatial_video_tiles(
    video: torch.Tensor,
    plan: SpatialTilePlan,
    region_mask: torch.Tensor,
    process_tile: Callable[[torch.Tensor, torch.Tensor, SpatialTile], torch.Tensor],
    *,
    mask_resize_mode: str = "nearest",
) -> SpatialTileRunResult:
    """Run spatial tiles with an explicit spatial-only callback and composite mask."""
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise MMH3ResourceError("Masked spatial tile runner expects a [T,H,W,C] video tensor")
    if not isinstance(plan, SpatialTilePlan):
        raise MMH3ResourceError("Masked spatial tile runner expects a validated SpatialTilePlan")
    if not callable(process_tile):
        raise MMH3ResourceError("Masked spatial tile runner requires a callable process_tile adapter")
    frames, height, width, channels = (int(value) for value in video.shape)
    if (width, height, frames) != (plan.target_width, plan.target_height, plan.frames):
        raise MMH3ResourceError("Masked spatial tile runner video shape does not match the tile plan target")
    if channels != 3 or not video.is_floating_point():
        raise MMH3ResourceError("Masked spatial tile runner requires floating-point RGB video")
    if video.element_size() != plan.element_bytes:
        raise MMH3ResourceError("Masked spatial tile runner tensor element size does not match the plan")

    source_mask_shape = list(region_mask.shape) if isinstance(region_mask, torch.Tensor) else []
    mask = prepare_spatial_region_mask(
        region_mask,
        target_width=width,
        target_height=height,
        resize_mode=mask_resize_mode,
    ).to(device=video.device)
    # The callback receives private tile clones, including in the masked path.
    source = video.detach()
    canvas = source.clone()
    claimed = torch.zeros((height, width), device=video.device, dtype=torch.bool)
    processed = 0
    skipped = 0
    max_sample_elements = 0
    max_result_elements = 0
    for tile in plan.tiles:
        write = tile.write_rect
        write_mask = mask[write.y : write.bottom, write.x : write.right]
        if not torch.any(write_mask > 0):
            skipped += 1
            continue
        sample = tile.sample_rect
        context = source if plan.context_source == "original" else canvas
        tile_input = context[:, sample.y : sample.bottom, sample.x : sample.right, :].clone()
        tile_mask = mask[sample.y : sample.bottom, sample.x : sample.right].clone()
        tile_result = process_tile(tile_input, tile_mask, tile)
        if not isinstance(tile_result, torch.Tensor) or tuple(tile_result.shape) != tuple(tile_input.shape):
            raise MMH3ResourceError(f"Masked spatial tile {tile.index} adapter changed the sample tensor contract")
        if tile_result.device != video.device or tile_result.dtype != video.dtype:
            raise MMH3ResourceError(f"Masked spatial tile {tile.index} adapter changed tensor device or dtype")
        offset_x = write.x - sample.x
        offset_y = write.y - sample.y
        write_result = tile_result[:, offset_y : offset_y + write.height, offset_x : offset_x + write.width, :]
        alpha = _composite_mask(plan, tile, claimed).to(dtype=video.dtype) * write_mask.to(dtype=video.dtype)
        current = canvas[:, write.y : write.bottom, write.x : write.right, :]
        alpha = alpha.unsqueeze(0).unsqueeze(-1)
        current.copy_(write_result * alpha + current * (1.0 - alpha))
        claimed_view = claimed[write.y : write.bottom, write.x : write.right]
        claimed_view.logical_or_(write_mask > 0)
        processed += 1
        max_sample_elements = max(max_sample_elements, tile_input.numel())
        max_result_elements = max(max_result_elements, tile_result.numel())

    report = _tile_run_report_from_plan(plan.to_dict())
    report["contract"] = "mmh3_spatial_masked_tile_run_v1"
    report["operation"] = "masked_spatial_video_tile_callback_composite"
    report["region_mask"] = {
        "source_shape": source_mask_shape,
        "target_shape": [height, width],
        "resize_mode": mask_resize_mode,
        "resized": source_mask_shape[-2:] != [height, width],
        "active_spatial_pixels": int(torch.count_nonzero(mask > 0).item()),
        "empty_write_tile_action": "skip_callback",
        "write_ownership": "mask_gates_composite_and_claimed_pixels",
        "timeline_semantics": "single_spatial_plane_shared_by_all_frames",
    }
    report["observed"].update(
        {
            "tiles_processed": processed,
            "tiles_skipped_empty_mask": skipped,
            "max_sample_tensor_elements": max_sample_elements,
            "max_result_tensor_elements": max_result_elements,
        }
    )
    return SpatialTileRunResult(canvas, report)


def validate_spatial_tile_plan_report(
    report: Mapping[str, Any],
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    frames: int | None = None,
) -> dict[str, Any]:
    value = deep_copy_json(dict(report))
    if value.get("version") != 1 or value.get("contract") != "mmh3_spatial_tile_plan_v1":
        raise MMH3ResourceError("Expected an MMH3 spatial tile plan report v1")
    if value.get("scope") != "spatial_video_geometry_and_composite_policy":
        raise MMH3ResourceError("Spatial tile report has an unsupported ownership scope")
    target = value.get("target")
    if not isinstance(target, dict):
        raise MMH3ResourceError("Spatial tile report is missing target geometry")
    expected = {"width": target_width, "height": target_height, "frames": frames}
    for key, expected_value in expected.items():
        if expected_value is not None and target.get(key) != int(expected_value):
            raise MMH3ResourceError(f"Spatial tile {key} does not match the owning process")
    tile = value.get("tile")
    memory = value.get("memory_accounting")
    if not isinstance(tile, dict) or not isinstance(memory, dict):
        raise MMH3ResourceError("Spatial tile report is missing tile or memory accounting")
    try:
        canonical = plan_spatial_tiles(
            target_width=target["width"],
            target_height=target["height"],
            frames=target["frames"],
            tile_width=tile["width"],
            tile_height=tile["height"],
            overlap=tile["requested_overlap"],
            context_padding=tile["context_padding"],
            align=tile["align"],
            traversal=value["traversal"],
            overlap_mode=value["ownership"]["overlap_mode"],
            blend_mode=value["ownership"]["blend_mode"],
            context_source=value["ownership"]["context_source"],
            element_bytes=memory["element_bytes"],
        ).to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise MMH3ResourceError("Spatial tile report is incomplete") from exc
    if value != canonical:
        raise MMH3ResourceError("Spatial tile report does not match its canonical geometry and ownership policy")
    return value


def validate_spatial_tile_run_report(
    report: Mapping[str, Any],
    *,
    plan: SpatialTilePlan | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = deep_copy_json(dict(report))
    if value.get("version") != 1 or value.get("contract") != "mmh3_spatial_tile_run_v1":
        raise MMH3ResourceError("Expected an MMH3 spatial tile run report v1")
    embedded_plan = validate_spatial_tile_plan_report(value.get("plan", {}))
    if plan is not None:
        expected_plan = plan.to_dict() if isinstance(plan, SpatialTilePlan) else validate_spatial_tile_plan_report(plan)
        if embedded_plan != expected_plan:
            raise MMH3ResourceError("Spatial tile run report does not match the owning tile plan")
    if value != _tile_run_report_from_plan(embedded_plan):
        raise MMH3ResourceError("Spatial tile run report does not match its canonical plan and observed accounting")
    return value


def validate_masked_spatial_tile_run_report(
    report: Mapping[str, Any],
    *,
    plan: SpatialTilePlan | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = deep_copy_json(dict(report))
    if value.get("version") != 1 or value.get("contract") != "mmh3_spatial_masked_tile_run_v1":
        raise MMH3ResourceError("Expected an MMH3 masked spatial tile run report v1")
    if value.get("operation") != "masked_spatial_video_tile_callback_composite":
        raise MMH3ResourceError("Masked spatial tile run report has an unsupported operation")
    embedded_plan = validate_spatial_tile_plan_report(value.get("plan", {}))
    if plan is not None:
        expected_plan = plan.to_dict() if isinstance(plan, SpatialTilePlan) else validate_spatial_tile_plan_report(plan)
        if embedded_plan != expected_plan:
            raise MMH3ResourceError("Masked spatial tile run report does not match the owning tile plan")
    mask = value.get("region_mask")
    observed = value.get("observed")
    if not isinstance(mask, dict) or not isinstance(observed, dict):
        raise MMH3ResourceError("Masked spatial tile run report is missing mask or observed facts")
    expected_mask_keys = {
        "source_shape",
        "target_shape",
        "resize_mode",
        "resized",
        "active_spatial_pixels",
        "empty_write_tile_action",
        "write_ownership",
        "timeline_semantics",
    }
    if set(mask) != expected_mask_keys:
        raise MMH3ResourceError("Masked spatial tile run report has a non-canonical region-mask schema")
    target = embedded_plan["target"]
    target_shape = [target["height"], target["width"]]
    source_shape = mask["source_shape"]
    if (
        not isinstance(source_shape, list)
        or len(source_shape) not in {2, 3}
        or any(not isinstance(item, int) or item <= 0 for item in source_shape)
        or (len(source_shape) == 3 and source_shape[0] != 1)
    ):
        raise MMH3ResourceError("Masked spatial tile run report has an invalid source mask shape")
    if mask["target_shape"] != target_shape or mask["resize_mode"] not in TILE_MASK_RESIZE_MODES:
        raise MMH3ResourceError("Masked spatial tile run report has invalid target or resize facts")
    if mask["resized"] is not (source_shape[-2:] != target_shape):
        raise MMH3ResourceError("Masked spatial tile run report resize fact is inconsistent")
    active_pixels = mask["active_spatial_pixels"]
    if not isinstance(active_pixels, int) or not 0 <= active_pixels <= target["width"] * target["height"]:
        raise MMH3ResourceError("Masked spatial tile run report has invalid active pixel accounting")
    if mask["empty_write_tile_action"] != "skip_callback":
        raise MMH3ResourceError("Masked spatial tile run report has unsupported empty-tile behavior")
    if mask["write_ownership"] != "mask_gates_composite_and_claimed_pixels":
        raise MMH3ResourceError("Masked spatial tile run report has unsupported mask ownership")
    if mask["timeline_semantics"] != "single_spatial_plane_shared_by_all_frames":
        raise MMH3ResourceError("Masked spatial tile run report must not treat mask planes as a timeline batch")
    processed = observed.get("tiles_processed")
    skipped = observed.get("tiles_skipped_empty_mask")
    sample_elements = observed.get("max_sample_tensor_elements")
    result_elements = observed.get("max_result_tensor_elements")
    if not all(isinstance(item, int) and item >= 0 for item in (processed, skipped, sample_elements, result_elements)):
        raise MMH3ResourceError("Masked spatial tile run report has invalid observed accounting")
    if processed + skipped != len(embedded_plan["tiles"]):
        raise MMH3ResourceError("Masked spatial tile processed/skipped counts do not cover the plan")
    planned_max = embedded_plan["memory_accounting"]["max_sample_spatial_pixels"] * target["frames"] * 3
    if sample_elements != result_elements or sample_elements > planned_max or (processed == 0) != (sample_elements == 0):
        raise MMH3ResourceError("Masked spatial tile tensor accounting is inconsistent")
    canonical = _tile_run_report_from_plan(embedded_plan)
    canonical["contract"] = "mmh3_spatial_masked_tile_run_v1"
    canonical["operation"] = "masked_spatial_video_tile_callback_composite"
    canonical["region_mask"] = deep_copy_json(mask)
    canonical["observed"].update(
        {
            "tiles_processed": processed,
            "tiles_skipped_empty_mask": skipped,
            "max_sample_tensor_elements": sample_elements,
            "max_result_tensor_elements": result_elements,
        }
    )
    if value != canonical:
        raise MMH3ResourceError("Masked spatial tile run report does not match canonical ownership and accounting")
    return value
