from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import MMH3ResourceError
from .resource_model import resource_facts


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
H3_FPS = 24.0
AUDIO_LATENT_FPS = 40.0


@dataclass(frozen=True)
class ReferenceCostItem:
    resource_id: str
    kind: str
    order: int
    source_width: int | None
    source_height: int | None
    source_frames: int | None
    source_duration: float | None
    effective_width: int | None
    effective_height: int | None
    effective_frames: int | None
    video_latent_t: int
    video_rows: int
    audio_latent_t: int
    audio_rows: int
    estimate_complete: bool
    notes: tuple[str, ...] = ()

    @property
    def total_rows(self) -> int:
        return self.video_rows + self.audio_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "kind": self.kind,
            "order": self.order,
            "source": {
                "width": self.source_width,
                "height": self.source_height,
                "frames": self.source_frames,
                "duration": self.source_duration,
            },
            "effective": {
                "width": self.effective_width,
                "height": self.effective_height,
                "frames": self.effective_frames,
                "video_latent_t": self.video_latent_t,
                "audio_latent_t": self.audio_latent_t,
            },
            "rows": {"video": self.video_rows, "audio": self.audio_rows, "total": self.total_rows},
            "estimate_complete": self.estimate_complete,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ReferenceCostReport:
    ref_image_size: str
    target_width: int
    target_height: int
    target_frames: int
    target_rows: int
    items: tuple[ReferenceCostItem, ...]
    reference_rows: int
    ratio_to_target: float | None
    level: str
    diagnostics: tuple[dict[str, Any], ...]

    @property
    def complete(self) -> bool:
        return all(item.estimate_complete for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "minimax_h3_packed_rows.v1",
            "ref_image_size": self.ref_image_size,
            "target": {
                "width": self.target_width,
                "height": self.target_height,
                "frames": self.target_frames,
                "rows": self.target_rows,
            },
            "reference_rows": self.reference_rows,
            "ratio_to_target": self.ratio_to_target,
            "level": self.level,
            "complete": self.complete,
            "items": [item.to_dict() for item in self.items],
            "diagnostics": list(self.diagnostics),
        }

    def summary(self) -> str:
        ratio = "unknown" if self.ratio_to_target is None else f"{self.ratio_to_target:.3f}× target"
        confidence = "complete" if self.complete else "partial"
        return f"Reference packed rows: {self.reference_rows:,} · {ratio} · level={self.level} · estimate={confidence}"


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and math.isfinite(number) else None


def _dimensions(metadata: Mapping[str, Any]) -> tuple[int | None, int | None]:
    dimensions = metadata.get("dimensions")
    if isinstance(dimensions, (list, tuple)) and len(dimensions) >= 2:
        return _positive_int(dimensions[0]), _positive_int(dimensions[1])
    width, height = _positive_int(metadata.get("width")), _positive_int(metadata.get("height"))
    if width and height:
        return width, height
    shape = metadata.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 3:
        # IMAGE [B,H,W,C] and frame batches [T,H,W,C] share these positions.
        return _positive_int(shape[-2]), _positive_int(shape[-3])
    return None, None


def _media_timing(metadata: Mapping[str, Any], kind: str) -> tuple[int | None, float | None]:
    duration = _positive_float(metadata.get("duration"))
    frames = _positive_int(metadata.get("frame_count"))
    if frames is None:
        shape = metadata.get("shape")
        if kind == "video" and isinstance(shape, (list, tuple)) and len(shape) == 4:
            frames = _positive_int(shape[0])
    if duration is None:
        samples = _positive_int(metadata.get("samples"))
        sample_rate = _positive_int(metadata.get("sample_rate"))
        if samples and sample_rate:
            duration = samples / sample_rate
    return frames, duration


def align_canvas(value: float) -> int:
    return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def image_reference_canvas(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    ref_image_size: str,
) -> tuple[int, int]:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise MMH3ResourceError("Reference and target dimensions must be positive")
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((target_width * target_height) / (source_width * source_height)))
    elif ref_image_size == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(source_width, source_height))
    else:
        raise MMH3ResourceError("ref_image_size must be match or max")
    return align_canvas(source_width * scale), align_canvas(source_height * scale)


def video_reference_canvas(source_width: int, source_height: int) -> tuple[int, int]:
    if min(source_width, source_height) <= 0:
        raise MMH3ResourceError("Reference video dimensions must be positive")
    ratio = source_width / source_height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    canvas = (align_canvas(nominal_width), align_canvas(nominal_height))
    if source_width * source_height < canvas[0] * canvas[1]:
        return align_canvas(source_width), align_canvas(source_height)
    return canvas


def align_reference_video_frames(frame_count: int) -> int:
    if frame_count < 5:
        return frame_count
    while frame_count % 17 != 5:
        frame_count -= 1
    return frame_count


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _spatial_rows(width: int, height: int) -> int:
    # H3 VAE downsamples /16 and PackedLayout patchifies each latent frame /2 again.
    return (width // 32) * (height // 32)


def _estimate_item(
    resource: Any,
    *,
    target_width: int,
    target_height: int,
    target_frames: int,
    ref_image_size: str,
) -> ReferenceCostItem:
    resource_id = resource.resource_id if hasattr(resource, "resource_id") else str(resource["id"])
    kind = str(resource.kind if hasattr(resource, "kind") else resource.get("kind") or "")
    if kind not in ("image", "video", "audio"):
        raise MMH3ResourceError(f"Reference cost supports image, video, or audio resources; got {kind!r}")
    if isinstance(resource, Mapping) and isinstance(resource.get("descriptor"), Mapping):
        observed = resource["descriptor"]
        shape = observed.get("shape") if isinstance(observed.get("shape"), Mapping) else {}
        timing = observed.get("timing") if isinstance(observed.get("timing"), Mapping) else {}
        metadata = dict(shape)
        metadata.update(timing)
        tensor = observed.get("tensor") if isinstance(observed.get("tensor"), Mapping) else {}
        if "shape" in tensor:
            metadata["shape"] = tensor["shape"]
        order = int(resource.get("order") or 0)
    else:
        metadata = resource.facts if hasattr(resource, "facts") else resource_facts(resource)
        order = int(resource.order if hasattr(resource, "order") and resource.order is not None else resource.get("order", 0))
    width, height = _dimensions(metadata)
    frames, duration = _media_timing(metadata, kind)
    effective_width = effective_height = effective_frames = None
    vt = video_rows = audio_t = audio_rows = 0
    notes: list[str] = []
    complete = True

    if kind == "image":
        if width and height:
            effective_width, effective_height = image_reference_canvas(
                width, height, target_width, target_height, ref_image_size
            )
            vt = 1
            video_rows = _spatial_rows(effective_width, effective_height)
        else:
            complete = False
            notes.append("missing image dimensions in manifest metadata")
    elif kind == "video":
        if width and height:
            effective_width, effective_height = video_reference_canvas(width, height)
        else:
            complete = False
            notes.append("missing video dimensions in manifest metadata")
        if duration is not None:
            effective_frames = max(1, round(duration * H3_FPS))
        elif frames is not None:
            effective_frames = frames
            notes.append("duration missing; assuming stored frame_count is already 24 fps")
        else:
            complete = False
            notes.append("missing video duration/frame_count in manifest metadata")
        if effective_frames is not None:
            effective_frames = min(effective_frames, target_frames)
            if effective_frames < 5:
                complete = False
                notes.append("native H3 rejects reference videos shorter than 5 frames at 24 fps")
            else:
                effective_frames = align_reference_video_frames(effective_frames)
                vt = video_latent_t(effective_frames)
        if effective_width and effective_height and vt:
            video_rows = _spatial_rows(effective_width, effective_height) * vt
    elif kind == "audio":
        if duration is not None:
            audio_t = max(1, round(duration * AUDIO_LATENT_FPS))
            audio_rows = audio_t * 2
        else:
            complete = False
            notes.append("missing audio duration or samples/sample_rate in manifest metadata")

    return ReferenceCostItem(
        resource_id,
        kind,
        order,
        width,
        height,
        frames,
        duration,
        effective_width,
        effective_height,
        effective_frames,
        vt,
        video_rows,
        audio_t,
        audio_rows,
        complete,
        tuple(notes),
    )


def estimate_reference_cost(
    resources: Sequence[Any],
    *,
    target_width: int,
    target_height: int,
    target_frames: int,
    ref_image_size: str = "match",
) -> ReferenceCostReport:
    if ref_image_size not in ("match", "max"):
        raise MMH3ResourceError("ref_image_size must be match or max")
    if min(target_width, target_height, target_frames) <= 0:
        raise MMH3ResourceError("Target width, height and frames must be positive for reference cost estimation")
    items = tuple(
        _estimate_item(
            resource,
            target_width=target_width,
            target_height=target_height,
            target_frames=target_frames,
            ref_image_size=ref_image_size,
        )
        for resource in resources
    )
    aligned_target_frames = target_frames
    while aligned_target_frames % 17 != 5:
        aligned_target_frames += 1
    target_video_rows = _spatial_rows(target_width, target_height) * video_latent_t(aligned_target_frames)
    target_audio_rows = round((aligned_target_frames / H3_FPS) * AUDIO_LATENT_FPS) * 2
    target_rows = target_video_rows + target_audio_rows
    reference_rows = sum(item.total_rows for item in items)
    ratio = reference_rows / target_rows if target_rows else None
    level = "unknown"
    if ratio is not None:
        level = "low" if ratio <= 0.5 else "moderate" if ratio <= 1.0 else "high" if ratio <= 2.0 else "extreme"
    diagnostics: list[dict[str, Any]] = []
    incomplete = [item.resource_id for item in items if not item.estimate_complete]
    if incomplete:
        diagnostics.append(
            {
                "severity": "warning",
                "code": "reference_cost_partial",
                "message": f"Cost estimate is partial because {len(incomplete)} reference resource(s) lack geometry/timing metadata.",
                "resource_ids": incomplete,
            }
        )
    if level in ("high", "extreme"):
        diagnostics.append(
            {
                "severity": "warning",
                "code": "reference_attention_cost_high",
                "message": (
                    f"Estimated reference rows are {ratio:.2f}× the generated AV rows. "
                    "Reference rows remain in attention at every denoising step; reduce references or use match sizing."
                ),
                "resource_ids": [item.resource_id for item in items],
            }
        )
    if ref_image_size == "max" and any(item.kind == "image" for item in items):
        diagnostics.append(
            {
                "severity": "info",
                "code": "reference_image_max_cost",
                "message": "ref_image_size=max preserves up to a 2048px short edge and can be several times slower than match.",
                "resource_ids": [item.resource_id for item in items if item.kind == "image"],
            }
        )
    return ReferenceCostReport(
        ref_image_size,
        target_width,
        target_height,
        target_frames,
        target_rows,
        items,
        reference_rows,
        ratio,
        level,
        tuple(diagnostics),
    )
