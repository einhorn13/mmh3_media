from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from .errors import MMH3ResourceError
from .util import deep_copy_json


def build_long_video_upscale_settings(
    *,
    source_id: str,
    source_width: int,
    source_height: int,
    fps: float,
    chunk_plan: Mapping[str, Any],
    geometry_mode: str,
    scale: float,
    target_width: int,
    target_height: int,
    target_megapixels: float,
    align: int,
    optimization_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if chunk_plan.get("contract") != "mmh3_long_video_chunk_plan_v1" or str(chunk_plan.get("source_id")) != str(source_id):
        raise MMH3ResourceError("Upscale settings require a matching long-video chunk plan")
    if float(fps) != 24.0:
        raise MMH3ResourceError("H3 long-video upscale currently requires 24 FPS input")
    width, height, align = int(source_width), int(source_height), int(align)
    if width < 1 or height < 1 or align < 32 or align % 32:
        raise MMH3ResourceError("Upscale source geometry/align is invalid")

    def aligned(value: float) -> int:
        return max(align, int(round(value / align)) * align)

    if geometry_mode == "scale":
        factor = float(scale)
        if not math.isfinite(factor) or not 1.0 <= factor <= 4.0:
            raise MMH3ResourceError("Upscale scale must be within 1–4x")
        out_width, out_height = aligned(width * factor), aligned(height * factor)
    elif geometry_mode == "target_megapixels":
        megapixels = float(target_megapixels)
        if not math.isfinite(megapixels) or not 0.1 <= megapixels <= 8.0:
            raise MMH3ResourceError("target_megapixels must be within 0.1–8.0")
        area = megapixels * 1024.0 * 1024.0
        aspect = width / height
        out_width, out_height = aligned(math.sqrt(area * aspect)), aligned(math.sqrt(area / aspect))
    elif geometry_mode == "target_dimensions":
        out_width, out_height = int(target_width), int(target_height)
        if out_width < width or out_height < height or out_width > 4096 or out_height > 4096:
            raise MMH3ResourceError("Explicit upscale target must not downscale and must stay within 4096px")
        if out_width % align or out_height % align:
            raise MMH3ResourceError("Explicit upscale target is not aligned")
    else:
        raise MMH3ResourceError(f"Unknown upscale geometry_mode {geometry_mode!r}")
    if out_width / width > 4.0 or out_height / height > 4.0:
        raise MMH3ResourceError("Upscale target exceeds the 4x spatial limit")
    settings = {
        "contract": "mmh3_long_video_upscale_settings_v1",
        "source_id": str(source_id),
        "chunk_plan_contract": chunk_plan["contract"],
        "source": {"width": width, "height": height, "fps": 24.0},
        "target": {"width": out_width, "height": out_height, "align": align},
        "geometry_mode": geometry_mode,
        "optimization_profile": deep_copy_json(dict(optimization_profile or {})),
        "audio_policy": "exact_source_owner",
        "assembly_policy": "bounded_streaming_ownership",
    }
    body = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    settings["settings_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return settings


def validate_upscale_chunk_artifact(settings: Mapping[str, Any], process_info: Mapping[str, Any]) -> None:
    if settings.get("contract") != "mmh3_long_video_upscale_settings_v1":
        raise MMH3ResourceError("Unknown long-video upscale settings contract")
    target = process_info.get("geometry", {}).get("target", {}) if isinstance(process_info.get("geometry"), Mapping) else {}
    expected = settings["target"]
    if int(target.get("width", 0)) != int(expected["width"]) or int(target.get("height", 0)) != int(expected["height"]):
        raise MMH3ResourceError("Upscale chunk target geometry differs from the ledger settings")
    actual_profile = process_info.get("refine", {}).get("execution_profile") if isinstance(process_info.get("refine"), Mapping) else None
    expected_profile = settings.get("optimization_profile") or None
    if expected_profile is not None and actual_profile != expected_profile:
        raise MMH3ResourceError("Upscale chunk optimization profile differs from the ledger settings")


__all__ = ["build_long_video_upscale_settings", "validate_upscale_chunk_artifact"]
