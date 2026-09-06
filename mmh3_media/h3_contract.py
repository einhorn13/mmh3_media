from __future__ import annotations

from typing import Any, Mapping

from .constants import AUDIO_LATENT_FPS, AUDIO_SAMPLE_RATE, FPS, H3_LATENT_ORIGINS, H3_TEMPORAL_GRID
from .errors import MMH3ResourceError
from .h3 import H3LatentInfo, validate_h3_latent_origin
from .util import deep_copy_json

H3_LATENT_CONTRACT_VERSION = 2
H3_LATENT_LAYOUT = "joint_av"
_BINDING_STATES = ("bound", "stale", "unknown")


def build_h3_latent_contract(
    info: H3LatentInfo,
    *,
    origin: str = "unknown",
    geometry_binding: str = "bound",
    time_binding: str = "bound",
    absolute_start_frame: int | None = 0,
    noise_mask: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    origin = validate_h3_latent_origin(origin)
    if geometry_binding not in _BINDING_STATES or time_binding not in _BINDING_STATES:
        raise MMH3ResourceError("H3 geometry/time binding must be bound, stale, or unknown")
    if absolute_start_frame is not None and (not isinstance(absolute_start_frame, int) or absolute_start_frame < 0):
        raise MMH3ResourceError("H3 absolute_start_frame must be null or a non-negative integer")
    contract = {
        "contract_version": H3_LATENT_CONTRACT_VERSION,
        "model_family": "minimax_h3",
        "layout": H3_LATENT_LAYOUT,
        "origin": origin,
        "streams": {
            "video": {"shape": list(info.video_shape)},
            "audio": {"shape": list(info.audio_shape)},
        },
        "canvas": {"width": info.width, "height": info.height},
        "timeline": {
            "frames": info.frames,
            "fps": info.fps,
            "audio_sample_rate": info.audio_sample_rate,
            "audio_latent_rate": info.audio_latent_rate,
            "temporal_grid": H3_TEMPORAL_GRID if info.frames is not None else "unknown",
            "absolute_start_frame": absolute_start_frame,
        },
        "batch": info.batch,
        "binding": {"geometry": geometry_binding, "time": time_binding},
        "noise_mask": deep_copy_json(dict(noise_mask or {"present": False})),
        "naive_latent_concat_safe": False,
    }
    validate_h3_latent_contract(contract)
    return contract


def validate_h3_latent_contract(contract: Any) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise MMH3ResourceError("Missing or invalid extensions.minimax_h3.latent contract")
    if contract.get("contract_version") != H3_LATENT_CONTRACT_VERSION:
        raise MMH3ResourceError(f"Unsupported H3 latent contract version {contract.get('contract_version')!r}")
    if contract.get("model_family") != "minimax_h3" or contract.get("layout") != H3_LATENT_LAYOUT:
        raise MMH3ResourceError("H3 latent contract must describe minimax_h3 joint_av layout")
    validate_h3_latent_origin(str(contract.get("origin", "unknown")))
    streams = contract.get("streams")
    if not isinstance(streams, dict):
        raise MMH3ResourceError("H3 latent contract streams are missing")
    video = streams.get("video") if isinstance(streams.get("video"), dict) else {}
    audio = streams.get("audio") if isinstance(streams.get("audio"), dict) else {}
    vshape, ashape = video.get("shape"), audio.get("shape")
    if not (isinstance(vshape, list) and len(vshape) == 5 and isinstance(ashape, list) and len(ashape) == 4):
        raise MMH3ResourceError("H3 latent contract stream shapes must be video[5D] and audio[4D]")
    canvas = contract.get("canvas")
    if not isinstance(canvas, dict) or not all(isinstance(canvas.get(k), int) and canvas[k] > 0 for k in ("width", "height")):
        raise MMH3ResourceError("H3 latent contract canvas is invalid")
    timeline = contract.get("timeline")
    if not isinstance(timeline, dict):
        raise MMH3ResourceError("H3 latent contract timeline is missing")
    if timeline.get("fps") != FPS or timeline.get("audio_sample_rate") != AUDIO_SAMPLE_RATE or timeline.get("audio_latent_rate") != AUDIO_LATENT_FPS:
        raise MMH3ResourceError("H3 latent contract timing constants do not match H3")
    frames = timeline.get("frames")
    grid = timeline.get("temporal_grid")
    if frames is not None:
        if not isinstance(frames, int) or frames < 5 or (frames - 5) % 17:
            raise MMH3ResourceError("H3 latent contract frames are off the 17k+5 grid")
        if grid != H3_TEMPORAL_GRID:
            raise MMH3ResourceError("On-grid H3 latent must declare the stock temporal grid")
        expected_audio_t = round((frames / FPS) * AUDIO_LATENT_FPS)
        if int(ashape[-1]) != expected_audio_t:
            raise MMH3ResourceError(
                f"H3 latent contract AV duration mismatch: {frames} frames require audio T40={expected_audio_t}, got {ashape[-1]}"
            )
    binding = contract.get("binding")
    if not isinstance(binding, dict) or binding.get("geometry") not in _BINDING_STATES or binding.get("time") not in _BINDING_STATES:
        raise MMH3ResourceError("H3 latent contract binding state is invalid")
    return deep_copy_json(contract)


def h3_latent_contract_from_resource(resource: Mapping[str, Any]) -> dict[str, Any]:
    ext = resource.get("extensions") if isinstance(resource, Mapping) else None
    mmh3 = ext.get("minimax_h3") if isinstance(ext, Mapping) else None
    latent = mmh3.get("latent") if isinstance(mmh3, Mapping) else None
    return validate_h3_latent_contract(latent)


def validate_h3_geometry_state(contract: Mapping[str, Any], *, width: int | None = None, height: int | None = None) -> None:
    c = validate_h3_latent_contract(contract)
    if c["binding"]["geometry"] != "bound":
        raise MMH3ResourceError(f"H3 latent geometry binding is {c['binding']['geometry']!r}, not bound")
    canvas = c["canvas"]
    if width is not None and canvas["width"] != width:
        raise MMH3ResourceError(f"H3 latent width {canvas['width']} does not match target {width}")
    if height is not None and canvas["height"] != height:
        raise MMH3ResourceError(f"H3 latent height {canvas['height']} does not match target {height}")


def validate_h3_continuation_compatibility(contract: Mapping[str, Any]) -> None:
    c = validate_h3_latent_contract(contract)
    if c["origin"] != "sampler_output":
        raise MMH3ResourceError(f"H3 continuation requires sampler_output provenance; got {c['origin']!r}")
    if c["batch"] != 1:
        raise MMH3ResourceError("H3 continuation currently requires batch=1")
    if c["timeline"]["temporal_grid"] != H3_TEMPORAL_GRID:
        raise MMH3ResourceError("H3 continuation requires the stock 17k+5 temporal grid")
    validate_h3_geometry_state(c)
    if c["binding"]["time"] != "bound":
        raise MMH3ResourceError(f"H3 continuation requires bound time state; got {c['binding']['time']!r}")
    if c["timeline"].get("absolute_start_frame") is None:
        raise MMH3ResourceError("H3 continuation requires explicit absolute_start_frame")


def validate_h3_refine_compatibility(contract: Mapping[str, Any], *, width: int | None = None, height: int | None = None) -> None:
    c = validate_h3_latent_contract(contract)
    if c["origin"] != "sampler_output":
        raise MMH3ResourceError(f"H3 high-sigma refine requires sampler_output provenance; got {c['origin']!r}")
    validate_h3_geometry_state(c, width=width, height=height)
    if c["binding"]["time"] != "bound":
        raise MMH3ResourceError(f"H3 refine requires bound time state; got {c['binding']['time']!r}")


def stale_h3_geometry_contract(contract: Mapping[str, Any], *, width: int, height: int) -> dict[str, Any]:
    c = validate_h3_latent_contract(contract)
    c["canvas"] = {"width": int(width), "height": int(height)}
    c["binding"]["geometry"] = "stale"
    return validate_h3_latent_contract(c)


