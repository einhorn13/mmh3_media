from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import MMH3ResourceError
from .util import json_dumps_canonical


IMAGE_TO_VIDEO_NODE_ID = "MiniMaxH3ImageToVideo"
REFERENCE_TO_VIDEO_NODE_ID = "MiniMaxH3ReferenceToVideo"
ADD_GUIDE_NODE_ID = "MiniMaxH3AddGuide"
TILE_GUIDER_NODE_ID = "UltimateSDUpscaleNoUpscaleGuider"


@dataclass(frozen=True)
class AutogrowContract:
    group: str
    prefix: str
    maximum: int

    def input_path(self, index: int) -> str:
        if index < 0 or index >= self.maximum:
            raise MMH3ResourceError(
                f"Native H3 autogrow group {self.group!r} accepts indexes 0..{self.maximum - 1}; got {index}"
            )
        return f"{self.group}.{self.prefix}{index}"


@dataclass(frozen=True)
class NativeH3Contract:
    node_id: str
    fingerprint: str
    output_types: tuple[str, ...]
    autogrow: tuple[AutogrowContract, ...] = ()

    def group(self, name: str) -> AutogrowContract:
        found = next((item for item in self.autogrow if item.group == name), None)
        if found is None:
            raise MMH3ResourceError(f"Native H3 node {self.node_id} lacks autogrow group {name!r}")
        return found


@dataclass(frozen=True)
class AddGuideContract:
    node_id: str
    fingerprint: str
    output_types: tuple[str, ...]
    frame_idx_min: int
    frame_idx_max: int

    @property
    def supports_arbitrary_positions(self) -> bool:
        return self.frame_idx_min < 0 < self.frame_idx_max


@dataclass(frozen=True)
class TileGuiderContract:
    node_id: str
    fingerprint: str
    output_types: tuple[str, ...]
    traversal_modes: tuple[str, ...]
    overlap_modes: tuple[str, ...]
    batch_default: int
    supports_region_mask: bool
    supports_anchor_context: bool


def _input_type(spec: Any) -> str | None:
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    return str(spec[0])


def _input_options(spec: Any) -> tuple[str, ...]:
    if not isinstance(spec, (list, tuple)) or not spec or not isinstance(spec[0], (list, tuple)):
        return ()
    return tuple(str(item) for item in spec[0])


def parse_tile_guider_contract(info: Mapping[str, Any]) -> TileGuiderContract:
    """Parse the legacy object-info schema for the replaceable H3 tile-guider backend."""
    if not isinstance(info, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {TILE_GUIDER_NODE_ID} is not an object")
    inputs = info.get("input")
    required = inputs.get("required") if isinstance(inputs, Mapping) else None
    optional = inputs.get("optional") if isinstance(inputs, Mapping) else None
    if not isinstance(required, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {TILE_GUIDER_NODE_ID} has no required-input map")
    optional = optional if isinstance(optional, Mapping) else {}

    expected_types = {
        "upscaled_image": "IMAGE",
        "guider": "GUIDER",
        "sampler": "SAMPLER",
        "sigmas": "SIGMAS",
        "vae": "VAE",
        "seed": "INT",
        "tile_width": "INT",
        "tile_height": "INT",
        "mask_blur": "INT",
        "tile_padding": "INT",
        "seam_fix_denoise": "FLOAT",
        "seam_fix_width": "INT",
        "seam_fix_mask_blur": "INT",
        "seam_fix_padding": "INT",
        "tiled_decode": "BOOLEAN",
        "batch_size": "INT",
    }
    missing = sorted(set(expected_types).difference(required))
    if missing:
        raise MMH3ResourceError(
            f"Tile guider node {TILE_GUIDER_NODE_ID} is missing required inputs: {', '.join(missing)}"
        )
    for name, expected_type in expected_types.items():
        actual_type = _input_type(required[name])
        if actual_type != expected_type:
            raise MMH3ResourceError(
                f"Tile guider node {TILE_GUIDER_NODE_ID} input {name!r} changed type: "
                f"expected {expected_type}, got {actual_type!r}"
            )
    forbidden = sorted({"image", "model", "positive", "negative", "upscale_by", "upscale_model"}.intersection(required))
    if forbidden:
        raise MMH3ResourceError(
            f"Tile guider node {TILE_GUIDER_NODE_ID} no-upscale boundary unexpectedly requires: {', '.join(forbidden)}"
        )

    traversal_modes = _input_options(required.get("mode_type"))
    overlap_modes = _input_options(required.get("tile_overlap_mode"))
    seam_modes = _input_options(required.get("seam_fix_mode"))
    if not {"Linear", "Chess", "None"}.issubset(traversal_modes):
        raise MMH3ResourceError(f"Tile guider node {TILE_GUIDER_NODE_ID} lacks the expected traversal modes")
    if not {"Ignore Overlap", "Reprocess Overlap", "Context Only Overlap"}.issubset(overlap_modes):
        raise MMH3ResourceError(f"Tile guider node {TILE_GUIDER_NODE_ID} lacks the expected overlap modes")
    if "None" not in seam_modes:
        raise MMH3ResourceError(f"Tile guider node {TILE_GUIDER_NODE_ID} cannot disable its separate seam-fix pass")

    batch_spec = required["batch_size"]
    batch_options = batch_spec[1] if isinstance(batch_spec, (list, tuple)) and len(batch_spec) > 1 else None
    batch_options = batch_options if isinstance(batch_options, Mapping) else {}
    batch_default = batch_options.get("default")
    if not isinstance(batch_default, int) or isinstance(batch_default, bool) or batch_default != 1:
        raise MMH3ResourceError(f"Tile guider node {TILE_GUIDER_NODE_ID} must default batch_size to 1")

    optional_types = {"mask": "MASK", "anchor_context": "BOOLEAN"}
    for name, expected_type in optional_types.items():
        if name not in optional or _input_type(optional[name]) != expected_type:
            raise MMH3ResourceError(
                f"Tile guider node {TILE_GUIDER_NODE_ID} lacks optional {name!r} input of type {expected_type}"
            )
    outputs = info.get("output")
    if not isinstance(outputs, (list, tuple)) or tuple(outputs) != ("IMAGE",):
        raise MMH3ResourceError(
            f"Tile guider node {TILE_GUIDER_NODE_ID} output contract changed; expected one IMAGE output"
        )
    fingerprint = hashlib.sha256(json_dumps_canonical(dict(info)).encode("utf-8")).hexdigest()
    return TileGuiderContract(
        TILE_GUIDER_NODE_ID,
        fingerprint,
        tuple(str(item) for item in outputs),
        traversal_modes,
        overlap_modes,
        batch_default,
        True,
        True,
    )


def parse_add_guide_contract(info: Mapping[str, Any]) -> AddGuideContract:
    """Parse the runtime V1 schema for stock ``MiniMaxH3AddGuide`` without importing ComfyUI."""
    if not isinstance(info, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {ADD_GUIDE_NODE_ID} is not an object")
    inputs = info.get("input")
    required = inputs.get("required") if isinstance(inputs, Mapping) else None
    optional = inputs.get("optional") if isinstance(inputs, Mapping) else None
    if not isinstance(required, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {ADD_GUIDE_NODE_ID} has no required-input map")
    optional = optional if isinstance(optional, Mapping) else {}

    expected_required = {"positive": "CONDITIONING", "latent": "LATENT", "frame_idx": "INT"}
    expected_optional = {"vae": "VAE", "audio_vae": "VAE", "image": "IMAGE", "audio": "AUDIO"}
    missing_required = sorted(set(expected_required).difference(required))
    missing_optional = sorted(set(expected_optional).difference(optional))
    if missing_required:
        raise MMH3ResourceError(
            f"Native H3 node {ADD_GUIDE_NODE_ID} is missing required inputs: {', '.join(missing_required)}"
        )
    if missing_optional:
        raise MMH3ResourceError(
            f"Native H3 node {ADD_GUIDE_NODE_ID} is missing optional inputs: {', '.join(missing_optional)}"
        )
    for name, expected_type in expected_required.items():
        actual_type = _input_type(required[name])
        if actual_type != expected_type:
            raise MMH3ResourceError(
                f"Native H3 node {ADD_GUIDE_NODE_ID} input {name!r} changed type: "
                f"expected {expected_type}, got {actual_type!r}"
            )
    for name, expected_type in expected_optional.items():
        actual_type = _input_type(optional[name])
        if actual_type != expected_type:
            raise MMH3ResourceError(
                f"Native H3 node {ADD_GUIDE_NODE_ID} input {name!r} changed type: "
                f"expected {expected_type}, got {actual_type!r}"
            )

    frame_spec = required["frame_idx"]
    options = frame_spec[1] if isinstance(frame_spec, (list, tuple)) and len(frame_spec) > 1 else None
    options = options if isinstance(options, Mapping) else {}
    minimum, maximum = options.get("min"), options.get("max")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or not isinstance(maximum, int) or isinstance(maximum, bool):
        raise MMH3ResourceError(
            f"Native H3 node {ADD_GUIDE_NODE_ID} frame_idx must declare integer min/max bounds"
        )
    if minimum >= 0 or maximum <= 0:
        raise MMH3ResourceError(
            f"Native H3 node {ADD_GUIDE_NODE_ID} frame_idx no longer supports both negative and positive positions: "
            f"min={minimum!r}, max={maximum!r}"
        )

    outputs = info.get("output")
    if not isinstance(outputs, (list, tuple)) or tuple(outputs) != ("CONDITIONING",):
        raise MMH3ResourceError(
            f"Native H3 node {ADD_GUIDE_NODE_ID} output contract changed; expected one CONDITIONING output"
        )
    fingerprint = hashlib.sha256(json_dumps_canonical(dict(info)).encode("utf-8")).hexdigest()
    return AddGuideContract(
        ADD_GUIDE_NODE_ID,
        fingerprint,
        tuple(str(item) for item in outputs),
        minimum,
        maximum,
    )


def parse_native_h3_contract(node_id: str, info: Mapping[str, Any]) -> NativeH3Contract:
    if not isinstance(info, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {node_id} is not an object")
    inputs = info.get("input")
    required = inputs.get("required") if isinstance(inputs, Mapping) else None
    optional = inputs.get("optional") if isinstance(inputs, Mapping) else None
    if not isinstance(required, Mapping):
        raise MMH3ResourceError(f"Runtime contract for {node_id} has no required-input map")
    optional = optional if isinstance(optional, Mapping) else {}

    expected_required = {
        IMAGE_TO_VIDEO_NODE_ID: {"clip", "vae", "prompt", "width", "height", "length"},
        REFERENCE_TO_VIDEO_NODE_ID: {"clip", "prompt", "width", "height", "length", "ref_image_size"},
    }.get(node_id)
    if expected_required is None:
        raise MMH3ResourceError(f"Unsupported native H3 node contract {node_id!r}")
    missing = sorted(expected_required.difference(required))
    if missing:
        raise MMH3ResourceError(f"Native H3 node {node_id} is missing required inputs: {', '.join(missing)}")

    if node_id == REFERENCE_TO_VIDEO_NODE_ID:
        # Current ComfyUI permits text-encoder-only references by making the
        # video/audio VAEs optional. Older H3 releases exposed both as required.
        # MMH3 supplies them when materializing Ref2VA, so accept either schema
        # placement while still failing closed if an input disappears or changes
        # type.
        for name in ("vae", "audio_vae"):
            input_map = required if name in required else optional
            if name not in input_map:
                raise MMH3ResourceError(f"Native H3 node {node_id} is missing input: {name}")
            actual_type = _input_type(input_map[name])
            if actual_type != "VAE":
                raise MMH3ResourceError(
                    f"Native H3 node {node_id} input {name!r} changed type: expected VAE, got {actual_type!r}"
                )

    outputs = info.get("output")
    if not isinstance(outputs, (list, tuple)) or tuple(outputs[:2]) != ("CONDITIONING", "LATENT"):
        raise MMH3ResourceError(
            f"Native H3 node {node_id} output contract changed; expected CONDITIONING, LATENT at indexes 0, 1"
        )

    groups: list[AutogrowContract] = []
    if node_id == REFERENCE_TO_VIDEO_NODE_ID:
        expected_groups = {
            "ref_images": ("ref_image_", 9),
            "ref_videos": ("ref_video_", 3),
            "ref_video_audios": ("ref_video_audio_", 3),
            "ref_audios": ("ref_audio_", 3),
        }
        for group, (expected_prefix, minimum_max) in expected_groups.items():
            spec = optional.get(group)
            if not isinstance(spec, (list, tuple)) or len(spec) < 2 or spec[0] != "COMFY_AUTOGROW_V3":
                raise MMH3ResourceError(f"Native H3 node {node_id} lacks V3 autogrow group {group!r}")
            options = spec[1] if isinstance(spec[1], Mapping) else {}
            template = options.get("template") if isinstance(options, Mapping) else None
            prefix = template.get("prefix") if isinstance(template, Mapping) else None
            maximum = template.get("max") if isinstance(template, Mapping) else None
            if prefix != expected_prefix or not isinstance(maximum, int) or maximum < minimum_max:
                raise MMH3ResourceError(
                    f"Native H3 autogrow {group!r} changed: prefix={prefix!r}, max={maximum!r}"
                )
            groups.append(AutogrowContract(group, prefix, maximum))

    fingerprint = hashlib.sha256(json_dumps_canonical(dict(info)).encode("utf-8")).hexdigest()
    return NativeH3Contract(node_id, fingerprint, tuple(str(item) for item in outputs), tuple(groups))


def require_native_h3_contract(mode: str) -> NativeH3Contract:
    node_id = REFERENCE_TO_VIDEO_NODE_ID if mode == "ref2va" else IMAGE_TO_VIDEO_NODE_ID
    try:
        import nodes as comfy_nodes  # type: ignore
    except Exception as exc:
        raise MMH3ResourceError("Native H3 contract probe requires a running ComfyUI node registry") from exc
    node_class = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(node_id)
    if node_class is None:
        raise MMH3ResourceError(
            f"Native node {node_id} is not registered. Update/restart ComfyUI before running MMH3 H3 Auto Condition."
        )
    try:
        info = node_class.GET_NODE_INFO_V1()
    except Exception as exc:
        raise MMH3ResourceError(f"Could not inspect runtime contract for native node {node_id}: {exc}") from exc
    return parse_native_h3_contract(node_id, info)
