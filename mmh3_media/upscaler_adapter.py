from __future__ import annotations

from .errors import MMH3ResourceError


UPSCALER_NODE = "MinimaxH3LatentUpscaler3D"


def _input_names(backend) -> set[str]:
    define_schema = getattr(backend, "define_schema", None)
    if callable(define_schema):
        return {item.id for item in define_schema().inputs}
    input_types = getattr(backend, "INPUT_TYPES", None)
    if callable(input_types):
        declared = input_types()
        return set(declared.get("required", {})) | set(declared.get("optional", {}))
    return set()


def resolve_upscaler_api(backend) -> str:
    """Identify the two supported public 3D-upscaler schemas."""
    names = _input_names(backend)
    common = {"latent", "model_name", "mode", "align", "device", "precision"}
    if not common.issubset(names):
        raise MMH3ResourceError(
            "Unsupported MinimaxH3LatentUpscaler3D schema: missing "
            + ", ".join(sorted(common - names))
        )
    if {"keep_proportion", "offload_after_upscale"}.issubset(names):
        return "plus_v1"
    if {"enable_temporal_chunking", "force_unload"}.issubset(names):
        return "legacy_v1"
    raise MMH3ResourceError(
        "Unsupported MinimaxH3LatentUpscaler3D schema. Install the maintained Plus fork "
        "or a legacy release exposing enable_temporal_chunking/force_unload."
    )


def build_upscaler_inputs(
    api: str,
    *,
    latent,
    model_name: str,
    target_width: int,
    target_height: int,
    align: int,
    device: str,
    precision: str,
    offload_after_upscale: bool,
    legacy_temporal_chunking: bool,
) -> dict:
    common = {
        "latent": latent,
        "model_name": model_name,
        "mode": "target dimensions",
        "mode.width": int(target_width),
        "mode.height": int(target_height),
        "align": int(align),
        "device": device,
        "precision": precision,
    }
    if api == "plus_v1":
        if device == "rocm":
            raise MMH3ResourceError("Upscaler-Plus supports cuda or cpu; select cpu for a non-CUDA runtime")
        return {
            **common,
            "keep_proportion": True,
            "offload_after_upscale": bool(offload_after_upscale),
        }
    if api == "legacy_v1":
        return {
            **common,
            "enable_temporal_chunking": bool(legacy_temporal_chunking),
            "force_unload": bool(offload_after_upscale),
        }
    raise MMH3ResourceError(f"Unknown learned-upscaler API {api!r}")
