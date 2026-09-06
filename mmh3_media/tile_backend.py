from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .errors import MMH3ResourceError
from .h3 import concat_h3_av_latent, is_nested_tensor, nested_parts, validate_h3_av_latent
from .runtime_contract import TILE_GUIDER_NODE_ID
from .util import deep_copy_json


TILE_BACKEND_OVERLAP_MODES = ("Ignore Overlap", "Reprocess Overlap", "Context Only Overlap")


@dataclass(frozen=True)
class TileBackendFinalizeResult:
    latent: dict[str, Any]
    report: dict[str, Any]

    def summary(self) -> str:
        target = self.report["target"]
        return (
            f"READY · F07 tile backend finalize · {target['width']}x{target['height']} / "
            f"{target['frames']}f · video=vae_encoded · audio=preserve_exact"
        )


def _validate_fingerprint(value: str) -> str:
    fingerprint = str(value or "").strip().lower()
    if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        raise MMH3ResourceError("F07 tile backend requires a 64-character runtime schema fingerprint")
    return fingerprint


def tile_backend_fingerprint_from_preflight(report: Mapping[str, Any]) -> str:
    """Extract the admitted tile-backend schema fingerprint from an F15 report."""
    if not isinstance(report, Mapping):
        raise MMH3ResourceError("F07 tile finalize requires an MMH3Preflight info object")
    if report.get("ready") is not True:
        raise MMH3ResourceError("F07 tile finalize requires a READY MMH3Preflight report")
    if report.get("operation") != "high_sigma_refine":
        raise MMH3ResourceError("F07 tile finalize requires high_sigma_refine preflight")
    facts = report.get("facts")
    if not isinstance(facts, Mapping):
        raise MMH3ResourceError("F07 tile finalize preflight is missing runtime facts")
    if facts.get("runtime_nodes_checked") is not True or facts.get("runtime_contracts_checked") is not True:
        raise MMH3ResourceError("F07 tile finalize requires runtime node and schema checks")
    required_nodes = facts.get("required_nodes")
    if not isinstance(required_nodes, list) or TILE_GUIDER_NODE_ID not in required_nodes:
        raise MMH3ResourceError(f"F07 tile finalize preflight must require {TILE_GUIDER_NODE_ID}")
    contract = facts.get("tile_guider_contract")
    if not isinstance(contract, Mapping):
        raise MMH3ResourceError("F07 tile finalize preflight has no admitted tile-guider contract")
    if contract.get("output_types") != ["IMAGE"]:
        raise MMH3ResourceError("F07 tile finalize requires an IMAGE-only tile backend")
    if contract.get("batch_default") != 1:
        raise MMH3ResourceError("F07 tile finalize requires tile backend batch_default=1")
    modes = contract.get("overlap_modes")
    if not isinstance(modes, list) or not set(TILE_BACKEND_OVERLAP_MODES).issubset(set(modes)):
        raise MMH3ResourceError("F07 tile finalize preflight is missing required overlap modes")
    return _validate_fingerprint(contract.get("fingerprint", ""))


def _finalize_report(
    *,
    fingerprint: str,
    overlap_mode: str,
    settings: Mapping[str, Any],
    width: int,
    height: int,
    frames: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "contract": "mmh3_f07_external_tile_finalize_v1",
        "operation": "external_tile_refine_reencode",
        "backend": {
            "node_id": TILE_GUIDER_NODE_ID,
            "schema_fingerprint": fingerprint,
            "overlap_mode": overlap_mode,
            "batch_size": 1,
            "settings": deep_copy_json(dict(settings)),
            "output_boundary": "decoded_rgb_image_only",
        },
        "target": {"width": width, "height": height, "frames": frames},
        "provenance": {
            "backend_video_output": "decoded",
            "video_latent": "vae_encoded",
            "audio_latent": "preserved_exact_source",
            "joint_latent": "derived",
            "sampler_output": False,
        },
        "audio_policy": "preserve_exact_source_latent_outside_tile_backend",
        "peak_vram_measured": False,
    }


def finalize_external_tile_video(
    video: torch.Tensor,
    source_audio_latent: dict[str, Any],
    encode_video: Callable[[torch.Tensor], dict[str, Any]],
    *,
    backend_node_id: str,
    backend_schema_fingerprint: str,
    overlap_mode: str,
    batch_size: int = 1,
    settings: Mapping[str, Any] | None = None,
) -> TileBackendFinalizeResult:
    """Convert a decoded tile-backend result into an honest derived H3 AV latent.

    The external backend owns sampling and returns decoded RGB only. ``encode_video``
    is an explicit VideoVAE boundary and must return a separated H3 video LATENT.
    Source audio bypasses both callbacks and is recombined by tensor identity.
    """
    if backend_node_id != TILE_GUIDER_NODE_ID:
        raise MMH3ResourceError(f"Unsupported F07 tile backend node {backend_node_id!r}")
    _validate_fingerprint(backend_schema_fingerprint)
    if overlap_mode not in TILE_BACKEND_OVERLAP_MODES:
        raise MMH3ResourceError(f"Unsupported F07 tile backend overlap mode {overlap_mode!r}")
    if int(batch_size) != 1:
        raise MMH3ResourceError("F07 H3 tile backend requires sequential batch_size=1")
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise MMH3ResourceError("F07 tile backend output must be a [T,H,W,C] IMAGE tensor")
    frames, height, width, channels = (int(item) for item in video.shape)
    if channels != 3 or not video.is_floating_point():
        raise MMH3ResourceError("F07 tile backend output must be floating-point RGB")
    if frames < 5 or (frames - 5) % 17:
        raise MMH3ResourceError("F07 tile backend output must use a legal H3 17k+5 frame count")
    if width % 32 or height % 32:
        raise MMH3ResourceError("F07 tile backend output width and height must be divisible by 32")
    if not callable(encode_video):
        raise MMH3ResourceError("F07 tile finalize requires a callable VideoVAE encoder")

    encoded_video = encode_video(video.detach().clone())
    if not isinstance(encoded_video, dict):
        raise MMH3ResourceError("F07 tile VideoVAE encoder must return a separated video LATENT")
    encoded_samples = encoded_video.get("samples")
    if is_nested_tensor(encoded_samples):
        raise MMH3ResourceError("F07 tile VideoVAE encoder returned a joint latent instead of separated video")
    if not isinstance(encoded_samples, torch.Tensor):
        raise MMH3ResourceError("F07 tile VideoVAE encoder must return a separated video LATENT")
    return finalize_external_tile_latents(
        encoded_video,
        source_audio_latent,
        backend_node_id=backend_node_id,
        backend_schema_fingerprint=backend_schema_fingerprint,
        overlap_mode=overlap_mode,
        batch_size=batch_size,
        settings=settings,
        expected_width=width,
        expected_height=height,
        expected_frames=frames,
    )


def finalize_external_tile_latents(
    encoded_video_latent: dict[str, Any],
    source_audio_latent: dict[str, Any],
    *,
    backend_node_id: str,
    backend_schema_fingerprint: str,
    overlap_mode: str,
    batch_size: int = 1,
    settings: Mapping[str, Any] | None = None,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
) -> TileBackendFinalizeResult:
    if backend_node_id != TILE_GUIDER_NODE_ID:
        raise MMH3ResourceError(f"Unsupported F07 tile backend node {backend_node_id!r}")
    fingerprint = _validate_fingerprint(backend_schema_fingerprint)
    if overlap_mode not in TILE_BACKEND_OVERLAP_MODES:
        raise MMH3ResourceError(f"Unsupported F07 tile backend overlap mode {overlap_mode!r}")
    if int(batch_size) != 1:
        raise MMH3ResourceError("F07 H3 tile backend requires sequential batch_size=1")
    if not isinstance(encoded_video_latent, dict):
        raise MMH3ResourceError("F07 tile finalize requires a separated encoded video LATENT")
    encoded_samples = encoded_video_latent.get("samples")
    if is_nested_tensor(encoded_samples) or not isinstance(encoded_samples, torch.Tensor):
        raise MMH3ResourceError("F07 tile finalize requires a separated encoded video LATENT")
    if encoded_video_latent.get("noise_mask") is not None:
        raise MMH3ResourceError("F07 tile VideoVAE encoder output noise_mask ownership is undefined")
    if not isinstance(source_audio_latent, dict) or not isinstance(source_audio_latent.get("samples"), torch.Tensor):
        raise MMH3ResourceError("F07 tile finalize requires a separated source audio LATENT")
    if source_audio_latent.get("noise_mask") is not None:
        raise MMH3ResourceError("F07 tile finalize does not accept a source audio noise_mask")

    joint = concat_h3_av_latent(encoded_video_latent, source_audio_latent)
    info = validate_h3_av_latent(joint, strict_audio_length=True)
    expected = (int(expected_width), int(expected_height), int(expected_frames))
    if (info.width, info.height, info.frames) != expected:
        raise MMH3ResourceError(
            "F07 tile VideoVAE output geometry or temporal grid does not match the decoded backend video"
        )
    if nested_parts(joint["samples"])[1] is not source_audio_latent["samples"]:
        raise MMH3ResourceError("F07 tile finalize failed exact source-audio tensor preservation")
    report = _finalize_report(
        fingerprint=fingerprint,
        overlap_mode=overlap_mode,
        settings=settings or {},
        width=expected[0],
        height=expected[1],
        frames=expected[2],
    )
    return TileBackendFinalizeResult(joint, report)


def validate_external_tile_finalize_report(
    report: Mapping[str, Any],
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    frames: int | None = None,
) -> dict[str, Any]:
    value = deep_copy_json(dict(report))
    if value.get("version") != 1 or value.get("contract") != "mmh3_f07_external_tile_finalize_v1":
        raise MMH3ResourceError("Expected an F07 external tile finalize report v1")
    backend = value.get("backend")
    target = value.get("target")
    if not isinstance(backend, Mapping) or not isinstance(target, Mapping):
        raise MMH3ResourceError("F07 external tile finalize report is missing backend or target facts")
    fingerprint = _validate_fingerprint(backend.get("schema_fingerprint", ""))
    overlap_mode = backend.get("overlap_mode")
    if overlap_mode not in TILE_BACKEND_OVERLAP_MODES:
        raise MMH3ResourceError("F07 external tile finalize report has an unsupported overlap mode")
    try:
        width, height, frame_count = int(target["width"]), int(target["height"]), int(target["frames"])
        settings = backend["settings"]
    except (KeyError, TypeError, ValueError) as exc:
        raise MMH3ResourceError("F07 external tile finalize report is incomplete") from exc
    if not isinstance(settings, Mapping):
        raise MMH3ResourceError("F07 external tile finalize settings must be an object")
    expected = {"width": target_width, "height": target_height, "frames": frames}
    actual = {"width": width, "height": height, "frames": frame_count}
    for key, expected_value in expected.items():
        if expected_value is not None and actual[key] != int(expected_value):
            raise MMH3ResourceError(f"F07 external tile finalize {key} does not match the owning process")
    canonical = _finalize_report(
        fingerprint=fingerprint,
        overlap_mode=str(overlap_mode),
        settings=settings,
        width=width,
        height=height,
        frames=frame_count,
    )
    if value != canonical:
        raise MMH3ResourceError("F07 external tile finalize report does not match its canonical provenance")
    return value
