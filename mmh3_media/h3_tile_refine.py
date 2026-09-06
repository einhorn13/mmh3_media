from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .errors import MMH3ResourceError
from .h3 import nested_parts, split_h3_av_latent, validate_h3_av_latent
from .latent_upscale import build_latent_upscale_refine_target
from .spatial_tiles import SpatialTile, SpatialTilePlan, SpatialTileRunResult, run_spatial_video_tiles
from .util import deep_copy_json


NATIVE_H3_TILE_CONTRACT = "mmh3_h3_native_tile_refine_v1"


@dataclass(frozen=True)
class NativeH3TileRefineResult:
    video: torch.Tensor
    spatial_run: SpatialTileRunResult
    report: dict[str, Any]

    def summary(self) -> str:
        observed = self.report["observed"]
        return (
            f"READY · native H3 tile refine · {observed['tiles_sampled']} tiles · "
            "video=sampled · audio=locked/discarded"
        )


def _plain_latent(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("samples"), torch.Tensor):
        raise MMH3ResourceError(f"Native H3 tile refine {label} must be a separated LATENT")
    return value


def _validate_native_report(report: dict[str, Any], plan: SpatialTilePlan) -> dict[str, Any]:
    value = deep_copy_json(report)
    if value.get("version") != 1 or value.get("contract") != NATIVE_H3_TILE_CONTRACT:
        raise MMH3ResourceError("Expected a native H3 tile refine report v1")
    if value.get("plan") != plan.to_dict():
        raise MMH3ResourceError("Native H3 tile refine report does not match its tile plan")
    if value.get("sampler") != {
        "batch_size": 1,
        "timeline_per_tile": "complete",
        "video_noise_mask": 1,
        "audio_noise_mask": 0,
        "sampled_audio_action": "discard",
    }:
        raise MMH3ResourceError("Native H3 tile refine report has a non-canonical sampler contract")
    observed = value.get("observed")
    if not isinstance(observed, dict) or observed.get("tiles_sampled") != len(plan.tiles):
        raise MMH3ResourceError("Native H3 tile refine report has invalid tile accounting")
    if observed.get("source_audio_identity_preserved_in_each_target") is not True:
        raise MMH3ResourceError("Native H3 tile refine report lost source-audio identity")
    return value


def validate_native_h3_tile_refine_report(
    report: dict[str, Any], *, plan: SpatialTilePlan
) -> dict[str, Any]:
    return _validate_native_report(report, plan)


def run_native_h3_tile_refine(
    video: torch.Tensor,
    source_audio_latent: dict[str, Any],
    plan: SpatialTilePlan,
    *,
    encode_video: Callable[[torch.Tensor], dict[str, Any]],
    sample_joint: Callable[[dict[str, Any], SpatialTile], dict[str, Any]],
    decode_video: Callable[[dict[str, Any]], torch.Tensor],
) -> NativeH3TileRefineResult:
    """Run H3 video refinement one spatial tile at a time.

    Every callback target is a full-timeline joint H3 latent. Video receives an
    all-one denoise mask; the exact source audio tensor receives an all-zero
    mask. Sampled audio is never authoritative and is discarded before decode.
    """
    _plain_latent(source_audio_latent, label="source audio")
    if source_audio_latent.get("noise_mask") is not None:
        raise MMH3ResourceError("Native H3 tile refine source audio must not carry a noise_mask")
    if not all(callable(item) for item in (encode_video, sample_joint, decode_video)):
        raise MMH3ResourceError("Native H3 tile refine requires encode, sample and decode adapters")

    audio_samples = source_audio_latent["samples"]
    sampled_tiles = 0

    def process_tile(tile_video: torch.Tensor, tile: SpatialTile) -> torch.Tensor:
        nonlocal sampled_tiles
        encoded = _plain_latent(encode_video(tile_video.detach().clone()), label="VideoVAE output")
        if encoded.get("noise_mask") is not None:
            raise MMH3ResourceError("Native H3 tile VideoVAE output must not carry a noise_mask")
        target = build_latent_upscale_refine_target(encoded, source_audio_latent)
        target_video, target_audio = nested_parts(target["samples"])
        video_mask, audio_mask = nested_parts(target["noise_mask"])
        if target_audio is not audio_samples:
            raise MMH3ResourceError("Native H3 tile target did not preserve source-audio tensor identity")
        if not torch.all(video_mask == 1) or not torch.all(audio_mask == 0):
            raise MMH3ResourceError("Native H3 tile target must use video=1/audio=0 stream masks")

        sampled = sample_joint(target, tile)
        validate_h3_av_latent(sampled, strict_audio_length=True)
        sampled_video, _sampled_audio = split_h3_av_latent(sampled)
        if tuple(sampled_video["samples"].shape) != tuple(target_video.shape):
            raise MMH3ResourceError("Native H3 tile sampler changed video latent geometry")
        decoded = decode_video(sampled_video)
        if not isinstance(decoded, torch.Tensor):
            raise MMH3ResourceError("Native H3 tile VideoVAE decoder did not return an IMAGE tensor")
        if decoded.ndim == 5:
            if int(decoded.shape[0]) != 1:
                raise MMH3ResourceError("Native H3 tile VideoVAE decoder returned a multi-batch video")
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        if decoded.ndim != 4 or int(decoded.shape[-1]) != 3:
            raise MMH3ResourceError("Native H3 tile VideoVAE decoder must return [T,H,W,C] RGB video")
        if int(decoded.shape[0]) < int(tile_video.shape[0]):
            raise MMH3ResourceError("Native H3 tile VideoVAE decoded fewer frames than the source timeline")
        sampled_tiles += 1
        return decoded[: tile_video.shape[0]].to(device=tile_video.device, dtype=tile_video.dtype)

    spatial = run_spatial_video_tiles(video, plan, process_tile)
    report = {
        "version": 1,
        "contract": NATIVE_H3_TILE_CONTRACT,
        "operation": "native_h3_spatial_tile_refine",
        "plan": plan.to_dict(),
        "spatial_run": deep_copy_json(spatial.report),
        "sampler": {
            "batch_size": 1,
            "timeline_per_tile": "complete",
            "video_noise_mask": 1,
            "audio_noise_mask": 0,
            "sampled_audio_action": "discard",
        },
        "final_audio_owner": "source_audio_latent_recombined_after_full_video_vae_encode",
        "observed": {
            "tiles_sampled": sampled_tiles,
            "source_audio_identity_preserved_in_each_target": True,
            "peak_vram_measured": False,
        },
    }
    _validate_native_report(report, plan)
    return NativeH3TileRefineResult(spatial.video, spatial, report)
