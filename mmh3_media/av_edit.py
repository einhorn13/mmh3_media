"""Independent stream masks. White means regenerate; existing protection always wins."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .errors import MMH3ResourceError
from .h3 import make_nested_tensor, nested_parts, validate_h3_av_latent


def _checked(mask, shape, label):
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(shape):
        raise MMH3ResourceError(f"{label} must have shape {tuple(shape)}")
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise MMH3ResourceError(f"{label} must contain finite values in [0,1]")
    return mask


def video_frame_groups(frames):
    """Causal H3 VAE groups: 1,4,4,4,4 repeated, ending with 1,4."""
    from .continuation import h3_video_t_from_frames
    count = h3_video_t_from_frames(frames)
    start = 0
    groups = []
    for index in range(count):
        end = start + (1 if index % 5 == 0 else 4)
        groups.append((start, end))
        start = end
    if start != frames:
        raise MMH3ResourceError("Invalid H3 frame grouping")
    return groups


def apply_av_edit_policy(latent, *, video_policy="mask", audio_policy="preserve",
                         video_mask=None, audio_intervals=()):
    info = validate_h3_av_latent(latent, strict_audio_length=True)
    if info.frames is None:
        raise MMH3ResourceError("Editing requires the H3 temporal grid")
    if video_policy not in {"preserve", "all", "mask"}:
        raise MMH3ResourceError("video_policy must be preserve, all or mask")
    if audio_policy not in {"preserve", "all", "intervals", "follow_video"}:
        raise MMH3ResourceError("Invalid audio_policy")
    video, audio = nested_parts(latent["samples"])
    vm = torch.full_like(video, float(video_policy == "all"))
    activity = torch.full((info.frames,), float(video_policy == "all"), device=video.device)
    if video_policy == "mask":
        if video_mask is None:
            raise MMH3ResourceError("video_policy=mask requires a video mask")
        mask = video_mask.unsqueeze(0) if video_mask.ndim == 2 else video_mask
        if mask.ndim != 3 or mask.shape[0] not in (1, info.frames):
            raise MMH3ResourceError("Video mask must be static [H,W]/[1,H,W] or full [frames,H,W]")
        _checked(mask, (mask.shape[0], info.height, info.width), "video_mask")
        mask = mask.to(device=video.device, dtype=torch.float32).expand(info.frames, -1, -1)
        activity = mask.amax(dim=(1, 2))
        # Max pooling aligns both latent pixels and the 2x2 DiT token footprint.
        pooled = F.adaptive_max_pool2d(mask[:, None], (video.shape[-2] // 2, video.shape[-1] // 2))
        pooled = pooled.repeat_interleave(2, -2).repeat_interleave(2, -1)
        for index, (start, end) in enumerate(video_frame_groups(info.frames)):
            vm[:, :, index] = pooled[start:end].amax(0)
    am = torch.full_like(audio, float(audio_policy == "all"))
    if audio_policy == "intervals":
        if not isinstance(audio_intervals, (list, tuple)):
            raise MMH3ResourceError("audio_intervals must be an array of [start_frame,end_frame)")
        for interval in audio_intervals:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise MMH3ResourceError("Each audio interval must contain two frame indices")
            start, end = interval
            if any(type(x) is not int for x in interval) or not 0 <= start < end <= info.frames:
                raise MMH3ResourceError("Audio interval is outside the target frame range")
            # Only wholly enclosed audio cells are editable: protection wins at boundaries.
            left, right = (start * 40 + 23) // 24, end * 40 // 24
            am[..., left:right] = 1
    elif audio_policy == "follow_video":
        for index in range(audio.shape[-1]):
            start = index * 24 // 40
            end = min(info.frames, ((index + 1) * 24 + 39) // 40)
            am[..., index] = activity[start:end].amin().to(audio)
    if latent.get("noise_mask") is not None:
        existing = nested_parts(latent["noise_mask"])
        if len(existing) != 2:
            raise MMH3ResourceError("Existing protection must contain both AV streams")
        vm *= _checked(existing[0], video.shape, "video protection").to(vm)
        am *= _checked(existing[1], audio.shape, "audio protection").to(am)
    report = {"contract": "mmh3_av_edit_v1", "video_policy": video_policy,
              "audio_policy": audio_policy, "audio_intervals": list(audio_intervals),
              "frames": info.frames, "semantics": "0=preserve,1=regenerate",
              "audio_preservation": "latent only; use original PCM for lossless delivery"}
    return {**latent, "noise_mask": make_nested_tensor([vm, am])}, report


def restore_av_protection(source, sampled):
    """Enforce exact protected values after sampling; fractional cells remain sampled."""
    validate_h3_av_latent(source, strict_audio_length=True)
    validate_h3_av_latent(sampled, strict_audio_length=True)
    if source.get("noise_mask") is None:
        raise MMH3ResourceError("Source has no AV protection mask")
    result = []
    for original, generated, mask in zip(nested_parts(source["samples"]),
                                         nested_parts(sampled["samples"]),
                                         nested_parts(source["noise_mask"]), strict=True):
        _checked(mask, original.shape, "protection")
        if generated.shape != original.shape:
            raise MMH3ResourceError("Sampled geometry differs from protected source")
        result.append(torch.where(mask.to(generated.device) == 0, original.to(generated), generated))
    # A finalized result is reusable; do not carry this run's editing mask into the next run.
    return {**{key: value for key, value in sampled.items() if key != "noise_mask"},
            "samples": make_nested_tensor(result)}


def select_edit_delivery_audio(packet, decoded_audio, *, delivery_policy="original_when_preserved"):
    if delivery_policy not in {"original_when_preserved", "decoded_latent"}:
        raise MMH3ResourceError("Invalid audio delivery policy")
    policy = packet.manifest.get("extensions", {}).get("minimax_h3", {}).get("av_edit_policy")
    if policy is None:
        raise MMH3ResourceError("Packet has no AV edit policy")
    if delivery_policy == "decoded_latent" or policy["audio_policy"] != "preserve":
        return decoded_audio
    if packet.get_primary("audio") is None and packet.get_primary("video") is None:
        raise MMH3ResourceError("No original PCM: connect source audio or explicitly choose decoded_latent delivery")
    # Reuse the source-PCM contract, including the native 40 Hz audio boundary.
    from .upscale_execution import select_upscale_audio
    return select_upscale_audio(packet, decoded_audio, "source_pcm", policy["frames"])[0]
