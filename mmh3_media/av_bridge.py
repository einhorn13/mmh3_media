"""Decoded two-clip bridge: encode on the target grid, never concatenate VAE latents."""
from __future__ import annotations

import math
import torch

from .constants import AUDIO_SAMPLE_RATE, FPS
from .decoded_continuation import _stereo_waveform
from .continuation import h3_video_t_from_frames, is_exact_h3_av_handover_boundary
from .av_edit import video_frame_groups
from .errors import MMH3ResourceError
from .h3 import make_nested_tensor, nested_parts, validate_h3_av_latent


def prepare_av_bridge(clip_a, clip_b, *, context_a=39, context_b=39, gap_frames=46,
                      missing_audio="error"):
    if missing_audio not in {"error", "silence", "generate"}:
        raise MMH3ResourceError("missing_audio must be error, silence or generate")
    if any(type(n) is not int or n < 1 for n in (context_a, context_b, gap_frames)):
        raise MMH3ResourceError("Both contexts and gap_frames must be positive integers")
    if not all(is_exact_h3_av_handover_boundary(n) for n in (context_a, context_b)):
        raise MMH3ResourceError("Context windows must use exact H3 AV boundaries: 39, 90, 141, ...")
    requested = context_a + gap_frames + context_b
    total = 5 + 17 * max(0, math.ceil((requested - 5) / 17))
    if total > 3600:
        raise MMH3ResourceError("Bridge target exceeds 3600 frames including both contexts")
    h3_video_t_from_frames(total)
    gap = total - context_a - context_b
    frames, waves, sources, missing = [], [], [], []
    canvas = None
    for label, clip, count, tail in (("A", clip_a, context_a, True), ("B", clip_b, context_b, False)):
        if not hasattr(clip, "get_components"):
            raise MMH3ResourceError(f"Clip {label} must be a decoded VIDEO")
        components = clip.get_components()
        pixels = components.images
        if float(components.frame_rate) != FPS:
            raise MMH3ResourceError(f"Clip {label} must be {FPS} FPS; normalize explicitly first")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[-1] != 3 or len(pixels) < count:
            raise MMH3ResourceError(f"Clip {label} requires at least {count} RGB frames")
        shape = tuple(pixels.shape[1:])
        if shape[0] % 32 or shape[1] % 32 or min(shape[:2]) < 32 or (canvas and shape != canvas):
            raise MMH3ResourceError("Both clips must share a canvas divisible by 32")
        canvas = shape
        start = len(pixels) - count if tail else 0
        frames.append(pixels[start:start + count])
        source_audio = getattr(components, "audio", None)
        absent = source_audio is None
        if absent and missing_audio == "error":
            raise MMH3ResourceError(f"Clip {label} has no audio; choose silence or generate explicitly")
        if absent:
            waveform = pixels.new_zeros((1, 2, round(count * AUDIO_SAMPLE_RATE / FPS)))
        else:
            raw_wave = source_audio.get("waveform") if isinstance(source_audio, dict) else None
            if isinstance(raw_wave, torch.Tensor) and raw_wave.ndim == 3 and raw_wave.shape[1] > 2:
                raise MMH3ResourceError(f"Clip {label} has multichannel audio; downmix explicitly first")
            waveform, rate = _stereo_waveform(source_audio)
            if rate != AUDIO_SAMPLE_RATE:
                raise MMH3ResourceError(f"Clip {label} audio must be 32000 Hz; resample explicitly")
            left, right = round(start * rate / FPS), round((start + count) * rate / FPS)
            if waveform.shape[-1] < right:
                raise MMH3ResourceError(f"Clip {label} audio does not cover its synchronized context")
            waveform = waveform[..., left:right]
        waves.append(waveform)
        missing.append(absent)
        sources.append({"clip": label, "source_frames": len(pixels), "context": [start, start + count]})
    target = frames[0].new_zeros((total, *canvas))
    target[:context_a] = frames[0]
    target[-context_b:] = frames[1].to(target)
    waveform = waves[0].new_zeros((1, 2, round(total * AUDIO_SAMPLE_RATE / FPS)))
    waveform[..., :waves[0].shape[-1]] = waves[0]
    waveform[..., -waves[1].shape[-1]:] = waves[1].to(waveform)
    report = {"contract": "mmh3_two_clip_av_bridge_v1", "fps": FPS, "frames": total,
              "context_a": context_a, "context_b": context_b,
              "requested_gap_frames": gap_frames, "gap_frames": gap,
              "missing_audio": missing_audio, "absent_audio": missing, "sources": sources,
              "assembly": {"order": ["A", "bridge", "B"], "bridge_keep_frames": [context_a, total - context_b],
                           "bridge_audio_keep_samples": [round(context_a * AUDIO_SAMPLE_RATE / FPS),
                                                         round((total - context_b) * AUDIO_SAMPLE_RATE / FPS)],
                           "keep_full_source_clips": True},
              "latent_origin": "vae_encoded", "round_trip": True}
    return target, waveform, report


def encode_av_bridge(prepared, *, video_vae, audio_vae):
    pixels, waveform, report = prepared
    if int(getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE)) != AUDIO_SAMPLE_RATE:
        raise MMH3ResourceError("Bridge requires the H3 32 kHz audio VAE")
    latent = {"samples": make_nested_tensor([video_vae.encode(pixels), audio_vae.encode(waveform.movedim(1, -1))])}
    info = validate_h3_av_latent(latent, strict_audio_length=True)
    if info.batch != 1 or info.frames != report["frames"] or info.height != pixels.shape[1] or info.width != pixels.shape[2]:
        raise MMH3ResourceError("VAE returned an incompatible bridge target")
    video, audio = nested_parts(latent["samples"])
    vm, am = torch.ones_like(video), torch.ones_like(audio)
    left, right = report["context_a"], report["frames"] - report["context_b"]
    protected_groups = []
    for index, (start, end) in enumerate(video_frame_groups(report["frames"])):
        if end <= left or start >= right:
            vm[:, :, index] = 0
            protected_groups.append([start, end])
    if not report["absent_audio"][0] or report["missing_audio"] != "generate":
        am[..., :left * 40 // 24] = 0
    if not report["absent_audio"][1] or report["missing_audio"] != "generate":
        am[..., (right * 40 + 23) // 24:] = 0
    latent["noise_mask"] = make_nested_tensor([vm, am])
    return latent, {**report, "protected_video_frame_groups": protected_groups,
                    "protection": "latent cells fully inside context; finalize after sampling before decode"}


def trim_bridge_components(components, report):
    if report.get("contract") != "mmh3_two_clip_av_bridge_v1":
        raise MMH3ResourceError("Expected a Two-Clip AV Bridge report")
    if float(components.frame_rate) != FPS or len(components.images) != report["frames"]:
        raise MMH3ResourceError("Bridge output timeline differs from its preparation report")
    left, right = report["assembly"]["bridge_keep_frames"]
    if not 0 <= left < right <= report["frames"] or right - left != report["gap_frames"]:
        raise MMH3ResourceError("Invalid bridge trim range")
    audio = components.audio
    if audio is not None:
        rate = audio["sample_rate"]
        start, end = round(left * rate / FPS), round(right * rate / FPS)
        if audio["waveform"].shape[-1] < end:
            raise MMH3ResourceError("Decoded bridge audio is too short for its trim range")
        audio = {**audio, "waveform": audio["waveform"][..., start:end]}
    return components.images[left:right], audio, components.frame_rate
