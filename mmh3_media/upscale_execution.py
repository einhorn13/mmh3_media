"""F07 execution contracts shared by nodes, reports and offline tests."""
import copy
import math

from .errors import MMH3ResourceError
from .generation_contract import validate_generation_model_family


def validate_upscaler_video_input(latent):
    import torch
    if not isinstance(latent, dict) or "samples" not in latent:
        raise MMH3ResourceError(
            "Upscaler requires video LATENT, not an MMH3 packet. Connect "
            "H3 Latent Upscale Prepare.video_latent (output 2), "
            "not Prepare.packet or Load.packet, to Upscale Preflight.video_latent."
        )
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor) or getattr(samples, "is_nested", False) or samples.ndim != 5 or samples.shape[1] != 24:
        raise MMH3ResourceError("Upscaler requires a plain video LATENT [B,24,T,H,W]; separate joint AV first")


def validate_sigma_values(values):
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 10001:
        raise MMH3ResourceError("Recorded sigmas must contain 2..10001 values")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise MMH3ResourceError("Sigmas must be finite nonnegative numbers")
    if values[0] <= 0 or values[-1] != 0 or any(a < b for a, b in zip(values, values[1:])):
        raise MMH3ResourceError("Sigmas must descend from positive noise to zero")


def build_refine_sigmas(recipe, calculate):
    """calculate(scheduler, total_steps) is injected; source_tail never calls it."""
    import torch
    report = copy.deepcopy(recipe)
    if report.get("contract") != "mmh3_f05_upscale_refine_sampling_v1":
        raise MMH3ResourceError("Expected a source-aware F07 sampling recipe")
    refine = report["refine"]
    steps = refine["steps"]
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise MMH3ResourceError("Invalid refine step count")
    mode = refine.get("schedule_mode", "regenerated_tail")
    if mode == "source_tail":
        source = report.get("source_sigmas")
        validate_sigma_values(source)
        if steps > len(source) - 1:
            raise MMH3ResourceError("Exact source tail cannot exceed the recorded intervals")
        dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64,
                 "torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16}.get(report.get("source_sigma_dtype", "torch.float32"))
        if dtype is None:
            raise MMH3ResourceError("Unsupported recorded sigma dtype")
        sigmas = torch.tensor(source[-steps-1:], dtype=dtype)
    elif mode == "regenerated_tail":
        denoise = refine["denoise"]
        if type(denoise) not in (float, int) or not math.isfinite(denoise) or not 0 < denoise <= 1:
            raise MMH3ResourceError("Refine denoise must be within 0..1")
        total = int(steps / denoise)
        if refine.get("scheduler_total_steps", total) != total:
            raise MMH3ResourceError("Refine sigma grid length disagrees with steps/denoise")
        sigmas = calculate(refine["scheduler"], total).detach().cpu()[-steps-1:]
    else:
        raise MMH3ResourceError("Unknown refine schedule mode")
    values = sigmas.tolist()
    validate_sigma_values(values)
    if len(values) != steps + 1:
        raise MMH3ResourceError("Scheduler returned an incorrect number of sigma intervals")
    report["executed_sigmas"] = values
    report["sigma_dtype"] = str(sigmas.dtype)
    return sigmas, report


def select_upscale_audio(packet, decoded_audio, policy, frames):
    import torch
    from .h3 import AUDIO_LATENT_FPS, h3_expected_audio_t
    if policy not in {"source_pcm", "decoded_latent"}:
        raise MMH3ResourceError("Unknown upscale audio policy")
    audio = decoded_audio
    if policy == "source_pcm":
        resource = packet.get_primary("audio")
        audio = packet.ref(resource["id"]).materialize() if resource else None
        if audio is None:
            video = packet.get_primary("video")
            if video:
                audio = getattr(packet.ref(video["id"]).materialize().get_components(), "audio", None)
    if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
        raise MMH3ResourceError("Selected audio is unavailable; source_pcm requires original PCM audio")
    waveform, rate = audio["waveform"], audio.get("sample_rate")
    if type(rate) is not int or rate <= 0 or waveform.ndim != 3 or waveform.shape[0] != 1:
        raise MMH3ResourceError("Invalid upscale audio waveform or sample rate")
    timeline_samples = round(frames * rate / 24)
    # H3 audio is quantized to a 40 Hz latent timeline.  Decoding that stream
    # retains the final latent hop, so e.g. 73 video frames produce T40=122 and
    # 97,600 PCM samples at 32 kHz rather than the unquantized 97,333 samples.
    h3_padded_samples = round(h3_expected_audio_t(frames) * rate / AUDIO_LATENT_FPS)
    sample_count = int(waveform.shape[-1])
    if policy == "source_pcm" and not any(
        abs(sample_count - expected) <= 1 for expected in (timeline_samples, h3_padded_samples)
    ):
        raise MMH3ResourceError(
            "Original PCM duration differs from both the video timeline and its H3 audio-latent boundary; "
            "align it explicitly"
        )
    return audio, {"policy": policy, "sample_rate": rate, "samples": waveform.shape[-1],
                   "frames": frames, "pcm_preserved": policy == "source_pcm",
                   "timeline_samples": timeline_samples, "h3_padded_samples": h3_padded_samples}


def upscale_preflight(packet, geometry, sampling, settings, optimization, available_nodes):
    from .latent_upscale import prepare_packet_latent_upscale, build_latent_upscale_refine_sampling
    if settings.get("device") not in {"cuda", "rocm", "cpu"} or settings.get("precision") not in {"fp32", "fp16", "bf16"}:
        raise MMH3ResourceError("Unsupported F07 device/precision")
    if settings.get("audio_policy") not in {"source_pcm", "decoded_latent"}:
        raise MMH3ResourceError("Unsupported F07 audio policy")
    refine = sampling["refine"]
    expected = build_latent_upscale_refine_sampling(packet, denoise_override=refine["denoise"],
        steps_override=refine.get("steps_override", 0), schedule_mode=refine.get("schedule_mode", "regenerated_tail")).info
    if sampling.get("source_sampling") != expected["source_sampling"] or sampling.get("task_family") != expected["task_family"]:
        raise MMH3ResourceError("F07 sampling recipe does not belong to the source packet")
    for key in ("steps", "sampler", "scheduler", "video_shift", "audio_shift", "scheduler_total_steps"):
        if refine.get(key) != expected["refine"].get(key):
            raise MMH3ResourceError("F07 sampling recipe disagrees with source/overrides: " + key)
    target = geometry["target"]
    prepared = prepare_packet_latent_upscale(packet, geometry_mode="target_dimensions",
        target_width=target["width"], target_height=target["height"], align=geometry["align"])
    if prepared.plan.to_dict()["target"] != target or prepared.plan.to_dict()["source"] != geometry["source"]:
        raise MMH3ResourceError("Preflight geometry disagrees with the source packet")
    family = validate_generation_model_family(sampling["task_family"],
        model_family=settings["model_family"], checkpoint_name=settings["refine_checkpoint"])
    if not family.ready or family.resolved_family == "unknown":
        raise MMH3ResourceError("F07 checkpoint/source family mismatch or unknown family: " + str(family.to_dict()))
    required = {"MinimaxH3LatentUpscaler3D", "MiniMaxH3SigmaShift", "MMH3H3RefineScheduler"}
    missing = required - set(available_nodes)
    if missing:
        raise MMH3ResourceError("Missing F07 nodes: " + ", ".join(sorted(missing)))
    if not isinstance(optimization, dict) or optimization.get("contract") != "mmh3_h3_model_optimizations_v1":
        raise MMH3ResourceError("F07 preflight requires the executed Optimizations profile")
    from .model_optimizations import ModelOptimizationPlan
    plan = ModelOptimizationPlan(optimization["enabled"], optimization["attention"]["mode"],
        optimization["torch"]["fp16_accumulation"], {"attention": optimization["attention"]["settings"]})
    for node in plan.required_nodes:
        if node not in available_nodes and not (node == "PathchSageAttentionKJ" and "PatchSageAttentionKJ" in available_nodes):
            raise MMH3ResourceError("Missing selected optimization node: " + node)
    if plan.attention_mode == "vdn_h3":
        raise MMH3ResourceError("VDN-H3 has no supported F07 low-sigma refine adapter")
    if settings["audio_policy"] == "source_pcm":
        select_upscale_audio(packet, None, "source_pcm", target["frames"])
    return {"contract": "mmh3_f07_preflight_v1", "ready": True, "geometry": geometry,
            "settings": settings, "model_family": family.to_dict(), "refine": refine,
            "optimizations": optimization}
