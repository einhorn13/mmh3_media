from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .archive import load_archive, save_archive
from .automation import VIDEO_SUFFIXES
from .constants import AUDIO_SAMPLE_RATE, FPS
from .core import MMH3Media
from .errors import MMH3Error, MMH3ResourceError
from .stitch import inspect_stitch_packet
from .resource_model import descriptor_from_media_metadata


RAW_IMPORT_CONTRACT = "mmh3_raw_video_import_report_v1"
RAW_IMPORT_SETTINGS_CONTRACT = "mmh3_raw_video_conform_v1"
PIXEL_FORMAT = "yuv420p"
AUDIO_SAMPLE_BYTES = 4  # ComfyUI AUDIO is float32 PCM in this repository.
AUDIO_PCM_WORKING_SET_MULTIPLIER = 3  # decode chunks + contiguous conform/save transient copies


@dataclass(frozen=True)
class RawVideoProbe:
    path: str
    width: int
    height: int
    fps: float | None
    pixel_format: str | None
    duration_seconds: float | None
    frame_count_hint: int | None
    audio_sample_rate: int | None
    audio_channels: int | None
    has_audio: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "pixel_format": self.pixel_format,
            "duration_seconds": self.duration_seconds,
            "frame_count_hint": self.frame_count_hint,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "has_audio": self.has_audio,
        }


@dataclass(frozen=True)
class ConformSettings:
    width: int
    height: int
    fps: int = FPS
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    audio_channels: int = 2
    pixel_format: str = PIXEL_FORMAT
    max_audio_pcm_bytes: int = 512 * 1024 * 1024

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": RAW_IMPORT_SETTINGS_CONTRACT,
            "canvas": [self.width, self.height],
            "fps": self.fps,
            "pixel_format": self.pixel_format,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "fps_policy": "preserve_all_decoded_frames_retime",
            "canvas_policy": "contain_letterbox_no_crop_no_distort",
            "audio_policy": "full_content_time_conform_to_frame_derived_pcm",
            "pcm_duration_policy": "round(frame_count * sample_rate / fps)",
            "audio_memory_limit_bytes": self.max_audio_pcm_bytes,
        }


@dataclass(frozen=True)
class RawImportResult:
    archive_path: Path
    report: dict[str, Any]


def canonical_pcm_samples(frame_count: int, *, fps: int, sample_rate: int) -> int:
    if frame_count < 0 or fps <= 0 or sample_rate <= 0:
        raise MMH3ResourceError("Invalid frame-derived PCM boundary arguments")
    return round(int(frame_count) * int(sample_rate) / int(fps))


def audio_pcm_bytes(samples: int, *, channels: int = 2) -> int:
    if samples < 0 or channels < 1:
        raise MMH3ResourceError("Invalid PCM estimate arguments")
    return int(samples) * int(channels) * AUDIO_SAMPLE_BYTES


def estimate_audio_pcm_bytes(
    duration_seconds: float | None,
    *,
    sample_rate: int,
    channels: int,
    safety_factor: float = 1.10,
) -> int | None:
    if duration_seconds is None or duration_seconds <= 0:
        return None
    samples = math.ceil(float(duration_seconds) * int(sample_rate) * float(safety_factor))
    return audio_pcm_bytes(samples, channels=channels)


def validate_target_overrides(
    *,
    target_width: int,
    target_height: int,
    target_fps: int,
    target_audio_sample_rate: int,
    target_audio_channels: int,
    max_audio_pcm_mb: int,
) -> None:
    if (target_width == 0) != (target_height == 0):
        raise MMH3ResourceError("target_width and target_height must both be 0 (auto) or both be set")
    if target_width < 0 or target_height < 0:
        raise MMH3ResourceError("Target canvas dimensions cannot be negative")
    if target_width and (target_width < 2 or target_height < 2):
        raise MMH3ResourceError("Target canvas must be at least 2x2")
    if target_fps < 1 or target_fps > 240:
        raise MMH3ResourceError("target_fps must be in 1..240")
    if target_audio_sample_rate < 8000 or target_audio_sample_rate > 192000:
        raise MMH3ResourceError("target_audio_sample_rate must be in 8000..192000")
    if target_audio_channels not in {1, 2}:
        raise MMH3ResourceError("target_audio_channels must be mono or stereo")
    if max_audio_pcm_mb < 1:
        raise MMH3ResourceError("max_audio_pcm_mb must be at least 1 MiB")


def _even(value: int) -> int:
    value = int(value)
    return value if value % 2 == 0 else value + 1


def canvas_fit(source_width: int, source_height: int, target_width: int, target_height: int) -> dict[str, int]:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise MMH3ResourceError("Canvas dimensions must be positive")
    scale = min(target_width / source_width, target_height / source_height)
    scaled_width = max(1, min(target_width, round(source_width * scale)))
    scaled_height = max(1, min(target_height, round(source_height * scale)))
    left = (target_width - scaled_width) // 2
    top = (target_height - scaled_height) // 2
    return {
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "pad_left": left,
        "pad_right": target_width - scaled_width - left,
        "pad_top": top,
        "pad_bottom": target_height - scaled_height - top,
    }


def _stream_duration_seconds(stream: Any, container: Any) -> float | None:
    duration = getattr(stream, "duration", None)
    time_base = getattr(stream, "time_base", None)
    if duration not in (None, 0) and time_base not in (None, 0):
        try:
            value = float(duration * time_base)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    container_duration = getattr(container, "duration", None)
    if container_duration not in (None, 0):
        try:
            # PyAV container duration is expressed in AV_TIME_BASE microseconds.
            value = float(container_duration) / 1_000_000.0
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return None


def probe_raw_video(path: str | Path, *, av_module: Any | None = None) -> RawVideoProbe:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() not in VIDEO_SUFFIXES:
        raise MMH3ResourceError(f"Unsupported raw video extension: {source.suffix or '<none>'}")
    if not source.is_file():
        raise MMH3ResourceError(f"Raw video does not exist: {source}")
    if av_module is None:
        try:
            import av as av_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - supplied by ComfyUI runtime
            raise MMH3ResourceError("Raw video import requires PyAV from the ComfyUI runtime") from exc
    try:
        with av_module.open(str(source), mode="r") as container:
            if not container.streams.video:
                raise MMH3ResourceError("Raw input contains no video stream")
            video = container.streams.video[0]
            width = int(getattr(video, "width", 0) or 0)
            height = int(getattr(video, "height", 0) or 0)
            if width < 1 or height < 1:
                raise MMH3ResourceError("Raw video metadata has invalid canvas dimensions")
            rate = getattr(video, "average_rate", None) or getattr(video, "base_rate", None)
            try:
                fps = float(rate) if rate not in (None, 0) else None
            except (TypeError, ValueError, ZeroDivisionError):
                fps = None
            frame_hint = int(getattr(video, "frames", 0) or 0) or None
            pix_fmt = getattr(getattr(video, "codec_context", None), "pix_fmt", None)
            duration = _stream_duration_seconds(video, container)
            audio = container.streams.audio[0] if container.streams.audio else None
            sample_rate = int(getattr(audio, "rate", 0) or 0) or None if audio is not None else None
            channels = int(getattr(audio, "channels", 0) or 0) or None if audio is not None else None
            return RawVideoProbe(
                str(source), width, height, fps, str(pix_fmt) if pix_fmt else None,
                duration, frame_hint, sample_rate, channels, audio is not None,
            )
    except MMH3ResourceError:
        raise
    except Exception as exc:
        raise MMH3ResourceError(f"Could not inspect raw video {source.name}: {exc}") from exc


def make_file_backed_video(path: str | Path):
    """Use the same ComfyUI VIDEO abstraction already used by this repository."""
    try:
        from comfy_api.latest import InputImpl  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime contract
        raise MMH3ResourceError("Raw video import requires current ComfyUI comfy_api.latest.InputImpl") from exc
    video = InputImpl.VideoFromFile(str(Path(path)))
    if not hasattr(video, "get_stream_source") or not hasattr(video, "save_to"):
        raise MMH3ResourceError("ComfyUI VideoFromFile lacks the file-backed VIDEO methods required by MMH3")
    return video


def _contain_frame(frame: Any, *, width: int, height: int, fit: dict[str, int], av_module: Any):
    image = frame.to_image().convert("RGB")
    scaled = image.resize((fit["scaled_width"], fit["scaled_height"]), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(scaled, (fit["pad_left"], fit["pad_top"]))
    return av_module.VideoFrame.from_image(canvas)


def _decode_audio_full(
    source: Path,
    *,
    av_module: Any,
    sample_rate: int,
    channels: int,
    max_pcm_bytes: int,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    layout = "stereo" if channels == 2 else "mono"
    chunks: list[torch.Tensor] = []
    total_samples = 0
    with av_module.open(str(source), mode="r") as container:
        if not container.streams.audio:
            return None, {"source_audio": "missing", "decoded_samples": 0, "decoded_pcm_bytes": 0}
        stream = container.streams.audio[0]
        resampler = av_module.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        for frame in container.decode(stream):
            produced = resampler.resample(frame)
            if produced is None:
                continue
            if not isinstance(produced, (list, tuple)):
                produced = [produced]
            for audio_frame in produced:
                array = np.asarray(audio_frame.to_ndarray(), dtype=np.float32)
                if array.ndim == 1:
                    array = array[None, :]
                if array.shape[0] != channels and array.ndim == 2 and array.shape[1] == channels:
                    array = array.T
                if array.shape[0] != channels:
                    raise MMH3ResourceError(
                        f"PyAV audio resampler returned {array.shape}; expected {channels} planar channels"
                    )
                tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
                total_samples += int(tensor.shape[-1])
                used = audio_pcm_bytes(total_samples, channels=channels)
                working_set = used * AUDIO_PCM_WORKING_SET_MULTIPLIER
                if working_set > max_pcm_bytes:
                    raise MMH3ResourceError(
                        "Decoded audio PCM working set exceeds configured memory limit: "
                        f"{working_set} > {max_pcm_bytes} bytes"
                    )
                chunks.append(tensor)
        flushed = resampler.resample(None)
        if flushed is not None:
            if not isinstance(flushed, (list, tuple)):
                flushed = [flushed]
            for audio_frame in flushed:
                array = np.asarray(audio_frame.to_ndarray(), dtype=np.float32)
                if array.ndim == 1:
                    array = array[None, :]
                if array.shape[0] != channels and array.ndim == 2 and array.shape[1] == channels:
                    array = array.T
                tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
                total_samples += int(tensor.shape[-1])
                used = audio_pcm_bytes(total_samples, channels=channels)
                working_set = used * AUDIO_PCM_WORKING_SET_MULTIPLIER
                if working_set > max_pcm_bytes:
                    raise MMH3ResourceError(
                        "Decoded audio PCM working set exceeds configured memory limit: "
                        f"{working_set} > {max_pcm_bytes} bytes"
                    )
                chunks.append(tensor)
    waveform = torch.cat(chunks, dim=-1) if chunks else torch.zeros((1, channels, 0), dtype=torch.float32)
    return waveform, {
        "source_audio": "decoded",
        "decoded_samples": int(waveform.shape[-1]),
        "decoded_pcm_bytes": audio_pcm_bytes(int(waveform.shape[-1]), channels=channels),
    }


def _time_conform_audio(waveform: torch.Tensor | None, target_samples: int, *, channels: int) -> tuple[torch.Tensor, str]:
    if target_samples < 0:
        raise MMH3ResourceError("Target audio length cannot be negative")
    if waveform is None or int(waveform.shape[-1]) == 0:
        return torch.zeros((1, channels, target_samples), dtype=torch.float32), "silence_for_missing_audio"
    if int(waveform.shape[-1]) == target_samples:
        return waveform.to(dtype=torch.float32), "none"
    if target_samples == 0:
        raise MMH3ResourceError("Cannot conform non-empty audio to a zero-frame video")
    # Uses the full source waveform: no head/tail crop. This changes playback rate when
    # preserving every source video frame requires retiming to the target FPS.
    return F.interpolate(waveform.to(dtype=torch.float32), size=target_samples, mode="linear", align_corners=False), "linear_time_conform_full_content"


def import_raw_video(
    source_path: str | Path,
    *,
    archive_path: str | Path,
    settings: ConformSettings,
    probe: RawVideoProbe | None = None,
    av_module: Any | None = None,
    video_factory: Callable[[str | Path], Any] = make_file_backed_video,
) -> RawImportResult:
    source = Path(source_path).expanduser().resolve()
    probe = probe or probe_raw_video(source, av_module=av_module)
    if av_module is None:
        try:
            import av as av_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - supplied by ComfyUI runtime
            raise MMH3ResourceError("Raw video import requires PyAV from the ComfyUI runtime") from exc

    estimate = estimate_audio_pcm_bytes(
        probe.duration_seconds,
        sample_rate=settings.audio_sample_rate,
        channels=settings.audio_channels,
    )
    estimated_working_set = (estimate * AUDIO_PCM_WORKING_SET_MULTIPLIER) if estimate is not None else None
    if estimated_working_set is not None and estimated_working_set > settings.max_audio_pcm_bytes:
        raise MMH3ResourceError(
            "Estimated audio PCM working set exceeds configured memory limit: "
            f"{estimated_working_set} > {settings.max_audio_pcm_bytes} bytes"
        )

    target_archive = Path(archive_path).expanduser().resolve()
    target_archive.parent.mkdir(parents=True, exist_ok=True)
    normalized_video = target_archive.with_suffix(".normalized.mp4")
    fit = canvas_fit(probe.width, probe.height, settings.width, settings.height)
    frame_count = 0
    try:
        with av_module.open(str(source), mode="r") as input_container, av_module.open(str(normalized_video), mode="w") as output_container:
            if not input_container.streams.video:
                raise MMH3ResourceError("Raw input contains no video stream")
            input_stream = input_container.streams.video[0]
            output_stream = output_container.add_stream("libx264", rate=settings.fps)
            output_stream.width = settings.width
            output_stream.height = settings.height
            output_stream.pix_fmt = settings.pixel_format
            for frame in input_container.decode(input_stream):
                conformed = _contain_frame(frame, width=settings.width, height=settings.height, fit=fit, av_module=av_module)
                conformed.pts = frame_count
                conformed.time_base = Fraction(1, settings.fps)
                for packet in output_stream.encode(conformed):
                    output_container.mux(packet)
                frame_count += 1
            for packet in output_stream.encode(None):
                output_container.mux(packet)
    except MMH3ResourceError:
        normalized_video.unlink(missing_ok=True)
        raise
    except Exception as exc:
        normalized_video.unlink(missing_ok=True)
        raise MMH3ResourceError(f"Raw video streaming conform failed for {source.name}: {exc}") from exc
    if frame_count < 1:
        normalized_video.unlink(missing_ok=True)
        raise MMH3ResourceError("Raw video decoded zero frames")

    target_samples = canonical_pcm_samples(frame_count, fps=settings.fps, sample_rate=settings.audio_sample_rate)
    final_pcm_bytes = audio_pcm_bytes(target_samples, channels=settings.audio_channels)
    final_working_set = final_pcm_bytes * AUDIO_PCM_WORKING_SET_MULTIPLIER
    if final_working_set > settings.max_audio_pcm_bytes:
        normalized_video.unlink(missing_ok=True)
        raise MMH3ResourceError(
            "Frame-derived output audio PCM working set exceeds configured memory limit: "
            f"{final_working_set} > {settings.max_audio_pcm_bytes} bytes"
        )
    waveform, audio_decode = _decode_audio_full(
        source,
        av_module=av_module,
        sample_rate=settings.audio_sample_rate,
        channels=settings.audio_channels,
        max_pcm_bytes=settings.max_audio_pcm_bytes,
    )
    waveform, audio_adjustment = _time_conform_audio(waveform, target_samples, channels=settings.audio_channels)
    audio = {"waveform": waveform, "sample_rate": settings.audio_sample_rate}

    video = video_factory(normalized_video)
    packet = MMH3Media.create(name=f"Imported {source.name}")
    video_facts = {
        "dimensions": [settings.width, settings.height],
        "fps": float(settings.fps),
        "frame_count": frame_count,
        "duration": frame_count / settings.fps,
    }
    packet, _ = packet.put_primary(
        video, kind="video", descriptor=descriptor_from_media_metadata("video", video_facts),
        extensions={"mmh3_media": {"raw_import": {"source_path": str(source)}}},
    )
    audio_facts = {
        "sample_rate": settings.audio_sample_rate,
        "channels": settings.audio_channels,
        "samples": target_samples,
        "duration": target_samples / settings.audio_sample_rate,
    }
    packet, _ = packet.put_primary(
        audio, kind="audio", descriptor=descriptor_from_media_metadata("audio", audio_facts),
        extensions={"mmh3_media": {"raw_import": True}},
    )
    report = {
        "contract": RAW_IMPORT_CONTRACT,
        "source": probe.to_dict(),
        "conform": settings.to_dict(),
        "video": {
            "decoded_frames": frame_count,
            "output_frames": frame_count,
            "frames_dropped": 0,
            "canvas_fit": fit,
            "storage": "temporary normalized MP4 byte-copied into MMH3; staging file removed after archive commit",
        },
        "audio": {
            **audio_decode,
            "output_samples": target_samples,
            "output_pcm_bytes": final_pcm_bytes,
            "adjustment": audio_adjustment,
            "duration_seconds": target_samples / settings.audio_sample_rate,
        },
        "estimate": {
            "audio_pcm_bytes_from_metadata": estimate,
            "audio_pcm_working_set_bytes_from_metadata": estimated_working_set,
            "audio_pcm_output_bytes": final_pcm_bytes,
            "audio_pcm_output_working_set_bound_bytes": final_working_set,
            "audio_pcm_working_set_multiplier": AUDIO_PCM_WORKING_SET_MULTIPLIER,
            "audio_pcm_limit_bytes": settings.max_audio_pcm_bytes,
            "full_pcm_materialization_required": True,
            "rgb_full_video_materialization": False,
            "rgb_frame_buffering": "O(1) frames; exact transient allocator peak requires runtime measurement",
        },
    }
    packet = packet.set_extension_value("mmh3_media", "raw_video_import", report)
    try:
        saved, _ = save_archive(packet, target_archive)
    finally:
        # save_archive byte-copies a file-backed VideoFromFile into the MMH3 archive.
        # The staging MP4 is no longer required after the atomic archive commit/failure.
        normalized_video.unlink(missing_ok=True)
    return RawImportResult(Path(saved.source_archive or target_archive), report)


def _resolve_path(raw_path: str, search_roots: Sequence[str | Path]) -> Path:
    source = Path(raw_path).expanduser()
    if source.is_absolute():
        return source.resolve()
    candidates = [(Path(root).expanduser().resolve() / source).resolve() for root in search_roots]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) > 1:
        raise MMH3ResourceError(f"Relative batch path is ambiguous across search roots: {raw_path}")
    if existing:
        return existing[0]
    return candidates[0] if candidates else source.resolve()


def _auto_canvas_from_plan(plan: dict[str, Any], *, search_roots: Sequence[str | Path], probe_fn: Callable[..., RawVideoProbe]) -> tuple[int, int]:
    for job in plan.get("jobs", []):
        if not isinstance(job, dict):
            continue
        raw_path = job.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        try:
            resolved = _resolve_path(raw_path, search_roots)
            if job.get("kind") == "video":
                probe = probe_fn(resolved)
                return _even(probe.width), _even(probe.height)
            if job.get("kind") == "mmh3":
                packet = load_archive(resolved, verify="manifest")
                fact, reasons = inspect_stitch_packet(packet, index=int(job.get("index") or 0))
                if not reasons and isinstance(fact.get("dimensions"), list):
                    return _even(int(fact["dimensions"][0])), _even(int(fact["dimensions"][1]))
        except (MMH3Error, OSError):
            continue
    raise MMH3ResourceError("Could not derive target canvas from batch metadata; set target_width/target_height manually")


def normalize_batch_plan(
    plan: dict[str, Any],
    *,
    search_roots: Sequence[str | Path],
    temp_root: str | Path,
    error_policy: str,
    target_width: int = 0,
    target_height: int = 0,
    target_fps: int = FPS,
    target_audio_sample_rate: int = AUDIO_SAMPLE_RATE,
    target_audio_channels: int = 2,
    max_audio_pcm_mb: int = 512,
    probe_fn: Callable[..., RawVideoProbe] = probe_raw_video,
    import_fn: Callable[..., RawImportResult] = import_raw_video,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(plan, dict) or plan.get("contract") != "mmh3_batch_input_plan_v1":
        raise MMH3ResourceError("Raw batch import requires mmh3_batch_input_plan_v1")
    if error_policy not in {"stop_on_error", "skip_invalid"}:
        raise MMH3ResourceError("Raw batch import error_policy must be stop_on_error or skip_invalid")
    validate_target_overrides(
        target_width=target_width, target_height=target_height, target_fps=target_fps,
        target_audio_sample_rate=target_audio_sample_rate, target_audio_channels=target_audio_channels,
        max_audio_pcm_mb=max_audio_pcm_mb,
    )
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise MMH3ResourceError("Raw batch import plan has no jobs")
    if target_width == 0:
        target_width, target_height = _auto_canvas_from_plan(plan, search_roots=search_roots, probe_fn=probe_fn)
    target_width, target_height = _even(target_width), _even(target_height)
    settings = ConformSettings(
        target_width, target_height, target_fps, target_audio_sample_rate,
        target_audio_channels, PIXEL_FORMAT, int(max_audio_pcm_mb) * 1024 * 1024,
    )
    session_dir = Path(tempfile.mkdtemp(prefix="mmh3_raw_batch_", dir=str(Path(temp_root))))
    output_jobs: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    total_pcm_bytes = 0
    for position, job in enumerate(jobs):
        item = {
            "position": position,
            "index": job.get("index") if isinstance(job, dict) else None,
            "job_id": job.get("job_id") if isinstance(job, dict) else None,
            "input_path": job.get("path") if isinstance(job, dict) else None,
            "kind": job.get("kind") if isinstance(job, dict) else None,
            "status": "rejected",
            "reasons": [],
        }
        try:
            if not isinstance(job, dict) or job.get("index") != position:
                raise MMH3ResourceError("job index is missing or does not match ordered position")
            raw_path = job.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise MMH3ResourceError("job path is missing")
            resolved = _resolve_path(raw_path, search_roots)
            item["resolved_path"] = str(resolved)
            if job.get("kind") == "mmh3":
                packet = load_archive(resolved, verify="manifest")
                fact, reasons = inspect_stitch_packet(packet, index=position, expected_dimensions=(settings.width, settings.height))
                if reasons:
                    raise MMH3ResourceError("; ".join(reasons))
                item["status"] = "passed_through"
                item["facts"] = fact
                existing_pcm_bytes = audio_pcm_bytes(int(fact.get("audio_samples") or 0), channels=2)
                item["estimated_audio_pcm_bytes"] = existing_pcm_bytes
                total_pcm_bytes += existing_pcm_bytes
                output_jobs.append({"index": len(output_jobs), "job_id": job.get("job_id"), "path": str(resolved), "kind": "mmh3"})
            elif job.get("kind") == "video":
                probe = probe_fn(resolved)
                estimate = estimate_audio_pcm_bytes(
                    probe.duration_seconds,
                    sample_rate=settings.audio_sample_rate,
                    channels=settings.audio_channels,
                )
                item["probe"] = probe.to_dict()
                item["estimated_audio_pcm_bytes"] = estimate
                archive_path = session_dir / f"{position:06d}_{job.get('job_id') or 'job'}.mmh3"
                result = import_fn(resolved, archive_path=archive_path, settings=settings, probe=probe)
                item["status"] = "imported"
                item["import_report"] = result.report
                total_pcm_bytes += int(result.report["audio"]["output_pcm_bytes"])
                output_jobs.append({"index": len(output_jobs), "job_id": job.get("job_id"), "path": str(result.archive_path), "kind": "mmh3"})
            else:
                raise MMH3ResourceError(f"unsupported job kind {job.get('kind')!r}")
        except (MMH3Error, OSError) as exc:
            item["reasons"].append(str(exc))
        items.append(item)
    rejected = [item for item in items if item["status"] == "rejected"]
    normalized_plan = {
        "contract": "mmh3_batch_input_plan_v1",
        "order": "provided",
        "jobs": output_jobs,
    }
    report = {
        "contract": RAW_IMPORT_CONTRACT,
        "ready": len(output_jobs) >= 2 and (not rejected or error_policy == "skip_invalid"),
        "source_contract": plan["contract"],
        "error_policy": error_policy,
        "conform": settings.to_dict(),
        "stitch_canonical": (
            settings.fps == FPS
            and settings.audio_sample_rate == AUDIO_SAMPLE_RATE
            and settings.audio_channels == 2
        ),
        "counts": {"planned": len(items), "accepted": len(output_jobs), "skipped": len(rejected)},
        "estimate": {
            "full_pcm_materialization_required": True,
            "configured_per_item_pcm_limit_bytes": settings.max_audio_pcm_bytes,
            "accepted_output_pcm_bytes_total": total_pcm_bytes,
            "rgb_full_video_materialization": False,
        },
        "items": items,
    }
    if rejected and error_policy == "stop_on_error":
        details = "; ".join(f"item {item['position']}: {', '.join(item['reasons'])}" for item in rejected)
        raise MMH3ResourceError("Raw batch import stopped by per-item validation: " + details)
    if len(output_jobs) < 2:
        raise MMH3ResourceError(f"Raw batch import requires at least two accepted inputs; got {len(output_jobs)}")
    return normalized_plan, report


__all__ = [
    "RAW_IMPORT_CONTRACT", "RAW_IMPORT_SETTINGS_CONTRACT", "PIXEL_FORMAT", "AUDIO_PCM_WORKING_SET_MULTIPLIER",
    "RawVideoProbe", "ConformSettings", "RawImportResult", "canonical_pcm_samples",
    "audio_pcm_bytes", "estimate_audio_pcm_bytes", "validate_target_overrides", "canvas_fit",
    "probe_raw_video", "make_file_backed_video", "import_raw_video", "normalize_batch_plan",
]
