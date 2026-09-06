from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .archive import get_resource_payload
from .constants import AUDIO_LATENT_FPS, FPS, H3_TEMPORAL_GRID
from .continuation import h3_video_t_from_frames
from .core import MMH3Media
from .errors import MMH3ResourceError
from .h3 import h3_expected_audio_t, make_nested_tensor, nested_parts, validate_h3_av_latent
from .h3_contract import h3_latent_contract_from_resource, validate_h3_latent_contract


@dataclass(frozen=True)
class H3LatentStitchCompatibility:
    ready: bool
    facts: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "contract": "mmh3_h3_latent_stitch_compatibility_v2",
            "segments": list(self.facts),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class H3LatentStitchPlan:
    output_frames: int
    output_video_t: int
    output_audio_t: int
    segments: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": True,
            "contract": "mmh3_h3_latent_stitch_v2",
            "assembly_domain": "h3_joint_av_latent",
            "ownership": "continuation_prefix_trim",
            "lineage_proof": "source_packet_id+latent_resource_id+latent_revision",
            "task_family_policy": "mixed_fl2va_ref2va_allowed",
            "output_frames": self.output_frames,
            "output_video_t": self.output_video_t,
            "output_audio_t": self.output_audio_t,
            "fps": FPS,
            "audio_latent_rate": AUDIO_LATENT_FPS,
            "segments": list(self.segments),
            "latent_blend_used": False,
            "naive_latent_concat_used": False,
            "audio_boundary_policy": "absolute_round(global_frame*40/24)",
        }

    def summary(self) -> str:
        return (
            f"READY · H3 latent stitch · {len(self.segments)} segments · "
            f"{self.output_frames}f · videoT={self.output_video_t} · audioT={self.output_audio_t}"
        )


@dataclass(frozen=True)
class H3LatentStitchResult:
    latent: dict[str, Any]
    plan: H3LatentStitchPlan


def _continuation_info(packet: MMH3Media) -> Mapping[str, Any] | None:
    last_process = _last_process(packet)
    if not isinstance(last_process, Mapping) or last_process.get("operation") != "continuation":
        return None
    info = last_process.get("info")
    return info if isinstance(info, Mapping) else None


def _last_process(packet: MMH3Media) -> Mapping[str, Any] | None:
    ext = packet.manifest.get("extensions", {})
    mmh3 = ext.get("mmh3_media") if isinstance(ext, Mapping) else None
    last_process = mmh3.get("last_process") if isinstance(mmh3, Mapping) else None
    return last_process if isinstance(last_process, Mapping) else None


def inspect_h3_latent_stitch_packets(packets: Sequence[MMH3Media]) -> H3LatentStitchCompatibility:
    if len(packets) < 2:
        return H3LatentStitchCompatibility(False, (), ("H3 latent stitch requires at least two ordered packets.",))
    reasons: list[str] = []
    facts: list[dict[str, Any]] = []
    expected_canvas: tuple[int, int] | None = None
    for index, packet in enumerate(packets):
        if not isinstance(packet, MMH3Media):
            reasons.append(f"Segment {index} is not an MMH3_MEDIA packet.")
            continue
        resource = packet.get_primary("latent")
        if resource is None or resource.get("kind") != "latent":
            reasons.append(f"Segment {index} has no primary H3 latent.")
            continue
        try:
            contract = h3_latent_contract_from_resource(resource)
        except MMH3ResourceError as exc:
            reasons.append(f"Segment {index}: {exc}")
            continue
        timeline = contract["timeline"]
        canvas = contract["canvas"]
        current_canvas = (int(canvas["width"]), int(canvas["height"]))
        if expected_canvas is None:
            expected_canvas = current_canvas
        elif current_canvas != expected_canvas:
            reasons.append(
                f"Segment {index} canvas {current_canvas[0]}x{current_canvas[1]} differs from "
                f"{expected_canvas[0]}x{expected_canvas[1]}."
            )
        if contract["batch"] != 1:
            reasons.append(f"Segment {index} requires batch=1 for latent stitch.")
        if contract["binding"]["geometry"] != "bound" or contract["binding"]["time"] != "bound":
            reasons.append(f"Segment {index} requires bound H3 geometry/time state.")
        if timeline.get("temporal_grid") != H3_TEMPORAL_GRID or not isinstance(timeline.get("frames"), int):
            reasons.append(f"Segment {index} must be on the stock H3 17k+5 temporal grid.")
        process = _last_process(packet) or {}
        process_info = process.get("info") if isinstance(process.get("info"), Mapping) else {}
        task_family = str(process_info.get("task_family") or "")
        if not task_family:
            mode = str(process.get("mode") or "")
            if mode == "ref2va":
                task_family = "ref2va"
            elif mode in {"t2va", "i2va", "l2va", "fl2va"}:
                task_family = "fl2va"
            else:
                task_family = "unknown"
        latent_revision = str((resource.get("content") or {}).get("revision") or "")
        if not latent_revision:
            reasons.append(f"Segment {index} latent is missing an immutable resource revision for lineage validation.")
        continuation = _continuation_info(packet) if index else None
        video_handover = 0
        audio_handover = 0
        source_packet_id = ""
        source_latent_resource_id = ""
        source_latent_revision = ""
        source_task_family = ""
        if index:
            if continuation is None:
                reasons.append(
                    f"Segment {index} is not provenance-linked as an MMH3 continuation; "
                    "independent H3 latents are not safe to stitch in latent space."
                )
            else:
                try:
                    video_handover = int(continuation.get("video_handover_frames"))
                    audio_handover = int(continuation.get("audio_handover_frames"))
                except (TypeError, ValueError):
                    reasons.append(f"Segment {index} continuation metadata has invalid handover lengths.")
                if video_handover <= 0 or audio_handover <= 0:
                    reasons.append(f"Segment {index} continuation metadata is missing positive AV handover lengths.")
                elif video_handover != audio_handover:
                    reasons.append(
                        f"Segment {index} uses different video/audio handovers "
                        f"({video_handover}f/{audio_handover}f); latent stitch v1 requires one shared AV ownership boundary."
                    )
                elif (video_handover - 5) % 17:
                    reasons.append(f"Segment {index} video handover {video_handover}f is not on the H3 17k+5 grid.")
                elif (video_handover * AUDIO_LATENT_FPS) % FPS:
                    reasons.append(
                        f"Segment {index} handover {video_handover}f is not an exact 40 Hz audio-latent boundary."
                    )
                source_packet_id = str(continuation.get("source_packet_id") or "")
                source_latent_resource_id = str(continuation.get("source_latent_resource_id") or "")
                source_latent_revision = str(continuation.get("source_latent_revision") or "")
                source_task_family = str(continuation.get("source_task_family") or "")
                missing_lineage = [
                    name
                    for name, value in (
                        ("source_packet_id", source_packet_id),
                        ("source_latent_resource_id", source_latent_resource_id),
                        ("source_latent_revision", source_latent_revision),
                    )
                    if not value
                ]
                if missing_lineage:
                    reasons.append(
                        f"Segment {index} continuation is missing source-lineage proof: "
                        + ", ".join(missing_lineage)
                        + ". Re-run continuation with the current MMH3 H3 Continuation Handover."
                    )
                elif facts:
                    previous = facts[-1]
                    expected = (
                        str(previous["packet_id"]),
                        str(previous["latent_resource_id"]),
                        str(previous["latent_revision"]),
                    )
                    actual = (source_packet_id, source_latent_resource_id, source_latent_revision)
                    if actual != expected:
                        reasons.append(
                            f"Segment {index} source-lineage proof does not point to segment {index - 1}: "
                            f"expected packet/resource/revision {expected}, got {actual}."
                        )
                    previous_family = str(previous.get("task_family") or "unknown")
                    if source_task_family and source_task_family != previous_family:
                        reasons.append(
                            f"Segment {index} source_task_family={source_task_family!r} does not match "
                            f"segment {index - 1} task_family={previous_family!r}."
                        )
                    source_frames = continuation.get("source_frames")
                    try:
                        if int(source_frames) != int(previous["frames"]):
                            reasons.append(
                                f"Segment {index} source_frames={source_frames} does not match "
                                f"segment {index - 1} frames={previous['frames']}."
                            )
                    except (TypeError, ValueError):
                        reasons.append(f"Segment {index} continuation metadata has invalid source_frames.")
        facts.append({
            "index": index,
            "packet_id": packet.manifest["id"],
            "latent_resource_id": resource["id"],
            "latent_revision": latent_revision,
            "origin": contract["origin"],
            "process_operation": str(process.get("operation") or ""),
            "process_mode": str(process.get("mode") or ""),
            "task_family": task_family,
            "canvas": list(current_canvas),
            "frames": timeline.get("frames"),
            "video_t": int(contract["streams"]["video"]["shape"][2]),
            "audio_t": int(contract["streams"]["audio"]["shape"][-1]),
            "absolute_start_frame": timeline.get("absolute_start_frame"),
            "video_handover_frames": video_handover,
            "audio_handover_frames": audio_handover,
            "source_packet_id": source_packet_id or None,
            "source_latent_resource_id": source_latent_resource_id or None,
            "source_latent_revision": source_latent_revision or None,
            "source_task_family": source_task_family or None,
            "family_transition": (
                None if index == 0 else f"{facts[-1].get('task_family', 'unknown')}->{task_family}"
            ),
        })
    return H3LatentStitchCompatibility(not reasons, tuple(facts), tuple(reasons))


def stitch_h3_continuation_latents(packets: Sequence[MMH3Media]) -> H3LatentStitchResult:
    compatibility = inspect_h3_latent_stitch_packets(packets)
    if not compatibility.ready:
        raise MMH3ResourceError("H3 latent stitch blocked: " + "; ".join(compatibility.reasons))

    videos: list[torch.Tensor] = []
    audios: list[torch.Tensor] = []
    segment_map: list[dict[str, Any]] = []
    output_frames = 0
    output_audio_t = 0

    for index, (packet, fact) in enumerate(zip(packets, compatibility.facts)):
        descriptor = packet.get_primary("latent")
        assert descriptor is not None
        latent = get_resource_payload(packet, descriptor)
        info = validate_h3_av_latent(latent, strict_audio_length=True)
        video, audio = nested_parts(latent["samples"])
        frames = int(fact["frames"])
        handover = int(fact["video_handover_frames"])

        if index == 0:
            video_piece = video
            audio_piece = audio
            frame_start = 0
            frame_end = frames
            video_trim_t = 0
            audio_trim_t = 0
            output_frames = frames
            output_audio_t = int(audio.shape[-1])
        else:
            video_trim_t = h3_video_t_from_frames(handover)
            audio_trim_t = handover * AUDIO_LATENT_FPS // FPS
            if video_trim_t >= int(video.shape[2]) or audio_trim_t >= int(audio.shape[-1]):
                raise MMH3ResourceError(f"Segment {index} handover consumes the entire latent segment")
            appended_frames = frames - handover
            frame_start = output_frames
            frame_end = output_frames + appended_frames
            video_piece = video[:, :, video_trim_t:]
            audio_tail = audio[..., audio_trim_t:]

            # H3 audio uses rounded 40 Hz length for a full 24 fps clip. Recompute each
            # seam against the absolute assembled timeline so per-clip rounding never drifts.
            target_audio_end = h3_expected_audio_t(frame_end)
            needed_audio_t = target_audio_end - output_audio_t
            if needed_audio_t < 1 or needed_audio_t > int(audio_tail.shape[-1]):
                raise MMH3ResourceError(
                    f"Segment {index} cannot satisfy absolute H3 audio boundary: need {needed_audio_t} latent ticks, "
                    f"tail provides {audio_tail.shape[-1]}"
                )
            audio_piece = audio_tail[..., :needed_audio_t]
            output_frames = frame_end
            output_audio_t = target_audio_end

        videos.append(video_piece)
        audios.append(audio_piece)
        segment_map.append({
            **fact,
            "output_frame_start": frame_start,
            "output_frame_end": frame_end,
            "video_trim_t": video_trim_t,
            "audio_trim_t": audio_trim_t,
            "appended_video_t": int(video_piece.shape[2]),
            "appended_audio_t": int(audio_piece.shape[-1]),
            "audio_tail_ticks_discarded_for_absolute_boundary": (
                0 if index == 0 else int(audio.shape[-1]) - audio_trim_t - int(audio_piece.shape[-1])
            ),
        })

    output_video = torch.cat(videos, dim=2)
    output_audio = torch.cat(audios, dim=-1)
    latent = {"samples": make_nested_tensor([output_video, output_audio])}
    info = validate_h3_av_latent(latent, strict_audio_length=True)
    if info.frames != output_frames:
        raise MMH3ResourceError(
            f"H3 latent stitch internal frame mismatch: tensor implies {info.frames}, plan says {output_frames}"
        )
    plan = H3LatentStitchPlan(
        output_frames=output_frames,
        output_video_t=int(output_video.shape[2]),
        output_audio_t=int(output_audio.shape[-1]),
        segments=tuple(segment_map),
    )
    return H3LatentStitchResult(latent, plan)
