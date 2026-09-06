from __future__ import annotations

import math

from .archive import get_resource_payload
from .core import MMH3Media
from .errors import MMH3ResourceError
from .resource_model import descriptor_from_media_metadata


def discard_stale_delivery(packet: MMH3Media) -> MMH3Media:
    """A replaced primary source cannot retain derived movies from its old revision."""
    source = packet.get_primary("video")
    if source is None:
        return packet
    out = packet
    for resource in packet.resources():
        info = resource.get("extensions", {}).get("mmh3_media", {}).get("decoded_upscale")
        if (isinstance(info, dict) and info.get("source_resource_id") == source["id"]
                and info.get("source_revision") != source["content"]["revision"]
                and resource["id"] not in packet.manifest["primary"].values()):
            out = out.remove(resource_id=resource["id"], record_history=False)
    return out


def upscale_decoded_video(packet: MMH3Media, *, enabled: bool = False, scale: float = 1.5,
                          quality: str = "HIGH", upscale=None, make_video=None):
    source = packet.get_primary("video")
    if source is None:
        raise MMH3ResourceError("Decoded upscale requires a primary video")
    if not math.isfinite(scale) or not 1.0 <= scale <= 4.0:
        raise MMH3ResourceError("Video upscale scale must be between 1 and 4")
    if quality not in {"LOW", "MEDIUM", "HIGH", "ULTRA"}:
        raise MMH3ResourceError("Unknown RTX VSR quality")
    video = get_resource_payload(packet, source)
    if not enabled or scale == 1.0:
        return packet, video, source["id"], "Original video"
    if upscale is None or make_video is None:
        raise MMH3ResourceError("RTX VSR is unavailable; select Original or install NVIDIA RTX nodes")
    color_space = video.get_color_space() if hasattr(video, "get_color_space") else "sRGB"
    if color_space != "sRGB":
        raise MMH3ResourceError("RTX VSR currently requires sRGB video; convert HDR explicitly first")
    bit_depth = video.get_bit_depth() if hasattr(video, "get_bit_depth") else 8
    components = video.get_components()
    images = components.images
    shape = tuple(images.shape)
    if len(shape) != 4 or shape[0] < 1 or shape[-1] != 3:
        raise MMH3ResourceError("RTX VSR requires a non-empty RGB video")
    frame_rate = components.frame_rate
    if float(frame_rate) <= 0:
        raise MMH3ResourceError("Video frame rate must be positive")
    expected_w = max(8, round(int(shape[2] * scale) / 8) * 8)
    expected_h = max(8, round(int(shape[1] * scale) / 8) * 8)
    output = upscale(images, scale, quality)
    if tuple(output.shape) != (shape[0], expected_h, expected_w, shape[-1]):
        raise MMH3ResourceError("RTX VSR changed frame count/channels or returned unexpected dimensions")
    # Audio and time base are passed through, never regenerated or resampled.
    result = make_video(output, components.audio, frame_rate, bit_depth=bit_depth)
    facts = {"dimensions": [expected_w, expected_h], "frame_count": shape[0],
             "fps": float(frame_rate), "duration": shape[0] / float(frame_rate)}
    info = {"backend": "RTXVideoSuperResolution", "scale": scale, "quality": quality,
            "source_resource_id": source["id"], "source_revision": source["content"]["revision"],
            "audio_policy": "source_passthrough", "frame_count": shape[0]}
    out, rid = packet.put_with_id(
        result, kind="video", role="auxiliary", name="Upscaled video", tags=["delivery", "upscaled"],
        descriptor=descriptor_from_media_metadata("video", facts),
        provenance={"input_resource_ids": [source["id"]], "operation": "decoded_upscale"},
        extensions={"mmh3_media": {"decoded_upscale": info}}, record_history=False,
    )
    # Preserve primary media and H3 generation geometry: the saved sampler state still owns continuation.
    out = out.record_operation("decoded_upscale", input_resource_ids=[source["id"]],
                               output_resource_ids=[rid], **info)
    return out, result, rid, f"RTX VSR · {expected_w}×{expected_h} · original audio"
