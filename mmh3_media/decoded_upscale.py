from __future__ import annotations

import math
import copy

from .archive import get_resource_payload
from .core import MMH3Media
from .errors import MMH3ResourceError
from .resource_model import descriptor_from_media_metadata


class _AudioOverrideVideo:
    """Defer RGB decoding until a consumer requests pixels or saves the video."""

    _mmh3_skip_components_metadata = True

    def __init__(self, source, audio):
        self.source, self.audio = source, audio

    def get_components(self):
        components = copy.copy(self.source.get_components())
        components.audio = self.audio
        return components

    def get_bit_depth(self):
        return self.source.get_bit_depth() if hasattr(self.source, "get_bit_depth") else 8

    def get_color_space(self):
        return self.source.get_color_space() if hasattr(self.source, "get_color_space") else "sRGB"

    def __getattr__(self, name):
        # Never expose the source stream: that would bypass the audio override.
        if name in {"get_dimensions", "get_duration", "get_frame_count", "get_frame_rate"}:
            return getattr(self.source, name)
        raise AttributeError(name)

    def _materialized(self):
        from comfy_api.latest import InputImpl
        return InputImpl.VideoFromComponents(self.get_components(),
                                             bit_depth=self.get_bit_depth(),
                                             color_space=self.get_color_space())

    def save_to(self, *args, **kwargs):
        return self._materialized().save_to(*args, **kwargs)

    def as_trimmed(self, *args, **kwargs):
        return self._materialized().as_trimmed(*args, **kwargs)

    def as_cropped(self, *args, **kwargs):
        return self._materialized().as_cropped(*args, **kwargs)


def discard_stale_delivery(packet: MMH3Media) -> MMH3Media:
    """Drop delivery movies whose recorded video or audio input is stale."""
    source = packet.get_primary("video")
    if source is None:
        return packet
    out = packet
    for resource in packet.resources():
        info = resource.get("extensions", {}).get("mmh3_media", {}).get("decoded_upscale")
        if not isinstance(info, dict) or resource["id"] in packet.manifest["primary"].values():
            continue
        stale_video = (info.get("source_resource_id") == source["id"]
                       and info.get("source_revision") != source["content"]["revision"])
        audio_id = info.get("audio_resource_id")
        audio = packet.get_by_id(audio_id) if audio_id else None
        stale_audio = bool(audio_id) and (audio is None or
                                         info.get("audio_revision") != audio["content"]["revision"])
        if stale_video or stale_audio:
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
    audio_resource = packet.get_primary("audio")
    rescale = enabled and scale != 1.0
    if not rescale and audio_resource is None:
        return packet, video, source["id"], "Original video"
    if not rescale:
        result = _AudioOverrideVideo(video, get_resource_payload(packet, audio_resource))
        input_ids = [source["id"], audio_resource["id"]]
        info = {"backend": "Original", "scale": 1.0,
                "source_resource_id": source["id"], "source_revision": source["content"]["revision"],
                "audio_resource_id": audio_resource["id"], "audio_revision": audio_resource["content"]["revision"],
                "audio_policy": "source_passthrough"}
        out, rid = packet.put_with_id(
            result, kind="video", role="auxiliary", name="Original video with selected audio",
            tags=["delivery"], descriptor=copy.deepcopy(source.get("descriptor", {})),
            provenance={"input_resource_ids": input_ids, "operation": "decoded_upscale"},
            extensions={"mmh3_media": {"decoded_upscale": info}}, record_history=False)
        out = out.record_operation("decoded_upscale", input_resource_ids=input_ids,
                                   output_resource_ids=[rid], **info)
        return out, result, rid, "Original video · selected source audio"
    if (rescale and upscale is None) or make_video is None:
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
    expected_w = max(8, round(int(shape[2] * scale) / 8) * 8) if rescale else shape[2]
    expected_h = max(8, round(int(shape[1] * scale) / 8) * 8) if rescale else shape[1]
    output = upscale(images, scale, quality) if rescale else images
    if tuple(output.shape) != (shape[0], expected_h, expected_w, shape[-1]):
        raise MMH3ResourceError("RTX VSR changed frame count/channels or returned unexpected dimensions")
    # Audio and time base are passed through, never regenerated or resampled.
    audio = get_resource_payload(packet, audio_resource) if audio_resource else components.audio
    result = make_video(output, audio, frame_rate, bit_depth=bit_depth)
    facts = {"dimensions": [expected_w, expected_h], "frame_count": shape[0],
             "fps": float(frame_rate), "duration": shape[0] / float(frame_rate)}
    info = {"backend": "RTXVideoSuperResolution" if rescale else "Original", "scale": scale if rescale else 1.0, "quality": quality,
            "source_resource_id": source["id"], "source_revision": source["content"]["revision"],
            "audio_policy": "source_passthrough", "frame_count": shape[0]}
    input_ids = [source["id"]]
    if audio_resource:
        input_ids.append(audio_resource["id"])
        info.update(audio_resource_id=audio_resource["id"], audio_revision=audio_resource["content"]["revision"])
    out, rid = packet.put_with_id(
        result, kind="video", role="auxiliary", name="Upscaled video" if rescale else "Original video with selected audio",
        tags=["delivery", "upscaled"] if rescale else ["delivery"],
        descriptor=descriptor_from_media_metadata("video", facts),
        provenance={"input_resource_ids": input_ids, "operation": "decoded_upscale"},
        extensions={"mmh3_media": {"decoded_upscale": info}}, record_history=False,
    )
    # Preserve primary media and H3 generation geometry: the saved sampler state still owns continuation.
    out = out.record_operation("decoded_upscale", input_resource_ids=input_ids,
                               output_resource_ids=[rid], **info)
    label = "RTX VSR" if rescale else "Original video"
    return out, result, rid, f"{label} · {expected_w}×{expected_h} · selected source audio"
