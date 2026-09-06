from __future__ import annotations

from typing import Any


def describe_media_payload(payload: Any, kind: str) -> dict[str, Any]:
    """Return cheap JSON metadata from an already-present typed payload without encoding or saving it."""
    metadata: dict[str, Any] = {}
    shape = getattr(payload, "shape", None)
    if shape is not None:
        try:
            metadata["shape"] = [int(value) for value in shape]
        except (TypeError, ValueError):
            pass

    if kind == "image" and shape is not None:
        try:
            if len(shape) >= 3:
                # ComfyUI IMAGE tensors are [B, H, W, C]. Keep explicit dimensions
                # alongside shape so manifest-only consumers do not need tensor knowledge.
                metadata["dimensions"] = [int(shape[-2]), int(shape[-3])]
        except (TypeError, ValueError, IndexError):
            pass

    if kind == "audio" and isinstance(payload, dict):
        waveform = payload.get("waveform")
        sample_rate = payload.get("sample_rate")
        wave_shape = getattr(waveform, "shape", None)
        try:
            rate = int(sample_rate)
            samples = int(wave_shape[-1])
            if rate > 0 and samples > 0:
                metadata.update(
                    {
                        "sample_rate": rate,
                        "samples": samples,
                        "duration": samples / rate,
                        "channels": int(wave_shape[-2]) if len(wave_shape) >= 2 else None,
                    }
                )
        except (TypeError, ValueError, IndexError):
            pass

    if kind == "video":
        for key, method in (
            ("dimensions", "get_dimensions"),
            ("duration", "get_duration"),
            ("frame_count", "get_frame_count"),
            ("bit_depth", "get_bit_depth"),
        ):
            if hasattr(payload, method):
                try:
                    value = getattr(payload, method)()
                    if key == "dimensions" and isinstance(value, (tuple, list)) and len(value) >= 2:
                        metadata[key] = [int(value[0]), int(value[1])]
                    elif key == "duration":
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except Exception:
                    pass
        if hasattr(payload, "get_components") and not bool(
            getattr(payload, "_mmh3_skip_components_metadata", False)
        ):
            try:
                components = payload.get_components()
                images = getattr(components, "images", None)
                frame_rate = float(getattr(components, "frame_rate", 0.0))
                image_shape = getattr(images, "shape", None)
                if image_shape is not None and len(image_shape) == 4:
                    metadata.setdefault("shape", [int(value) for value in image_shape])
                    metadata.setdefault("dimensions", [int(image_shape[2]), int(image_shape[1])])
                    metadata.setdefault("frame_count", int(image_shape[0]))
                    if frame_rate > 0:
                        metadata.setdefault("duration", int(image_shape[0]) / frame_rate)
                        metadata.setdefault("fps", frame_rate)
            except Exception:
                pass
    return metadata
