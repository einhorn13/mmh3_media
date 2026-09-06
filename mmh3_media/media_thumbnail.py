"""CPU-only single-frame thumbnails shared by Load and Save."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

from .errors import MMH3ResourceError


def thumbnail_webp(image: Image.Image) -> bytes:
    if image.width * image.height > 64_000_000:
        raise MMH3ResourceError("Image is too large for an automatic thumbnail")
    image = ImageOps.exif_transpose(image)
    image.thumbnail((640, 480), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.convert("RGB").save(output, format="WEBP", quality=80)
    return output.getvalue()


def thumbnail_from_file(source, kind: str) -> bytes:
    if kind == "image":
        try:
            with Image.open(source) as image:
                return thumbnail_webp(image)
        except Image.DecompressionBombError as exc:
            raise MMH3ResourceError("Image is too large for an automatic thumbnail") from exc
    if kind != "video":
        raise MMH3ResourceError("Automatic thumbnails support images and videos only")
    try:
        import av
    except ImportError as exc:
        raise MMH3ResourceError("Video thumbnails require PyAV (included with ComfyUI)") from exc
    # Explicit container demuxers exclude playlists/external-media URL resolution.
    suffix = Path(str(getattr(source, "name", source))).suffix.lower()
    container_format = {".mp4": "mov", ".mov": "mov", ".m4v": "mov", ".mkv": "matroska",
                        ".webm": "matroska", ".avi": "avi"}.get(suffix)
    if not container_format:
        raise MMH3ResourceError("Video container is not supported for automatic previews")
    # Only decode the first displayed frame, not all frames/get_components().
    try:
        with av.open(source, mode="r", format=container_format,
                     options={"protocol_whitelist": "file,pipe", "enable_drefs": "0"}) as container:
            if not container.streams.video:
                raise MMH3ResourceError("Video contains no video stream")
            stream = container.streams.video[0]
            if stream.width * stream.height > 64_000_000:
                raise MMH3ResourceError("Video frame is too large for an automatic thumbnail")
            for frame in container.decode(stream):
                return thumbnail_webp(frame.to_image())
    except av.error.FFmpegError as exc:
        raise MMH3ResourceError("Video frame could not be decoded") from exc
    raise MMH3ResourceError("Video contains no decodable frames")
