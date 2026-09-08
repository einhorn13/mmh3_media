from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from .errors import MMH3ResourceError
from .h3 import (
    is_nested_tensor,
    make_nested_tensor,
    nested_parts,
    validate_h3_av_latent,
)
from .util import is_jsonable, safe_filename_component


def infer_kind(payload: Any) -> str:
    if isinstance(payload, dict):
        if "samples" in payload:
            return "latent"
        if "waveform" in payload and "sample_rate" in payload:
            return "audio"
    if isinstance(payload, torch.Tensor):
        if payload.ndim == 4 and payload.shape[-1] in (1, 3, 4):
            return "image"
        if payload.ndim in (2, 3):
            return "mask"
    if hasattr(payload, "save_to") and hasattr(payload, "get_components"):
        return "video"
    if isinstance(payload, (str, int, float, bool, list, tuple, dict)) or payload is None:
        if is_jsonable(payload):
            return "json"
    raise MMH3ResourceError(
        "Could not infer resource type. Expected LATENT, IMAGE, VIDEO, AUDIO, MASK, or JSON-compatible value."
    )



def validate_payload_contract(payload: Any, kind: str, *, descriptor: dict[str, Any] | None = None, extensions: dict[str, Any] | None = None) -> None:
    """Fail fast on payload shapes/types without performing serialization or decoding."""
    if kind == "latent":
        if not isinstance(payload, dict) or "samples" not in payload:
            raise MMH3ResourceError("LATENT resource must be a dict containing 'samples'")
        h3_ext = (extensions or {}).get("minimax_h3", {}) if isinstance(extensions, dict) else {}
        if isinstance(h3_ext, dict) and isinstance(h3_ext.get("latent"), dict):
            validate_h3_av_latent(payload)
            return
        samples = payload["samples"]
        if not isinstance(samples, torch.Tensor) and not is_nested_tensor(samples):
            raise MMH3ResourceError("LATENT 'samples' must be a tensor or NestedTensor")
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) or is_nested_tensor(value):
                continue
            if not is_jsonable(value):
                raise MMH3ResourceError(
                    f"LATENT field {key!r} is neither tensor-like nor JSON-safe; refusing to silently drop it"
                )
        return
    if kind == "image":
        if not isinstance(payload, torch.Tensor) or payload.ndim != 4:
            raise MMH3ResourceError(f"IMAGE must be [B,H,W,C], got {getattr(payload, 'shape', None)}")
        if payload.shape[0] != 1:
            raise MMH3ResourceError(
                f"One MMH3 image resource represents one image; got batch={payload.shape[0]}. Split the batch into separate resources first."
            )
        if payload.shape[-1] not in (1, 3, 4):
            raise MMH3ResourceError(f"IMAGE channels must be 1, 3, or 4; got {payload.shape[-1]}")
        return
    if kind == "mask":
        if not isinstance(payload, torch.Tensor) or payload.ndim not in (2, 3):
            raise MMH3ResourceError(f"MASK must be [H,W] or [B,H,W], got {getattr(payload, 'shape', None)}")
        return
    if kind == "audio":
        if not isinstance(payload, dict) or "waveform" not in payload or "sample_rate" not in payload:
            raise MMH3ResourceError("AUDIO must contain waveform and sample_rate")
        waveform = payload["waveform"]
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1:
            raise MMH3ResourceError(
                f"AUDIO waveform must be [1,C,T] for one media asset, got {getattr(waveform, 'shape', None)}"
            )
        if waveform.shape[1] < 1 or waveform.shape[1] > 32:
            raise MMH3ResourceError(f"Unsupported audio channel count: {waveform.shape[1]}")
        try:
            sample_rate = int(payload["sample_rate"])
        except Exception as e:
            raise MMH3ResourceError("AUDIO sample_rate must be an integer") from e
        if sample_rate <= 0:
            raise MMH3ResourceError("AUDIO sample_rate must be positive")
        return
    if kind == "video":
        if not hasattr(payload, "save_to"):
            raise MMH3ResourceError("VIDEO resource does not expose ComfyUI's save_to() API")
        return
    if kind == "json":
        if not is_jsonable(payload):
            raise MMH3ResourceError("JSON resource payload is not JSON serializable")
        return
    raise MMH3ResourceError(f"No serializer for resource kind {kind!r}")

def canonical_v03_archive_path(res: dict[str, Any], *, video_suffix: str | None = None) -> str:
    """Stable schema-2 member path based on resource identity, not semantic ordering."""
    kind = str(res["kind"])
    rid = safe_filename_component(str(res["id"]))
    ext = {
        "latent": ".safetensors",
        "image": ".png",
        "video": video_suffix or ".mp4",
        "audio": ".wav",
        "mask": ".safetensors",
        "json": ".json",
    }.get(kind, ".bin")
    return f"resources/{safe_filename_component(kind)}/{rid}{ext}"


def _tensor_field_to_safetensors(
    key: str,
    value: Any,
    tensors: dict[str, torch.Tensor],
    field_layout: dict[str, Any],
    *,
    h3_samples: bool = False,
) -> None:
    if isinstance(value, torch.Tensor):
        name = key if key != "samples" else "samples"
        tensors[name] = value.detach().to(device="cpu").contiguous()
        field_layout[key] = {"type": "tensor", "keys": [name]}
        return
    if is_nested_tensor(value):
        parts = nested_parts(value)
        names = []
        if h3_samples and len(parts) == 2:
            tensor_names = ["video", "audio"]
        else:
            tensor_names = [f"{key}.{i}" for i in range(len(parts))]
        for name, part in zip(tensor_names, parts):
            if not isinstance(part, torch.Tensor):
                raise MMH3ResourceError(f"LATENT field {key!r} contains a non-tensor nested part")
            tensors[name] = part.detach().to(device="cpu").contiguous()
            names.append(name)
        field_layout[key] = {"type": "nested", "keys": names}
        return
    raise MMH3ResourceError(f"LATENT field {key!r} is not a tensor/NestedTensor")


def _serialization_descriptor(res: dict[str, Any]) -> dict[str, Any]:
    descriptor = res.get("descriptor")
    if not isinstance(descriptor, dict):
        return {}
    storage = descriptor.get("serialization")
    return storage if isinstance(storage, dict) else {}


def _has_h3_latent_contract(res: dict[str, Any]) -> bool:
    extensions = res.get("extensions")
    if not isinstance(extensions, dict):
        return False
    h3 = extensions.get("minimax_h3")
    return isinstance(h3, dict) and isinstance(h3.get("latent"), dict)


def serialize_latent(payload: dict, path: Path, *, h3: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict) or "samples" not in payload:
        raise MMH3ResourceError("LATENT resource must be a dict containing 'samples'")
    if h3:
        validate_h3_av_latent(payload)
    tensors: dict[str, torch.Tensor] = {}
    field_layout: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) or is_nested_tensor(value):
            _tensor_field_to_safetensors(key, value, tensors, field_layout, h3_samples=(h3 and key == "samples"))
        else:
            if not is_jsonable(value):
                raise MMH3ResourceError(
                    f"LATENT field {key!r} is neither tensor-like nor JSON-safe; refusing to silently drop it"
                )
            extras[key] = value
    if not tensors:
        raise MMH3ResourceError("LATENT contains no tensor fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_save_file(tensors, str(path))
    return {
        "serializer": "mmh3.safetensors.latent.v1",
        "media_type": "application/x-safetensors",
        "descriptor_patch": {
            "latent_layout": field_layout,
            **({"latent_extras": extras} if extras else {}),
        },
    }


def _load_owned_tensors(path: Path) -> dict[str, torch.Tensor]:
    # CPU safetensors may retain file-backed storage after load_file returns.
    # Archive extraction files are deleted immediately after deserialization,
    # so every returned tensor must own storage independent of that file.
    tensors = safe_load_file(str(path), device="cpu")
    for key in tensors:
        tensors[key] = tensors[key].clone()
    return tensors


def deserialize_latent(path: Path, res: dict[str, Any]) -> dict:
    tensors = _load_owned_tensors(path)
    storage = _serialization_descriptor(res)
    layout = storage.get("latent_layout")
    if not isinstance(layout, dict):
        raise MMH3ResourceError(f"Latent resource {res['id']} is missing required latent_layout descriptor")
    out: dict[str, Any] = {}
    for field, spec in layout.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("keys"), list):
            raise MMH3ResourceError(f"Invalid latent layout for field {field!r}")
        keys = spec["keys"]
        try:
            parts = [tensors[k] for k in keys]
        except KeyError as exc:
            raise MMH3ResourceError(f"Latent safetensors is missing field tensor {exc.args[0]!r}") from exc
        if spec.get("type") == "nested":
            out[field] = make_nested_tensor(parts)
        elif spec.get("type") == "tensor" and len(parts) == 1:
            out[field] = parts[0]
        else:
            raise MMH3ResourceError(f"Invalid latent field layout for {field!r}")
    extras = storage.get("latent_extras")
    if isinstance(extras, dict):
        out.update(extras)
    if _has_h3_latent_contract(res):
        validate_h3_av_latent(out)
    return out


def serialize_image(payload: torch.Tensor, path: Path) -> dict[str, Any]:
    if not isinstance(payload, torch.Tensor) or payload.ndim != 4:
        raise MMH3ResourceError(f"IMAGE must be [B,H,W,C], got {getattr(payload, 'shape', None)}")
    if payload.shape[0] != 1:
        raise MMH3ResourceError(
            f"One MMH3 image resource represents one image; got batch={payload.shape[0]}. Split the batch into separate resources first."
        )
    if payload.shape[-1] not in (1, 3, 4):
        raise MMH3ResourceError(f"IMAGE channels must be 1, 3, or 4; got {payload.shape[-1]}")
    arr = payload[0].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    arr = np.rint(arr * 255.0).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
        mode = "L"
    else:
        mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path, format="PNG", optimize=False)
    return {
        "serializer": "mmh3.png8.image.v1",
        "media_type": "image/png",
        "descriptor_patch": {
            "shape": list(payload.shape),
            "encoding": "png8",
            "note": "Comfy IMAGE tensors are archived as standard 8-bit PNG; latent/mask tensors remain exact safetensors.",
        },
    }


def deserialize_image(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        if im.mode not in ("L", "RGB", "RGBA"):
            im = im.convert("RGB")
        arr = np.asarray(im).copy()
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).to(dtype=torch.float32).div_(255.0).unsqueeze(0)


def serialize_mask(payload: torch.Tensor, path: Path) -> dict[str, Any]:
    if not isinstance(payload, torch.Tensor) or payload.ndim not in (2, 3):
        raise MMH3ResourceError(f"MASK must be [H,W] or [B,H,W], got {getattr(payload, 'shape', None)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_save_file({"mask": payload.detach().to(device="cpu").contiguous()}, str(path))
    return {
        "serializer": "mmh3.safetensors.mask.v1",
        "media_type": "application/x-safetensors",
        "descriptor_patch": {"shape": list(payload.shape), "dtype": str(payload.dtype)},
    }


def deserialize_mask(path: Path) -> torch.Tensor:
    data = _load_owned_tensors(path)
    if "mask" not in data:
        raise MMH3ResourceError("Mask safetensors is missing 'mask'")
    return data["mask"]


def _write_float32_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    # RIFF/WAVE IEEE float32 (format code 3), exact for a float32 Comfy waveform.
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise MMH3ResourceError(
            f"AUDIO waveform must be [1,C,T] for one media asset, got {tuple(waveform.shape)}"
        )
    channels = int(waveform.shape[1])
    if channels < 1 or channels > 32:
        raise MMH3ResourceError(f"Unsupported audio channel count: {channels}")
    pcm = waveform[0].detach().to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous().numpy().astype("<f4", copy=False)
    data = pcm.tobytes(order="C")
    data_size = len(data)
    riff_size = 36 + data_size
    if riff_size > 0xFFFFFFFF:
        raise MMH3ResourceError("Audio is too large for RIFF WAV (>4 GiB); split it or use a video/container resource")
    block_align = channels * 4
    byte_rate = int(sample_rate) * block_align
    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 3, channels, int(sample_rate), byte_rate, block_align, 32),
            b"data",
            struct.pack("<I", data_size),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(data)


def _read_float32_wav(path: Path) -> tuple[torch.Tensor, int]:
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            raise MMH3ResourceError("Unsupported WAV: expected RIFF")
        _ = f.read(4)
        if f.read(4) != b"WAVE":
            raise MMH3ResourceError("Unsupported WAV: expected WAVE")
        fmt = None
        data = None
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                break
            raw_size = f.read(4)
            if len(raw_size) != 4:
                break
            size = struct.unpack("<I", raw_size)[0]
            payload = f.read(size)
            if size & 1:
                f.read(1)
            if chunk_id == b"fmt ":
                fmt = payload
            elif chunk_id == b"data":
                data = payload
        if fmt is None or data is None or len(fmt) < 16:
            raise MMH3ResourceError("Invalid WAV: missing fmt/data chunk")
        audio_format, channels, sample_rate, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
        if audio_format != 3 or bits != 32 or block_align != channels * 4:
            raise MMH3ResourceError(
                "This MMH3 audio loader expects the plugin's IEEE float32 WAV. "
                f"Got format={audio_format}, bits={bits}."
            )
        arr = np.frombuffer(data, dtype="<f4")
        if arr.size % channels:
            raise MMH3ResourceError("Invalid WAV data length")
        arr = arr.reshape(-1, channels).copy()
        waveform = torch.from_numpy(arr).transpose(0, 1).contiguous().unsqueeze(0)
        return waveform, int(sample_rate)


def serialize_audio(payload: dict, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or "waveform" not in payload or "sample_rate" not in payload:
        raise MMH3ResourceError("AUDIO must contain waveform and sample_rate")
    waveform = payload["waveform"]
    sample_rate = int(payload["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        raise MMH3ResourceError("AUDIO waveform must be a torch tensor")
    if sample_rate <= 0:
        raise MMH3ResourceError("AUDIO sample_rate must be positive")
    _write_float32_wav(path, waveform, sample_rate)
    return {
        "serializer": "mmh3.wav.float32.v1",
        "media_type": "audio/wav",
        "descriptor_patch": {
            "sample_rate": sample_rate,
            "channels": int(waveform.shape[1]) if waveform.ndim >= 2 else None,
            "samples": int(waveform.shape[-1]),
            "encoding": "pcm_f32le",
        },
    }


def deserialize_audio(path: Path, res: dict[str, Any]) -> dict:
    serializer = res.get("serializer")
    if serializer != "mmh3.wav.float32.v1":
        raise MMH3ResourceError(
            f"Unsupported AUDIO serializer {serializer!r}"
        )
    waveform, sample_rate = _read_float32_wav(path)
    return {"waveform": waveform, "sample_rate": sample_rate}


def video_source_path(payload: Any) -> tuple[Path | None, str | None]:
    """Return an untouched on-disk source and suffix when byte-copy is semantically valid."""
    if not hasattr(payload, "get_stream_source"):
        return None, None
    try:
        if hasattr(payload, "get_active_trim_window"):
            start, duration = payload.get_active_trim_window()
            if float(start) != 0.0 or float(duration) != 0.0:
                return None, None
        src = payload.get_stream_source()
        if isinstance(src, (str, os.PathLike)):
            p = Path(src)
            if p.is_file():
                return p, p.suffix.lower() or ".mp4"
    except Exception:
        pass
    return None, None


def serialize_video(payload: Any, path: Path) -> dict[str, Any]:
    if not hasattr(payload, "save_to"):
        raise MMH3ResourceError("VIDEO resource does not expose ComfyUI's save_to() API")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.save_to(str(path))
    metadata: dict[str, Any] = {"container": path.suffix.lower().lstrip(".")}
    save_stats = getattr(payload, "last_save_stats", None)
    if isinstance(save_stats, dict):
        metadata["streaming_stats"] = {
            str(key): int(value)
            for key, value in save_stats.items()
            if isinstance(value, (int, np.integer))
        }
    for key, method in (
        ("dimensions", "get_dimensions"),
        ("duration", "get_duration"),
        ("frame_count", "get_frame_count"),
        ("bit_depth", "get_bit_depth"),
    ):
        if hasattr(payload, method):
            try:
                value = getattr(payload, method)()
                metadata[key] = list(value) if isinstance(value, tuple) else value
            except Exception:
                pass
    return {
        "serializer": "comfy.video.v1",
        "media_type": "video/mp4",
        "descriptor_patch": metadata,
    }


def deserialize_video(path: Path):
    try:
        from comfy_api.latest import InputImpl  # type: ignore
    except Exception as e:
        raise MMH3ResourceError("VIDEO extraction requires a current ComfyUI comfy_api.latest runtime") from e
    return InputImpl.VideoFromFile(str(path))


def serialize_json(payload: Any, path: Path) -> dict[str, Any]:
    if not is_jsonable(payload):
        raise MMH3ResourceError("JSON resource payload is not JSON serializable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "serializer": "mmh3.json.v1",
        "media_type": "application/json",
        "descriptor_patch": {},
    }


def deserialize_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_payload(payload: Any, res: dict[str, Any], path: Path) -> dict[str, Any]:
    kind = res["kind"]
    if kind == "latent":
        return serialize_latent(payload, path, h3=_has_h3_latent_contract(res))
    if kind == "image":
        return serialize_image(payload, path)
    if kind == "mask":
        return serialize_mask(payload, path)
    if kind == "audio":
        return serialize_audio(payload, path)
    if kind == "video":
        return serialize_video(payload, path)
    if kind == "json":
        return serialize_json(payload, path)
    raise MMH3ResourceError(f"No serializer for resource kind {kind!r}")


def deserialize_payload(path: Path, res: dict[str, Any]) -> Any:
    kind = res["kind"]
    if kind == "latent":
        return deserialize_latent(path, res)
    if kind == "image":
        return deserialize_image(path)
    if kind == "mask":
        return deserialize_mask(path)
    if kind == "audio":
        return deserialize_audio(path, res)
    if kind == "video":
        return deserialize_video(path)
    if kind == "json":
        return deserialize_json(path)
    raise MMH3ResourceError(f"No loader for resource kind {kind!r}")
