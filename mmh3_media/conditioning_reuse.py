"""Bounded, process-only H3 preparation reuse. No packet resources or files written."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading
import weakref

import torch

from .util import json_dumps_canonical
from .h3 import is_nested_tensor, nested_parts


_tokens = {}
_sequence = 0
_token_lock = threading.RLock()


def _identity(value):
    global _sequence
    with _token_lock:
        entry = _tokens.get(id(value))
        if entry is not None and entry[0]() is value:
            return entry[1]
        address = id(value)
        try:
            reference = weakref.ref(value, lambda ref: _forget(address, ref))
        except TypeError:
            raise ValueError("Untrackable runtime object")
        _sequence += 1
        _tokens[address] = (reference, _sequence)
        return _sequence


def _forget(address, reference):
    with _token_lock:
        if _tokens.get(address, (None,))[0] is reference:
            _tokens.pop(address, None)


def _signature(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, bytes):
        return ["bytes", hashlib.sha256(value).hexdigest()]
    if isinstance(value, torch.Tensor):
        try:
            version = value._version
        except RuntimeError:
            # Comfy executes nodes in inference mode, whose tensors have no
            # mutation counter. Hash exact content in bounded CPU blocks instead.
            digest = hashlib.sha256()
            _hash_tensor(digest, value)
            version = digest.hexdigest()
        return ["tensor", _identity(value), version, str(value.dtype), str(value.device), list(value.shape)]
    if is_nested_tensor(value):
        return [_signature(v) for v in nested_parts(value)]
    if isinstance(value, dict):
        return {str(k): _signature(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_signature(v) for v in value]
    return ["object", _identity(value)]


def _hash_tensor(digest, value):
    if value.numel() <= 65536:
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        return
    dimension = next(i for i, size in enumerate(value.shape) if size > 1)
    size = max(1, 65536 // (value.numel() // value.shape[dimension]))
    for chunk in value.split(size, dim=dimension):
        _hash_tensor(digest, chunk)


def _model_signature(value):
    if value is None:
        return None
    patcher = getattr(value, "patcher", None)
    result = [_identity(value), _signature(getattr(value, "tokenizer", None)),
              _signature(getattr(value, "layer_idx", None))]
    if patcher is not None:
        result.extend([_identity(patcher), str(getattr(patcher, "patches_uuid", "")),
                       _signature(getattr(patcher, "model", None)),
                       _signature(getattr(patcher, "model_options", {})),
                       _signature(getattr(patcher, "object_patches", {}))])
    return result


def conditioning_key(packet, resolved, *, clip, video_vae, audio_vae, contract,
                     ref_image_size="match", ref_image_short_edge=None):
    """Seed, packet metadata and downstream sampler settings do not condition H3."""
    try:
        selected = []
        for resource in resolved.selected_resources:
            descriptor = packet.get_by_id(resource.resource_id)
            selected.append({"resource": resource.to_dict(), "content": descriptor["content"],
                             "live": _signature(packet.payloads.get(resource.resource_id))})
        value = {"contract": contract.fingerprint, "mode": resolved.mode,
                 "values": {k: resolved.value(k) for k in ("prompt", "width", "height", "frames", "fps")},
                 "resources": selected, "order": [r.to_dict() for r in resolved.reference_order],
                 "models": [_model_signature(m) for m in (clip, video_vae, audio_vae)],
                 "reference_size": [ref_image_size, ref_image_short_edge],
                 "reference_conditioning": packet.manifest.get("extensions", {}).get("minimax_h3", {}).get("reference_conditioning", "encoder_and_vae")}
        return hashlib.sha256(json_dumps_canonical(value).encode()).hexdigest()
    except (ValueError, TypeError, RuntimeError):
        return ""  # Unknown/mutable inputs safely run the native preparation.


@dataclass
class _TensorCopy:
    value: torch.Tensor
    device: torch.device


@dataclass
class _NestedCopy:
    kind: type
    parts: tuple


def _bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if is_nested_tensor(value):
        return sum(_bytes(v) for v in nested_parts(value))
    if isinstance(value, dict):
        return sum(_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_bytes(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        return 0
    raise ValueError("Opaque conditioning cannot be safely cloned")


def _copy(value, *, restore=False):
    if restore and isinstance(value, _TensorCopy):
        return value.value.to(device=value.device).clone()
    if restore and isinstance(value, _NestedCopy):
        return value.kind(tuple(_copy(v, restore=True) for v in value.parts))
    if isinstance(value, torch.Tensor):
        return _TensorCopy(value.detach().to(device="cpu").clone(), value.device)
    if is_nested_tensor(value):
        return _NestedCopy(type(value), tuple(_copy(v) for v in nested_parts(value)))
    if isinstance(value, dict):
        return {k: _copy(v, restore=restore) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_copy(v, restore=restore) for v in value)
    return value


class ConditioningMemory:
    def __init__(self, max_bytes=256 * 1024 * 1024, max_entries=4):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            try:
                copied = _copy(entry[1], restore=True)
            except (RuntimeError, ValueError, TypeError):
                self._entries.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return copied

    def put(self, key, positive, latent):
        if not key:
            return False
        try:
            count = _bytes((positive, latent))
            if count > self.max_bytes or self.max_entries < 1:
                return False
            with self._lock:
                self._entries.pop(key, None)
                while self._entries and (len(self._entries) >= self.max_entries
                        or sum(e[0] for e in self._entries.values()) + count > self.max_bytes):
                    self._entries.popitem(last=False)
                self._entries[key] = (count, _copy((positive, latent)))
            return True
        except (ValueError, RuntimeError, TypeError):
            return False

    def clear(self):
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def stats(self):
        with self._lock:
            return {"entries": len(self._entries), "bytes": sum(e[0] for e in self._entries.values()),
                    "max_bytes": self.max_bytes, "hits": self.hits, "misses": self.misses,
                    "storage": "memory_only"}


conditioning_memory = ConditioningMemory()
