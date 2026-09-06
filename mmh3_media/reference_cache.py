from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .archive import get_resource_payload
from .core import MMH3Media
from .errors import MMH3ResourceError
from .reference_cost import ReferenceCostItem, estimate_reference_cost
from .h3_resource_semantics import cache_contract, reference_contract


REFERENCE_CACHE_VERSION = 1
REFERENCE_CACHE_SCOPE = "minimax_h3_native_vae_reference_block"


def _source_token(descriptor: dict[str, Any]) -> tuple[str, str]:
    content = descriptor.get("content") if isinstance(descriptor.get("content"), dict) else {}
    revision = content.get("revision")
    if isinstance(revision, str) and revision:
        return "revision", revision
    digest = content.get("digest")
    if isinstance(digest, str) and digest:
        return "digest", digest
    raise MMH3ResourceError(
        f"Reference {descriptor.get('id')!r} has neither content revision nor digest; "
        "re-put it with the current writer or save it before creating a persistent exact cache"
    )


@dataclass(frozen=True)
class ReferenceCacheSpec:
    key: str
    source_resource_id: str
    source_token_kind: str
    source_token: str
    source_kind: str
    source_role: str
    ref_image_size: str
    target_width: int
    target_height: int
    target_frames: int
    effective_width: int
    effective_height: int
    effective_frames: int | None
    video_latent_t: int
    vae_fingerprint: str
    native_contract_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": REFERENCE_CACHE_VERSION,
            "scope": REFERENCE_CACHE_SCOPE,
            "exactness": "exact_vae_block_only_qwen_not_cached",
            "key": self.key,
            "source": {
                "resource_id": self.source_resource_id,
                "token_kind": self.source_token_kind,
                "token": self.source_token,
                "kind": self.source_kind,
                "role": self.source_role,
            },
            "processing": {
                "ref_image_size": self.ref_image_size,
                "target_width": self.target_width,
                "target_height": self.target_height,
                "target_frames": self.target_frames,
                "effective_width": self.effective_width,
                "effective_height": self.effective_height,
                "effective_frames": self.effective_frames,
                "video_latent_t": self.video_latent_t,
                "fps": 24.0,
                "canvas_multiple": 32,
            },
            "encoder": {
                "vae_fingerprint": self.vae_fingerprint,
                "native_contract_fingerprint": self.native_contract_fingerprint,
            },
        }


@dataclass(frozen=True)
class ReferenceCacheLookup:
    spec: ReferenceCacheSpec
    descriptor: dict[str, Any] | None
    stale_resource_ids: tuple[str, ...]

    @property
    def hit(self) -> bool:
        return self.descriptor is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit": self.hit,
            "spec": self.spec.to_dict(),
            "cache_resource": self.descriptor,
            "stale_resource_ids": list(self.stale_resource_ids),
        }


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_reference_cache_spec(
    packet: MMH3Media,
    source_resource_id: str,
    *,
    target_width: int,
    target_height: int,
    target_frames: int,
    ref_image_size: str,
    vae_fingerprint: str,
    native_contract_fingerprint: str,
) -> ReferenceCacheSpec:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    source_resource_id = str(source_resource_id or "").strip()
    raw_descriptor = packet.get_by_id(source_resource_id)
    if raw_descriptor is None:
        raise MMH3ResourceError(f"Reference source {source_resource_id!r} does not exist")
    descriptor = packet.ref(source_resource_id).descriptor
    reference_contract(descriptor)
    if descriptor.get("kind") not in ("image", "video"):
        raise MMH3ResourceError("VAE reference cache supports image or video reference sources")
    if not str(vae_fingerprint or "").strip():
        raise MMH3ResourceError("vae_fingerprint is required; a filename alone is not an exact model identity")
    if not str(native_contract_fingerprint or "").strip():
        raise MMH3ResourceError("native_contract_fingerprint is required")
    token_kind, token = _source_token(raw_descriptor)
    report = estimate_reference_cost(
        (descriptor,),
        target_width=int(target_width),
        target_height=int(target_height),
        target_frames=int(target_frames),
        ref_image_size=ref_image_size,
    )
    item: ReferenceCostItem = report.items[0]
    if not item.estimate_complete or not item.effective_width or not item.effective_height:
        raise MMH3ResourceError(
            f"Reference {source_resource_id!r} lacks complete geometry/timing metadata required for an exact cache"
        )
    unsigned = {
        "version": REFERENCE_CACHE_VERSION,
        "scope": REFERENCE_CACHE_SCOPE,
        "source": {
            "resource_id": descriptor["id"],
            "token_kind": token_kind,
            "token": token,
            "kind": descriptor["kind"],
            "role": descriptor["role"],
        },
        "processing": {
            "ref_image_size": ref_image_size,
            "target_width": int(target_width),
            "target_height": int(target_height),
            "target_frames": int(target_frames),
            "effective_width": item.effective_width,
            "effective_height": item.effective_height,
            "effective_frames": item.effective_frames,
            "video_latent_t": item.video_latent_t,
            "fps": 24.0,
            "canvas_multiple": 32,
        },
        "encoder": {
            "vae_fingerprint": str(vae_fingerprint).strip(),
            "native_contract_fingerprint": str(native_contract_fingerprint).strip(),
        },
    }
    return ReferenceCacheSpec(
        _fingerprint_payload(unsigned),
        descriptor["id"],
        token_kind,
        token,
        descriptor["kind"],
        descriptor["role"],
        ref_image_size,
        int(target_width),
        int(target_height),
        int(target_frames),
        int(item.effective_width),
        int(item.effective_height),
        item.effective_frames,
        int(item.video_latent_t),
        str(vae_fingerprint).strip(),
        str(native_contract_fingerprint).strip(),
    )


def lookup_reference_cache(packet: MMH3Media, spec: ReferenceCacheSpec) -> ReferenceCacheLookup:
    hit = None
    stale: list[str] = []
    for raw_descriptor in packet.resources():
        descriptor = packet.ref(raw_descriptor["id"]).descriptor
        if descriptor.get("role") != "intermediate" or descriptor.get("kind") != "latent":
            continue
        cache = cache_contract(descriptor)
        source = cache.get("source") if isinstance(cache, dict) else None
        if not isinstance(source, dict) or source.get("resource_id") != spec.source_resource_id:
            continue
        if cache.get("key") == spec.key:
            hit = raw_descriptor
        else:
            stale.append(raw_descriptor["id"])
    return ReferenceCacheLookup(spec, hit, tuple(stale))


def put_reference_cache(packet: MMH3Media, spec: ReferenceCacheSpec, latent: Any) -> tuple[MMH3Media, str]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise MMH3ResourceError("Reference cache latent must be a LATENT dict containing samples")
    samples = latent["samples"]
    shape = getattr(samples, "shape", None)
    if shape is None or len(shape) != 5:
        raise MMH3ResourceError(f"Reference cache samples must be [B,24,T,H,W], got {shape}")
    if int(shape[0]) != 1 or int(shape[1]) != 24:
        raise MMH3ResourceError(f"Reference cache requires one H3 video latent with 24 channels, got {tuple(shape)}")
    expected_h, expected_w = spec.effective_height // 16, spec.effective_width // 16
    if (int(shape[3]), int(shape[4])) != (expected_h, expected_w):
        raise MMH3ResourceError(
            f"Cache latent spatial shape {(int(shape[3]), int(shape[4]))} does not match fingerprinted "
            f"native VAE shape {(expected_h, expected_w)}"
        )
    if spec.source_kind == "video" and int(shape[2]) != spec.video_latent_t:
        raise MMH3ResourceError(
            f"Cache video latent T={int(shape[2])} does not match fingerprinted native T={spec.video_latent_t}"
        )
    lookup = lookup_reference_cache(packet, spec)
    extensions = {"minimax_h3": {"cache": spec.to_dict()}}
    descriptor = {"tensor": {"shape": [int(value) for value in shape]}}
    existing_id = None if lookup.descriptor is None else str(lookup.descriptor["id"])
    return packet.put_with_id(
        latent, kind="latent", role="intermediate", order=None,
        mode="replace" if existing_id else "add", resource_id=existing_id or "",
        descriptor=descriptor, extensions=extensions,
    )


def materialize_reference_cache(packet: MMH3Media, lookup: ReferenceCacheLookup) -> Any:
    if lookup.descriptor is None:
        raise MMH3ResourceError("Reference cache miss")
    return get_resource_payload(packet, lookup.descriptor)


def compare_reference_cache_latents(original: Any, cached: Any) -> dict[str, Any]:
    """Compare two typed latent payloads without tolerating shape/dtype drift."""
    if not isinstance(original, dict) or not isinstance(cached, dict):
        raise MMH3ResourceError("Both reference latents must be LATENT dictionaries")
    left, right = original.get("samples"), cached.get("samples")
    if left is None or right is None or not hasattr(left, "shape") or not hasattr(right, "shape"):
        raise MMH3ResourceError("Both reference latents must contain tensor-like samples")
    left_shape, right_shape = tuple(int(value) for value in left.shape), tuple(int(value) for value in right.shape)
    left_dtype, right_dtype = str(left.dtype), str(right.dtype)
    same_shape, same_dtype = left_shape == right_shape, left_dtype == right_dtype
    exact = False
    max_abs_error = None
    if same_shape:
        import torch

        exact = bool(torch.equal(left, right)) and same_dtype
        max_abs_error = float((left.to(torch.float32) - right.to(torch.float32)).abs().max().item())
    return {
        "exact": exact,
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "original_shape": list(left_shape),
        "cached_shape": list(right_shape),
        "original_dtype": left_dtype,
        "cached_dtype": right_dtype,
        "max_abs_error": max_abs_error,
    }
