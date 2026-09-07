from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import torch

from .archive import get_resource_payload
from .control_contract import (
    H3_CONTROLNET_ALGORITHMS,
    ControlProviderProvenance,
    H3ControlConfiguration,
    control_configuration_from_dict,
    get_control_configuration,
)
from .control_preflight import H3_CONTROL_APPLY_NODE_ID
from .core import MMH3Media
from .errors import MMH3ResourceError
from .h3_resource_semantics import control_resource_kind, control_usage
from .util import deep_copy_json, json_dumps_canonical


H3_CONTROL_APPLY_PROCESS_CONTRACT = "mmh3_h3_control_apply_v1"

_PROVIDER_CAPABILITY_FIELDS = {
    "provider_id": "provider",
    "checkpoint_sha256": "checkpoint_sha256",
    "quantization": "quantization",
    "dtype": "dtype",
    "base_family": "base_family",
    "adaln_form": "adaln_form",
    "vae_fingerprint": "vae_fingerprint",
}


@dataclass(frozen=True)
class H3ControlProviderAdapter:
    algorithm: str
    quantization: str
    apply_node_id: str = H3_CONTROL_APPLY_NODE_ID


@dataclass(frozen=True)
class H3ControlApplyPlan:
    adapter: H3ControlProviderAdapter
    config: H3ControlConfiguration
    capability_fingerprint: str
    target_width: int
    target_height: int
    target_frames: int
    pass_through: bool
    capability: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": H3_CONTROL_APPLY_PROCESS_CONTRACT,
            "version": 1,
            "ready": True,
            "algorithm": self.adapter.algorithm,
            "quantization": self.adapter.quantization,
            "apply_node_id": self.adapter.apply_node_id,
            "capability_fingerprint": self.capability_fingerprint,
            "capability": deep_copy_json(dict(self.capability or {})),
            "target_width": self.target_width,
            "target_height": self.target_height,
            "target_frames": self.target_frames,
            "pass_through": self.pass_through,
            "applied": not self.pass_through,
            "control": self.config.to_dict(),
        }


@dataclass(frozen=True)
class H3ControlExpansion:
    positive: Any
    graph: dict[str, Any]
    plan: H3ControlApplyPlan


@dataclass(frozen=True)
class MaterializedControlVideo:
    frames: torch.Tensor
    resource_id: str
    role: str
    source_fps: float
    target_fps: float
    source_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "role": self.role,
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "source_frames": self.source_frames,
            "output_frames": int(self.frames.shape[0]),
            "width": int(self.frames.shape[2]),
            "height": int(self.frames.shape[1]),
            "spatial_resize": False,
            "temporal_alignment": "deferred_to_control_apply_policy",
        }

    def info_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


_PROVIDERS = {
    "fun_controlnet_union_bf16": H3ControlProviderAdapter("fun_controlnet_union_bf16", "bf16"),
    "fun_controlnet_union_int8_convrot": H3ControlProviderAdapter(
        "fun_controlnet_union_int8_convrot", "int8_convrot"
    ),
}


def control_provider_for_algorithm(algorithm: str) -> H3ControlProviderAdapter:
    try:
        return _PROVIDERS[str(algorithm or "").strip().lower()]
    except KeyError as exc:
        raise MMH3ResourceError(
            f"Algorithm {algorithm!r} has no H3 ControlNet provider adapter; expected one of {H3_CONTROLNET_ALGORITHMS}"
        ) from exc


def h3_fun_control_video_required(config: H3ControlConfiguration) -> bool:
    """Return whether the high-level H3 Fun owner must materialize structural control video.

    Structural-only control modes always require it. Inpaint owns mask/source-video as its
    primary contract and requests a supplemental structural stream only when a resource ID
    is explicitly configured. The only currently accepted supplemental kind is Pose, which
    is validated when that resource is materialized.
    """
    if not isinstance(config, H3ControlConfiguration):
        raise MMH3ResourceError("Expected an immutable H3ControlConfiguration")
    if config.control_kind == "inpaint":
        return bool(str(config.control_video_resource_id or "").strip())
    return True


def _capability_fingerprint(capability: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json_dumps_canonical(deep_copy_json(dict(capability))).encode("utf-8")
    ).hexdigest()


def _validate_bound_capability(
    config: H3ControlConfiguration,
    capability: Mapping[str, Any],
    fingerprint: str,
) -> None:
    if not capability:
        raise MMH3ResourceError("Control capability material is empty")
    if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
        raise MMH3ResourceError("Control capability lacks a lowercase SHA-256 fingerprint")
    if _capability_fingerprint(capability) != fingerprint:
        raise MMH3ResourceError("Control capability material does not match its fingerprint")
    for provider_field, capability_field in _PROVIDER_CAPABILITY_FIELDS.items():
        if getattr(config.provider, provider_field) != str(capability.get(capability_field) or ""):
            raise MMH3ResourceError(
                f"Control provider field {provider_field!r} does not match capability material"
            )


def build_control_pass_through_info(config: H3ControlConfiguration) -> dict[str, Any]:
    """Describe neutral control without loading or claiming a provider capability."""
    if not isinstance(config, H3ControlConfiguration):
        raise MMH3ResourceError("Expected an immutable H3ControlConfiguration")
    if config.strength != 0.0:
        raise MMH3ResourceError("Control pass-through info requires strength=0")
    adapter = control_provider_for_algorithm(config.effective_algorithm)
    return {
        "contract": H3_CONTROL_APPLY_PROCESS_CONTRACT,
        "version": 1,
        "ready": True,
        "algorithm": adapter.algorithm,
        "quantization": adapter.quantization,
        "apply_node_id": adapter.apply_node_id,
        "capability_fingerprint": config.provider.capability_fingerprint,
        "capability": {},
        "target_width": None,
        "target_height": None,
        "target_frames": None,
        "pass_through": True,
        "applied": False,
        "control": config.to_dict(),
    }


def bind_control_capability(
    config: H3ControlConfiguration, preflight_info: Mapping[str, Any]
) -> H3ControlConfiguration:
    if not isinstance(preflight_info, Mapping) or preflight_info.get("ready") is not True:
        raise MMH3ResourceError("H3 Control Apply requires a READY MMH3 Control Preflight report")
    facts = preflight_info.get("facts")
    if not isinstance(facts, Mapping):
        raise MMH3ResourceError("Control preflight report has no facts object")
    requested = str(facts.get("requested_algorithm") or "")
    effective = str(facts.get("effective_algorithm") or "")
    if requested != config.requested_algorithm or effective != config.effective_algorithm:
        raise MMH3ResourceError("Control preflight algorithm selection does not match the packet configuration")
    fingerprint = str(facts.get("capability_fingerprint") or "")
    capability = facts.get("capability")
    if not isinstance(capability, Mapping):
        raise MMH3ResourceError("Control preflight report has no capability material")
    checkpoint = facts.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    base = facts.get("base")
    base = base if isinstance(base, Mapping) else {}
    provider = replace(
        config.provider,
        capability_fingerprint=fingerprint,
        checkpoint_sha256=str(checkpoint.get("sha256") or config.provider.checkpoint_sha256),
        quantization=str(checkpoint.get("quantization") or config.provider.quantization),
        dtype=str(checkpoint.get("dtype") or config.provider.dtype),
        base_family=str(checkpoint.get("base_family") or base.get("family") or config.provider.base_family),
        adaln_form=str(checkpoint.get("adaln_form") or base.get("adaln_form") or config.provider.adaln_form),
        vae_fingerprint=str(base.get("vae_fingerprint") or config.provider.vae_fingerprint),
    )
    bound = replace(config, provider=provider)
    _validate_bound_capability(bound, capability, fingerprint)
    return bound


def build_control_apply_plan(
    config: H3ControlConfiguration, preflight_info: Mapping[str, Any]
) -> H3ControlApplyPlan:
    config = bind_control_capability(config, preflight_info)
    adapter = control_provider_for_algorithm(config.effective_algorithm)
    facts = preflight_info["facts"]
    capability = facts["capability"]
    if str(capability.get("quantization") or "") != adapter.quantization:
        raise MMH3ResourceError(
            f"Preflight quantization {capability.get('quantization')!r} does not match adapter {adapter.quantization!r}"
        )
    media = facts.get("media")
    media = media if isinstance(media, Mapping) else {}
    try:
        width = int(media["width"])
        height = int(media["height"])
        frames = int(media["target_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MMH3ResourceError("Control preflight report lacks exact target media geometry") from exc
    return H3ControlApplyPlan(
        adapter=adapter,
        config=config,
        capability_fingerprint=config.provider.capability_fingerprint,
        target_width=width,
        target_height=height,
        target_frames=frames,
        pass_through=config.strength == 0.0,
        capability=deep_copy_json(dict(capability)),
    )


def normalize_control_apply_process_info(
    packet: MMH3Media, value: Mapping[str, Any]
) -> tuple[dict[str, Any], H3ControlConfiguration]:
    """Admit one Apply report for atomic result packing.

    The packet carries the user configuration; the report may enrich only its
    provider proof. Result packing, not Apply, commits that proof to packet state.
    """
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    if not isinstance(value, Mapping):
        raise MMH3ResourceError("control_process_info must be an object")
    if value.get("contract") != H3_CONTROL_APPLY_PROCESS_CONTRACT or value.get("version") != 1:
        raise MMH3ResourceError(
            f"Unsupported control apply process contract {value.get('contract')!r}/{value.get('version')!r}"
        )
    if value.get("ready") is not True:
        raise MMH3ResourceError("Control apply process report is not READY")
    control = value.get("control")
    if not isinstance(control, Mapping):
        raise MMH3ResourceError("Control apply process report has no control configuration")
    candidate = control_configuration_from_dict(control)
    recorded = get_control_configuration(packet)
    if recorded is None:
        raise MMH3ResourceError("Packet has no configured F16 control contract")
    if candidate.provider.provider_id != recorded.provider.provider_id:
        raise MMH3ResourceError("Control apply provider ID does not match the packet configuration")
    if replace(candidate, provider=recorded.provider) != recorded:
        raise MMH3ResourceError("Control apply configuration does not match the packet configuration")
    for field in (
        "checkpoint_sha256",
        "quantization",
        "dtype",
        "base_family",
        "adaln_form",
        "vae_fingerprint",
    ):
        expected = getattr(recorded.provider, field)
        actual = getattr(candidate.provider, field)
        if expected and expected != actual:
            raise MMH3ResourceError(
                f"Control apply provider field {field!r} conflicts with the packet configuration"
            )
    adapter = control_provider_for_algorithm(candidate.effective_algorithm)
    if value.get("algorithm") != adapter.algorithm or value.get("quantization") != adapter.quantization:
        raise MMH3ResourceError("Control apply adapter identity does not match the effective algorithm")
    fingerprint = candidate.provider.capability_fingerprint
    if value.get("capability_fingerprint") != fingerprint:
        raise MMH3ResourceError("Control apply report fingerprint does not match its bound configuration")
    pass_through = candidate.strength == 0.0
    if value.get("pass_through") is not pass_through or value.get("applied") is pass_through:
        raise MMH3ResourceError("Control apply/off semantics are inconsistent with control strength")
    capability = value.get("capability")
    if pass_through:
        if candidate.provider != recorded.provider:
            raise MMH3ResourceError("Pass-through control report cannot enrich provider capability")
        if capability not in ({}, None):
            raise MMH3ResourceError("Pass-through control report must not claim materialized capability")
    else:
        if not isinstance(capability, Mapping):
            raise MMH3ResourceError("Control apply report has no canonical capability material")
        _validate_bound_capability(candidate, capability, fingerprint)
    normalized = deep_copy_json(dict(value))
    return normalized, candidate


def materialize_control_video(
    packet: MMH3Media,
    *,
    usage: str,
    resource_id: str = "",
    target_fps: float = 24.0,
) -> MaterializedControlVideo:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    if usage not in ("control_video", "inpaint_source"):
        raise MMH3ResourceError("Control video usage must be 'control_video' or 'inpaint_source'")
    config = get_control_configuration(packet)
    if config is None:
        raise MMH3ResourceError("Packet has no configured F16 control contract")
    configured_id = (
        config.control_video_resource_id if usage == "control_video" else config.inpaint_source_resource_id
    )
    selected_id = str(resource_id or "").strip() or configured_id
    if not selected_id:
        raise MMH3ResourceError(f"Control configuration has no resource ID for usage {usage!r}")
    if configured_id and selected_id != configured_id:
        raise MMH3ResourceError(
            f"Selected {usage} resource {selected_id!r} does not match configured ID {configured_id!r}"
        )
    raw_descriptor = packet.get_by_id(selected_id)
    if raw_descriptor is None:
        raise MMH3ResourceError(f"Control resource {selected_id!r} does not exist")
    descriptor = packet.ref(selected_id).descriptor
    actual_usage = control_usage(descriptor)
    if actual_usage != usage or descriptor.get("kind") != "video":
        raise MMH3ResourceError(
            f"Control resource {selected_id!r} is {descriptor.get('role')}/{descriptor.get('kind')} with H3 usage {actual_usage!r}, expected usage={usage!r}/video"
        )
    if usage == "control_video":
        declared_kind = control_resource_kind(descriptor)
        if declared_kind is None:
            expected = "pose" if config.control_kind == "inpaint" else config.control_kind
            raise MMH3ResourceError(
                f"Control resource {selected_id!r} has no structural control_kind metadata; "
                f"declare {expected!r} in extensions.minimax_h3.control.control_kind"
            )
        if config.control_kind == "inpaint":
            if declared_kind != "pose":
                raise MMH3ResourceError(
                    f"Inpaint supplemental control supports only Pose, but resource {selected_id!r} "
                    f"declares control_kind={declared_kind!r}"
                )
        elif config.control_kind in ("canny", "depth", "hed", "mlsd", "pose"):
            if declared_kind != config.control_kind:
                raise MMH3ResourceError(
                    f"Control resource {selected_id!r} declares control_kind={declared_kind!r}, "
                    f"but MMH3 Control Configure selected {config.control_kind!r}; hidden mode substitution is forbidden"
                )
        else:
            raise MMH3ResourceError(
                f"Control video resource {selected_id!r} is not legal for control_kind={config.control_kind!r}"
            )
    try:
        target_fps_value = float(target_fps)
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError("Control target_fps must be numeric") from exc
    if target_fps_value <= 0 or not math.isfinite(target_fps_value):
        raise MMH3ResourceError("Control target_fps must be a positive finite value")
    video = get_resource_payload(packet, raw_descriptor)
    if not hasattr(video, "get_components"):
        raise MMH3ResourceError(f"Control video {selected_id!r} does not expose get_components()")
    components = video.get_components()
    frames = getattr(components, "images", None)
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[0] < 1:
        raise MMH3ResourceError("Control video frames must be non-empty IMAGE [T,H,W,C]")
    try:
        source_fps = float(getattr(components, "frame_rate", None))
    except (TypeError, ValueError) as exc:
        raise MMH3ResourceError(f"Control video {selected_id!r} has invalid frame rate") from exc
    if source_fps <= 0 or not math.isfinite(source_fps):
        raise MMH3ResourceError(f"Control video {selected_id!r} has invalid frame rate {source_fps!r}")
    output = frames
    if abs(source_fps - target_fps_value) >= 1e-6:
        target_count = max(1, int(round(frames.shape[0] * target_fps_value / source_fps)))
        positions = torch.arange(target_count, device=frames.device, dtype=torch.float64)
        indexes = torch.floor(positions * source_fps / target_fps_value).to(torch.long)
        output = frames.index_select(0, indexes.clamp_(0, frames.shape[0] - 1))
    return MaterializedControlVideo(
        output,
        selected_id,
        str(descriptor.get("role") or "control"),
        source_fps,
        target_fps_value,
        int(frames.shape[0]),
    )


def _validate_spatial_tensor(value: Any, *, label: str, width: int, height: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise MMH3ResourceError(f"{label} must be a tensor")
    if value.ndim == 4:
        actual_height, actual_width = int(value.shape[1]), int(value.shape[2])
    elif value.ndim == 3:
        actual_height, actual_width = int(value.shape[1]), int(value.shape[2])
    else:
        raise MMH3ResourceError(f"{label} must be [T,H,W,C] or [T,H,W], got {tuple(value.shape)}")
    if actual_width != width or actual_height != height:
        raise MMH3ResourceError(
            f"{label} geometry {actual_width}x{actual_height} does not match preflight target {width}x{height}; silent resize is forbidden"
        )


def align_control_timeline(
    value: torch.Tensor,
    *,
    target_frames: int,
    policy: str,
    label: str,
) -> torch.Tensor:
    frames = int(value.shape[0])
    if frames == target_frames:
        return value
    if policy == "strict":
        raise MMH3ResourceError(
            f"strict temporal policy requires {target_frames} {label} frames; got {frames}"
        )
    if policy == "trim":
        if frames < target_frames:
            raise MMH3ResourceError(f"trim cannot extend {label} from {frames} to {target_frames} frames")
        return value[:target_frames]
    if policy == "pad_hold_last":
        if frames > target_frames:
            raise MMH3ResourceError(
                f"pad_hold_last cannot shorten {label} from {frames} to {target_frames} frames"
            )
        if frames == 0:
            raise MMH3ResourceError(f"pad_hold_last cannot extend an empty {label} timeline")
        repeats = (target_frames - frames,) + (1,) * (value.ndim - 1)
        return torch.cat((value, value[-1:].repeat(repeats)), dim=0)
    raise MMH3ResourceError(f"Unsupported control temporal policy {policy!r}")


def prepare_control_inputs(
    plan: H3ControlApplyPlan,
    *,
    control_video: torch.Tensor | None,
    mask: torch.Tensor | None,
    source_video: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if plan.pass_through:
        return None, None, None
    config = plan.config
    if config.control_kind == "inpaint":
        if mask is None or source_video is None:
            raise MMH3ResourceError("Inpaint apply requires both mask and source_video")
    elif control_video is None:
        raise MMH3ResourceError(f"Control kind {config.control_kind!r} requires control_video")
    if mask is not None and source_video is None:
        raise MMH3ResourceError("mask cannot be applied without source_video")
    if source_video is not None and mask is None:
        raise MMH3ResourceError("source_video cannot be applied without mask")

    prepared: list[torch.Tensor | None] = []
    for label, value in (
        ("control_video", control_video),
        ("mask", mask),
        ("source_video", source_video),
    ):
        if value is None:
            prepared.append(None)
            continue
        _validate_spatial_tensor(
            value,
            label=label,
            width=plan.target_width,
            height=plan.target_height,
        )
        prepared.append(
            align_control_timeline(
                value,
                target_frames=plan.target_frames,
                policy=config.temporal_policy,
                label=label,
            )
        )
    return prepared[0], prepared[1], prepared[2]


def validate_masked_edit_inputs(
    source_video: torch.Tensor, mask: torch.Tensor, control_video: torch.Tensor | None = None
) -> dict[str, Any]:
    if not isinstance(source_video, torch.Tensor) or source_video.ndim != 4 or source_video.shape[-1] not in (3, 4):
        raise MMH3ResourceError("Masked edit source_video must be IMAGE frames [T,H,W,3|4]")
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
        raise MMH3ResourceError("Masked edit mask must be [T,H,W]")
    if tuple(source_video.shape[:3]) != tuple(mask.shape):
        raise MMH3ResourceError(
            f"Masked edit source/mask timelines and geometry differ: source={tuple(source_video.shape[:3])}, mask={tuple(mask.shape)}"
        )
    if control_video is not None:
        if not isinstance(control_video, torch.Tensor) or control_video.ndim != 4:
            raise MMH3ResourceError("Masked edit control_video must be IMAGE frames [T,H,W,C]")
        if tuple(control_video.shape[:3]) != tuple(source_video.shape[:3]):
            raise MMH3ResourceError("Masked edit control_video must exactly match source timeline and geometry")
    return {
        "frames": int(source_video.shape[0]),
        "height": int(source_video.shape[1]),
        "width": int(source_video.shape[2]),
        "mask_policy": "white_regenerate",
        "has_control_video": control_video is not None,
    }


def build_h3_control_expansion(
    plan: H3ControlApplyPlan,
    *,
    positive: Any,
    control_net: Any,
    vae: Any,
    control_video: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    source_video: torch.Tensor | None = None,
    graph_builder_factory: Callable[[], Any] | None = None,
) -> H3ControlExpansion:
    if plan.pass_through:
        return H3ControlExpansion(positive, {}, plan)
    if control_net is None or vae is None:
        raise MMH3ResourceError("Active H3 Control Apply requires both control_net and vae")
    control_video, mask, source_video = prepare_control_inputs(
        plan,
        control_video=control_video,
        mask=mask,
        source_video=source_video,
    )
    if graph_builder_factory is None:
        from comfy_execution.graph_utils import GraphBuilder  # type: ignore

        graph_builder_factory = GraphBuilder
    graph = graph_builder_factory()
    inputs: dict[str, Any] = {
        "positive": positive,
        "control_net": control_net,
        "vae": vae,
        "strength": plan.config.strength,
        "start_percent": plan.config.start_percent,
        "end_percent": plan.config.end_percent,
    }
    if control_video is not None:
        inputs["control_video"] = control_video
    if mask is not None:
        inputs["mask"] = mask
        inputs["source_video"] = source_video
    apply = graph.node(plan.adapter.apply_node_id, **inputs)
    return H3ControlExpansion(apply.out(0), graph.finalize(), plan)
