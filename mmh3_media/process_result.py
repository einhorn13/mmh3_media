from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .archive import get_resource_payload
from .core import MMH3Media
from .errors import MMH3ResourceError
from .h3 import validate_h3_av_latent, validate_h3_latent_origin
from .media_metadata import describe_media_payload
from .resource_model import descriptor_from_media_metadata
from .h3_contract import build_h3_latent_contract
from .h3_resource_semantics import find_context_resource, make_context_contract
from .lora_provenance import normalize_generation_loras
from .control_contract import H3_CONTROLNET_ALGORITHMS, get_control_configuration
from .control_provider import normalize_control_apply_process_info
from .util import deep_copy_json, is_jsonable


PROCESS_RESULT_VERSION = 1


@dataclass(frozen=True)
class PackedH3Result:
    packet: MMH3Media
    resource_ids: Mapping[str, str]
    operation: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROCESS_RESULT_VERSION,
            "operation": self.operation,
            "mode": self.mode,
            "resource_ids": dict(self.resource_ids),
        }

    def summary(self) -> str:
        resources = ", ".join(f"{role}={resource_id}" for role, resource_id in self.resource_ids.items())
        return f"PACKED · {self.operation} · mode={self.mode or 'unspecified'} · {resources}"


@dataclass(frozen=True)
class PrimaryPacketView:
    latent: Any
    video: Any
    audio: Any
    first_frame: Any
    last_frame: Any
    resource_ids: Mapping[str, str]
    generation: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_ids": dict(self.resource_ids),
            "has": {key: key in self.resource_ids for key in ("latent", "video", "audio", "first_frame", "last_frame")},
            "generation": deep_copy_json(dict(self.generation)),
        }


def _selected_input_resource_ids(process_info: Mapping[str, Any]) -> list[str]:
    selected = process_info.get("selected_resources")
    if not isinstance(selected, list):
        return []
    output: list[str] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        resource_id = item.get("id", item.get("resource_id"))
        if isinstance(resource_id, str):
            output.append(resource_id)
    return output


def _resolved_generation_provenance(process_info: Mapping[str, Any]) -> dict[str, Any]:
    values = process_info.get("values")
    if not isinstance(values, Mapping):
        return {}
    generation: dict[str, Any] = {}
    prompt = values.get("prompt")
    if isinstance(prompt, Mapping) and isinstance(prompt.get("value"), str):
        generation["prompt"] = prompt["value"]
    seed = values.get("seed")
    if isinstance(seed, Mapping):
        value = seed.get("value")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            generation["seed"] = value
    return generation


def pack_h3_result(
    packet: MMH3Media,
    *,
    latent: Any = None,
    video: Any = None,
    audio: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
    operation: str = "generate",
    mode: str = "",
    status: str = "",
    process_info: Mapping[str, Any] | None = None,
    latent_origin: str = "sampler_output",
    applied_loras: Any = None,
    control_process_info: Mapping[str, Any] | None = None,
) -> PackedH3Result:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    operation = str(operation or "").strip()
    if not operation:
        raise MMH3ResourceError("Process operation must be a non-empty string")
    process_info = deep_copy_json(dict(process_info or {}))
    if not is_jsonable(process_info):
        raise MMH3ResourceError("process_info must be JSON serializable")
    if "control" in process_info:
        raise MMH3ResourceError(
            "process_info.control is reserved; pass the dedicated control_process_info argument"
        )
    configured_control = get_control_configuration(packet)
    requires_control_proof = bool(
        configured_control is not None
        and configured_control.effective_algorithm in H3_CONTROLNET_ALGORITHMS
    )
    normalized_control_info = None
    bound_control = None
    if control_process_info is not None:
        normalized_control_info, bound_control = normalize_control_apply_process_info(
            packet, control_process_info
        )
        process_info["control"] = normalized_control_info
    elif requires_control_proof:
        raise MMH3ResourceError(
            "A capability-bound ControlNet packet requires control_process_info from MMH3 H3 Control Apply"
        )
    outputs = {"latent": latent, "video": video, "audio": audio, "first_frame": first_frame, "last_frame": last_frame}
    if all(payload is None for payload in outputs.values()):
        raise MMH3ResourceError("Pack H3 Result requires at least one connected result resource")

    out = packet
    resource_ids: dict[str, str] = {}
    if latent is not None:
        origin = validate_h3_latent_origin(latent_origin)
        latent_info = validate_h3_av_latent(latent)
        extensions = {"minimax_h3": {"latent": build_h3_latent_contract(latent_info, origin=origin)}}
        existing = out.get_primary("latent")
        out, resource_ids["latent"] = out.put_with_id(
            latent, kind="latent", role="auxiliary", mode="replace" if existing else "add",
            resource_id=existing["id"] if existing else "", extensions=extensions, record_history=False,
        )
        out = out.set_primary("latent", resource_ids["latent"])
        generation_patch = {
            "width": latent_info.width, "height": latent_info.height, "fps": latent_info.fps,
            "audio_sample_rate": latent_info.audio_sample_rate, "audio_latent_rate": latent_info.audio_latent_rate,
        }
        if latent_info.frames is not None:
            generation_patch["frames"] = latent_info.frames
        out = out.edit_metadata(merge_patch_json={"generation": generation_patch})

    for key, payload, kind in (("video", video, "video"), ("audio", audio, "audio")):
        if payload is None:
            continue
        existing = out.get_primary(kind)
        facts = describe_media_payload(payload, kind)
        out, resource_ids[key] = out.put_with_id(
            payload, kind=kind, role="auxiliary", mode="replace" if existing else "add",
            resource_id=existing["id"] if existing else "", descriptor=descriptor_from_media_metadata(kind, facts),
            tags=["output"], record_history=False,
        )
        out = out.set_primary(kind, resource_ids[key])

    if video is not None:
        from .decoded_upscale import discard_stale_delivery
        out = discard_stale_delivery(out)

    for key, payload, usage, order in (("first_frame", first_frame, "first_frame", 0), ("last_frame", last_frame, "last_frame", 1)):
        if payload is None:
            continue
        existing = find_context_resource(out, usage)
        facts = describe_media_payload(payload, "image")
        extensions = {"minimax_h3": {"context": make_context_contract(usage=usage)}}
        out, resource_ids[key] = out.put_with_id(
            payload, kind="image", role="context", order=order, mode="replace" if existing else "add",
            resource_id=existing["id"] if existing else "", descriptor=descriptor_from_media_metadata("image", facts),
            extensions=extensions, tags=[usage.replace("_", "-")], record_history=False,
        )

    mode = str(mode or "").strip()
    generation_provenance = _resolved_generation_provenance(process_info)
    if mode:
        generation_provenance["task"] = mode
    if generation_provenance:
        out = out.edit_metadata(merge_patch_json={"generation": generation_provenance})
    if applied_loras is not None:
        normalized_applied_loras = normalize_generation_loras(applied_loras)
        out = out.edit_metadata(merge_patch_json={"generation": {"loras": normalized_applied_loras}})
        process_info["applied_loras"] = deep_copy_json(normalized_applied_loras)
    if bound_control is not None:
        out = out.set_extension_value("mmh3_media", "control", bound_control.to_dict())
    process_record = {
        "version": PROCESS_RESULT_VERSION,
        "operation": operation,
        "mode": mode,
        "status": str(status or ""),
        "input_resource_ids": _selected_input_resource_ids(process_info),
        "output_resource_ids": dict(resource_ids),
        "info": process_info,
    }
    out = out.set_extension_value("mmh3_media", "last_process", process_record)
    out = out.record_operation(
        "process_result",
        operation=operation,
        mode=mode,
        input_resource_ids=process_record["input_resource_ids"],
        output_resource_ids=resource_ids,
    )
    from .save_previews import cache_materialized_previews

    out = cache_materialized_previews(out, result_ids=resource_ids)
    return PackedH3Result(out, resource_ids, operation, mode)


def unpack_primary(packet: MMH3Media) -> PrimaryPacketView:
    """Materialize explicit primary AV resources and H3 first/last context resources."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    payloads: dict[str, Any] = {"latent": None, "video": None, "audio": None, "first_frame": None, "last_frame": None}
    resource_ids: dict[str, str] = {}
    for kind in ("latent", "video", "audio"):
        descriptor = packet.get_primary(kind)
        if descriptor is not None:
            payloads[kind] = get_resource_payload(packet, descriptor)
            resource_ids[kind] = descriptor["id"]
    for key in ("first_frame", "last_frame"):
        descriptor = find_context_resource(packet, key)
        if descriptor is not None:
            payloads[key] = get_resource_payload(packet, descriptor)
            resource_ids[key] = descriptor["id"]
    return PrimaryPacketView(
        latent=payloads["latent"], video=payloads["video"], audio=payloads["audio"],
        first_frame=payloads["first_frame"], last_frame=payloads["last_frame"],
        resource_ids=resource_ids, generation=deep_copy_json(packet.manifest.get("generation", {})),
    )
