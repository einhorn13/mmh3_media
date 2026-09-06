from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .core import MMH3Media
from .errors import MMH3ResourceError
from .constants import REFERENCE_PRESETS, REFERENCE_PURPOSES
from .reference_cost import ReferenceCostReport, estimate_reference_cost
from .h3_contract import h3_latent_contract_from_resource
from .h3_resource_semantics import context_usage, reference_enabled, reference_purposes, reference_video_binding
from .resource_model import resource_facts


INTENTS = ("condition/generate",)
MODES = ("auto", "t2va", "i2va", "fl2va", "l2va", "ref2va")
POLICIES = ("auto", "prefer_references", "prefer_keyframes", "strict")

_MODE_ALIASES = {
    "t2v": "t2va",
    "t2va": "t2va",
    "i2v": "i2va",
    "i2va": "i2va",
    "fl2v": "fl2va",
    "fl2va": "fl2va",
    "l2v": "l2va",
    "l2va": "l2va",
    "r2v": "ref2va",
    "r2va": "ref2va",
    "ref2v": "ref2va",
    "ref2va": "ref2va",
}


@dataclass(frozen=True)
class ResolutionDiagnostic:
    severity: str
    code: str
    message: str
    resource_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "resource_ids": list(self.resource_ids),
        }


@dataclass(frozen=True)
class ResourceDescriptor:
    resource_id: str
    kind: str
    role: str
    order: int | None
    path: str
    facts_json: str
    usage: str | None = None

    @staticmethod
    def from_canonical(value: Mapping[str, Any]) -> "ResourceDescriptor":
        role = str(value.get("role") or "auxiliary")
        kind = str(value.get("kind") or "")
        facts = resource_facts(value)
        usage = None
        if role == "reference":
            facts["reference_control"] = {
                "enabled": reference_enabled(value),
                "purposes": list(reference_purposes(value)),
            }
            binding = reference_video_binding(value)
            if binding:
                facts["reference_binding"] = {"video_resource_id": binding}
            usage = "reference"
        elif role == "context":
            usage = context_usage(value)
        if kind == "latent":
            try:
                facts["h3_latent"] = h3_latent_contract_from_resource(value)
            except MMH3ResourceError:
                pass
        order = value.get("order")
        return ResourceDescriptor(
            resource_id=str(value["id"]),
            kind=kind,
            role=role,
            order=int(order) if isinstance(order, int) else None,
            path=str(value.get("path") or ""),
            facts_json=json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            usage=usage,
        )

    @property
    def facts(self) -> dict[str, Any]:
        return json.loads(self.facts_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.resource_id,
            "kind": self.kind,
            "role": self.role,
            "order": self.order,
            "path": self.path,
            "facts": self.facts,
            **({"usage": self.usage} if self.usage is not None else {}),
        }


@dataclass(frozen=True)
class ResolvedValue:
    name: str
    value: Any
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "source": self.source}


@dataclass(frozen=True)
class ReferenceItem:
    resource: ResourceDescriptor
    prompt_tag: str
    paired_video_resource_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"resource": self.resource.to_dict(), "prompt_tag": self.prompt_tag}
        if self.paired_video_resource_id:
            out["paired_video_resource_id"] = self.paired_video_resource_id
        return out


@dataclass(frozen=True)
class ResolvedReferenceSet:
    preset: str
    resources: tuple[ResourceDescriptor, ...]
    presentation: tuple[ReferenceItem, ...]
    disabled_resources: tuple[ResourceDescriptor, ...]
    filtered_resources: tuple[ResourceDescriptor, ...]
    diagnostics: tuple[ResolutionDiagnostic, ...]
    cost: ReferenceCostReport | None = None

    @property
    def ready(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "ready": self.ready,
            "resources": [item.to_dict() for item in self.resources],
            "presentation": [item.to_dict() for item in self.presentation],
            "disabled_resources": [item.to_dict() for item in self.disabled_resources],
            "filtered_resources": [item.to_dict() for item in self.filtered_resources],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "cost": None if self.cost is None else self.cost.to_dict(),
        }


@dataclass(frozen=True)
class UnusedResource:
    resource: ResourceDescriptor
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"resource": self.resource.to_dict(), "reason": self.reason}


@dataclass(frozen=True)
class ResolvedMMH3Inputs:
    intent: str
    mode: str
    policy: str
    ready: bool
    values: tuple[ResolvedValue, ...]
    selected_resources: tuple[ResourceDescriptor, ...]
    reference_order: tuple[ReferenceItem, ...]
    unused_resources: tuple[UnusedResource, ...]
    diagnostics: tuple[ResolutionDiagnostic, ...]
    reference_preset: str = "all"
    ref_image_size: str = "match"
    reference_cost: ReferenceCostReport | None = None

    def value(self, name: str, default: Any = None) -> Any:
        return next((item.value for item in self.values if item.name == name), default)

    def source(self, name: str, default: str = "missing") -> str:
        return next((item.source for item in self.values if item.name == name), default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "mode": self.mode,
            "policy": self.policy,
            "reference_preset": self.reference_preset,
            "ref_image_size": self.ref_image_size,
            "ready": self.ready,
            "values": {item.name: item.to_dict() for item in self.values},
            "selected_resources": [item.to_dict() for item in self.selected_resources],
            "reference_order": [item.to_dict() for item in self.reference_order],
            "unused_resources": [item.to_dict() for item in self.unused_resources],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "reference_cost": None if self.reference_cost is None else self.reference_cost.to_dict(),
        }

    def summary(self) -> str:
        state = "READY" if self.ready else "BLOCKED"
        selected = ", ".join(f"{r.role}/{r.kind} order={r.order}" for r in self.selected_resources) or "none"
        lines = [
            f"{state} · {self.intent} → {self.mode} · policy={self.policy} · refs={self.reference_preset}",
            " · ".join(
                f"{name}={self.value(name)!r} ({self.source(name)})"
                for name in ("prompt", "seed", "width", "height", "frames", "fps")
            ),
            f"Selected resources: {selected}",
        ]
        for item in self.diagnostics:
            lines.append(f"{item.severity.upper()} [{item.code}] {item.message}")
        if self.reference_cost is not None:
            lines.append(self.reference_cost.summary())
        return "\n".join(lines)


def _normalize_mode(value: Any, *, allow_auto: bool) -> str | None:
    text = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if allow_auto and text in ("", "auto"):
        return "auto"
    return _MODE_ALIASES.get(text)


def _descriptor_h3_value(resources: tuple[ResourceDescriptor, ...], key: str) -> Any:
    latent = next((item for item in resources if item.kind == "latent" and isinstance(item.facts.get("h3_latent"), dict)), None)
    if latent is None:
        return None
    contract = latent.facts["h3_latent"]
    if key in ("width", "height"):
        return contract.get("canvas", {}).get(key)
    if key in ("frames", "fps"):
        return contract.get("timeline", {}).get(key)
    return None


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _resolve_values(
    generation: Mapping[str, Any],
    resources: tuple[ResourceDescriptor, ...],
    overrides: Mapping[str, Any],
    diagnostics: list[ResolutionDiagnostic],
) -> tuple[ResolvedValue, ...]:
    out: list[ResolvedValue] = []

    prompt_override = overrides.get("prompt")
    if isinstance(prompt_override, str) and prompt_override != "":
        out.append(ResolvedValue("prompt", prompt_override, "override"))
    elif isinstance(generation.get("prompt"), str):
        out.append(ResolvedValue("prompt", generation.get("prompt"), "packet_metadata"))
    else:
        out.append(ResolvedValue("prompt", "", "default"))

    seed_override = _nonnegative_int(overrides.get("seed"))
    seed_packet = _nonnegative_int(generation.get("seed"))
    if seed_override is not None:
        out.append(ResolvedValue("seed", seed_override, "override"))
    elif seed_packet is not None:
        out.append(ResolvedValue("seed", seed_packet, "packet_metadata"))
    else:
        out.append(ResolvedValue("seed", 0, "default"))

    for key in ("width", "height", "frames"):
        override = _positive_int(overrides.get(key))
        packet_value = _positive_int(generation.get(key))
        latent_value = _positive_int(_descriptor_h3_value(resources, key))
        if override is not None:
            out.append(ResolvedValue(key, override, "override"))
        elif packet_value is not None:
            out.append(ResolvedValue(key, packet_value, "packet_metadata"))
        elif latent_value is not None:
            out.append(ResolvedValue(key, latent_value, "packet_resource"))
        else:
            out.append(ResolvedValue(key, None, "missing"))
            diagnostics.append(ResolutionDiagnostic("error", f"missing_{key}", f"No usable {key} override or packet value."))

    fps_override = _positive_float(overrides.get("fps"))
    fps_packet = _positive_float(generation.get("fps"))
    fps_latent = _positive_float(_descriptor_h3_value(resources, "fps"))
    if fps_override is not None:
        out.append(ResolvedValue("fps", fps_override, "override"))
    elif fps_packet is not None:
        out.append(ResolvedValue("fps", fps_packet, "packet_metadata"))
    elif fps_latent is not None:
        out.append(ResolvedValue("fps", fps_latent, "packet_resource"))
    else:
        out.append(ResolvedValue("fps", 24.0, "default"))

    width = next(item.value for item in out if item.name == "width")
    height = next(item.value for item in out if item.name == "height")
    frames = next(item.value for item in out if item.name == "frames")
    if width is not None and width % 32:
        diagnostics.append(ResolutionDiagnostic("warning", "width_not_multiple_32", f"Resolved width {width} is not divisible by 32."))
    if height is not None and height % 32:
        diagnostics.append(ResolutionDiagnostic("warning", "height_not_multiple_32", f"Resolved height {height} is not divisible by 32."))
    if frames is not None and (frames - 5) % 17:
        diagnostics.append(ResolutionDiagnostic("warning", "frames_off_h3_grid", f"Resolved frame count {frames} is not on the H3 17n+5 grid."))
    return tuple(out)


def _requirements_met(mode: str, has_first: bool, has_last: bool, has_refs: bool) -> bool:
    return {
        "t2va": True,
        "i2va": has_first,
        "l2va": has_last,
        "fl2va": has_first and has_last,
        "ref2va": has_refs,
    }[mode]


def _keyframe_mode(has_first: bool, has_last: bool) -> str:
    if has_first and has_last:
        return "fl2va"
    if has_last:
        return "l2va"
    if has_first:
        return "i2va"
    return "t2va"


def _reference_order(
    resources: tuple[ResourceDescriptor, ...], diagnostics: list[ResolutionDiagnostic]
) -> tuple[ReferenceItem, ...]:
    pictures = tuple(r for r in resources if r.kind == "image")
    videos = tuple(r for r in resources if r.kind == "video")
    audios = tuple(r for r in resources if r.kind == "audio")
    video_ids = {item.resource_id for item in videos}
    paired: dict[str, list[ResourceDescriptor]] = {}
    standalone: list[ResourceDescriptor] = []

    for audio in audios:
        binding = audio.facts.get("reference_binding")
        video_id = binding.get("video_resource_id") if isinstance(binding, dict) else None
        if isinstance(video_id, str) and video_id:
            if video_id in video_ids:
                paired.setdefault(video_id, []).append(audio)
            else:
                standalone.append(audio)
                diagnostics.append(
                    ResolutionDiagnostic(
                        "warning",
                        "orphan_reference_audio_binding",
                        f"audio reference order={audio.order} points to missing video resource {video_id!r}; treating it as standalone.",
                        (audio.resource_id,),
                    )
                )
        else:
            standalone.append(audio)

    result: list[ReferenceItem] = []
    audio_index = 0
    for index, picture in enumerate(pictures, start=1):
        result.append(ReferenceItem(picture, f"<Picture {index}>"))
    for index, video in enumerate(videos, start=1):
        soundtracks = paired.get(video.resource_id, [])
        if len(soundtracks) > 1:
            diagnostics.append(
                ResolutionDiagnostic(
                    "error",
                    "multiple_video_soundtracks",
                    f"video reference order={video.order} has {len(soundtracks)} bound audio resources; native H3 accepts one soundtrack per reference video.",
                    tuple(item.resource_id for item in soundtracks),
                )
            )
        for audio in soundtracks:
            audio_index += 1
            result.append(ReferenceItem(audio, f"<Audio {audio_index}>", video.resource_id))
        result.append(ReferenceItem(video, f"<Video {index}>"))
    for audio in standalone:
        audio_index += 1
        result.append(ReferenceItem(audio, f"<Audio {audio_index}>"))
    return tuple(result)


def _reference_control(resource: ResourceDescriptor) -> tuple[bool, tuple[str, ...]]:
    control = resource.facts.get("reference_control")
    if not isinstance(control, dict):
        return True, ("unknown",)
    enabled = control.get("enabled", True)
    if not isinstance(enabled, bool):
        raise MMH3ResourceError(
            f"Resource {resource.resource_id!r} reference_control.enabled must be boolean"
        )
    raw_purposes = control.get("purposes", ["unknown"])
    if not isinstance(raw_purposes, list) or not raw_purposes:
        raise MMH3ResourceError(
            f"Resource {resource.resource_id!r} reference_control.purposes must be a non-empty array"
        )
    purposes: list[str] = []
    for purpose in raw_purposes:
        if purpose not in REFERENCE_PURPOSES:
            raise MMH3ResourceError(
                f"Resource {resource.resource_id!r} reference purpose {purpose!r} is unsupported"
            )
        if purpose not in purposes:
            purposes.append(purpose)
    return enabled, tuple(purposes)


def _preset_matches(preset: str, purposes: tuple[str, ...]) -> bool:
    if preset in ("all", "balanced"):
        return True
    wanted = {
        "identity": {"identity"},
        "motion": {"motion"},
        "style": {"style", "detail"},
        "voice": {"voice"},
    }[preset]
    return bool(wanted.intersection(purposes))


def resolve_reference_set(
    packet: MMH3Media,
    *,
    preset: str = "all",
    target_width: int | None = None,
    target_height: int | None = None,
    target_frames: int | None = None,
    ref_image_size: str = "match",
) -> ResolvedReferenceSet:
    """Resolve reference descriptors, bindings and H3 presentation without materializing payloads."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    if preset not in REFERENCE_PRESETS:
        raise MMH3ResourceError(f"Unsupported reference preset {preset!r}; expected one of {REFERENCE_PRESETS}")
    kind_rank = {"image": 0, "video": 1, "audio": 2}
    all_refs = tuple(
        sorted(
            (ResourceDescriptor.from_canonical(ref.descriptor) for ref in packet.filter_resource_refs(role="reference")),
            key=lambda value: (kind_rank.get(value.kind, 99), value.order is None, value.order or 0, value.resource_id),
        )
    )
    active: list[ResourceDescriptor] = []
    disabled: list[ResourceDescriptor] = []
    filtered: list[ResourceDescriptor] = []
    diagnostics: list[ResolutionDiagnostic] = []
    for resource in all_refs:
        enabled, purposes = _reference_control(resource)
        if not enabled:
            disabled.append(resource)
        elif not _preset_matches(preset, purposes):
            filtered.append(resource)
        else:
            active.append(resource)

    presentation = _reference_order(tuple(active), diagnostics)
    pictures = [item for item in active if item.kind == "image"]
    videos = [item for item in active if item.kind == "video"]
    standalone_audio = [
        item for item in presentation if item.resource.kind == "audio" and item.paired_video_resource_id is None
    ]
    for label, limited_resources, maximum in (
        ("image reference", pictures, 9),
        ("video reference", videos, 3),
        ("standalone audio reference", [item.resource for item in standalone_audio], 3),
    ):
        count = len(limited_resources)
        if count > maximum:
            diagnostics.append(
                ResolutionDiagnostic(
                    "error",
                    "reference_limit_exceeded",
                    f"Selected {count} {label} resources, but native H3 accepts at most {maximum}.",
                    tuple(item.resource_id for item in limited_resources),
                )
            )
    if filtered:
        diagnostics.append(
            ResolutionDiagnostic(
                "info",
                "references_filtered_by_preset",
                f"Preset {preset!r} filtered {len(filtered)} enabled reference resource(s).",
                tuple(item.resource_id for item in filtered),
            )
        )
    if disabled:
        diagnostics.append(
            ResolutionDiagnostic(
                "info",
                "references_explicitly_disabled",
                f"{len(disabled)} reference resource(s) are explicitly disabled.",
                tuple(item.resource_id for item in disabled),
            )
        )
    cost = None
    if target_width and target_height and target_frames:
        cost = estimate_reference_cost(
            active,
            target_width=int(target_width),
            target_height=int(target_height),
            target_frames=int(target_frames),
            ref_image_size=ref_image_size,
        )
    return ResolvedReferenceSet(
        preset,
        tuple(active),
        presentation,
        tuple(disabled),
        tuple(filtered),
        tuple(diagnostics),
        cost,
    )


def resolve_packet(
    packet: MMH3Media,
    *,
    intent: str = "condition/generate",
    mode: str = "auto",
    policy: str = "auto",
    reference_preset: str = "all",
    ref_image_size: str = "match",
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedMMH3Inputs:
    """Resolve a packet without reading or decoding any resource payload."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    if intent not in INTENTS:
        raise MMH3ResourceError(f"Unsupported resolver intent {intent!r}; expected one of {INTENTS}")
    if policy not in POLICIES:
        raise MMH3ResourceError(f"Unsupported resolver policy {policy!r}; expected one of {POLICIES}")
    if reference_preset not in REFERENCE_PRESETS:
        raise MMH3ResourceError(
            f"Unsupported reference preset {reference_preset!r}; expected one of {REFERENCE_PRESETS}"
        )
    if ref_image_size not in ("match", "max"):
        raise MMH3ResourceError("ref_image_size must be match or max")
    requested_mode = _normalize_mode(mode, allow_auto=True)
    if requested_mode is None or requested_mode not in MODES:
        raise MMH3ResourceError(f"Unsupported H3 mode {mode!r}; expected one of {MODES}")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise MMH3ResourceError("Resolver overrides must be a mapping")

    reference_set = resolve_reference_set(packet, preset=reference_preset)
    diagnostics: list[ResolutionDiagnostic] = list(reference_set.diagnostics)
    resources = tuple(
        ResourceDescriptor.from_canonical(item)
        for item in sorted(
            packet.resources(),
            key=lambda r: (str(r.get("role")), r.get("order") is None, int(r.get("order") or 0), str(r.get("id"))),
        )
    )
    contexts = tuple(
        ResourceDescriptor.from_canonical(ref.descriptor)
        for ref in packet.filter_resource_refs(role="context")
        if context_usage(ref.descriptor) in ("first_frame", "last_frame")
    )
    first = next((r for r in contexts if r.usage == "first_frame"), None)
    last = next((r for r in contexts if r.usage == "last_frame"), None)
    refs = reference_set.resources
    has_first, has_last, has_refs = first is not None, last is not None, bool(refs)
    stored_task_raw = packet.manifest.get("generation", {}).get("task")
    stored_task = _normalize_mode(stored_task_raw, allow_auto=False)

    if requested_mode != "auto":
        resolved_mode = requested_mode
    elif has_refs and (has_first or has_last):
        if policy == "strict":
            resolved_mode = stored_task if stored_task and _requirements_met(stored_task, has_first, has_last, has_refs) else "ref2va"
            diagnostics.append(
                ResolutionDiagnostic(
                    "error",
                    "ambiguous_keyframes_and_references",
                    "Packet contains both keyframes and references; choose a mode or a non-strict preference policy.",
                    tuple(r.resource_id for r in refs + tuple(x for x in (first, last) if x is not None)),
                )
            )
        elif policy == "prefer_keyframes":
            resolved_mode = _keyframe_mode(has_first, has_last)
            diagnostics.append(ResolutionDiagnostic("warning", "references_not_selected", "Keyframes won over references by policy."))
        else:
            resolved_mode = "ref2va"
            diagnostics.append(ResolutionDiagnostic("warning", "keyframes_not_selected", "References won over keyframes by policy."))
    elif stored_task and _requirements_met(stored_task, has_first, has_last, has_refs):
        resolved_mode = stored_task
    else:
        if stored_task_raw and stored_task is None:
            diagnostics.append(ResolutionDiagnostic("warning", "unsupported_packet_task", f"generation.task={stored_task_raw!r} is not a base H3 generation mode."))
        elif stored_task and not _requirements_met(stored_task, has_first, has_last, has_refs):
            diagnostics.append(ResolutionDiagnostic("warning", "stale_packet_task", f"generation.task={stored_task!r} lacks its required packet resources; auto-detected mode instead."))
        resolved_mode = "ref2va" if has_refs else _keyframe_mode(has_first, has_last)

    missing_by_mode = {
        "i2va": (() if has_first else ("first_frame",)),
        "l2va": (() if has_last else ("last_frame",)),
        "fl2va": tuple(name for name, present in (("first_frame", has_first), ("last_frame", has_last)) if not present),
        "ref2va": (() if has_refs else ("reference media",)),
        "t2va": (),
    }[resolved_mode]
    if missing_by_mode:
        diagnostics.append(ResolutionDiagnostic("error", "missing_mode_resources", f"Mode {resolved_mode} requires: {', '.join(missing_by_mode)}."))

    selected: list[ResourceDescriptor] = []
    if resolved_mode == "i2va" and first:
        selected.append(first)
    elif resolved_mode == "l2va" and last:
        selected.append(last)
    elif resolved_mode == "fl2va":
        selected.extend(item for item in (first, last) if item is not None)
    elif resolved_mode == "ref2va":
        selected.extend(refs)

    selected_ids = {item.resource_id for item in selected}
    extra_media = tuple(r for r in refs + tuple(x for x in (first, last) if x is not None) if r.resource_id not in selected_ids)
    if extra_media and not any(d.code == "ambiguous_keyframes_and_references" for d in diagnostics):
        severity = "error" if policy == "strict" else "warning"
        diagnostics.append(
            ResolutionDiagnostic(
                severity,
                "unused_compatible_media",
                f"Mode {resolved_mode} does not use: " + ", ".join(f"{r.role}/{r.kind} order={r.order}" for r in extra_media) + ".",
                tuple(r.resource_id for r in extra_media),
            )
        )

    reference_order = reference_set.presentation
    disabled_ids = {item.resource_id for item in reference_set.disabled_resources}
    filtered_ids = {item.resource_id for item in reference_set.filtered_resources}
    unused = tuple(
        UnusedResource(
            resource,
            (
                "reference explicitly disabled"
                if resource.resource_id in disabled_ids
                else "reference filtered by preset"
                if resource.resource_id in filtered_ids
                else "compatible media not selected by resolved mode"
                if resource.resource_id in {item.resource_id for item in refs + tuple(x for x in (first, last) if x is not None)}
                else f"role is not consumed by intent {intent}"
            ),
        )
        for resource in resources
        if resource.resource_id not in selected_ids
    )
    values = _resolve_values(packet.manifest.get("generation", {}), resources, overrides or {}, diagnostics)
    reference_cost = None
    cost_width = _positive_int(next((item.value for item in values if item.name == "width"), None))
    cost_height = _positive_int(next((item.value for item in values if item.name == "height"), None))
    cost_frames = _positive_int(next((item.value for item in values if item.name == "frames"), None))
    if resolved_mode == "ref2va" and refs and cost_width and cost_height and cost_frames:
        reference_cost = estimate_reference_cost(
            refs,
            target_width=cost_width,
            target_height=cost_height,
            target_frames=cost_frames,
            ref_image_size=ref_image_size,
        )
        diagnostics.extend(
            ResolutionDiagnostic(
                str(item.get("severity", "info")),
                str(item.get("code", "reference_cost")),
                str(item.get("message", "")),
                tuple(str(resource_id) for resource_id in item.get("resource_ids", [])),
            )
            for item in reference_cost.diagnostics
        )
    if resolved_mode == "ref2va" and reference_order:
        prompt = str(next((item.value for item in values if item.name == "prompt"), "") or "")
        tags = tuple(item.prompt_tag for item in reference_order)
        if not any(tag in prompt for tag in tags):
            diagnostics.append(
                ResolutionDiagnostic(
                    "warning",
                    "reference_tags_not_in_prompt",
                    "Prompt does not contain any resolved H3 reference tag. Available tags: " + ", ".join(tags),
                    tuple(item.resource.resource_id for item in reference_order),
                )
            )
    ready = not any(item.severity == "error" for item in diagnostics)
    return ResolvedMMH3Inputs(
        intent=intent,
        mode=resolved_mode,
        policy=policy,
        reference_preset=reference_preset,
        ref_image_size=ref_image_size,
        ready=ready,
        values=values,
        selected_resources=tuple(selected),
        reference_order=reference_order,
        unused_resources=unused,
        diagnostics=tuple(diagnostics),
        reference_cost=reference_cost,
    )
