from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .constants import AUDIO_LATENT_FPS, AUDIO_SAMPLE_RATE, FPS
from .core import MMH3Media
from .inspection import inspect_packet
from .resource_model import resource_facts
from .h3_contract import validate_h3_continuation_compatibility, validate_h3_refine_compatibility
from .lora_provenance import get_generation_loras
from .runtime_contract import ADD_GUIDE_NODE_ID, TILE_GUIDER_NODE_ID, parse_add_guide_contract, parse_tile_guider_contract
from .generation_contract import validate_generation_model_family
from .resolution import MODES, resolve_packet


PREFLIGHT_OPERATIONS = (
    "inspect",
    "generate",
    "continuation",
    "high_sigma_refine",
    "decoded_high_sigma_refine",
    "stitch",
    "masked_edit",
)


@dataclass(frozen=True)
class PreflightDiagnostic:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class PreflightReport:
    operation: str
    diagnostics: tuple[PreflightDiagnostic, ...]
    facts: dict[str, Any]

    @property
    def ready(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def summary(self) -> str:
        head = "READY" if self.ready else "BLOCKED"
        counts = {level: sum(item.severity == level for item in self.diagnostics) for level in ("error", "warning", "info")}
        lines = [f"{head} · F15 {self.operation} · errors={counts['error']} warnings={counts['warning']}"]
        lines.extend(f"{item.severity.upper()} [{item.code}] {item.message}" for item in self.diagnostics)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "operation": self.operation,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "facts": self.facts,
        }


def _add(items: list[PreflightDiagnostic], severity: str, code: str, message: str) -> None:
    items.append(PreflightDiagnostic(severity, code, message))


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _decoded_media_facts(
    packet: MMH3Media,
    diagnostics: list[PreflightDiagnostic],
) -> dict[str, Any]:
    """Validate primary decoded AV descriptors without touching loaded payloads."""
    facts: dict[str, Any] = {"video": None, "audio": None, "duration_tolerance_seconds": 1.0 / FPS}
    video = packet.get_primary("video")
    audio = packet.get_primary("audio")
    video_duration = audio_duration = None

    if video is not None:
        metadata = resource_facts(video)
        frame_count = _positive_int(metadata.get("frames"))
        video_duration = _positive_float(metadata.get("duration"))
        video_fps = _positive_float(metadata.get("fps"))
        fps_source = "descriptor"
        if video_fps is None and frame_count is not None and video_duration is not None:
            video_fps = frame_count / video_duration
            fps_source = "derived_frame_count_over_duration"
        if video_duration is None and frame_count is not None and video_fps is not None:
            video_duration = frame_count / video_fps
        if video_fps is None:
            _add(
                diagnostics,
                "warning",
                "decoded_video_fps_unknown",
                "Primary decoded video descriptor cannot prove its FPS; payload was kept lazy.",
            )
        elif abs(video_fps - FPS) > 1e-6:
            _add(
                diagnostics,
                "error",
                "decoded_video_fps_not_24",
                f"Primary decoded video resolves to {video_fps:g} FPS; H3 requires {FPS} FPS.",
            )
        if video_duration is None:
            _add(
                diagnostics,
                "warning",
                "decoded_video_duration_unknown",
                "Primary decoded video descriptor has no usable duration; payload was kept lazy.",
            )
        facts["video"] = {
            "resource_id": video.get("id"),
            "fps": video_fps,
            "fps_source": fps_source if video_fps is not None else None,
            "frame_count": frame_count,
            "duration": video_duration,
        }

    if audio is not None:
        metadata = resource_facts(audio)
        sample_rate = _positive_int(metadata.get("sample_rate"))
        channels = _positive_int(metadata.get("channels"))
        samples = _positive_int(metadata.get("samples"))
        audio_duration = _positive_float(metadata.get("duration"))
        if audio_duration is None and samples is not None and sample_rate is not None:
            audio_duration = samples / sample_rate
        if sample_rate is None:
            _add(
                diagnostics,
                "warning",
                "decoded_audio_sample_rate_unknown",
                "Primary decoded audio descriptor cannot prove its sample rate; payload was kept lazy.",
            )
        elif sample_rate != AUDIO_SAMPLE_RATE:
            _add(
                diagnostics,
                "error",
                "decoded_audio_sample_rate_not_32000",
                f"Primary decoded audio uses {sample_rate} Hz; H3 requires {AUDIO_SAMPLE_RATE} Hz.",
            )
        if channels is None:
            _add(
                diagnostics,
                "warning",
                "decoded_audio_channels_unknown",
                "Primary decoded audio metadata cannot prove stereo channel layout; payload was kept lazy.",
            )
        elif channels != 2:
            _add(
                diagnostics,
                "error",
                "decoded_audio_not_stereo",
                f"Primary decoded audio has {channels} channel(s); H3 requires stereo audio.",
            )
        if audio_duration is None:
            _add(
                diagnostics,
                "warning",
                "decoded_audio_duration_unknown",
                "Primary decoded audio metadata has no usable duration; payload was kept lazy.",
            )
        facts["audio"] = {
            "resource_id": audio.get("id"),
            "sample_rate": sample_rate,
            "channels": channels,
            "samples": samples,
            "duration": audio_duration,
        }

    if video_duration is not None and audio_duration is not None:
        delta = abs(video_duration - audio_duration)
        facts["duration_delta_seconds"] = delta
        if delta > (1.0 / FPS) + 1e-9:
            _add(
                diagnostics,
                "error",
                "decoded_av_duration_mismatch",
                f"Primary decoded video/audio durations differ by {delta:.6f}s "
                f"({video_duration:.6f}s vs {audio_duration:.6f}s); tolerance is one {FPS} FPS frame.",
            )
    return facts


def preflight_packet(
    packet: MMH3Media,
    *,
    operation: str = "inspect",
    runtime_nodes: Sequence[str] | None = None,
    runtime_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    required_nodes: Sequence[str] = (),
    lora_inventory: Mapping[str, str | None] | None = None,
    allow_unknown_loras: bool = False,
    generation_mode: str = "auto",
    model_family: str = "auto",
    checkpoint_name: str = "",
    allow_experimental_model_family: bool = False,
) -> PreflightReport:
    if operation not in PREFLIGHT_OPERATIONS:
        raise ValueError(f"Unsupported preflight operation {operation!r}")
    info = inspect_packet(packet)
    diagnostics: list[PreflightDiagnostic] = []
    geometry = info["geometry"]
    width, height, frames, fps = (geometry.get(key) for key in ("width", "height", "frames", "fps"))

    for warning in info.get("warnings", ()):  # existing latent/manifest owner remains authoritative
        _add(diagnostics, "warning", "packet_inspection", str(warning))
    if operation != "inspect":
        for key, value in (("width", width), ("height", height), ("frames", frames)):
            if value is None:
                _add(diagnostics, "error", f"missing_{key}", f"Operation {operation} requires resolved {key}.")
    if width is not None and int(width) % 32:
        _add(diagnostics, "error", "width_not_multiple_32", f"Width {width} is not divisible by 32.")
    if height is not None and int(height) % 32:
        _add(diagnostics, "error", "height_not_multiple_32", f"Height {height} is not divisible by 32.")
    if frames is not None and (int(frames) < 5 or (int(frames) - 5) % 17):
        _add(diagnostics, "error", "frames_off_h3_grid", f"Frame count {frames} is not on the H3 17n+5 grid.")
    if fps is not None and float(fps) != float(FPS):
        _add(diagnostics, "error", "fps_not_24", f"H3 requires 24 FPS; packet resolves to {fps}.")

    has_latent = bool(info["has"]["latent"])
    if operation in ("continuation", "high_sigma_refine") and not has_latent:
        _add(diagnostics, "error", "missing_h3_latent", f"{operation} requires a primary H3 latent.")
    contract = info.get("h3_latent_contract") or {}
    origin = contract.get("origin", "unknown") if isinstance(contract, dict) else "unknown"
    if operation in ("continuation", "high_sigma_refine") and has_latent:
        try:
            if operation == "continuation":
                validate_h3_continuation_compatibility(contract)
            else:
                validate_h3_refine_compatibility(contract, width=int(width) if width is not None else None, height=int(height) if height is not None else None)
        except Exception as e:
            code = "unsafe_latent_origin" if origin != "sampler_output" else "unsafe_h3_latent_contract"
            _add(diagnostics, "error", code, str(e))
    if operation == "stitch":
        _add(diagnostics, "warning", "raw_latent_concat_forbidden", "Preflight never approves raw H3 latent concatenation; use a seam/stitch adapter.")

    loras = get_generation_loras(packet)
    if operation in ("high_sigma_refine", "decoded_high_sigma_refine"):
        if loras is None:
            severity = "warning" if allow_unknown_loras else "error"
            _add(diagnostics, severity, "lora_provenance_unknown", "Source generation LoRA stack is not recorded.")
        elif not loras:
            _add(diagnostics, "info", "lora_stack_empty", "Source generation explicitly used no LoRAs.")
        else:
            for index, entry in enumerate(loras):
                name = entry["name"]
                if not entry.get("reapply_for_high_sigma", True):
                    _add(diagnostics, "warning", "lora_reapply_opt_out", f"LoRA #{index + 1} {name!r} opts out of high-sigma reapply.")
                if lora_inventory is not None:
                    if name not in lora_inventory:
                        _add(diagnostics, "error", "lora_unavailable", f"Recorded LoRA {name!r} is unavailable.")
                    elif entry.get("sha256") and lora_inventory[name] != entry["sha256"]:
                        _add(diagnostics, "error", "lora_hash_mismatch", f"Recorded LoRA {name!r} SHA-256 does not match the available file.")

    task = str(info.get("generation", {}).get("task") or "")
    has = info["has"]
    refs = info["refs"]
    requirements = {
        "i2va": has["first_frame"], "l2va": has["last_frame"],
        "fl2va": has["first_frame"] and has["last_frame"],
        "ref2va": sum(int(value) for value in refs.values()) > 0,
    }
    if task in requirements and not requirements[task]:
        _add(diagnostics, "error", "task_resources_inconsistent", f"generation.task={task!r} lacks its required packet resources.")

    decoded_media = _decoded_media_facts(packet, diagnostics)
    if operation == "stitch":
        if not has["video"]:
            _add(diagnostics, "error", "missing_decoded_video", "Decoded-domain stitch requires primary video.")
        if not has["audio"]:
            _add(diagnostics, "error", "missing_decoded_audio", "Decoded-domain stitch requires primary audio.")
    if operation == "decoded_high_sigma_refine":
        if not has["video"]:
            _add(diagnostics, "error", "missing_decoded_video", "Decoded F07 refine requires primary video.")
        if not has["audio"]:
            _add(diagnostics, "warning", "missing_decoded_audio", "Decoded F07 refine requires explicit silence policy when primary audio is absent.")
    generation_contract = None
    resolved_generation_mode = task or None
    if operation == "generate":
        if generation_mode not in MODES:
            raise ValueError(f"Unsupported generation_mode {generation_mode!r}")
        resolved_generation_mode = resolve_packet(packet, mode=generation_mode).mode
        generation_contract = validate_generation_model_family(
            resolved_generation_mode,
            model_family=model_family,
            checkpoint_name=checkpoint_name,
            allow_experimental=allow_experimental_model_family,
        )
        for item in generation_contract.diagnostics:
            _add(diagnostics, item.severity, item.code, item.message)

    requested = list(dict.fromkeys(str(item) for item in required_nodes if str(item)))
    if operation == "generate":
        requested.append("MiniMaxH3ReferenceToVideo" if resolved_generation_mode == "ref2va" else "MiniMaxH3ImageToVideo")
    if runtime_nodes is None and requested:
        _add(diagnostics, "warning", "runtime_capabilities_unknown", "Runtime node registry was not supplied; required nodes were not verified.")
    elif runtime_nodes is not None:
        available = set(runtime_nodes)
        for node_id in dict.fromkeys(requested):
            if node_id not in available:
                _add(diagnostics, "error", "required_node_missing", f"Required runtime node {node_id!r} is not registered.")
    add_guide_facts = None
    if ADD_GUIDE_NODE_ID in requested and runtime_nodes is not None and ADD_GUIDE_NODE_ID in set(runtime_nodes):
        if runtime_contracts is None:
            _add(
                diagnostics,
                "warning",
                "add_guide_contract_unknown",
                "MiniMaxH3AddGuide is registered, but its schema was not supplied; arbitrary-position inputs were not verified.",
            )
        elif ADD_GUIDE_NODE_ID not in runtime_contracts:
            _add(
                diagnostics,
                "error",
                "add_guide_contract_missing",
                "MiniMaxH3AddGuide is registered, but its runtime schema could not be inspected.",
            )
        else:
            try:
                contract = parse_add_guide_contract(runtime_contracts[ADD_GUIDE_NODE_ID])
            except Exception as exc:
                _add(diagnostics, "error", "add_guide_contract_invalid", str(exc))
            else:
                add_guide_facts = {
                    "fingerprint": contract.fingerprint,
                    "frame_idx_min": contract.frame_idx_min,
                    "frame_idx_max": contract.frame_idx_max,
                    "supports_arbitrary_positions": contract.supports_arbitrary_positions,
                    "output_types": list(contract.output_types),
                }
    tile_guider_facts = None
    if TILE_GUIDER_NODE_ID in requested and runtime_nodes is not None and TILE_GUIDER_NODE_ID in set(runtime_nodes):
        if runtime_contracts is None:
            _add(
                diagnostics,
                "warning",
                "tile_guider_contract_unknown",
                f"{TILE_GUIDER_NODE_ID} is registered, but its schema was not supplied; tile inputs were not verified.",
            )
        elif TILE_GUIDER_NODE_ID not in runtime_contracts:
            _add(
                diagnostics,
                "error",
                "tile_guider_contract_missing",
                f"{TILE_GUIDER_NODE_ID} is registered, but its runtime schema could not be inspected.",
            )
        else:
            try:
                contract = parse_tile_guider_contract(runtime_contracts[TILE_GUIDER_NODE_ID])
            except Exception as exc:
                _add(diagnostics, "error", "tile_guider_contract_invalid", str(exc))
            else:
                tile_guider_facts = {
                    "fingerprint": contract.fingerprint,
                    "output_types": list(contract.output_types),
                    "traversal_modes": list(contract.traversal_modes),
                    "overlap_modes": list(contract.overlap_modes),
                    "batch_default": contract.batch_default,
                    "supports_region_mask": contract.supports_region_mask,
                    "supports_anchor_context": contract.supports_anchor_context,
                }
    if operation == "masked_edit":
        _add(
            diagnostics,
            "error",
            "masked_edit_adapter_todo",
            "Stock H3 has a nested per-stream AV denoise-mask path, but MMH3 does not yet compile and runtime-verify user edit masks against it.",
        )

    facts = {
        "geometry": geometry,
        "h3": {"fps": FPS, "audio_sample_rate": AUDIO_SAMPLE_RATE, "audio_latent_rate": AUDIO_LATENT_FPS},
        "latent_origin": origin,
        "generation_task": task or None,
        "resolved_generation_mode": resolved_generation_mode,
        "generation_model_contract": None if generation_contract is None else generation_contract.to_dict(),
        "required_nodes": list(dict.fromkeys(requested)),
        "runtime_nodes_checked": runtime_nodes is not None,
        "runtime_contracts_checked": runtime_contracts is not None,
        "add_guide_contract": add_guide_facts,
        "tile_guider_contract": tile_guider_facts,
        "decoded_media": decoded_media,
        "lora_provenance": "unknown" if loras is None else ("empty" if not loras else "recorded"),
    }
    return PreflightReport(operation, tuple(diagnostics), facts)
