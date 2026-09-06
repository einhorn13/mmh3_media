from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .errors import MMH3ResourceError
from .util import deep_copy_json, is_jsonable


LORA_PURPOSES = (
    "unknown",
    "identity",
    "style",
    "detail",
    "motion",
    "quality",
    "acceleration",
    "other",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_generation_loras(value: Any) -> list[dict[str, Any]]:
    """Validate and normalize an ordered source-generation LoRA stack.

    Array order and duplicate entries are deliberately preserved: applying the
    same LoRA twice or changing application order can be semantically relevant.
    """
    if not isinstance(value, (list, tuple)):
        raise MMH3ResourceError("generation.loras must be an array")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise MMH3ResourceError(f"generation.loras[{index}] must be an object")
        if not is_jsonable(raw):
            raise MMH3ResourceError(f"generation.loras[{index}] must be JSON serializable")
        entry = deep_copy_json(dict(raw))
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MMH3ResourceError(f"generation.loras[{index}].name must be a non-empty string")
        entry["name"] = name.strip().replace("\\", "/")

        strength_model = entry.get("strength_model")
        if isinstance(strength_model, bool):
            raise MMH3ResourceError(f"generation.loras[{index}].strength_model must be a finite number")
        try:
            entry["strength_model"] = float(strength_model)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MMH3ResourceError(
                f"generation.loras[{index}].strength_model must be a finite number"
            ) from exc
        if not math.isfinite(entry["strength_model"]):
            raise MMH3ResourceError(f"generation.loras[{index}].strength_model must be a finite number")

        strength_clip = entry.get("strength_clip")
        if strength_clip is not None:
            if isinstance(strength_clip, bool):
                raise MMH3ResourceError(
                    f"generation.loras[{index}].strength_clip must be a finite number or null"
                )
            try:
                strength_clip = float(strength_clip)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MMH3ResourceError(
                    f"generation.loras[{index}].strength_clip must be a finite number or null"
                ) from exc
            if not math.isfinite(strength_clip):
                raise MMH3ResourceError(
                    f"generation.loras[{index}].strength_clip must be a finite number or null"
                )
        entry["strength_clip"] = strength_clip

        sha256 = entry.get("sha256")
        if sha256 is not None:
            if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256.lower()):
                raise MMH3ResourceError(
                    f"generation.loras[{index}].sha256 must be a 64-character hexadecimal digest"
                )
            entry["sha256"] = sha256.lower()

        purpose = entry.get("purpose", "unknown")
        if purpose not in LORA_PURPOSES:
            raise MMH3ResourceError(
                f"generation.loras[{index}].purpose must be one of {LORA_PURPOSES}"
            )
        entry["purpose"] = purpose
        reapply = entry.get("reapply_for_high_sigma", True)
        if not isinstance(reapply, bool):
            raise MMH3ResourceError(
                f"generation.loras[{index}].reapply_for_high_sigma must be boolean"
            )
        entry["reapply_for_high_sigma"] = reapply

        for optional_string in ("loader", "source"):
            optional = entry.get(optional_string)
            if optional is not None and (not isinstance(optional, str) or not optional.strip()):
                raise MMH3ResourceError(
                    f"generation.loras[{index}].{optional_string} must be a non-empty string when present"
                )
            if isinstance(optional, str):
                entry[optional_string] = optional.strip().replace("\\", "/")
        normalized.append(entry)
    return normalized


def get_generation_loras(packet) -> tuple[dict[str, Any], ...] | None:
    """Return None for unknown provenance and an ordered tuple when recorded."""
    generation = packet.manifest.get("generation", {})
    if "loras" not in generation:
        return None
    return tuple(normalize_generation_loras(generation["loras"]))


def set_generation_loras(packet, loras: Sequence[Mapping[str, Any]]):
    normalized = normalize_generation_loras(loras)
    out = packet.edit_metadata(merge_patch_json={"generation": {"loras": normalized}})
    return out.record_operation(
        "generation_loras_set",
        count=len(normalized),
        names=[entry["name"] for entry in normalized],
    )


def clear_generation_loras(packet):
    out = packet.edit_metadata(merge_patch_json={"generation": {"loras": None}})
    return out.record_operation("generation_loras_clear")


def lora_provenance_summary(packet) -> str:
    loras = get_generation_loras(packet)
    if loras is None:
        return "LoRA provenance: unknown (generation.loras is not recorded)"
    if not loras:
        return "LoRA provenance: recorded; source generation used no LoRAs"
    reapply = sum(1 for entry in loras if entry["reapply_for_high_sigma"])
    names = ", ".join(entry["name"] for entry in loras)
    return f"LoRA provenance: {len(loras)} recorded in application order; high-sigma reapply={reapply} · {names}"
