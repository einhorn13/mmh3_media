from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .core import MMH3Media
from .errors import MMH3ResourceError
from .lora_provenance import get_generation_loras
from .util import deep_copy_json


@dataclass(frozen=True)
class LoRAReapplyDiagnostic:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class LoRAReapplyPlan:
    provenance: str
    applied: tuple[dict[str, Any], ...]
    skipped: tuple[dict[str, Any], ...]
    diagnostics: tuple[LoRAReapplyDiagnostic, ...]

    @property
    def ready(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def summary(self) -> str:
        state = "READY" if self.ready else "BLOCKED"
        return (
            f"{state} · F07 source LoRA reapply · provenance={self.provenance} · "
            f"apply={len(self.applied)} skipped={len(self.skipped)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "ready": self.ready,
            "provenance": self.provenance,
            "applied": [deep_copy_json(item) for item in self.applied],
            "skipped": [deep_copy_json(item) for item in self.skipped],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class LoRAReapplyExpansion:
    model: Any
    clip: Any
    graph: dict[str, Any]


def build_high_sigma_lora_plan(
    packet: MMH3Media,
    *,
    lora_inventory: Mapping[str, str | None] | None = None,
    allow_unknown: bool = False,
    allow_missing: bool = False,
) -> LoRAReapplyPlan:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    recorded = get_generation_loras(packet)
    diagnostics: list[LoRAReapplyDiagnostic] = []
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if recorded is None:
        diagnostics.append(
            LoRAReapplyDiagnostic(
                "warning" if allow_unknown else "error",
                "lora_provenance_unknown",
                "Source generation LoRA stack is not recorded; continuing requires an explicit override.",
            )
        )
        return LoRAReapplyPlan("unknown", (), (), tuple(diagnostics))
    if not recorded:
        diagnostics.append(
            LoRAReapplyDiagnostic("info", "lora_stack_empty", "Source generation explicitly used no LoRAs.")
        )
        return LoRAReapplyPlan("empty", (), (), tuple(diagnostics))

    for index, source in enumerate(recorded):
        entry = deep_copy_json(source)
        entry["source_index"] = index
        name = entry["name"]
        if not entry.get("reapply_for_high_sigma", True):
            skipped.append({**entry, "skip_reason": "source_opt_out"})
            diagnostics.append(
                LoRAReapplyDiagnostic(
                    "info", "lora_source_opt_out", f"LoRA #{index + 1} {name!r} opted out of high-sigma reapply."
                )
            )
            continue
        loader = entry.get("loader")
        if loader not in (None, "LoraLoader", "LoraLoaderModelOnly"):
            diagnostics.append(
                LoRAReapplyDiagnostic(
                    "error",
                    "lora_loader_unsupported",
                    f"Recorded LoRA #{index + 1} {name!r} used unsupported loader semantics {loader!r}.",
                )
            )
            skipped.append({**entry, "skip_reason": "unsupported_loader"})
            continue
        if loader == "LoraLoaderModelOnly" and entry.get("strength_clip") is not None:
            diagnostics.append(
                LoRAReapplyDiagnostic(
                    "error",
                    "lora_loader_contract_inconsistent",
                    f"Recorded LoRA #{index + 1} {name!r} declares model-only loading with a CLIP strength.",
                )
            )
            skipped.append({**entry, "skip_reason": "loader_contract_inconsistent"})
            continue
        if lora_inventory is not None and name not in lora_inventory:
            severity = "warning" if allow_missing else "error"
            diagnostics.append(
                LoRAReapplyDiagnostic(severity, "lora_unavailable", f"Recorded LoRA #{index + 1} {name!r} is unavailable.")
            )
            skipped.append({**entry, "skip_reason": "missing_override" if allow_missing else "missing"})
            continue
        available_hash = None if lora_inventory is None else lora_inventory.get(name)
        if entry.get("sha256") and available_hash != entry["sha256"]:
            diagnostics.append(
                LoRAReapplyDiagnostic(
                    "error",
                    "lora_hash_mismatch" if available_hash else "lora_hash_unverified",
                    f"Recorded LoRA #{index + 1} {name!r} SHA-256 was not matched exactly.",
                )
            )
            skipped.append({**entry, "skip_reason": "hash_mismatch" if available_hash else "hash_unverified"})
            continue
        applied.append(entry)
    return LoRAReapplyPlan("recorded", tuple(applied), tuple(skipped), tuple(diagnostics))


def build_lora_reapply_expansion(
    plan: LoRAReapplyPlan,
    *,
    model: Any,
    clip: Any,
    graph_builder_factory: Callable[[], Any] | None = None,
) -> LoRAReapplyExpansion:
    if not plan.ready:
        raise MMH3ResourceError(plan.summary())
    if graph_builder_factory is None:
        from comfy_execution.graph_utils import GraphBuilder  # type: ignore

        graph_builder_factory = GraphBuilder
    graph = graph_builder_factory()
    current_model = model
    current_clip = clip
    for entry in plan.applied:
        if entry.get("strength_clip") is None:
            node = graph.node(
                "LoraLoaderModelOnly",
                model=current_model,
                lora_name=entry["name"],
                strength_model=float(entry["strength_model"]),
            )
            current_model = node.out(0)
        else:
            node = graph.node(
                "LoraLoader",
                model=current_model,
                clip=current_clip,
                lora_name=entry["name"],
                strength_model=float(entry["strength_model"]),
                strength_clip=float(entry["strength_clip"]),
            )
            current_model = node.out(0)
            current_clip = node.out(1)
    return LoRAReapplyExpansion(current_model, current_clip, graph.finalize())
