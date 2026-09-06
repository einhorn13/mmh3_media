from __future__ import annotations

from dataclasses import dataclass

from .errors import MMH3ResourceError


H3_MODEL_FAMILIES = ("auto", "unknown", "fl2va", "ref2va", "hybrid")
H3_GENERATION_MODES = ("t2va", "i2va", "fl2va", "l2va", "ref2va")
H3_CONTINUATION_FAMILIES = ("auto", "fl2va", "ref2va")


@dataclass(frozen=True)
class ModelFamilyDiagnostic:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class GenerationModelContract:
    mode: str
    expected_family: str
    resolved_family: str
    family_source: str
    checkpoint_name: str
    experimental_opt_in: bool
    diagnostics: tuple[ModelFamilyDiagnostic, ...]

    @property
    def ready(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "mode": self.mode,
            "expected_family": self.expected_family,
            "resolved_family": self.resolved_family,
            "family_source": self.family_source,
            "checkpoint_name": self.checkpoint_name or None,
            "experimental_opt_in": self.experimental_opt_in,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def resolve_h3_continuation_family(
    requested_family: str,
    *,
    has_references: bool,
) -> tuple[str, str]:
    """Resolve continuation conditioning independently from the source latent family.

    Continuation state is carried by the protected joint-AV latent prefix. The model
    family used to generate the unknown future is a separate choice:

    - FL2VA continuation uses text-only native H3 conditioning (``t2va``) because
      the preserved prefix already owns the temporal start state. Packet keyframes
      and references are intentionally not consumed by this branch.
    - Ref2VA continuation uses ordinary ``ref2va`` conditioning and therefore
      requires at least one active packet reference.
    - ``auto`` selects Ref2VA when active references exist, otherwise FL2VA.

    The returned pair is ``(task_family, native_conditioning_mode)``.
    """
    requested = str(requested_family or "auto").strip().lower()
    if requested not in H3_CONTINUATION_FAMILIES:
        raise MMH3ResourceError(
            f"Unsupported H3 continuation family {requested!r}; "
            f"expected one of {H3_CONTINUATION_FAMILIES}"
        )
    family = ("ref2va" if has_references else "fl2va") if requested == "auto" else requested
    if family == "ref2va" and not has_references:
        raise MMH3ResourceError(
            "Ref2VA continuation requires at least one active packet reference. "
            "Add/reference media upstream or select FL2VA continuation."
        )
    return family, ("ref2va" if family == "ref2va" else "t2va")


def infer_h3_model_family(checkpoint_name: str) -> str:
    """Infer only explicit task-family tokens from a checkpoint identifier.

    Unknown names stay unknown; this function never guesses from a ComfyUI/core version.
    """
    normalized = str(checkpoint_name or "").strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    has_fl2va = "fl2va" in normalized
    has_ref2va = "ref2va" in normalized
    if "hybrid" in normalized or (has_fl2va and has_ref2va):
        return "hybrid"
    if has_ref2va:
        return "ref2va"
    if has_fl2va:
        return "fl2va"
    return "unknown"


def validate_generation_model_family(
    mode: str,
    *,
    model_family: str = "auto",
    checkpoint_name: str = "",
    allow_experimental: bool = False,
) -> GenerationModelContract:
    mode = str(mode or "").strip().lower()
    if mode not in H3_GENERATION_MODES:
        raise MMH3ResourceError(
            f"Unsupported H3 generation mode {mode!r}; expected one of {H3_GENERATION_MODES}"
        )
    declared = str(model_family or "auto").strip().lower()
    if declared not in H3_MODEL_FAMILIES:
        raise MMH3ResourceError(
            f"Unsupported H3 model family {declared!r}; expected one of {H3_MODEL_FAMILIES}"
        )

    expected = "ref2va" if mode == "ref2va" else "fl2va"
    inferred = infer_h3_model_family(checkpoint_name)
    diagnostics: list[ModelFamilyDiagnostic] = []
    if declared == "auto":
        resolved = inferred
        source = "checkpoint_name" if inferred != "unknown" else "unknown"
    else:
        resolved = declared
        source = "explicit"
        if inferred != "unknown" and resolved != inferred:
            diagnostics.append(
                ModelFamilyDiagnostic(
                    "error",
                    "model_family_declaration_mismatch",
                    f"Declared model family {resolved!r} conflicts with checkpoint name {checkpoint_name!r}, "
                    f"which identifies {inferred!r}.",
                )
            )

    if resolved == "unknown":
        diagnostics.append(
            ModelFamilyDiagnostic(
                "warning",
                "model_family_unknown",
                f"Cannot prove that the selected checkpoint belongs to the required {expected!r} family for mode {mode!r}.",
            )
        )
    elif resolved == "hybrid":
        if allow_experimental:
            diagnostics.append(
                ModelFamilyDiagnostic(
                    "warning",
                    "experimental_model_family",
                    f"Experimental hybrid checkpoint was explicitly allowed for mode {mode!r}; baseline parity is not implied.",
                )
            )
        else:
            diagnostics.append(
                ModelFamilyDiagnostic(
                    "error",
                    "experimental_model_family_requires_opt_in",
                    "Hybrid FL2VA/Ref2VA checkpoints require explicit experimental opt-in.",
                )
            )
    elif resolved != expected:
        diagnostics.append(
            ModelFamilyDiagnostic(
                "error",
                "model_family_mismatch",
                f"Mode {mode!r} requires the {expected!r} checkpoint family; got {resolved!r}.",
            )
        )
    else:
        diagnostics.append(
            ModelFamilyDiagnostic(
                "info",
                "model_family_match",
                f"Mode {mode!r} and checkpoint family {resolved!r} are consistent.",
            )
        )

    return GenerationModelContract(
        mode,
        expected,
        resolved,
        source,
        str(checkpoint_name or "").strip(),
        bool(allow_experimental),
        tuple(diagnostics),
    )
