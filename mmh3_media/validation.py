from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import FORMAT_NAME, SCHEMA_VERSION
from .errors import MMH3FormatError
from .schema import validate_manifest


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors), "warnings": list(self.warnings)}


def validate_packet_manifest(manifest: Any) -> ValidationReport:
    """Single internal validation entrypoint used by core/archive/nodes.

    This validates the only published MMH3_MEDIA schema: schema v1. The function is
    intentionally side-effect free so later validators can add resource/H3 checks without changing nodes.
    """
    try:
        validate_manifest(manifest, allow_unknown_kinds=False)
    except Exception as e:
        return ValidationReport(False, (str(e),), ())
    warnings: list[str] = []
    if manifest.get("format") != FORMAT_NAME:
        return ValidationReport(False, (f"format must be {FORMAT_NAME}",), ())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return ValidationReport(False, (f"schema_version must be {SCHEMA_VERSION}",), ())
    if not manifest.get("name"):
        warnings.append("packet name is empty")
    return ValidationReport(True, (), tuple(warnings))


def require_valid_manifest(manifest: Any) -> None:
    report = validate_packet_manifest(manifest)
    if not report.valid:
        raise MMH3FormatError("; ".join(report.errors))
