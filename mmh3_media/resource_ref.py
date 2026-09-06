from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import MMH3ResourceError
from .util import deep_copy_json
from .resource_model import validate_resource_descriptor

if TYPE_CHECKING:
    from .core import MMH3Media


@dataclass(frozen=True)
class MMH3ResourceRef:
    """Lazy handle to one resource in one immutable MMH3 packet snapshot.

    The handle never materializes payload data during construction or metadata
    inspection.  Because :class:`MMH3Media` updates are immutable, a ref keeps
    snapshot semantics: callers must explicitly rebind it to a newer packet.
    """

    packet: "MMH3Media"
    resource_id: str

    def __post_init__(self) -> None:
        from .core import MMH3Media

        if not isinstance(self.packet, MMH3Media):
            raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise MMH3ResourceError("resource_id must be a non-empty string")
        if self.packet.get_by_id(self.resource_id) is None:
            raise MMH3ResourceError(f"Resource {self.resource_id!r} does not exist")

    @property
    def packet_id(self) -> str:
        return str(self.packet.manifest["id"])

    @property
    def descriptor(self) -> dict[str, Any]:
        """Return the canonical v0.3 resource descriptor without payload I/O."""
        resource = self.packet.get_by_id(self.resource_id)
        if resource is None:
            raise MMH3ResourceError(f"Resource {self.resource_id!r} does not exist")
        validate_resource_descriptor(resource)
        return deep_copy_json(resource)

    @property
    def kind(self) -> str:
        return str(self.descriptor["kind"])

    @property
    def role(self) -> str:
        return str(self.descriptor["role"])

    @property
    def order(self) -> int | None:
        value = self.descriptor["order"]
        return int(value) if value is not None else None

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(self.descriptor["tags"])

    def materialize(self) -> Any:
        from .archive import get_resource_payload

        resource = self.packet.get_by_id(self.resource_id)
        if resource is None:
            raise MMH3ResourceError(f"Resource {self.resource_id!r} does not exist")
        return get_resource_payload(self.packet, resource)

    def preview_info(self) -> dict[str, Any]:
        """Return descriptor/cached-representation preview data without source I/O."""
        from .preview import preview_info

        return preview_info(self.packet, target=self.resource_id)

    def representations(self, *, kind: str | None = None, fresh_only: bool = False):
        return self.packet.representations_for(self.resource_id, kind=kind, fresh_only=fresh_only)

    def fingerprint(self) -> str:
        """Return a deterministic logical fingerprint without payload I/O."""

        descriptor = self.descriptor
        descriptor.pop("path", None)
        payload = {
            "packet_id": self.packet_id,
            "resource": descriptor,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def rebind(self, packet: "MMH3Media") -> "MMH3ResourceRef":
        """Bind this stable resource ID to a newer snapshot of the same packet."""

        from .core import MMH3Media

        if not isinstance(packet, MMH3Media):
            raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
        if str(packet.manifest.get("id", "")) != self.packet_id:
            raise MMH3ResourceError(
                f"Cannot rebind resource {self.resource_id!r} from packet {self.packet_id!r} "
                f"to different packet {packet.manifest.get('id')!r}"
            )
        return packet.ref(self.resource_id)
