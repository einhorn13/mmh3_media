from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .core import MMH3Media
from .errors import MMH3ResourceError
from .util import deep_copy_json, json_dumps_canonical, utc_now_iso
from .h3_resource_semantics import find_context_resource


CHAIN_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")


@dataclass(frozen=True)
class ChainValidation:
    ready: bool
    chain_id: str
    head_segment_id: str
    active_segments: int
    invalidated_segments: int
    should_reanchor: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "contract": "mmh3_chain_v1",
            "chain_id": self.chain_id,
            "head_segment_id": self.head_segment_id,
            "active_segments": self.active_segments,
            "invalidated_segments": self.invalidated_segments,
            "should_reanchor": self.should_reanchor,
            "reasons": list(self.reasons),
        }

    def summary(self) -> str:
        status = "READY" if self.ready else "BLOCKED"
        suffix = " · re-anchor recommended" if self.should_reanchor else ""
        return (
            f"{status} · F04 chain {self.chain_id or '(missing)'} · "
            f"active={self.active_segments} invalidated={self.invalidated_segments}{suffix}"
        )


@dataclass(frozen=True)
class ChainCommitResult:
    packet: MMH3Media
    segment_id: str
    scene_id: str
    revision: int
    filename_prefix: str
    validation: ChainValidation


@dataclass(frozen=True)
class RerollSourceValidation:
    ready: bool
    target_segment_id: str
    expected_parent_segment_id: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "contract": "mmh3_chain_reroll_source_v1",
            "target_segment_id": self.target_segment_id,
            "expected_parent_segment_id": self.expected_parent_segment_id,
            "reasons": list(self.reasons),
        }


def _clean_id(value: str, *, label: str) -> str:
    value = str(value or "").strip()
    if not _ID_RE.fullmatch(value):
        raise MMH3ResourceError(
            f"{label} must contain only letters, digits, '.', '_' or '-' and be at most 120 characters"
        )
    return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _derived_id(prefix: str, *parts: Any) -> str:
    seed = ":".join(str(part) for part in parts)
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"


def _chain_from_packet(packet: MMH3Media) -> dict[str, Any] | None:
    namespace = packet.manifest.get("extensions", {}).get("mmh3_media")
    if not isinstance(namespace, dict):
        return None
    chain = namespace.get("chain")
    return deep_copy_json(chain) if isinstance(chain, dict) else None


def _packet_state_fingerprint(packet: MMH3Media, output_resource_ids: dict[str, str]) -> str:
    resources: list[dict[str, Any]] = []
    for role, resource_id in sorted(output_resource_ids.items()):
        descriptor = packet.get_by_id(resource_id)
        if descriptor is None:
            raise MMH3ResourceError(
                f"Last process output {role} points to missing resource {resource_id!r}"
            )
        resources.append(
            {
                "role": role,
                "id": resource_id,
                "kind": descriptor.get("kind"),
                "revision": (descriptor.get("content") or {}).get("revision"),
            }
        )
    payload = {
        "packet_id": packet.manifest["id"],
        "generation": packet.manifest.get("generation", {}),
        "resources": resources,
    }
    return hashlib.sha256(json_dumps_canonical(payload).encode("utf-8")).hexdigest()


def _last_process(packet: MMH3Media) -> dict[str, Any]:
    namespace = packet.manifest.get("extensions", {}).get("mmh3_media")
    process = namespace.get("last_process") if isinstance(namespace, dict) else None
    if not isinstance(process, dict):
        raise MMH3ResourceError("F04 segment commit requires extensions.mmh3_media.last_process")
    outputs = process.get("output_resource_ids")
    if not isinstance(outputs, dict) or not outputs:
        raise MMH3ResourceError("F04 segment commit requires last_process output_resource_ids")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in outputs.items()):
        raise MMH3ResourceError("last_process output_resource_ids must be a string map")
    return deep_copy_json(process)


def _primary_resource_ids(packet: MMH3Media) -> dict[str, str]:
    output: dict[str, str] = {}
    for kind in ("latent", "video", "audio"):
        descriptor = packet.get_primary(kind)
        if descriptor is not None:
            output[kind] = descriptor["id"]
    for usage in ("first_frame", "last_frame"):
        descriptor = find_context_resource(packet, usage)
        if descriptor is not None:
            output[usage] = descriptor["id"]
    return output


def _safe_archive_prefix(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise MMH3ResourceError("archive_prefix must be a safe portable relative path")
    if path.suffix.lower() == ".mmh3":
        path = path.with_suffix("")
    return path.as_posix()


def start_chain(
    packet: MMH3Media,
    *,
    name: str = "",
    chain_id: str = "",
    scene_id: str = "",
    reanchor_every_segments: int = 4,
) -> MMH3Media:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    existing = _chain_from_packet(packet)
    if existing is not None:
        report = validate_chain(packet, require_head_state=False)
        if not report.ready:
            raise MMH3ResourceError("Existing chain manifest is invalid: " + "; ".join(report.reasons))
        return packet
    reanchor_every_segments = int(reanchor_every_segments)
    if reanchor_every_segments < 1:
        raise MMH3ResourceError("reanchor_every_segments must be at least one")
    resolved_chain_id = _clean_id(chain_id, label="chain_id") if chain_id else _new_id("chain")
    resolved_scene_id = (
        _clean_id(scene_id, label="scene_id")
        if scene_id
        else _derived_id("scene", resolved_chain_id, 0)
    )
    chain = {
        "version": CHAIN_VERSION,
        "chain_id": resolved_chain_id,
        "name": str(name or packet.manifest.get("name") or "untitled chain"),
        "active_scene_id": resolved_scene_id,
        "head_segment_id": "",
        "policy": {
            "reanchor_every_segments": reanchor_every_segments,
            "primary_payload_retention": "latest_packet_only",
            "prior_segment_retention": "manifest_and_archive_reference",
        },
        "segments": [],
    }
    root_outputs = _primary_resource_ids(packet)
    if root_outputs:
        chain["root_state"] = {
            "packet_manifest_id": packet.manifest["id"],
            "output_resource_ids": root_outputs,
            "packet_state_fingerprint": _packet_state_fingerprint(packet, root_outputs),
        }
    return packet.set_extension_value("mmh3_media", "chain", chain).record_operation(
        "chain_start", chain_id=resolved_chain_id, scene_id=resolved_scene_id
    )


def _default_archive_prefix(chain_id: str, scene_id: str, segment_id: str, revision: int) -> str:
    return f"mmh3/chains/{chain_id}/{scene_id}/{segment_id}_r{revision:03d}"


def validate_chain(packet: MMH3Media, *, require_head_state: bool = True) -> ChainValidation:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    chain = _chain_from_packet(packet)
    if chain is None:
        return ChainValidation(False, "", "", 0, 0, False, ("Packet has no chain manifest.",))
    reasons: list[str] = []
    if chain.get("version") != CHAIN_VERSION:
        reasons.append(f"Unsupported chain version {chain.get('version')!r}.")
    chain_id = str(chain.get("chain_id") or "")
    try:
        _clean_id(chain_id, label="chain_id")
        _clean_id(str(chain.get("active_scene_id") or ""), label="active_scene_id")
    except MMH3ResourceError as exc:
        reasons.append(str(exc))
    segments = chain.get("segments")
    if not isinstance(segments, list):
        segments = []
        reasons.append("chain.segments must be an array.")
    ids: set[str] = set()
    indexes: set[int] = set()
    by_id: dict[str, dict[str, Any]] = {}
    for item in segments:
        if not isinstance(item, dict):
            reasons.append("Every chain segment must be an object.")
            continue
        segment_id = str(item.get("segment_id") or "")
        try:
            _clean_id(segment_id, label="segment_id")
        except MMH3ResourceError as exc:
            reasons.append(str(exc))
        if segment_id in ids:
            reasons.append(f"Duplicate segment_id {segment_id!r}.")
        ids.add(segment_id)
        by_id[segment_id] = item
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            reasons.append(f"Segment {segment_id!r} has invalid index.")
        elif index in indexes:
            reasons.append(f"Duplicate segment index {index}.")
        else:
            indexes.add(index)
        if item.get("status") not in {"active", "invalidated"}:
            reasons.append(f"Segment {segment_id!r} has invalid status.")
        parent = item.get("parent_segment_id")
        if parent and parent not in ids:
            reasons.append(f"Segment {segment_id!r} parent is missing or not earlier in the chain.")
    head = str(chain.get("head_segment_id") or "")
    if head:
        head_item = by_id.get(head)
        if head_item is None:
            reasons.append("head_segment_id does not exist.")
        elif head_item.get("status") != "active":
            reasons.append("head_segment_id is not active.")
        elif require_head_state:
            outputs = head_item.get("output_resource_ids")
            if not isinstance(outputs, dict):
                reasons.append("Head segment has no output_resource_ids.")
            else:
                try:
                    actual = _packet_state_fingerprint(packet, outputs)
                    if actual != head_item.get("packet_state_fingerprint"):
                        reasons.append("Current packet resources do not match the recorded chain head state.")
                except MMH3ResourceError as exc:
                    reasons.append(str(exc))
    elif segments:
        reasons.append("Non-empty chain has no head_segment_id.")

    active = [item for item in segments if isinstance(item, dict) and item.get("status") == "active"]
    invalidated = [item for item in segments if isinstance(item, dict) and item.get("status") == "invalidated"]
    policy = chain.get("policy") if isinstance(chain.get("policy"), dict) else {}
    interval = policy.get("reanchor_every_segments", 4)
    try:
        interval = max(1, int(interval))
    except (TypeError, ValueError):
        interval = 4
        reasons.append("reanchor_every_segments is invalid.")
    since_reanchor = 0
    for item in reversed(active):
        if item.get("handover") == "reanchor":
            break
        since_reanchor += 1
    return ChainValidation(
        not reasons,
        chain_id,
        head,
        len(active),
        len(invalidated),
        bool(active) and since_reanchor >= interval,
        tuple(reasons),
    )


def validate_reroll_source(
    chain_packet: MMH3Media,
    source_packet: MMH3Media,
    *,
    target_segment_id: str,
) -> RerollSourceValidation:
    target_segment_id = _clean_id(target_segment_id, label="target_segment_id")
    chain_report = validate_chain(chain_packet, require_head_state=True)
    reasons = list(chain_report.reasons)
    chain = _chain_from_packet(chain_packet)
    if chain is None:
        reasons.append("Authoritative packet has no chain manifest.")
        return RerollSourceValidation(False, target_segment_id, "", tuple(reasons))
    target = next(
        (item for item in chain.get("segments", []) if item.get("segment_id") == target_segment_id),
        None,
    )
    if target is None:
        reasons.append(f"Reroll target {target_segment_id!r} does not exist.")
        return RerollSourceValidation(False, target_segment_id, "", tuple(reasons))
    if target.get("status") != "active":
        reasons.append(f"Reroll target {target_segment_id!r} is not active.")
    parent_id = str(target.get("parent_segment_id") or "")
    if parent_id:
        expected = next(
            (item for item in chain.get("segments", []) if item.get("segment_id") == parent_id),
            None,
        )
    else:
        expected = chain.get("root_state")
    if not isinstance(expected, dict):
        reasons.append("The target parent/root state is not recorded in the chain manifest.")
    else:
        outputs = expected.get("output_resource_ids")
        if not isinstance(outputs, dict):
            reasons.append("The target parent/root state has no output_resource_ids.")
        else:
            try:
                actual = _packet_state_fingerprint(source_packet, outputs)
                if actual != expected.get("packet_state_fingerprint"):
                    reasons.append("Selected reroll source packet does not match the target parent state.")
            except MMH3ResourceError as exc:
                reasons.append(str(exc))
    return RerollSourceValidation(not reasons, target_segment_id, parent_id, tuple(reasons))


def commit_chain_segment(
    packet: MMH3Media,
    *,
    action: str = "append",
    target_segment_id: str = "",
    archive_prefix: str = "",
    chain_packet: MMH3Media | None = None,
) -> ChainCommitResult:
    if action not in {"append", "reroll", "reanchor"}:
        raise MMH3ResourceError("Chain action must be append, reroll or reanchor")
    ledger_packet = chain_packet if chain_packet is not None else packet
    ledger_packet = start_chain(ledger_packet)
    validation = validate_chain(ledger_packet, require_head_state=chain_packet is not None)
    if not validation.ready:
        raise MMH3ResourceError("Cannot commit invalid chain: " + "; ".join(validation.reasons))
    if action == "reroll" and chain_packet is None:
        raise MMH3ResourceError(
            "Reroll commit requires the authoritative chain_packet separately from the regenerated result packet"
        )
    chain = _chain_from_packet(ledger_packet)
    assert chain is not None
    packet = packet.set_extension_value("mmh3_media", "chain", chain)
    process = _last_process(packet)
    outputs = process["output_resource_ids"]
    fingerprint = _packet_state_fingerprint(packet, outputs)
    segments = chain["segments"]
    now = utc_now_iso()

    if action == "reroll":
        target_segment_id = _clean_id(target_segment_id, label="target_segment_id")
        target = next((item for item in segments if item.get("segment_id") == target_segment_id), None)
        if target is None:
            raise MMH3ResourceError(f"Reroll target {target_segment_id!r} does not exist")
        if target.get("status") != "active":
            raise MMH3ResourceError(f"Reroll target {target_segment_id!r} is not active")
        old_revision = {
            key: deep_copy_json(target.get(key))
            for key in (
                "revision",
                "packet_state_fingerprint",
                "output_resource_ids",
                "operation",
                "mode",
                "archive_prefix",
                "updated_at",
            )
        }
        target.setdefault("superseded_revisions", []).append(old_revision)
        revision = int(target.get("revision", 1)) + 1
        scene_id = target["scene_id"]
        for item in segments:
            if int(item.get("index", -1)) > int(target["index"]) and item.get("status") == "active":
                item["status"] = "invalidated"
                item["invalidated_by"] = target_segment_id
                item["invalidated_at"] = now
        segment = target
        handover = target.get("handover", "continuation")
    else:
        revision = 1
        next_index = max((int(item.get("index", -1)) for item in segments), default=-1) + 1
        if action == "reanchor":
            scene_ordinal = len({item.get("scene_id") for item in segments})
            scene_id = _derived_id("scene", chain["chain_id"], scene_ordinal)
            chain["active_scene_id"] = scene_id
            scene_index = 0
            handover = "reanchor"
        else:
            scene_id = chain["active_scene_id"]
            scene_index = 1 + max(
                (int(item.get("scene_index", -1)) for item in segments if item.get("scene_id") == scene_id),
                default=-1,
            )
            handover = "initial" if not segments else "continuation"
        target_segment_id = _derived_id("seg", chain["chain_id"], scene_id, next_index)
        segment = {
            "segment_id": target_segment_id,
            "scene_id": scene_id,
            "index": next_index,
            "scene_index": scene_index,
            "parent_segment_id": chain.get("head_segment_id") or "",
            "created_at": now,
            "superseded_revisions": [],
        }
        segments.append(segment)

    resolved_prefix = _safe_archive_prefix(
        archive_prefix
        or _default_archive_prefix(chain["chain_id"], scene_id, target_segment_id, revision)
    )
    segment.update(
        {
            "revision": revision,
            "status": "active",
            "handover": handover,
            "packet_manifest_id": packet.manifest["id"],
            "packet_state_fingerprint": fingerprint,
            "operation": str(process.get("operation") or ""),
            "mode": str(process.get("mode") or ""),
            "output_resource_ids": deep_copy_json(outputs),
            "archive_prefix": resolved_prefix,
            "updated_at": now,
        }
    )
    segment.pop("invalidated_by", None)
    segment.pop("invalidated_at", None)
    chain["head_segment_id"] = target_segment_id
    chain["active_scene_id"] = scene_id
    out = packet.set_extension_value("mmh3_media", "chain", chain).record_operation(
        "chain_segment_commit",
        chain_id=chain["chain_id"],
        scene_id=scene_id,
        segment_id=target_segment_id,
        revision=revision,
        action=action,
        invalidated_segment_ids=[
            item["segment_id"]
            for item in segments
            if item.get("status") == "invalidated" and item.get("invalidated_by") == target_segment_id
        ],
        archive_prefix=resolved_prefix,
    )
    final_validation = validate_chain(out, require_head_state=True)
    if not final_validation.ready:
        raise MMH3ResourceError("Committed chain failed validation: " + "; ".join(final_validation.reasons))
    return ChainCommitResult(
        out,
        target_segment_id,
        scene_id,
        revision,
        resolved_prefix,
        final_validation,
    )
