"""Read-only review projection. Chain acceptance and execution acceptance are distinct.

No files are opened and no resources are materialized. Explicit candidate bindings
are caller-supplied observations of saved artifacts, never inferred from filenames,
packet IDs (which survive edits), job order, or attempt numbers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .automation_execution import _validate_ledger
from .chain import validate_chain
from .core import MMH3Media
from .errors import MMH3ResourceError
from .util import json_dumps_canonical

PROJECT_REVIEW_CONTRACT = "mmh3_project_review_v1"


def _hash(value: Any) -> str:
    return hashlib.sha256(json_dumps_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReviewCandidateBinding:
    """Verified artifact-to-chain observation; not a mutation or persistence record."""

    job_id: str
    candidate_id: str
    segment_id: str
    revision: int
    packet_state_fingerprint: str


@dataclass(frozen=True)
class ReviewCandidate:
    job_id: str
    candidate_id: str
    attempt: int
    packet_path: str | None
    artifact_digest: str | None
    selected: bool
    execution_accepted: bool
    accepted: bool
    segment_id: str | None
    revision: int | None
    artifact_json: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["artifact"] = json.loads(result.pop("artifact_json"))
        return result


@dataclass(frozen=True)
class ReviewJob:
    job_id: str
    state: str
    attempts: int
    selected_candidate_id: str | None
    candidates: tuple[ReviewCandidate, ...]
    artifact_json: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        result["artifact"] = json.loads(result.pop("artifact_json"))
        return result


@dataclass(frozen=True)
class ReviewSegment:
    segment_id: str
    scene_id: str
    index: int
    status: str
    active_revision: int
    parent_segment_id: str
    packet_state_fingerprint: str
    archive_prefix: str
    candidates: tuple[ReviewCandidate, ...]
    selected_candidate_id: str | None
    record_json: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        result["record"] = json.loads(result.pop("record_json"))
        return result


@dataclass(frozen=True)
class ProjectReviewState:
    project_id: str
    name: str
    chain_id: str | None
    active_scene_id: str | None
    head_segment_id: str | None
    head_state_matches: bool | None
    head_state_reasons: tuple[str, ...]
    segments: tuple[ReviewSegment, ...]
    jobs: tuple[ReviewJob, ...]
    execution_revision: int | None
    draft_json: str
    source_digest: str

    @property
    def state_digest(self) -> str:
        return project_state_digest(self)

    def to_dict(self) -> dict[str, Any]:
        result = _state_dict(self)
        result["state_digest"] = self.state_digest
        return result


def _state_dict(state: ProjectReviewState) -> dict[str, Any]:
    result = asdict(state)
    result["contract"] = PROJECT_REVIEW_CONTRACT
    result["segments"] = [segment.to_dict() for segment in state.segments]
    result["jobs"] = [job.to_dict() for job in state.jobs]
    result["draft"] = json.loads(result.pop("draft_json"))
    return result


def project_state_digest(review_state: ProjectReviewState) -> str:
    """Hash the snapshot, excluding the derived digest itself; no mtime or I/O."""
    return _hash(_state_dict(review_state))


def build_project_review_state(
    packet: MMH3Media,
    execution_ledger: Mapping[str, Any] | None = None,
    *,
    candidate_bindings: tuple[ReviewCandidateBinding, ...] = (),
) -> ProjectReviewState:
    """Project existing state without inventing a common job/segment identity.

    ``segments`` is authoritative chain history; ``jobs`` retains all execution
    states, including unbound candidates and legacy direct-completion artifacts.
    A candidate is chain-accepted only when execution selected it AND its explicit
    binding matches the active chain revision/fingerprint. Bindings to superseded
    revisions remain visible but cannot become accepted. Missing preview, creation
    time, seam and storage metadata are left absent in the artifact/record objects.

    ``execution_revision`` is only the F18 counter. Chain v1 has no revision counter;
    use ``state_digest`` for snapshot identity, not a fabricated global revision.
    This digest is not on-disk artifact validation or a concurrency lock.
    """
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    manifest = packet.manifest
    namespace = manifest.get("extensions", {}).get("mmh3_media", {})
    chain = namespace.get("chain")
    if chain is not None:
        report = validate_chain(packet, require_head_state=False)
        if not report.ready:
            raise MMH3ResourceError("Invalid review chain: " + "; ".join(report.reasons))
    records = (chain or {}).get("segments", [])
    head_report = validate_chain(packet) if chain is not None else None
    for record in records:
        if (type(record.get("revision")) is not int or record["revision"] < 1
                or not isinstance(record.get("scene_id"), str)
                or not isinstance(record.get("packet_state_fingerprint"), str)
                or not record["packet_state_fingerprint"]):
            raise MMH3ResourceError("Invalid review chain segment revision/state")
    by_id = {record["segment_id"]: record for record in records}
    if execution_ledger is not None:
        _validate_ledger(execution_ledger)
        revision = execution_ledger.get("revision")
        if type(revision) is not int or revision < 0:
            raise MMH3ResourceError("Invalid execution ledger revision")
    else:
        revision = None
    entries = (execution_ledger or {}).get("jobs", [])
    known_candidates = {
        (entry["job_id"], candidate["candidate_id"])
        for entry in entries for candidate in entry.get("candidates", [])
    }
    bindings = {}
    for binding in candidate_bindings:
        key = (binding.job_id, binding.candidate_id)
        if key not in known_candidates or key in bindings:
            raise MMH3ResourceError("Unknown or duplicate review candidate binding")
        record = by_id.get(binding.segment_id)
        if record is None or type(binding.revision) is not int or binding.revision < 1:
            raise MMH3ResourceError("Invalid review binding segment/revision")
        versions = [record, *record.get("superseded_revisions", [])]
        matches = [version for version in versions if version.get("revision") == binding.revision]
        if (len(matches) != 1 or not binding.packet_state_fingerprint
                or matches[0].get("packet_state_fingerprint") != binding.packet_state_fingerprint):
            raise MMH3ResourceError("Review binding does not match recorded revision fingerprint")
        bindings[key] = binding
    jobs = []
    linked: dict[str, list[ReviewCandidate]] = {}
    for entry in entries:
        candidates = []
        selected_id = entry.get("selected_candidate_id")
        for item in entry.get("candidates", []):
            selected = selected_id == item["candidate_id"]
            execution_accepted = selected and entry["state"] == "completed"
            if execution_accepted and entry.get("artifact") != item:
                raise MMH3ResourceError("Accepted execution artifact differs from selected candidate")
            binding = bindings.get((entry["job_id"], item["candidate_id"]))
            record = by_id[binding.segment_id] if binding else None
            accepted = bool(execution_accepted and record
                            and record["status"] == "active"
                            and record["revision"] == binding.revision)
            candidate = ReviewCandidate(
                job_id=entry["job_id"], candidate_id=item["candidate_id"],
                attempt=item.get("attempt", 0), packet_path=item.get("packet_path"),
                artifact_digest=item.get("sha256"), selected=selected,
                execution_accepted=execution_accepted, accepted=accepted,
                segment_id=binding.segment_id if binding else None,
                revision=binding.revision if binding else None,
                artifact_json=json_dumps_canonical(item),
            )
            candidates.append(candidate)
            if binding:
                linked.setdefault(binding.segment_id, []).append(candidate)
        jobs.append(ReviewJob(
            job_id=entry["job_id"], state=entry["state"], attempts=entry["attempts"],
            selected_candidate_id=selected_id,
            candidates=tuple(sorted(candidates, key=lambda item: item.candidate_id)),
            artifact_json=json_dumps_canonical(entry.get("artifact")), error=entry.get("error"),
        ))
    segments = []
    for record in records:
        candidates = tuple(sorted(linked.get(record["segment_id"], []),
                                  key=lambda item: (item.job_id, item.candidate_id)))
        accepted = [candidate for candidate in candidates if candidate.accepted]
        if len(accepted) > 1:
            raise MMH3ResourceError("Multiple accepted candidates bound to one chain revision")
        segments.append(ReviewSegment(
            segment_id=record["segment_id"], scene_id=record["scene_id"], index=record["index"],
            status="accepted" if record["status"] == "active" else "invalidated",
            active_revision=record["revision"], parent_segment_id=record.get("parent_segment_id", ""),
            packet_state_fingerprint=record["packet_state_fingerprint"],
            archive_prefix=record.get("archive_prefix", ""), candidates=candidates,
            selected_candidate_id=accepted[0].candidate_id if accepted else None,
            record_json=json_dumps_canonical(record),
        ))
    # Resource content.revision is semantic identity; save-added digest/size and
    # archive location must not invalidate an otherwise identical packet snapshot.
    source_digest = _hash({
        "packet_id": manifest["id"], "generation": manifest.get("generation", {}),
        "name": manifest.get("name"), "notes": manifest.get("notes"),
        "primary": manifest.get("primary", {}),
        "extensions": manifest.get("extensions", {}),
        "resources": sorted([
            {"id": resource["id"], "kind": resource["kind"],
             "name": resource.get("name"), "role": resource.get("role"),
             "order": resource.get("order"), "tags": resource.get("tags"),
             "extensions": resource.get("extensions", {}),
             "revision": (resource.get("content") or {}).get("revision")}
            for resource in manifest.get("resources", [])
        ], key=lambda resource: resource["id"]),
        "execution_ledger": execution_ledger,
    })
    draft = namespace.get("segment_plan")
    if draft and draft.get("state") == "accepted":
        draft = None
    return ProjectReviewState(
        project_id=(chain or {}).get("chain_id") or manifest["id"],
        name=(chain or {}).get("name") or manifest.get("name", ""),
        chain_id=(chain or {}).get("chain_id"), active_scene_id=(chain or {}).get("active_scene_id"),
        head_segment_id=(chain or {}).get("head_segment_id") or None,
        head_state_matches=head_report.ready if head_report else None,
        head_state_reasons=head_report.reasons if head_report else (),
        segments=tuple(sorted(segments, key=lambda segment: segment.index)),
        jobs=tuple(jobs), execution_revision=revision,
        draft_json=json_dumps_canonical(draft), source_digest=source_digest,
    )


__all__ = [
    "PROJECT_REVIEW_CONTRACT", "ProjectReviewState", "ReviewSegment", "ReviewJob",
    "ReviewCandidate", "ReviewCandidateBinding", "build_project_review_state", "project_state_digest",
]
