"""Guarded review actions and publication of a dedicated authoritative .mmh3.

Reopen prepares a replacement; it does not abandon the accepted head before a
replacement exists. Filesystem publication is deliberately separate from pure
domain actions. F18 ledger transitions remain owned by automation_execution.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .archive import _durable_replace, load_archive, save_archive
from .chain import inspect_chain_reroll, validate_chain
from .core import MMH3Media
from .errors import MMH3ResourceError
from .project_review import ReviewCandidateBinding, build_project_review_state
from .segment_plan import SegmentPlan, prepare_segment, review_segment
from .util import deep_copy_json, sha256_file, utc_now_iso


class ProjectStateConflict(MMH3ResourceError):
    """The caller's review snapshot is stale; obtain a fresh preview."""


class ProjectPublicationBusy(MMH3ResourceError):
    """Another process is publishing to this authoritative archive."""


@dataclass(frozen=True)
class ChainMutationPreview:
    action: str
    expected_state_digest: str
    target_segment_id: str
    current_revision: int
    next_revision: int
    affected_segment_ids: tuple[str, ...]
    invalidated_segment_ids: tuple[str, ...]
    affected_candidates: tuple[tuple[str, str], ...]
    archive_prefixes: tuple[str, ...]
    new_head_segment_id: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"contract": "mmh3_chain_mutation_preview_v1", "effective_on": "accept",
                "bytes_affected": None, **asdict(self)}


@dataclass(frozen=True)
class ProjectSegmentAcceptance:
    packet: MMH3Media
    previous_state_digest: str
    state_digest: str
    segment_id: str
    revision: int
    invalidated_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProjectPublication:
    packet: MMH3Media
    authoritative_path: str
    previous_snapshot_path: str
    operation_id: str
    state_digest: str
    already_applied: bool


def _require_digest(actual: str, expected: str, label: str = "Project state") -> None:
    if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not hmac.compare_digest(actual, expected)):
        raise ProjectStateConflict(f"{label} changed; refresh the review and preview before retrying")


def preview_chain_mutation(
    chain_packet: MMH3Media, *, target_segment_id: str, action: str = "reroll",
    execution_ledger: Mapping[str, Any] | None = None,
    candidate_bindings: tuple[ReviewCandidateBinding, ...] = (),
) -> ChainMutationPreview:
    """Pure blast-radius report using the same chain-v1 rule as the actual commit.

    Paths are recorded archive prefixes, not verified files. Unknown/unbound F18
    candidates cannot be assigned to a segment and are reported as a limitation.
    """
    if action not in {"reroll", "reopen"}:
        raise MMH3ResourceError("Preview action must be reroll or reopen")
    impact = inspect_chain_reroll(chain_packet, target_segment_id=target_segment_id)
    state = build_project_review_state(chain_packet, execution_ledger, candidate_bindings=candidate_bindings)
    affected = (impact.target_segment_id, *impact.invalidated_segment_ids)
    segments = [segment for segment in state.segments if segment.segment_id in affected]
    warnings = ["Chain v1 invalidates active segments after the target index; artifacts are preserved."]
    if action == "reopen":
        warnings.append("Reopen prepares a draft; the accepted head changes only when its replacement is accepted.")
    if any(candidate.segment_id is None for job in state.jobs for candidate in job.candidates):
        warnings.append("Unbound execution candidates exist; their lineage impact is unknown.")
    return ChainMutationPreview(
        action=action, expected_state_digest=state.state_digest,
        target_segment_id=impact.target_segment_id, current_revision=impact.current_revision,
        next_revision=impact.current_revision + 1, affected_segment_ids=affected,
        invalidated_segment_ids=impact.invalidated_segment_ids,
        affected_candidates=tuple((candidate.job_id, candidate.candidate_id)
            for segment in segments for candidate in segment.candidates
            if candidate.revision == segment.active_revision),
        archive_prefixes=tuple(segment.archive_prefix for segment in segments if segment.archive_prefix),
        new_head_segment_id=impact.target_segment_id, warnings=tuple(warnings),
    )


def prepare_project_reopen(
    chain_packet: MMH3Media, source_packet: MMH3Media, *, target_segment_id: str,
    expected_state_digest: str, prompts: str = "", seed: int = -1,
    frames: int = 124, context_frames: int = 39, reanchor_image=None,
    execution_ledger: Mapping[str, Any] | None = None,
    candidate_bindings: tuple[ReviewCandidateBinding, ...] = (),
) -> SegmentPlan:
    preview = preview_chain_mutation(chain_packet, target_segment_id=target_segment_id,
        action="reopen", execution_ledger=execution_ledger, candidate_bindings=candidate_bindings)
    _require_digest(preview.expected_state_digest, expected_state_digest)
    return prepare_segment(source_packet, action="Reroll accepted", chain_packet=chain_packet,
        target_segment_id=target_segment_id, prompts=prompts, seed=seed, frames=frames,
        context_frames=context_frames, reanchor_image=reanchor_image)


def accept_project_segment(
    chain_packet: MMH3Media, candidate_packet: MMH3Media, *, expected_state_digest: str,
    execution_ledger: Mapping[str, Any] | None = None,
    candidate_bindings: tuple[ReviewCandidateBinding, ...] = (),
) -> ProjectSegmentAcceptance:
    """Validate and accept an interactive draft in memory; neither input is mutated.

    The returned packet is not yet published. Call publish_project_segment for a
    serialized, crash-safe disk boundary. This function does not accept F18 jobs.
    """
    state = build_project_review_state(chain_packet, execution_ledger, candidate_bindings=candidate_bindings)
    _require_digest(state.state_digest, expected_state_digest)
    if not state.head_state_matches:
        raise MMH3ResourceError("Authoritative packet does not match its chain head")
    namespace = candidate_packet.manifest.get("extensions", {}).get("mmh3_media", {})
    plan = namespace.get("segment_plan")
    if not isinstance(plan, dict) or plan.get("state") != "draft":
        raise MMH3ResourceError("Project acceptance requires an unaccepted interactive segment draft")
    invalidated = ()
    if plan.get("action") == "reroll":
        invalidated = inspect_chain_reroll(chain_packet, target_segment_id=plan.get("target_segment_id", "")).invalidated_segment_ids
    accepted, _, _ = review_segment(candidate_packet, accept=True, chain_packet=chain_packet)
    accepted_state = build_project_review_state(accepted, execution_ledger, candidate_bindings=candidate_bindings)
    head = next(segment for segment in accepted_state.segments if segment.segment_id == accepted_state.head_segment_id)
    return ProjectSegmentAcceptance(accepted, state.state_digest, accepted_state.state_digest,
                                    head.segment_id, head.active_revision, invalidated)


@contextmanager
def _publication_lock(path: Path):
    """Nonblocking OS lock, released by process exit; keep the lock inode stable."""
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ProjectPublicationBusy("Another project publication is in progress") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _managed_directory(parent: Path, name: str) -> Path:
    path = parent / name
    if path.is_symlink() or path.resolve() != path:
        raise MMH3ResourceError("Project history directory escapes its managed parent")
    path.mkdir(exist_ok=True)
    return path


def _archive_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".mmh3" or not path.is_file():
        raise MMH3ResourceError("Project publication requires existing .mmh3 archive paths")
    return path


def project_history_directory(target: Path) -> Path:
    """One namespace for publication locks and immutable predecessor snapshots."""
    root = _managed_directory(target.parent, ".mmh3_review")
    token = hashlib.sha256(os.path.normcase(str(target)).encode("utf-8")).hexdigest()[:24]
    return _managed_directory(root, token)


@contextmanager
def project_archive_lock(target: Path):
    lock_path = project_history_directory(target) / "publication.lock"
    if lock_path.is_symlink():
        raise MMH3ResourceError("Project lock must not be a symlink")
    with _publication_lock(lock_path):
        yield


def publish_project_segment(
    authoritative_path: str | Path, candidate_path: str | Path, *,
    expected_state_digest: str, expected_candidate_sha256: str, operation_id: str,
) -> ProjectPublication:
    """Publish an interactive draft to a dedicated authoritative archive.

    Uses a process-shared OS lock, re-reads authoritative state, preserves a complete
    previous snapshot, stages and validates the replacement, then atomically switches
    the authoritative .mmh3. A receipt inside that same archive makes retries safe.
    No separate project manifest or F18 ledger write is introduced. All writers of
    this dedicated path must use this service (ordinary Save must use other paths).
    expected_candidate_sha256 must be captured when the candidate is reviewed, so
    replacing that file cannot silently substitute an unreviewed generation.
    """
    if not isinstance(operation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", operation_id):
        raise MMH3ResourceError("operation_id must be a portable identifier of at most 120 characters")
    target, source = _archive_path(authoritative_path), _archive_path(candidate_path)
    if target == source or os.path.samefile(target, source):
        raise MMH3ResourceError("Candidate and authoritative archive must be separate files")
    root = project_history_directory(target)
    with project_archive_lock(target):
        current_sha, _ = sha256_file(target)
        current = load_archive(target, verify="full")
        if sha256_file(target)[0] != current_sha:
            raise ProjectStateConflict("Authoritative archive changed while reading its state")
        report = validate_chain(current, require_head_state=True)
        if not report.ready:
            raise MMH3ResourceError("Invalid authoritative chain: " + "; ".join(report.reasons))
        candidate_sha, _ = sha256_file(source)
        _require_digest(candidate_sha, expected_candidate_sha256, "Candidate artifact")
        request = {"expected_state_digest": expected_state_digest, "candidate_sha256": candidate_sha}
        namespace = current.manifest.get("extensions", {}).get("mmh3_media", {})
        receipts = deep_copy_json(namespace.get("project_publications", []))
        if not isinstance(receipts, list) or any(not isinstance(item, dict) for item in receipts):
            raise MMH3ResourceError("Malformed project publication receipts")
        receipt_ids = [item.get("operation_id") for item in receipts]
        if (any(not isinstance(value, str) or not value for value in receipt_ids)
                or len(set(receipt_ids)) != len(receipt_ids)
                or any(not isinstance(item.get("previous_snapshot"), str)
                       or not re.fullmatch(r"[0-9a-f]{64}\.mmh3", item["previous_snapshot"])
                       for item in receipts)):
            raise MMH3ResourceError("Malformed or duplicate project publication receipts")
        previous = next((item for item in receipts if item.get("operation_id") == operation_id), None)
        if previous is not None:
            if previous.get("request") != request:
                raise ProjectStateConflict("operation_id was already used for another publication request")
            return ProjectPublication(current, str(target), str(root / previous["previous_snapshot"]),
                operation_id, build_project_review_state(current).state_digest, True)
        current_state = build_project_review_state(current)
        _require_digest(current_state.state_digest, expected_state_digest)
        candidate = load_archive(source, verify="full")
        acceptance = accept_project_segment(current, candidate, expected_state_digest=expected_state_digest)
        managed_candidates = deep_copy_json(namespace.get("project_candidates", []))
        if managed_candidates:
            selected = next((item for item in managed_candidates if item.get("id") == candidate_sha), None)
            if selected is None or selected.get("status") != "review" or not selected.get("selected"):
                raise MMH3ResourceError("Select an active project candidate before accepting it")
            selected.update(status="accepted", selected=False, accepted_segment_id=acceptance.segment_id,
                            accepted_revision=acceptance.revision)
        previous_name = f"{current_sha}.mmh3"
        receipt = {"operation_id": operation_id, "request": request,
                   "previous_snapshot": previous_name, "created_at": utc_now_iso(),
                   "segment_id": acceptance.segment_id, "revision": acceptance.revision,
                   "invalidated_segment_ids": list(acceptance.invalidated_segment_ids)}
        accepted = acceptance.packet.set_extension_value("mmh3_media", "project_publications", [*receipts, receipt])
        if "project_candidates" in namespace:
            accepted = accepted.set_extension_value("mmh3_media", "project_candidates", managed_candidates)
        if "project_segment_sources" in namespace:
            accepted = accepted.set_extension_value("mmh3_media", "project_segment_sources", namespace["project_segment_sources"])
        accepted = accepted.record_operation("project_segment_publication", **receipt)
        previous_path = root / previous_name
        if previous_path.is_symlink():
            raise MMH3ResourceError("Previous snapshot must not be a symlink")
        with tempfile.TemporaryDirectory(prefix="publication_", dir=root) as directory:
            stage = Path(directory)
            # Save a byte-identical predecessor before changing any authoritative state.
            if previous_path.exists():
                if sha256_file(previous_path)[0] != current_sha:
                    raise MMH3ResourceError("Previous snapshot hash mismatch")
            else:
                backup = stage / "previous.mmh3"
                shutil.copyfile(target, backup)
                if sha256_file(backup)[0] != current_sha:
                    raise ProjectStateConflict("Authoritative archive changed while preserving its snapshot")
                load_archive(backup, verify="full")
                _durable_replace(backup, previous_path)
            prepared, _ = save_archive(accepted, stage / "accepted.mmh3")
            prepared = load_archive(prepared.source_archive, verify="full")
            report = validate_chain(prepared, require_head_state=True)
            if not report.ready:
                raise MMH3ResourceError("Prepared publication has invalid chain: " + "; ".join(report.reasons))
            if build_project_review_state(prepared).state_digest != build_project_review_state(accepted).state_digest:
                raise MMH3ResourceError("Prepared publication changed semantic project state")
            if sha256_file(target)[0] != current_sha or sha256_file(source)[0] != candidate_sha:
                raise ProjectStateConflict("An input archive changed during publication; refresh before retrying")
            _durable_replace(Path(prepared.source_archive), target)
        published = load_archive(target, verify="manifest")
        return ProjectPublication(published, str(target), str(previous_path), operation_id,
                                  build_project_review_state(published).state_digest, False)


__all__ = [
    "ChainMutationPreview", "ProjectSegmentAcceptance", "ProjectPublication",
    "ProjectStateConflict", "ProjectPublicationBusy", "preview_chain_mutation",
    "prepare_project_reopen", "accept_project_segment", "publish_project_segment",
]
