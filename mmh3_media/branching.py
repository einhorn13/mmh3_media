"""Independent branch state, preserving native resource payloads and revisions."""
from __future__ import annotations

import uuid

from .chain import _last_process, _packet_state_fingerprint, commit_chain_segment, start_chain, validate_chain
from .core import MMH3Media
from .errors import MMH3ResourceError
from .project_review import build_project_review_state
from .segment_plan import review_segment
from .util import deep_copy_json, utc_now_iso


def build_project_branch(source: MMH3Media, *, source_project_id: str,
                         source_candidate_id: str = "", name: str = "") -> MMH3Media:
    """Adopt one complete source result as segment zero of a new independent chain.

    This is a pure operation. Disk independence is established by saving all archive
    members at the caller's publication boundary; no reference/COW mode is supported.
    A candidate may be unaccepted or stale against its project, but its own saved
    draft/result proof must remain valid. No model execution or latent resizing occurs.
    """
    if source_candidate_id:
        review_segment(source, accept=False)
    elif not validate_chain(source, require_head_state=True).ready:
        raise MMH3ResourceError("Branch source must be an accepted chain head or a complete draft candidate")
    source_state = build_project_review_state(source)
    process = _last_process(source)
    fingerprint = _packet_state_fingerprint(source, process["output_resource_ids"])
    head = next((segment for segment in source_state.segments
                 if segment.segment_id == source_state.head_segment_id), None)
    manifest = deep_copy_json(source.manifest)
    manifest["id"] = "mmh3_" + uuid.uuid4().hex
    manifest["created_at"] = utc_now_iso()
    manifest["name"] = str(name).strip() or source_state.name + " — branch"
    namespace = manifest.setdefault("extensions", {}).setdefault("mmh3_media", {})
    for key in ("chain", "segment_plan", "project_candidates", "project_publications", "project_segment_sources", "project_storage_operations", "branch_request"):
        namespace.pop(key, None)
    namespace["branch"] = {
        "contract": "mmh3_branch_v1", "artifact_policy": "copy",
        "source_project_id": source_project_id, "source_packet_id": source.manifest["id"],
        "source_chain_id": source_state.chain_id, "source_candidate_id": source_candidate_id or None,
        "source_segment_id": head.segment_id if head and not source_candidate_id else None,
        "source_revision": head.active_revision if head and not source_candidate_id else None,
        "source_state_fingerprint": fingerprint, "source_state_digest": source_state.state_digest,
        "created_at": utc_now_iso(),
    }
    branch = MMH3Media(manifest, source.source_archive, dict(source.payloads), dict(source.origins),
                       source.verify_mode, True, dict(source.representation_payloads), dict(source.representation_origins))
    branch = start_chain(branch, name=manifest["name"])
    # The copied clip is an accepted anchor of the new branch, not a newly generated take.
    branch = commit_chain_segment(branch).packet
    return branch.record_operation("project_branch", **namespace["branch"])


__all__ = ["build_project_branch"]
