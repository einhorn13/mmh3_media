"""Filesystem boundary for the Project Manager. IDs/selectors, never client paths."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from .archive import _durable_replace, load_archive, save_archive
from .chain import validate_chain
from .branching import build_project_branch
from .errors import MMH3Error, MMH3ResourceError
from .project_actions import (
    _managed_directory, _require_digest, accept_project_segment, project_archive_lock,
    prepare_project_reopen, preview_chain_mutation, publish_project_segment,
)
from .project_review import build_project_review_state
from .segment_plan import prepare_segment
from .util import deep_copy_json, json_dumps_canonical, sha256_file


class ProjectManager:
    def __init__(self, input_root, output_root):
        self.roots = {"input": Path(input_root).resolve(), "output": Path(output_root).resolve()}

    def resolve(self, selector):
        if not isinstance(selector, str) or "::" not in selector:
            raise MMH3ResourceError("Select an archive from ComfyUI input/output")
        area, relative = selector.split("::", 1)
        root = self.roots.get(area)
        if root is None or not relative or Path(relative).is_absolute() or ":" in relative:
            raise MMH3ResourceError("Invalid archive selector")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".mmh3" or not path.is_file():
            raise MMH3ResourceError("Archive is missing or outside ComfyUI input/output")
        return path

    def selector(self, path):
        path = Path(path).resolve()
        for area, root in self.roots.items():
            if path.is_relative_to(root):
                return area + "::" + path.relative_to(root).as_posix()
        raise MMH3ResourceError("Archive is outside ComfyUI input/output")

    def directory(self, project_id):
        if not isinstance(project_id, str) or not re.fullmatch(r"[0-9a-f]{32}", project_id):
            raise MMH3ResourceError("Invalid project ID")
        base = self.roots["output"] / "mmh3_projects"
        path = base / project_id
        if path.resolve() != path or not (path / "current.mmh3").is_file():
            raise MMH3ResourceError("Project is missing or outside its managed directory")
        return path

    def current(self, project_id):
        path = self.directory(project_id) / "current.mmh3"
        if path.resolve() != path:
            raise MMH3ResourceError("Authoritative archive must remain inside its managed project")
        return path

    def list_projects(self):
        base = self.roots["output"] / "mmh3_projects"
        projects = []
        if not base.is_dir():
            return projects
        for directory in sorted(base.iterdir()):
            if len(projects) >= 200:
                break
            try:
                packet = load_archive(self.current(directory.name), verify="manifest")
                state = build_project_review_state(packet)
                projects.append({"id": directory.name, "name": state.name, "chain_id": state.chain_id})
            except (MMH3Error, OSError):
                continue
        return projects

    def create(self, selector):
        packet = load_archive(self.resolve(selector), verify="full")
        if not validate_chain(packet).ready or not build_project_review_state(packet).head_segment_id:
            raise MMH3ResourceError("Start a project from a saved accepted chain segment")
        base = _managed_directory(self.roots["output"], "mmh3_projects")
        project_id = uuid.uuid4().hex
        directory = _managed_directory(base, project_id)
        # Imported candidates belong to the source workspace, not this new project.
        packet = packet.set_extension_value("mmh3_media", "project_candidates", [])
        packet = packet.set_extension_value("mmh3_media", "project_publications", [])
        packet = packet.set_extension_value("mmh3_media", "project_segment_sources", [])
        save_archive(packet, directory / "current.mmh3")
        return self.state(project_id)

    def _records(self, packet):
        records = packet.manifest.get("extensions", {}).get("mmh3_media", {}).get("project_candidates", [])
        if not isinstance(records, list) or any(not isinstance(item, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("id", ""))) for item in records):
            raise MMH3ResourceError("Malformed project candidates")
        return deep_copy_json(records)

    def candidate(self, project_id, candidate_id):
        if not isinstance(candidate_id, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate_id):
            raise MMH3ResourceError("Invalid candidate ID")
        current = load_archive(self.current(project_id), verify="manifest")
        if not any(item["id"] == candidate_id for item in self._records(current)):
            raise MMH3ResourceError("Candidate does not belong to this project")
        path = self.directory(project_id) / "candidates" / (candidate_id + ".mmh3")
        if not path.exists():
            path = self.directory(project_id) / "trash" / (candidate_id + ".mmh3")
        if path.resolve() != path or not path.is_file():
            raise MMH3ResourceError("Candidate file is missing or unsafe")
        return path

    def state(self, project_id):
        packet = load_archive(self.current(project_id), verify="manifest")
        view = build_project_review_state(packet).to_dict()
        chain = packet.manifest["extensions"]["mmh3_media"]["chain"]
        chain_hash = hashlib.sha256(json_dumps_canonical(chain).encode()).hexdigest()
        records = self._records(packet)
        for item in records:
            item["stale"] = item["ledger_sha256"] != chain_hash and item["status"] != "accepted"
            if item["status"] == "accepted" and not any(s["segment_id"] == item.get("accepted_segment_id")
                    and s["revision"] == item.get("accepted_revision") and s["status"] == "active" for s in chain["segments"]):
                item["stale"] = True
        return {"id": project_id, "state": view, "candidates": records,
                "current_file": self.selector(self.current(project_id))}

    def _edit(self, project_id, expected, edit):
        path = self.current(project_id)
        with project_archive_lock(path):
            packet = load_archive(path, verify="full")
            _require_digest(build_project_review_state(packet).state_digest, expected)
            records = self._records(packet)
            edit(packet, records)
            updated = packet.set_extension_value("mmh3_media", "project_candidates", records)
            save_archive(updated, path)
        return self.state(project_id)

    def add_candidate(self, project_id, selector, expected):
        source = self.resolve(selector)
        def edit(packet, records):
            digest = sha256_file(source)[0]
            if any(item["id"] == digest for item in records) and self.candidate(project_id, digest).parent.name == "trash":
                raise MMH3ResourceError("Restore this candidate from trash instead of importing it again")
            candidate = load_archive(source, verify="full")
            accept_project_segment(packet, candidate, expected_state_digest=expected)
            directory = _managed_directory(self.directory(project_id), "candidates")
            destination = directory / (digest + ".mmh3")
            if destination.is_symlink():
                raise MMH3ResourceError("Unsafe candidate destination")
            if not destination.exists():
                temporary = directory / (uuid.uuid4().hex + ".tmp")
                try:
                    with temporary.open("xb") as output, source.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output, 8 * 1024 * 1024)
                    if sha256_file(temporary)[0] != digest:
                        raise MMH3ResourceError("Candidate changed while copying")
                    _durable_replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            if sha256_file(destination)[0] != digest or sha256_file(source)[0] != digest:
                raise MMH3ResourceError("Candidate changed while importing; add it again")
            if any(item["id"] == digest for item in records):
                return
            plan = candidate.manifest["extensions"]["mmh3_media"]["segment_plan"]
            records.append({"id": digest, "name": source.stem, "status": "review", "selected": False,
                "ledger_sha256": plan["ledger_sha256"], "target_segment_id": plan["target_segment_id"],
                "action": plan["action"], "prompt": plan["prompt"], "seed": plan["seed"]})
        return self._edit(project_id, expected, edit)

    def choose(self, project_id, candidate_id, expected, action):
        if self.candidate(project_id, candidate_id).parent.name == "trash":
            raise MMH3ResourceError("Restore this candidate from trash before reviewing it")
        def edit(packet, records):
            item = next(item for item in records if item["id"] == candidate_id)
            if item["status"] == "accepted":
                raise MMH3ResourceError("Accepted revisions cannot be rejected or switched; reopen the segment")
            if action == "select":
                candidate = load_archive(self.candidate(project_id, candidate_id), verify="manifest")
                accept_project_segment(packet, candidate, expected_state_digest=expected)
                for record in records:
                    record["selected"] = record["id"] == candidate_id
                item["status"] = "review"
            elif action == "reject":
                item.update(status="rejected", selected=False)
            else:
                raise MMH3ResourceError("Unknown review action")
        return self._edit(project_id, expected, edit)

    def accept(self, project_id, candidate_id, expected, operation_id):
        candidate = self.candidate(project_id, candidate_id)
        publish_project_segment(self.current(project_id), candidate,
            expected_state_digest=expected, expected_candidate_sha256=candidate_id, operation_id=operation_id)
        return self.state(project_id)

    def impact(self, project_id, candidate_id, expected):
        packet = load_archive(self.current(project_id), verify="manifest")
        _require_digest(build_project_review_state(packet).state_digest, expected)
        candidate = load_archive(self.candidate(project_id, candidate_id), verify="manifest")
        accept_project_segment(packet, candidate, expected_state_digest=expected)
        plan = candidate.manifest["extensions"]["mmh3_media"]["segment_plan"]
        if plan["action"] == "reroll":
            return preview_chain_mutation(packet, target_segment_id=plan["target_segment_id"]).to_dict()
        return {"action": plan["action"], "affected_segment_ids": [], "invalidated_segment_ids": []}

    def handoff(self, project_id, expected, target_segment_id="", parent_file="", prompts="", seed=0):
        packet = load_archive(self.current(project_id), verify="manifest")
        _require_digest(build_project_review_state(packet).state_digest, expected)
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise MMH3ResourceError("Seed must be an integer between 0 and 4294967295")
        if target_segment_id:
            from .project_timeline import project_timeline
            timeline = project_timeline(self, project_id)
            target = next((s for s in timeline["segments"] if s["segment_id"] == target_segment_id), None)
            parent = next((s for s in timeline["segments"] if target and s["segment_id"] == target["parent_segment_id"]), None)
            if parent and parent["file"]:
                parent_file = parent["file"]
            source = load_archive(self.resolve(parent_file), verify="manifest")
            plan = prepare_project_reopen(packet, source, target_segment_id=target_segment_id,
                expected_state_digest=expected, prompts=prompts, seed=seed)
        else:
            parent_file = self.selector(self.current(project_id))
            plan = prepare_segment(packet, action="Continue", prompts=prompts, seed=seed)
        return {"source_file": parent_file, "chain_file": self.selector(self.current(project_id)),
                "action": "Reroll accepted" if target_segment_id else "Continue",
                "target_segment_id": target_segment_id, "prompts": plan.info["prompts"], "seed": seed}

    def attach_segment(self, project_id, selector, expected):
        with project_archive_lock(self.current(project_id)):
            packet = load_archive(self.current(project_id), verify="full")
            _require_digest(build_project_review_state(packet).state_digest, expected)
            source = self._branch_source(project_id, source_file=selector)
            load_archive(source, verify="full")
            digest = sha256_file(source)[0]
            directory = _managed_directory(self.directory(project_id), "segments")
            destination = directory / (digest + ".mmh3")
            if destination.resolve() != destination:
                raise MMH3ResourceError("Unsafe segment destination")
            if not destination.exists():
                with tempfile.TemporaryDirectory(prefix=".attach_", dir=directory) as temporary:
                    staged = Path(temporary) / "segment.mmh3"
                    shutil.copyfile(source, staged)
                    _require_digest(sha256_file(staged)[0], digest, "Saved segment")
                    _durable_replace(staged, destination)
            _require_digest(sha256_file(destination)[0], digest, "Saved segment")
            refs = packet.manifest["extensions"]["mmh3_media"].get("project_segment_sources", [])
            packet = packet.set_extension_value("mmh3_media", "project_segment_sources", list(dict.fromkeys([*refs, digest])))
            save_archive(packet, self.current(project_id))
        return self.state(project_id)

    def _branch_source(self, project_id, candidate_id="", source_file=""):
        if candidate_id and source_file:
            raise MMH3ResourceError("Choose one branch source")
        if candidate_id:
            path = self.candidate(project_id, candidate_id)
            if path.parent.name == "trash":
                raise MMH3ResourceError("Restore this candidate before creating a branch")
            return path
        if not source_file:
            return self.current(project_id)
        path = self.resolve(source_file)
        source = build_project_review_state(load_archive(path, verify="manifest"))
        current = build_project_review_state(load_archive(self.current(project_id), verify="manifest"))
        if not source.head_state_matches or source.chain_id != current.chain_id:
            raise MMH3ResourceError("Saved branch source does not belong to this chain")
        head = next((segment for segment in source.segments if segment.segment_id == source.head_segment_id), None)
        target = next((segment for segment in current.segments if head and segment.segment_id == head.segment_id), None)
        if target is None:
            raise MMH3ResourceError("Saved branch segment is not recorded in this project")
        record = target.to_dict()["record"]
        versions = [record, *record.get("superseded_revisions", [])]
        if not any(item.get("revision") == head.active_revision
                   and item.get("packet_state_fingerprint") == head.packet_state_fingerprint for item in versions):
            raise MMH3ResourceError("Saved branch revision does not match recorded lineage")
        return path

    def branch_preview(self, project_id, expected, candidate_id="", source_file=""):
        current = load_archive(self.current(project_id), verify="manifest")
        _require_digest(build_project_review_state(current).state_digest, expected)
        path = self._branch_source(project_id, candidate_id, source_file)
        digest, size = sha256_file(path)
        if candidate_id:
            _require_digest(digest, candidate_id, "Candidate artifact")
        source = load_archive(path, verify="manifest")
        # Validate the source without allocating a branch ID or touching payloads.
        if candidate_id:
            from .segment_plan import review_segment
            review_segment(source, accept=False)
        elif not validate_chain(source).ready:
            raise MMH3ResourceError("Invalid accepted branch source")
        return {"artifact_policy": "copy", "source_file": self.selector(path),
                "candidate_id": candidate_id, "source_sha256": digest, "source_archive_bytes": size,
                "expected_state_digest": expected, "source_unchanged": True,
                "description": "Copy this result into an independent branch with a new chain. No source files are changed."}

    def branch(self, project_id, expected, source_sha256, operation_id, candidate_id="", source_file="", name=""):
        if not isinstance(operation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", operation_id):
            raise MMH3ResourceError("Invalid branch operation ID")
        target_id = hashlib.sha256(("branch:" + project_id + ":" + operation_id).encode()).hexdigest()[:32]
        base = _managed_directory(self.roots["output"], "mmh3_projects")
        destination = base / target_id
        request = {"source_project_id": project_id, "expected_state_digest": expected,
                   "source_sha256": source_sha256, "candidate_id": candidate_id,
                   "source_file": source_file, "name": name, "operation_id": operation_id}
        with project_archive_lock(self.current(project_id)):
            if destination.exists():
                packet = load_archive(self.current(target_id), verify="manifest")
                if packet.manifest["extensions"]["mmh3_media"].get("branch_request") != request:
                    from .project_actions import ProjectStateConflict
                    raise ProjectStateConflict("Branch operation ID was already used for a different request")
                return self.state(target_id)
            preview = self.branch_preview(project_id, expected, candidate_id, source_file)
            _require_digest(preview["source_sha256"], source_sha256, "Branch source")
            source_path = self._branch_source(project_id, candidate_id, source_file)
            source = load_archive(source_path, verify="full")
            branch = build_project_branch(source, source_project_id=project_id,
                                          source_candidate_id=candidate_id, name=name)
            branch = branch.set_extension_value("mmh3_media", "branch_request", request)
            with tempfile.TemporaryDirectory(prefix=".branch_", dir=base) as directory:
                staged = Path(directory) / "current.mmh3"
                save_archive(branch, staged)
                if not validate_chain(load_archive(staged, verify="full")).ready:
                    raise MMH3ResourceError("Staged branch failed chain validation")
                _require_digest(sha256_file(source_path)[0], source_sha256, "Branch source")
                # The complete project directory appears in one rename; no half project is listed.
                os.rename(directory, destination)
        return self.state(target_id)
