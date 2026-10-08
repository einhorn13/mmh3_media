"""Persistent, sequential ComfyUI orchestration using the existing F18 contracts."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from .automation_execution import (
    acquire_next_execution_job, accept_execution_candidate, commit_execution_artifact,
    commit_execution_candidate, load_execution_ledger, save_execution_ledger,
    transition_execution_job,
)
from .automation_runner import materialize_job_workflow, materialize_assembly_workflow, stamp_submission_workflow, recover_saved_artifact
from .errors import MMH3ResourceError
from .project_actions import _publication_lock

BOOT_ID = uuid.uuid4().hex
ROOT = Path(__file__).resolve().parents[1]


def project_directory(output_root, prefix):
    raw = str(prefix).strip().replace("\\", "/")
    relative = Path(raw)
    if not raw or relative.is_absolute() or any(p in {".", ".."} or ":" in p for p in relative.parts):
        raise MMH3ResourceError("Project must be a relative output directory without '..' or ':'")
    root = Path(output_root).resolve()
    directory = (root / relative).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise MMH3ResourceError("Project directory escapes the output directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def lock_project(directory):
    return _publication_lock(Path(directory) / ".studio.lock")


def store(directory, ledger):
    save_execution_ledger(ledger, Path(directory) / "ledger.json")


def read(directory):
    return load_execution_ledger(Path(directory) / "ledger.json")


def summary(ledger):
    rows = []
    for entry in ledger["jobs"]:
        rows.append(f"{entry['job_id']}: {entry['state']} · takes={len(entry.get('candidates', []))}")
    return f"{ledger['status']}\n" + "\n".join(rows)


def review_action(ledger, action, candidate_id=""):
    reviews = [job for job in ledger["jobs"] if job["state"] == "review"]
    if len(reviews) != 1:
        raise MMH3ResourceError("Accept/reroll requires exactly one shot waiting for review")
    entry = reviews[0]
    if action == "accept":
        candidates = entry.get("candidates", [])
        selected = candidate_id.strip() or (candidates[-1]["candidate_id"] if candidates else "")
        return accept_execution_candidate(ledger, entry["job_id"], selected)
    return transition_execution_job(ledger, entry["job_id"], "reroll")


def acquire(directory, ledger, output_root, prompt_id):
    if any(job["state"] == "running" for job in ledger["jobs"]):
        raise MMH3ResourceError("A shot is already running. After a failed/cancelled run choose recover before retrying.")
    if any(job["state"] == "review" for job in ledger["jobs"]):
        raise MMH3ResourceError("Accept the current take or choose reroll before generating another shot")
    updated, lease = acquire_next_execution_job(ledger)
    if lease is None:
        raise MMH3ResourceError("No shot is ready. Accept the current take or assemble completed shots.")
    prefix = Path(directory).relative_to(Path(output_root).resolve()).as_posix()
    lease["filename_prefix"] = f"{prefix}/shots/{lease['job_id']}/take_{lease['attempt']:03d}"
    receipt = {"contract": "mmh3_ui_submission_v1", "job_id": lease["job_id"], "lease_id": lease["lease_id"]}
    if (Path(directory) / "source.mmh3").is_file():
        from .studio_projects import materialize_scene
        materialize_scene(directory, updated, lease)
    updated["studio"]["submission"] = {"receipt": receipt, "filename_prefix": lease["filename_prefix"], "prompt_id": str(prompt_id), "boot_id": BOOT_ID, "lease": lease}
    store(directory, updated)
    return updated, lease


def recover(directory, ledger, output_root, running_prompt_ids, failed_prompt_ids):
    submission = ledger["studio"].get("submission")
    if not submission or not any(j["state"] == "running" for j in ledger["jobs"]):
        return ledger
    prompt_id = submission["prompt_id"]
    if prompt_id in running_prompt_ids:
        raise MMH3ResourceError("The original prompt is still queued/running; recovery cannot duplicate it")
    packet_path = recover_saved_artifact(output_root, submission)
    lease = submission["lease"]
    if packet_path:
        updated = commit_saved(ledger, lease, packet_path)
    elif prompt_id in failed_prompt_ids or submission.get("boot_id") != BOOT_ID:
        updated = transition_execution_job(ledger, lease["job_id"], "fail", lease_id=lease["lease_id"], error="Interrupted UI prompt; no matching saved artifact")
    else:
        raise MMH3ResourceError("Prompt outcome is unknown. Recovery retained the lease to avoid a duplicate generation.")
    updated["studio"].pop("submission", None)
    store(directory, updated)
    return updated


def commit_saved(ledger, lease, packet_path):
    function = commit_execution_candidate if ledger["effective_settings"].get("review_policy") == "candidate_required" else commit_execution_artifact
    updated = function(ledger, lease["job_id"], lease_id=lease["lease_id"], packet_path=packet_path)
    scene = ledger.get("studio", {}).get("scene_sources", {}).get(lease["job_id"])
    if scene:
        entry = next(e for e in updated["jobs"] if e["job_id"] == lease["job_id"])
        if entry.get("candidates"):
            entry["candidates"][-1].update(scene_revision=scene["revision"], scene_source=scene["path"],
                scene_prompt=scene.get("source_prompt", scene["prompt"]), scene_references=scene.get("references"))
    return updated


def job_workflow(ledger, lease):
    template = json.loads((ROOT / "automation/workflows/mmh3_f18_image_lipsync_api.json").read_text(encoding="utf-8"))
    workflow = materialize_job_workflow(template, ledger, lease)
    settings = ledger["studio"]["render"]
    workflow["1"]["inputs"]["path_override"] = ledger["effective_settings"]["source_archive"]
    for key, input_name, setting in (("6", "clip_name", "clip_name"), ("7", "vae_name", "video_vae_name"), ("8", "vae_name", "audio_vae_name"), ("10", "unet_name", "model_name")):
        workflow[key]["inputs"][input_name] = settings[setting]
    workflow["9"]["inputs"].update(prompt_override=settings["prompt"], width_override=settings["width"], height_override=settings["height"])
    scene = ledger["studio"].get("scene_sources", {}).get(lease["job_id"])
    if scene:
        workflow["1"]["inputs"]["path_override"] = scene["path"]
        if "scene_id" in scene:
            # Master insertion changes native reference numbering. Compile only after it.
            workflow["3"]["inputs"]["mode"] = "add"
            workflow["StudioMasterAlias"] = {"class_type": "MMH3ReferenceAlias", "inputs": {
                "packet": ["3", 0], "resource_id": ["3", 1], "alias": "master_audio",
                "scenes_json": json.dumps([scene["scene_id"]])}}
            workflow["StudioSceneReferences"] = {"class_type": "MMH3SceneReferences", "inputs": {
                "packet": ["StudioMasterAlias", 0], "scene_id": scene["scene_id"], "prompt": scene["source_prompt"]}}
            workflow["9"]["inputs"].update(packet=["StudioSceneReferences", 0], prompt_override=["StudioSceneReferences", 1])
        else:
            workflow["9"]["inputs"]["prompt_override"] = scene["prompt"]
    workflow["15"]["inputs"]["steps"] = settings["steps"]
    return stamp_submission_workflow(workflow, ledger["studio"]["submission"]["receipt"])


def assembly_workflow(ledger, prefix):
    template = json.loads((ROOT / "automation/workflows/mmh3_f18_long_video_lipsync_assembly_api.json").read_text(encoding="utf-8"))
    template["3"]["inputs"]["filename_prefix"] = prefix + "/final_video"
    template["4"]["inputs"]["filename_prefix"] = prefix + "/final"
    return materialize_assembly_workflow(template, ledger)


def expand_api(graph, workflow, external_nodes=None):
    """Convert a flattened API template into real ComfyUI expansion links."""
    built, visiting = dict(external_nodes or {}), set()
    def build(key):
        if key in built:
            return built[key]
        if key in visiting:
            raise MMH3ResourceError("Expansion dependency cycle")
        visiting.add(key)
        definition = workflow[key]
        def resolve(value):
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and (value[0] in workflow or value[0] in built) and type(value[1]) is int:
                return build(value[0]).out(value[1])
            return value
        built[key] = graph.node(definition["class_type"], **{name: resolve(value) for name, value in definition["inputs"].items()})
        visiting.remove(key)
        return built[key]
    for key in workflow:
        build(key)
    return built
