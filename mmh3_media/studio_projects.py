"""Studio views and revision-checked editing for the shared Project Manager."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from . import long_video_ui as studio
from .errors import MMH3ResourceError
from .project_actions import ProjectStateConflict


def digest(ledger):
    return hashlib.sha256(json.dumps(ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def directory(output_root, project_id):
    if not isinstance(project_id, str) or not project_id.startswith("studio::"):
        raise MMH3ResourceError("Invalid Studio project ID")
    prefix = project_id[8:]
    relative = Path(prefix)
    root = Path(output_root).resolve()
    target = (root / relative).resolve()
    if not prefix or relative.is_absolute() or any(p in {".", ".."} or ":" in p for p in relative.parts) or not target.is_relative_to(root) or target == root:
        raise MMH3ResourceError("Studio project is outside the output directory")
    if not (target / "ledger.json").is_file():
        raise MMH3ResourceError("Studio project does not exist")
    return target


def list_projects(output_root):
    root = Path(output_root).resolve()
    if not root.is_dir():
        return []
    result = []
    for file in root.rglob("ledger.json"):
        if len(result) >= 200:
            break
        try:
            parent = file.parent.resolve()
            if not parent.is_relative_to(root):
                continue
            ledger = studio.read(parent)
            if not isinstance(ledger.get("studio"), dict) or not (parent / "source.mmh3").is_file():
                continue
            prefix = parent.relative_to(root).as_posix()
            result.append({"id": "studio::" + prefix, "name": prefix, "kind": "studio"})
        except (MMH3ResourceError, ValueError, OSError):
            continue
    return result


def scene_settings(ledger, entry):
    index = str(entry["job"]["index"])
    return {"revision": 0, "prompt": ledger["studio"].get("render", {}).get("prompt", ""),
            "references": None, **ledger["studio"].get("scenes", {}).get(index, {})}


def state(output_root, project_id):
    path = directory(output_root, project_id)
    ledger = studio.read(path)
    running = any(e["state"] == "running" for e in ledger["jobs"])
    duration_editable, pending_suffix = {}, True
    for entry in reversed(ledger["jobs"]):
        pending_suffix = pending_suffix and entry["state"] == "pending" and not entry["attempts"] and not entry.get("candidates")
        duration_editable[entry["job_id"]] = bool(pending_suffix and not running)
    scenes = []
    for entry in ledger["jobs"]:
        job = entry["job"]
        settings = scene_settings(ledger, entry)
        scenes.append({"id": entry["job_id"], "index": job["index"], "status": entry["state"],
            "start_seconds": job["write_start_frame"] / 24, "duration_seconds": (job["write_end_frame"] - job["write_start_frame"]) / 24,
            "generation_seconds": job["generation_frames"] / 24, "settings": settings,
            "editable": not running and entry["state"] in {"pending", "failed", "review"},
            "duration_editable": duration_editable[entry["job_id"]],
            "candidates": [{**c, "preview": ledger["studio"].get("previews", {}).get(c["packet_path"])} for c in entry.get("candidates", [])]})
    from .archive import load_archive
    from .reference_cards import reference_slots, reference_card_settings
    source = load_archive(path / "source.mmh3", verify="manifest")
    aliases = source.manifest.get("extensions", {}).get("minimax_h3", {}).get("reference_schedule", {}).get("aliases", {})
    names = {item["resource_id"]: "@" + name for name, item in aliases.items()}
    intent = {c["key"]: c for c in reference_card_settings(source)}
    references = [{"key": key, "kind": r["kind"], "resource_id": r["id"], "name": names.get(r["id"], r.get("name", key)),
        "enabled": intent[key]["enabled"], "scenes": intent[key]["scenes"]} for key, r in reference_slots(source)]
    for scene in scenes:
        if scene["settings"]["references"] is None:
            scene_id = "scene_" + str(scene["index"] + 1)
            scene["settings"]["references"] = [c["key"] for c in reference_card_settings(source, scene_id)
                if c["enabled"] and scene_id in c["scenes"]]
    return {"id": project_id, "name": project_id[8:], "kind": "studio", "state": {"state_digest": digest(ledger)},
            "scenes": scenes, "references": references, "status": ledger["status"], "render": ledger["studio"].get("render", {}),
            "final_path": ledger["studio"].get("final_path", ""), "timeline": {"slots": scenes, "duration_seconds": ledger["plan"]["total_frames"] / 24}}


def _reflow(ledger, entry, duration):
    """Replan only never-rendered suffixes, preserving accepted ownership and proofs."""
    from .automation import _build_h3_audio_try_chunk
    from .automation_execution import create_execution_ledger, _canonical_hash, _validate_ledger
    from .automation_lipsync import _job_settings, _settings_with_hash
    index = ledger["jobs"].index(entry)
    suffix = ledger["jobs"][index:]
    if any(e["state"] != "pending" or e["attempts"] or e.get("candidates") for e in suffix):
        raise MMH3ResourceError("Duration changes require a never-rendered pending suffix; prompts/references can create a new take revision")
    plan = copy.deepcopy(ledger["plan"])
    start = entry["job"]["write_start_frame"]
    samples = plan["settings"]["master_audio_samples"]
    chunks = list(plan["chunks"][:index])
    offset = 0
    while start < plan["total_frames"]:
        if offset == 0:
            seconds = duration
        elif offset < len(suffix):
            job = suffix[offset]["job"]
            seconds = (job["write_end_frame"] - job["write_start_frame"]) / 24
        else:
            seconds = ledger["studio"].get("default_scene_seconds", 8.0)
        chunk = _build_h3_audio_try_chunk(source_id=plan["source_id"], index=len(chunks), total_frames=plan["total_frames"],
            total_audio_samples=samples, audio_sample_rate=32000, write_start_frame=start, shot_duration_seconds=seconds,
            context_seconds=plan["settings"].get("context_seconds", 0.5), min_generation_frames=124, max_generation_frames=362)
        chunks.append(chunk)
        start, offset = chunk["write_end_frame"], offset + 1
    plan["chunks"] = chunks
    settings = copy.deepcopy(ledger["effective_settings"])
    settings["jobs"] = [_job_settings(settings, j) for j in chunks]
    settings = _settings_with_hash(settings)
    rebuilt = create_execution_ledger(plan, operation=ledger["operation"], effective_settings=settings)
    ledger["jobs"] = ledger["jobs"][:index] + rebuilt["jobs"][index:]
    ledger["plan"], ledger["plan_sha256"] = plan, _canonical_hash(plan)
    ledger["effective_settings"], ledger["effective_settings_sha256"] = settings, _canonical_hash(settings)
    valid_indices = {str(e["job"]["index"]) for e in ledger["jobs"]}
    ledger["studio"]["scenes"] = {key: value for key, value in ledger["studio"].get("scenes", {}).items() if key in valid_indices}
    _validate_ledger(ledger)


def edit(output_root, project_id, expected, job_id, prompt, duration, references):
    path = directory(output_root, project_id)
    with studio.lock_project(path):
        ledger = studio.read(path)
        if expected != digest(ledger):
            raise ProjectStateConflict("Studio changed; refresh before saving scene settings")
        if any(e["state"] == "running" for e in ledger["jobs"]):
            raise MMH3ResourceError("Finish the running scene before editing the plan")
        entry = next((e for e in ledger["jobs"] if e["job_id"] == job_id), None)
        if entry is None or entry["state"] not in {"pending", "failed", "review"}:
            raise MMH3ResourceError("Scene is not editable; choose a pending, failed or reviewed scene")
        if not isinstance(prompt, str) or len(prompt) > 100000:
            raise MMH3ResourceError("Scene prompt must be text")
        seconds = float(duration)
        if not math.isfinite(seconds) or not 0 < seconds <= 15.0:
            raise MMH3ResourceError("Scene duration must be within 0–15 seconds")
        from .archive import load_archive
        from .reference_cards import reference_slots
        available = {key for key, _ in reference_slots(load_archive(path / "source.mmh3", verify="manifest"))}
        if references is not None and (not isinstance(references, list) or any(not isinstance(r, str) or r not in available for r in references)
                or len(set(references)) != len(references)):
            raise MMH3ResourceError("Select existing reference cards")
        index = str(entry["job"]["index"])
        current = scene_settings(ledger, entry)
        revision = ledger["revision"]
        actual = (entry["job"]["write_end_frame"] - entry["job"]["write_start_frame"]) / 24
        if abs(seconds - actual) > 1 / 48:
            _reflow(ledger, entry, seconds)
        elif entry["state"] == "review":
            ledger = studio.review_action(ledger, "reroll")
        ledger["studio"].setdefault("scenes", {})[index] = {"prompt": prompt, "references": references, "revision": current["revision"] + 1}
        ledger["studio"].pop("final_path", None)
        ledger["revision"] = revision + 1
        studio.store(path, ledger)
    return state(output_root, project_id)


def materialize_scene(directory, ledger, lease):
    """Save a revision-specific source; master AUDIO ID/revision remains unchanged."""
    from .archive import load_archive, save_archive
    from .reference_cards import reference_card_settings, apply_reference_cards
    entry = next(e for e in ledger["jobs"] if e["job_id"] == lease["job_id"])
    settings = scene_settings(ledger, entry)
    packet = load_archive(Path(directory) / "source.mmh3", verify="on_access")
    selected = settings["references"]
    prompt = settings["prompt"]
    aliases = packet.manifest.get("extensions", {}).get("minimax_h3", {}).get("reference_schedule", {}).get("aliases", {})
    if "master_audio" in aliases:
        raise MMH3ResourceError("The alias @master_audio is reserved for Studio's master slice; rename the library alias")
    scene_id = "scene_" + str(entry["job"]["index"] + 1)
    cards = []
    for intent in reference_card_settings(packet, scene_id):
        enabled = intent["key"] in selected if selected is not None else intent["enabled"] and scene_id in intent["scenes"]
        cards.append({**intent, "enabled": enabled, "scenes": [scene_id]})
    conditioning = packet.manifest.get("extensions", {}).get("minimax_h3", {}).get("reference_conditioning", "encoder_and_vae")
    source_prompt = prompt
    # The master slice is added by the job graph. Compile its alias after that insertion.
    preview_prompt = re.sub(r"@master_audio\b", "<Audio 1>", prompt)
    packet = packet.edit_metadata(merge_patch_json={"generation": {"prompt": preview_prompt}})
    packet, prompt, _ = apply_reference_cards(packet, json.dumps(cards), scene_id, preview_prompt, conditioning)
    packet = packet.edit_metadata(merge_patch_json={"generation": {"prompt": prompt}})
    path = Path(directory) / "scene_sources" / f"{lease['job_id']}_r{settings['revision']}_a{lease['attempt']}_{lease['lease_id']}.mmh3"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MMH3ResourceError("Scene source revision already exists; recover the previous attempt")
    save_archive(packet, path)
    ledger["studio"].setdefault("scene_sources", {})[lease["job_id"]] = {"path": str(path), "prompt": prompt,
        "source_prompt": source_prompt, "scene_id": scene_id, "references": selected, "revision": settings["revision"]}


def review(output_root, project_id, expected, job_id, action, candidate_id=""):
    path = directory(output_root, project_id)
    with studio.lock_project(path):
        ledger = studio.read(path)
        if expected != digest(ledger):
            raise ProjectStateConflict("Studio changed; refresh before reviewing")
        if any(e["state"] == "running" for e in ledger["jobs"]):
            raise MMH3ResourceError("Finish the running scene before reviewing")
        entry = next((e for e in ledger["jobs"] if e["job_id"] == job_id and e["state"] == "review"), None)
        if entry is None or action not in {"accept", "reroll"}:
            raise MMH3ResourceError("Select a scene waiting for review")
        updated = studio.review_action(ledger, action, candidate_id)
        studio.store(path, updated)
    return state(output_root, project_id)


def reference_thumbnail(output_root, project_id, key):
    from io import BytesIO
    from PIL import Image
    from .archive import load_archive, get_resource_payload
    from .reference_cards import reference_slots
    from .keyframes import prepare_keyframe
    packet = load_archive(directory(output_root, project_id) / "source.mmh3", verify="on_access")
    resource = dict(reference_slots(packet)).get(key)
    if resource is None or resource["kind"] != "image":
        raise MMH3ResourceError("No image thumbnail for this reference")
    image, _ = prepare_keyframe(get_resource_payload(packet, resource)[:1], 256, 256, "contain")
    output = BytesIO()
    Image.fromarray((image[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")).save(output, format="PNG")
    return output.getvalue()
