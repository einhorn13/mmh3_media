"""Stable resource aliases compiled through the existing native reference resolver."""
from __future__ import annotations

import hashlib
import json
import re

from .errors import MMH3ResourceError
from .reference_management import configure_reference
from .resolution import resolve_reference_set

ALIAS = re.compile(r"@([A-Za-z_][A-Za-z_0-9]*)")


def parse_schedule_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MMH3ResourceError(f"Duplicate schedule key/alias {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(text, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise MMH3ResourceError(f"Invalid reference schedule JSON: {exc}") from exc


def add_reference_alias(packet, resource_id, alias, scenes):
    current = packet.manifest.get("extensions", {}).get("minimax_h3", {}).get("reference_schedule", {})
    aliases = dict(current.get("aliases", {}))
    if alias in aliases and aliases[alias]["resource_id"] != resource_id:
        raise MMH3ResourceError(f"Alias @{alias} already belongs to another resource")
    if not isinstance(scenes, list) or any(not isinstance(s, str) or not s for s in scenes):
        raise MMH3ResourceError("scenes must be stable scene IDs")
    aliases[alias] = {"resource_id": resource_id, "scenes": scenes}
    scene_ids = list(dict.fromkeys([*current.get("scenes", []), *scenes]))
    return configure_reference_schedule(packet, {"scenes": scene_ids, "aliases": aliases})


def configure_reference_schedule(packet, schedule):
    if not isinstance(schedule, dict) or set(schedule) - {"scenes", "aliases"}:
        raise MMH3ResourceError("Schedule requires scenes and aliases")
    scenes = schedule.get("scenes")
    aliases = schedule.get("aliases")
    if not isinstance(scenes, list) or not scenes or any(not isinstance(s, str) or not s for s in scenes):
        raise MMH3ResourceError("scenes must be nonempty stable scene IDs")
    if len(set(scenes)) != len(scenes) or not isinstance(aliases, dict):
        raise MMH3ResourceError("Duplicate scene ID or invalid aliases")
    normalized = {}
    for name, item in aliases.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or not isinstance(item, dict):
            raise MMH3ResourceError(f"Invalid alias {name!r}")
        if set(item) - {"resource_id", "scenes", "range", "revision"}:
            raise MMH3ResourceError(f"Unknown schedule fields for @{name}")
        resource = packet.get_by_id(item.get("resource_id"))
        if resource is None or resource.get("role") != "reference" or resource.get("kind") not in {"image", "video", "audio"}:
            raise MMH3ResourceError(f"@{name} must name an existing reference resource")
        selected = item.get("scenes", scenes)
        if "range" in item:
            if "scenes" in item or not isinstance(item["range"], list) or len(item["range"]) != 2:
                raise MMH3ResourceError(f"@{name}: use scenes or an inclusive range of scene IDs")
            start, end = item["range"]
            if start not in scenes or end not in scenes or scenes.index(start) > scenes.index(end):
                raise MMH3ResourceError(f"@{name}: invalid scene range")
            selected = scenes[scenes.index(start):scenes.index(end) + 1]
        if not isinstance(selected, list) or any(s not in scenes for s in selected) or len(set(selected)) != len(selected):
            raise MMH3ResourceError(f"@{name}: invalid scene IDs")
        revision = packet.ref(resource["id"]).descriptor.get("content", {}).get("revision")
        if "revision" in item and item["revision"] != revision:
            raise MMH3ResourceError(f"@{name}: resource revision mismatch")
        normalized[name] = {"resource_id": resource["id"], "revision": revision, "scenes": list(selected)}
    out = {"contract": "mmh3_reference_schedule_v1", "scenes": list(scenes), "aliases": normalized}
    return packet.set_extension_value("minimax_h3", "reference_schedule", out)


def compile_scene_references(packet, scene_id, prompt):
    schedule = packet.manifest.get("extensions", {}).get("minimax_h3", {}).get("reference_schedule")
    if schedule is None:
        if ALIAS.search(prompt):
            raise MMH3ResourceError("Named reference used without a reference schedule")
        return packet, prompt, {"active": [], "prompt": prompt}
    if scene_id not in schedule["scenes"]:
        raise MMH3ResourceError(f"Unknown scene ID {scene_id!r}")
    aliases = schedule["aliases"]
    active = {name: item for name, item in aliases.items() if scene_id in item["scenes"]}
    for name in ALIAS.findall(prompt):
        if name not in aliases:
            raise MMH3ResourceError(f"Unknown reference @{name}")
        if name not in active:
            raise MMH3ResourceError(f"Reference @{name} is inactive in scene {scene_id}")
    out = packet
    active_ids = {item["resource_id"] for item in active.values()}
    # Schedule owns inclusion only for its resources; unscheduled native references remain compatible.
    for resource_id in sorted({item["resource_id"] for item in aliases.values()}):
        out = configure_reference(out, resource_id, inclusion="include" if resource_id in active_ids else "exclude").packet
    for name, item in active.items():
        if out.ref(item["resource_id"]).descriptor.get("content", {}).get("revision") != item["revision"]:
            raise MMH3ResourceError(f"Reference @{name} changed revision; reconfigure its schedule")
    resolved = resolve_reference_set(out)
    if not resolved.ready:
        raise MMH3ResourceError("; ".join(d.message for d in resolved.diagnostics if d.severity == "error"))
    tags = {item.resource.resource_id: item.prompt_tag for item in resolved.presentation}
    for name, item in active.items():
        if item["resource_id"] not in tags:
            raise MMH3ResourceError(f"Reference @{name} has no active native label")
    compiled = ALIAS.sub(lambda match: tags[active[match[1]]["resource_id"]], prompt)
    report = {"scene_id": scene_id, "prompt": compiled, "source_prompt": prompt, "schedule": schedule,
              "active": {name: {**item, "prompt_tag": tags[item["resource_id"]]} for name, item in active.items()},
              "reference_order": [item.to_dict() for item in resolved.presentation],
              "resource_content": {item.resource.resource_id: out.ref(item.resource.resource_id).descriptor["content"]
                                   for item in resolved.presentation}}
    report["fingerprint"] = hashlib.sha256(json.dumps(report, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out = out.set_extension_value("minimax_h3", "compiled_references", report)
    out = out.edit_metadata(merge_patch_json={"generation": {"prompt": compiled}})
    return out, compiled, report


def validate_scene_compilation(packet, prompt):
    context = packet.manifest.get("extensions", {}).get("minimax_h3", {})
    if "reference_schedule" not in context:
        return
    compiled = context.get("compiled_references")
    if not compiled:
        raise MMH3ResourceError("Connect MMH3 Scene References before conditioning a scheduled packet")
    _, expected, report = compile_scene_references(packet, compiled["scene_id"], compiled["source_prompt"])
    if report["fingerprint"] != compiled["fingerprint"] or prompt != expected:
        raise MMH3ResourceError("Scheduled references or prompt changed after preview; rerun MMH3 Scene References")
    # Recompilation may restore inclusion; reject a downstream change to the actual active set too.
    current = resolve_reference_set(packet)
    if [item.to_dict() for item in current.presentation] != compiled["reference_order"]:
        raise MMH3ResourceError("Reference inclusion/order changed after scene preview")
