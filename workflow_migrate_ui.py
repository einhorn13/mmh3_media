from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


OLD_FL2V_TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
LOCAL_FL2V_TURBO = "minimax\\turbo\\minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"

CURRENT_INPUT_ORDERS = {
    "MMH3PackH3Result": (
        "packet",
        "operation",
        "mode",
        "status",
        "process_info_json",
        "latent_origin",
        "latent",
        "video",
        "audio",
        "first_frame",
        "last_frame",
        "generation_settings_json",
        "sampling_profile_json",
        "optimization_profile_json",
        "applied_loras_json",
        "control_process_info_json",
    ),
    "MMH3ReferenceConfigure": (
        "packet",
        "resource_id",
        "inclusion",
        "order",
        "purposes_json",
        "binding_action",
        "video_resource_id",
    ),
}

DYNAMIC_MIN_HEIGHT = {
    "MMH3Load": 470,
    "MMH3Save": 470,
    "LoadImage": 360,
    "MMH3Create": 430,
    "MMH3H3SamplingPreset": 260,
    "SaveVideo": 420,
}

TITLE_REPLACEMENTS = {
    "Authoritative .mmh3 Archive": "Save MMH3",
    "Save Reusable .mmh3 File": "Save MMH3",
    "H3 Baseline / Optimization Profile": "Sampling / Optimizations",
}



def _widget_value_map(node: dict[str, Any]) -> dict[str, Any]:
    values = node.get("widgets_values", [])
    if not isinstance(values, list):
        return {}
    # Linked widgets still occupy serialized positions in ComfyUI.
    widgets = [item for item in node.get("inputs", []) if isinstance(item.get("widget"), dict)]
    include_linked = len(values) == len(widgets)
    result: dict[str, Any] = {}
    value_index = 0
    for item in node.get("inputs", []):
        if not isinstance(item, dict) or not isinstance(item.get("widget"), dict) or (not include_linked and item.get("link") is not None):
            continue
        if value_index >= len(values):
            raise ValueError(f"node {node.get('id')}:{node.get('type')} has incomplete widget values")
        result[str(item.get("name"))] = values[value_index]
        value_index += 1
    if value_index != len(values):
        raise ValueError(
            f"node {node.get('id')}:{node.get('type')} has {len(values) - value_index} unmapped widget values"
        )
    return result


def _rebuild_widgets(node: dict[str, Any], values: dict[str, Any]) -> None:
    node["widgets_values"] = [
        values[str(item.get("name"))]
        for item in node.get("inputs", [])
        if isinstance(item, dict)
        and isinstance(item.get("widget"), dict)
        and str(item.get("name")) in values
    ]


def _retarget_links(workflow: dict[str, Any], node_id: Any, old_inputs: list[dict[str, Any]], new_inputs: list[dict[str, Any]]) -> None:
    old_index = {str(item.get("name")): index for index, item in enumerate(old_inputs)}
    new_index = {str(item.get("name")): index for index, item in enumerate(new_inputs)}
    for link in workflow.get("links", []):
        if not isinstance(link, list) or len(link) < 5 or link[3] != node_id:
            continue
        slot = link[4]
        if not isinstance(slot, int) or slot < 0 or slot >= len(old_inputs):
            continue
        name = str(old_inputs[slot].get("name"))
        if name not in new_index:
            if old_inputs[slot].get("link") is not None:
                raise ValueError(f"cannot remove linked input {node_id}.{name}")
            continue
        link[4] = new_index[name]


def _reorder_known_node(workflow: dict[str, Any], node: dict[str, Any]) -> bool:
    expected = CURRENT_INPUT_ORDERS.get(str(node.get("type")))
    if expected is None:
        return False
    old_inputs = node.get("inputs", [])
    if not isinstance(old_inputs, list):
        return False
    actual = tuple(str(item.get("name")) for item in old_inputs if isinstance(item, dict))
    if actual == expected:
        return False
    by_name = {str(item.get("name")): item for item in old_inputs if isinstance(item, dict)}
    missing = [name for name in expected if name not in by_name]
    if missing == ["sampling_profile_json"] and node.get("type") == "MMH3PackH3Result":
        by_name["sampling_profile_json"] = {
            "name": "sampling_profile_json",
            "type": "STRING",
            "widget": {"name": "sampling_profile_json"},
            "link": None,
        }
    elif missing:
        raise ValueError(f"node {node.get('id')}:{node.get('type')} cannot be reordered; missing {missing}")
    if node.get("type") == "MMH3PackH3Result" and len(node.get("widgets_values", [])) == 10:
        values = dict(zip(("operation", "mode", "status", "process_info_json", "latent_origin",
                           "generation_settings_json", "sampling_profile_json", "optimization_profile_json",
                           "applied_loras_json", "control_process_info_json"), node["widgets_values"]))
    elif node.get("type") == "MMH3ReferenceConfigure" and len(node.get("widgets_values", [])) == 6:
        values = dict(zip(expected[1:], node["widgets_values"]))
    else:
        values = _widget_value_map(node)
    values.setdefault("sampling_profile_json", "")
    new_inputs = [by_name[name] for name in expected]
    _retarget_links(workflow, node.get("id"), old_inputs, new_inputs)
    node["inputs"] = new_inputs
    _rebuild_widgets(node, values)
    return True


def _migrate_metadata(node: dict[str, Any]) -> bool:
    inputs = node.get("inputs", [])
    names = [item.get("name") for item in inputs if isinstance(item, dict)] if isinstance(inputs, list) else []
    if "task_action" in names and "seed_action" in names:
        return False
    if "task" not in names or "seed" not in names:
        return False
    values = _widget_value_map(node)
    task = values.get("task", "")
    seed = values.get("seed", -1)
    by_name = {str(item.get("name")): item for item in inputs if isinstance(item, dict)}
    order = (
        "packet", "name", "task_action", "task", "prompt", "seed_action", "seed", "tags",
        "notes_action", "notes", "custom_json_merge_patch",
    )
    by_name["task_action"] = {"name": "task_action", "type": "COMBO", "widget": {"name": "task_action"}, "link": None}
    by_name["seed_action"] = {"name": "seed_action", "type": "COMBO", "widget": {"name": "seed_action"}, "link": None}
    values["task_action"] = "set" if task else "keep"
    values["seed_action"] = "set" if isinstance(seed, int) and seed >= 0 else "keep"
    node["inputs"] = [by_name[name] for name in order]
    _rebuild_widgets(node, values)
    return True


def _normalize_metadata_task(node: dict[str, Any]) -> bool:
    values = _widget_value_map(node)
    task = values.get("task")
    # The value is ignored while task_action=keep, but ComfyUI still validates
    # the serialized Combo value before execution.
    replacement = "t2va" if task in {"", "not specified"} else "fl2va" if task == "masked_video_inpaint" else None
    if replacement is None:
        return False
    values["task"] = replacement
    _rebuild_widgets(node, values)
    return True


def _reserve_dynamic_space(workflow: dict[str, Any]) -> bool:
    nodes = [node for node in workflow.get("nodes", []) if isinstance(node, dict)]
    changed = False
    for node in sorted(nodes, key=lambda item: float(item.get("pos", [0, 0])[1])):
        minimum = DYNAMIC_MIN_HEIGHT.get(str(node.get("type")))
        pos, size = node.get("pos"), node.get("size")
        if minimum is None or not isinstance(pos, list) or not isinstance(size, list) or len(pos) < 2 or len(size) < 2:
            continue
        old_height = float(size[1])
        if node.get("type") in {"MMH3Load", "MMH3Save"} and float(size[0]) < 380:
            size[0] = 380
            changed = True
        if old_height >= minimum:
            continue
        delta = minimum - old_height
        x, y, width = float(pos[0]), float(pos[1]), float(size[0])
        old_bottom = y + old_height
        containing_groups = []
        for group in workflow.get("groups", []):
            box = group.get("bounding") if isinstance(group, dict) else None
            if (
                isinstance(box, list) and len(box) >= 4
                and x >= float(box[0]) and y >= float(box[1])
                and x + width <= float(box[0]) + float(box[2]) and old_bottom <= float(box[1]) + float(box[3])
            ):
                containing_groups.append(group)
        size[1] = minimum
        for other in nodes:
            if other is node:
                continue
            other_pos = other.get("pos")
            if not isinstance(other_pos, list) or len(other_pos) < 2:
                continue
            if abs(float(other_pos[0]) - x) <= 24 and float(other_pos[1]) >= old_bottom:
                other_pos[1] = float(other_pos[1]) + delta
        for group in containing_groups:
            group["bounding"][3] = float(group["bounding"][3]) + delta
        changed = True
    return changed


def _repair_contracts(workflow: dict[str, Any]) -> set[str]:
    changes: set[str] = set()
    nodes = {node["id"]: node for node in workflow.get("nodes", [])}
    links = {link[0]: link for link in workflow.get("links", [])}
    for node in nodes.values():
        before = json.dumps(node, sort_keys=True)
        kind = node.get("type")
        inputs = node.get("inputs", [])
        values = node.get("widgets_values", [])
        if kind == "MMH3Create":
            for item in inputs:
                if item["name"] in {"task.first_frame", "task.last_frame"}:
                    item["name"] = item["name"].removeprefix("task.")
            for name in ("first_frame", "last_frame"):
                if not any(item["name"] == name for item in inputs):
                    inputs.append({"name": name, "type": "IMAGE", "link": None})
            ordered = [item for item in inputs if item["name"] not in {"first_frame", "last_frame"}]
            ordered.extend(next(item for item in inputs if item["name"] == name) for name in ("first_frame", "last_frame"))
            _retarget_links(workflow, node["id"], inputs, ordered)
            node["inputs"] = ordered
            if len(values) == 5:
                values.insert(3, "fixed")
        elif kind in CURRENT_INPUT_ORDERS:
            defaults = ({"operation": "generate", "mode": "", "status": "", "process_info_json": "",
                         "latent_origin": "unknown", "generation_settings_json": "", "sampling_profile_json": "",
                         "optimization_profile_json": "", "applied_loras_json": "", "control_process_info_json": ""}
                        if kind == "MMH3PackH3Result" else
                        {"resource_id": "", "inclusion": "keep", "order": -1, "purposes_json": "[]",
                         "binding_action": "keep", "video_resource_id": ""})
            if len(values) == len(defaults):
                # Runtime widget order is independent of serialized socket order.
                mapped = dict(zip(defaults, values))
            else:
                mapped = _widget_value_map(node)
            for item in inputs:
                if item["name"] in defaults:
                    item["widget"] = {"name": item["name"]}
            defaults.update(mapped)
            node["widgets_values"] = list(defaults.values())
        elif kind == "SaveVideo":
            for item in inputs:
                if item["name"] == "codec":
                    item["name"] = "format.codec"
                if item["name"] in {"format", "format.codec"}:
                    item["widget"] = {"name": item["name"]}
            while len(values) < 3:
                values.append("auto")
        if before != json.dumps(node, sort_keys=True):
            changes.add(f"contract:{kind}")

    def upstream_settings(node: dict[str, Any], visited: set[int]) -> dict[str, Any] | None:
        if node["id"] in visited:
            return None
        visited.add(node["id"])
        if node["type"] == "MMH3H3GenerationSettings":
            return node
        # Follow the packet path, not arbitrary MODEL/media dependencies.
        for item in node.get("inputs", []):
            if item.get("type") == "MMH3_MEDIA" and item.get("link") in links:
                found = upstream_settings(nodes[links[item["link"]][1]], visited)
                if found is not None:
                    return found
        return None

    next_link = max(links, default=0)
    for node in nodes.values():
        if node.get("type") not in {"MMH3H3AutoCondition", "MMH3H3ContinuationCondition"}:
            continue
        if all(item.get("link") is not None for item in node.get("inputs", [])
               if item["name"] in {"width_override", "height_override", "frames_override"}):
            continue
        settings = upstream_settings(node, set())
        if settings is None:
            candidates = [item for item in nodes.values() if item.get("type") == "MMH3H3GenerationSettings"]
            if len(candidates) == 1:
                settings = candidates[0]
        if settings is None:
            raise ValueError(f"no upstream Video Settings for node {node['id']}")
        for slot, item in enumerate(node.get("inputs", [])):
            name = item["name"].removesuffix("_override")
            if name not in {"width", "height", "frames"} or item.get("link") is not None:
                continue
            output_slot = next(i for i, out in enumerate(settings["outputs"]) if out["name"] == name)
            next_link += 1
            item["link"] = next_link
            settings["outputs"][output_slot]["links"] = (settings["outputs"][output_slot].get("links") or []) + [next_link]
            workflow["links"].append([next_link, settings["id"], output_slot, node["id"], slot, "INT"])
            changes.add("generation-links")
    for node in nodes.values():
        if node.get("type") != "MMH3H3ContinuationHandover":
            continue
        candidates = [item for item in nodes.values() if item.get("type") == "MMH3H3GenerationSettings"]
        if len(candidates) != 1:
            raise ValueError("Direct continuation needs one unambiguous Video Settings node")
        settings = candidates[0]
        if not any(item["name"] == "target_width" for item in node["inputs"]):
            old_inputs = settings["inputs"]
            new_inputs = [item for item in old_inputs if not item["name"].startswith("resolution.")]
            _retarget_links(workflow, settings["id"], old_inputs, new_inputs)
            settings["inputs"] = new_inputs
            settings["widgets_values"] = ["Source", settings["widgets_values"][-1]]
        for name, output_slot in (("target_width", 1), ("target_height", 2), ("target_frames", 3)):
            if not any(item["name"] == name for item in node["inputs"]):
                node["inputs"].append({"name": name, "type": "INT", "link": None})
            slot = next(i for i, item in enumerate(node["inputs"]) if item["name"] == name)
            item = node["inputs"][slot]
            if item.get("link") is not None:
                continue
            next_link += 1
            item["link"] = next_link
            settings["outputs"][output_slot]["links"] = (settings["outputs"][output_slot].get("links") or []) + [next_link]
            workflow["links"].append([next_link, settings["id"], output_slot, node["id"], slot, "INT"])
            changes.add("continuation-settings-links")
    if workflow.get("last_link_id", 0) < next_link:
        workflow["last_link_id"] = next_link
        changes.add("link-counter")
    return changes


def migrate_workflow(workflow: dict[str, Any]) -> set[str]:
    changes = _repair_contracts(workflow)
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("type") == "MMH3H3SamplingPreset":
            if not any(i.get("name") == "model" for i in node.get("inputs", [])):
                node.setdefault("inputs", []).append({"name": "model", "type": "MODEL", "link": None})
                changes.add("sampling-model-socket")
            for name, kind in (("model", "MODEL"), ("applied_loras_json", "STRING")):
                if not any(o.get("name") == name for o in node.get("outputs", [])):
                    node.setdefault("outputs", []).append({"name": name, "type": kind, "links": None})
                    changes.add("sampling-model-socket")
        if _reorder_known_node(workflow, node):
            changes.add(f"schema:{node.get('type')}")
        if node.get("type") == "MMH3Metadata" and _migrate_metadata(node):
            changes.add("schema:MMH3Metadata")
        title = node.get("title")
        if title in TITLE_REPLACEMENTS:
            node["title"] = TITLE_REPLACEMENTS[title]
            changes.add("titles")
        if node.get("type") == "LoraLoaderModelOnly":
            values = node.get("widgets_values", [])
            if isinstance(values, list) and values and values[0] == OLD_FL2V_TURBO:
                values[0] = LOCAL_FL2V_TURBO
                changes.add("local-turbo-lora")
        if node.get("type") == "MMH3Metadata" and _normalize_metadata_task(node):
            changes.add("metadata-task")
        if node.get("type") == "MarkdownNote" and isinstance(node.get("widgets_values"), list) and node["widgets_values"]:
            text = str(node["widgets_values"][0])
            if "Default fast path: packaged Comfy-Org FL2V Turbo LoRA" in text:
                node["widgets_values"][0] = (
                    "Default fast path: packaged FL2V Turbo 8-step 768p LoRA at strength 1.0 with the "
                    "**turbo (8 steps)** preset: Euler + simple, video/audio shift 6/3. This adapter is for "
                    "the FL2VA task family (T2VA / I2VA / L2VA / FL2VA and FL2VA-based continuation).\n\n"
                    "**Stock quality/reference fallback:** set the Turbo LoRA node to **Bypass** and select "
                    "**standard (20 steps)** in **H3 Sampling Preset**."
                )
                changes.add("sampling-notes")
            elif "Standard LoRAs belong upstream" in text:
                node["widgets_values"][0] = (
                    "LoRAs remain explicit upstream through a standard ComfyUI loader. The default "
                    "**standard (20 steps)** preset uses res_multistep + simple and video/audio shift 12/3. "
                    "For Ref2VA Turbo use **turbo (4 steps)** with the matching Ref2V 4-step LoRA; do not use "
                    "the FL2V 8-step LoRA on Ref2VA."
                )
                changes.add("sampling-notes")
    if _reserve_dynamic_space(workflow):
        changes.add("dynamic-layout")
    return changes


def unify_f02(workflow: dict[str, Any]) -> set[str]:
    """One continuation graph: optional endpoint plus existing advanced audio controls."""
    if "nodes" not in workflow:
        if any(n.get("class_type") == "MMH3H3ContinuationGuide" for n in workflow.values()):
            return set()
        guide_id = str(max(map(int, workflow)) + 1)
        conditioning = next(k for k, n in workflow.items() if n["class_type"] in {"MMH3H3AutoCondition", "MMH3H3ContinuationCondition"})
        handover = next(k for k, n in workflow.items() if n["class_type"] == "MMH3H3ContinuationHandover")
        vae = workflow[conditioning]["inputs"]["video_vae"]
        for node in workflow.values():
            if node["class_type"] == "BasicGuider":
                node["inputs"]["conditioning"] = [guide_id, 0]
        workflow[guide_id] = {"class_type": "MMH3H3ContinuationGuide", "inputs": {
            "positive": [conditioning, 0], "latent": [handover, 1], "video_vae": vae}}
        return {"unified-f02"}

    nodes = workflow["nodes"]
    if any(n["type"] == "MMH3H3ContinuationGuide" for n in nodes):
        return set()
    by_type = {n["type"]: n for n in nodes}
    condition = by_type.get("MMH3H3ContinuationCondition") or by_type["MMH3H3AutoCondition"]
    handover = by_type["MMH3H3ContinuationHandover"]
    vae_link = next(link for link in workflow["links"] if link[0] == next(
        i["link"] for i in condition["inputs"] if i["name"] == "video_vae"))
    vae_id, vae_slot = vae_link[1:3]
    next_id = max(n["id"] for n in nodes) + 1
    image_id, guide_id, note_id = next_id, next_id + 1, next_id + 2
    image = {"id": image_id, "type": "LoadImage", "title": "End Frame — optional (enable to use)",
             "pos": [90, 1450], "size": [380, 360], "flags": {}, "order": 0, "mode": 2,
             "inputs": [{"name": "image", "type": "COMBO", "link": None, "widget": {"name": "image"}}],
             "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}, {"name": "MASK", "type": "MASK", "links": None}],
             "properties": {"Node name for S&R": "LoadImage"}, "widgets_values": [""]}
    guide = {"id": guide_id, "type": "MMH3H3ContinuationGuide", "pos": [1590, 500],
             "size": [360, 150], "flags": {}, "order": len(nodes), "mode": 0,
             "inputs": [{"name": name, "type": kind, "link": None} for name, kind in
                        (("positive", "CONDITIONING"), ("latent", "LATENT"), ("video_vae", "VAE"), ("last_frame", "IMAGE"))],
             "outputs": [{"name": "positive", "type": "CONDITIONING", "links": []}],
             "properties": {"Node name for S&R": "MMH3H3ContinuationGuide"}, "widgets_values": []}
    note = {"id": note_id, "type": "MarkdownNote", "title": "Continuation controls",
            "pos": [1590, 740], "size": [360, 310], "flags": {}, "order": len(nodes) + 1, "mode": 0,
            "inputs": [], "outputs": [], "properties": {}, "widgets_values": [
                "**Ordinary continuation:** leave End Frame muted.\n\n"
                "**End frame:** select an image and set its LoadImage node Mode to Always (Ctrl+M toggles mute). "
                "Mute it again or disconnect last_frame to return to ordinary continuation.\n\n"
                "**Audio transition:** expand Advanced on Handover. Audio overlap 0 follows video; feather 0 is off. "
                "For the former feather example use video=39, audio=90, feather=6 and a target longer than 90 frames.\n\n"
                "**Duration includes overlap**, not just new frames. Direct continuation keeps Source resolution."
            ]}
    # Keep existing consumer links; interpose the guide only on positive conditioning.
    for link in workflow["links"]:
        if link[1:3] == [condition["id"], 0]:
            condition["outputs"][0]["links"].remove(link[0])
            link[1] = guide_id
            guide["outputs"][0]["links"].append(link[0])
    nodes.extend([image, guide, note])
    lookup = {n["id"]: n for n in nodes}
    for source, slot, target_slot, kind in ((condition["id"], 0, 0, "CONDITIONING"),
            (handover["id"], 1, 1, "LATENT"), (vae_id, vae_slot, 2, "VAE"), (image_id, 0, 3, "IMAGE")):
        link_id = max(link[0] for link in workflow["links"]) + 1
        workflow["links"].append([link_id, source, slot, guide_id, target_slot, kind])
        output = lookup[source]["outputs"][slot]
        output["links"] = (output.get("links") or []) + [link_id]
        guide["inputs"][target_slot]["link"] = link_id
    handover["title"] = "Continuation Handover"
    workflow["last_node_id"] = note_id
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    return {"unified-f02"}


def migrate_directory(root: Path, *, write: bool) -> tuple[int, dict[str, int]]:
    changed_files = 0
    totals: dict[str, int] = {}
    for path in sorted(root.glob("mmh3_*.json"), key=lambda item: item.name.casefold()):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict):
            continue
        changes = unify_f02(workflow) if path.name in {
            "mmh3_f02_continuation.json",
            "mmh3_f02_continuation_api.json",
            "mmh3_f02_ref2va_continuation.json",
            "mmh3_f02_ref2va_continuation_api.json",
        } else set()
        changes |= migrate_workflow(workflow) if "nodes" in workflow else migrate_api(workflow)
        if not changes:
            continue
        changed_files += 1
        for change in changes:
            totals[change] = totals.get(change, 0) + 1
        if write:
            path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed_files, totals


def migrate_api(prompt: dict[str, Any]) -> set[str]:
    changes: set[str] = set()
    settings_ids = [key for key, node in prompt.items() if node.get("class_type") == "MMH3H3GenerationSettings"]
    for node in prompt.values():
        if node.get("class_type") == "MMH3H3ContinuationHandover" and len(settings_ids) == 1:
            for name, slot in (("target_width", 1), ("target_height", 2), ("target_frames", 3)):
                value = [settings_ids[0], slot]
                if node["inputs"].get(name) != value:
                    node["inputs"][name] = value
                    changes.add("api-continuation-settings")
    semantic_outputs = {
        "MMH3H3SamplingPreset": {"sampling_profile_json": 0, "steps": 1, "shift_video": 2,
                                 "shift_audio": 3, "sampler_name": 4, "scheduler": 5},
        "MMH3H3GenerationSettings": {"width_override": 1, "height_override": 2, "frames_override": 3,
                                     "generation_settings_json": 4},
    }
    for node in prompt.values():
        for name, value in list(node.get("inputs", {}).items()):
            if node.get("class_type") == "MMH3Create" and name in {"task.first_frame", "task.last_frame"}:
                node["inputs"][name.removeprefix("task.")] = node["inputs"].pop(name)
                changes.add("api-keyframes")
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
                source = prompt[str(value[0])].get("class_type")
                expected = semantic_outputs.get(source, {}).get(name)
                if expected is not None and value[1] != expected:
                    value[1] = expected
                    changes.add("api-output-indices")
    return changes


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair checked-in MMH3 UI/API workflow contracts.")
    parser.add_argument("workflow_root", nargs="*", type=Path)
    parser.add_argument("--write", action="store_true", help="Write migrations. Without this flag the command is a dry run.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base = Path(__file__).resolve().parent
    roots = tuple(args.workflow_root) if args.workflow_root else (base / "example_workflows", base / "automation" / "workflows", base / "tests" / "fixtures" / "workflows")
    changed = 0
    totals: dict[str, int] = {}
    for root in roots:
        count, partial = migrate_directory(root, write=args.write)
        changed += count
        for key, value in partial.items(): totals[key] = totals.get(key, 0) + value
    mode = "written" if args.write else "dry-run"
    details = " ".join(f"{name}={count}" for name, count in sorted(totals.items()))
    print(f"workflow_migration={mode} files={changed}" + (f" {details}" if details else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
