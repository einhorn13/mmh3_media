from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


REMOVE_METADATA_UI = {
    "mmh3_f01_fl2va.json",
    "mmh3_f01_ref2va.json",
    "mmh3_f13_reference_management.json",
}

KEEP_GEOMETRY_PATCH_UI = {"mmh3_f15_preflight.json"}


def _widget_values(node: dict[str, Any]) -> dict[str, Any]:
    values = node.get("widgets_values", [])
    result: dict[str, Any] = {}
    index = 0
    for item in node.get("inputs", []):
        if not isinstance(item, dict) or not isinstance(item.get("widget"), dict) or item.get("link") is not None:
            continue
        if index >= len(values):
            raise ValueError(f"node {node.get('id')}:{node.get('type')} has incomplete widget values")
        result[str(item.get("name"))] = values[index]
        index += 1
    if index != len(values):
        raise ValueError(f"node {node.get('id')}:{node.get('type')} has unmapped widget values")
    return result


def _set_widget(node: dict[str, Any], name: str, value: Any) -> None:
    values = _widget_values(node)
    if name not in values:
        raise ValueError(f"node {node.get('id')}:{node.get('type')} has no widget {name!r}")
    values[name] = value
    node["widgets_values"] = [
        values[str(item.get("name"))]
        for item in node.get("inputs", [])
        if isinstance(item, dict) and isinstance(item.get("widget"), dict) and item.get("link") is None
    ]


def _generation_patch(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    generation = parsed.get("generation", {}) if isinstance(parsed, dict) else {}
    return generation if isinstance(generation, dict) else {}


def _metadata_inputs(packet_link: Any) -> list[dict[str, Any]]:
    return [
        {"name": "packet", "type": "MMH3_MEDIA", "link": packet_link},
        {"name": "name", "type": "STRING", "link": None, "widget": {"name": "name"}},
        {"name": "tags_action", "type": "COMBO", "link": None, "widget": {"name": "tags_action"}},
        {"name": "tags", "type": "STRING", "link": None, "widget": {"name": "tags"}},
        {"name": "notes_action", "type": "COMBO", "link": None, "widget": {"name": "notes_action"}},
        {"name": "notes", "type": "STRING", "link": None, "widget": {"name": "notes"}},
        {
            "name": "custom_json_merge_patch",
            "type": "STRING",
            "link": None,
            "widget": {"name": "custom_json_merge_patch"},
        },
    ]


def _replace_output_source(workflow: dict[str, Any], node: dict[str, Any]) -> None:
    packet_input = next(item for item in node.get("inputs", []) if item.get("name") == "packet")
    input_link_id = packet_input.get("link")
    links = workflow.get("links", [])
    input_link = next(link for link in links if link[0] == input_link_id)
    source_id, source_slot = input_link[1], input_link[2]
    output_link_ids = set((node.get("outputs") or [{}])[0].get("links") or [])
    source_node = next(item for item in workflow.get("nodes", []) if item.get("id") == source_id)
    source_output = source_node.get("outputs", [])[source_slot]
    source_links = source_output.setdefault("links", [])
    if source_links is None:
        source_links = []
        source_output["links"] = source_links
    if input_link_id in source_links:
        source_links.remove(input_link_id)
    for link in links:
        if link[0] in output_link_ids:
            link[1], link[2] = source_id, source_slot
            if link[0] not in source_links:
                source_links.append(link[0])
    workflow["links"] = [link for link in links if link[0] != input_link_id]
    workflow["nodes"] = [item for item in workflow.get("nodes", []) if item is not node]


def _migrate_create_widgets(workflow: dict[str, Any]) -> bool:
    changed = False
    for node in workflow.get("nodes", []):
        if node.get("type") != "MMH3Create":
            continue
        values = _widget_values(node)
        if values.get("seed_mode") == "not set" and values.get("recorded_seed") != 0:
            values["recorded_seed"] = 0
            node["widgets_values"] = [
                values[str(item.get("name"))]
                for item in node.get("inputs", [])
                if isinstance(item, dict) and isinstance(item.get("widget"), dict) and item.get("link") is None
            ]
            changed = True
    return changed


def _migrate_ui(path: Path, workflow: dict[str, Any]) -> set[str]:
    changes: set[str] = set()
    if _migrate_create_widgets(workflow):
        changes.add("create-seed-default")
    metadata_nodes = [node for node in workflow.get("nodes", []) if node.get("type") == "MMH3Metadata"]
    if not metadata_nodes:
        return changes
    if len(metadata_nodes) != 1:
        raise ValueError(f"{path.name}: expected at most one MMH3Metadata node")
    metadata = metadata_nodes[0]
    metadata_names = tuple(str(item.get("name")) for item in metadata.get("inputs", []))
    if metadata_names == (
        "packet",
        "name",
        "tags_action",
        "tags",
        "notes_action",
        "notes",
        "custom_json_merge_patch",
    ):
        return changes
    old = _widget_values(metadata)
    generation = _generation_patch(old.get("custom_json_merge_patch"))

    auto_nodes = [node for node in workflow.get("nodes", []) if node.get("type") == "MMH3H3AutoCondition"]
    if old.get("prompt"):
        if len(auto_nodes) != 1:
            raise ValueError(f"{path.name}: cannot move prompt without one MMH3H3AutoCondition")
        _set_widget(auto_nodes[0], "prompt_override", old["prompt"])
        changes.add("prompt-to-conditioning")
    if old.get("seed_action") == "set":
        if len(auto_nodes) != 1:
            raise ValueError(f"{path.name}: cannot move seed without one MMH3H3AutoCondition")
        _set_widget(auto_nodes[0], "seed_override", int(old["seed"]))
        changes.add("seed-to-conditioning")

    settings_nodes = [node for node in workflow.get("nodes", []) if node.get("type") == "MMH3H3GenerationSettings"]
    frames = generation.get("frames")
    if frames is not None and settings_nodes:
        if len(settings_nodes) != 1:
            raise ValueError(f"{path.name}: expected one MMH3H3GenerationSettings")
        _set_widget(settings_nodes[0], "timeline_mode", "custom")
        _set_widget(settings_nodes[0], "custom_frames", int(frames))
        changes.add("timeline-to-settings")

    if path.name == "mmh3_f13_reference_management.json":
        report = next(node for node in workflow["nodes"] if node.get("type") == "MMH3ReferenceReport")
        for key, widget in (("width", "target_width_override"), ("height", "target_height_override"), ("frames", "target_frames_override")):
            _set_widget(report, widget, generation[key])
        changes.add("geometry-to-report")

    if path.name in REMOVE_METADATA_UI:
        _replace_output_source(workflow, metadata)
        changes.add("remove-metadata")
    else:
        tags = str(old.get("tags") or "")
        keep_patch = old.get("custom_json_merge_patch", "") if path.name in KEEP_GEOMETRY_PATCH_UI else ""
        packet_link = next(item for item in metadata["inputs"] if item.get("name") == "packet").get("link")
        metadata["inputs"] = _metadata_inputs(packet_link)
        metadata["outputs"] = [{"name": "packet", "type": "MMH3_MEDIA", "links": (metadata.get("outputs") or [{}])[0].get("links")}]
        metadata["widgets_values"] = [
            str(old.get("name") or ""),
            "set" if tags else "keep",
            tags,
            str(old.get("notes_action") or "keep"),
            str(old.get("notes") or ""),
            keep_patch,
        ]
        size = metadata.get("size")
        if isinstance(size, list) and len(size) >= 2:
            size[1] = 260
        changes.add("metadata-schema")
    return changes


def _replace_api_refs(workflow: dict[str, Any], old_id: str, replacement: list[Any]) -> None:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == old_id and value[1] == 0:
                inputs[key] = list(replacement)


def _migrate_api(path: Path, workflow: dict[str, Any]) -> set[str]:
    changes: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "MMH3Create":
            inputs = node.get("inputs", {})
            if inputs.get("seed_mode") == "not set" and inputs.get("recorded_seed") != 0:
                inputs["recorded_seed"] = 0
                changes.add("create-seed-default")

    metadata_items = [(key, node) for key, node in workflow.items() if isinstance(node, dict) and node.get("class_type") == "MMH3Metadata"]
    if not metadata_items:
        return changes
    if len(metadata_items) != 1:
        raise ValueError(f"{path.name}: expected one MMH3Metadata")
    node_id, metadata = metadata_items[0]
    old = metadata["inputs"]
    if tuple(old) == (
        "packet",
        "name",
        "tags_action",
        "tags",
        "notes_action",
        "notes",
        "custom_json_merge_patch",
    ):
        return changes
    source = list(old["packet"])
    auto = next((node for node in workflow.values() if isinstance(node, dict) and node.get("class_type") == "MMH3H3AutoCondition"), None)
    if old.get("prompt"):
        if auto is None:
            raise ValueError(f"{path.name}: no conditioning node for prompt")
        auto["inputs"]["prompt_override"] = old["prompt"]
        changes.add("prompt-to-conditioning")
    if old.get("seed_action") == "set":
        if auto is None:
            raise ValueError(f"{path.name}: no conditioning node for seed")
        auto["inputs"]["seed_override"] = int(old["seed"])
        changes.add("seed-to-conditioning")

    if path.name == "mmh3_f02_continuation_api.json":
        metadata["inputs"] = {
            "packet": source,
            "name": old["name"],
            "tags_action": "keep",
            "tags": "",
            "notes_action": old.get("notes_action", "keep"),
            "notes": old.get("notes", ""),
            "custom_json_merge_patch": "",
        }
        changes.add("metadata-schema")
    else:
        _replace_api_refs(workflow, str(node_id), source)
        del workflow[node_id]
        changes.add("remove-metadata")

    if path.name == "mmh3_f13_reference_management_api.json":
        workflow["6"]["inputs"].update(
            target_width_override=256,
            target_height_override=256,
            target_frames_override=39,
        )
    if path.name == "mmh3_f18_long_video_lipsync_api.json":
        workflow["5a"]["inputs"].pop("kind_override", None)
        changes.add("reference-schema")
    return changes


def migrate_directory(root: Path, *, write: bool) -> tuple[int, dict[str, int]]:
    changed_files = 0
    totals: dict[str, int] = {}
    for path in sorted(root.glob("mmh3_*.json"), key=lambda item: item.name.casefold()):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        changes = _migrate_api(path, workflow) if path.name.endswith("_api.json") else _migrate_ui(path, workflow)
        if not changes:
            continue
        changed_files += 1
        for change in changes:
            totals[change] = totals.get(change, 0) + 1
        if write:
            path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed_files, totals


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate MMH3 Metadata consumers to the descriptive-only node contract.")
    parser.add_argument("workflow_root", nargs="?", type=Path, default=Path(__file__).resolve().parent / "example_workflows")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    changed, totals = migrate_directory(args.workflow_root, write=args.write)
    mode = "written" if args.write else "dry-run"
    print(f"metadata_workflow_migration={mode} files={changed} " + " ".join(f"{key}={value}" for key, value in sorted(totals.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
