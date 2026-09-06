from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STOCK_H3_BASELINE = {
    "steps": 20,
    "video_shift": 12.0,
    "audio_shift": 3.0,
    "sampler": "res_multistep",
    "scheduler": "simple",
    "sigma_preset": "scheduler_generated",
}

TURBO_LORA_RECIPES: dict[str, dict[str, Any]] = {
    "minimax\\turbo\\minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors": {
        "task_family": "fl2va",
        "profile": "turbo (4 steps)",
        "strength_model": 1.0,
    },
    "minimax\\minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": {
        "task_family": "ref2va",
        "profile": "turbo (4 steps)",
        "strength_model": 1.0,
    },
    "minimax\\turbo\\minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors": {
        "task_family": "fl2va",
        "profile": "turbo (8 steps)",
        "strength_model": 1.0,
    },
}

LOCAL_FL2V_TURBO = "minimax\\turbo\\minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
DEPRECATED_FL2V_TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


_GROUP_PREFIX = re.compile(r"^(\d{2})\s*·")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_OUTPUT_TYPES = {"SaveVideo", "MMH3Save"}
_RUNTIME_COMBO_TYPES = {
    "CLIPLoader",
    "LoraLoaderModelOnly",
    "UNETLoader",
    "VAELoader",
}
_EXPENSIVE_TYPES = {"SamplerCustomAdvanced", "KSampler", "KSamplerAdvanced"}
_BYPASS_MODES = {2, 4}
_DYNAMIC_MIN_HEIGHT = {"LoadImage": 360, "MMH3Create": 430, "SaveVideo": 420, "MMH3Load": 470, "MMH3Save": 470}
_DEPRECATED_TITLES = {
    "Authoritative .mmh3 Archive",
    "H3 Baseline / Optimization Profile",
}


@dataclass(frozen=True)
class ReviewIssue:
    path: Path
    message: str

    def __str__(self) -> str:
        return f"{self.path.name}: {self.message}"


def _issue(path: Path, scope: str, message: str) -> ReviewIssue:
    prefix = f"{scope}: " if scope else ""
    return ReviewIssue(path, prefix + message)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _node_rect(node: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    pos, size = node.get("pos"), node.get("size")
    if not (isinstance(pos, Sequence) and len(pos) >= 2 and isinstance(size, Sequence) and len(size) >= 2):
        return None
    x, y, w, h = pos[0], pos[1], size[0], size[1]
    if not all(_finite_number(v) for v in (x, y, w, h)) or float(w) <= 0 or float(h) <= 0:
        return None
    return float(x), float(y), float(x) + float(w), float(y) + float(h)


def _overlap_area(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ar, br = _node_rect(a), _node_rect(b)
    if ar is None or br is None:
        return 0.0
    ax1, ay1, ax2, ay2 = ar
    bx1, by1, bx2, by2 = br
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def _inside_group(group: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    bounding = group.get("bounding")
    rect = _node_rect(node)
    if not (isinstance(bounding, Sequence) and len(bounding) >= 4 and rect is not None):
        return False
    gx, gy, gw, gh = bounding[:4]
    if not all(_finite_number(v) for v in (gx, gy, gw, gh)) or float(gw) <= 0 or float(gh) <= 0:
        return False
    x1, y1, x2, y2 = rect
    return x1 >= float(gx) and y1 >= float(gy) and x2 <= float(gx) + float(gw) and y2 <= float(gy) + float(gh)


def _type_tokens(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _types_compatible(a: Any, b: Any) -> bool:
    left, right = _type_tokens(a), _type_tokens(b)
    if not left or not right or "*" in left or "*" in right:
        return True
    return bool(left & right)


def _normalize_link(link: Any) -> tuple[Any, Any, Any, Any, Any, Any] | None:
    if isinstance(link, list) and len(link) >= 6:
        return tuple(link[:6])  # type: ignore[return-value]
    if isinstance(link, dict):
        keys = ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")
        if all(key in link for key in keys):
            return tuple(link[key] for key in keys)  # type: ignore[return-value]
    return None


def _is_active(node: Mapping[str, Any]) -> bool:
    return node.get("mode", 0) not in _BYPASS_MODES


def _graph_structure_issues(
    path: Path,
    graph: Mapping[str, Any],
    *,
    scope: str,
    check_canvas: bool,
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    nodes = graph.get("nodes")
    links = graph.get("links", [])
    if not isinstance(nodes, list):
        return [_issue(path, scope, "graph has no nodes array")]
    if not isinstance(links, list):
        return [_issue(path, scope, "graph has no links array")]

    node_by_id: dict[Any, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            issues.append(_issue(path, scope, f"node[{index}] is not an object"))
            continue
        node_id = node.get("id")
        if node_id is None:
            issues.append(_issue(path, scope, f"node[{index}] has no id"))
        elif node_id in node_by_id:
            issues.append(_issue(path, scope, f"duplicate node ID {node_id}"))
        else:
            node_by_id[node_id] = node
        node_type = node.get("type")
        if not isinstance(node_type, str) or not node_type:
            issues.append(_issue(path, scope, f"node {node_id} has no type"))
        if check_canvas and _node_rect(node) is None:
            issues.append(_issue(path, scope, f"node {node_id} has invalid pos/size geometry"))
        if not isinstance(node.get("inputs", []), list):
            issues.append(_issue(path, scope, f"node {node_id} inputs is not an array"))
        if not isinstance(node.get("outputs", []), list):
            issues.append(_issue(path, scope, f"node {node_id} outputs is not an array"))

    link_by_id: dict[Any, tuple[Any, Any, Any, Any, Any, Any]] = {}
    for raw in links:
        link = _normalize_link(raw)
        if link is None:
            issues.append(_issue(path, scope, f"malformed link: {raw!r}"))
            continue
        link_id, source_id, source_slot, target_id, target_slot, declared_type = link
        if link_id in link_by_id:
            issues.append(_issue(path, scope, f"duplicate link ID {link_id}"))
        link_by_id[link_id] = link
        source, target = node_by_id.get(source_id), node_by_id.get(target_id)
        virtual_source = scope.startswith("subgraph ") and source_id == -10
        virtual_target = scope.startswith("subgraph ") and target_id == -20
        if source is None and not virtual_source:
            issues.append(_issue(path, scope, f"link {link_id} references missing source node {source_id}"))
            continue
        if target is None and not virtual_target:
            issues.append(_issue(path, scope, f"link {link_id} references missing target node {target_id}"))
            continue
        if virtual_source or virtual_target:
            # ComfyUI subgraphs serialize their interface as virtual -10/-20 endpoints.
            # Interface linkIds are checked separately from ordinary node backlinks.
            continue
        source_outputs, target_inputs = source.get("outputs", []), target.get("inputs", [])
        if not isinstance(source_slot, int) or source_slot < 0 or source_slot >= len(source_outputs):
            issues.append(_issue(path, scope, f"link {link_id} has invalid source socket {source_id}.{source_slot}"))
            continue
        if not isinstance(target_slot, int) or target_slot < 0 or target_slot >= len(target_inputs):
            issues.append(_issue(path, scope, f"link {link_id} has invalid target socket {target_id}.{target_slot}"))
            continue
        source_type = source_outputs[source_slot].get("type") if isinstance(source_outputs[source_slot], dict) else None
        target_type = target_inputs[target_slot].get("type") if isinstance(target_inputs[target_slot], dict) else None
        if not _types_compatible(source_type, declared_type):
            issues.append(_issue(path, scope, f"link {link_id} declared type {declared_type!r} disagrees with source {source_type!r}"))
        if not _types_compatible(source_type, target_type):
            issues.append(_issue(path, scope, f"link {link_id} type mismatch {source_type!r} -> {target_type!r}"))

    # Validate both halves of ComfyUI's redundant link serialization.
    for node_id, node in node_by_id.items():
        for index, item in enumerate(node.get("inputs", [])):
            if not isinstance(item, dict):
                issues.append(_issue(path, scope, f"node {node_id} input {index} is not an object"))
                continue
            link_id = item.get("link")
            if link_id is None:
                continue
            link = link_by_id.get(link_id)
            if link is None:
                issues.append(_issue(path, scope, f"node {node_id} input {index} references missing link {link_id}"))
            elif link[3] != node_id or link[4] != index:
                issues.append(_issue(path, scope, f"node {node_id} input {index} disagrees with link {link_id}"))

        for index, item in enumerate(node.get("outputs", [])):
            if not isinstance(item, dict):
                issues.append(_issue(path, scope, f"node {node_id} output {index} is not an object"))
                continue
            output_links = item.get("links")
            if output_links is None:
                continue
            if not isinstance(output_links, list):
                issues.append(_issue(path, scope, f"node {node_id} output {index} links is not an array/null"))
                continue
            if len(output_links) != len(set(output_links)):
                issues.append(_issue(path, scope, f"node {node_id} output {index} contains duplicate link IDs"))
            for link_id in output_links:
                link = link_by_id.get(link_id)
                if link is None:
                    issues.append(_issue(path, scope, f"node {node_id} output {index} references missing link {link_id}"))
                elif link[1] != node_id or link[2] != index:
                    issues.append(_issue(path, scope, f"node {node_id} output {index} disagrees with link {link_id}"))

    if check_canvas:
        live_nodes = [node for node in node_by_id.values() if _is_active(node)]
        for i, node in enumerate(live_nodes):
            for other in live_nodes[i + 1 :]:
                if _overlap_area(node, other) > 0:
                    issues.append(
                        _issue(
                            path,
                            scope,
                            f"node overlap: {node.get('id')}:{node.get('title') or node.get('type')} and "
                            f"{other.get('id')}:{other.get('title') or other.get('type')}",
                        )
                    )

        groups = graph.get("groups", [])
        if not isinstance(groups, list):
            issues.append(_issue(path, scope, "groups is not an array"))
        else:
            numeric_prefixes: list[int] = []
            for group_index, group in enumerate(groups):
                if not isinstance(group, dict):
                    issues.append(_issue(path, scope, f"group[{group_index}] is not an object"))
                    continue
                title = str(group.get("title", ""))
                bounding = group.get("bounding")
                if not (isinstance(bounding, list) and len(bounding) >= 4 and all(_finite_number(v) for v in bounding[:4]) and bounding[2] > 0 and bounding[3] > 0):
                    issues.append(_issue(path, scope, f"group {title!r} has invalid bounding box"))
                    continue
                if not any(_inside_group(group, node) for node in node_by_id.values()):
                    issues.append(_issue(path, scope, f"empty visual group {title!r}"))
                match = _GROUP_PREFIX.match(title)
                if match:
                    numeric_prefixes.append(int(match.group(1)))
            if numeric_prefixes:
                expected = list(range(1, len(numeric_prefixes) + 1))
                if numeric_prefixes != expected:
                    issues.append(_issue(path, scope, f"non-contiguous semantic group order {numeric_prefixes}, expected {expected}"))

    return issues


def _h3_sampling_link_issues(path: Path, workflow: Mapping[str, Any]) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "MMH3H3SamplingPreset":
            continue
        model = next((item for item in node.get("inputs", []) if item.get("name") == "model"), None)
        if not model or model.get("link") is None:
            issues.append(ReviewIssue(path, "H3 Sampling requires a connected base MODEL"))
        for name in ("model", "applied_loras_json"):
            output = next((item for item in node.get("outputs", []) if item.get("name") == name), None)
            if not output or not output.get("links"):
                issues.append(ReviewIssue(path, f"H3 Sampling {name} output is not connected"))
        output = next((item for item in node.get("outputs", []) if item.get("name") == "video_shift"), None)
        if not output or not output.get("links"):
            issues.append(ReviewIssue(path, "H3 Sampling video_shift is not connected to the sampler subgraph"))
    return issues


def _output_reachability_issues(path: Path, workflow: Mapping[str, Any]) -> list[ReviewIssue]:
    nodes = [node for node in workflow.get("nodes", []) if isinstance(node, dict)]
    node_by_id = {node.get("id"): node for node in nodes}
    reverse: dict[Any, set[Any]] = {node_id: set() for node_id in node_by_id}
    for raw in workflow.get("links", []):
        link = _normalize_link(raw)
        if link is not None and link[1] in node_by_id and link[3] in node_by_id:
            reverse[link[3]].add(link[1])

    terminals = {node.get("id") for node in nodes if node.get("type") in _OUTPUT_TYPES and _is_active(node)}
    if not terminals:
        # Only insist on output nodes for workflows that actually contain sampling work/subgraphs.
        has_sampling = any(node.get("type") in _EXPENSIVE_TYPES for node in nodes) or bool(workflow.get("definitions", {}).get("subgraphs", []))
        return [ReviewIssue(path, "sampling workflow has no visible save/output node")] if has_sampling else []

    reachable = set(terminals)
    frontier = list(terminals)
    while frontier:
        current = frontier.pop()
        for upstream in reverse.get(current, ()):
            if upstream not in reachable:
                reachable.add(upstream)
                frontier.append(upstream)

    issues: list[ReviewIssue] = []
    for node in nodes:
        if _is_active(node) and node.get("type") in _EXPENSIVE_TYPES and node.get("id") not in reachable:
            issues.append(ReviewIssue(path, f"active expensive node {node.get('id')}:{node.get('type')} cannot reach SaveVideo/MMH3Save"))
    return issues


def _runtime_input_spec(schema: Mapping[str, Any], input_name: str) -> Any:
    sections = schema.get("input", {})
    if not isinstance(sections, Mapping):
        return None
    for section_name in ("required", "optional"):
        section = sections.get(section_name, {})
        if isinstance(section, Mapping) and input_name in section:
            return section[input_name]
    return None


def _runtime_combo_options(spec: Any) -> tuple[Any, ...]:
    if not isinstance(spec, list) or not spec:
        return ()
    if isinstance(spec[0], list):
        return tuple(spec[0])
    if spec[0] != "COMBO" or len(spec) < 2 or not isinstance(spec[1], Mapping):
        return ()
    options = spec[1].get("options", [])
    return tuple(options) if isinstance(options, list) else ()


def _runtime_schema_issues(
    path: Path,
    workflow: Mapping[str, Any],
    node_schemas: Mapping[str, Any],
) -> list[ReviewIssue]:
    """Validate serialized UI sockets/widgets against a live read-only /object_info snapshot."""
    issues: list[ReviewIssue] = []
    graphs: list[tuple[str, Mapping[str, Any]]] = [("canvas", workflow)]
    definitions = workflow.get("definitions", {})
    if isinstance(definitions, Mapping):
        subgraphs = definitions.get("subgraphs", [])
        if isinstance(subgraphs, list):
            graphs.extend(
                (f"subgraph {graph.get('name') or graph.get('id')}", graph)
                for graph in subgraphs
                if isinstance(graph, Mapping)
            )

    for scope, graph in graphs:
        for node in graph.get("nodes", []):
            if not isinstance(node, Mapping):
                continue
            node_type = node.get("type")
            if not isinstance(node_type, str) or _UUID.fullmatch(node_type):
                continue
            schema = node_schemas.get(node_type)
            if not isinstance(schema, Mapping):
                if node_type.startswith("MMH3"):
                    issues.append(_issue(path, scope, f"runtime /object_info is missing MMH3 node type {node_type}"))
                continue

            node_inputs = node.get("inputs", [])
            if not isinstance(node_inputs, list):
                continue
            actual_names = tuple(
                item.get("name") for item in node_inputs if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            )
            if node_type.startswith("MMH3"):
                order = schema.get("input_order", {})
                expected_names: tuple[str, ...] = ()
                if isinstance(order, Mapping):
                    expected_names = tuple(
                        name
                        for section_name in ("required", "optional")
                        for name in (order.get(section_name, []) if isinstance(order.get(section_name, []), list) else [])
                        if isinstance(name, str) and name
                    )
                if actual_names != expected_names:
                    issues.append(
                        _issue(
                            path,
                            scope,
                            f"node {node.get('id')}:{node_type} input schema drift: "
                            f"stored={actual_names!r}, runtime={expected_names!r}",
                        )
                    )
                    continue

            if not (node_type.startswith("MMH3") or node_type in _RUNTIME_COMBO_TYPES):
                continue
            raw_values = node.get("widgets_values", [])
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            value_index = 0
            for item in node_inputs:
                if not isinstance(item, Mapping) or not isinstance(item.get("widget"), Mapping) or item.get("link") is not None:
                    continue
                if value_index >= len(values):
                    issues.append(_issue(path, scope, f"node {node.get('id')}:{node_type} has no stored value for widget {item.get('name')}"))
                    break
                value = values[value_index]
                value_index += 1
                input_name = item.get("name")
                options = _runtime_combo_options(_runtime_input_spec(schema, str(input_name)))
                if options and value not in options:
                    issues.append(
                        _issue(
                            path,
                            scope,
                            f"node {node.get('id')}:{node_type}.{input_name} stores {value!r}, "
                            "which is absent from runtime options",
                        )
                    )
    return issues


def _ui_issues(
    path: Path,
    workflow: dict[str, Any],
    *,
    node_schemas: Mapping[str, Any] | None = None,
) -> list[ReviewIssue]:
    issues = _graph_structure_issues(path, workflow, scope="canvas", check_canvas=True)
    nodes = [node for node in workflow.get("nodes", []) if isinstance(node, dict)]

    for node in nodes:
        node_type = str(node.get("type", ""))
        values = node.get("widgets_values", [])
        inputs = {item.get("name"): item for item in node.get("inputs", [])}
        if node_type in {"MMH3H3AutoCondition", "MMH3H3ContinuationCondition"}:
            for name in ("width_override", "height_override", "frames_override"):
                if inputs.get(name, {}).get("link") is None:
                    issues.append(ReviewIssue(path, f"node {node.get('id')}: required {name} is not connected"))
        if node_type == "MMH3PackH3Result":
            if len(values) != 10 or values[4] not in {"unknown", "sampler_output", "vae_encoded", "derived"}:
                issues.append(ReviewIssue(path, f"node {node.get('id')}: invalid PackH3Result widget order/latent_origin"))
        if node_type == "MMH3ReferenceConfigure":
            if (len(values) != 6 or values[1] not in {"keep", "include", "exclude"}
                    or not isinstance(values[2], int) or values[4] not in {"keep", "set", "clear"}):
                issues.append(ReviewIssue(path, f"node {node.get('id')}: invalid ReferenceConfigure widget order"))
        if node_type == "MMH3Create":
            if len(values) != 6 or values[3] not in {"fixed", "increment", "decrement", "randomize"}:
                issues.append(ReviewIssue(path, f"node {node.get('id')}: missing seed control widget"))
            if any(name in inputs for name in ("task.first_frame", "task.last_frame")):
                issues.append(ReviewIssue(path, f"node {node.get('id')}: obsolete dynamic keyframe sockets"))
        if node_type == "SaveVideo" and (len(values) != 3 or "format.codec" not in inputs):
            issues.append(ReviewIssue(path, f"node {node.get('id')}: incomplete SaveVideo format/codec widgets"))
        size = node.get("size", [])
        minimum = _DYNAMIC_MIN_HEIGHT.get(node_type)
        if minimum is not None and isinstance(size, Sequence) and len(size) >= 2 and _finite_number(size[1]) and float(size[1]) < minimum:
            issues.append(ReviewIssue(path, f"node {node.get('id')}:{node_type} reserves height {size[1]}, expected at least {minimum} for expanded content"))
        if node.get("title") in _DEPRECATED_TITLES:
            issues.append(ReviewIssue(path, f"node {node.get('id')} uses deprecated user-facing title {node.get('title')!r}"))
        if node_type == "LoraLoaderModelOnly":
            values = node.get("widgets_values", [])
            if isinstance(values, list) and values and values[0] == DEPRECATED_FL2V_TURBO:
                issues.append(ReviewIssue(path, f"node {node.get('id')} uses deprecated non-local Turbo LoRA name; expected {LOCAL_FL2V_TURBO!r}"))

    subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
    if not isinstance(subgraphs, list):
        issues.append(ReviewIssue(path, "definitions.subgraphs is not an array"))
        subgraphs = []
    subgraph_ids: set[Any] = set()
    for index, subgraph in enumerate(subgraphs):
        if not isinstance(subgraph, dict):
            issues.append(ReviewIssue(path, f"subgraph[{index}] is not an object"))
            continue
        subgraph_id = subgraph.get("id")
        if not isinstance(subgraph_id, str) or not _UUID.fullmatch(subgraph_id):
            issues.append(ReviewIssue(path, f"subgraph[{index}] has invalid UUID {subgraph_id!r}"))
        elif subgraph_id in subgraph_ids:
            issues.append(ReviewIssue(path, f"duplicate subgraph definition {subgraph_id}"))
        else:
            subgraph_ids.add(subgraph_id)
        issues.extend(_graph_structure_issues(path, subgraph, scope=f"subgraph {subgraph.get('name') or subgraph_id}", check_canvas=False))
        inner_types = {node.get("type") for node in subgraph.get("nodes", []) if isinstance(node, dict)}
        if {"SamplerCustomAdvanced", "CreateVideo"}.issubset(inner_types) and subgraph.get("name") != "H3 Sampling + AV Decode":
            issues.append(ReviewIssue(path, "sampler/decode subgraph lacks semantic H3 Sampling + AV Decode name"))

    for node in nodes:
        node_type = node.get("type")
        if isinstance(node_type, str) and _UUID.fullmatch(node_type):
            if node_type not in subgraph_ids:
                issues.append(ReviewIssue(path, f"subgraph instance {node.get('id')} references missing definition {node_type}"))
            else:
                definition = next((sg for sg in subgraphs if isinstance(sg, dict) and sg.get("id") == node_type), None)
                if definition:
                    expected_in, expected_out = len(definition.get("inputs", [])), len(definition.get("outputs", []))
                    if len(node.get("inputs", [])) != expected_in or len(node.get("outputs", [])) != expected_out:
                        issues.append(ReviewIssue(path, f"subgraph instance {node.get('id')} socket count drift: {len(node.get('inputs', []))}/{len(node.get('outputs', []))} vs definition {expected_in}/{expected_out}"))

    issues.extend(_h3_sampling_link_issues(path, workflow))
    issues.extend(_output_reachability_issues(path, workflow))
    if node_schemas is not None:
        issues.extend(_runtime_schema_issues(path, workflow, node_schemas))
    return issues


def _api_issues(path: Path, prompt: dict[str, Any]) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    node_ids = set(map(str, prompt))
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            issues.append(ReviewIssue(path, f"API node {node_id} is not an object"))
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or not class_type:
            issues.append(ReviewIssue(path, f"API node {node_id} has no class_type"))
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            issues.append(ReviewIssue(path, f"API node {node_id} has no inputs object"))
            continue
        if node.get("class_type") == "MMH3H3SamplingPreset" and "model" not in inputs:
            issues.append(ReviewIssue(path, f"API node {node_id}: H3 Sampling requires a connected base MODEL"))
        for input_name, value in inputs.items():
            if not (isinstance(value, list) and len(value) == 2):
                continue
            if not isinstance(value[1], int):
                # Two-item literal arrays can exist; only classify as a link when the source resolves.
                continue
            source_id, output_index = str(value[0]), value[1]
            if source_id not in node_ids:
                issues.append(ReviewIssue(path, f"API node {node_id}.{input_name} references missing source {source_id}"))
            if output_index < 0:
                issues.append(ReviewIssue(path, f"API node {node_id}.{input_name} uses negative output index {output_index}"))
            source_type = prompt.get(source_id, {}).get("class_type")
            expected = {
                "MMH3H3SamplingPreset": {"sampling_profile_json": 0, "steps": 1, "shift_video": 2,
                                         "shift_audio": 3, "sampler_name": 4, "scheduler": 5,
                                         "model": 6, "applied_loras_json": 7},
                "MMH3H3GenerationSettings": {"width_override": 1, "height_override": 2, "frames_override": 3,
                                             "generation_settings_json": 4},
            }.get(source_type, {}).get(input_name)
            if expected is not None and output_index != expected:
                issues.append(ReviewIssue(path, f"API node {node_id}.{input_name} uses stale output index {output_index}; expected {expected}"))

    return issues


def review_workflow(
    path: str | Path,
    *,
    node_schemas: Mapping[str, Any] | None = None,
) -> list[ReviewIssue]:
    workflow_path = Path(path)
    try:
        data = json.loads(workflow_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [ReviewIssue(workflow_path, f"cannot load JSON: {exc}")]
    if isinstance(data, dict) and "nodes" in data:
        return _ui_issues(workflow_path, data, node_schemas=node_schemas)
    if isinstance(data, dict):
        return _api_issues(workflow_path, data)
    return [ReviewIssue(workflow_path, "workflow root must be a JSON object")]


def review_directory(
    root: str | Path,
    *,
    node_schemas: Mapping[str, Any] | None = None,
    ui_only: bool = False,
) -> list[ReviewIssue]:
    directory = Path(root)
    issues: list[ReviewIssue] = []
    for path in sorted(directory.glob("mmh3_*.json"), key=lambda item: item.name.casefold()):
        if ui_only and path.name.endswith("_api.json"):
            continue
        issues.extend(review_workflow(path, node_schemas=node_schemas))
    return issues


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static MMH3 UI/API workflow review without starting ComfyUI.")
    parser.add_argument("workflow_root", nargs="?", type=Path, default=Path(__file__).resolve().parent / "example_workflows")
    parser.add_argument("--ui-only", action="store_true", help="Review user-facing UI workflows and skip *_api.json files.")
    parser.add_argument(
        "--object-info-url",
        help="Optionally validate UI sockets and Combo values against a running ComfyUI /object_info endpoint without queuing nodes.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    node_schemas = None
    if args.object_info_url:
        try:
            with urllib.request.urlopen(args.object_info_url, timeout=15) as response:
                node_schemas = json.load(response)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"cannot load runtime object_info: {exc}")
            return 2
        if not isinstance(node_schemas, dict):
            print("runtime object_info root must be an object")
            return 2
    issues = review_directory(args.workflow_root, node_schemas=node_schemas, ui_only=args.ui_only)
    if issues:
        for issue in issues:
            print(issue)
        return 1
    paths = list(args.workflow_root.glob("mmh3_*.json"))
    if args.ui_only:
        paths = [path for path in paths if not path.name.endswith("_api.json")]
    runtime = " runtime_schema=checked" if node_schemas is not None else ""
    print(f"workflow_review=OK workflows={len(paths)}{runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
