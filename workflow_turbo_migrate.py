"""Install a visible Turbo LoRA selection block in shipped generation/refine graphs."""
import json
from pathlib import Path

CONSUMERS = {"MMH3H3SamplingPreset", "MMH3H3RefineLoRAs", "MMH3H3StitchUpscale"}
VALUES = {"mode": "auto", "lora_1": "None", "strength_1": 1.0,
          "lora_2": "None", "strength_2": 1.0, "lora_3": "None", "strength_3": 1.0, "lora_entries_json": ""}


def refresh(data):
    """Refresh the editor without changing IDs, links, order or stored selections."""
    before = json.dumps(data)
    scopes = [data['graph'], *data.get('subgraphs', {}).values()] if 'graph' in data else [data, *data.get('definitions', {}).get('subgraphs', [])]
    for scope in scopes:
        collection = scope.get('nodes', scope)
        nodes = collection.values() if isinstance(collection, dict) else collection
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get('type', node.get('class_type')) != 'MMH3H3TurboLoRAs':
                continue
            if 'class_type' in node:
                node['inputs'].setdefault('lora_entries_json', '')
                node.setdefault('_meta', {})['title'] = 'Load LoRAs'
            elif 'values' in node:
                node['values'].setdefault('lora_entries_json', '')
                node.setdefault('layout', {})['title'] = 'Load LoRAs'
            else:
                node['title'] = 'Load LoRAs'
                if not any(p['name'] == 'lora_entries_json' for p in node['inputs']):
                    node['inputs'].append(dict(name='lora_entries_json', type='STRING', link=None,
                                               widget={'name': 'lora_entries_json'}, shape=7))
                    node['widgets_values'].append('')
    return json.dumps(data) != before


def migrate_recipe(data):
    scope = data["graph"]
    nodes = scope["nodes"]
    consumers = [n for n in nodes.values() if n["type"] in CONSUMERS]
    if not consumers or any(n["type"] == "MMH3H3TurboLoRAs" for n in nodes.values()):
        return refresh(data)
    position = [consumers[0]["layout"]["pos"][0], min(n["layout"]["pos"][1] for n in nodes.values()) - 360]
    identity = max(n["id"] for n in nodes.values()) + 1
    nodes["TurboLoRAs"] = dict(id=identity, type="MMH3H3TurboLoRAs", values=dict(VALUES), bindings={},
                              layout=dict(title="Turbo LoRAs", pos=position, size=[400, 270], order=0, mode=0, flags={}, properties={}))
    for n in consumers:
        n["bindings"]["turbo_loras_json"] = ["TurboLoRAs", "turbo_loras_json"]
    refresh(data)
    return True


def migrate_ui(data):
    nodes = data["nodes"]
    consumers = [n for n in nodes if n["type"] in CONSUMERS]
    if not consumers or any(n["type"] == "MMH3H3TurboLoRAs" for n in nodes):
        return refresh(data)
    identity = max([data.get("last_node_id", 0)] + [n["id"] for n in nodes]) + 1
    link_id = max([data.get("last_link_id", 0)] + [l[0] for l in data["links"]])
    position = [consumers[0]["pos"][0], min(n["pos"][1] for n in nodes) - 360]
    links = []
    for n in consumers:
        field = next((p for p in n["inputs"] if p["name"] == "turbo_loras_json"), None)
        if field is None:
            field = dict(name="turbo_loras_json", type="STRING", link=None, shape=7)
            n["inputs"].append(field)
        link_id += 1
        field["link"] = link_id
        links.append(link_id)
        data["links"].append([link_id, identity, 0, n["id"], n["inputs"].index(field), "STRING"])
    nodes.append(dict(id=identity, type="MMH3H3TurboLoRAs", title="Turbo LoRAs", pos=position,
        size=[400, 270], flags={}, order=0, mode=0,
        inputs=[dict(name=k, type="FLOAT" if k.startswith("strength") else "STRING" if k == 'lora_entries_json' else "COMBO", link=None, widget={"name": k}) for k in VALUES],
        outputs=[dict(name="turbo_loras_json", type="STRING", links=links, slot_index=0)],
        properties={"Node name for S&R": "MMH3H3TurboLoRAs"}, widgets_values=list(VALUES.values())))
    data.update(last_node_id=identity, last_link_id=link_id)
    refresh(data)
    return True


def migrate_api(data):
    consumers = [n for n in data.values() if isinstance(n, dict) and n.get("class_type") in CONSUMERS]
    if not consumers or any(isinstance(n, dict) and n.get("class_type") == "MMH3H3TurboLoRAs" for n in data.values()):
        return refresh(data)
    key = "mmh3_turbo_loras"
    if key in data:
        raise ValueError("Turbo LoRA node ID already used")
    data[key] = dict(class_type="MMH3H3TurboLoRAs", inputs=dict(VALUES), _meta={"title": "Turbo LoRAs"})
    for n in consumers:
        n["inputs"]["turbo_loras_json"] = [key, 0]
    refresh(data)
    return True


def main():
    root = Path(__file__).resolve().parent
    owned = set()
    for path in (root / "workflow_recipes").glob("*.recipe.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        owned.update((root / p).resolve() for p in data["outputs"].values())
        if migrate_recipe(data):
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for directory in ("example_workflows", "automation/workflows", "tests/fixtures/workflows"):
        for path in (root / directory).glob("*.json"):
            if path.resolve() in owned:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if (migrate_ui if "nodes" in data else migrate_api)(data):
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
