"""Add independent Sage controls without shifting existing positional widgets."""
import copy
import json
from pathlib import Path

TYPES = {"MMH3H3ModelOptimizations", "MMH3H3StitchUpscale"}


def migrate(data):
    if isinstance(data, list):
        for child in data:
            migrate(child)
        return
    if not isinstance(data, dict):
        return
    kind = data.get("type", data.get("class_type"))
    if kind in TYPES:
        inputs = data.get("values", data.get("inputs", {}))
        if isinstance(inputs, dict):
            if inputs.get("attention") == "SageAttention (KJ)":
                inputs["attention"] = "Default"
                inputs["sage_attention"] = inputs.pop("attention.sage_mode", "auto")
                inputs["sage_allow_compile"] = inputs.pop("attention.sage_allow_compile", False)
            inputs.setdefault("sage_attention", "disabled")
            inputs.setdefault("sage_allow_compile", False)
        elif not any(p["name"] == "sage_attention" for p in inputs):
            # Existing widget sockets give their serialized order, including
            # the selected DynamicCombo's children. No seed controls here.
            fields = [p for p in inputs if "widget" in p]
            values = dict(zip((p["name"] for p in fields), data.get("widgets_values", [])))
            mode, compile = "disabled", False
            if values.get("attention") == "SageAttention (KJ)":
                values["attention"] = "Default"
                mode = values.pop("attention.sage_mode", "auto")
                compile = values.pop("attention.sage_allow_compile", False)
                inputs[:] = [p for p in inputs if not p["name"].startswith("attention.sage_")]
            slot = next(i for i, p in enumerate(inputs) if p["name"] == "fp16_accumulation") + 1
            inputs[slot:slot] = [
                dict(name="sage_attention", type="COMBO", link=None, widget={"name": "sage_attention"}),
                dict(name="sage_allow_compile", type="BOOLEAN", link=None, widget={"name": "sage_allow_compile"}),
            ]
            values.update(sage_attention=mode, sage_allow_compile=compile)
            data["widgets_values"] = [values.get(p["name"], "") for p in inputs if "widget" in p]
            data["size"][1] += 48
    # Repair link target slots by the input carrying each link, both root and
    # subgraph layouts. This also covers removal of old Sage child sockets.
    for child in list(data.values()):
        migrate(child)
    if isinstance(data.get("nodes"), list) and isinstance(data.get("links"), list):
        targets = {p["link"]: (n["id"], i) for n in data["nodes"]
                   for i, p in enumerate(n.get("inputs", [])) if p.get("link") is not None}
        for link in data["links"]:
            key = link["id"] if isinstance(link, dict) else link[0]
            if key in targets:
                node, slot = targets[key]
                if isinstance(link, dict):
                    link.update(target_id=node, target_slot=slot)
                else:
                    link[3:5] = [node, slot]


def main():
    root = Path(__file__).resolve().parent
    for directory in ("workflow_recipes", "example_workflows", "automation/workflows", "tests/fixtures/workflows", "subgraphs"):
        for path in (root / directory).glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            old = copy.deepcopy(data)
            migrate(data)
            if data != old:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
