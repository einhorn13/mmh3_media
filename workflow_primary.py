"""Generate the six primary entry workflows without replacing advanced examples."""
import argparse
import json
from pathlib import Path

from workflow_generate import compile_recipe
from workflow_schema import local_schemas, expanded_inputs, is_widget, default_value

ROOT = Path(__file__).resolve().parent
MODELS = {"fl2va_checkpoint": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
          "ref2va_checkpoint": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
          "checkpoint": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
          "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
          "video_vae_name": "minimax_h3_video_vae_int8_convrot.safetensors", "audio_vae_name": "minimax_h3_audio_vae_fp32.safetensors"}


def primary_recipe(name, definitions):
    schemas = local_schemas({t for _, t, _, _ in definitions})
    nodes = {}
    for i, (key, kind, overrides, bindings) in enumerate(definitions, 1):
        fields = expanded_inputs(schemas[kind], overrides)
        values = {f["name"]: default_value(f) for f in fields if is_widget(f)}
        values.update({k: v for k, v in MODELS.items() if k in values})
        values.update(overrides)
        nodes[key] = {"id": i, "type": kind, "values": values, "bindings": bindings,
                      "layout": {"pos": [80 + ((i - 1) % 3) * 470, 80 + ((i - 1) // 3) * 650], "size": [420, 530], "properties": {"Node name for S&R": kind}}}
        if "seed" in values:
            nodes[key]["controls"] = {"seed": "fixed"}
    return {"version": 1, "outputs": {"ui": f"example_workflows/primary/{name}.json", "api": f"workflow_recipes/generated/{name}_api.json"},
            "graph": {"layout": {"groups": []}, "nodes": nodes}, "subgraphs": {}}


def generated():
    generate = primary_recipe("01_generate", [
        ("Create", "MMH3Create", {"task": "Video (optional frames)", "prompt": "A cinematic tracking shot. Natural movement and synchronized environmental sound."}, {}),
        ("References", "MMH3ReferenceCards", {}, {"packet": ["Create", "packet"]}),
        ("Settings", "MMH3H3GenerationSettings", {"resolution": "Custom", "resolution.width": 1344, "resolution.height": 768, "duration_seconds": 5.2}, {"packet": ["References", "packet"]}),
        ("Keyframes", "MMH3PacketKeyframes", {}, {"packet": ["Settings", "packet"], "width": ["Settings", "width"], "height": ["Settings", "height"]}),
        ("Generate", "MMH3Generate", {}, {"packet": ["Keyframes", "packet"], "width": ["Settings", "width"], "height": ["Settings", "height"], "frames": ["Settings", "frames"]}),
        ("Video", "MMH3SaveVideo", {"filename_prefix": "video/MMH3_Generate"}, {"video": ["Generate", "video"], "packet": ["Generate", "packet"]}),
        ("Save", "MMH3Save", {"filename_prefix": "mmh3/MMH3_Generate"}, {"packet": ["Video", "packet"]}),
    ])
    refine = primary_recipe("04_refine", [
        ("Load", "MMH3Load", {}, {}), ("Refine", "MMH3H3Refine", {}, {"packet": ["Load", "packet"]}),
        ("Video", "MMH3SaveVideo", {"filename_prefix": "video/MMH3_Refine"}, {"video": ["Refine", "video"], "packet": ["Refine", "packet"]}),
        ("Save", "MMH3Save", {"filename_prefix": "mmh3/MMH3_Refine"}, {"packet": ["Video", "packet"]}),
    ])
    files = {}
    for recipe in (generate, refine):
        ui, _ = compile_recipe(recipe)
        files[recipe["outputs"]["ui"]] = ui
    for target, source in [("02_continue", "mmh3_f04_chain_append"), ("03_edit", "mmh3_independent_av_edit"),
                           ("05_assemble", "mmh3_f18_batch_stitch"), ("06_studio", "mmh3_f18_long_video_studio")]:
        files[f"example_workflows/primary/{target}.json"] = json.loads((ROOT / f"example_workflows/{source}.json").read_text(encoding="utf-8"))
    studio = next(n for n in files["example_workflows/primary/06_studio.json"]["nodes"] if n["type"] == "MMH3LongVideoStudio")
    if not any(i["name"] == "references" for i in studio["inputs"]):
        studio["inputs"].append({"name": "references", "type": "MMH3_MEDIA", "link": None})
    studio["widgets_values"][2] = studio["widgets_values"][2].replace("<Audio 1>", "@master_audio")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = generated()
    for recipe, method in (("mmh3_f07_latent_upscale", "full"), ("mmh3_f07_native_tile_upscale", "tile")):
        _, prompt = compile_recipe(json.loads((ROOT / f"workflow_recipes/{recipe}.recipe.json").read_text(encoding="utf-8")))
        files[f"automation/workflows/mmh3_refine_{method}_api.json"] = {k:n for k,n in prompt.items() if n["class_type"] not in {"MMH3Save", "MMH3SaveVideo"}}
    for name, value in files.items():
        destination = ROOT / name
        text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != text:
                raise SystemExit("Primary workflow drift: " + name)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
