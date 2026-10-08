"""One refinement interface, expanded through the canonical F07 graphs."""
import json
from pathlib import Path

from .errors import MMH3ResourceError
from .keyframes import validate_refinement_mask
from .long_video_ui import expand_api


def refinement_template(packet, *, method, scale, checkpoint, clip_name, video_vae_name, audio_vae_name,
                        denoise, steps, audio_policy, tile_width=640, tile_height=384, overlap=64, mask=None):
    if method not in {"full_frame", "tiles", "masked_tiles"}:
        raise MMH3ResourceError("Unknown refinement method")
    if method == "masked_tiles" and mask is None:
        raise MMH3ResourceError("Masked refinement needs a MASK input")
    if method != "masked_tiles" and mask is not None:
        raise MMH3ResourceError("Select masked_tiles to use the connected mask")
    if method == "masked_tiles":
        validate_refinement_mask(mask)
    name = "full" if method == "full_frame" else "tile"
    template = json.loads((Path(__file__).resolve().parents[1] / f"automation/workflows/mmh3_refine_{name}_api.json").read_text(encoding="utf-8"))
    def find(kind):
        found = [(k, n) for k, n in template.items() if n["class_type"] == kind]
        if len(found) != 1:
            raise MMH3ResourceError(f"Refine template requires exactly one {kind}")
        return found[0]
    load_key, _ = find("MMH3Load")
    prepare_key, prepare = find("MMH3H3LatentUpscalePrepare")
    prepare["inputs"].update(geometry_mode="scale", scale=scale)
    _, settings = find("MMH3H3UpscaleSettings")
    settings["inputs"].update(refine_checkpoint=checkpoint, audio_policy=audio_policy)
    _, clip = find("CLIPLoader")
    clip["inputs"]["clip_name"] = clip_name
    for n in template.values():
        if n["class_type"] == "VAELoader":
            n["inputs"]["vae_name"] = audio_vae_name if "audio" in n["inputs"]["vae_name"] else video_vae_name
    _, sampling = find("MMH3H3UpscaleRefineSampling")
    sampling["inputs"].update(denoise_override=denoise, steps_override=steps)
    pack_key, pack = find("MMH3PackH3Result")
    if method != "full_frame":
        tile_key, tile = find("MMH3H3NativeTileRefine")
        tile["inputs"].update(tile_width=tile_width, tile_height=tile_height, overlap=overlap)
        if method == "masked_tiles":
            template["MaskRestore"] = {"class_type": "MMH3RefineMaskRestore", "inputs": {
                "source_video": tile["inputs"]["video"], "refined_video": [tile_key, 1],
                "video_vae": tile["inputs"]["video_vae"], "audio_latent": [prepare_key, 2]}}
            # Final delivery uses the exact pixel composite, not another VAE decode.
            for n in template.values():
                if n["class_type"] == "MMH3CreateVideo":
                    n["inputs"]["images"] = ["MaskRestore", 1]
                    n["inputs"]["decode_info_json"] = ""
            pack["inputs"]["latent"] = ["MaskRestore", 0]
            pack["inputs"]["mode"] = "masked_pixel_composite"
    return template, load_key, pack_key


class _Ports:
    def __init__(self, values):
        self.values = values

    def out(self, index):
        return self.values[index]


def expand_refinement(graph, packet, options, mask=None):
    template, source, result = refinement_template(packet, mask=mask, **options)
    template.pop(source)
    if mask is not None:
        template["MaskRestore"]["inputs"]["mask"] = mask
    nodes = expand_api(graph, template, external_nodes={source: _Ports([packet])})
    pack = template[result]
    video_ref = pack["inputs"]["video"]
    return nodes[result].out(0), nodes[video_ref[0]].out(video_ref[1])
