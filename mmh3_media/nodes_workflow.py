"""Compact user-facing nodes built on the established H3 operations."""
from __future__ import annotations

import json

from .node_support import CATEGORY, MMH3, _packet, io, ui, folder_paths
from .errors import MMH3ResourceError
from .keyframes import prepare_keyframe
from .reference_cards import apply_reference_cards
from .generation_contract import validate_generation_model_family


class MMH3H3TaskModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3TaskModel", display_name="H3 Model for Task", category=CATEGORY,
            description="Load only the checkpoint for the packet's task. FL2VA and Ref2VA selections remain explicit; incompatible selections fail before loading.",
            inputs=[MMH3.Input("packet"),
                    io.Combo.Input("fl2va_checkpoint", options=folder_paths.get_filename_list("diffusion_models")),
                    io.Combo.Input("ref2va_checkpoint", options=folder_paths.get_filename_list("diffusion_models"))],
            outputs=[io.Model.Output("model"), io.Combo.Output("task_family"), io.String.Output("checkpoint")], enable_expand=True)

    @classmethod
    def execute(cls, packet, fl2va_checkpoint, ref2va_checkpoint):
        packet = _packet(packet)
        task = packet.manifest.get("generation", {}).get("task", "t2va")
        family = "ref2va" if task == "ref2va" else "fl2va"
        name = ref2va_checkpoint if family == "ref2va" else fl2va_checkpoint
        from .fasth3_v2 import is_fasth3_v2_checkpoint
        fast_v2 = is_fasth3_v2_checkpoint(name)
        if fast_v2 and task != "t2va":
            raise MMH3ResourceError("FastH3 V2 currently supports text-to-audio-video only; choose the base model for image/reference tasks")
        checked = validate_generation_model_family(task, model_family="auto", checkpoint_name=name)
        if not checked.ready or checked.resolved_family == "unknown":
            raise MMH3ResourceError("Selected checkpoint does not match the task: " + str(checked.to_dict()))
        from comfy_execution.graph_utils import GraphBuilder
        graph = GraphBuilder()
        model = (graph.node("MMH3FastH3V2Model", checkpoint=name) if fast_v2
                 else graph.node("UNETLoader", unet_name=name, weight_dtype="default"))
        return io.NodeOutput(model.out(0), family, name, expand=graph.finalize())


class MMH3Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3Generate", display_name="H3 Generate", category=CATEGORY,
            description="Unified text/first-last-frame/reference generation. Only the matching selected task checkpoint is loaded. Advanced sampling and controls remain available in the detailed workflows.",
            inputs=[MMH3.Input("packet"), io.Int.Input("width", default=1344, min=32, max=4096, step=32),
                    io.Int.Input("height", default=768, min=32, max=4096, step=32), io.Int.Input("frames", default=124, min=5, max=362, step=17),
                    io.Combo.Input("fl2va_checkpoint", options=folder_paths.get_filename_list("diffusion_models")),
                    io.Combo.Input("ref2va_checkpoint", options=folder_paths.get_filename_list("diffusion_models")),
                    io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                    io.Combo.Input("video_vae_name", options=folder_paths.get_filename_list("vae")),
                    io.Combo.Input("audio_vae_name", options=folder_paths.get_filename_list("vae")),
                    io.Combo.Input("preset", options=["Standard - 20 steps", "Turbo - 4 steps", "Turbo - 8 steps", "FastH3 V2 - 8 steps"], default="Standard - 20 steps")],
            outputs=[MMH3.Output("packet"), io.Video.Output("video")], enable_expand=True)

    @classmethod
    def execute(cls, packet, width, height, frames, fl2va_checkpoint, ref2va_checkpoint, clip_name, video_vae_name, audio_vae_name,
                preset="Standard - 20 steps"):
        if width <= 0 or height <= 0 or width % 32 or height % 32 or not 5 <= frames <= 362 or (frames - 5) % 17:
            raise MMH3ResourceError("Generation requires a 32-pixel canvas grid and 5+17k frames (up to 362); use Studio for longer sequences")
        from comfy_execution.graph_utils import GraphBuilder
        graph = GraphBuilder()
        packet = _packet(packet)
        from .fasth3_v2 import FASTH3_V2_LABEL, is_fasth3_v2_checkpoint
        fast = is_fasth3_v2_checkpoint(fl2va_checkpoint) and packet.manifest.get("generation", {}).get("task", "t2va") != "ref2va"
        if (preset == FASTH3_V2_LABEL) != fast:
            raise MMH3ResourceError("FastH3 V2 checkpoint and FastH3 V2 preset must be selected together")
        model = graph.node("MMH3H3TaskModel", packet=packet, fl2va_checkpoint=fl2va_checkpoint, ref2va_checkpoint=ref2va_checkpoint)
        clip = graph.node("CLIPLoader", clip_name=clip_name, type="minimax", device="default")
        video_vae = graph.node("VAELoader", vae_name=video_vae_name)
        audio_vae = graph.node("VAELoader", vae_name=audio_vae_name)
        condition = graph.node("MMH3H3AutoCondition", packet=packet, clip=clip.out(0), video_vae=video_vae.out(0),
                               audio_vae=audio_vae.out(0), prompt_override="", seed_override=-1,
                               width_override=width, height_override=height, frames_override=frames)
        sampling = graph.node("MMH3H3SamplingPreset", model=model.out(0), task_family=model.out(1), profile=preset)
        shifted = graph.node("MiniMaxH3SigmaShift", model=sampling.out(6), shift_video=sampling.out(2), shift_audio=sampling.out(3))
        guider = graph.node("BasicGuider", model=shifted.out(0), conditioning=condition.out(0))
        noise = graph.node("RandomNoise", noise_seed=condition.out(2))
        sampler = graph.node("KSamplerSelect", sampler_name=sampling.out(4))
        scheduler = graph.node("MMH3H3Scheduler", model=shifted.out(0), scheduler=sampling.out(5), steps=sampling.out(1), denoise=1.0)
        result = graph.node("SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0), sampler=sampler.out(0),
                            sigmas=scheduler.out(0), latent_image=condition.out(1))
        separated = graph.node("MMH3H3AVSeparate", av_latent=result.out(0))
        decoded = graph.node("MMH3H3VideoDecode", samples=separated.out(0), vae=video_vae.out(0), decode_mode="vae", trt_decoder="auto")
        audio = graph.node("VAEDecodeAudio", samples=separated.out(1), vae=audio_vae.out(0))
        video = graph.node("MMH3CreateVideo", images=decoded.out(0), audio=audio.out(0), fps=24.0, bit_depth=8,
                           color_space="sRGB", decode_info_json=decoded.out(1))
        packed = graph.node("MMH3PackH3Result", packet=condition.out(3), latent=result.out(0), video=video.out(0), audio=audio.out(0),
                            operation="generate", mode=condition.out(4), status=condition.out(5), process_info_json=condition.out(6),
                            sampling_profile_json=sampling.out(0), applied_loras_json=sampling.out(7), latent_origin="sampler_output")
        return io.NodeOutput(packed.out(0), video.out(0), expand=graph.finalize())


class MMH3KeyframePrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3KeyframePrepare", display_name="H3 Prepare Keyframe", category=CATEGORY,
            description="Preview the exact pixel canvas before connecting a first or last frame. crop preserves proportions and trims edges; contain adds borders; stretch is explicit.",
            inputs=[io.Image.Input("image"), io.Int.Input("width", default=1344, min=32, max=4096, step=32),
                    io.Int.Input("height", default=768, min=32, max=4096, step=32),
                    io.Combo.Input("fit", options=["crop", "contain", "stretch"], default="crop"),
                    io.Float.Input("background", default=0.0, min=0.0, max=1.0, step=0.05)],
            outputs=[io.Image.Output("image"), io.String.Output("info_json")])

    @classmethod
    def execute(cls, image, width, height, fit="crop", background=0.0):
        prepared, report = prepare_keyframe(image, width, height, fit, background)
        return io.NodeOutput(prepared, json.dumps(report), ui=ui.PreviewImage(prepared))


class MMH3PacketKeyframes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3PacketKeyframes", display_name="H3 Prepare Packet Keyframes", category=CATEGORY,
            description="Fit first/last-frame resources to the actual generation canvas. No frame inputs is a pass-through; reference images keep their independent geometry.",
            inputs=[MMH3.Input("packet"), io.Int.Input("width", default=1344, min=32, max=4096, step=32),
                    io.Int.Input("height", default=768, min=32, max=4096, step=32),
                    io.Combo.Input("fit", options=["crop", "contain", "stretch"], default="crop"),
                    io.Float.Input("background", default=0.0, min=0.0, max=1.0, step=0.05)],
            outputs=[MMH3.Output("packet"), io.String.Output("info_json")])

    @classmethod
    def execute(cls, packet, width, height, fit="crop", background=0.0):
        from .archive import get_resource_payload
        from .media_metadata import describe_media_payload
        from .resource_model import descriptor_from_media_metadata
        out, reports, previews = _packet(packet), [], []
        for resource in out.manifest["resources"]:
            usage = resource.get("extensions", {}).get("minimax_h3", {}).get("context", {}).get("usage")
            if resource["kind"] != "image" or usage not in {"first_frame", "last_frame"}:
                continue
            image, report = prepare_keyframe(get_resource_payload(out, resource), width, height, fit, background)
            out, _ = out.put_with_id(image, kind="image", role=resource["role"], order=resource.get("order"),
                resource_id=resource["id"], mode="replace", extensions=resource.get("extensions"),
                descriptor=descriptor_from_media_metadata("image", describe_media_payload(image, "image")))
            reports.append({"usage": usage, **report})
            previews.extend(ui.PreviewImage(image[:1]).as_dict()["images"])
        out = out.set_extension_value("minimax_h3", "keyframe_preparation", reports)
        return io.NodeOutput(out, json.dumps(reports), ui={"images": previews})


class MMH3ReferenceCards(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3ReferenceCards", display_name="H3 Reference Cards", category=CATEGORY,
            description="Visual aliases, roles and scene activation. Queue once to discover connected references. Encoder-only affects every reference in this packet, not individual cards.",
            inputs=[MMH3.Input("packet"), io.String.Input("scene_id", default="scene_1"),
                    io.Combo.Input("conditioning", options=["encoder_and_vae", "encoder_only"], default="encoder_and_vae"),
                    io.String.Input("cards_json", default="[]", multiline=True, advanced=True),
                    io.String.Input("prompt", default="", multiline=True, tooltip="Blank inherits the packet prompt; @aliases compile to native media labels.")],
            outputs=[MMH3.Output("packet"), io.String.Output("prompt"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, packet, scene_id="scene_1", conditioning="encoder_and_vae", cards_json="[]", prompt=""):
        out, compiled, report = apply_reference_cards(_packet(packet), cards_json, scene_id, prompt, conditioning)
        # Image previews are saved by ComfyUI; only reference images are thumbnails.
        from .archive import get_resource_payload
        for card in report["cards"]:
            if card["kind"] == "image":
                images = get_resource_payload(out, out.get_by_id(card["resource_id"]))
                from .keyframes import prepare_keyframe
                thumbnail, _ = prepare_keyframe(images[:1], 256, 256, "contain")
                card["preview"] = ui.PreviewImage(thumbnail).as_dict()["images"][0]
        return io.NodeOutput(out, compiled, json.dumps(report, ensure_ascii=False), ui={"mmh3_reference_cards": [report]})


class MMH3H3Refine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3Refine", display_name="H3 Refine", category=CATEGORY,
            description="One F07 interface: full frame, tiles or masked tile composite. Preserves source audio and reuses source LoRAs/sampling. Masked mode composites selected pixels onto the upscaled baseline, not trained inpainting.",
            inputs=[MMH3.Input("packet"), io.Combo.Input("method", options=["full_frame", "tiles", "masked_tiles"], default="full_frame"),
                    io.Float.Input("scale", default=1.5, min=1.0, max=4.0, step=0.05),
                    io.Combo.Input("checkpoint", options=folder_paths.get_filename_list("diffusion_models")),
                    io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                    io.Combo.Input("video_vae_name", options=folder_paths.get_filename_list("vae")),
                    io.Combo.Input("audio_vae_name", options=folder_paths.get_filename_list("vae")),
                    io.Float.Input("denoise", default=0.25, min=0.01, max=1.0, step=0.01),
                    io.Int.Input("steps", default=0, min=0, max=100, tooltip="0 inherits source steps."),
                    io.Combo.Input("audio_policy", options=["source_pcm", "decoded_latent"], default="source_pcm"),
                    io.Int.Input("tile_width", default=640, min=32, max=4096, step=32, advanced=True),
                    io.Int.Input("tile_height", default=384, min=32, max=4096, step=32, advanced=True),
                    io.Int.Input("overlap", default=64, min=0, max=4096, step=32, advanced=True), io.Mask.Input("mask", optional=True)],
            outputs=[MMH3.Output("packet"), io.Video.Output("video")], enable_expand=True)

    @classmethod
    def execute(cls, packet, method, scale, checkpoint, clip_name, video_vae_name, audio_vae_name, denoise, steps,
                audio_policy="source_pcm", tile_width=640, tile_height=384, overlap=64, mask=None):
        from comfy_execution.graph_utils import GraphBuilder
        from .refine_workflow import expand_refinement
        graph = GraphBuilder()
        options = dict(method=method, scale=scale, checkpoint=checkpoint, clip_name=clip_name,
                       video_vae_name=video_vae_name, audio_vae_name=audio_vae_name, denoise=denoise,
                       steps=steps, audio_policy=audio_policy, tile_width=tile_width, tile_height=tile_height, overlap=overlap)
        outputs = expand_refinement(graph, _packet(packet), options, mask)
        return io.NodeOutput(*outputs, expand=graph.finalize())


class MMH3RefineMaskRestore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3RefineMaskRestore", category=CATEGORY + "/Internal",
            inputs=[io.Image.Input("source_video"), io.Image.Input("refined_video"), io.Mask.Input("mask"),
                    io.Vae.Input("video_vae"), io.Latent.Input("audio_latent")],
            outputs=[io.Latent.Output("latent"), io.Image.Output("video")])

    @classmethod
    def execute(cls, source_video, refined_video, mask, video_vae, audio_latent):
        from .keyframes import composite_refinement_mask
        from .h3 import make_nested_tensor
        image = composite_refinement_mask(source_video, refined_video, mask)
        video = video_vae.encode(image)
        return io.NodeOutput({"samples": make_nested_tensor([video, audio_latent["samples"]])}, image)
