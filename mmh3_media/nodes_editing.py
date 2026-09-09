from __future__ import annotations

import inspect
import json

from .node_support import CATEGORY, MMH3, _packet, io
from .errors import MMH3ResourceError
from .av_edit import apply_av_edit_policy, restore_av_protection
from .av_bridge import prepare_av_bridge, encode_av_bridge
from .scheduled_references import configure_reference_schedule, compile_scene_references, parse_schedule_json


def require_joint_masks():
    import comfy.samplers
    source = inspect.getsource(comfy.samplers.CFGGuider)
    if "denoise_mask.is_nested" not in source or "pack_latents(denoise_masks)" not in source:
        raise MMH3ResourceError("This ComfyUI sampler does not expose the supported joint AV mask contract")


class MMH3AVEditPolicy(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3AVEditPolicy", display_name="MMH3 AV Edit Policy", category=CATEGORY,
            inputs=[MMH3.Input("packet"), io.Latent.Input("latent"),
                    io.Combo.Input("video_policy", options=["mask", "all", "preserve"], default="mask"),
                    io.Combo.Input("audio_policy", options=["preserve", "all", "intervals", "follow_video"], default="preserve"),
                    io.String.Input("audio_intervals_json", default="[]", multiline=True),
                    io.Mask.Input("video_mask", optional=True)],
            outputs=[MMH3.Output("packet"), io.Latent.Output("latent"), io.String.Output("info_json")])

    @classmethod
    def execute(cls, packet, latent, video_policy, audio_policy, audio_intervals_json, video_mask=None):
        require_joint_masks()
        out, report = apply_av_edit_policy(latent, video_policy=video_policy, audio_policy=audio_policy,
            video_mask=video_mask, audio_intervals=json.loads(audio_intervals_json))
        packet = _packet(packet).set_extension_value("minimax_h3", "av_edit_policy", report)
        return io.NodeOutput(packet, out, json.dumps(report, ensure_ascii=False, indent=2))


class MMH3AVProtectionRestore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3AVProtectionRestore", display_name="MMH3 AV Protection Restore", category=CATEGORY,
            inputs=[io.Latent.Input("protected_source"), io.Latent.Input("sampled")],
            outputs=[io.Latent.Output("latent")])

    @classmethod
    def execute(cls, protected_source, sampled):
        return io.NodeOutput(restore_av_protection(protected_source, sampled))


class MMH3TwoClipAVBridge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3TwoClipAVBridge", display_name="MMH3 Two-Clip AV Bridge", category=CATEGORY,
            inputs=[MMH3.Input("packet_a"), MMH3.Input("packet_b"),
                    io.Vae.Input("video_vae"), io.Vae.Input("audio_vae"),
                    io.Int.Input("context_a", default=39, min=1, max=1000),
                    io.Int.Input("context_b", default=39, min=1, max=1000),
                    io.Int.Input("gap_frames", default=46, min=1, max=3600),
                    io.Combo.Input("missing_audio", options=["error", "silence", "generate"], default="error")],
            outputs=[MMH3.Output("packet"), io.Latent.Output("latent"), io.String.Output("info_json"),
                     io.Int.Output("frames"), io.Int.Output("width"), io.Int.Output("height")])

    @classmethod
    def execute(cls, packet_a, packet_b, video_vae, audio_vae, context_a, context_b, gap_frames, missing_audio):
        require_joint_masks()
        packets = [_packet(packet_a), _packet(packet_b)]
        videos, provenance = [], []
        from types import SimpleNamespace
        for packet in packets:
            descriptor = packet.get_primary("video")
            if descriptor is None:
                raise MMH3ResourceError("Both bridge packets need decoded primary video")
            video = packet.ref(descriptor["id"]).materialize()
            components = video.get_components()
            audio = packet.get_primary("audio")
            if audio is not None:
                selected = packet.ref(audio["id"]).materialize()
                values = SimpleNamespace(images=components.images, frame_rate=components.frame_rate, audio=selected)
                video = SimpleNamespace(get_components=lambda values=values: values)
            videos.append(video)
            provenance.append({"packet_id": packet.manifest["id"], "video_resource_id": descriptor["id"],
                               "video_content": packet.ref(descriptor["id"]).descriptor.get("content"),
                               "audio_resource_id": audio["id"] if audio else None,
                               "audio_content": packet.ref(audio["id"]).descriptor.get("content") if audio else None})
        prepared = prepare_av_bridge(*videos, context_a=context_a, context_b=context_b,
                                     gap_frames=gap_frames, missing_audio=missing_audio)
        latent, report = encode_av_bridge(prepared, video_vae=video_vae, audio_vae=audio_vae)
        report["provenance"] = provenance
        from .core import MMH3Media
        height, width = prepared[0].shape[1:3]
        generation = {"prompt": packets[0].manifest.get("generation", {}).get("prompt", ""),
                      "seed": packets[0].manifest.get("generation", {}).get("seed", 0),
                      "width": width, "height": height, "frames": report["frames"], "fps": 24}
        packet = MMH3Media.create(name="Two-Clip AV Bridge", generation=generation)
        packet = packet.set_extension_value("minimax_h3", "av_bridge", report)
        return io.NodeOutput(packet, latent, json.dumps(report, ensure_ascii=False, indent=2), report["frames"], width, height)


class MMH3ReferenceSchedule(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3ReferenceSchedule", display_name="MMH3 Reference Schedule", category=CATEGORY,
            inputs=[MMH3.Input("packet"), io.String.Input("schedule_json", default='{"scenes":["scene_1"],"aliases":{}}', multiline=True)],
            outputs=[MMH3.Output("packet")])

    @classmethod
    def execute(cls, packet, schedule_json):
        return io.NodeOutput(configure_reference_schedule(_packet(packet), parse_schedule_json(schedule_json)))


class MMH3SceneReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3SceneReferences", display_name="MMH3 Scene References / Prompt Preview", category=CATEGORY,
            inputs=[MMH3.Input("packet"), io.String.Input("scene_id", default="scene_1"),
                    io.String.Input("prompt", default="", multiline=True, dynamic_prompts=False)],
            outputs=[MMH3.Output("packet"), io.String.Output("prompt"), io.String.Output("info_json")], is_output_node=True)

    @classmethod
    def execute(cls, packet, scene_id, prompt):
        from comfy_api.latest import ui
        out, compiled, report = compile_scene_references(_packet(packet), scene_id, prompt)
        return io.NodeOutput(out, compiled, json.dumps(report, ensure_ascii=False, indent=2), ui=ui.PreviewText(compiled))


class MMH3ReferenceAlias(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3ReferenceAlias", display_name="MMH3 Reference Alias", category=CATEGORY,
            inputs=[MMH3.Input("packet"), io.String.Input("resource_id", default="", force_input=True),
                    io.String.Input("alias", default="hero_face"),
                    io.String.Input("scenes_json", default='["scene_1"]', multiline=True)],
            outputs=[MMH3.Output("packet")])

    @classmethod
    def execute(cls, packet, resource_id, alias, scenes_json):
        from .scheduled_references import add_reference_alias
        return io.NodeOutput(add_reference_alias(_packet(packet), resource_id, alias, json.loads(scenes_json)))


class MMH3AVEditAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3AVEditAudio", display_name="MMH3 AV Edit Delivery Audio", category=CATEGORY,
            inputs=[MMH3.Input("packet"), io.Audio.Input("decoded_audio"),
                    io.Combo.Input("delivery_policy", options=["original_when_preserved", "decoded_latent"], default="original_when_preserved")],
            outputs=[io.Audio.Output("audio")])

    @classmethod
    def execute(cls, packet, decoded_audio, delivery_policy):
        from .av_edit import select_edit_delivery_audio
        return io.NodeOutput(select_edit_delivery_audio(_packet(packet), decoded_audio, delivery_policy=delivery_policy))


class MMH3BridgeMiddle(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3BridgeMiddle", display_name="MMH3 Bridge Middle for Assembly", category=CATEGORY,
            inputs=[io.Video.Input("video"), io.String.Input("bridge_info_json", default="{}", force_input=True)],
            outputs=[io.Video.Output("video")])

    @classmethod
    def execute(cls, video, bridge_info_json):
        from comfy_api.latest import InputImpl, Types
        from .av_bridge import trim_bridge_components
        images, audio, frame_rate = trim_bridge_components(video.get_components(), json.loads(bridge_info_json))
        out = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=frame_rate))
        return io.NodeOutput(out)
