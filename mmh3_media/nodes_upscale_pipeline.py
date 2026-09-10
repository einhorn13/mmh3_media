"""Explicit settings, execution gate, sigma recording and PCM delivery for F07."""
from .node_support import CATEGORY, MMH3, io, json, _packet, _parse_object
from .upscale_execution import build_refine_sigmas, upscale_preflight, select_upscale_audio


class MMH3H3UpscaleSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3UpscaleSettings", display_name="MMH3 H3 Upscale Settings", category=CATEGORY,
            inputs=[
                io.String.Input("upscaler_model", default="minimax_h3_latent_upscaler_3d_bf16.safetensors"),
                io.String.Input("refine_checkpoint", default="minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
                io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp32", "fp16", "bf16"], default="bf16"),
                io.Combo.Input("model_family", options=["auto", "fl2va", "ref2va"], default="auto"),
                io.Combo.Input("audio_policy", options=["source_pcm", "decoded_latent"], default="source_pcm"),
            ], outputs=[io.String.Output("settings_json"), io.AnyType.Output("upscaler_model"),
                io.AnyType.Output("refine_checkpoint"), io.AnyType.Output("device"), io.AnyType.Output("precision"),
                io.String.Output("upscaler_model_text")])

    @classmethod
    def execute(cls, upscaler_model, refine_checkpoint, device, precision, model_family, audio_policy):
        from .errors import MMH3ResourceError
        if not upscaler_model.strip() or not refine_checkpoint.strip():
            raise MMH3ResourceError("F07 model filenames must not be empty")
        settings = dict(upscaler_model=upscaler_model, refine_checkpoint=refine_checkpoint,
                        device=device, precision=precision, model_family=model_family, audio_policy=audio_policy)
        return io.NodeOutput(json.dumps(settings), upscaler_model, refine_checkpoint, device, precision, upscaler_model)


class MMH3H3UpscalePreflight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3UpscalePreflight", display_name="MMH3 H3 Upscale Preflight", category=CATEGORY,
            inputs=[io.Latent.Input("video_latent"), io.Model.Input("model"), MMH3.Input("packet"),
                io.String.Input("geometry_json"), io.String.Input("refine_sampling_json"),
                io.String.Input("settings_json"), io.String.Input("optimization_profile_json")],
            outputs=[io.Latent.Output("video_latent"), io.Model.Output("model"),
                io.String.Output("preflight_json")])

    @classmethod
    def execute(cls, packet, video_latent, model, geometry_json, refine_sampling_json, settings_json, optimization_profile_json):
        from .upscale_execution import validate_upscaler_video_input
        validate_upscaler_video_input(video_latent)
        import nodes
        report = upscale_preflight(_packet(packet), _parse_object(geometry_json, "geometry"),
            _parse_object(refine_sampling_json, "sampling"), _parse_object(settings_json, "settings"),
            _parse_object(optimization_profile_json, "optimizations"), nodes.NODE_CLASS_MAPPINGS)
        return io.NodeOutput(video_latent, model, json.dumps(report))


class MMH3H3RefineScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3RefineScheduler", display_name="MMH3 H3 Refine Scheduler", category=CATEGORY,
            inputs=[io.Model.Input("model"), io.String.Input("refine_sampling_json")],
            outputs=[io.Sigmas.Output("SIGMAS"), io.String.Output("refine_sampling_json")])

    @classmethod
    def execute(cls, model, refine_sampling_json):
        import comfy.samplers
        sigmas, report = build_refine_sigmas(_parse_object(refine_sampling_json, "sampling"),
            lambda scheduler, steps: comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps))
        return io.NodeOutput(sigmas, json.dumps(report))


class MMH3H3UpscaleAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3UpscaleAudio", display_name="MMH3 H3 Upscale Audio", category=CATEGORY,
            inputs=[MMH3.Input("packet"), io.Audio.Input("decoded_audio"),
                io.String.Input("settings_json"), io.Int.Input("frames", default=5, min=1)],
            outputs=[io.Audio.Output("audio"), io.String.Output("audio_delivery_json")])

    @classmethod
    def execute(cls, packet, decoded_audio, settings_json, frames):
        settings = _parse_object(settings_json, "settings")
        audio, report = select_upscale_audio(_packet(packet), decoded_audio, settings["audio_policy"], frames)
        return io.NodeOutput(audio, json.dumps(report))
