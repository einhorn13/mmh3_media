from __future__ import annotations

from .decoded_upscale import upscale_decoded_video
from .node_support import CATEGORY, MMH3, MMH3ResourceError, _packet, io, ui, InputImpl, Types
from .video_output import save_output_video, build_video_decode_graph
from .video_output import DECODE_MODES, trt_decoder_options, DecodedVideo


class MMH3SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SaveVideo", display_name="MMH3 Save Video", category=CATEGORY,
            description="Save video once and pass its encoded file to MMH3 Save through the output packet.",
            inputs=[io.Video.Input("video"),
                    io.String.Input("filename_prefix", default="video/MMH3"),
                    io.Combo.Input("format", options=["auto", "mp4", "mkv", "webm"], default="auto"),
                    io.Combo.Input("codec", options=["auto", "h264", "av1"], default="auto"),
                    MMH3.Input("packet", optional=True)],
            outputs=[io.Video.Output("video"), MMH3.Output("packet")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video, filename_prefix, format="auto", codec="auto", packet=None):
        import os
        import folder_paths
        from comfy.cli_args import args

        if format not in {"auto", "mp4", "mkv", "webm"} or codec not in {"auto", "h264", "av1"}:
            raise MMH3ResourceError("Unsupported video format or codec")
        format = ("webm" if codec == "av1" else "mp4") if format == "auto" else format
        width, height = video.get_dimensions()
        directory, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        filename = f"{name}_{counter:05}_.{format}"
        metadata = {}
        if not args.disable_metadata:
            metadata.update(cls.hidden.extra_pnginfo or {})
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
        saved, out = save_output_video(
            video, os.path.join(directory, filename), packet=_packet(packet) if packet is not None else None,
            reader_factory=InputImpl.VideoFromFile, format=Types.VideoContainer(format),
            codec=Types.VideoCodec(codec), metadata=metadata or None,
        )
        return io.NodeOutput(saved, out, ui=ui.PreviewVideo([ui.SavedResult(filename, subfolder, io.FolderType.output)]))


class MMH3H3VideoDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3H3VideoDecode", display_name="H3 Video Decode", category=CATEGORY,
            description="Final video decode: connected VAE, temporal taeh3 draft, or H3 TensorRT. Encoding and audio are unchanged. Connect decode_info_json to MMH3 Create Video to archive provenance.",
            inputs=[io.Latent.Input("samples"), io.Vae.Input("vae"),
                    io.Combo.Input("decode_mode", options=DECODE_MODES, default="vae", display_name="Video decoder"),
                    io.Combo.Input("trt_decoder", options=trt_decoder_options(), default="auto", advanced=True)],
            outputs=[io.Image.Output("IMAGE"), io.String.Output("decode_info_json")],
        )

    @classmethod
    def execute(cls, samples, vae, decode_mode="vae", trt_decoder="auto", draft_tae=None):
        import json
        import nodes
        if draft_tae is not None:
            decode_mode = 'draft' if draft_tae else decode_mode
        from .video_output import load_video_decoder
        vae, report = load_video_decoder(vae, decode_mode, trt_decoder, nodes.NODE_CLASS_MAPPINGS)
        images = nodes.NODE_CLASS_MAPPINGS['VAEDecode']().decode(vae, samples)[0]
        # A long stitched latent must decode to its full AV timeline, never a silently truncated engine tile.
        from .h3 import is_nested_tensor, nested_parts, h3_frame_count_from_video_t
        latent = samples['samples']
        latent = nested_parts(latent)[0] if is_nested_tensor(latent) else latent
        expected = 1 if latent.shape[2] == 1 else h3_frame_count_from_video_t(latent.shape[2])
        if expected is not None and len(images) != expected * latent.shape[0]:
            raise MMH3ResourceError(f'{decode_mode} decoder returned {len(images)} frames; expected {expected * latent.shape[0]}')
        report['frames'] = len(images)
        return io.NodeOutput(images, json.dumps(report))


class MMH3CreateVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MMH3CreateVideo', display_name='MMH3 Create Video', category=CATEGORY,
            inputs=[io.Image.Input('images'), io.Float.Input('fps', default=24.0, min=1.0, max=120.0),
                    io.Audio.Input('audio', optional=True),
                    io.Combo.Input('bit_depth', options=['auto', 8, 10], default=8, optional=True),
                    io.Combo.Input('color_space', options=['sRGB', 'HDR', 'HDR PQ'], default='sRGB', optional=True),
                    io.String.Input('decode_info_json', default='', optional=True, force_input=True)],
            outputs=[io.Video.Output('VIDEO')])

    @classmethod
    def execute(cls, images, fps=24.0, audio=None, bit_depth=8, color_space='sRGB', decode_info_json=''):
        import json
        from comfy_extras.nodes_video import CreateVideo
        from .video_output import video_decode_metadata
        video = CreateVideo.execute(images, fps, audio, bit_depth, color_space)[0]
        if decode_info_json:
            video = DecodedVideo(video, json.loads(decode_info_json))
            video_decode_metadata(video)
        return io.NodeOutput(video)


class MMH3H3DecodeVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MMH3H3DecodeVideo', display_name='H3 Decode to Video', category=CATEGORY,
            description='Final H3 delivery without a full IMAGE batch. Auto streams supported TRT/TAEH3 videos over 15 seconds. Off retains ordinary decode. Audio and the joint AV latent remain unchanged.',
            inputs=[io.Latent.Input('samples'), io.Vae.Input('vae'),
                    io.Combo.Input('decode_mode', options=DECODE_MODES, default='vae'),
                    io.Combo.Input('trt_decoder', options=trt_decoder_options(), default='auto', advanced=True),
                    io.Combo.Input('streaming', options=['off', 'auto', 'stream'], default='auto'),
                    io.Audio.Input('audio', optional=True)],
            outputs=[io.Video.Output('VIDEO'), io.String.Output('decode_info_json')])

    @classmethod
    def execute(cls, samples, vae, decode_mode='vae', trt_decoder='auto', streaming='auto', audio=None):
        import json
        import nodes
        from .streaming_decode import video_latent, backend_kind, select_stream, StreamingDecodedVideo
        from .video_output import load_video_decoder
        vae, report = load_video_decoder(vae, decode_mode, trt_decoder, nodes.NODE_CLASS_MAPPINGS)
        z, frames = video_latent(samples)
        kind = backend_kind(vae.first_stage_model)
        active = select_stream(streaming, frames, kind is not None)
        report.update(frames=frames, streaming='stream' if active else 'off', streaming_requested=streaming)
        if active:
            report['temporal_method'] = 'persistent_memblocks' if kind == 'draft' else '7_token_overlap_blend'
            video = StreamingDecodedVideo(vae, z, frames, kind, audio, report)
        else:
            from comfy_extras.nodes_video import CreateVideo
            images = nodes.NODE_CLASS_MAPPINGS['VAEDecode']().decode(vae, samples)[0]
            if len(images) != frames:
                raise MMH3ResourceError(f'Decoder returned {len(images)} frames; expected {frames}')
            video = DecodedVideo(CreateVideo.execute(images, 24.0, audio, 8, 'sRGB')[0], report)
        return io.NodeOutput(video, json.dumps(report))


class MMH3VideoUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3VideoUpscale", display_name="MMH3 Video Upscale", category=CATEGORY,
            description="Optional RTX VSR after decoding. Keeps the original video, audio and H3 latent; adds a separate delivery video to the packet.",
            inputs=[MMH3.Input("packet"),
                    io.DynamicCombo.Input("upscale", options=[
                        io.DynamicCombo.Option("Original", []),
                        io.DynamicCombo.Option("RTX VSR", [
                            io.Float.Input("scale", default=1.5, min=1.0, max=4.0, step=0.05),
                            io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="HIGH", advanced=True),
                        ]),
                    ])],
            outputs=[MMH3.Output("packet"), io.Video.Output("video"),
                     io.String.Output("resource_id"), io.String.Output("summary")],
        )

    @classmethod
    def execute(cls, packet, upscale):
        choice = upscale if isinstance(upscale, dict) else {"upscale": upscale}
        mode = choice.get("upscale", "Original")
        if mode not in {"Original", "RTX VSR"}:
            raise MMH3ResourceError("Choose Original or RTX VSR")
        scale = float(choice.get("scale", 1.5))
        backend = None
        if mode == "RTX VSR" and scale != 1.0:
            import nodes
            backend = nodes.NODE_CLASS_MAPPINGS.get("RTXVideoSuperResolution")
            if backend is None:
                raise MMH3ResourceError("RTX VSR is unavailable; select Original or install NVIDIA RTX nodes")
            schema = backend.define_schema()
            if tuple(i.id for i in schema.inputs) != ("images", "resize_type", "quality"):
                raise MMH3ResourceError("Unsupported RTX VSR node schema")

        def apply(images, factor, quality):
            return backend.execute(images=images, resize_type={"resize_type": "scale by multiplier", "scale": factor},
                                   quality=quality)[0]

        def make_video(images, audio, frame_rate, *, bit_depth):
            return InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=frame_rate), bit_depth=bit_depth)

        out, video, rid, summary = upscale_decoded_video(
            _packet(packet), enabled=mode == "RTX VSR", scale=scale,
            quality=str(choice.get("quality", "HIGH")), upscale=apply if backend else None,
            make_video=make_video,
        )
        return io.NodeOutput(out, video, rid, summary, ui=ui.PreviewText(summary))
