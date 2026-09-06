from __future__ import annotations

from .decoded_upscale import upscale_decoded_video
from .node_support import CATEGORY, MMH3, MMH3ResourceError, _packet, io, ui, InputImpl, Types


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
