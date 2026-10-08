"""Internal lazy expansion: hits skip native encoding, misses preserve native H3."""
from .node_support import io, MMH3, CATEGORY
from .comfy_h3_expansion import build_h3_expansion
from .conditioning_reuse import conditioning_key, conditioning_memory
from .runtime_contract import require_native_h3_contract

H3_INPUTS = io.Custom("MMH3_H3_RESOLVED")


class MMH3H3ReuseConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3ReuseConditioning", display_name="H3 Conditioning · Memory Reuse",
            category=f"{CATEGORY}/Internal", inputs=[MMH3.Input("packet"), H3_INPUTS.Input("resolved"),
                io.Clip.Input("clip"), io.Vae.Input("video_vae"),
                io.String.Input("ref_image_size"), io.Int.Input("ref_image_short_edge", min=-1, max=2048),
                io.Vae.Input("audio_vae", optional=True)],
            outputs=[io.Conditioning.Output("positive"), io.Latent.Output("latent")], enable_expand=True)

    @classmethod
    def execute(cls, packet, resolved, clip, video_vae, ref_image_size, ref_image_short_edge, audio_vae=None):
        from comfy_execution.graph_utils import GraphBuilder
        contract = require_native_h3_contract(resolved.mode)
        sizing = dict(ref_image_size=ref_image_size,
                      ref_image_short_edge=None if ref_image_short_edge < 0 else ref_image_short_edge)
        key = conditioning_key(packet, resolved, clip=clip, video_vae=video_vae, audio_vae=audio_vae,
                               contract=contract, **sizing)
        cached = conditioning_memory.get(key) if key else None
        if cached is not None:
            return io.NodeOutput(*cached)
        graph = GraphBuilder()
        expansion = build_h3_expansion(packet, resolved, clip=clip, video_vae=video_vae, audio_vae=audio_vae,
            contract=contract, graph_builder_factory=lambda: graph, **sizing)
        stored = graph.node("MMH3H3RememberConditioning", key=key,
                            positive=expansion.positive, latent=expansion.latent)
        return io.NodeOutput(stored.out(0), stored.out(1), expand=graph.finalize())


class MMH3H3RememberConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3RememberConditioning", display_name="H3 Conditioning · Remember",
            category=f"{CATEGORY}/Internal", inputs=[io.String.Input("key"),
                io.Conditioning.Input("positive"), io.Latent.Input("latent")],
            outputs=[io.Conditioning.Output("positive"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, key, positive, latent):
        conditioning_memory.put(key, positive, latent)
        return io.NodeOutput(positive, latent)
