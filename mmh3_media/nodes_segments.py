from __future__ import annotations

from .node_support import CATEGORY, MMH3, _packet, io, ui
from .segment_plan import SEGMENT_ACTIONS, prepare_segment, review_segment


def build_segment_expansion(plan, *, clip, video_vae, audio_vae, width: int, height: int,
                            task_family: str, graph_builder_factory=None):
    if graph_builder_factory is None:
        from comfy_execution.graph_utils import GraphBuilder
        graph_builder_factory = GraphBuilder
    graph = graph_builder_factory()
    common = dict(packet=plan.packet, prompt_override="", seed_override=-1, clip=clip,
                  video_vae=video_vae, width_override=width, height_override=height,
                  frames_override=plan.info["timing"]["generated_frames"])
    if audio_vae is not None:
        common["audio_vae"] = audio_vae
    if plan.info["operation"] == "continuation":
        handover = graph.node("MMH3H3ContinuationHandover", packet=plan.packet,
                              target_frames=common["frames_override"],
                              video_handover_frames=plan.info["timing"]["context_frames"],
                              audio_handover_frames=0, audio_feather_frames=0,
                              target_width=width, target_height=height)
        condition = graph.node("MMH3H3ContinuationCondition", **common,
                               task_family=task_family, handover_info_json=handover.out(3))
        outputs = (condition.out(0), handover.out(1), condition.out(1), condition.out(2),
                   condition.out(3), condition.out(4), condition.out(5), condition.out(6))
    else:
        condition = graph.node("MMH3H3AutoCondition", **common)
        outputs = tuple(condition.out(i) for i in range(8))
    return outputs, graph.finalize()


class MMH3H3SegmentPrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3H3SegmentPrepare", display_name="Prepare H3 Segment", category=CATEGORY,
            description="One accepted parent, one prompt plan. Repeating a draft never advances the plan. New scene uses fresh generation; Continue preserves the exact AV prefix.",
            inputs=[
                MMH3.Input("packet", display_name="Source"),
                io.Combo.Input("action", options=list(SEGMENT_ACTIONS), default="Continue"),
                io.String.Input("prompts", default="", multiline=True, dynamic_prompts=False,
                                tooltip="Separate segment prompts with a line containing ---. One prompt repeats; blank inherits the saved plan. Describe the preserved opening before changing the scene."),
                io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="-1 inherits the saved seed. Change this to try another draft."),
                io.Clip.Input("clip"), io.Vae.Input("video_vae"), io.Vae.Input("audio_vae", optional=True),
                io.Int.Input("width", default=0, min=0, max=16384, force_input=True),
                io.Int.Input("height", default=0, min=0, max=16384, force_input=True),
                io.Int.Input("frames", default=124, min=5, max=3600, force_input=True),
                io.Int.Input("context_frames", default=39, min=39, max=3600, step=51, advanced=True),
                io.Combo.Input("task_family", options=["auto", "fl2va", "ref2va"], default="auto", advanced=True,
                               tooltip="Future conditioning for Continue. First segment/New scene use the source packet's generation mode."),
                MMH3.Input("chain_packet", optional=True, tooltip="Reroll only: latest accepted chain. Source must be the target's parent."),
                io.String.Input("target_segment_id", default="", advanced=True, tooltip="Reroll only. Blank selects the last accepted segment."),
                io.Image.Input("reanchor_image", optional=True,
                               tooltip="Reanchor only: first frame of a fresh scene, without the previous AV prefix. Required again when rerolling a reanchored scene."),
            ],
            outputs=[io.Conditioning.Output("positive"), io.Latent.Output("latent"),
                     io.Int.Output("seed"), MMH3.Output("packet"), io.String.Output("mode"),
                     io.String.Output("status"), io.String.Output("info_json"),
                     io.Combo.Output("task_family"), io.String.Output("operation"),
                     io.String.Output("timing")], enable_expand=True,
        )

    @classmethod
    def execute(cls, packet, action, prompts, seed, clip, video_vae, width, height, frames,
                context_frames=39, task_family="auto", audio_vae=None, chain_packet=None,
                target_segment_id="", reanchor_image=None):
        plan = prepare_segment(_packet(packet), action=action, prompts=prompts, seed=seed,
                               frames=frames, context_frames=context_frames,
                               chain_packet=_packet(chain_packet) if chain_packet is not None else None,
                               target_segment_id=target_segment_id, reanchor_image=reanchor_image)
        outputs, graph = build_segment_expansion(plan, clip=clip, video_vae=video_vae,
                                                 audio_vae=audio_vae, width=width, height=height,
                                                 task_family=task_family)
        return io.NodeOutput(*outputs, plan.info["operation"], plan.summary, expand=graph,
                             ui=ui.PreviewText(plan.summary))


class MMH3SegmentReview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SegmentReview", display_name="Review H3 Segment", category=CATEGORY,
            description="Draft keeps the accepted parent unchanged. Accept records this result and enables the next prompt. Save the accepted packet and load it as Source to continue.",
            inputs=[MMH3.Input("packet"),
                    io.Combo.Input("decision", options=["Draft", "Accept"], default="Draft"),
                    MMH3.Input("chain_packet", optional=True, tooltip="Connect the same accepted chain as Prepare for rerolls.")],
            outputs=[MMH3.Output("packet"), io.String.Output("filename_prefix"), io.String.Output("summary")],
        )

    @classmethod
    def execute(cls, packet, decision="Draft", chain_packet=None):
        if decision not in {"Draft", "Accept"}:
            raise ValueError("Decision must be Draft or Accept")
        out, prefix, summary = review_segment(_packet(packet), accept=decision == "Accept",
                                              chain_packet=_packet(chain_packet) if chain_packet is not None else None)
        return io.NodeOutput(out, prefix, summary, ui=ui.PreviewText(summary))
