from __future__ import annotations

from .stitch import resolve_stitch_transition
from .latent_stitch import inspect_h3_latent_stitch_packets, stitch_h3_continuation_latents

from .node_support import (
    CATEGORY,
    MMH3,
    MMH3ResourceError,
    _packet,
    build_streaming_stitch,
    commit_chain_segment,
    inspect_stitch_packets,
    io,
    json,
    materialize_streaming_segment,
    pack_h3_result,
    start_chain,
    unpack_primary,
    validate_chain,
    validate_reroll_source,
)

class MMH3ChainStart(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ChainStart",
            display_name="MMH3 Chain Start",
            category=CATEGORY,
            description="Initialize an embedded F04 chain manifest. Existing valid chain state is preserved unchanged.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("name", default=""),
                io.String.Input("chain_id", default="", advanced=True),
                io.String.Input("scene_id", default="", advanced=True),
                io.Int.Input("reanchor_every_segments", default=4, min=1, max=1000, advanced=True),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("summary"), io.String.Output("info_json")],
        )

    @classmethod
    def execute(
        cls,
        packet,
        name: str,
        chain_id: str,
        scene_id: str,
        reanchor_every_segments: int,
    ) -> io.NodeOutput:
        out = start_chain(
            _packet(packet),
            name=name,
            chain_id=chain_id,
            scene_id=scene_id,
            reanchor_every_segments=int(reanchor_every_segments),
        )
        report = validate_chain(out, require_head_state=False)
        return io.NodeOutput(out, report.summary(), json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


class MMH3ChainCommit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ChainCommit",
            display_name="MMH3 Chain Commit Segment",
            category=CATEGORY,
            description=(
                "Commit the latest atomic process result as append, reroll or re-anchor state. "
                "Returns the deterministic archive prefix for MMH3 Save."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("action", options=["append", "reroll", "reanchor"], default="append"),
                io.String.Input("target_segment_id", default="", advanced=True),
                io.String.Input("archive_prefix", default="", advanced=True),
                MMH3.Input(
                    "chain_packet",
                    optional=True,
                    tooltip="Required for reroll: latest authoritative ledger packet, separate from regenerated result.",
                ),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.String.Output("segment_id"),
                io.String.Output("scene_id"),
                io.Int.Output("revision"),
                io.String.Output("filename_prefix"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        action: str,
        target_segment_id: str,
        archive_prefix: str,
        chain_packet=None,
    ) -> io.NodeOutput:
        result = commit_chain_segment(
            _packet(packet),
            action=action,
            target_segment_id=target_segment_id,
            archive_prefix=archive_prefix,
            chain_packet=_packet(chain_packet) if chain_packet is not None else None,
        )
        info = result.validation.to_dict()
        info.update(
            {
                "segment_id": result.segment_id,
                "scene_id": result.scene_id,
                "revision": result.revision,
                "filename_prefix": result.filename_prefix,
            }
        )
        return io.NodeOutput(
            result.packet,
            result.segment_id,
            result.scene_id,
            result.revision,
            result.filename_prefix,
            result.validation.summary(),
            json.dumps(info, ensure_ascii=False, indent=2),
        )


class MMH3ChainValidate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ChainValidate",
            display_name="MMH3 Chain Validate Resume",
            category=CATEGORY,
            description="Validate chain topology and prove that current primary resources match the recorded head.",
            inputs=[
                MMH3.Input("packet"),
                io.Boolean.Input("require_head_state", default=True, advanced=True),
                io.Boolean.Input("fail_on_invalid", default=True, advanced=True),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Boolean.Output("ready"),
                io.Boolean.Output("should_reanchor"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(cls, packet, require_head_state: bool, fail_on_invalid: bool) -> io.NodeOutput:
        packet = _packet(packet)
        report = validate_chain(packet, require_head_state=bool(require_head_state))
        if bool(fail_on_invalid) and not report.ready:
            raise MMH3ResourceError("F04 chain resume blocked: " + "; ".join(report.reasons))
        return io.NodeOutput(
            packet,
            report.ready,
            report.should_reanchor,
            report.summary(),
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        )


class MMH3ChainValidateRerollSource(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3ChainValidateRerollSource",
            display_name="MMH3 Chain Validate Reroll Source",
            category=CATEGORY,
            description=(
                "Prove that a separately loaded source packet matches the parent/root state required "
                "to regenerate the selected segment."
            ),
            inputs=[
                MMH3.Input("chain_packet"),
                MMH3.Input("source_packet"),
                io.String.Input("target_segment_id"),
            ],
            outputs=[MMH3.Output("source_packet"), io.String.Output("summary"), io.String.Output("info_json")],
        )

    @classmethod
    def execute(cls, chain_packet, source_packet, target_segment_id: str) -> io.NodeOutput:
        chain_packet = _packet(chain_packet)
        source_packet = _packet(source_packet)
        report = validate_reroll_source(
            chain_packet,
            source_packet,
            target_segment_id=target_segment_id,
        )
        if not report.ready:
            raise MMH3ResourceError("F04 reroll source blocked: " + "; ".join(report.reasons))
        summary = (
            f"READY · reroll {report.target_segment_id} from "
            f"{report.expected_parent_segment_id or 'chain root'}"
        )
        return io.NodeOutput(
            source_packet,
            summary,
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        )


class MMH3VideoStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3VideoStitch",
            display_name="MMH3 Video Stitch",
            category=CATEGORY,
            description=(
                "Streaming decoded RGB/PCM stitch. Auto Seamless analyzes scene continuity and motion inside "
                "the fixed overlap, choosing either a crossfade or a motion-safe ownership cut. Seam-local "
                "exposure/white-balance/contrast correction is suppressed for detected scene changes and decays "
                "back to the source look."
            ),
            inputs=[
                io.Autogrow.Input("segments",
                    template=io.Autogrow.TemplateNames(MMH3.Input("segment"),
                        names=[f"segment_{index}" for index in range(1, 67)], min=2),
                    display_name="Segments"),
                io.Combo.Input("transition", options=["Auto Seamless", "Cut", "Crossfade"], default="Auto Seamless"),
                io.Int.Input("transition_frames", default=0, min=0, max=240, advanced=True,
                             tooltip="0 = automatic (about 0.25 s, shortened for small clips). Positive = manual frames."),
                io.Combo.Input("color_match", options=["Auto", "Off", "Always"], default="Auto", advanced=True,
                               tooltip="Auto corrects exposure/white-balance/contrast only for continuity seams; detected scene changes are left untouched. Always overrides that safeguard."),
                io.Float.Input("color_match_strength", default=0.85, min=0.0, max=1.0, step=0.05, advanced=True),
                io.Int.Input("color_match_decay_frames", default=24, min=0, max=240, advanced=True,
                             tooltip="Fade color/exposure correction back to the source look over this many frames."),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Video.Output("video"),
                io.Audio.Output("audio"),
                io.String.Output("summary"),
                io.String.Output("assembly_report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        segments,
        transition: str,
        transition_frames: int,
        color_match: str,
        color_match_strength: float,
        color_match_decay_frames: int,
    ) -> io.NodeOutput:
        packets = []
        for key in sorted(segments or {}, key=lambda name: int(name.rsplit("_", 1)[-1])):
            if segments[key] is not None:
                packets.append(_packet(segments[key]))
        compatibility = inspect_stitch_packets(packets)
        if not compatibility.ready:
            raise MMH3ResourceError("F05 video stitch compatibility blocked: " + "; ".join(compatibility.reasons))
        video_mode, audio_mode, overlap_frames = resolve_stitch_transition(
            transition, transition_frames, [int(fact["frame_count"]) for fact in compatibility.facts])
        facts = list(compatibility.facts)
        dimensions = facts[0].get("dimensions")
        if not isinstance(dimensions, list) or len(dimensions) != 2:
            raise MMH3ResourceError("F05 streaming assembly requires manifest dimensions")
        streams = [materialize_streaming_segment(packet, fact) for packet, fact in zip(packets, facts)]
        color_mode = {"Auto": "auto", "Off": "off", "Always": "always"}[color_match]
        result = build_streaming_stitch(
            streams, dimensions=(int(dimensions[0]), int(dimensions[1])),
            video_mode=video_mode, audio_mode=audio_mode, overlap_frames=int(overlap_frames),
            seam_strategy="auto_seamless" if transition == "Auto Seamless" else "manual",
            color_match_mode=color_mode, color_match_strength=float(color_match_strength),
            color_match_decay_frames=int(color_match_decay_frames))
        video = result.video
        report = result.plan.to_dict()
        report["compatibility"] = compatibility.to_dict()
        report["assembly_mode"] = "streaming_decoded"
        report["transition_length_mode"] = "none" if transition == "Cut" else ("auto" if transition_frames == 0 else "manual")
        report["rgb_buffer_bound_frames"] = 0 if video_mode == "cut" else 2 * int(overlap_frames)
        report["auto_seamless_note"] = (
            "Auto Seamless is evaluated during streaming decode inside the fixed overlap. It preserves total "
            "timeline length, chooses crossfade or ownership_cut from scene/motion evidence, and records per-seam "
            "decisions in StreamingStitchedVideo save stats."
        )
        report["color_match_note"] = (
            "Seam-local exposure/white-balance/contrast correction is evaluated during streaming decode. Auto "
            "mode suppresses correction for detected scene changes; actual decisions are stored in save stats."
        )
        packed = pack_h3_result(
            packets[-1], video=video, audio=result.audio, operation="video_stitch",
            mode=f"{video_mode}+{audio_mode}", status=result.plan.summary(), process_info=report)
        out = packed.packet.set_extension_value("mmh3_media", "assembly", report)
        return io.NodeOutput(out, video, result.audio, result.plan.summary(),
                             json.dumps(report, ensure_ascii=False, indent=2))


class MMH3H3LatentStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3LatentStitch",
            display_name="MMH3 H3 Latent Stitch",
            category=CATEGORY,
            description=(
                "Assemble provenance-linked H3 continuation segments in joint AV latent space. "
                "Trims duplicated continuation prefixes, preserves H3 video phase, and resolves audio ticks "
                "against absolute 40 Hz boundaries. Independent latents and latent blending are rejected."
            ),
            inputs=[
                io.Autogrow.Input("segments",
                    template=io.Autogrow.TemplateNames(MMH3.Input("segment"),
                        names=[f"segment_{index}" for index in range(1, 67)], min=2),
                    display_name="Continuation Segments"),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Latent.Output("latent"),
                io.String.Output("summary"),
                io.String.Output("assembly_report_json"),
            ],
        )

    @classmethod
    def execute(cls, segments) -> io.NodeOutput:
        packets = []
        for key in sorted(segments or {}, key=lambda name: int(name.rsplit("_", 1)[-1])):
            if segments[key] is not None:
                packets.append(_packet(segments[key]))
        compatibility = inspect_h3_latent_stitch_packets(packets)
        if not compatibility.ready:
            raise MMH3ResourceError("F05 H3 latent stitch compatibility blocked: " + "; ".join(compatibility.reasons))
        result = stitch_h3_continuation_latents(packets)
        report = result.plan.to_dict()
        report["compatibility"] = compatibility.to_dict()
        report["decode_required"] = True
        report["decode_hint"] = "Connect latent to stock VAEDecode + VAEDecodeAudio, then CreateVideo."
        packed = pack_h3_result(
            packets[-1], latent=result.latent, operation="latent_stitch", mode="h3_continuation",
            status=result.plan.summary(), process_info=report, latent_origin="derived")
        out = packed.packet.set_extension_value("mmh3_media", "assembly", report)
        return io.NodeOutput(out, result.latent, result.plan.summary(),
                             json.dumps(report, ensure_ascii=False, indent=2))


class MMH3Unpack(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Unpack",
            display_name="MMH3 Unpack",
            category=CATEGORY,
            description="Explicit interoperability escape hatch. Materializes primary singleton resources; use typed Get nodes when only one heavy resource is needed.",
            inputs=[MMH3.Input("packet")],
            outputs=[
                io.Latent.Output("latent"),
                io.Video.Output("video"),
                io.Audio.Output("audio"),
                io.Image.Output("first_frame"),
                io.Image.Output("last_frame"),
                io.Boolean.Output("has_latent"),
                io.Boolean.Output("has_video"),
                io.Boolean.Output("has_audio"),
                io.Boolean.Output("has_first_frame"),
                io.Boolean.Output("has_last_frame"),
                io.String.Output("prompt"),
                io.Int.Output("seed"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("frames"),
                io.Float.Output("fps"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(cls, packet) -> io.NodeOutput:
        view = unpack_primary(_packet(packet))
        generation = view.generation

        def integer(name: str) -> int:
            try:
                return int(generation.get(name) or 0)
            except (TypeError, ValueError):
                return 0

        try:
            fps = float(generation.get("fps") or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        has = {role: role in view.resource_ids for role in ("latent", "video", "audio", "first_frame", "last_frame")}
        return io.NodeOutput(
            view.latent,
            view.video,
            view.audio,
            view.first_frame,
            view.last_frame,
            has["latent"],
            has["video"],
            has["audio"],
            has["first_frame"],
            has["last_frame"],
            str(generation.get("prompt") or ""),
            integer("seed"),
            integer("width"),
            integer("height"),
            integer("frames"),
            fps,
            json.dumps(view.to_dict(), ensure_ascii=False, indent=2),
        )
