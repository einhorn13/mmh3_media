from __future__ import annotations

import json
from pathlib import Path

from .automation import H3_AUDIO_TIMELINE_PLANNERS, plan_batch_inputs, plan_h3_audio_interactive_sequence, plan_h3_audio_timeline_chunks, plan_long_video_chunks, resolve_chunk_execution
from .automation_batch_stitch import inspect_batch_stitch_plan, prepare_batch_stitch
from .raw_video_import import normalize_batch_plan
from .resource_model import resource_facts
from .automation_adapters import build_audio_timeline_chunk_expansion, build_video_chunk_expansion, trim_audio_samples, trim_audio_samples_padded
from .automation_estimate import build_prequeue_estimate, prequeue_summary
from .automation_assembly import assemble_chunk_packets
from .automation_upscale import build_long_video_upscale_settings
from .automation_lipsync import (
    MODES as AUDIO_SYNC_MODES,
    AUDIO_OUTPUT_MODES,
    build_h3_audio_sync_chunk_proof,
    build_long_video_audio_sync_settings,
    validate_lipsync_chunk_proof,
    validate_audio_sync_source,
)
from .automation_execution import (
    acquire_next_execution_job,
    append_interactive_audio_try,
    build_chunk_assembly_map,
    accept_execution_candidate,
    build_execution_summary,
    commit_execution_artifact,
    commit_execution_candidate,
    create_execution_ledger,
    deterministic_artifact_prefix,
    finalize_interactive_audio_sequence,
    save_execution_ledger,
    set_execution_audio_delivery_policy,
    select_resume_jobs,
    transition_execution_job,
)
from .node_support import (
    CATEGORY,
    MMH3,
    MMH3ResourceError,
    _packet,
    _safe_output_prefix,
    build_streaming_stitch,
    folder_paths,
    inspect_packet,
    io,
    materialize_streaming_segment,
    pack_h3_result,
)


class MMH3BatchInputPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3BatchInputPlan",
            display_name="MMH3 Batch Input Plan",
            category=CATEGORY,
            description="Create an ordered, deterministic planned-job manifest without opening or processing media.",
            inputs=[
                io.String.Input("paths_json", default="[]", multiline=True),
                io.Combo.Input("order", options=["natural", "provided"], default="natural"),
            ],
            outputs=[io.String.Output("plan_json"), io.Int.Output("jobs"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, paths_json: str, order: str) -> io.NodeOutput:
        try:
            paths = json.loads(paths_json)
        except json.JSONDecodeError as exc:
            raise MMH3ResourceError(f"paths_json is invalid: {exc}") from exc
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise MMH3ResourceError("paths_json must be an array of file path strings")
        plan = plan_batch_inputs(paths, order=order)
        count = len(plan["jobs"])
        return io.NodeOutput(json.dumps(plan, ensure_ascii=False, indent=2), count, f"READY · batch plan · jobs={count}")


class MMH3BatchPreQueueEstimate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3BatchPreQueueEstimate",
            display_name="MMH3 Batch Pre-Queue Estimate",
            category=CATEGORY,
            description=(
                "Inspect MMH3 manifests without decoding media and report planned jobs, exact known AV totals, "
                "output naming, error policy and explicit unknowns before batch execution."
            ),
            inputs=[
                io.String.Input("plan_json", default="{}", multiline=True),
                io.Combo.Input("video_mode", options=["cut", "crossfade"], default="cut"),
                io.Combo.Input("audio_mode", options=["cut", "half_cosine"], default="cut"),
                io.Int.Input("overlap_frames", default=0, min=0, max=240),
                io.Combo.Input("error_policy", options=["stop_on_error", "skip_invalid"], default="stop_on_error"),
                io.String.Input("output_prefix", default="mmh3_automation/batch_stitch/final"),
            ],
            outputs=[
                io.String.Output("estimate_json"),
                io.String.Output("preflight_json"),
                io.String.Output("summary"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, plan_json: str, video_mode: str, audio_mode: str, overlap_frames: int, error_policy: str, output_prefix: str) -> io.NodeOutput:
        plan = _json_object(plan_json, "plan_json")
        prepared = inspect_batch_stitch_plan(
            plan,
            search_roots=(folder_paths.get_input_directory(), folder_paths.get_output_directory()),
            error_policy=error_policy,
            video_mode=video_mode,
            audio_mode=audio_mode,
            overlap_frames=int(overlap_frames),
        )
        settings = {
            "assembly_mode": "streaming",
            "video_mode": video_mode,
            "audio_mode": audio_mode,
            "overlap_frames": int(overlap_frames),
        }
        estimate = build_prequeue_estimate(
            plan,
            operation="batch_stitch",
            effective_settings=settings,
            output_prefix=output_prefix,
            error_policy=error_policy,
            batch_preflight=prepared.report,
        )
        return io.NodeOutput(
            json.dumps(estimate, ensure_ascii=False, indent=2),
            json.dumps(prepared.report, ensure_ascii=False, indent=2),
            prequeue_summary(estimate),
        )


class MMH3BatchNormalizeImport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3BatchNormalizeImport",
            display_name="MMH3 Batch Normalize / Import Raw Video",
            category=CATEGORY,
            description=(
                "Normalize raw VIDEO jobs into temporary decoded MMH3 archives using a file-backed "
                "video path and bounded full-PCM audio decode, then emit a stitch-ready batch plan."
            ),
            inputs=[
                io.String.Input("plan_json", default="{}", multiline=True),
                io.Int.Input("target_width", default=0, min=0, max=16384),
                io.Int.Input("target_height", default=0, min=0, max=16384),
                io.Int.Input("target_fps", default=24, min=1, max=240),
                io.Int.Input("target_audio_sample_rate", default=32000, min=8000, max=192000),
                io.Int.Input("target_audio_channels", default=2, min=1, max=2),
                io.Int.Input("max_audio_pcm_mb", default=512, min=1, max=32768, advanced=True),
                io.Combo.Input(
                    "error_policy",
                    options=["stop_on_error", "skip_invalid"],
                    default="stop_on_error",
                ),
            ],
            outputs=[
                io.String.Output("normalized_plan_json"),
                io.String.Output("report_json"),
                io.Int.Output("accepted"),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        plan_json: str,
        target_width: int,
        target_height: int,
        target_fps: int,
        target_audio_sample_rate: int,
        target_audio_channels: int,
        max_audio_pcm_mb: int,
        error_policy: str,
    ) -> io.NodeOutput:
        normalized, report = normalize_batch_plan(
            _json_object(plan_json, "plan_json"),
            search_roots=(
                folder_paths.get_input_directory(),
                folder_paths.get_output_directory(),
            ),
            temp_root=folder_paths.get_temp_directory(),
            error_policy=error_policy,
            target_width=int(target_width),
            target_height=int(target_height),
            target_fps=int(target_fps),
            target_audio_sample_rate=int(target_audio_sample_rate),
            target_audio_channels=int(target_audio_channels),
            max_audio_pcm_mb=int(max_audio_pcm_mb),
        )
        accepted = int(report["counts"]["accepted"])
        canonical = "canonical" if report["stitch_canonical"] else "noncanonical"
        status = (
            f"READY · raw normalize/import · accepted={accepted} · "
            f"skipped={report['counts']['skipped']} · {canonical}"
        )
        return io.NodeOutput(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            json.dumps(report, ensure_ascii=False, indent=2),
            accepted,
            status,
        )


class MMH3BatchStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3BatchStitch",
            display_name="MMH3 Batch Stitch",
            category=CATEGORY,
            description=(
                "Validate and stream-stitch an arbitrary ordered batch of decoded MMH3 archives. "
                "Reports every rejected item and supports fail-fast or skip-invalid execution."
            ),
            inputs=[
                io.String.Input("plan_json", default="{}", multiline=True),
                io.Combo.Input("video_mode", options=["cut", "crossfade"], default="cut"),
                io.Combo.Input("audio_mode", options=["cut", "half_cosine"], default="cut"),
                io.Int.Input("overlap_frames", default=0, min=0, max=240),
                io.Combo.Input(
                    "error_policy",
                    options=["stop_on_error", "skip_invalid"],
                    default="stop_on_error",
                ),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Video.Output("video"),
                io.Audio.Output("audio"),
                io.String.Output("assembly_map_json"),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        plan_json: str,
        video_mode: str,
        audio_mode: str,
        overlap_frames: int,
        error_policy: str,
    ) -> io.NodeOutput:
        prepared = prepare_batch_stitch(
            _json_object(plan_json, "plan_json"),
            search_roots=(
                folder_paths.get_input_directory(),
                folder_paths.get_output_directory(),
            ),
            error_policy=error_policy,
            video_mode=video_mode,
            audio_mode=audio_mode,
            overlap_frames=int(overlap_frames),
        )
        facts = [
            item["facts"]
            for item in prepared.report["items"]
            if item["status"] == "accepted"
        ]
        dimensions = facts[0]["dimensions"]
        segments = [
            materialize_streaming_segment(packet, fact)
            for packet, fact in zip(prepared.packets, facts)
        ]
        result = build_streaming_stitch(
            segments,
            dimensions=(int(dimensions[0]), int(dimensions[1])),
            video_mode=video_mode,
            audio_mode=audio_mode,
            overlap_frames=int(overlap_frames),
        )
        input_pcm_bytes = sum(
            int(fact.get("audio_samples") or 0) * 2 * 4
            for fact in facts
        )
        output_pcm_bytes = int(result.plan.output_audio_samples) * 2 * 4
        assembly_map = {
            "contract": "mmh3_batch_stitch_assembly_map_v1",
            "ready": True,
            "preflight": prepared.report,
            "result": result.plan.to_dict(),
            "rgb_buffer_bound_frames": 0 if video_mode == "cut" else 2 * int(overlap_frames),
            "audio_pcm_materialization": {
                "input_pcm_bytes_total": input_pcm_bytes,
                "output_pcm_bytes": output_pcm_bytes,
                "full_input_pcm_materialized": True,
                "note": "F05 streaming keeps VIDEO file-backed but ComfyUI AUDIO is full float32 PCM; peak allocator overhead requires runtime measurement.",
            },
        }
        packed = pack_h3_result(
            prepared.packets[-1],
            video=result.video,
            audio=result.audio,
            operation="batch_stitch",
            mode=f"{video_mode}+{audio_mode}",
            status=result.plan.summary(),
            process_info=assembly_map,
        )
        packet = packed.packet.set_extension_value("mmh3_media", "batch_assembly", assembly_map)
        return io.NodeOutput(
            packet,
            result.video,
            result.audio,
            json.dumps(assembly_map, ensure_ascii=False, indent=2),
            prepared.summary() + f" · output={result.plan.output_frames}f",
        )


class MMH3LongVideoChunkPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3LongVideoChunkPlan",
            display_name="MMH3 Long Video Chunk Plan",
            category=CATEGORY,
            description="Plan bounded-memory long-video jobs with exact frame/audio ownership and optional scene boundaries.",
            inputs=[
                MMH3.Input("packet"),
                io.Float.Input("chunk_duration_seconds", default=10.0, min=1.0, max=600.0, step=0.5),
                io.Float.Input("overlap_seconds", default=1.0, min=0.0, max=60.0, step=0.25),
                io.Combo.Input("boundary_mode", options=["fixed", "scene_aware"], default="fixed"),
                io.String.Input("scene_cuts_json", default="[]", multiline=True, advanced=True),
                io.Float.Input("scene_search_seconds", default=1.5, min=0.0, max=30.0, step=0.25, advanced=True),
                io.Float.Input("min_tail_seconds", default=2.0, min=0.1, max=60.0, step=0.1, advanced=True),
                io.Combo.Input("min_tail_policy", options=["merge_previous", "keep", "drop"], default="merge_previous"),
                io.Int.Input("audio_sample_rate", default=32000, min=8000, max=192000, advanced=True),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("plan_json"), io.Int.Output("chunks"), io.String.Output("status"), io.String.Output("estimate_json")],
        )

    @classmethod
    def execute(
        cls,
        packet,
        chunk_duration_seconds: float,
        overlap_seconds: float,
        boundary_mode: str,
        scene_cuts_json: str,
        scene_search_seconds: float,
        min_tail_seconds: float,
        min_tail_policy: str,
        audio_sample_rate: int,
    ) -> io.NodeOutput:
        value = _packet(packet)
        info = inspect_packet(value)
        geometry = info.get("geometry", {})
        try:
            scene_cuts = json.loads(scene_cuts_json or "[]")
        except json.JSONDecodeError as exc:
            raise MMH3ResourceError(f"scene_cuts_json is invalid: {exc}") from exc
        if not isinstance(scene_cuts, list) or not all(isinstance(item, int) for item in scene_cuts):
            raise MMH3ResourceError("scene_cuts_json must be an array of frame indices")
        total_frames = geometry.get("frames")
        fps = geometry.get("fps")
        if total_frames in (None, 0) or fps in (None, 0):
            raise MMH3ResourceError("Long-video planning requires packet frame count and FPS metadata")
        plan = plan_long_video_chunks(
            source_id=str(info.get("id") or "unknown"),
            total_frames=int(total_frames),
            fps=float(fps),
            chunk_duration_seconds=chunk_duration_seconds,
            overlap_seconds=overlap_seconds,
            boundary_mode=boundary_mode,
            scene_cuts=scene_cuts,
            scene_search_seconds=scene_search_seconds,
            min_tail_seconds=min_tail_seconds,
            min_tail_policy=min_tail_policy,
            audio_sample_rate=audio_sample_rate,
        )
        plan_dict = plan.to_dict()
        estimate = build_prequeue_estimate(
            plan_dict,
            operation="long_video_chunking",
            effective_settings=plan.settings,
        )
        return io.NodeOutput(
            value,
            json.dumps(plan_dict, ensure_ascii=False, indent=2),
            len(plan.chunks),
            plan.summary(),
            json.dumps(estimate, ensure_ascii=False, indent=2),
        )


class MMH3LongAudioTimelinePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3LongAudioTimelinePlan",
            display_name="MMH3 H3 Audio Master Timeline",
            category=CATEGORY,
            description=(
                "Plan legal 17n+5 H3 generation windows directly from the immutable master audio. "
                "Use this for image-reference singing/performance generation without a source video."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Float.Input("generation_duration_seconds", default=8.0, min=5.0, max=15.1, step=0.1),
                io.Float.Input("context_seconds", default=0.5, min=0.0, max=4.0, step=0.1),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("plan_json"), io.Int.Output("chunks"), io.String.Output("status"), io.String.Output("estimate_json")],
        )

    @classmethod
    def execute(cls, packet, generation_duration_seconds: float, context_seconds: float) -> io.NodeOutput:
        value = _packet(packet)
        info = inspect_packet(value)
        audio = value.get_primary("audio")
        if audio is None:
            raise MMH3ResourceError("Audio-master timeline requires an explicit primary audio resource")
        facts = resource_facts(audio)
        sample_rate = int(facts.get("sample_rate") or 0)
        samples = int(facts.get("samples") or 0)
        if sample_rate != 32000:
            raise MMH3ResourceError(f"H3 audio-master timeline requires 32000 Hz source audio; got {sample_rate or 'unknown'}")
        if samples < 1:
            raise MMH3ResourceError("Audio-master timeline requires exact source audio sample-count metadata")
        plan = plan_h3_audio_timeline_chunks(
            source_id=str(info.get("id") or "unknown"),
            total_audio_samples=samples,
            audio_sample_rate=sample_rate,
            generation_duration_seconds=float(generation_duration_seconds),
            context_seconds=float(context_seconds),
        )
        plan_dict = plan.to_dict()
        estimate = build_prequeue_estimate(plan_dict, operation="long_video_lipsync", effective_settings=plan.settings)
        return io.NodeOutput(
            value,
            json.dumps(plan_dict, ensure_ascii=False, indent=2),
            len(plan.chunks),
            f"READY · H3 audio master · chunks={len(plan.chunks)} · window={plan.settings['generation_frames']}f · master={samples/sample_rate:.3f}s",
            json.dumps(estimate, ensure_ascii=False, indent=2),
        )


class MMH3InteractiveAudioTimelinePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3InteractiveAudioTimelinePlan",
            display_name="MMH3 F18 Interactive Music Timeline",
            category=CATEGORY,
            description=(
                "Start a human-in-the-loop F18 music-video timeline. The first delivered shot can use any "
                "duration; its larger H3 17n+5 generation window and exact master-audio slice are derived automatically."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Float.Input("first_shot_duration_seconds", default=5.0, min=0.1, max=15.0, step=0.1),
                io.Float.Input("context_seconds", default=0.5, min=0.0, max=4.0, step=0.1),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("plan_json"), io.String.Output("status"), io.String.Output("estimate_json")],
        )

    @classmethod
    def execute(cls, packet, first_shot_duration_seconds: float, context_seconds: float) -> io.NodeOutput:
        value = _packet(packet)
        info = inspect_packet(value)
        audio = value.get_primary("audio")
        if audio is None:
            raise MMH3ResourceError("Interactive F18 timeline requires an explicit primary master audio resource")
        facts = resource_facts(audio)
        sample_rate = int(facts.get("sample_rate") or 0)
        samples = int(facts.get("samples") or 0)
        if sample_rate != 32000 or samples < 1:
            raise MMH3ResourceError("Interactive F18 timeline requires exact 32000 Hz master-audio sample metadata")
        plan = plan_h3_audio_interactive_sequence(
            source_id=str(info.get("id") or "unknown"),
            total_audio_samples=samples,
            audio_sample_rate=sample_rate,
            first_shot_duration_seconds=float(first_shot_duration_seconds),
            context_seconds=float(context_seconds),
        )
        plan_dict = plan.to_dict()
        first = plan_dict["chunks"][0]
        estimate = build_prequeue_estimate(plan_dict, operation="long_video_lipsync", effective_settings=plan.settings)
        return io.NodeOutput(
            value,
            json.dumps(plan_dict, ensure_ascii=False, indent=2),
            (
                f"READY · F18 interactive · shot={first['actual_shot_duration_seconds']:.3f}s "
                f"H3={first['generation_frames']}f · cursor={first['audio_write_end']/sample_rate:.3f}s"
            ),
            json.dumps(estimate, ensure_ascii=False, indent=2),
        )


class MMH3LongVideoUpscaleSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3LongVideoUpscaleSettings",
            display_name="MMH3 Long Video Upscale Settings",
            category=CATEGORY,
            description="Freeze one target canvas and optimization profile for every chunk before queueing long-video F07 upscale.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("chunk_plan_json", default="{}", multiline=True),
                io.Combo.Input("geometry_mode", options=["scale", "target_megapixels", "target_dimensions"], default="scale"),
                io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05),
                io.Int.Input("target_width", default=0, min=0, max=4096, step=32, advanced=True),
                io.Int.Input("target_height", default=0, min=0, max=4096, step=32, advanced=True),
                io.Float.Input("target_megapixels", default=2.1, min=0.1, max=8.0, step=0.1),
                io.Int.Input("align", default=32, min=32, max=512, step=32, advanced=True),
                io.String.Input("optimization_profile_json", default="{}", multiline=True, advanced=True),
            ],
            outputs=[io.String.Output("settings_json"), io.Int.Output("target_width"), io.Int.Output("target_height"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, packet, chunk_plan_json: str, geometry_mode: str, scale: float, target_width: int, target_height: int, target_megapixels: float, align: int, optimization_profile_json: str) -> io.NodeOutput:
        value = _packet(packet)
        info = inspect_packet(value)
        geometry = info.get("geometry", {})
        settings = build_long_video_upscale_settings(
            source_id=str(info.get("id") or ""),
            source_width=int(geometry.get("width") or 0),
            source_height=int(geometry.get("height") or 0),
            fps=float(geometry.get("fps") or 0),
            chunk_plan=_json_object(chunk_plan_json, "chunk_plan_json"),
            geometry_mode=geometry_mode,
            scale=float(scale),
            target_width=int(target_width),
            target_height=int(target_height),
            target_megapixels=float(target_megapixels),
            align=int(align),
            optimization_profile=_json_object(optimization_profile_json, "optimization_profile_json"),
        )
        target = settings["target"]
        return io.NodeOutput(json.dumps(settings, ensure_ascii=False, indent=2), int(target["width"]), int(target["height"]), f"READY · long-video upscale · {target['width']}x{target['height']}")


class MMH3LongVideoAudioSyncSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3LongVideoAudioSyncSettings",
            display_name="MMH3 Long Video H3 Audio Sync Settings",
            category=CATEGORY,
            description="Freeze native H3 audio-driven or lipsync behavior while keeping source audio immutable for final assembly.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("chunk_plan_json", default="{}", multiline=True),
                io.Combo.Input("mode", options=list(AUDIO_SYNC_MODES), default="lipsync"),
                io.Combo.Input("source_mode", options=["auto", "video_reference", "image_reference"], default="auto", optional=True),
                io.Boolean.Input("require_candidate_review", default=True, optional=True),
                io.Combo.Input("audio_output_mode", options=list(AUDIO_OUTPUT_MODES), default="master_only", optional=True),
                io.Float.Input("master_gain_db", default=0.0, min=-120.0, max=24.0, step=0.5, advanced=True, optional=True),
                io.Float.Input("generated_gain_db", default=-18.0, min=-120.0, max=24.0, step=0.5, advanced=True, optional=True),
                io.Boolean.Input("peak_limit", default=True, advanced=True, optional=True),
                io.String.Input("tracks_json", default="[]", multiline=True),
            ],
            outputs=[io.String.Output("settings_json"), io.Int.Output("tracks"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, packet, chunk_plan_json: str, mode: str, tracks_json: str, source_mode: str = "auto", require_candidate_review: bool = False, audio_output_mode: str = "master_only", master_gain_db: float = 0.0, generated_gain_db: float = -18.0, peak_limit: bool = True) -> io.NodeOutput:
        value = _packet(packet)
        info = inspect_packet(value)
        chunk_plan = _json_object(chunk_plan_json, "chunk_plan_json")
        try:
            tracks = json.loads(tracks_json)
        except json.JSONDecodeError as exc:
            raise MMH3ResourceError(f"tracks_json is invalid: {exc}") from exc
        if not isinstance(tracks, list) or not all(isinstance(item, dict) for item in tracks):
            raise MMH3ResourceError("tracks_json must be an array of track objects")
        audio = value.get_primary("audio")
        if audio is None:
            raise MMH3ResourceError("H3 audio sync requires an explicit primary audio resource")
        audio_facts = resource_facts(audio)
        audio_rate = int(audio_facts.get("sample_rate") or 0)
        planned_rate = int(chunk_plan.get("audio_sample_rate") or 0)
        if audio_rate and planned_rate and audio_rate != planned_rate:
            raise MMH3ResourceError(f"H3 audio-sync rate mismatch: packet={audio_rate} Hz plan={planned_rate} Hz")
        settings = build_long_video_audio_sync_settings(
            source_id=str(info.get("id") or ""),
            source_archive=str(value.source_archive or ""),
            source_audio_resource_id=str(audio["id"]),
            source_audio_revision=str((audio.get("content") or {}).get("revision") or (audio.get("content") or {}).get("digest") or "unknown"),
            chunk_plan=chunk_plan,
            tracks=tracks,
            mode=mode,
            source_channels=int(audio_facts.get("channels") or audio_facts.get("channel_count") or 0),
            source_audio_samples=int(audio_facts.get("samples") or 0),
            source_mode=source_mode,
            require_candidate_review=bool(require_candidate_review),
            audio_output_mode=audio_output_mode,
            master_gain_db=float(master_gain_db),
            generated_gain_db=float(generated_gain_db),
            peak_limit=bool(peak_limit),
        )
        return io.NodeOutput(
            json.dumps(settings, ensure_ascii=False, indent=2),
            len(settings["tracks"]),
            f"READY · H3 {mode} · audio={settings['delivery_policy']['audio_output_mode']} · tracks={len(settings['tracks'])} · channels={settings['source_audio']['channels'] or 'unknown'}",
        )


class MMH3H3AudioSyncProof(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3AudioSyncProof",
            display_name="MMH3 H3 Audio Sync Proof",
            category=CATEGORY,
            description="Record frozen native-H3 conditioning provenance for audio-driven/lipsync chunks before packing the result.",
            inputs=[
                io.String.Input("settings_json", default="{}", multiline=True),
                io.String.Input("job_json", default="{}", multiline=True),
            ],
            outputs=[io.String.Output("process_info_json"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, settings_json: str, job_json: str) -> io.NodeOutput:
        settings = _json_object(settings_json, "settings_json")
        job = _json_object(job_json, "job_json")
        proof = build_h3_audio_sync_chunk_proof(settings, job)
        process_info = {"audio_sync": proof, "lipsync": proof}
        validate_lipsync_chunk_proof(settings, job, process_info)
        return io.NodeOutput(
            json.dumps(process_info, ensure_ascii=False, indent=2),
            f"READY · native H3 {proof['mode']} proof · job={job.get('job_id', '?')}",
        )


class MMH3AutomationAudioChunk(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3AutomationAudioChunk",
            display_name="MMH3 Automation Audio-Master Chunk",
            category=CATEGORY,
            description=(
                "Trim one immutable master-audio slice for an H3 generation job, right-pad only the legal "
                "conditioning tail when required, and preserve packet image/reference resources."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("plan_json", default="{}", multiline=True),
                io.Int.Input("chunk_index", default=0, min=0, max=999999),
                io.String.Input("job_id", default="", advanced=True),
                io.String.Input("settings_json", default="{}", multiline=True, optional=True, advanced=True),
            ],
            outputs=[MMH3.Output("packet"), io.Audio.Output("audio"), io.String.Output("context_json"), io.Int.Output("generation_frames"), io.String.Output("status")],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, packet, plan_json: str, chunk_index: int, job_id: str, settings_json: str = "{}") -> io.NodeOutput:
        value = _packet(packet)
        plan = _json_object(plan_json, "plan_json")
        settings = _json_object(settings_json, "settings_json")
        if settings:
            validate_audio_sync_source(value, settings)
        if str(plan.get("source_id")) != str(value.manifest.get("id")):
            raise MMH3ResourceError("Chunk plan source_id does not match the input packet")
        if value.get_primary("audio") is None:
            raise MMH3ResourceError("Audio-master chunk requires an explicit primary audio resource")
        planner = str((plan.get("settings") or {}).get("planner") or "")
        if planner not in H3_AUDIO_TIMELINE_PLANNERS:
            raise MMH3ResourceError("Audio-master chunk requires an H3 audio timeline or interactive plan")
        selection = resolve_chunk_execution(plan, chunk_index=int(chunk_index), job_id=job_id)
        expansion = build_audio_timeline_chunk_expansion(selection, packet=value)
        return io.NodeOutput(expansion.packet, expansion.audio, expansion.context_json, expansion.generation_frames, expansion.status, expand=expansion.graph)


class MMH3AutomationVideoChunk(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3AutomationVideoChunk",
            display_name="MMH3 Automation Video Chunk",
            category=CATEGORY,
            description="Lazily trim one planned VIDEO chunk and its optional AUDIO, then return a self-describing MMH3 packet.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("plan_json", default="{}", multiline=True),
                io.Int.Input("chunk_index", default=0, min=0, max=999999),
                io.String.Input("job_id", default="", advanced=True, tooltip="Stable selector; when set, overrides chunk_index."),
                io.String.Input("settings_json", default="{}", multiline=True, optional=True, advanced=True),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Video.Output("video"),
                io.Audio.Output("audio"),
                io.String.Output("context_json"),
                io.String.Output("status"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, packet, plan_json: str, chunk_index: int, job_id: str, settings_json: str = "{}") -> io.NodeOutput:
        value = _packet(packet)
        plan = _json_object(plan_json, "plan_json")
        settings = _json_object(settings_json, "settings_json")
        if settings:
            validate_audio_sync_source(value, settings)
        if str(plan.get("source_id")) != str(value.manifest.get("id")):
            raise MMH3ResourceError("Chunk plan source_id does not match the input packet")
        if value.get_primary("video") is None:
            raise MMH3ResourceError("Automation video chunk requires an explicit primary video resource")
        selection = resolve_chunk_execution(plan, chunk_index=int(chunk_index), job_id=job_id)
        expansion = build_video_chunk_expansion(
            selection,
            packet=value,
            has_audio=value.get_primary("audio") is not None,
        )
        return io.NodeOutput(
            expansion.packet,
            expansion.video,
            expansion.audio,
            expansion.context_json,
            expansion.status,
            expand=expansion.graph,
        )


class MMH3TrimAudioSamples(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3TrimAudioSamples",
            display_name="MMH3 Trim Audio Samples",
            category=CATEGORY,
            description="Trim AUDIO by exact absolute PCM sample boundaries; avoids independent time-rounding drift.",
            inputs=[
                io.Audio.Input("audio"),
                io.Int.Input("start_sample", default=0, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Int.Input("end_sample", default=1, min=1, max=0x7FFFFFFFFFFFFFFF),
            ],
            outputs=[io.Audio.Output("audio")],
        )

    @classmethod
    def execute(cls, audio, start_sample: int, end_sample: int) -> io.NodeOutput:
        try:
            trimmed = trim_audio_samples(audio, start_sample, end_sample)
        except ValueError as exc:
            raise MMH3ResourceError(str(exc)) from exc
        return io.NodeOutput(trimmed)


class MMH3TrimAudioSamplesPadded(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3TrimAudioSamplesPadded",
            display_name="MMH3 Trim Audio Samples + EOF Pad",
            category=CATEGORY,
            description="Sample-exact audio trim that right-pads only beyond EOF; intended for legal H3 conditioning windows.",
            inputs=[
                io.Audio.Input("audio"),
                io.Int.Input("start_sample", default=0, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Int.Input("end_sample", default=1, min=1, max=0x7FFFFFFFFFFFFFFF),
            ],
            outputs=[io.Audio.Output("audio")],
        )

    @classmethod
    def execute(cls, audio, start_sample: int, end_sample: int) -> io.NodeOutput:
        try:
            trimmed = trim_audio_samples_padded(audio, start_sample, end_sample)
        except ValueError as exc:
            raise MMH3ResourceError(str(exc)) from exc
        return io.NodeOutput(trimmed)


def _json_object(value: str, label: str) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise MMH3ResourceError(f"{label} is invalid: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MMH3ResourceError(f"{label} must be a JSON object")
    return parsed


class MMH3AutomationLedger(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3AutomationLedger",
            display_name="MMH3 Automation Ledger",
            category=CATEGORY,
            description="Acquire/commit resumable jobs with attempt leases, deterministic save prefixes and stale-worker protection.",
            inputs=[
                io.String.Input("plan_json", default="{}", multiline=True),
                io.String.Input("ledger_json", default="{}", multiline=True, advanced=True),
                io.Combo.Input("action", options=["initialize", "inspect", "acquire_next", "append_next", "finalize_sequence", "set_audio_delivery", "start", "commit_artifact", "commit_candidate", "accept_candidate", "reroll", "complete", "fail", "cancel", "reset"], default="initialize"),
                io.String.Input("job_id", default="", advanced=True),
                io.String.Input("lease_id", default="", advanced=True),
                io.String.Input("operation", default="process"),
                io.Combo.Input("error_policy", options=["stop_on_error", "continue_on_error"], default="stop_on_error"),
                io.String.Input("effective_settings_json", default="{}", multiline=True, advanced=True),
                io.String.Input("artifact_json", default="{}", multiline=True, advanced=True),
                io.String.Input("packet_path", default="", advanced=True, tooltip="Existing .mmh3 path returned by MMH3 Save for commit_artifact."),
                io.String.Input("packet_id", default="", advanced=True),
                io.String.Input("candidate_id", default="", advanced=True, optional=True, tooltip="Candidate to accept when action=accept_candidate."),
                io.String.Input("candidate_label", default="", advanced=True, optional=True),
                io.String.Input("candidate_notes", default="", multiline=True, advanced=True, optional=True),
                io.Combo.Input("audio_output_mode", options=list(AUDIO_OUTPUT_MODES), default="master_only", advanced=True, optional=True),
                io.Float.Input("master_gain_db", default=0.0, min=-120.0, max=24.0, step=0.5, advanced=True, optional=True),
                io.Float.Input("generated_gain_db", default=-18.0, min=-120.0, max=24.0, step=0.5, advanced=True, optional=True),
                io.Boolean.Input("peak_limit", default=True, advanced=True, optional=True),
                io.Float.Input("next_shot_duration_seconds", default=5.0, min=0.1, max=15.0, step=0.1, advanced=True, optional=True),
                io.Float.Input("next_context_seconds", default=0.5, min=0.0, max=4.0, step=0.1, advanced=True, optional=True),
                io.String.Input("error", default="", multiline=True, advanced=True),
                io.Combo.Input("resume_mode", options=["pending_and_failed", "failed_only", "include_cancelled"], default="pending_and_failed"),
            ],
            outputs=[
                io.String.Output("ledger_json"),
                io.String.Output("resume_job_ids_json"),
                io.String.Output("status"),
                io.Int.Output("revision"),
                io.String.Output("active_job_id"),
                io.String.Output("active_lease_id"),
                io.String.Output("filename_prefix"),
                io.String.Output("job_json"),
                io.String.Output("estimate_json"),
                io.String.Output("estimate_summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        plan_json: str,
        ledger_json: str,
        action: str,
        job_id: str,
        lease_id: str,
        operation: str,
        error_policy: str,
        effective_settings_json: str,
        artifact_json: str,
        packet_path: str,
        packet_id: str,
        error: str,
        resume_mode: str,
        candidate_id: str = "",
        candidate_label: str = "",
        candidate_notes: str = "",
        audio_output_mode: str = "master_only",
        master_gain_db: float = 0.0,
        generated_gain_db: float = -18.0,
        peak_limit: bool = True,
        next_shot_duration_seconds: float = 5.0,
        next_context_seconds: float = 0.5,
    ) -> io.NodeOutput:
        active: dict = {}
        if action == "initialize":
            ledger = create_execution_ledger(
                _json_object(plan_json, "plan_json"),
                operation=operation,
                effective_settings=_json_object(effective_settings_json, "effective_settings_json"),
                error_policy=error_policy,
            )
        else:
            ledger = _json_object(ledger_json, "ledger_json")
            if action == "acquire_next":
                ledger, lease = acquire_next_execution_job(ledger, mode=resume_mode)
                active = lease or {}
            elif action == "append_next":
                ledger = append_interactive_audio_try(
                    ledger,
                    shot_duration_seconds=float(next_shot_duration_seconds),
                    context_seconds=float(next_context_seconds),
                )
            elif action == "finalize_sequence":
                ledger = finalize_interactive_audio_sequence(ledger)
            elif action == "set_audio_delivery":
                ledger = set_execution_audio_delivery_policy(
                    ledger, audio_output_mode=audio_output_mode, master_gain_db=float(master_gain_db),
                    generated_gain_db=float(generated_gain_db), peak_limit=bool(peak_limit),
                )
            elif action == "commit_artifact":
                ledger = commit_execution_artifact(
                    ledger,
                    job_id.strip(),
                    lease_id=lease_id.strip(),
                    packet_path=packet_path.strip(),
                    packet_id=packet_id.strip(),
                )
            elif action == "commit_candidate":
                ledger = commit_execution_candidate(
                    ledger,
                    job_id.strip(),
                    lease_id=lease_id.strip(),
                    packet_path=packet_path.strip(),
                    packet_id=packet_id.strip(),
                    label=candidate_label,
                    notes=candidate_notes,
                )
            elif action == "accept_candidate":
                ledger = accept_execution_candidate(ledger, job_id.strip(), candidate_id.strip())
            elif action != "inspect":
                artifact = _json_object(artifact_json, "artifact_json") if action == "complete" else None
                ledger = transition_execution_job(
                    ledger,
                    job_id.strip(),
                    action,
                    artifact=artifact,
                    error=error,
                    lease_id=lease_id.strip(),
                )
            if not active:
                running = [item for item in ledger["jobs"] if item["state"] == "running"]
                if len(running) == 1:
                    item = running[0]
                    active = {
                        "job_id": item["job_id"],
                        "lease_id": item["lease_id"],
                        "filename_prefix": "",
                        "job": item["job"],
                    }
        resume_ids = select_resume_jobs(ledger, mode=resume_mode)
        status = f"READY · automation ledger · status={ledger['status']} resume_jobs={len(resume_ids)}"
        first_job_id = ledger["jobs"][0]["job_id"] if ledger.get("jobs") else ""
        prefix_root = ""
        if first_job_id:
            prefix_root = deterministic_artifact_prefix(ledger, first_job_id).rsplit("/", 1)[0]
        estimate = build_prequeue_estimate(
            ledger["plan"],
            operation=str(ledger.get("operation") or "unknown"),
            effective_settings=ledger.get("effective_settings") or {},
            output_prefix=prefix_root,
            error_policy=str(ledger.get("error_policy") or ""),
        )
        estimate_text = json.dumps(estimate, ensure_ascii=False, indent=2)
        estimate_status = prequeue_summary(estimate)
        return io.NodeOutput(
            json.dumps(ledger, ensure_ascii=False, indent=2),
            json.dumps(list(resume_ids), ensure_ascii=False),
            status,
            int(ledger["revision"]),
            str(active.get("job_id") or ""),
            str(active.get("lease_id") or ""),
            str(active.get("filename_prefix") or ""),
            json.dumps(active.get("job") or {}, ensure_ascii=False, indent=2),
            estimate_text,
            estimate_status,
        )


class MMH3AutomationReport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3AutomationReport",
            display_name="MMH3 Automation Report",
            category=CATEGORY,
            description="Build a read-only post-run summary with artifact integrity, timeline diagnostics and a safe next-action recommendation.",
            inputs=[io.String.Input("ledger_json", default="{}", multiline=True)],
            outputs=[
                io.String.Output("summary_json"),
                io.String.Output("recommendation"),
                io.Boolean.Output("assembly_ready"),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(cls, ledger_json: str) -> io.NodeOutput:
        ledger = _json_object(ledger_json, "ledger_json")
        summary = build_execution_summary(ledger)
        counts = summary["counts"]
        status = (
            "REPORT · "
            f"completed={counts['completed']} review={counts.get('review', 0)} failed={counts['failed']} cancelled={counts['cancelled']} "
            f"pending={counts['pending']} running={counts['running']} "
            f"assembly={'ready' if summary['assembly']['ready'] else 'blocked'} "
            f"next={summary['recommendation']}"
        )
        return io.NodeOutput(
            json.dumps(summary, ensure_ascii=False, indent=2),
            str(summary["recommendation"]),
            bool(summary["assembly"]["ready"]),
            status,
        )


class MMH3AutomationCheckpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3AutomationCheckpoint",
            display_name="MMH3 Automation Checkpoint",
            category=CATEGORY,
            description="Atomically overwrite one resumable ledger JSON inside the ComfyUI output directory.",
            inputs=[
                io.String.Input("ledger_json", default="{}", multiline=True),
                io.String.Input("filename_prefix", default="mmh3_automation/checkpoints/ledger"),
            ],
            outputs=[io.String.Output("ledger_json"), io.String.Output("path"), io.String.Output("status")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, ledger_json: str, filename_prefix: str) -> io.NodeOutput:
        ledger = _json_object(ledger_json, "ledger_json")
        relative = _safe_output_prefix(filename_prefix)
        target = (Path(folder_paths.get_output_directory()) / relative).with_suffix(".json").resolve()
        output_root = Path(folder_paths.get_output_directory()).resolve()
        try:
            target.relative_to(output_root)
        except ValueError as exc:
            raise MMH3ResourceError("Checkpoint path escapes the ComfyUI output directory") from exc
        saved = save_execution_ledger(ledger, target)
        return io.NodeOutput(
            json.dumps(ledger, ensure_ascii=False, indent=2),
            str(saved),
            f"SAVED · automation checkpoint · revision={ledger['revision']}",
        )


class MMH3AutomationAssemblyGate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3AutomationAssemblyGate",
            display_name="MMH3 Automation Assembly Gate",
            category=CATEGORY,
            description="Fail-closed final assembly map for completed contiguous long-video chunks.",
            inputs=[io.String.Input("ledger_json", default="{}", multiline=True)],
            outputs=[io.String.Output("assembly_map_json"), io.Int.Output("segments"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, ledger_json: str) -> io.NodeOutput:
        assembly = build_chunk_assembly_map(_json_object(ledger_json, "ledger_json"))
        count = len(assembly["segments"])
        return io.NodeOutput(
            json.dumps(assembly, ensure_ascii=False, indent=2),
            count,
            f"READY · chunk assembly · segments={count} frames={assembly['output_frames']}",
        )


class MMH3AutomationAssembleChunks(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3AutomationAssembleChunks",
            display_name="MMH3 Automation Assemble Chunks",
            category=CATEGORY,
            description="Verify committed chunk archives, remove read overlap by ownership and build one bounded-streaming VIDEO with exact PCM.",
            inputs=[io.String.Input("assembly_map_json", default="{}", multiline=True)],
            outputs=[
                MMH3.Output("packet"),
                io.Video.Output("video"),
                io.Audio.Output("audio"),
                io.String.Output("report_json"),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(cls, assembly_map_json: str) -> io.NodeOutput:
        result = assemble_chunk_packets(_json_object(assembly_map_json, "assembly_map_json"))
        count = len(result.report["segments"])
        return io.NodeOutput(
            result.packet,
            result.video,
            result.audio,
            json.dumps(result.report, ensure_ascii=False, indent=2),
            f"READY · bounded assembly · chunks={count} frames={result.report['output_frames']}",
        )


__all__ = [
    "MMH3AutomationAssemblyGate",
    "MMH3AutomationAssembleChunks",
    "MMH3AutomationCheckpoint",
    "MMH3AutomationLedger",
    "MMH3AutomationAudioChunk",
    "MMH3AutomationVideoChunk",
    "MMH3BatchInputPlan",
    "MMH3BatchPreQueueEstimate",
    "MMH3BatchNormalizeImport",
    "MMH3BatchStitch",
    "MMH3LongAudioTimelinePlan",
    "MMH3InteractiveAudioTimelinePlan",
    "MMH3LongVideoChunkPlan",
    "MMH3LongVideoUpscaleSettings",
    "MMH3LongVideoAudioSyncSettings",
    "MMH3H3AudioSyncProof",
    "MMH3TrimAudioSamples",
    "MMH3TrimAudioSamplesPadded",
]
