from __future__ import annotations

from .h3_contract import h3_latent_contract_from_resource, validate_h3_latent_contract

from .node_support import (
    CATEGORY,
    H3_LATENT_ORIGINS,
    MMH3,
    MMH3ResourceError,
    _packet,
    build_h3_continuation_handover,
    concat_h3_av_latent,
    deep_copy_json,
    encode_decoded_prefix,
    get_resource_payload,
    io,
    json,
    prepare_decoded_prefix,
    split_h3_av_latent,
)

class MMH3H3AudioVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3AudioVAE", display_name="H3 Audio VAE · Preserve Onset", category=CATEGORY,
            description="Prevent generic center cropping before H3 audio encoding. Uses a private VAE wrapper; source PCM and shared loader state are unchanged.",
            inputs=[io.Vae.Input("audio_vae")], outputs=[io.Vae.Output("audio_vae")],
        )

    @classmethod
    def execute(cls, audio_vae) -> io.NodeOutput:
        from .audio_vae import preserve_h3_audio_onset
        return io.NodeOutput(preserve_h3_audio_onset(audio_vae))


class MMH3H3ContinuationGuide(io.ComfyNode):
    """Optional endpoint conditioning, independent of the preserved AV prefix."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3ContinuationGuide", display_name="Continuation End Frame", category=CATEGORY,
            description="Optionally guide the end of a continuation. Without an image, conditioning is unchanged.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent"),
                io.Vae.Input("video_vae"),
                io.Image.Input("last_frame", optional=True, tooltip="Optional endpoint. Only the first image of a batch is used."),
            ],
            outputs=[io.Conditioning.Output("positive")],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, positive, latent, video_vae, last_frame=None) -> io.NodeOutput:
        if last_frame is None:
            return io.NodeOutput(positive)
        from comfy_execution.graph_utils import GraphBuilder

        graph = GraphBuilder()
        guide = graph.node("MiniMaxH3AddGuide", positive=positive, latent=latent,
                           vae=video_vae, image=last_frame[:1], frame_idx=-1)
        return io.NodeOutput(guide.out(0), expand=graph.finalize())


class MMH3H3AVSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3AVSeparate",
            display_name="MMH3 H3 AV Separate",
            category=CATEGORY,
            description="Strictly split a MiniMax H3 joint AV latent into video and audio branches. Preserves auxiliary LATENT fields and split noise masks without copying tensor storage.",
            inputs=[io.Latent.Input("av_latent")],
            outputs=[
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, av_latent) -> io.NodeOutput:
        video_latent, audio_latent = split_h3_av_latent(av_latent)
        return io.NodeOutput(video_latent, audio_latent)


class MMH3H3AVCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3AVCombine",
            display_name="MMH3 H3 AV Combine",
            category=CATEGORY,
            description="Combine separated MiniMax H3 video/audio latents into one joint AV latent. Rejects batch/layout/duration mismatches instead of silently trimming or padding audio.",
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        return io.NodeOutput(concat_h3_av_latent(video_latent, audio_latent))


class MMH3H3Provenance(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3Provenance",
            display_name="MMH3 H3 Provenance",
            category=CATEGORY,
            description="Declare how the stored H3 AV latent was produced without loading or rewriting the latent payload.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input(
                    "origin",
                    options=list(H3_LATENT_ORIGINS),
                    default="sampler_output",
                    tooltip="Use sampler_output only when you know this is the direct output of an H3 sampler. This changes metadata only.",
                ),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("status")],
        )

    @classmethod
    def execute(cls, packet, origin: str) -> io.NodeOutput:
        packet = _packet(packet)
        res = packet.get_primary("latent")
        if res is None:
            raise MMH3ResourceError("Packet has no primary H3 latent")
        contract = h3_latent_contract_from_resource(res)
        contract["origin"] = origin
        contract = validate_h3_latent_contract(contract)
        extensions = deep_copy_json(res.get("extensions", {}))
        extensions.setdefault("minimax_h3", {})["latent"] = contract
        out = packet.update_resource(res["id"], extensions=extensions)
        binding = contract["binding"]
        status = f"origin={origin}; geometry={binding['geometry']}; time={binding['time']}"
        return io.NodeOutput(out, status)


class MMH3H3Compatibility(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3Compatibility",
            display_name="MMH3 H3 Compatibility",
            category=CATEGORY,
            description="Manifest-only structural compatibility check for two H3 AV packets before a future seam/stitch adapter. Never approves naive latent concatenation.",
            inputs=[MMH3.Input("packet_a"), MMH3.Input("packet_b")],
            outputs=[
                io.Boolean.Output("seam_compatible"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(cls, packet_a, packet_b) -> io.NodeOutput:
        left = _packet(packet_a).get_primary("latent")
        right = _packet(packet_b).get_primary("latent")
        reasons: list[str] = []
        warnings: list[str] = []
        if left is None or right is None:
            reasons.append("both packets require a primary H3 latent")
        else:
            a = h3_latent_contract_from_resource(left)
            b = h3_latent_contract_from_resource(right)
            if a["canvas"] != b["canvas"]:
                reasons.append("canvas geometry differs")
            for key in ("fps", "audio_sample_rate", "audio_latent_rate"):
                if a["timeline"].get(key) != b["timeline"].get(key):
                    reasons.append(f"timeline {key} differs")
            if a["binding"]["geometry"] != "bound" or b["binding"]["geometry"] != "bound":
                reasons.append("geometry binding is not bound")
            if a["binding"]["time"] != "bound" or b["binding"]["time"] != "bound":
                reasons.append("time binding is not bound")
            if a.get("origin") != "sampler_output" or b.get("origin") != "sampler_output":
                warnings.append("one or both latents are not direct sampler outputs")
        result = {
            "compatible": not reasons,
            "reasons": reasons,
            "warnings": warnings,
            "naive_latent_concat_safe": False,
        }
        summary = (
            "H3 seam source compatibility: OK. Naive latent concatenation remains unsafe."
            if result["compatible"] else
            "H3 seam source compatibility: NO — " + "; ".join(reasons)
        )
        if warnings:
            summary += " Warnings: " + " | ".join(warnings)
        return io.NodeOutput(bool(result["compatible"]), summary, json.dumps(result, ensure_ascii=False, indent=2))


class MMH3H3ContinuationHandover(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3ContinuationHandover",
            display_name="MMH3 H3 Continuation Handover",
            category=CATEGORY,
            description=(
                "Build a fresh joint AV sampling target from the tail of a direct H3 sampler output. "
                "Creates native per-stream noise masks; never concatenates latents."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Int.Input("target_frames", default=124, min=5, max=3600, step=17),
                io.Int.Input(
                    "video_handover_frames",
                    default=39,
                    min=39,
                    max=3600,
                    step=51,
                    tooltip="Exact synchronized boundaries are 39, 90, 141, 192, ... frames.",
                ),
                io.Int.Input(
                    "audio_handover_frames",
                    default=0,
                    min=0,
                    max=3600,
                    step=3,
                    advanced=True,
                    tooltip="0 follows video_handover_frames; otherwise an exact 40 Hz boundary in pixel frames.",
                ),
                io.Int.Input(
                    "audio_feather_frames",
                    default=0,
                    min=0,
                    max=3600,
                    step=3,
                    advanced=True,
                    tooltip="Half-cosine denoise ramp inside the copied audio prefix.",
                ),
                io.Int.Input("target_width", default=0, min=0, optional=True, force_input=True,
                             tooltip="Connect Video Settings width. Direct continuation must keep source resolution."),
                io.Int.Input("target_height", default=0, min=0, optional=True, force_input=True,
                             tooltip="Connect Video Settings height. Direct continuation must keep source resolution."),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Latent.Output("latent"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        target_frames: int,
        video_handover_frames: int,
        audio_handover_frames: int,
        audio_feather_frames: int,
        target_width: int = 0,
        target_height: int = 0,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        resource = packet.get_primary("latent")
        if resource is None:
            raise MMH3ResourceError("Packet has no primary H3 latent for F02 continuation")
        contract = h3_latent_contract_from_resource(resource)
        origin = str(contract.get("origin", "unknown"))
        if origin != "sampler_output":
            raise MMH3ResourceError(
                f"F02 continuation requires H3 latent origin='sampler_output'; got {origin!r}"
            )
        source_latent = get_resource_payload(packet, resource)
        result = build_h3_continuation_handover(
            source_latent,
            source_origin=origin,
            target_frames=int(target_frames),
            video_handover_frames=int(video_handover_frames),
            audio_handover_frames=None if int(audio_handover_frames) == 0 else int(audio_handover_frames),
            audio_feather_frames=int(audio_feather_frames),
            target_width=int(target_width),
            target_height=int(target_height),
        )
        process = packet.manifest.get("extensions", {}).get("mmh3_media", {}).get("last_process", {})
        process = process if isinstance(process, dict) else {}
        process_info = process.get("info") if isinstance(process.get("info"), dict) else {}
        source_mode = str(process.get("mode") or packet.manifest.get("generation", {}).get("task") or "")
        source_task_family = str(process_info.get("task_family") or "")
        if not source_task_family:
            if source_mode == "ref2va":
                source_task_family = "ref2va"
            elif source_mode in {"t2va", "i2va", "l2va", "fl2va"}:
                source_task_family = "fl2va"
            else:
                source_task_family = "unknown"
        info = result.plan.to_dict()
        info.update({
            "summary": result.plan.summary(),
            "source_packet_id": packet.manifest["id"],
            "source_latent_resource_id": resource["id"],
            "source_latent_revision": (resource.get("content") or {}).get("revision"),
            "source_process_operation": str(process.get("operation") or ""),
            "source_process_mode": source_mode,
            "source_task_family": source_task_family,
        })
        return io.NodeOutput(
            packet,
            result.latent,
            result.plan.summary(),
            json.dumps(info, ensure_ascii=False, indent=2),
        )


class MMH3H3DecodedContinuation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3DecodedContinuation",
            display_name="MMH3 H3 Decoded Continuation",
            category=CATEGORY,
            description=(
                "Build an H3 continuation target from the synchronized tail of decoded packet video/audio. "
                "Normalizes to 24 FPS, stereo 32 kHz, VAE-encodes the prefix and records the lossy round trip."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("target_frames", default=124, min=5, max=3600, step=17),
                io.Int.Input(
                    "prefix_frames",
                    default=39,
                    min=39,
                    max=3600,
                    step=51,
                    tooltip="Exact synchronized boundaries are 39, 90, 141, ... frames.",
                ),
                io.Combo.Input(
                    "missing_audio_policy",
                    options=["error", "silence"],
                    default="error",
                    tooltip="silence explicitly permits missing or too-short source audio.",
                ),
                io.Int.Input("width", default=0, min=0, max=8192, step=32, advanced=True),
                io.Int.Input("height", default=0, min=0, max=8192, step=32, advanced=True),
                io.Int.Input(
                    "audio_feather_frames",
                    default=0,
                    min=0,
                    max=3600,
                    step=3,
                    advanced=True,
                ),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.Latent.Output("latent"),
                io.String.Output("summary"),
                io.String.Output("info_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        packet,
        video_vae,
        audio_vae,
        target_frames: int,
        prefix_frames: int,
        missing_audio_policy: str,
        width: int,
        height: int,
        audio_feather_frames: int,
    ) -> io.NodeOutput:
        packet = _packet(packet)
        video_resource = packet.get_primary("video")
        if video_resource is None or video_resource.get("kind") != "video":
            raise MMH3ResourceError("Packet has no decoded primary video for F03 continuation")
        video = get_resource_payload(packet, video_resource)
        audio_resource = packet.get_primary("audio")
        audio = get_resource_payload(packet, audio_resource) if audio_resource is not None else None
        prepared = prepare_decoded_prefix(
            video,
            audio=audio,
            prefix_frames=int(prefix_frames),
            width=int(width),
            height=int(height),
            missing_audio_policy=missing_audio_policy,
        )
        result = encode_decoded_prefix(
            prepared,
            video_vae=video_vae,
            audio_vae=audio_vae,
            target_frames=int(target_frames),
            audio_feather_frames=int(audio_feather_frames),
        )
        context = result.plan.to_dict()
        context["target"] = result.continuation.plan.to_dict()
        context["source_resource_ids"] = {
            "video": video_resource["id"],
            **({"audio": audio_resource["id"]} if audio_resource is not None else {}),
        }
        existing = next((
            ref.descriptor for ref in packet.filter_resource_refs(role="auxiliary", kind="json")
            if "continuation_context" in ref.descriptor.get("tags", [])
        ), None)
        out = packet.put(
            context, kind="json", role="auxiliary", order=None,
            mode="replace" if existing else "add",
            resource_id=str(existing["id"]) if existing else "",
            tags=["continuation_context"],
            extensions={"minimax_h3": {"continuation": {
                "contract": context["contract"], "latent_origin": "vae_encoded"
            }}},
        )
        return io.NodeOutput(
            out,
            result.latent,
            result.plan.summary(),
            json.dumps(context, ensure_ascii=False, indent=2),
        )
