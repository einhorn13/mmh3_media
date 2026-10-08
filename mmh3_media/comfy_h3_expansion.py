from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .core import MMH3Media
from .audio_vae import preserve_h3_audio_onset
from .errors import MMH3ResourceError
from .resolution import ResolvedMMH3Inputs
from .runtime_contract import NativeH3Contract, require_native_h3_contract


@dataclass(frozen=True)
class H3Expansion:
    positive: Any
    latent: Any
    graph: dict[str, Any]
    contract_fingerprint: str


def _get_node(graph, node_id: str, packet: MMH3Media, descriptor, missing: str = "error"):
    return graph.node(
        node_id,
        packet=packet,
        role="auxiliary",
        order=-1,
        resource_id=descriptor.resource_id,
        missing=missing,
    )


def build_h3_expansion(
    packet: MMH3Media,
    resolved: ResolvedMMH3Inputs,
    *,
    clip: Any,
    video_vae: Any,
    audio_vae: Any = None,
    ref_image_size: str = "match",
    ref_image_short_edge: int | None = None,
    contract: NativeH3Contract | None = None,
    graph_builder_factory: Callable[[], Any] | None = None,
    reuse_conditioning: bool = False,
) -> H3Expansion:
    if not resolved.ready:
        raise MMH3ResourceError(resolved.summary())
    contract = contract or require_native_h3_contract(resolved.mode)
    if graph_builder_factory is None:
        from comfy_execution.graph_utils import GraphBuilder  # type: ignore

        graph_builder_factory = GraphBuilder
    graph = graph_builder_factory()
    if reuse_conditioning:
        inputs = dict(packet=packet, resolved=resolved, clip=clip, video_vae=video_vae,
                      ref_image_size=ref_image_size,
                      ref_image_short_edge=ref_image_short_edge if ref_image_short_edge is not None else -1)
        if audio_vae is not None:
            inputs["audio_vae"] = audio_vae
        cached = graph.node("MMH3H3ReuseConditioning", **inputs)
        return H3Expansion(cached.out(0), cached.out(1), graph.finalize(), contract.fingerprint)
    common = {
        "clip": clip,
        "vae": video_vae,
        "prompt": resolved.value("prompt"),
        "width": resolved.value("width"),
        "height": resolved.value("height"),
        "length": resolved.value("frames"),
    }

    if resolved.mode != "ref2va":
        selected = {item.usage: item for item in resolved.selected_resources if item.usage}
        if "first_frame" in selected:
            common["first_frame"] = _get_node(graph, "MMH3GetImage", packet, selected["first_frame"]).out(0)
        if "last_frame" in selected:
            common["last_frame"] = _get_node(graph, "MMH3GetImage", packet, selected["last_frame"]).out(0)
        native = graph.node(contract.node_id, **common)
        return H3Expansion(native.out(0), native.out(1), graph.finalize(), contract.fingerprint)

    encoder_only = packet.manifest.get("extensions", {}).get("minimax_h3", {}).get("reference_conditioning") == "encoder_only"
    if encoder_only:
        # Newer native contracts accept encoder-only refs. Never silently degrade on an older host.
        import nodes
        schema = nodes.NODE_CLASS_MAPPINGS[contract.node_id].define_schema()
        for name in ("vae", "audio_vae"):
            field = next((i for i in schema.inputs if i.id == name), None)
            if field is None or not field.optional:
                raise MMH3ResourceError("This H3 runtime does not support encoder-only references")
        common.pop("vae", None)
    if audio_vae is None and not encoder_only:
        raise MMH3ResourceError("Ref2VA requires audio_vae because the native H3 reference node requires that input")
    pictures = sorted((item for item in resolved.selected_resources if item.kind == "image"), key=lambda r: (r.order is None, r.order or 0))
    videos = sorted((item for item in resolved.selected_resources if item.kind == "video"), key=lambda r: (r.order is None, r.order or 0))
    audios = sorted((item for item in resolved.selected_resources if item.kind == "audio"), key=lambda r: (r.order is None, r.order or 0))
    paired_by_video: dict[str, str] = {}
    paired_audio_ids: set[str] = set()
    for item in resolved.reference_order:
        if item.paired_video_resource_id:
            if item.paired_video_resource_id in paired_by_video:
                raise MMH3ResourceError(
                    f"Video reference {item.paired_video_resource_id!r} has more than one paired soundtrack"
                )
            paired_by_video[item.paired_video_resource_id] = item.resource.resource_id
            paired_audio_ids.add(item.resource.resource_id)

    dynamic: dict[str, Any] = {}
    picture_group = contract.group("ref_images")
    for index, descriptor in enumerate(pictures):
        image_node = _get_node(graph, "MMH3GetImage", packet, descriptor)
        image = image_node.out(0)
        if ref_image_short_edge is not None:
            image = graph.node(
                "MMH3H3ReferenceImageResize", image=image, short_edge=int(ref_image_short_edge)
            ).out(0)
        dynamic[picture_group.input_path(index)] = image

    video_group = contract.group("ref_videos")
    video_audio_group = contract.group("ref_video_audios")
    for index, descriptor in enumerate(videos):
        video_node = graph.node(
            "MMH3H3VideoReference",
            packet=packet,
            resource_id=descriptor.resource_id,
            paired_audio_resource_id=paired_by_video.get(descriptor.resource_id, ""),
            include_embedded_audio=False,
            target_fps=24.0,
        )
        dynamic[video_group.input_path(index)] = video_node.out(0)
        dynamic[video_audio_group.input_path(index)] = video_node.out(1)

    audio_group = contract.group("ref_audios")
    standalone = [item for item in audios if item.resource_id not in paired_audio_ids]
    for index, descriptor in enumerate(standalone):
        audio_node = _get_node(graph, "MMH3GetAudio", packet, descriptor)
        dynamic[audio_group.input_path(index)] = audio_node.out(0)

    audio_inputs = {} if encoder_only else {"audio_vae": preserve_h3_audio_onset(audio_vae)}
    native = graph.node(
        contract.node_id,
        **common,
        **audio_inputs,
        ref_image_size=ref_image_size,
        **dynamic,
    )
    return H3Expansion(native.out(0), native.out(1), graph.finalize(), contract.fingerprint)
