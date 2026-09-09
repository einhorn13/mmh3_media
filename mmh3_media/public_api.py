from __future__ import annotations

from typing import Any, Mapping

from .av_edit import apply_av_edit_policy, restore_av_protection, select_edit_delivery_audio
from .av_bridge import prepare_av_bridge, encode_av_bridge, trim_bridge_components
from .scheduled_references import configure_reference_schedule, compile_scene_references, add_reference_alias
from .archive import get_resource_payload
from .core import MMH3Media
from .resource_ref import MMH3ResourceRef
from .errors import MMH3Error, MMH3FormatError, MMH3IntegrityError, MMH3ResourceError
from .resolution import ResolvedMMH3Inputs, ResolvedReferenceSet, resolve_packet, resolve_reference_set
from .reference_management import ReferenceConfigurationResult, configure_reference
from .process_result import PackedH3Result, PrimaryPacketView, pack_h3_result, unpack_primary
from .lora_provenance import (
    clear_generation_loras,
    get_generation_loras,
    lora_provenance_summary,
    normalize_generation_loras,
    set_generation_loras,
)
from .util import deep_copy_json
from .media_metadata import describe_media_payload
from .resource_model import descriptor_from_media_metadata
from .reference_cost import ReferenceCostReport, estimate_reference_cost
from .reference_cache import (
    ReferenceCacheLookup,
    ReferenceCacheSpec,
    build_reference_cache_spec,
    compare_reference_cache_latents,
    lookup_reference_cache,
    materialize_reference_cache,
    put_reference_cache,
)
from .preflight import PREFLIGHT_OPERATIONS, PreflightDiagnostic, PreflightReport, preflight_packet
from .generation_contract import (
    H3_GENERATION_MODES,
    H3_MODEL_FAMILIES,
    GenerationModelContract,
    ModelFamilyDiagnostic,
    infer_h3_model_family,
    validate_generation_model_family,
)
from .continuation import (
    H3ContinuationPlan,
    H3ContinuationResult,
    build_h3_continuation_handover,
    build_h3_target_from_prefix,
    exact_h3_av_handover_boundaries,
    h3_video_t_from_frames,
    is_exact_h3_av_handover_boundary,
)
from .decoded_continuation import (
    DecodedContinuationPlan,
    DecodedContinuationResult,
    PreparedDecodedPrefix,
    encode_decoded_prefix,
    prepare_decoded_prefix,
)
from .chain import ChainCommitResult, ChainValidation, RerollSourceValidation, commit_chain_segment, start_chain, validate_chain, validate_reroll_source
from .stitch import DecodedSegment, StitchCompatibility, StitchPlan, StitchResult, inspect_stitch_packets, materialize_decoded_segment, pcm_boundary, stitch_decoded_segments
from .streaming_stitch import StreamingSegment, StreamingStitchResult, StreamingStitchedVideo, build_streaming_stitch, iter_stitched_rgb_frames, materialize_streaming_segment
from .lora_reapply import LoRAReapplyDiagnostic, LoRAReapplyExpansion, LoRAReapplyPlan, build_high_sigma_lora_plan, build_lora_reapply_expansion
from .latent_upscale import UPSCALE_GEOMETRY_MODES, LatentUpscalePlan, LatentUpscaleProcessReport, PreparedLatentUpscale, build_latent_upscale_process_report, build_latent_upscale_refine_target, plan_latent_upscale_geometry, prepare_decoded_packet_latent_upscale, prepare_packet_latent_upscale
from .spatial_tiles import TILE_BLEND_MODES, TILE_CONTEXT_SOURCES, TILE_MASK_RESIZE_MODES, TILE_OVERLAP_MODES, TILE_TRAVERSALS, SpatialTile, SpatialTilePlan, SpatialTileRunResult, TileRect, plan_spatial_tiles, prepare_spatial_region_mask, run_masked_spatial_video_tiles, run_spatial_video_tiles, validate_masked_spatial_tile_run_report, validate_spatial_tile_plan_report, validate_spatial_tile_run_report
from .h3_tile_refine import NATIVE_H3_TILE_CONTRACT, NativeH3TileRefineResult, run_native_h3_tile_refine, validate_native_h3_tile_refine_report
from .tile_backend import TILE_BACKEND_OVERLAP_MODES, TileBackendFinalizeResult, finalize_external_tile_latents, finalize_external_tile_video, tile_backend_fingerprint_from_preflight, validate_external_tile_finalize_report
from .control_contract import (
    H3_CONTROL_ALGORITHMS,
    H3_CONTROL_ALGORITHM_CONTEXTS,
    H3_CONTROL_AUTO_PREFERENCES,
    H3_CONTROL_CONTEXT_ALGORITHMS,
    H3_CONTROLNET_ALGORITHMS,
    H3_CONTROL_KINDS,
    H3_CONTROL_KIND_OPTIONS,
    H3_CONTROL_RESOURCE_ROLES,
    H3_CONTROL_TEMPORAL_POLICIES,
    ControlDiagnostic,
    ControlAlgorithmResolution,
    ControlPreprocessor,
    ControlProviderProvenance,
    H3ControlConfiguration,
    build_control_configuration,
    control_configuration_from_dict,
    get_control_configuration,
    set_control_configuration,
    resolve_control_algorithm,
)
from .control_preflight import (
    CONTROLNET_LOADER_NODE_ID,
    H3_CONTROL_APPLY_NODE_ID,
    H3ControlCapabilityReport,
    RuntimeNodeContract,
    parse_controlnet_loader_contract,
    parse_h3_control_apply_contract,
    preflight_h3_control,
)
from .control_provider import (
    H3ControlApplyPlan,
    H3ControlExpansion,
    H3ControlProviderAdapter,
    MaterializedControlVideo,
    align_control_timeline,
    bind_control_capability,
    build_control_apply_plan,
    build_control_pass_through_info,
    build_h3_control_expansion,
    control_provider_for_algorithm,
    prepare_control_inputs,
    normalize_control_apply_process_info,
    materialize_control_video,
    validate_masked_edit_inputs,
)
from .control_tiles import (
    H3_CONTROL_TILE_CONTRACT,
    H3TileControlInputs,
    apply_control_to_conditioning,
    build_tiled_control_process_info,
    clone_guider_with_conditioning,
    crop_control_inputs_for_tile,
)
from .sampling_presets import (
    CUSTOM_PROFILE,
    PROFILE_PRESETS,
    SAMPLING_PRESET_CONTRACT,
    SAMPLING_PROFILES,
    STANDARD_PROFILE,
    TURBO_4_PROFILE,
    TURBO_8_PROFILE,
    TURBO_RECIPES,
    SamplingPreset,
    build_sampling_preset,
)
from .model_optimizations import (
    ATTENTION_MODES,
    FP16_ACCUMULATION_MODES,
    MODEL_OPTIMIZATION_CONTRACT,
    ModelOptimizationExpansion,
    ModelOptimizationPlan,
    build_model_optimization_expansion,
    build_model_optimization_plan,
)
from .generation_settings import GenerationSettings, adapt_h3_canvas, build_generation_settings
from .automation import ChunkExecutionSelection, ChunkPlan, VIDEO_SUFFIXES, plan_batch_inputs, plan_long_video_chunks, resolve_chunk_execution
from .raw_video_import import ConformSettings, RawImportResult, RawVideoProbe, import_raw_video, normalize_batch_plan, probe_raw_video
from .automation_adapters import VideoChunkExpansion, build_video_chunk_expansion, trim_audio_samples
from .automation_estimate import PREQUEUE_ESTIMATE_CONTRACT, build_prequeue_estimate, prequeue_summary
from .automation_assembly import ChunkAssemblyResult, assemble_chunk_packets, load_immutable_lipsync_audio
from .automation_upscale import build_long_video_upscale_settings, validate_upscale_chunk_artifact
from .automation_lipsync import (
    PROOF_CONTRACT as LIPSYNC_PROOF_CONTRACT,
    SETTINGS_CONTRACT as LIPSYNC_SETTINGS_CONTRACT,
    build_h3_audio_sync_chunk_proof,
    build_long_video_audio_sync_settings,
    validate_lipsync_chunk_proof,
)
from .automation_execution import EXECUTION_CONTRACT, JOB_STATES, acquire_next_execution_job, build_chunk_assembly_map, commit_execution_artifact, create_execution_ledger, deterministic_artifact_prefix, load_execution_ledger, save_execution_ledger, select_resume_jobs, transition_execution_job


MMH3_ADAPTER_API_VERSION = 1


def list_resource_refs(
    packet: MMH3Media,
    *,
    kind: str | None = None,
    role: str | None = None,
    tags: tuple[str, ...] | list[str] | None = None,
) -> tuple[MMH3ResourceRef, ...]:
    """Return lazy handles filtered by canonical v0.3 resource fields."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    return packet.filter_resource_refs(kind=kind, role=role, tags=tags)


def select_resource_ref(packet: MMH3Media, *, resource_id: str) -> MMH3ResourceRef:
    """Select one resource by stable ID."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    return packet.ref(resource_id)


def primary_resource_ref(packet: MMH3Media, *, kind: str) -> MMH3ResourceRef | None:
    """Return the explicitly bound primary resource for a native media kind."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    return packet.primary(kind)


def set_primary_resource(packet: MMH3Media, *, kind: str, resource_id: str | None) -> MMH3Media:
    """Return a new packet snapshot with the primary binding updated or cleared."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    return packet.set_primary(kind, resource_id)




def list_representation_records(
    packet: MMH3Media,
    *,
    target: str,
    kind: str | None = None,
    fresh_only: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return detached derived representation records without source materialization."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    return packet.representations_for(target, kind=kind, fresh_only=fresh_only)


def resource_preview_info(packet: MMH3Media, *, resource_id: str) -> dict[str, Any]:
    """Return descriptor-first preview metadata for one resource without payload I/O."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    return packet.ref(resource_id).preview_info()


def packet_preview_info(packet: MMH3Media) -> dict[str, Any]:
    """Return descriptor-first packet preview metadata without payload I/O."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    from .preview import preview_info
    from .representations import PACKET_TARGET

    return preview_info(packet, target=PACKET_TARGET)

def list_resources(
    packet: MMH3Media,
    *,
    kind: str | None = None,
    role: str | None = None,
    tags: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return detached canonical v0.3 descriptors without materializing payloads.

    Descriptor listing intentionally walks the manifest directly instead of building
    resource refs and resolving each ID again.  This keeps the common metadata-only
    path linear for large packets while preserving ref validation for handle APIs.
    """
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    wanted_tags = {
        str(tag).strip().lower().replace("_", "-")
        for tag in (tags or ())
        if str(tag).strip()
    }
    selected = [
        resource
        for resource in packet.manifest["resources"]
        if (kind is None or resource["kind"] == kind)
        and (role is None or resource["role"] == role)
        and (not wanted_tags or wanted_tags.issubset(set(resource["tags"])))
    ]
    selected.sort(
        key=lambda resource: (
            resource.get("order") is None,
            resource.get("order") if resource.get("order") is not None else 0,
            resource["id"],
        )
    )
    return tuple(deep_copy_json(resource) for resource in selected)


def select_resource(packet: MMH3Media, *, resource_id: str) -> dict[str, Any]:
    """Return one detached canonical v0.3 descriptor by stable resource ID."""
    return select_resource_ref(packet, resource_id=resource_id).descriptor


def materialize_resource(packet: MMH3Media, resource_id: str) -> Any:
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    selected = packet.get_by_id(resource_id)
    if selected is None:
        raise MMH3ResourceError(f"Resource {resource_id!r} does not exist")
    return get_resource_payload(packet, selected)


def put_resource(
    packet: MMH3Media,
    payload: Any,
    *,
    kind: str,
    role: str = "auxiliary",
    order: int | None = None,
    mode: str = "add",
    resource_id: str = "",
    name: str = "",
    tags: tuple[str, ...] | list[str] | None = None,
    descriptor: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> MMH3Media:
    """Put one canonical v0.3 resource."""
    if not isinstance(packet, MMH3Media):
        raise MMH3ResourceError("Expected an MMH3_MEDIA packet")
    observed = descriptor_from_media_metadata(kind, describe_media_payload(payload, kind))
    observed.update(deep_copy_json(dict(descriptor or {})))
    return packet.put(
        payload, kind=kind, role=role, order=order, mode=mode, resource_id=resource_id,
        name=name, tags=tags, descriptor=observed, provenance=deep_copy_json(dict(provenance or {})),
        extensions=deep_copy_json(dict(extensions or {})),
    )



__all__ = [
    "apply_av_edit_policy", "restore_av_protection", "select_edit_delivery_audio",
    "prepare_av_bridge", "encode_av_bridge", "trim_bridge_components",
    "configure_reference_schedule", "compile_scene_references", "add_reference_alias",
    "MMH3_ADAPTER_API_VERSION",
    "list_resource_refs",
    "select_resource_ref",
    "list_representation_records",
    "resource_preview_info",
    "packet_preview_info",
    "MMH3Error",
    "MMH3FormatError",
    "MMH3IntegrityError",
    "MMH3ResourceError",
    "H3_CONTROL_ALGORITHMS",
    "H3_CONTROL_ALGORITHM_CONTEXTS",
    "H3_CONTROL_AUTO_PREFERENCES",
    "H3_CONTROL_CONTEXT_ALGORITHMS",
    "H3_CONTROLNET_ALGORITHMS",
    "H3_CONTROL_KINDS",
    "H3_CONTROL_KIND_OPTIONS",
    "H3_CONTROL_RESOURCE_ROLES",
    "H3_CONTROL_TEMPORAL_POLICIES",
    "CONTROLNET_LOADER_NODE_ID",
    "H3_CONTROL_APPLY_NODE_ID",
    "ControlDiagnostic",
    "ControlAlgorithmResolution",
    "ControlPreprocessor",
    "ControlProviderProvenance",
    "H3ControlConfiguration",
    "H3ControlCapabilityReport",
    "RuntimeNodeContract",
    "H3ControlApplyPlan",
    "H3ControlExpansion",
    "H3ControlProviderAdapter",
    "MaterializedControlVideo",
    "H3TileControlInputs",
    "H3_CONTROL_TILE_CONTRACT",
    "H3_GENERATION_MODES",
    "H3_MODEL_FAMILIES",
    "GenerationModelContract",
    "H3ContinuationPlan",
    "H3ContinuationResult",
    "DecodedContinuationPlan",
    "DecodedContinuationResult",
    "PreparedDecodedPrefix",
    "ChainCommitResult",
    "ChainValidation",
    "RerollSourceValidation",
    "DecodedSegment",
    "StitchCompatibility",
    "StitchPlan",
    "StitchResult",
    "StreamingSegment",
    "StreamingStitchResult",
    "StreamingStitchedVideo",
    "LoRAReapplyDiagnostic",
    "LoRAReapplyExpansion",
    "LoRAReapplyPlan",
    "LatentUpscalePlan",
    "LatentUpscaleProcessReport",
    "PreparedLatentUpscale",
    "UPSCALE_GEOMETRY_MODES",
    "TILE_BLEND_MODES",
    "TILE_CONTEXT_SOURCES",
    "TILE_MASK_RESIZE_MODES",
    "TILE_OVERLAP_MODES",
    "TILE_TRAVERSALS",
    "TILE_BACKEND_OVERLAP_MODES",
    "TileRect",
    "SpatialTile",
    "SpatialTilePlan",
    "SpatialTileRunResult",
    "NativeH3TileRefineResult",
    "NATIVE_H3_TILE_CONTRACT",
    "TileBackendFinalizeResult",
    "ModelFamilyDiagnostic",
    "PackedH3Result",
    "PrimaryPacketView",
    "ReferenceConfigurationResult",
    "ResolvedMMH3Inputs",
    "ResolvedReferenceSet",
    "ReferenceCostReport",
    "ReferenceCacheLookup",
    "ReferenceCacheSpec",
    "PREFLIGHT_OPERATIONS",
    "PreflightDiagnostic",
    "PreflightReport",
    "clear_generation_loras",
    "configure_reference",
    "build_reference_cache_spec",
    "build_h3_continuation_handover",
    "build_h3_target_from_prefix",
    "build_streaming_stitch",
    "build_high_sigma_lora_plan",
    "build_lora_reapply_expansion",
    "plan_latent_upscale_geometry",
    "plan_spatial_tiles",
    "prepare_spatial_region_mask",
    "run_masked_spatial_video_tiles",
    "validate_masked_spatial_tile_run_report",
    "run_spatial_video_tiles",
    "run_native_h3_tile_refine",
    "validate_native_h3_tile_refine_report",
    "finalize_external_tile_video",
    "finalize_external_tile_latents",
    "tile_backend_fingerprint_from_preflight",
    "validate_external_tile_finalize_report",
    "validate_spatial_tile_plan_report",
    "validate_spatial_tile_run_report",
    "prepare_packet_latent_upscale",
    "prepare_decoded_packet_latent_upscale",
    "build_latent_upscale_process_report",
    "build_latent_upscale_refine_target",
    "compare_reference_cache_latents",
    "describe_media_payload",
    "estimate_reference_cost",
    "encode_decoded_prefix",
    "commit_chain_segment",
    "exact_h3_av_handover_boundaries",
    "lookup_reference_cache",
    "materialize_reference_cache",
    "get_generation_loras",
    "infer_h3_model_family",
    "inspect_stitch_packets",
    "h3_video_t_from_frames",
    "is_exact_h3_av_handover_boundary",
    "list_resources",
    "lora_provenance_summary",
    "materialize_resource",
    "materialize_decoded_segment",
    "materialize_streaming_segment",
    "pcm_boundary",
    "pack_h3_result",
    "put_resource",
    "put_reference_cache",
    "preflight_packet",
    "prepare_decoded_prefix",
    "resolve_packet",
    "resolve_reference_set",
    "select_resource",
    "set_generation_loras",
    "start_chain",
    "stitch_decoded_segments",
    "iter_stitched_rgb_frames",
    "normalize_generation_loras",
    "unpack_primary",
    "validate_generation_model_family",
    "build_control_configuration",
    "control_configuration_from_dict",
    "get_control_configuration",
    "set_control_configuration",
    "resolve_control_algorithm",
    "parse_controlnet_loader_contract",
    "parse_h3_control_apply_contract",
    "preflight_h3_control",
    "align_control_timeline",
    "bind_control_capability",
    "build_control_apply_plan",
    "build_control_pass_through_info",
    "build_h3_control_expansion",
    "control_provider_for_algorithm",
    "prepare_control_inputs",
    "normalize_control_apply_process_info",
    "materialize_control_video",
    "validate_masked_edit_inputs",
    "crop_control_inputs_for_tile",
    "apply_control_to_conditioning",
    "clone_guider_with_conditioning",
    "build_tiled_control_process_info",
    "validate_chain",
    "validate_reroll_source",
    "ATTENTION_MODES",
    "CUSTOM_PROFILE",
    "FP16_ACCUMULATION_MODES",
    "MODEL_OPTIMIZATION_CONTRACT",
    "PROFILE_PRESETS",
    "SAMPLING_PRESET_CONTRACT",
    "SAMPLING_PROFILES",
    "STANDARD_PROFILE",
    "TURBO_4_PROFILE",
    "TURBO_8_PROFILE",
    "TURBO_RECIPES",
    "ModelOptimizationExpansion",
    "ModelOptimizationPlan",
    "SamplingPreset",
    "build_model_optimization_expansion",
    "build_model_optimization_plan",
    "build_sampling_preset",
    "GenerationSettings",
    "adapt_h3_canvas",
    "build_generation_settings",
    "ChunkPlan",
    "ChunkExecutionSelection",
    "VideoChunkExpansion",
    "ChunkAssemblyResult",
    "VIDEO_SUFFIXES",
    "RawVideoProbe",
    "ConformSettings",
    "RawImportResult",
    "probe_raw_video",
    "import_raw_video",
    "normalize_batch_plan",
    "plan_batch_inputs",
    "plan_long_video_chunks",
    "PREQUEUE_ESTIMATE_CONTRACT",
    "build_prequeue_estimate",
    "prequeue_summary",
    "resolve_chunk_execution",
    "build_video_chunk_expansion",
    "trim_audio_samples",
    "assemble_chunk_packets",
    "load_immutable_lipsync_audio",
    "build_long_video_upscale_settings",
    "validate_upscale_chunk_artifact",
    "LIPSYNC_PROOF_CONTRACT",
    "LIPSYNC_SETTINGS_CONTRACT",
    "validate_lipsync_chunk_proof",
    "EXECUTION_CONTRACT",
    "JOB_STATES",
    "build_chunk_assembly_map",
    "acquire_next_execution_job",
    "commit_execution_artifact",
    "create_execution_ledger",
    "deterministic_artifact_prefix",
    "load_execution_ledger",
    "save_execution_ledger",
    "select_resume_jobs",
    "transition_execution_job",
]
