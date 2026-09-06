from __future__ import annotations

import logging

from .node_support import ComfyExtension, io, override
from .nodes_packet import MMH3Create, MMH3Load, MMH3Save, MMH3Put, MMH3Remove, MMH3Move, MMH3Metadata, MMH3GenerationLoRAs, MMH3H3RefineLoRAs
from .nodes_conditioning import MMH3Inspect, MMH3Preflight, MMH3ReferenceConfigure, MMH3ReferenceReport, MMH3ReferenceCacheStore, MMH3ReferenceCacheGet, MMH3ReferenceCacheVerify, MMH3ResolveReport, MMH3H3VideoReference, MMH3H3ReferenceImageResize, MMH3H3AutoCondition, MMH3H3ContinuationCondition, MMH3PackH3Result
from .nodes_chain import MMH3ChainStart, MMH3ChainCommit, MMH3ChainValidate, MMH3ChainValidateRerollSource, MMH3VideoStitch, MMH3H3LatentStitch, MMH3Unpack
from .nodes_upscale import MMH3H3LatentUpscalePrepare, MMH3H3LatentUpscaleTarget, MMH3H3LatentStitchUpscaleTarget, MMH3H3UpscaleRefineSampling, MMH3H3DecodedUpscalePrepare, MMH3H3LatentUpscaleReport, MMH3H3ExternalTileFinalize, MMH3H3NativeTileRefine
from .nodes_h3 import MMH3H3AVSeparate, MMH3H3AVCombine, MMH3H3Provenance, MMH3H3Compatibility, MMH3H3ContinuationHandover, MMH3H3DecodedContinuation, MMH3H3ContinuationGuide
from .nodes_utility import MMH3Compare, MMH3Export, MMH3Preview, _MMH3GetBase, MMH3GetLatent, MMH3GetImage, MMH3GetVideo, MMH3GetAudio, MMH3GetMask, MMH3GetJSON
from .nodes_control import MMH3ControlConfigure, MMH3ControlPreflight, MMH3ControlVideo, MMH3MaskedEditCondition, MMH3H3ControlApply
from .nodes_optimization import MMH3H3FP16AccumulationPatch, MMH3H3ModelOptimizations, MMH3H3SamplingPreset, MMH3H3SLAApply
from .nodes_settings import MMH3H3GenerationSettings, MMH3H3ReferenceImageSettings
from .nodes_segments import MMH3H3SegmentPrepare, MMH3SegmentReview
from .nodes_stitch_upscale import MMH3H3StitchUpscale, MMH3H3UpscaleAVRestore
from .nodes_delivery import MMH3VideoUpscale
from .nodes_automation import MMH3AutomationAssembleChunks, MMH3AutomationAssemblyGate, MMH3AutomationCheckpoint, MMH3AutomationLedger, MMH3AutomationReport, MMH3AutomationVideoChunk, MMH3BatchInputPlan, MMH3BatchNormalizeImport, MMH3BatchPreQueueEstimate, MMH3BatchStitch, MMH3H3AudioSyncProof, MMH3LongVideoAudioSyncSettings, MMH3LongVideoChunkPlan, MMH3LongVideoUpscaleSettings, MMH3TrimAudioSamples
from .upstream_backports.minimax_h3_forward_patch import BackportCompatibilityError
from .upstream_backports.minimax_h3_fun_pr15860 import ensure_minimax_h3_fun_backport


LOGGER = logging.getLogger(__name__)


class MMH3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        node_list = [
            MMH3Create,
            MMH3Load,
            MMH3Save,
            MMH3Put,
            MMH3Remove,
            MMH3Move,
            MMH3Metadata,
            MMH3GenerationLoRAs,
            MMH3H3RefineLoRAs,
            MMH3Inspect,
            MMH3Preflight,
            MMH3H3GenerationSettings,
            MMH3H3ReferenceImageSettings,
            MMH3BatchInputPlan,
            MMH3BatchPreQueueEstimate,
            MMH3BatchNormalizeImport,
            MMH3BatchStitch,
            MMH3LongVideoChunkPlan,
            MMH3LongVideoUpscaleSettings,
            MMH3LongVideoAudioSyncSettings,
            MMH3H3AudioSyncProof,
            MMH3AutomationVideoChunk,
            MMH3TrimAudioSamples,
            MMH3AutomationLedger,
            MMH3AutomationCheckpoint,
            MMH3AutomationReport,
            MMH3AutomationAssemblyGate,
            MMH3AutomationAssembleChunks,
            MMH3ReferenceConfigure,
            MMH3ReferenceReport,
            MMH3ReferenceCacheStore,
            MMH3ReferenceCacheGet,
            MMH3ReferenceCacheVerify,
            MMH3ResolveReport,
            MMH3H3VideoReference,
            MMH3H3ReferenceImageResize,
            MMH3H3AutoCondition,
            MMH3H3ContinuationCondition,
            MMH3PackH3Result,
            MMH3ControlConfigure,
            MMH3ControlPreflight,
            MMH3ControlVideo,
            MMH3MaskedEditCondition,
            MMH3H3ControlApply,
            MMH3ChainStart,
            MMH3H3SegmentPrepare,
            MMH3SegmentReview,
            MMH3VideoUpscale,
            MMH3ChainCommit,
            MMH3ChainValidate,
            MMH3ChainValidateRerollSource,
            MMH3VideoStitch,
            MMH3H3LatentStitch,
            MMH3H3StitchUpscale,
            MMH3H3UpscaleAVRestore,
            MMH3Unpack,
            MMH3Export,
            MMH3Compare,
            MMH3Preview,
            MMH3H3LatentUpscalePrepare,
            MMH3H3DecodedUpscalePrepare,
            MMH3H3LatentUpscaleTarget,
            MMH3H3LatentStitchUpscaleTarget,
            MMH3H3UpscaleRefineSampling,
            MMH3H3LatentUpscaleReport,
            MMH3H3NativeTileRefine,
            MMH3H3ExternalTileFinalize,
            MMH3H3SLAApply,
            MMH3H3SamplingPreset,
            MMH3H3ModelOptimizations,
            MMH3H3FP16AccumulationPatch,
            MMH3H3AVSeparate,
            MMH3H3AVCombine,
            MMH3H3Provenance,
            MMH3H3Compatibility,
            MMH3H3ContinuationHandover,
            MMH3H3ContinuationGuide,
            MMH3H3DecodedContinuation,
            MMH3GetLatent,
            MMH3GetImage,
            MMH3GetVideo,
            MMH3GetAudio,
            MMH3GetMask,
            MMH3GetJSON,
        ]
        try:
            status, apply_class = ensure_minimax_h3_fun_backport(io)
            LOGGER.info("MiniMax H3 Fun runtime support: %s", status.detail)
            if apply_class is not None:
                node_list.append(apply_class)
        except BackportCompatibilityError as error:
            # Keep the rest of MMH3 usable. F16 preflight will remain fail-closed
            # because MiniMaxH3FunControlNetApply is not registered.
            LOGGER.error("MiniMax H3 Fun PR #15860 backport disabled: %s", error)
        return node_list
