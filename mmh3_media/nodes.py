from __future__ import annotations

import logging

from .node_support import ComfyExtension, io, override
from .nodes_packet import MMH3Create, MMH3Load, MMH3Save, MMH3Put, MMH3Remove, MMH3Move, MMH3Metadata, MMH3GenerationLoRAs, MMH3H3RefineLoRAs
from .nodes_conditioning import MMH3Inspect, MMH3Preflight, MMH3ReferenceConfigure, MMH3ReferenceReport, MMH3ReferenceCacheStore, MMH3ReferenceCacheGet, MMH3ReferenceCacheVerify, MMH3ResolveReport, MMH3H3VideoReference, MMH3H3ReferenceImageResize, MMH3H3AutoCondition, MMH3H3ContinuationCondition, MMH3PackH3Result
from .nodes_editing import MMH3AVEditPolicy, MMH3AVProtectionRestore, MMH3TwoClipAVBridge, MMH3ReferenceSchedule, MMH3SceneReferences, MMH3ReferenceAlias, MMH3AVEditAudio, MMH3BridgeMiddle
from .nodes_chain import MMH3ChainStart, MMH3ChainCommit, MMH3ChainValidate, MMH3ChainValidateRerollSource, MMH3VideoStitch, MMH3H3LatentStitch, MMH3Unpack
from .nodes_upscale_pipeline import MMH3H3UpscaleSettings, MMH3H3UpscalePreflight, MMH3H3RefineScheduler, MMH3H3UpscaleAudio
from .nodes_upscale import MMH3H3LearnedUpscale, MMH3H3LatentUpscalePrepare, MMH3H3LatentUpscaleTarget, MMH3H3LatentStitchUpscaleTarget, MMH3H3UpscaleRefineSampling, MMH3H3DecodedUpscalePrepare, MMH3H3LatentUpscaleReport, MMH3H3ExternalTileFinalize, MMH3H3NativeTileRefine
from .nodes_h3 import MMH3H3AVSeparate, MMH3H3AVCombine, MMH3H3Provenance, MMH3H3Compatibility, MMH3H3ContinuationHandover, MMH3H3DecodedContinuation, MMH3H3ContinuationGuide
from .nodes_utility import MMH3Compare, MMH3Export, MMH3Preview, _MMH3GetBase, MMH3GetLatent, MMH3GetImage, MMH3GetVideo, MMH3GetAudio, MMH3GetMask, MMH3GetJSON
from .nodes_control import MMH3ControlConfigure, MMH3ControlPreflight, MMH3ControlVideo, MMH3MaskedEditCondition, MMH3H3ControlApply, MMH3H3FunControl
from .nodes_optimization import MMH3H3OptimizationRecord, MMH3H3FP16AccumulationPatch, MMH3H3ModelOptimizations, MMH3H3SamplingPreset, MMH3H3SLAApply, MMH3H3VDNApply
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
            MMH3AVEditPolicy,
            MMH3AVProtectionRestore,
            MMH3TwoClipAVBridge,
            MMH3ReferenceSchedule,
            MMH3SceneReferences,
            MMH3ReferenceAlias,
            MMH3AVEditAudio,
            MMH3BridgeMiddle,
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
            MMH3H3FunControl,
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
            MMH3H3LearnedUpscale,
            MMH3H3LatentUpscalePrepare,
            MMH3H3DecodedUpscalePrepare,
            MMH3H3LatentUpscaleTarget,
            MMH3H3LatentStitchUpscaleTarget,
            MMH3H3UpscaleRefineSampling,
            MMH3H3UpscaleSettings,
            MMH3H3UpscalePreflight,
            MMH3H3RefineScheduler,
            MMH3H3UpscaleAudio,
            MMH3H3LatentUpscaleReport,
            MMH3H3NativeTileRefine,
            MMH3H3ExternalTileFinalize,
            MMH3H3SLAApply,
            MMH3H3VDNApply,
            MMH3H3SamplingPreset,
            MMH3H3OptimizationRecord,
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
        # Current ComfyUI (PR #15975) owns H3 Fun as a MODEL_PATCH. Never install
        # the older #15860 forward monkey-patch on a core that exposes that contract.
        try:
            from comfy.ldm.minimax import controlnet as _h3_control  # type: ignore
            from comfy_extras import nodes_minimax_h3 as _h3_nodes  # type: ignore
            current_fun_patch = (
                callable(getattr(_h3_control, "is_minimax_h3_fun_state_dict", None))
                and getattr(_h3_control, "MiniMaxH3FunControl", None) is not None
                and getattr(_h3_nodes, "MiniMaxH3FunControlPatch", None) is not None
            )
        except Exception:
            current_fun_patch = False
        if current_fun_patch:
            LOGGER.info("MiniMax H3 Fun runtime support: current MODEL_PATCH contract (#15975)")
        else:
            try:
                status, apply_class = ensure_minimax_h3_fun_backport(io)
                LOGGER.info("MiniMax H3 Fun legacy runtime support: %s", status.detail)
                if apply_class is not None:
                    node_list.append(apply_class)
            except BackportCompatibilityError as error:
                LOGGER.error("MiniMax H3 Fun PR #15860 compatibility backport disabled: %s", error)
        return node_list
