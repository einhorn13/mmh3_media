from __future__ import annotations

from .nodes_optimization import h3_optimization_inputs
from .node_support import CATEGORY, MMH3, MMH3ResourceError, _packet, io, ui
from .stitch_upscale import (DEFAULT_UPSCALER, UPSCALER_NODE, UPSCALE_ASPECTS,
    build_stitch_upscale_expansion, restore_upscale_av, resolve_stitch_upscale_geometry)
from .upscaler_adapter import resolve_upscaler_api


class MMH3H3StitchUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        return io.Schema(
            node_id='MMH3H3StitchUpscale', display_name='H3 Upscale + Stitch', category=CATEGORY,
            description='Upscale an ordered continuation chain, preserve exact AV seams, then decode once. Add more segment inputs; no duplicated generation blocks.',
            inputs=[
                io.Autogrow.Input('segments', template=io.Autogrow.TemplateNames(
                    MMH3.Input('segment'), names=[f'segment_{i}' for i in range(1, 67)], min=2)),
                io.Clip.Input('clip'), io.Vae.Input('video_vae'), io.Vae.Input('audio_vae'),
                io.DynamicCombo.Input('resolution', options=[
                    io.DynamicCombo.Option('Scale', [
                        io.Float.Input('scale', default=2.0, min=1.0, max=4.0, step=0.05,
                                       tooltip='Multiply MMH3 source dimensions; preserve source proportions.')]),
                    io.DynamicCombo.Option('Megapixels', [
                        io.Float.Input('megapixels', default=1.0, min=0.1, max=8.0, step=0.1,
                                       tooltip='1 MP = 1024×1024 pixels. Dimensions round to multiples of 32.'),
                        io.Combo.Input('aspect_ratio', options=list(UPSCALE_ASPECTS), default='Source',
                                       tooltip='Source reads MMH3 proportions. Other ratios stretch the image; no crop.')]),
                    io.DynamicCombo.Option('Dimensions', [
                        io.Int.Input('width', default=1344, min=32, max=4096, step=32),
                        io.Int.Input('height', default=768, min=32, max=4096, step=32)]),
                ]),
                io.Float.Input('denoise', default=0.0, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip='0 inherits the source-aware amount. 0.05–0.50 is recommended; stronger refinement may replace source structure.'),
                io.String.Input('upscaler_model', default=DEFAULT_UPSCALER, advanced=True,
                                tooltip='Installed model filename in latent_upscale_models.'),
                io.Model.Input('fl2va_model', optional=True, tooltip='Unpatched FL2VA base for T2VA/I2VA/FL2VA segments.'),
                io.Model.Input('ref2va_model', optional=True, tooltip='Unpatched Ref2VA base, needed only when the chain includes Ref2VA segments.'),
                io.Int.Input('steps_override', default=0, min=0, max=100, optional=True, advanced=True,
                             tooltip='Upscale denoising steps; 0 inherits source. Manual sigmas take priority.'),
                io.String.Input('manual_sigmas', default='', multiline=True, optional=True, advanced=True,
                                tooltip='Optional descending sigmas ending in 0, e.g. 0.5, 0.25, 0.1, 0. Overrides steps and denoise.'),
                io.Combo.Input('turbo_override', options=['Source'] + folder_paths.get_filename_list('loras'),
                               default='Source', optional=True, advanced=True,
                               tooltip='Upscale-only Turbo LoRA at strength 1. Replaces source acceleration adapters; other LoRAs retain their order and strengths. Match the model family. Sampler/AV shifts remain from source.'),
                *h3_optimization_inputs(optional=True),
                io.Boolean.Input('force_unload', default=True, advanced=True, optional=True,
                                 tooltip='Unload the learned upscaler after each part to save VRAM. Disable only with enough memory.'),
            ],
            outputs=[MMH3.Output('packet'), io.Video.Output('video'), io.Latent.Output('latent'),
                     io.String.Output('assembly_report_json')], enable_expand=True,
        )

    @classmethod
    def execute(cls, segments, clip, video_vae, audio_vae, resolution, denoise, upscaler_model,
                fl2va_model=None, ref2va_model=None, steps_override=0, manual_sigmas='', turbo_override='Source',
                attention='Default', fp16_accumulation='Default', force_unload=True):
        packets = [_packet(segments[key]) for key in sorted(segments, key=lambda key: int(key.rsplit('_', 1)[1]))
                   if segments[key] is not None]
        import nodes
        backend = nodes.NODE_CLASS_MAPPINGS.get(UPSCALER_NODE)
        if backend is None:
            raise MMH3ResourceError('Install Minimax H3 Latent Upscaler (3D) to use Upscale + Stitch')
        upscaler_api = resolve_upscaler_api(backend)
        plan = resolve_stitch_upscale_geometry(packets, resolution)
        outputs, graph, summary = build_stitch_upscale_expansion(
            packets, width=plan.target_width, height=plan.target_height, denoise=denoise,
            models={'fl2va': fl2va_model, 'ref2va': ref2va_model}, clip=clip,
            video_vae=video_vae, audio_vae=audio_vae, upscaler_model=upscaler_model,
            steps_override=steps_override, manual_sigmas=manual_sigmas,
            turbo_override='' if turbo_override == 'Source' else turbo_override,
            attention=attention, fp16_accumulation=fp16_accumulation, force_unload=force_unload,
            upscaler_api=upscaler_api)
        summary = f'{plan.source_width}×{plan.source_height} → {summary}'
        if plan.warnings:
            summary += '\n' + '\n'.join(plan.warnings)
        return io.NodeOutput(*outputs, expand=graph, ui=ui.PreviewText(summary))


class MMH3H3UpscaleAVRestore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MMH3H3UpscaleAVRestore', display_name='H3 Restore Upscale AV',
                         category=CATEGORY, inputs=[io.Latent.Input('sampled'), io.Latent.Input('target')],
                         outputs=[io.Latent.Output('latent')], is_dev_only=True)

    @classmethod
    def execute(cls, sampled, target):
        return io.NodeOutput(restore_upscale_av(sampled, target))
