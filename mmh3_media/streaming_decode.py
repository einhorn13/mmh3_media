"""Bounded H3 final decode. Temporal state belongs to one complete video.

Adapters follow ComfyUI TAEHV's serial MemBlock execution and H3VAE_TRT's
7-token/5-stride temporal blend. Never restart TAE memory at a stitched seam.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile

import torch

from .errors import MMH3ResourceError
from .h3 import h3_frame_count_from_video_t, is_nested_tensor, nested_parts

STREAM_MODES = ['off', 'auto', 'stream']
AUTO_STREAM_FRAMES = 360


def video_latent(samples):
    z = samples['samples']
    z = nested_parts(z)[0] if is_nested_tensor(z) else z
    if not isinstance(z, torch.Tensor) or z.ndim != 5 or z.shape[:2] != (1, 24):
        raise MMH3ResourceError('Streaming H3 delivery requires video latent [1,24,T,H,W]')
    frames = 1 if z.shape[2] == 1 else h3_frame_count_from_video_t(z.shape[2])
    if frames is None or min(z.shape[-2:]) < 1:
        raise MMH3ResourceError('Invalid H3 temporal/spatial latent grid')
    return z, frames


def backend_kind(model):
    if (type(model).__name__ == 'TAEHV' and getattr(model, 'is_h3', False)
            and model.patch_size == 2 and model.t_upscale == 4 and model.frames_to_trim == 3
            and model.latent_format is None):
        names = [type(b).__name__ for b in model.decoder]
        if names.count('MemBlock') == 9 and names.count('TGrow') == 3 and 'TPool' not in names:
            return 'draft'
    if (type(model).__name__ == 'MiniMaxH3TRTVAE'
            and getattr(model, 'decoder_runner', None) is not None
            and all(getattr(model, k, None) == v for k, v in {
                'tokens_chunk_size': 5, 'token_overlap': 2, 'frame_pre_padding': 3,
                'frame_overlap': 5, 'vae_ratio_t': 4, 'vae_ratio': 16,
                'token_drop': 3, 'tile_size': 256}.items())
            and all(callable(getattr(model, k, None)) for k in ('tiled_decode', 'blend', '_finalize_pixels'))):
        return 'trt'
    return None


def select_stream(mode, frames, supported):
    if mode not in STREAM_MODES:
        raise MMH3ResourceError(f'Unknown streaming mode {mode!r}')
    if mode == 'stream' and not supported:
        raise MMH3ResourceError('Streaming requires a supported native TAEH3 or H3VAE_TRT decoder; use off for ordinary decode')
    return mode == 'stream' or (mode == 'auto' and supported and frames > AUTO_STREAM_FRAMES)


def iter_taeh3(model, z, *, device, dtype):
    """Serial TAEHV execution with a bounded DFS queue and persistent memory."""
    memory = [None] * len(model.decoder)
    raw_index = 0
    for t in range(z.shape[2]):
        xt = model.process_in(z[:, :, t:t+1].to(device=device, dtype=dtype))[:, :, 0]
        queue = deque([(xt, 0)])
        del xt
        while queue:
            xt, i = queue.popleft()
            if i == len(model.decoder):
                # Native H3 removes the first 3 of every 20 raw frames. For
                # legal 5k+2 latents the terminal padding is entirely discarded.
                keep = raw_index >= 3 if z.shape[2] == 1 else raw_index % 20 >= 3
                raw_index += 1
                if keep:
                    rgb = torch.nn.functional.pixel_shuffle(xt, model.patch_size).clamp_(0, 1)
                    yield rgb[0].movedim(0, -1).float().cpu()
                    del rgb
            else:
                block = model.decoder[i]
                name = type(block).__name__
                if name == 'MemBlock':
                    result = block(xt, xt * 0 if memory[i] is None else memory[i])
                    memory[i] = xt.detach().clone()
                    queue.appendleft((result, i + 1))
                    del result
                elif name == 'TGrow':
                    grown = block(xt)
                    for part in reversed(grown.chunk(block.stride, dim=0)):
                        queue.appendleft((part, i + 1))
                    del grown, part
                else:
                    queue.appendleft((block(xt), i + 1))
            del xt


def iter_trt(model, z, *, device, dtype):
    """Keep only one spatially tiled temporal window and its five-frame tail."""
    expected = 1 if z.shape[2] == 1 else h3_frame_count_from_video_t(z.shape[2])
    overlap = None
    emitted = 0
    for start in range(0, max(1, z.shape[2] - 2), 5):
        clip = z[:, :, start:start+7].to(device=device, dtype=dtype)
        if clip.shape[2] < 7:
            clip = torch.cat((clip, clip[:, :, -1:].expand(-1, -1, 7-clip.shape[2], -1, -1)), dim=2)
        clip = clip * model.latents_std.to(clip) + model.latents_mean.to(clip)
        pixels = model.tiled_decode(clip)
        del clip
        if tuple(pixels.shape[:3]) != (1, 3, 28):
            raise MMH3ResourceError('TRT temporal tile must decode to [1,3,28,H,W]')
        if z.shape[2] == 1:
            chunk = pixels[:, :, -1:]
        else:
            chunk = pixels[:, :, 3:20]
            if overlap is not None:
                chunk = model.blend(overlap, chunk, 5, dim=-3)
            overlap = pixels[:, :, 23:28].clone()
        chunk = model._finalize_pixels(chunk)
        for t in range(min(chunk.shape[2], expected - emitted)):
            yield chunk[0, :, t].movedim(0, -1).float().cpu()
            emitted += 1
        del pixels, chunk
    if emitted < expected and overlap is not None:
        tail = model._finalize_pixels(overlap)
        for t in range(min(tail.shape[2], expected - emitted)):
            yield tail[0, :, t].movedim(0, -1).float().cpu()


def iter_decoder(vae, z, kind):
    import comfy.model_management as mm
    import comfy.model_prefetch
    model = vae.first_stage_model
    shape = (1, 24, min(z.shape[2], 7), *z.shape[-2:])
    with torch.inference_mode(), mm.cuda_device_context(vae.device):
        if kind == 'trt':
            if min(z.shape[-2:]) < 16:
                raise MMH3ResourceError('Static TRT streaming engine requires a canvas of at least 256×256')
            if model.encoder_runner is not None:
                model.encoder_runner.offload_to_ram()
            vae.patcher.current_mode = 'decode'
            vae.patcher.size = vae.dec_size
        with comfy.model_prefetch.pause_malloc_graph():
            mm.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(shape, vae.vae_dtype),
                               force_full_load=getattr(vae, 'disable_offload', False))
        if kind == 'trt':
            runner = model.decoder_runner
            runner.load_to_gpu()
            # The provider's infer ignores set_input_shape/execute return codes.
            # Validate its static contract before it can write an output buffer.
            import tensorrt as trt
            engine = runner.engine
            if (tuple(engine.get_tensor_shape('latent_tile')) != (1, 24, 7, 16, 16)
                    or tuple(engine.get_tensor_shape('pixel_tile')) != (1, 3, 28, 256, 256)
                    or engine.get_tensor_dtype('latent_tile') != trt.float16
                    or engine.get_tensor_dtype('pixel_tile') != trt.float16):
                raise MMH3ResourceError('Unsupported TRT engine IO profile for streaming decode')
        iterator = iter_trt if kind == 'trt' else iter_taeh3
        for image in iterator(model, z, device=vae.device, dtype=vae.vae_dtype):
            mm.throw_exception_if_processing_interrupted()
            yield image


class StreamingDecodedVideo:
    """VIDEO that decodes into the encoder, never into a full IMAGE batch."""
    _mmh3_skip_components_metadata = True

    def __init__(self, vae, z, frames, kind, audio, report):
        self.vae, self.z, self.frames, self.kind = vae, z, frames, kind
        self.audio = audio
        self.mmh3_decode = dict(report)
        self._temporary = None
        self.last_save_stats = {}
        if audio is not None:
            waveform = audio['waveform']
            if waveform.ndim != 3 or waveform.shape[:2] != (1, 2) or audio['sample_rate'] != 32000:
                raise MMH3ResourceError('H3 streaming requires stereo AUDIO [1,2,S] at 32000 Hz')

    def get_dimensions(self): return (self.z.shape[-1] * 16, self.z.shape[-2] * 16)
    def get_frame_count(self): return self.frames
    def get_frame_rate(self): return Fraction(24)
    def get_duration(self): return self.frames / 24
    def get_bit_depth(self): return 8
    def get_color_space(self): return 'sRGB'

    def get_components(self):
        raise MMH3ResourceError('Streaming VIDEO has no full IMAGE batch. Use streaming=off for RGB consumers.')

    def get_stream_source(self):
        if self._temporary is None:
            temporary = tempfile.TemporaryDirectory(prefix='mmh3_decode_')
            try:
                self.save_to(str(Path(temporary.name) / 'video.mp4'))
            except BaseException:
                temporary.cleanup()
                raise
            self._temporary = temporary
        return str(Path(self._temporary.name) / 'video.mp4')

    def save_to(self, path, format='auto', codec='auto', metadata=None,
                bit_depth=None, crf=None, color_space=None):
        import av
        if bit_depth not in (None, 8, 'auto') or color_space not in (None, 'sRGB'):
            raise MMH3ResourceError('Streaming H3 delivery supports 8-bit sRGB')
        container = getattr(format, 'value', format)
        encoder = getattr(codec, 'value', codec)
        if container == 'auto':
            suffix = Path(path).suffix.lower() if isinstance(path, (str, os.PathLike)) else '.mp4'
            container = {'.mkv': 'matroska', '.webm': 'webm'}.get(suffix, 'mp4')
        container = 'matroska' if container == 'mkv' else container
        encoder = ('libsvtav1' if container == 'webm' else 'h264') if encoder == 'auto' else encoder
        encoder = 'libsvtav1' if encoder == 'av1' else encoder
        count = 0
        audio_pos = 0
        # Match CreateVideo's untrimmed PCM contract, including its quantized tail.
        audio_len = self.audio['waveform'].shape[-1] if self.audio is not None else 0
        iterator = iter_decoder(self.vae, self.z, self.kind)
        try:
            options = {}
            if container == 'mp4':
                options['movflags'] = 'use_metadata_tags+faststart' if isinstance(path, (str, os.PathLike)) else 'use_metadata_tags'
            with av.open(path, mode='w', format=container, options=options) as output:
                for key, value in (metadata or {}).items():
                    output.metadata[key] = value if isinstance(value, str) else json.dumps(value)
                video = output.add_stream(encoder, rate=Fraction(24))
                video.width, video.height = self.get_dimensions()
                video.pix_fmt = 'yuv420p'
                if crf is not None:
                    video.options = {'crf': str(crf)}
                audio = output.add_stream('libopus' if container == 'webm' else 'aac', rate=32000, layout='stereo') if audio_len else None

                def write_audio(until):
                    nonlocal audio_pos
                    while audio_pos < min(until, audio_len):
                        end = min(audio_pos + 1024, audio_len, until)
                        data = self.audio['waveform'][0, :, audio_pos:end].float().cpu().contiguous().numpy()
                        frame = av.AudioFrame.from_ndarray(data, format='fltp', layout='stereo')
                        frame.sample_rate, frame.pts, frame.time_base = 32000, audio_pos, Fraction(1, 32000)
                        for packet in audio.encode(frame): output.mux(packet)
                        audio_pos = end

                for image in iterator:
                    if count >= self.frames or tuple(image.shape) != (video.height, video.width, 3):
                        raise MMH3ResourceError('Streaming decoder returned unexpected frame count or dimensions')
                    if not torch.isfinite(image).all():
                        raise MMH3ResourceError('Streaming decoder returned non-finite pixels')
                    rgb = image.clamp(0, 1).mul(255).to(torch.uint8).numpy()
                    frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                    frame.pts, frame.time_base = count, Fraction(1, 24)
                    for packet in video.encode(frame): output.mux(packet)
                    count += 1
                    write_audio(count * 32000 // 24)
                    del image, rgb, frame
                if count != self.frames:
                    raise MMH3ResourceError(f'Streaming decoded {count} frames; expected {self.frames}')
                for packet in video.encode(None): output.mux(packet)
                write_audio(audio_len)
                if audio is not None:
                    for packet in audio.encode(None): output.mux(packet)
        finally:
            iterator.close()
        self.last_save_stats = {'encoded_frames': count, 'audio_samples': audio_pos,
                                'max_temporal_latents': 7 if self.kind == 'trt' else 1}
