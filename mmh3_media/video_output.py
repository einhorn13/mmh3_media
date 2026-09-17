"""One encoded delivery file shared by video output and an MMH3 archive."""
from __future__ import annotations

import copy
import os
import uuid
from dataclasses import replace
from pathlib import Path

from .errors import MMH3ResourceError

DECODE_MODES = ['vae', 'draft', 'trt']
TRT_LOADER = 'MiniMaxH3TRTVAELoader'


def trt_decoder_options():
    import folder_paths
    files = set()
    for root in folder_paths.get_folder_paths('vae'):
        base = Path(root)
        files.update(p.relative_to(base).as_posix() for p in base.rglob('*.engine')
                     if p.is_file() and not ('encoder' in p.name.lower() and 'decoder' not in p.name.lower()))
    return ['auto', *sorted(files)]


def resolve_trt_decoder(runtime_nodes, selected):
    loader = runtime_nodes.get(TRT_LOADER)
    if loader is None:
        raise MMH3ResourceError('TRT decode requires ComfyUI-H3VAE_TRT and a compiled H3 decoder engine')
    schema = loader.INPUT_TYPES().get('required', {})
    if not {'decoder', 'encoder'} <= schema.keys() or getattr(loader, 'RETURN_TYPES', ()) != ('VAE',):
        raise MMH3ResourceError('Unsupported MiniMaxH3TRTVAELoader contract')
    options = list(schema['decoder'][0])
    engines = [p for p in options if str(p).lower().endswith('.engine')]
    if selected == 'auto':
        candidates = [p for p in engines if 'decod' in str(p).lower()]
        if len(candidates) != 1:
            raise MMH3ResourceError('Select one compiled H3 TRT decoder engine explicitly; auto needs exactly one decoder')
        selected = candidates[0]
    matches = [p for p in engines if p.replace('\\', '/') == selected.replace('\\', '/')]
    if len(matches) != 1:
        raise MMH3ResourceError(f'TRT decoder engine {selected!r} is not available')
    if 'encoder' in matches[0].lower() and 'decoder' not in matches[0].lower():
        raise MMH3ResourceError('Select a TRT decoder engine, not an encoder engine')
    return loader, matches[0]


def decode_report(vae, mode, engine=None):
    model = getattr(vae, 'first_stage_model', None)
    name = type(model).__name__ if model is not None else type(vae).__name__
    # Report the actual supplied decoder even if a custom graph wires it to VAE.
    runner = getattr(model, 'decoder_runner', None)
    if runner is not None:
        mode = 'trt'
        engine = engine or Path(runner.model_path).name
    elif 'taeh' in name.lower():
        mode = 'draft'
    return {'version': 1, 'mode': mode, 'quality': 'draft' if mode == 'draft' else 'full',
            'backend': 'tensorrt' if mode == 'trt' else 'pytorch', 'decoder_class': name,
            'weights': engine if mode == 'trt' else TAEH3_FILENAME if mode == 'draft' else 'connected_vae'}


class DecodedVideo:
    """VIDEO adapter carrying explicit decode provenance until it is archived."""
    def __init__(self, video, report):
        self.video = video
        self.mmh3_decode = copy.deepcopy(report)

    def __getattr__(self, name):
        return getattr(self.video, name)


def video_decode_metadata(video):
    report = getattr(video, 'mmh3_decode', None)
    if report is None:
        return None
    if not isinstance(report, dict) or report.get('mode') not in DECODE_MODES or report.get('version') != 1:
        raise MMH3ResourceError('Invalid video decode provenance')
    return copy.deepcopy(report)


class SavedVideo:
    """File-backed VIDEO; refuse to archive a deleted/replaced output file."""

    def __init__(self, path, reader):
        self.path = Path(path).resolve()
        self.reader = reader
        self.stamp = self._stamp()

    def _stamp(self):
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def _mmh3_existing_video_path(self):
        try:
            if self._stamp() == self.stamp:
                return self.path
        except OSError:
            pass
        raise MMH3ResourceError("Saved video changed or was removed; run Save Video again before saving MMH3")

    def get_stream_source(self):
        return str(self._mmh3_existing_video_path())

    def get_components(self):
        self._mmh3_existing_video_path()
        return self.reader.get_components()

    def save_to(self, *args, **kwargs):
        self._mmh3_existing_video_path()
        return self.reader.save_to(*args, **kwargs)

    def __getattr__(self, name):
        # Transform methods belong to the reader and return a new VIDEO, without
        # our byte-copy capability. Crops/audio edits must never reuse old bytes.
        return getattr(self.reader, name)


def save_output_video(video, path, *, packet=None, reader_factory, **encode_options):
    """Encode once, then bind exactly the matching packet payloads to its file.

    Other resources (including original latent and lossless PCM) stay untouched.
    Replacing the storage representation does not create a new semantic revision,
    just as archive serialization itself does not change resource revisions.
    """
    matches = [] if packet is None else [
        res['id'] for res in packet.resources()
        if res['kind'] == 'video' and packet.payloads.get(res['id']) is video
    ]
    if packet is not None and not matches:
        raise MMH3ResourceError("Save Video needs the packet containing this exact VIDEO; connect the matching Pack/delivery output")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        video.save_to(str(temporary), **encode_options)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise MMH3ResourceError("Video encoder did not produce a non-empty output file")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    saved = SavedVideo(path, reader_factory(str(path)))
    if packet is None:
        return saved, None
    payloads = dict(packet.payloads)
    for rid in matches:
        payloads[rid] = saved
    return saved, replace(packet, manifest=copy.deepcopy(packet.manifest), payloads=payloads, dirty=True)


TAEH3_FILENAME = "taeh3.safetensors"
TAEH3_URL = "https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taeh3.safetensors"


def validate_taeh3_file(path):
    """Reject the older 2D preview weights and other video model families."""
    if path is None or not Path(path).is_file():
        raise MMH3ResourceError(
            f"Draft TAE needs models/vae_approx/{TAEH3_FILENAME}. Download {TAEH3_URL} or switch Draft TAE off.")
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        keys = weights.keys()
        if ("decoder.1.weight" not in keys or "decoder.22.bias" not in keys
                or weights.get_slice("decoder.1.weight").get_shape()[1:] != [24, 3, 3]
                or weights.get_slice("decoder.22.bias").get_shape() != [12]):
            raise MMH3ResourceError("Draft TAE requires madebyollin temporal taeh3 weights, not the older 2D MiniMax preview model.")


def build_video_decode_graph(samples, vae, *, draft_tae, runtime_nodes, graph):
    if "VAEDecode" not in runtime_nodes:
        raise MMH3ResourceError("Missing VAEDecode. Update ComfyUI.")
    if draft_tae:
        loader = runtime_nodes.get("VAELoader")
        if "taeh3" not in getattr(loader, "video_taes", ()):
            raise MMH3ResourceError("Draft TAE requires ComfyUI with native taeh3 support. Update ComfyUI or switch Draft TAE off.")
        vae = graph.node("VAELoader", vae_name=TAEH3_FILENAME).out(0)
    decoded = graph.node("VAEDecode", samples=samples, vae=vae)
    return decoded.out(0), graph.finalize()
