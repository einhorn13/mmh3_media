"""File-backed PCM keeps the Comfy AUDIO tensor contract without a giant concat."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from .errors import MMH3ResourceError

PCM_BLOCK_SAMPLES = 32768


class _OwnedPCM(np.memmap):
    def __new__(cls, channels, samples, dtype):
        temporary = tempfile.TemporaryDirectory(prefix="mmh3_pcm_")
        try:
            result = super().__new__(cls, Path(temporary.name) / "pcm.bin", dtype=dtype,
                                     mode="w+", shape=(1, channels, samples))
            result._temporary = temporary
            return result
        except BaseException:
            temporary.cleanup()
            raise

    def __array_finalize__(self, parent):
        super().__array_finalize__(parent)
        self._temporary = None  # Only the root mapping owns the directory.

    def __del__(self):
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            # Torch storage retains the root ndarray, including through tensor views.
            self._mmap.close()
            temporary.cleanup()


def allocate_pcm(channels, samples, *, dtype=torch.float32):
    if not 1 <= channels <= 32 or samples < 1 or dtype not in {torch.float32, torch.float64}:
        raise MMH3ResourceError("File-backed PCM requires positive samples and float32/64 audio")
    array = _OwnedPCM(channels, samples, np.float64 if dtype == torch.float64 else np.float32)
    return torch.from_numpy(array)


def pcm_blocks(waveform, *, start=0, end=None):
    end = waveform.shape[-1] if end is None else end
    for position in range(start, end, PCM_BLOCK_SAMPLES):
        yield position, waveform[..., position:min(position + PCM_BLOCK_SAMPLES, end)]


def copy_pcm(destination, start, source):
    if destination.shape[:-1] != source.shape[:-1] or start < 0 or start + source.shape[-1] > destination.shape[-1]:
        raise MMH3ResourceError("PCM copy shape/ownership mismatch")
    for position, block in pcm_blocks(source):
        destination[..., start + position:start + position + block.shape[-1]].copy_(block)


def finite_pcm(waveform):
    return all(torch.isfinite(block).all().item() for _, block in pcm_blocks(waveform))


def limit_pcm(waveform, enabled):
    if not finite_pcm(waveform):
        raise MMH3ResourceError("Delivery audio contains NaN or Inf")
    peak = max((float(block.abs().max()) for _, block in pcm_blocks(waveform)), default=0.0) if enabled else 0.0
    scale = 0.98 / peak if peak > 0.98 else 1.0
    if scale != 1.0:
        for _, block in pcm_blocks(waveform):
            block.mul_(scale)
    return scale
