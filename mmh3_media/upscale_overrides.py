"""Explicit upscale-only sampling and acceleration overrides."""
import math
import re

from .errors import MMH3ResourceError
from .lora_provenance import get_generation_loras, set_generation_loras


def parse_upscale_sigmas(text):
    if not text.strip():
        return None
    try:
        values = [float(value) for value in re.split(r'[\s,]+', text.strip().strip('[]')) if value]
    except ValueError as exc:
        raise MMH3ResourceError('Sigmas must be numbers separated by commas or spaces') from exc
    if not 2 <= len(values) <= 101 or any(not math.isfinite(v) or not 0 <= v <= 3.402823466e38 for v in values):
        raise MMH3ResourceError('Provide 2–101 finite nonnegative sigmas')
    if values[-1] != 0 or any(a <= b for a, b in zip(values, values[1:])):
        raise MMH3ResourceError('Sigmas must strictly decrease and end with 0')
    return values


def _source_sampling_adapter_name(packet):
    sampling = packet.manifest.get('extensions', {}).get('mmh3_media', {}).get('last_process', {}).get('info', {}).get('sampling', {})
    adapter = sampling.get('adapter') or {}
    return str(adapter.get('name') or sampling.get('recommended_lora') or '').replace('\\', '/')


def _looks_like_acceleration(entry, *, adapter_name=''):
    """Classify only trajectory/acceleration adapters; creative purposes always win."""
    purpose = str(entry.get('purpose') or 'unknown')
    name = str(entry.get('name') or '').replace('\\', '/')
    if purpose == 'acceleration' or (adapter_name and name == adapter_name):
        return True
    if purpose != 'unknown':
        return False
    lowered = name.lower()
    return any(token in lowered for token in (
        'turbo', 'pdd', 'minimax_h3_acc', 'minimax-h3-acc', 'fasth3', 'fast_h3',
    ))


def drop_upscale_acceleration_loras(packet):
    """Remove source trajectory adapters while preserving creative/style/content LoRAs.

    VDN owns its trained trajectory adapter.  Reusing a source H3 Turbo/PDD/FastH3
    acceleration LoRA underneath VDN creates an unsupported stacked trajectory, so callers
    switching architecture families must drop only acceleration entries and keep the rest in
    exact recorded order.
    """
    recorded = get_generation_loras(packet)
    if recorded is None:
        raise MMH3ResourceError('Record source LoRAs before changing the acceleration trajectory')
    adapter_name = _source_sampling_adapter_name(packet)
    kept = [entry for entry in recorded if not _looks_like_acceleration(entry, adapter_name=adapter_name)]
    return set_generation_loras(packet, kept)


def replace_upscale_turbo(packet, name, digest=None):
    recorded = get_generation_loras(packet)
    if recorded is None:
        raise MMH3ResourceError('Record source LoRAs before overriding Turbo; creative LoRAs must be known')
    name = name.replace('\\', '/')
    replacement = dict(name=name, strength_model=1.0, strength_clip=None,
        purpose='acceleration', loader='LoraLoaderModelOnly', reapply_for_high_sigma=True)
    if digest:
        replacement['sha256'] = digest
    adapter_name = _source_sampling_adapter_name(packet)
    result = []
    inserted = False
    for entry in recorded:
        if _looks_like_acceleration(entry, adapter_name=adapter_name):
            if not inserted:
                result.append(replacement)
                inserted = True
        else:
            result.append(entry)
    if not inserted:
        result.insert(0, replacement)
    return set_generation_loras(packet, result)


__all__ = [
    'drop_upscale_acceleration_loras',
    'parse_upscale_sigmas',
    'replace_upscale_turbo',
]
