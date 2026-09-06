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


def replace_upscale_turbo(packet, name, digest=None):
    recorded = get_generation_loras(packet)
    if recorded is None:
        raise MMH3ResourceError('Record source LoRAs before overriding Turbo; creative LoRAs must be known')
    name = name.replace('\\', '/')
    replacement = dict(name=name, strength_model=1.0, strength_clip=None,
        purpose='acceleration', loader='LoraLoaderModelOnly', reapply_for_high_sigma=True)
    if digest:
        replacement['sha256'] = digest
    sampling = packet.manifest.get('extensions', {}).get('mmh3_media', {}).get('last_process', {}).get('info', {}).get('sampling', {})
    adapter = sampling.get('adapter') or {}
    adapter_name = str(adapter.get('name') or sampling.get('recommended_lora') or '').replace('\\', '/')
    result = []
    inserted = False
    for entry in recorded:
        # Explicit creative purposes win over filename heuristics.
        accelerated = entry['purpose'] == 'acceleration' or entry['name'] == adapter_name
        if entry['purpose'] == 'unknown':
            accelerated |= any(token in entry['name'].lower() for token in
                               ('turbo', 'pdd', 'minimax_h3_acc', 'minimax-h3-acc', 'fasth3', 'fast_h3'))
        if accelerated:
            if not inserted:
                result.append(replacement)
                inserted = True
        else:
            result.append(entry)
    if not inserted:
        result.insert(0, replacement)
    return set_generation_loras(packet, result)
