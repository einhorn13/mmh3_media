"""Explicit Turbo adapter selection shared by sampling and source refinement."""
import json
import math

from .errors import MMH3ResourceError


def turbo_lora_mode(text):
    if not text or not text.strip():
        return "auto"
    config = json.loads(text)
    return config.get("mode", "auto")


def parse_turbo_loras(text):
    if not text or not text.strip():
        return None
    try:
        config = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise MMH3ResourceError("Invalid Turbo LoRA configuration JSON") from exc
    if not isinstance(config, dict) or config.get("version") != 1:
        raise MMH3ResourceError("Unsupported Turbo LoRA configuration")
    mode = config.get("mode")
    if mode == "auto":
        return None
    if mode == "disabled":
        return []
    if mode not in {"custom", "extension"} or not isinstance(config.get("loras"), list):
        raise MMH3ResourceError("Turbo LoRA mode must be auto, custom, extension or disabled")
    result, names = [], set()
    for item in config["loras"]:
        if not isinstance(item, dict):
            raise MMH3ResourceError("Turbo LoRA entries must be objects")
        name = str(item.get("name") or "").strip().replace("\\", "/")
        try:
            strength = float(item.get("strength", 1))
        except (ValueError, TypeError) as exc:
            raise MMH3ResourceError("Turbo LoRA strength must be a finite number") from exc
        if not name or name == "None" or name in names:
            raise MMH3ResourceError("Select distinct Turbo LoRA files")
        if not math.isfinite(strength) or not -10 <= strength <= 10:
            raise MMH3ResourceError("Turbo LoRA strength must be finite and within -10..10")
        names.add(name)
        if strength:
            result.append({"name": name, "strength": strength})
    if not config["loras"]:
        raise MMH3ResourceError("Select at least one custom Turbo LoRA or choose Disabled")
    return result


def replace_source_turbo_loras(packet, selection, *, extension=False):
    from .lora_provenance import get_generation_loras, set_generation_loras
    from .upscale_overrides import _looks_like_acceleration, _source_sampling_adapter_name
    recorded = get_generation_loras(packet)
    if recorded is None:
        raise MMH3ResourceError("Record source LoRAs before overriding Turbo; creative LoRAs must be known")
    replacements = [dict(name=e["name"], strength_model=e["strength"], strength_clip=None,
                         purpose="acceleration", loader="LoraLoaderModelOnly") for e in selection]
    if extension:
        return set_generation_loras(packet, list(recorded) + replacements)
    result, inserted = [], False
    adapter_name = _source_sampling_adapter_name(packet)
    for entry in recorded:
        if _looks_like_acceleration(entry, adapter_name=adapter_name):
            if not inserted:
                result.extend(replacements)
                inserted = True
        else:
            result.append(entry)
    if not inserted:
        result = replacements + result
    return set_generation_loras(packet, result)
