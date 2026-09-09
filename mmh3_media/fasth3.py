from __future__ import annotations

import math

from .errors import MMH3ResourceError


FASTH3_PROFILE = "fasth3 dense experimental (6 steps)"


def _read_adapter(path: str, validate=None):
    """Stream the hash, then memory-map tensors; do not keep a second adapter-sized byte buffer."""
    import hashlib
    from pathlib import Path
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = Path(path)
    before = source.stat()
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        shapes = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
    validation = validate(shapes, metadata) if validate else None
    weights = load_file(str(source), device="cpu")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise MMH3ResourceError("Adapter file changed while loading; retry from a stable file")
    return weights, metadata, digest, validation


def validate_converted_fasth3(shapes: dict, metadata: dict, base_shapes: dict) -> set[str]:
    """Validate the converted artifact against logical base shapes, without executing tensors.

    Converter metadata identifies the conversion, not the student's training provenance.
    The caller must select a conversion of dense-datafree, never a converted VSA student.
    """
    if any("to_gate_compress" in k or "vsa" in k.lower() for k in shapes):
        raise MMH3ResourceError("VSA adapters require a different architecture; use dense-datafree")
    if metadata.get("converted_by") != "lora_convert_h3" or metadata.get("adaln_mode") != "drop":
        raise MMH3ResourceError("FastH3 requires lora_convert_h3 output with adaln_mode=drop")
    if not all(any(k.startswith(f"diffusion_model.{name}") for k in base_shapes)
               for name in ("video_patch_proj.", "audio_patch_proj.")):
        raise MMH3ResourceError("FastH3 requires a MiniMax H3 joint-AV base model")
    suffixes = (".lora_A.weight", ".lora_B.weight", ".alpha", ".diff_b", ".diff")
    modules: dict[str, dict] = {}
    for key, shape in shapes.items():
        suffix = next((s for s in suffixes if key.endswith(s)), None)
        if not key.startswith("diffusion_model.") or suffix is None or "adaln_proj" in key:
            raise MMH3ResourceError(f"Unsupported FastH3 tensor: {key}")
        modules.setdefault(key[:-len(suffix)], {})[suffix] = tuple(shape)
    targets: set[str] = set()
    pairs = 0
    for module, parts in modules.items():
        weight, bias = module + ".weight", module + ".bias"
        a, b = parts.get(".lora_A.weight"), parts.get(".lora_B.weight")
        if a is not None or b is not None:
            expected = tuple(base_shapes.get(weight, ()))
            if a is None or b is None or len(a) != 2 or len(b) != 2 or len(expected) != 2:
                raise MMH3ResourceError(f"Incomplete/unmapped FastH3 pair: {module}")
            if a[0] < 1 or b[1] != a[0] or (b[0], a[1]) != expected:
                raise MMH3ResourceError(f"FastH3 shape mismatch: {module}")
            if ".diff" in parts:
                raise MMH3ResourceError(f"Conflicting LoRA and full delta: {module}")
            pairs += 1
            targets.add(weight)
        if ".alpha" in parts and (a is None or math.prod(parts[".alpha"]) != 1):
            raise MMH3ResourceError(f"Invalid FastH3 alpha: {module}")
        for suffix, target in ((".diff", weight), (".diff_b", bias)):
            if suffix in parts:
                if target not in base_shapes or parts[suffix] != tuple(base_shapes[target]):
                    raise MMH3ResourceError(f"FastH3 delta shape mismatch: {target}")
                targets.add(target)
    if not pairs:
        raise MMH3ResourceError("FastH3 adapter contains no usable LoRA pairs")
    return targets


def apply_fasth3(model, path: str, name: str):
    """Runtime-only, CPU adapter loading; reject partial applications and stacked patches."""
    import torch
    import comfy.lora
    from .optimization_contract import require_unoptimized_sampling_model
    require_unoptimized_sampling_model(model)

    if getattr(model, "patches", None) or getattr(model, "object_patches", None):
        raise MMH3ResourceError("FastH3 experimental profile requires an unpatched base model")
    if getattr(model, "model_options", {}).get("transformer_options"):
        raise MMH3ResourceError("Apply FastH3 before attention/cache patches")
    base_shapes = {k: tuple(v.shape) for k, v in model.model.state_dict().items()}
    weights, metadata, digest, expected = _read_adapter(
        path, lambda shapes, meta: validate_converted_fasth3(shapes, meta, base_shapes))
    for key, tensor in weights.items():
        if key.endswith(".alpha") and (not torch.isfinite(tensor).all() or tensor.item() <= 0):
            raise MMH3ResourceError(f"Invalid FastH3 alpha value: {key}")
    mapping = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(weights, mapping, log_missing=False)
    if set(patches) != expected:
        raise MMH3ResourceError("FastH3 mapping is incomplete in this ComfyUI version")
    patched = model.clone()
    applied = set(patched.add_patches(patches, 1.0))
    if applied != expected:
        raise MMH3ResourceError("FastH3 was only partially applied; the base model is unchanged")
    provenance = {"name": name.replace("\\", "/"), "sha256": digest,
                  "strength_model": 1.0, "strength_clip": None, "purpose": "acceleration",
                  "loader": "LoraLoaderModelOnly", "source": "NikoDemon80/lora_convert_h3",
                  "conversion": metadata, "profile": FASTH3_PROFILE,
                  "validated_target_count": len(applied), "runtime_validated": False}
    patched.set_attachments("mmh3_sampling_adapter", FASTH3_PROFILE)
    return patched, provenance


def apply_packaged_turbo(model, recommended_name: str):
    """Couple a known packaged recipe to one installed adapter; never guess a replacement."""
    from pathlib import PurePosixPath
    import folder_paths
    import comfy.lora
    from .optimization_contract import require_unoptimized_sampling_model
    require_unoptimized_sampling_model(model)
    import comfy.lora_convert

    if getattr(model, "patches", None) or getattr(model, "object_patches", None):
        raise MMH3ResourceError("Automatic Turbo loading requires the unpatched base model")
    basename = PurePosixPath(recommended_name.replace("\\", "/")).name
    matches = [n for n in folder_paths.get_filename_list("loras")
               if PurePosixPath(n.replace("\\", "/")).name == basename]
    if len(matches) != 1:
        raise MMH3ResourceError(f"Turbo requires exactly one installed {basename}; found {len(matches)}")
    name = matches[0]
    path = folder_paths.get_full_path("loras", name)
    weights, _, digest, _ = _read_adapter(path)
    weights = comfy.lora_convert.convert_lora(weights)
    mapping = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(weights, mapping)
    if not patches:
        raise MMH3ResourceError("Turbo adapter did not map to this base model")
    patched = model.clone()
    if set(patched.add_patches(patches, 1.0)) != set(patches):
        raise MMH3ResourceError("Turbo adapter was only partially applied")
    return patched, {"name": name.replace("\\", "/"), "sha256": digest,
                     "strength_model": 1.0, "strength_clip": None,
                     "purpose": "acceleration", "loader": "LoraLoaderModelOnly"}
