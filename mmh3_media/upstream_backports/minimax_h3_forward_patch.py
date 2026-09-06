from __future__ import annotations

import hashlib
from dataclasses import dataclass


PR_URL = "https://github.com/Comfy-Org/ComfyUI/pull/15860"
PR_COMMIT = "6410a035e3c211055f86ee7445ee1fea85783f18"


class BackportCompatibilityError(RuntimeError):
    """The installed core no longer matches the bounded backport patch points."""


@dataclass(frozen=True)
class ForwardPatchResult:
    source: str
    original_sha256: str


_OLD_SIGNATURE = (
    "def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, "
    "denoise_mask=None, audio_denoise_mask=None, **kwargs):"
)
_NEW_SIGNATURE = (
    "def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, "
    "denoise_mask=None, audio_denoise_mask=None, control=None, **kwargs):"
)
_ROPE_MARKER = (
    "    rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)\n"
)
_CONTROL_INIT = """

    # TEMPORARY UPSTREAM BACKPORT: ComfyUI PR #15860.
    control_streams = []
    if control is not None:
        audio_pos = layout.audio_pos.to(device)
        for entry in control.get("minimax_fun", ()):
            cm = entry["model"]
            control_streams.append({"model": cm, "c": cm.init_stream(h, entry["latent"].to(device), layout, t_emb),
                                    "strength": entry["strength"], "next": 0})
"""
_BLOCK_MARKER = (
    "        else:\n"
    "            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)\n"
)
_CONTROL_STEP = """        for s in control_streams:
            j = s["next"]
            if j < len(s["model"].injection_layers) and s["model"].injection_layers[j] == i:
                s["c"], skip = s["model"].step(j, s["c"], t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
                skip[audio_pos] = 0
                h.add_(skip, alpha=s["strength"])
                s["next"] = j + 1
"""


def patch_forward_source(source: str) -> ForwardPatchResult:
    """Apply the three-line-surface PR change to an inspected core method.

    This fails closed unless every expected patch point occurs exactly once. The
    installed core method remains authoritative; we do not vendor a full stale copy.
    """

    original_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    checks = {
        "legacy signature": source.count(_OLD_SIGNATURE),
        "rope marker": source.count(_ROPE_MARKER),
        "block marker": source.count(_BLOCK_MARKER),
    }
    invalid = [f"{name}={count}" for name, count in checks.items() if count != 1]
    if invalid:
        raise BackportCompatibilityError(
            "MiniMaxH3Model._forward does not match PR #15860 patch points: " + ", ".join(invalid)
        )

    patched = source.replace(_OLD_SIGNATURE, _NEW_SIGNATURE, 1)
    patched = patched.replace(_ROPE_MARKER, _ROPE_MARKER + _CONTROL_INIT, 1)
    patched = patched.replace(_BLOCK_MARKER, _BLOCK_MARKER + _CONTROL_STEP, 1)
    return ForwardPatchResult(source=patched, original_sha256=original_sha256)
