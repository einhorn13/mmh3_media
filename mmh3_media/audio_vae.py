from __future__ import annotations

from copy import copy
from typing import Any


def preserve_h3_audio_onset(vae: Any) -> Any:
    """Disable generic center cropping on a private H3 VAE wrapper.

    H3's encoder owns right padding to its 800-sample hop. Cropping before
    encoding drops leading PCM. Do not mutate the shared loader output, and
    leave other VAEs and host versions already containing the fix untouched.
    """
    if (getattr(vae, "audio_sample_rate", None) != 32000
            or getattr(vae, "latent_channels", None) != 32
            or getattr(vae, "downscale_ratio", None) != 800
            or not getattr(vae, "crop_input", False)):
        return vae
    private = copy(vae)
    private.crop_input = False
    return private
