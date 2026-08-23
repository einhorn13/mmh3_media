FORMAT_NAME = "MMH3_MEDIA"
SCHEMA_VERSION = 1
FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENT_FPS = 40
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STEREO_CHANNELS = 2
LATENT_SPATIAL_DIVISOR = 16
DIT_SPATIAL_PATCH = 2

SUPPORTED_KINDS = {"latent", "image", "video", "audio", "mask", "json"}
KNOWN_ROLES = (
    "h3_av_latent",
    "video",
    "audio",
    "first_frame",
    "last_frame",
    "picture_ref",
    "video_ref",
    "audio_ref",
    "preview",
    "mask",
    "continuation_context",
    "custom",
)

SINGLETON_ROLES = {
    "h3_av_latent",
    "video",
    "audio",
    "first_frame",
    "last_frame",
    "preview",
}
ORDERED_ROLES = {"picture_ref", "video_ref", "audio_ref", "mask", "continuation_context"}
ROLE_KIND_COMPAT = {
    "h3_av_latent": {"latent"},
    "video": {"video"},
    "audio": {"audio"},
    "first_frame": {"image"},
    "last_frame": {"image"},
    "picture_ref": {"image"},
    "video_ref": {"video"},
    "audio_ref": {"audio"},
    "preview": {"image"},
    "mask": {"mask"},
    # Continuation adapters may carry a latent, a preservation mask, or JSON handover state.
    "continuation_context": {"latent", "mask", "json"},
    "custom": SUPPORTED_KINDS,
}

VERIFY_MODES = ("manifest", "on_access", "full")

H3_LATENT_CONTEXT_VERSION = 1
H3_LATENT_ORIGINS = ("unknown", "sampler_output", "vae_encoded", "derived")
H3_TEMPORAL_GRID = "h3_causal_17k5"

PREVIEW_CACHE_ROLE = "preview"
PREVIEW_CACHE_VERSION = 1
