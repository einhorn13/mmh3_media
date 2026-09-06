FORMAT_NAME = "MMH3_MEDIA"
SCHEMA_VERSION = 2
FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENT_FPS = 40
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STEREO_CHANNELS = 2
LATENT_SPATIAL_DIVISOR = 16
DIT_SPATIAL_PATCH = 2

REFERENCE_PRESETS = ("all", "balanced", "identity", "motion", "style", "voice")
REFERENCE_PURPOSES = ("unknown", "identity", "motion", "style", "detail", "voice", "environment", "other")
VERIFY_MODES = ("manifest", "on_access", "full")

H3_LATENT_ORIGINS = ("unknown", "sampler_output", "vae_encoded", "derived")
H3_TEMPORAL_GRID = "h3_causal_17k5"
