# MMH3 automation workflow templates

This directory contains the canonical ComfyUI API-format templates required by installed F18 automation. They are not user-facing canvas examples and must not be copied into `tests/fixtures/workflows/`.

Long-video audio-sync is intentionally staged:

1. Queue `mmh3_f18_long_video_audio_driven_setup_api.json` or `mmh3_f18_long_video_lipsync_setup_api.json` after setting the source packet and ledger prefix.
2. Run the resulting ledger with `automation_runner.py --workflow automation/workflows/mmh3_f18_long_video_<mode>_api.json --ledger <ledger.json> ...`.
3. When the ledger is complete, pass the matching `--assembly-workflow automation/workflows/mmh3_f18_long_video_<mode>_assembly_api.json`.

`mmh3_f18_chunk_runner_api.json` is the generic chunk-only execution template. `mmh3_f18_batch_stitch_setup_api.json` is the canonical batch-stitch pre-queue/setup template and is retained because it represents the unique preflight-estimate automation scenario.
