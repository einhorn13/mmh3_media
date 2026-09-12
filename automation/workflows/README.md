# MMH3 automation workflow templates

This directory contains the canonical ComfyUI API-format templates required by installed F18 automation. They are not user-facing canvas examples and must not be copied into `tests/fixtures/workflows/`.

Long-video audio-sync is intentionally staged. F18 supports two source shapes:

- **Video-reference:** `mmh3_f18_long_video_lipsync_setup_api.json` + `mmh3_f18_long_video_lipsync_api.json`. The source VIDEO provides body/camera motion and its soundtrack provides timing.
- **Image-reference + master song:** `mmh3_f18_image_lipsync_setup_api.json` + `mmh3_f18_image_lipsync_api.json`. The packet keeps Ref2VA image references plus an explicit primary 32 kHz master AUDIO. This setup now starts an **interactive timeline**: each accepted shot owns an arbitrary user-facing duration, while F18 derives the larger legal H3 `17n+5` generation window and exact master-audio slice automatically.

## Interactive music-video flow

1. Build/save the source MMH3 packet. Keep the final song as the explicit primary AUDIO and character images as references.
2. Queue `mmh3_f18_image_lipsync_setup_api.json`. The first shot defaults to 5 s and candidate review is required.
3. Run the chunk template. The preview VIDEO uses the original driven audio slice so lipsync is easy to judge, while the saved MMH3 candidate stores the **decoded H3-generated audio** from the joint AV latent.
4. Review the candidate. Use `accept_candidate` to approve it or `reroll` to preserve it and generate another attempt.
5. After acceptance, use `MMH3 Automation Ledger` action `append_next` and set `next_shot_duration_seconds` (for example `3.5`, `5`, or `8`). F18 starts exactly at the previous accepted audio/frame cursor, chooses a legal H3 window, and extracts the correct absolute master-audio slice. Appending is blocked while the current shot is pending/running/review/failed.
6. Repeat generate -> review -> accept -> append. When the current prefix is enough, use `finalize_sequence`; if the full master has been covered, finalization is automatic. Early finalization records the untouched song tail explicitly as excluded rather than silently dropping it.
7. Assemble only after finalization. The assembly gate remains fail-closed for unaccepted candidates, ownership gaps/overlaps, changed artifacts, or an open interactive sequence.
8. Upscale only after edit/selection lock. Prefer timeline-preserving decoded/VSR upscale first; any generative refine should be followed by lipsync revalidation.

The execution templates derive `__MMH3_ATTEMPT_SEED__` from the job ID and attempt number. A reroll gets a different seed and save prefix; rematerializing the same attempt remains deterministic. An earlier candidate can still be accepted after a failed/cancelled reroll or before starting another one, but never while a generation lease is running.

Chunk nodes receive the frozen `settings_json` and check primary audio ID/revision before conditioning. Replacing the master requires a new setup. The optional inputs preserve older API workflows; new audio-sync templates always connect them. Existing v2 proof records without the newer review/source fields remain readable only against their original legacy settings.

Lipsync templates use **H3 Audio VAE · Preserve Onset** before the native audio guide. This prevents generic VAE center cropping on affected H3 runtimes, leaving H3's own right-padding behavior intact. The guard uses a private VAE wrapper; it does not alter the original song or the shared loader. MMH3 reference conditioning, decoded continuation, bridge and decoded latent-upscale encoding apply the same guard internally.

## Final audio modes

`MMH3 Long Video H3 Audio Sync Settings` sets the initial delivery policy. Delivery mixing is intentionally decoupled from the H3 generation proof: after candidates are generated/reviewed, `MMH3 Automation Ledger` action `set_audio_delivery` can switch mode/gains before assembly without regenerating video or invalidating lipsync provenance. Do not change delivery policy while a generation lease is running.

Available delivery policies:

- `master_only` (safe default): deliver only the immutable original song.
- `master_plus_generated`: mix the immutable song with the selected candidates' decoded H3 audio. Defaults are `master_gain_db=0` and `generated_gain_db=-18`. This can retain footsteps, clothing noise, breathing and ambience, but the generated layer is the **full H3 mix**, not an isolated FX stem, so it may also contain generated vocal/music energy.
- `generated_only`: deliver only the selected candidates' decoded H3 audio while still using the original song as the timing/ownership master.

The mixer preserves the master channel count (mono/stereo), performs deterministic mono/stereo conversion for generated audio, checks exact PCM ownership, tolerates only a small decode-tail shortfall (up to 250 ms, padded with silence), and can apply deterministic peak safety normalization after mix. `master_only` with 0 dB gain is left bit-for-bit at tensor level apart from archive/container encoding.

The image-reference path treats separate shots as independent performance shots. It does not claim hidden-state continuity across cuts; use the project's continuation/latent-handoff tools when one visually continuous take must span several H3 generations.

`mmh3_f18_chunk_runner_api.json` is the generic chunk-only execution template. `mmh3_f18_batch_stitch_setup_api.json` is the canonical batch-stitch pre-queue/setup template and is retained because it represents the unique preflight-estimate automation scenario.
