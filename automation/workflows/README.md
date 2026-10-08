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

## Runner recovery, pause and cancellation

Run `automation_runner.py` with the same ledger/checkpoint when reconnecting. The runner saves a submission intent before contacting ComfyUI, then retains the returned prompt ID and exact attempt ownership. On restart it reconciles running jobs with the queue/history before submitting anything else. A lost response or timeout keeps the attempt running; it does not automatically reroll it. Only one runner may hold a checkpoint at a time. Legacy running entries without a submission receipt require inspecting the old queue before resetting them.

Supply `--output-root <local ComfyUI output directory>` to recover finished `.mmh3` artifacts after history has been cleared. The runner inserts a metadata-only receipt before each MMH3 Save. Recovery accepts only one archive matching the exact token, job, attempt, lease, plan and settings; filenames alone are insufficient. Keep the output directory and checkpoint together. A missing/ambiguous result remains pending reconciliation rather than triggering duplicate GPU work.

From a second terminal, send a control request using the same checkpoint:

```console
python automation_runner.py --ledger ledger.json --request pause
python automation_runner.py --ledger ledger.json --request cancel
python automation_runner.py --ledger ledger.json --request resume
```

With a separate checkpoint, also pass the original `--checkpoint` value. `pause` finishes and saves the current segment, then stops before the next. `cancel` targets only that runner's remote prompt and stops after reconciliation. Cancellation requires ComfyUI's `/api/jobs/<id>/cancel` endpoint; an older host never falls back to global interruption. A result that finished concurrently with cancellation is kept. `resume` clears the control request; rerun the original generation command if the runner has already exited. Ctrl+C requests pause; a second Ctrl+C requests targeted cancellation. An optional `--control` path overrides the default `<checkpoint>.control.json`.

Decoded assembly PCM is backed by temporary disk storage. Copying, mixing, WAV read/write and final video encoding work in blocks of at most 32,768 samples, while preserving the regular ComfyUI AUDIO tensor interface. The files live until the last tensor storage reference is released; tensor consumers that explicitly materialize a full copy can still use proportional RAM. OS file caching is outside this block-buffer limit, and temporary disk space scales with audio length.

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

## Complete ComfyUI canvas

Open [Long Video Studio](../../example_workflows/mmh3_f18_long_video_studio.json) for image-reference generation over the full master-audio timeline. The Studio node materializes the canonical image-lipsync and final-assembly templates as native ComfyUI expansions. There is no second queue client or manual JSON transfer.

**Generate next → review the preview → Accept take → Generate next**, then **Assemble** when every scene is accepted. **Reroll** preserves earlier candidates; select a take in the dropdown before accepting it. **Refresh** reads the checkpoint. **Recover** checks prompt ownership and exact saved execution receipts; it refuses to restart a prompt still running or with an unknown outcome. The first run freezes the image, resampled stereo/mono master and render settings in `output/<project>`. Later runs resume this saved project; a new name starts different inputs/settings.

For batch operation set `review_takes=false` and `auto_continue=true` before creating a project. The default review mode runs one scene per prompt; automatic operation expands successive scenes within one prompt and can retain more executor cache state. Shots share identity references and audio timing, while cuts remain independent shots. Use continuation workflows for uninterrupted visual motion.

New node classes require a ComfyUI restart and browser refresh after updating this extension.
