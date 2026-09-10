# ComfyUI-MMH3-Media

**Generate videos with audio, continue scenes, and assemble clips with MiniMax H3 in ComfyUI.**

MMH3 Media is a set of custom nodes and ready-to-use workflows. Generate videos from text, first and last frames, or references; try different continuations; combine segments; and upscale the result.

Results are saved as `.mmh3` files. Along with the media, the file stores settings, references, and the internal generation state — latents. This lets you return to a project later and continue the scene. For playback and publishing, use the workflow's video output and **Save Video**; `.mmh3` is a working project file, not a video-player format.

## What you can do

- Generate video with audio from text, keyframes, or image/video/audio references.
- Continue saved generations, compare takes, and assemble compatible segments.
- Upscale a joint AV latent or refine decoded video in spatial tiles.
- Edit video and audio independently, with explicit masks and preservation rules.
- Generate a new AV section between two clips with **Two-Clip AV Bridge**.
- Address references by names such as `@hero_face` and activate them by scene.

[Installation](#installation) · [First video](#your-first-video) · [Workflows](#choosing-a-workflow) · [AV editing](#independent-video-and-audio-editing) · [Bridge](#two-clip-av-bridge) · [Named references](#named-references-by-scene) · [Troubleshooting](#saving-and-faq)

## Installation

1. Set up ComfyUI with MiniMax H3 support and the models used by the examples. This package does not include model weights.
2. Download the repository via **Code → Download ZIP** and extract it to `ComfyUI/custom_nodes/ComfyUI_mmh3_media`. The `__init__.py` file must be located directly inside this folder, without an extra nested repository directory.
3. Place the models in the folders listed below and select them in the workflow loaders. The table lists the files used by the bundled examples; other variants require a compatible ComfyUI setup.
4. Restart ComfyUI, refresh the browser, and open a JSON file from [example_workflows](example_workflows).

Alternatively, install with Git from your ComfyUI directory:

```console
cd custom_nodes
git clone https://github.com/einhorn13/mmh3_media.git ComfyUI_mmh3_media
```

Use either Git or ZIP installation in a single folder; duplicate copies can register the same nodes twice.

The basic nodes use ComfyUI's existing Python dependencies. Keep the host's compatible PyTorch/CUDA environment; this package does not provide a separate model runtime. Specialized workflows may require external nodes and models. A ComfyUI installation without MiniMax H3 support is not sufficient.

### Updating

For a Git installation, run inside the installed `ComfyUI_mmh3_media` directory:

```console
git pull --ff-only
```

Restart ComfyUI, refresh the browser, and reopen the updated example JSON. Keep your customized workflows separately so you can compare their settings with the new examples. For a ZIP installation, replace the old node folder with the new release while retaining any personal files separately.

### Models for your first run

| Component | File used in examples | Folder inside ComfyUI |
| --- | --- | --- |
| Text/frame generation | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| Reference-based generation | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| Text Encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |
| Turbo for text/frames | `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors` | `models/loras/` |
| Turbo for references | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | `models/loras/` |

To get started, choose one generation path; you do not need both diffusion models at the same time. By default, **H3 Sampling** uses the 4-step Turbo profile and automatically applies the matching LoRA. If the Turbo model is unavailable, select **Standard (20 steps)**. Do not manually apply the same Turbo LoRA again.

## Your first video

1. Drag [mmh3_f01_fl2va.json](example_workflows/mmh3_f01_fl2va.json) onto the ComfyUI canvas.
2. Select the models you installed in the loaders.
3. In **MMH3 Create**, select **Video (optional frames)** and write a prompt describing what happens in the scene, how the camera moves, and what audio you want.
4. Connect the frames you need, or leave both frame inputs disconnected for text-only generation.
5. In **H3 Video Settings**, set the aspect ratio, resolution, and duration. For your first test, use a short clip at a modest resolution.
6. Select a profile in **H3 Sampling**, run the workflow, and review the result. Save the `.mmh3` file with **MMH3 Save** if you plan to continue the scene.

| Connected frames | Result |
| --- | --- |
| None | Text-to-video — T2VA |
| First only | Animation from the starting image — I2VA |
| Last only | Video with a specified final frame — L2VA |
| First and last | Transition between two frames — FL2VA |

For reference-based generation, open [mmh3_f01_ref2va.json](example_workflows/mmh3_f01_ref2va.json), select the Ref2VA model, and add images, video, or audio to the **References to video** inputs. New inputs appear as you connect them. **Reference Image Size** controls the size of image references.

## Choosing a workflow

The [example_workflows](example_workflows) directory contains JSON files to open on the ComfyUI canvas. [F18 automation templates](automation/workflows/README.md) use API format and follow a separate setup → execution → assembly process.

### Generation and continuation

| Workflow | Purpose |
| --- | --- |
| [F01 · Text and Frames](example_workflows/mmh3_f01_fl2va.json) | Create the first video from a prompt and an optional first and/or last frame. |
| [F01 · References](example_workflows/mmh3_f01_ref2va.json) | Generate using images, video, and audio as references. |
| [F02 · Continuation](example_workflows/mmh3_f02_continuation.json) | Continue a saved H3 segment from an `.mmh3` file; an end frame can also be specified. |
| [F02 · Continuation with References](example_workflows/mmh3_f02_ref2va_continuation.json) | Generate a continuation with Ref2VA and active references. |
| [F03 · Decoded Video Continuation](example_workflows/mmh3_f03_decoded_continuation.json) | Continue a video without a saved H3 state. `missing_audio_policy=error` is strict by default; select `silence` only to permit missing or too-short source audio. This uses VAE re-encoding, so the handover is not a lossless copy. |
| [F04 · Segment Workflow](example_workflows/mmh3_f04_chain_append.json) | One workspace for Continue, Reroll accepted, New scene and Reanchor from a new first frame, with Draft → Accept review. |

### Stitching and upscaling

| Workflow | Purpose |
| --- | --- |
| [F05 · Video Stitch](example_workflows/mmh3_f05_stitch.json) | Combine finished clips with audio. Supports Auto Seamless, Cut, and Crossfade, plus color matching at the transition. |
| [F05 · Latent Stitch](example_workflows/mmh3_f05_latent_stitch.json) | Assemble sequential H3 continuations, remove the repeated opening context, and decode the final video with audio only once. |
| [F05 · Upscale + Stitch](example_workflows/mmh3_f05_latent_stitch_upscale.json) | Upscale a linked continuation chain and assemble it into a single video. Requires the external `MinimaxH3LatentUpscaler3D` node and its weights. |
| [F07 · Latent Upscale](example_workflows/mmh3_f07_latent_upscale.json) | Upscale an H3 latent and refine the result. Optional SLA is selected in **H3 Optimizations**; the normal default does not require the SLA backend. |
| [F07 · Native Tile Upscale](example_workflows/mmh3_f07_native_tile_upscale.json) | Refine decoded video in spatial tiles, then re-encode it while retaining the source audio latent. This includes VAE round trips. |
| [F07 · Progressive Handoff](example_workflows/mmh3_f07_progressive_handoff.json) | Experimental single-schedule low-grid → learned 3D handoff at the target resolution. Requires Flow-Aligned Regenerate and the maintained Upscaler-Plus provider. |

The ordinary F07 examples use the external `MinimaxH3LatentUpscaler3D` node and its weights through **MMH3 H3 Learned Upscale**. The adapter supports both the legacy node schema and the maintained Upscaler-Plus schema. Full-sequence temporal inference is the default; legacy temporal chunking remains an explicit low-VRAM fallback because it can change temporal normalization/context. `offload_after_upscale` is enabled in the examples to release learned-upscaler VRAM before the H3 refine pass.

The separate Progressive Handoff example requires [`MiniMax-H3-Flow-Aligned-Regenerate`](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate) and [`Comfyui_Minimax_h3_latent_Upscaler-Plus`](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus). It starts with the conservative tested settings `source_scale=0.70`, fixed handoff `0.35`, direction-only guidance and learned 3D transfer. Treat it as experimental and compare decoded media with the ordinary F01 → F07 path.

**Refine step override:** in **MMH3 H3 Upscale Refine Sampling**, set `steps_override` to `0` to inherit the source count, or a positive number to choose the actual number of second-pass sampling steps. `denoise_override=0` keeps source-aware defaults; explicit values may use the full `0–1` denoise domain, with `0.05–0.50` retained as the recommended upscale-refine range. Values above that range can replace source structure. Step count and denoise are separate controls: 8 steps at denoise 0.25 still runs 8 sampling steps, using the last 8 intervals of a 32-step scheduler grid. The summary and saved process report record the effective steps and override. Turbo step overrides are experimental; more steps are not necessarily better. These controls are shared by the separate Latent Upscale and Native Tile examples.

**F07 settings and checks:** **H3 Upscale Settings** owns the exact upscaler/refine checkpoint filenames, device, precision and audio delivery policy. Its outputs drive both the working nodes and the saved report. For custom checkpoint names, declare `model_family` explicitly if `auto` cannot identify FL2VA or Ref2VA. **H3 Upscale Preflight** is enabled in the example and checks source provenance, geometry, checkpoint family, step/denoise values, required nodes and the selected Optimizations profile before execution. Its diagnostic output is also recorded in the process report. It does not estimate available VRAM or prove visual quality.

**Sigma modes:** `schedule_mode=regenerated_tail` is the default described above. `source_tail` reuses the final recorded sigma values without generating or interpolating a grid. In this mode, `steps_override=0` selects `floor(recorded_intervals × denoise)` intervals (at least one); a positive override selects that many existing intervals, limited to half the recorded trajectory. **H3 Refine Scheduler** records the actual sigma tensor and its dtype in the result packet. Exact mode requires these values; older packets without them must use `regenerated_tail` first. When used with Native Tile, the recorded schedule is shared by the tile sampler calls.

**Audio delivery:** `source_pcm` is the example default: it uses the packet's primary audio, or embedded video audio if primary audio is absent, without another audio VAE round trip in the delivered signal. Missing PCM or a duration mismatch stops execution; it never silently substitutes decoded audio. Select `decoded_latent` explicitly only when original PCM is unavailable or a new Audio VAE decode is intended. The same selected audio is sent to video assembly and the saved packet. The current graph still computes the connected audio decode even when PCM delivery is selected.

**Repeat upscale:** an F07 result can be loaded for another pass while retaining the source sampling profile, task family and previous effective step count. The latent remains marked `derived`; only a matching recorded F07 output is accepted. Arbitrary derived latents are still rejected. Load the updated example JSON after updating the nodes to obtain the new settings, scheduler and delivery wiring. These paths have offline contract/workflow coverage; end-to-end GPU validation is still pending.


### VDN-H3 / Video DeltaNet optimization

**H3 Optimizations** exposes two experimental VDN-H3 choices: **VDN-H3 · FL2VA** and **VDN-H3 · Ref2VA**. They use `Saganaki22/ComfyUI-VDN-H3` as an optional runtime backend while MMH3 keeps responsibility for task-family selection, validation, graph wiring, and optimization provenance. Install that custom node separately and place a compatible VDN stage directory under `ComfyUI/models/vdn/`.

Use the matching **H3 Sampling** preset before enabling VDN: **VDN-H3 DMD - 8 steps (experimental)** or **VDN-H3 Stage-B - 50 steps (experimental)**. The default DMD path targets `stage-dmd-step-250` with the VDN-owned turbo adapter enabled. Ordinary H3 Turbo/FastH3 acceleration adapters are not stacked with VDN, and VDN must not be combined with Sol-Attn or H3 SLA. **Upscale + Stitch** currently rejects VDN because its low-sigma refine tail cannot guarantee the trained VDN trajectory.

F05 Upscale + Stitch and F07 offer **Attention** and **FP16 accumulation** controls, applied after the source LoRAs. **Default** keeps ComfyUI's settings; other attention modes may require external nodes. For the former F07 SLA recipe choose **Attention = H3 SLA** and **FP16 accumulation = Disabled**; SLA defaults are `0.90 / 64 / 4096 / 1`, with audio protection enabled and `comfy_kitchen` as the dense backend. The external `H3SLAAttention` backend is required only when SLA is selected. Refinement follows the source packet's sampler and a short tail of its sampling trajectory. Keep **force_unload** enabled to save VRAM; disabling it can avoid repeated upscaler loading when enough memory is available.

**Video Stitch** is intended for independent clips. **Latent Stitch** requires a continuous chain of saved H3 continuations in their original order: arbitrary videos, missing segments, or independently generated segments will not work. For latent assembly, use matching video/audio overlap values and disable audio feather. Keep the original segment order and use the compatibility reports before saving the final assembly.

### Additional and experimental workflows

| Workflow | Purpose |
| --- | --- |
| [F13 · Reference Management](example_workflows/mmh3_f13_reference_management.json) | Configure reference inclusion, order and purpose; add stable aliases, select a scene, and preview native prompt tags. |
| [F15 · Preflight](example_workflows/mmh3_f15_preflight.json) | Check input compatibility before generation or processing. |
| F16 · [Controlled Generation](example_workflows/mmh3_f16_f01_controlled_generation.json), [Pose + Ref2VA](example_workflows/mmh3_f16_f01_ref2va_pose_generation.json) | One FL2VA graph for Canny, Depth, HED, MLSD, or Pose; Ref2VA Pose remains separate because it has a different reference/conditioning contract. |
| F16 · [Continuation](example_workflows/mmh3_f16_f04_controlled_continuation.json), [Full Frame Refine](example_workflows/mmh3_f16_f07_controlled_full_frame_refine.json), [Tile Refine](example_workflows/mmh3_f16_f07_controlled_native_tile_refine.json) | Apply controls to continuation or frame refinement. |
| F16 · [Masked Refine](example_workflows/mmh3_f16_f07_masked_native_tile_refine.json), [Inpaint · optional Pose](example_workflows/mmh3_f16_f10_masked_inpaint.json) | Modify an area selected by a mask; the same inpaint workflow can optionally add lazy Pose control. |
| [F18 · Batch Stitch](example_workflows/mmh3_f18_batch_stitch.json) | Prepare and assemble a batch of clips. |
| [F18 · Long Video Lipsync](example_workflows/mmh3_f18_long_video_lipsync.json) and [Unified Assembly](example_workflows/mmh3_f18_long_video_assembly.json) | Process a long video in chunks with audio synchronization, then assemble the completed ledger. |
| F18 · Audio Driven: [automation templates](automation/workflows/README.md) | Long-form video driven by audio. Use the documented setup → execution → assembly automation path; these remain separate stateful stages. |

F16 Controlled Generation selects Canny/Depth/HED/MLSD/Pose in `MMH3ControlConfigure`. Its control resource metadata must declare the same structural type, and the selected H3 Fun provider/checkpoint family must match; mismatches fail instead of falling back. F16 requires a compatible H3 Fun runtime and models; runtime compatibility of these examples still needs separate execution and is not claimed here. F18 workflows are multi-stage pipelines involving plan preparation, chunk execution, and assembly rather than a single JSON run. See the linked canvas examples and the [F18 automation instructions](automation/workflows/README.md) for their respective entry points.

## Independent video and audio editing

Open [Independent AV Edit](example_workflows/mmh3_independent_av_edit.json) and choose a source `.mmh3` containing a joint H3 AV latent. In **MMH3 AV Edit Policy**, set the two streams independently:

| Intended change | `video_policy` | `audio_policy` |
| --- | --- | --- |
| Whole video, keep audio | `all` | `preserve` |
| Masked video region, keep audio | `mask` | `preserve` |
| Audio only | `preserve` | `all` or `intervals` |
| Both streams | `all` or `mask` | `all` or `intervals` |

The example starts with whole-video editing and preserved audio. To edit a region, choose `mask` and connect a MASK: white permits changes, black protects the source. A mask must match the source resolution and contain either one static frame or the full target timeline. H3's temporal/spatial grid determines the effective edit area; it may extend beyond the selected pixels.

Audio intervals are half-open frame ranges on the current target at **24 FPS**. For example, `[[24,48]]` selects frames 24 through 47. Audio follows video-mask activity only when you explicitly select `follow_video`. Existing protection takes precedence over new masks.

Keep **AV Protection Restore** between sampling and decoding. **AV Edit Delivery Audio** uses original PCM when audio is preserved; if the packet has no original PCM, explicitly select `decoded_latent` delivery. Preserving an audio latent does not guarantee identical audio after VAE decoding, and video export may re-encode the soundtrack.

## Two-Clip AV Bridge

Open [Two-Clip AV Bridge](example_workflows/mmh3_two_clip_av_bridge.json) and select two `.mmh3` sources with decoded primary video. The node prepares a target from **A's tail → new section → B's beginning** and protects both context windows during sampling.

- Both sources must use **24 FPS**, the same resolution divisible by 32, and synchronized mono/stereo **32 kHz** audio. Normalize incompatible sources beforehand.
- Context windows use exact H3 AV boundaries: **39, 90, 141… frames**. `gap_frames` requests the new section's length; the complete target rounds up to H3's `17k+5` grid. For example, 39 + 45 + 39 becomes 39 + **46** + 39 = 124 frames.
- Missing audio stops preparation by default. Choose `silence` to protect silence or `generate` to generate audio where a source has no soundtrack. An existing but too-short soundtrack is rejected.
- **Save MMH3** stores the full generated target with contexts. **Bridge Middle for Assembly → Save Video** exports only the new section. Assemble **full A → exported section → full B**, without repeating the contexts.

Bridge requires video/audio VAE encoding and decoding. Protected latent values do not guarantee a pixel-identical or visually seamless boundary.

**Validation status:** independent AV editing and Bridge have CPU contract checks; end-to-end execution of these new examples with real H3 models has not yet been verified. Start with a short test before using them in a longer project. Joint AV mask support is required in the ComfyUI sampler.

## Named references by scene

Open [F13 · Reference Management](example_workflows/mmh3_f13_reference_management.json). It demonstrates:

**Reference Configure → Reference Alias → Scene References / Prompt Preview → Reference Report**

1. Connect the reference's `resource_id` from **Put** or **Reference Configure** to **Reference Alias**.
2. Set an alias such as `hero_face` and scene IDs such as `["scene_1","scene_2"]`. Chain additional Alias nodes for other resources.
3. Select a scene in **Scene References** and write a prompt such as `Keep the identity of @hero_face in a cinematic tracking shot.`
4. Inspect the compiled prompt, then connect that node's packet to **AutoCondition** in your generation workflow. F13 itself is a reference-setup/preview example.

For a complete schedule, **Reference Schedule** accepts JSON:

```json
{
  "scenes": ["intro", "dialogue", "outro"],
  "aliases": {
    "hero_face": {"resource_id": "res_REPLACE_WITH_ACTUAL_ID", "range": ["intro", "dialogue"]},
    "voice": {"resource_id": "res_REPLACE_WITH_ACTUAL_AUDIO_ID", "scenes": ["dialogue"]}
  }
}
```

Ranges include both endpoints and resolve to stable scene IDs. Aliases bind to resource IDs and content revisions, rather than their current position. The native resolver assigns Picture/Video/Audio tags and handles bound video soundtracks. Replacing a reference's content requires reconfiguring its alias.

Unknown or inactive aliases, stale revisions and invalid bindings fail explicitly. Run the preview again after changing the prompt or reference settings. Existing prompts with native tags such as `<Picture 1>` remain supported, but numeric tags depend on the active reference order.

## Continue a scene and choose the best take

1. Open **F04 · Segment Workflow** and load the original or most recently accepted `.mmh3` file into **Source**.
2. Select **Continue**, write a prompt, and leave **Review** in **Draft** mode. Multiple prompts can be separated with a line containing `---`.
3. Run the draft. To try another take, change the seed and run again from the same Source.
4. When you are happy with the result, select **Accept** and run the graph with the same seed and settings.
5. In **Save**, click **Continue from this result**. The saved package becomes the Source for the next segment. The button does not start generation automatically.
6. For the final video, assemble compatible continuations with **F05 · Latent Stitch**.

Continuation duration includes repeated context from the previous segment, so the amount of newly generated footage will be shorter. Use the **generated / context / new seconds** report as your guide. **New scene** starts a new scene; use **Video Stitch** to edit independent scenes together.

## Saving and FAQ

For **Reroll accepted**, set Source to the target segment's parent and choose the latest accepted chain in the second Load. For **Reanchor**, choose a new first frame in the image loader and use the FL2VA model. Selecting an action enables its bundled helper loader and mutes unused ones. Reanchor starts a fresh shot without the previous audio/video prefix. Rerolling a reanchored shot requires its anchor image again; enable its loader with Ctrl+M if needed. Use Video Stitch between independent shots.

- **How do I open a result later?** Use **MMH3 Load**, select the `.mmh3` file, and press **Refresh (R)** if needed. Keep `path override` empty when selecting from the file list; a non-empty override takes precedence.
- **Why is my model missing from the list?** Check the model folder and its compatibility with your ComfyUI setup. Select the file you actually installed rather than the filename used by the example.
- **Why do I see unknown nodes?** Make sure the package loaded after restarting ComfyUI. Specialized graphs also require external nodes; start with F01 to verify the basic installation.
- **Running out of memory?** Reduce the resolution and duration, and work with short segments. Upscaling and the final decode of a long chain also require memory; there is no single minimum VRAM requirement that applies to every workflow.
- **Can I upscale only the finished video?** In graphs with **MMH3 Video Upscale**, select **RTX VSR**. This requires the external `RTXVideoSuperResolution` node from Comfy-Org NVIDIA RTX nodes. **Original** keeps the source video unchanged.
- **Can older `.mmh3` files be opened?** Version v0.3 can read schema 2 archives. Archives from before v0.3 are not converted automatically.
- **SageAttention (KJ) reports a missing node?** This option requires ComfyUI-KJNodes and its SageAttention dependencies. The adapter recognizes the registered `PathchSageAttentionKJ` spelling and the compatible `PatchSageAttentionKJ` alias. If neither loads, inspect ComfyUI startup errors. Choose **Default** when you do not need that optional backend.
- **Do the inputs look wrong after an update?** Restart ComfyUI, refresh the browser, and reopen the current workflow JSON.

The public repository includes the node code, canvas examples and F18 automation templates. Development tests, local build tools and internal documentation are not needed to install or run the node.

