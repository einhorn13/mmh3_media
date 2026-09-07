# ComfyUI-MMH3-Media

**Generate videos with audio, continue scenes, and assemble clips with MiniMax H3 in ComfyUI.**

MMH3 Media is a set of custom nodes and ready-to-use workflows. Generate videos from text, first and last frames, or references; try different continuations; combine segments; and upscale the result.

Results are saved as `.mmh3` files. Along with the media, the file stores settings, references, and the internal generation state — latents. This lets you return to a project later and continue the scene. For playback and publishing, use the workflow's video output and **Save Video**; `.mmh3` is a working project file, not a video-player format.

## Installation

1. Set up ComfyUI with MiniMax H3 support and the models used by the examples. This package does not include model weights.
2. Download the repository via **Code → Download ZIP** and extract it to `ComfyUI/custom_nodes/ComfyUI_mmh3_media`. The `__init__.py` file must be located directly inside this folder, without an extra nested repository directory.
3. Place the models in the folders listed below and select them in the workflow loaders. The table lists the files used by the bundled examples; other variants require a compatible ComfyUI setup.
4. Restart ComfyUI, refresh the browser, and open a JSON file from [example_workflows](example_workflows).

The basic nodes require no additional Python packages beyond ComfyUI's own dependencies. Specialized workflows may require external nodes and models.

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

The `example_workflows` directory contains user-facing ComfyUI canvas workflows only. Programmatic test fixtures live under `tests/fixtures/workflows/`; installed F18 automation templates live under `automation/workflows/`.

### Generation and continuation

| Workflow | Purpose |
| --- | --- |
| [F01 · Text and Frames](example_workflows/mmh3_f01_fl2va.json) | Create the first video from a prompt and an optional first and/or last frame. |
| [F01 · References](example_workflows/mmh3_f01_ref2va.json) | Generate using images, video, and audio as references. |
| [F02 · Continuation](example_workflows/mmh3_f02_continuation.json) | Continue a saved H3 segment from an `.mmh3` file; an end frame can also be specified. |
| [F02 · Continuation with References](example_workflows/mmh3_f02_ref2va_continuation.json) | Generate a continuation with Ref2VA and active references. |
| [F03 · Decoded Video Continuation](example_workflows/mmh3_f03_decoded_continuation.json) | Continue a video without a saved H3 state. `missing_audio_policy=error` is strict by default; select `silence` only to permit missing or too-short source audio. Existing source audio is still preserved. |
| [F04 · Segment Workflow](example_workflows/mmh3_f04_chain_append.json) | One workspace for Continue, Reroll accepted, New scene and Reanchor from a new first frame, with Draft → Accept review. |

### Stitching and upscaling

| Workflow | Purpose |
| --- | --- |
| [F05 · Video Stitch](example_workflows/mmh3_f05_stitch.json) | Combine finished clips with audio. Supports Auto Seamless, Cut, and Crossfade, plus color matching at the transition. |
| [F05 · Latent Stitch](example_workflows/mmh3_f05_latent_stitch.json) | Assemble sequential H3 continuations, remove the repeated opening context, and decode the final video with audio only once. |
| [F05 · Upscale + Stitch](example_workflows/mmh3_f05_latent_stitch_upscale.json) | Upscale a linked continuation chain and assemble it into a single video. Requires the external `MinimaxH3LatentUpscaler3D` node and its weights. |
| [F07 · Latent Upscale](example_workflows/mmh3_f07_latent_upscale.json) | Upscale an H3 latent and refine the result. Optional SLA is selected in **H3 Optimizations**; the normal default does not require the SLA backend. |
| [F07 · Native Tile Upscale](example_workflows/mmh3_f07_native_tile_upscale.json) | Refine the image in tiles — separate regions of the frame. |

The F07 examples use the external `MinimaxH3LatentUpscaler3D` node and its weights. Install them before running an upscale workflow.

### VDN-H3 / Video DeltaNet optimization

**H3 Optimizations** exposes two experimental VDN-H3 choices: **VDN-H3 · FL2VA** and **VDN-H3 · Ref2VA**. They use `Saganaki22/ComfyUI-VDN-H3` as an optional runtime backend while MMH3 keeps responsibility for task-family selection, validation, graph wiring, and optimization provenance. Install that custom node separately and place a compatible VDN stage directory under `ComfyUI/models/vdn/`.

Use the matching **H3 Sampling** preset before enabling VDN: **VDN-H3 DMD - 8 steps (experimental)** or **VDN-H3 Stage-B - 50 steps (experimental)**. The default DMD path targets `stage-dmd-step-250` with the VDN-owned turbo adapter enabled. Ordinary H3 Turbo/FastH3 acceleration adapters are not stacked with VDN, and VDN must not be combined with Sol-Attn or H3 SLA. **Upscale + Stitch** currently rejects VDN because its low-sigma refine tail cannot guarantee the trained VDN trajectory.

F05 Upscale + Stitch and F07 offer **Attention** and **FP16 accumulation** controls, applied after the source LoRAs. **Default** keeps ComfyUI's settings; other attention modes may require external nodes. For the former F07 SLA recipe choose **Attention = H3 SLA** and **FP16 accumulation = Disabled**; SLA defaults are `0.90 / 64 / 4096 / 1`, with audio protection enabled and `comfy_kitchen` as the dense backend. The external `H3SLAAttention` backend is required only when SLA is selected. Refinement follows the source packet's sampler and a short tail of its sampling trajectory. Keep **force_unload** enabled to save VRAM; disabling it can avoid repeated upscaler loading when enough memory is available.

**Video Stitch** is intended for independent clips. **Latent Stitch** requires a continuous chain of saved H3 continuations in their original order: arbitrary videos, missing segments, or independently generated segments will not work. For this type of assembly, use matching video/audio overlap values and disable audio feather; see the [workflow guide](docs/WORKFLOWS.md#f02-продолжение) for details.

### Additional and experimental workflows

| Workflow | Purpose |
| --- | --- |
| [F13 · Reference Management](example_workflows/mmh3_f13_reference_management.json) | Enable or disable references and change their order and role. |
| [F15 · Preflight](example_workflows/mmh3_f15_preflight.json) | Check input compatibility before generation or processing. |
| F16 · [Controlled Generation](example_workflows/mmh3_f16_f01_controlled_generation.json), [Pose + Ref2VA](example_workflows/mmh3_f16_f01_ref2va_pose_generation.json) | One FL2VA graph for Canny, Depth, HED, MLSD, or Pose; Ref2VA Pose remains separate because it has a different reference/conditioning contract. |
| F16 · [Continuation](example_workflows/mmh3_f16_f04_controlled_continuation.json), [Full Frame Refine](example_workflows/mmh3_f16_f07_controlled_full_frame_refine.json), [Tile Refine](example_workflows/mmh3_f16_f07_controlled_native_tile_refine.json) | Apply controls to continuation or frame refinement. |
| F16 · [Masked Refine](example_workflows/mmh3_f16_f07_masked_native_tile_refine.json), [Inpaint · optional Pose](example_workflows/mmh3_f16_f10_masked_inpaint.json) | Modify an area selected by a mask; the same inpaint workflow can optionally add lazy Pose control. |
| [F18 · Batch Stitch](example_workflows/mmh3_f18_batch_stitch.json) | Prepare and assemble a batch of clips. |
| [F18 · Long Video Lipsync](example_workflows/mmh3_f18_long_video_lipsync.json) and [Unified Assembly](example_workflows/mmh3_f18_long_video_assembly.json) | Process a long video in chunks with audio synchronization, then assemble the completed ledger. |
| F18 · Audio Driven: [automation templates](automation/workflows/README.md) | Long-form video driven by audio. Use the documented setup → execution → assembly automation path; these remain separate stateful stages. |

F16 Controlled Generation selects Canny/Depth/HED/MLSD/Pose in `MMH3ControlConfigure`. Its control resource metadata must declare the same structural type, and the selected H3 Fun provider/checkpoint family must match; mismatches fail instead of falling back. F16 requires a compatible H3 Fun runtime and models; runtime compatibility of these examples still needs separate execution and is not claimed here. F18 workflows are multi-stage pipelines involving plan preparation, chunk execution, and assembly rather than a single JSON run. Additional settings are described in the [workflow guide](docs/WORKFLOWS.md).

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

- **How do I open a result later?** Use **MMH3 Load**, select the `.mmh3` file, and press **Refresh (R)** if needed. A non-empty `path override` takes precedence over the file list.
- **Why is my model missing from the list?** Check the model folder and its compatibility with your ComfyUI setup. Select the file you actually installed rather than the filename used by the example.
- **Why do I see unknown nodes?** Make sure the package loaded after restarting ComfyUI. Specialized graphs also require external nodes; start with F01 to verify the basic installation.
- **Running out of memory?** Reduce the resolution and duration, and work with short segments. Upscaling and the final decode of a long chain also require memory; there is no single minimum VRAM requirement that applies to every workflow.
- **Can I upscale only the finished video?** In graphs with **MMH3 Video Upscale**, select **RTX VSR**. This requires the external `RTXVideoSuperResolution` node from Comfy-Org NVIDIA RTX nodes. **Original** keeps the source video unchanged.
- **Can older `.mmh3` files be opened?** Version v0.3 can read schema 2 archives. Archives from before v0.3 are not converted automatically.
- **Do the inputs look wrong after an update?** Restart ComfyUI, refresh the browser, and reopen the current workflow JSON.

For detailed continuation, transition, reference, and upscaling settings, see the [workflow guide](docs/WORKFLOWS.md).

