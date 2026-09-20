# MMH3 Media for ComfyUI

**Create video with audio. Continue the scene. Choose the best take.**

Custom nodes and ready-to-use workflows for **MiniMax H3** in ComfyUI — from your first prompt to a finished sequence.

[Get started](#installation) · [First video](#your-first-video) · [Workflows](#choosing-a-workflow) · [Advanced controls](#advanced-controls) · [FAQ](#saving-and-faq)

## What you can do

- **Generate** from text, first/last frames, or image, video and audio references.
- **Build a sequence** with continuations, alternate takes and Draft → Accept review.
- **Refine the result** with upscaling, masks, independent audio/video editing and clip bridges.
- **Return to your work** with `.mmh3` project files that keep media, settings, references and saved generation state (latents).

**A typical workflow:** Generate → Save `.mmh3` → Continue & review → Stitch → Export video

> Save a video for playback and sharing. Keep the `.mmh3` file to continue or edit the generation later.

## Installation

You need a **ComfyUI installation with MiniMax H3 support** and compatible model weights. This package uses ComfyUI's existing Python environment; it does not include models or replace your PyTorch/CUDA setup.

From your ComfyUI directory:

```console
cd custom_nodes
git clone https://github.com/einhorn13/mmh3_media.git ComfyUI_mmh3_media
```

Alternatively, use **Code → Download ZIP** and extract into `ComfyUI/custom_nodes/ComfyUI_mmh3_media`. Place `__init__.py` directly inside that folder. Keep only one installed copy.

Add the models below, restart ComfyUI, and refresh your browser.

<details>
<summary><strong>Models and folders</strong></summary>

These are the filenames used by the examples. Choose **FL2VA** for text/frames or **Ref2VA** for references; you only need the diffusion model and matching Turbo LoRA for your chosen path.

| Component | Example filename | Folder in ComfyUI |
| --- | --- | --- |
| Text / frames | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| References | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |
| FL2VA Turbo | `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors` | `models/loras/` |
| Ref2VA Turbo | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | `models/loras/` |

Select the files you actually installed in each loader. Other variants need a compatible host setup. Turbo LoRAs are optional if you use **Standard (20 steps)**.

</details>

### Updating

Run `git pull --ff-only` inside the installed node folder, then restart ComfyUI, refresh the browser and reopen the updated example JSON. Keep customized workflows separately. For ZIP installs, replace the old package with the new release while preserving your personal files.

## Your first video

1. Drag [**F01 · Text and Frames**](example_workflows/mmh3_f01_fl2va.json) onto the ComfyUI canvas.
2. Select your models in the loaders.
3. In **MMH3 Create**, choose **Video (optional frames)** and describe the scene, motion and sound.
4. Leave frame inputs empty for text-to-video, or connect a first frame, a last frame, or both.
5. Set resolution and duration in **H3 Video Settings**. Start with a short clip at a modest resolution.
6. Run the workflow. Save the video to share it and the `.mmh3` to keep working.

**H3 Sampling** defaults to **4-step Turbo** and loads its matching LoRA automatically. Do not apply it twice. If you do not have that LoRA, choose **Standard (20 steps)**.

Want to use references? Open [**F01 · References**](example_workflows/mmh3_f01_ref2va.json), select the Ref2VA model and connect images, video or audio. Reference inputs expand as you connect them.

## Choosing a workflow

Open these JSON files on the ComfyUI canvas. Start with F01, then choose the next step for your project.

| I want to… | Open |
| --- | --- |
| Generate a video | F01 · [Text and Frames](example_workflows/mmh3_f01_fl2va.json) / [References](example_workflows/mmh3_f01_ref2va.json) |
| Continue a saved generation | F02 · [Continuation](example_workflows/mmh3_f02_continuation.json) / [With References](example_workflows/mmh3_f02_ref2va_continuation.json) |
| Continue a video without saved latents | F03 · [Decoded Video Continuation](example_workflows/mmh3_f03_decoded_continuation.json) |
| Build a scene and compare takes | F04 · [Segment Workflow](example_workflows/mmh3_f04_chain_append.json) |
| Join independent clips | F05 · [Video Stitch](example_workflows/mmh3_f05_stitch.json) |
| Assemble a continuation chain | F05 · [Latent Stitch](example_workflows/mmh3_f05_latent_stitch.json) / [Upscale + Stitch](example_workflows/mmh3_f05_latent_stitch_upscale.json) |
| Upscale a saved result | F07 · [Latent Upscale](example_workflows/mmh3_f07_latent_upscale.json) / [Native Tile Upscale](example_workflows/mmh3_f07_native_tile_upscale.json) |
| Edit video and audio separately | [Independent AV Edit](example_workflows/mmh3_independent_av_edit.json) |
| Generate a transition between two clips | [Two-Clip AV Bridge](example_workflows/mmh3_two_clip_av_bridge.json) |

**Choose the right stitch:** Video Stitch joins independent clips with Cut, Crossfade or Auto Seamless. Latent Stitch needs a compatible, unbroken H3 continuation chain in its original order; it removes repeated context and decodes once. Use matching video/audio overlap and disable audio feather for latent assembly.

F03 re-encodes decoded video, so the handover is not lossless. Missing or short audio stops it by default; select `missing_audio_policy=silence` only when you want to allow that fallback.

<details>
<summary><strong>More workflows: references, controls and long-video automation</strong></summary>

| Use case | Workflow |
| --- | --- |
| Organize references and aliases | F13 · [Reference Management](example_workflows/mmh3_f13_reference_management.json) |
| Check input compatibility | F15 · [Preflight](example_workflows/mmh3_f15_preflight.json) |
| Generate with structural controls | F16 · [Controlled Generation](example_workflows/mmh3_f16_f01_controlled_generation.json) / [Pose + Ref2VA](example_workflows/mmh3_f16_f01_ref2va_pose_generation.json) |
| Continue or refine with controls | F16 · [Continuation](example_workflows/mmh3_f16_f04_controlled_continuation.json) / [Full Frame](example_workflows/mmh3_f16_f07_controlled_full_frame_refine.json) / [Tile Refine](example_workflows/mmh3_f16_f07_controlled_native_tile_refine.json) |
| Edit a masked area | F16 · [Masked Refine](example_workflows/mmh3_f16_f07_masked_native_tile_refine.json) / [Inpaint with optional Pose](example_workflows/mmh3_f16_f10_masked_inpaint.json) |
| Assemble batches or long videos | F18 · [Batch Stitch](example_workflows/mmh3_f18_batch_stitch.json) / [Long Video Lipsync](example_workflows/mmh3_f18_long_video_lipsync.json) / [Unified Assembly](example_workflows/mmh3_f18_long_video_assembly.json) |
| Try progressive generation/upscaling | F07 · [Progressive Handoff](example_workflows/mmh3_f07_progressive_handoff.json) — experimental |

F16 needs a compatible H3 Fun runtime and matching control models; these examples still need runtime validation. F18 uses separate **setup → execution → assembly** stages; follow the [automation guide](automation/workflows/README.md) for API templates, audio-driven generation and interactive music-video workflows.

Progressive Handoff requires [Flow-Aligned Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate) and [Upscaler-Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus).

</details>

## Continue a scene and choose the best take

1. Open **F04 · Segment Workflow** and load your latest accepted `.mmh3` as **Source**.
2. Select **Continue**, write the next prompt and run in **Draft** mode.
3. Change the seed to try another take from the same Source.
4. Choose **Accept** and run with the same seed and settings.
5. In **Save**, click **Continue from this result** to prepare the next segment. Queue it when ready.

The duration includes repeated context from the previous segment. Check the **generated / context / new seconds** report for the actual added footage.

**Project Manager** brings saved segments and candidates together. Open it from Load, Save or Segment Review to compare takes, accept candidates, prepare replacements and preview the final assembly. **Assemble and save MP4** exports the accepted revisions; each segment needs decoded video/audio and recorded continuation timing.

Use **New scene** or **Reanchor** for a fresh shot, then Video Stitch to combine independent shots. Reanchor needs a first-frame image and the FL2VA model. Rerolling that shot needs the anchor image again.

## Advanced controls

Optional workflows and backends need their own nodes and models. Start with the default settings, then enable the features you need.

<details>
<summary><strong>Video decoding and streaming</strong></summary>

| Video decoder | Setup |
| --- | --- |
| `vae` | Default: use the connected Video VAE. |
| `draft` | Put [taeh3.safetensors](https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taeh3.safetensors) in `models/vae_approx/`; requires native taeh3 support. Exported and archived video both use draft quality. |
| `trt` | Install [ComfyUI-H3VAE_TRT](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT), compile a compatible decoder engine and select it in **trt_decoder**. `auto` works only when exactly one decoder engine is available. |

The older `tae_minimax_h3.safetensors` is not a substitute for temporal taeh3. Missing or incompatible TRT engines fail explicitly.

For long F05 latent assemblies, **H3 Decode to Video** can stream `draft` or `trt` frames directly to the encoder. `auto` enables streaming above 360 frames (15 seconds); `stream` requires a supported decoder; `off` uses full decode. This limits decoded RGB memory, but models, latents and decoded audio still use memory. Nodes needing a full IMAGE batch require full decode.

TRT streaming supports the FP16 static decoder profile `[1,24,7,16,16] → [1,3,28,256,256]`, a canvas of at least 256×256 and one video per delivery. Other engines are not assumed compatible.

When customizing graphs, preserve **H3 Video Decode → MMH3 Create Video** connections for both IMAGE and `decode_info_json`. Keep **MMH3 Save Video → MMH3 Save** connected through the packet so the archive reuses the encoded video.

</details>

<details>
<summary><strong>Upscaling and preserved audio</strong></summary>

F07 and Upscale + Stitch use the external `MinimaxH3LatentUpscaler3D` node and its weights. Both legacy and Upscaler-Plus schemas are supported. Generate and save with F01, load the result in F07, then refine and save. A matching saved F07 result can be used for another pass, up to 4× per spatial axis per pass.

Repeat-upscale behavior has offline contract coverage; end-to-end GPU validation is still pending.

- **H3 Upscale Settings** selects checkpoints, device, precision and audio delivery. Keep Preflight enabled; it checks compatibility, not available VRAM.
- `steps_override=0` inherits the source count. A positive value sets the actual refinement steps.
- `denoise_override=0` uses source-aware defaults. Explicit values support 0–1; 0.05–0.50 is the recommended refinement range. Higher values can replace source structure.
- `regenerated_tail` is the default schedule. `source_tail` needs the actual sigma trajectory recorded in the source packet.
- `source_pcm` preserves original audio without another audio VAE round trip in the delivered signal. Choose `decoded_latent` explicitly if source PCM is unavailable. Export codecs may still re-encode audio.
- Keep upscaler offloading enabled to release VRAM before refinement. Full-sequence inference is the default; legacy temporal chunking is a fallback that can change the result.

For decoded-only upscaling, **MMH3 Video Upscale → RTX VSR** requires the external `RTXVideoSuperResolution` node. **Original** leaves the source video unchanged.

</details>

<details>
<summary><strong>Sampling, LoRAs and attention</strong></summary>

**Custom** sampling exposes the host's registered samplers and schedulers, plus built-in `beta57`. Availability does not guarantee good results for every H3 configuration.

**Load LoRAs** offers an expandable file/strength list. Connect it to Sampling, Refine Source LoRAs or Upscale + Stitch:

| Mode | Behavior |
| --- | --- |
| `auto` | Keep preset or recorded source adapters. |
| `custom` | Replace the acceleration stack. |
| `extension` | Keep the stack and append selected LoRAs. |
| `disabled` | Remove the acceleration stack. |

Creative source LoRAs retain their order during refinement. FastH3 uses its own adapter loader and requires `auto`.

**H3 Optimizations** separates Attention, **SageAttention (KJ)** and **FP16 accumulation**. Default attention uses the host settings. Sage needs ComfyUI-KJNodes and its dependencies; it supplies dense attention without replacing sparse kernels. H3 SLA requires its external backend; Sage composition uses SLA's `auto` dense backend.

**VDN-H3 is experimental.** Install `Saganaki22/ComfyUI-VDN-H3`, place its compatible stages in `models/vdn/`, and match FL2VA/Ref2VA with the DMD 8-step or Stage-B 50-step preset. Do not stack ordinary Turbo/FastH3 adapters, Sol or SLA with VDN. Upscale + Stitch does not support VDN. Custom LoRAs replace VDN's built-in Turbo adapter; extension mode retains it.

**TaoMate presets are experimental:** FL2VA 3/6-step and Ref2VA 8-step. They need `taomate_3step_lora_avg_rank_19_bf16.safetensors`; the 6-step preset also uses `Motion_BoosterV2.safetensors`. Keep **H3 Scheduler**, the preset step count and `denoise=1` for fixed schedules. Automatic upscale/refine schedule derivation rejects explicit-sigma source trajectories. No quality or speed benchmark is claimed.

</details>

<details>
<summary><strong>Independent video and audio editing</strong></summary>

### Independent video and audio editing

Load a `.mmh3` with a joint H3 AV latent in **Independent AV Edit**. Set the streams separately in **MMH3 AV Edit Policy**:

| Change | Video policy | Audio policy |
| --- | --- | --- |
| Whole video, keep audio | `all` | `preserve` |
| Masked region, keep audio | `mask` | `preserve` |
| Audio only | `preserve` | `all` or `intervals` |

Masks use white for editable areas and black for protected areas. Match the source resolution and provide one static frame or the full timeline. H3's grid may enlarge the effective edit area.

Audio intervals use half-open frame ranges at 24 FPS: `[[24,48]]` selects frames 24–47. Audio follows mask activity only with `follow_video`.

Keep **AV Protection Restore** before decoding. Use original PCM for preserved audio, or explicitly choose `decoded_latent` if PCM is unavailable. Preserving latents does not guarantee identical decoded pixels or audio.

This workflow has CPU contract coverage; end-to-end real-model execution is not yet verified. It requires joint AV mask support in the sampler.

</details>

<details>
<summary><strong>Two-Clip AV Bridge</strong></summary>

### Two-Clip AV Bridge

Load two `.mmh3` files with decoded primary video. The workflow builds **A's tail → new section → B's beginning**, protecting both context windows.

- Use 24 FPS, matching resolution divisible by 32, and synchronized mono/stereo 32-kHz audio.
- Context windows use H3 boundaries: 39, 90, 141… frames. The requested gap rounds to the H3 grid; check its effective length.
- Missing audio stops preparation unless you explicitly select `silence` or `generate`. Too-short existing audio is rejected.
- Save MMH3 keeps the contexts. **Bridge Middle for Assembly → Save Video** exports only the new section. Assemble **full A → new section → full B**.

The VAE round trip is not lossless and does not guarantee a seamless boundary. CPU contracts are covered; end-to-end real-model execution is not yet verified.

</details>

<details>
<summary><strong>Named references by scene</strong></summary>

### Named references by scene

In **F13 · Reference Management**, connect a resource ID to **Reference Alias**, name it (for example `hero_face`) and assign scene IDs. Select a scene in **Scene References** and write a prompt such as:

> Keep the identity of @hero_face in a cinematic tracking shot.

Preview the compiled prompt, then connect the resulting packet to **AutoCondition** in your generation workflow. F13 sets up references; it does not generate a video.

Aliases bind to resource IDs and content revisions. Reconfigure an alias when replacing its content. Unknown, inactive or stale aliases fail explicitly; native tags such as `<Picture 1>` remain supported but depend on active reference order.

</details>

## Saving and FAQ

| Question | Answer |
| --- | --- |
| How do I reopen a result? | Use **MMH3 Load** and **Refresh (R)**. Leave `path override` empty to use the file list. |
| Why are nodes missing? | Restart ComfyUI and check startup errors. Specialized workflows need additional nodes; start with F01 to check the base installation. |
| Why is a model missing? | Check its folder and select the filename you installed in the loader. |
| Running out of memory? | Lower resolution/duration and work in short segments. Upscaling and long-chain decoding also need memory; requirements depend on the workflow. |
| Inputs look wrong after an update? | Restart ComfyUI, refresh the browser and reopen the current example JSON. |
| Can I open old archives? | The package reads schema-2 `.mmh3` archives. Pre-v0.3 archives are not converted automatically. |
| How do I remove a resource? | Use **MMH3 Remove**, then save the returned packet. The original archive stays unchanged. |

See all [canvas examples](example_workflows), the [F18 automation guide](automation/workflows/README.md), or [AGENTS.md](AGENTS.md) for code ownership and development checks.
