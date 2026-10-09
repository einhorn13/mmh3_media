# MMH3 Media for ComfyUI

**Create video with audio. Continue a scene. Choose the best take.**

Custom nodes and ready-to-use workflows for MiniMax H3. Save `.mmh3` files to keep media, settings and generation state for later editing.

[Install](#installation) · [Start](#first-video) · [Workflows](#workflows) · [Models](#models-and-optional-features) · [Help](#quick-help)

## Installation

Use a ComfyUI installation with MiniMax H3 support. From its directory:

```console
cd custom_nodes
git clone https://github.com/einhorn13/mmh3_media.git ComfyUI_mmh3_media
```

Or download the ZIP and extract it into `ComfyUI/custom_nodes/ComfyUI_mmh3_media`, with `__init__.py` directly inside that folder. Keep one installed copy.

Add the required models, restart ComfyUI and refresh your browser. No extra dependencies are needed for the base package; keep ComfyUI's existing Python/PyTorch environment.

## First video

1. Drag [01 Generate](example_workflows/primary/01_generate.json) onto the canvas.
2. Select your diffusion model, text encoder and video/audio VAEs in **H3 Generate**.
3. Enter a prompt in **MMH3 Create**. Leave frame inputs empty for text-to-video, or connect first/last frames.
4. Set resolution and duration in **H3 Video Settings**. Start with a short clip.
5. Run. Save the video to share it and the `.mmh3` to continue working.

**Standard, 20 steps** is the starting preset. Turbo needs the matching acceleration LoRA. A selected task loads only its matching checkpoint.

## Workflows

| Workflow | Use it to |
| --- | --- |
| [01 Generate](example_workflows/primary/01_generate.json) | Generate from text, frames or references. |
| [02 Continue](example_workflows/primary/02_continue.json) | Extend a scene, reroll and review takes. |
| [03 Edit](example_workflows/primary/03_edit.json) | Edit video and audio independently. |
| [04 Refine](example_workflows/primary/04_refine.json) | Upscale the full frame or masked regions. |
| [05 Assemble](example_workflows/primary/05_assemble.json) | Join clips and export a sequence. |
| [06 Studio](example_workflows/primary/06_studio.json) | Build a long performance video scene by scene. |

In Studio: **Generate next → review → Accept take**, then Assemble when every scene is accepted. Project Manager keeps references, scene settings and saved takes together. Use continuation workflows for an uninterrupted shot.

[Advanced examples](example_workflows) cover control, inpainting, latent stitching and clip bridges. [API templates](automation/workflows) support batch automation; `python automation_runner.py --help` lists runner options. Specialized paths may require additional nodes and models.

## Models and optional features

Choose the filenames actually installed on your computer.

| Component | ComfyUI folder |
| --- | --- |
| H3 FL2VA for text/frames, or Ref2VA for references | `models/diffusion_models/` |
| H3 text encoder | `models/text_encoders/` |
| H3 video VAE and audio VAE | `models/vae/` |
| Matching Turbo and creative LoRAs | `models/loras/` |

- **Load LoRAs:** search by part of a filename and build an ordered list. Custom replaces preset adapters; Extension adds to them. Auto uses the preset or recorded source adapters.
- **FastH3 V2:** use a complete [Comfy checkpoint](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy) and the **FastH3 V2 - 8 steps** preset for text-to-audio-video without references. Native VSA is applied automatically; keep Load LoRAs in Auto. GPU quality and speed are not yet verified here.
- **Refine:** requires the maintained [H3 Latent Upscaler Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus) and its weights. Older upscaler APIs are unsupported.
- **Optimizations:** choose one attention strategy. FP16 accumulation and SageAttention can be enabled independently when their backends are available. Native Sol replaces the removed SolAttn_triton integration.

## Quick help

| Problem | Try this |
| --- | --- |
| Missing nodes or models | Check startup errors, model folders and selected filenames. |
| Out of memory | Reduce resolution or duration; generate long videos in segments. |
| Reopen a result | Use **MMH3 Load** and Refresh (R). Keep `path override` empty for the file list. |
| Preserve audio during edits | Keep the source PCM delivery policy; changing video need not regenerate audio. |
| Inputs changed after an update | Restart ComfyUI, refresh the browser and open the latest example JSON. |
| Old archive will not load | Schema-2 `.mmh3` archives are supported; pre-v0.3 files are not converted. |

To update a Git installation, run `git pull --ff-only` in this node folder. Keep personal workflows separately. For ZIP installs, replace the package with the new version.

For a bug report, include the ComfyUI version, workflow JSON and first console error.

Licensed under [LICENSE](LICENSE).
