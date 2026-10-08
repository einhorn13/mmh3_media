# Six primary workflows

Drag the JSON for your next operation onto ComfyUI. Restart ComfyUI and refresh the browser after updating the extension. Existing F01–F18 workflows and node IDs remain supported.

| Workflow | Use |
| --- | --- |
| 01 Generate | One graph for text, first/last frames and image/video/audio references. Choose the task in Create and select both checkpoint filenames in Generate; only the matching task model is loaded. Standard 20-step sampling is the initial preset. |
| 02 Continue | Existing F04 segment preparation/review, including new scenes, reanchor and rerolls. Direct latent continuation needs a compatible saved source. Decoded-only continuation remains the detailed F03 example. |
| 03 Edit | Independent video/audio policies and protected regions. Trained H3-Fun inpainting and two-clip bridges remain detailed examples. |
| 04 Refine | One H3 Refine node for full frame, tiles and masked tile compositing. Select a checkpoint matching the saved source task. Requires the existing external H3 latent upscaler. |
| 05 Assemble | Batch normalization/import and decoded stitching. Compatible continuation chains can use the detailed F05 latent stitch instead. |
| 06 Studio | Identity/reference library plus master audio; generate, review and assemble scenes. Open Project Manager for scene settings. |

## Keyframes

For FastH3 V2 in 01 Generate, select a complete `fasth3_8step_v2` Comfy checkpoint in the FL2VA field and the matching `FastH3 V2 - 8 steps` preset. Use text-to-audio-video with no keyframes or references. Native VSA is applied automatically; keep Load LoRAs in auto mode.

Generate prepares first/last frames **after** Video Settings resolves the actual canvas. Crop preserves proportions and trims edges; contain preserves the full image with borders; stretch deliberately changes proportions. Prepared frames appear as previews. With no keyframes this step passes the packet through. Reference images retain their independent geometry. The standalone H3 Prepare Keyframe node supports other graphs.

## Reference cards

Queue once to discover connected references. Edit aliases, active state, roles and scene IDs in H3 Reference Cards; subsequent runs compile `@hero_face` into the actual native tag and show the compiled prompt. Card slots remain stable when a newly created packet has different resource IDs. Refresh/review cards when replacing or reordering references.

Encoder-only conditioning applies to **all** references in the packet and requires a native runtime with optional reference VAEs; it is distinct from normal latent references. Decoding generated AV still uses the ordinary VAEs. Roles record intent, while the directing prompt specifies the attributes to transfer.

## Refinement

All methods use the existing F07 source sampling, source LoRA provenance, preflight and audio protection. `source_pcm` is the default delivery policy; choose `decoded_latent` explicitly if PCM is unavailable. Masked tiles refine the upscaled video then composite white/editable pixels onto the upscaled baseline, preserving black/protected pixels in delivered RGB. This is pixel compositing, not H3-Fun trained inpainting; the derived latent is re-encoded. A static mask or the complete frame timeline is supported.

## Shared Project Manager

Studio and continuation projects appear in the same project selector. Studio scene cards show delivered duration, legal generation window, reference selections and saved takes. Scene edits use a checked project digest and settings revision. Changing a reviewed prompt/reference selection keeps old takes and prepares another take. Accepted scenes remain locked.

Duration edits apply only to a never-rendered pending suffix. The suffix is replanned against absolute master-audio ownership; accepted scenes and their proofs stay intact. Generation/assembly controls require the Studio node for that project on the current canvas. Global sources/models remain frozen; create a new project to change them. An optional References packet on Studio freezes a library created with Reference Cards.

Use `@master_audio` for Studio's current master slice, especially when the library also contains audio references. Studio compiles aliases after inserting that slice so native numbering remains correct. The alias is reserved; library references need different names.

Developer regeneration: `python workflow_primary.py`; verification: `python workflow_primary.py --check`. This uses the repository's local development helpers `workflow_generate.py`, `workflow_schema.py` and F07 recipes. They are unnecessary for normal ComfyUI execution. Canonical full/tile refinement API templates are compiled from those recipes and shipped in `automation/workflows`.
