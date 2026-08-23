# ComfyUI-MMH3-Media

**Release:** 0.2.0  
**Packet schema:** 2

`MMH3_MEDIA` is an H3-aware media/checkpoint packet for ComfyUI. One `.mmh3` can carry the original MiniMax H3 joint AV latent together with final video/audio, keyframes, refs, masks, metadata, creator notes and a disposable preview cache.

## 0.2 contract

0.2 intentionally drops schema-v1 compatibility. A v1 archive is rejected instead of being guessed/migrated. Latents require explicit `latent_layout`; unknown resource kinds are rejected by the current schema. This keeps the format small and deterministic while it is still early.

`packet.json` now always contains a top-level `notes` string. It is free-form human/creator text and defaults to `""`. Machine-specific data belongs in `extensions` or resource metadata.

## Nodes

- **MMH3 Create** — starts a packet; can take H3 latent/video/audio/keyframes and initial notes.
- **MMH3 Load** — lazy manifest/ZIP-directory load; the prominent **REFRESH .MMH3 FILES** control is placed at the bottom of the node.
- **MMH3 Save** — atomic ZIP64 save; unchanged resources are byte-copied when possible.
- **MMH3 Export** — unpacks a self-consistent `packet.json` + ordinary resource tree to an output directory.
- **MMH3 Put / Remove / Move** — resource management with stable resource IDs and dense ordered slots.
- **MMH3 Metadata** — name/generation/tags/notes/custom metadata. `notes_action=keep|set|clear` prevents accidental note deletion.
- **MMH3 Inspect** — manifest-only summary; returns notes plus direct `prompt`, `task`, `seed`, and `name` outputs for downstream workflows.
- **MMH3 Compare** — semantic packet diff. Preview cache is ignored by default.
- **MMH3 Preview** — reuse a fresh preview cache or regenerate a contact sheet from `h3_av_latent` through a supplied H3-compatible Fast/Tiny VAE.
- **MMH3 H3 AV Separate / Combine** — strict zero-copy stream split/reassembly for MiniMax H3 joint AV latents; masks are preserved and duration/batch/layout mismatches are rejected instead of silently repaired.
- **MMH3 H3 Provenance** — declare latent origin without touching payload bytes.
- **MMH3 H3 Compatibility** — manifest-only preflight for future H3-aware seam/stitch adapters; never approves naive latent concatenation.
- Typed lazy Get nodes for Latent/Image/Video/Audio/Mask/JSON.

## Preview is cache

`preview` is a singleton IMAGE resource stored as `preview/preview.png` (contact sheet). It is disposable and can be rebuilt.

- The preview resource has **no `sha256` field** and is skipped by `verify=full` checksum verification.
- Its metadata records the source H3 latent `resource_id` and, after Save, the source latent's final SHA-256.
- `MMH3 Preview / if_missing_or_stale` reuses only a matching cache; otherwise it requires `fast_vae` and regenerates.
- `refresh` always regenerates; `cache_only` never invokes the decoder.
- `MMH3 Compare` ignores preview by default so a regenerated thumbnail does not make two assets semantically different.

A compatible Fast/Tiny VAE object must expose `decode()`. 0.2 does not bundle model weights or auto-download a decoder.

## Export

`MMH3 Export` writes a normal directory tree under ComfyUI output. `include_preview_cache=false` removes both the preview file and its descriptor from the exported `packet.json`, so the result remains internally consistent.

## Notes

Creator text is deliberately separate from generation metadata:

```json
{
  "notes": "Approved take. Keep the original voice; use latent for 2x upscale."
}
```

Use `MMH3 Metadata` to set/clear it. `MMH3 Inspect` and `MMH3 Metadata` expose the current value as an output. `MMH3 Inspect` also exposes stored generation `prompt`, `task`, `seed`, and packet `name` directly, so downstream conditioning does not need to parse `info_json`.

## Validation architecture

There is no `MMH3 Validate` node. Validation is an internal service layer (`validation.py`) used by core/archive operations. This keeps validation reusable for Save/Load/Export/future batch tooling without adding another user-facing node. Current validation covers schema-v1 structure, supported kinds, selectors/slots, role-kind compatibility, safe paths, and archive/resource integrity boundaries. H3 payload validation remains H3-specific and runs when an H3 latent is inserted/accessed.

## H3 downstream use

```text
MMH3 Load -> Get Latent -> H3 AV Separate -> upscale video -> H3 AV Combine -> H3 refine
MMH3 Load -> Get Latent -> H3 continuation adapter
Load A + Load B -> H3 Compatibility -> future seam/stitch adapter
```

The packet preserves exact joint AV latent state. `naive_latent_concat_safe` remains false by design.

0.2 currently has 44 passing core tests, including schema-v1 rejection, explicit latent layout, notes, Export, Compare, preview checksum exclusion/cache freshness, lazy round-trip, H3 provenance/compatibility, exact tensor/mask/audio serialization, path safety and atomic save.
