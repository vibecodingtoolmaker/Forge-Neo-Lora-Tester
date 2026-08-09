# Forge Neo LoRA Tester – Project Status

Last updated: 2026-08-09

Repository: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester

Local development checkout: `H:\DEVELOPMENT-sd-webui-forge-neo\extensions\Forge-Neo-Lora-Tester`

This file is the handoff document for a new Codex chat. Read `AGENTS.md` first,
then this document, `README.md`, and the current Git diff before changing code.

## Current state

- Current public release: **v0.2.0-beta.1** (GitHub pre-release, 2026-08-05).
- Current development snapshot: **v0.3.0-alpha.2** on `develop` (unreleased and
  intentionally untagged).
- Previous development snapshot: **v0.3.0-alpha.1**, commit `7ad1cbc`.
- Release tag: `v0.2.0-beta.1` -> `main` commit `14c6dae`.
- Release feature commit on `main`: `c27c621`.
- Current `develop` release commit: `2bb1bee`.
- Current `develop` feature commit: `af2432a`.
- The development installation must normally remain on branch `develop`.
- At the start of this handoff update, `develop` was clean and synchronized with
  `origin/develop`. Re-check `git status --short --branch`; the new handoff files
  may still be local and uncommitted.
- `main` intentionally contains only the public extension package. The test suite
  is intentionally retained on `develop` and omitted from `main`.
- License: `AGPL-3.0-only`; copyright holder: `vibecodingtoolmaker`.

## Proven working functionality

- txt2img and img2img integration through Forge's always-on script lifecycle.
- The extension fully switches off when its accordion/checkbox is disabled.
- Folder-scoped LoRA inventory with optional recursive subfolder inclusion.
- First activation and manual refresh populate the LoRA list and editable table.
- Trigger words from `.civitai.info` and `.json`, plus per-LoRA manual overrides.
- Positive, negative, ascending, and descending weight ranges.
- One fixed seed for the baseline and every LoRA/weight cell, including seed `-1`
  after the first random seed has been resolved.
- Disk spooling of every completed source cell; decoded full-size images are not
  intentionally retained in Forge's result list.
- Matrix-only cleanup is recovery-safe: source cells are deleted only after a
  complete run and successful validation of every matrix page.
- Interrupted runs create partial matrices and retain recoverable source images.
- Optional checkpoint, text encoder, and VAE unload before matrix composition;
  enabled by default and the main practical RAM-saving mechanism.
- Multiple matrix pages with compact labels and automatic page packing.
- Individual images are retained in dedicated txt2img/img2img run directories when
  the safe output-retention mode is selected.

## Changes implemented on 2026-08-05

### Extreme Run Mode

- Added a small opt-in **Extreme Run Mode** checkbox under Advanced Options.
- Normal limit remains 500 total matrix cells.
- Extreme limit is fixed at 10,000 total matrix cells.
- The generated baseline/reference counts as one matrix cell.
- Runs above the selected limit are visibly blocked; they are no longer silently
  truncated after 500 cells.
- The UI warns that extreme runs may take hours or days, use large amounts of disk
  space, and expose long-duration Forge/driver/model/extension/system instability.

### Matrix dimensions and repeated reference

- Increased `MAX_MATRIX_DIMENSION` from 60,000 to **65,000 pixels** per axis.
- Automatic page splitting remains mandatory. The 65,000 limit is independent of
  available RAM/VRAM and stays below common roughly 65.5k encoder boundaries.
- Reference generation is now enabled by default; the user must actively disable it.
- The reference is removed from the normal LoRA-cell sequence and rendered as a
  dedicated, centered reference row at the top of every successful matrix page.
  This avoids duplicating or reordering LoRA results.
- Page-size/RAM estimation includes the repeated reference row.
- Reference row files participate in cleanup, manifests, and recovery output.

### Validation and release

- Real-world stress test completed successfully:
  - 211 LoRAs
  - 13 weights per LoRA (`-3` through `3` in `0.5` steps)
  - 2,743 total LoRA cells reported by the user
  - 832 x 1216 pixels per generated image
  - total runtime: 4 h 24 min
- Seven `unittest` tests pass with Forge's Python environment.
- Forge-Python syntax compilation passes.
- `main`, `develop`, and tag `v0.2.0-beta.1` were pushed atomically.
- GitHub pre-release published with stress-test and upgrade notes.

## Changes implemented for v0.3.0-alpha.2 (develop, unreleased)

### Pillow/Gradio-safe matrix page area

- Confirmed a post-build Gallery failure on a valid `3082 x 63904` matrix page:
  196,952,128 pixels exceeded Pillow's 178,956,970-pixel hard error threshold.
- Added a dynamic ceiling of at most 89,000,000 pixels per matrix page, kept at or
  below the active Pillow warning limit. The existing 65,000-pixel per-axis ceiling
  remains independent and unchanged.
- Page packing now splits before either layout ceiling is crossed and reports each
  page's dimensions and total pixel count in the console.
- An individually oversized labeled row is rejected with recovery retained instead
  of being returned to Gradio as an unsafe disk fallback.
- Added regression coverage for the exact offending dimensions and area-driven page
  splitting with the reference repeated on every page.
- All 29 development tests pass with Forge's Python environment; Ruff, formatting,
  Forge-Python syntax, JavaScript syntax, and diff checks pass.

## Changes implemented for v0.3.0-alpha.1 (develop, unreleased)

### SDXL-compatible Embedding Test

- Added a **Test Type** switch between the existing LoRA workflow and a new
  textual-inversion Embedding workflow.
- Added reusable Forge UI-preset/model-family helpers. The explicit browser value has
  priority, with `shared.opts.forge_preset` as the pre-load fallback; `xl` maps to the
  SDXL/Pony/Illustrious family.
- Before model load, Embedding inventory reads only Safetensors headers and exposes
  files containing both `clip_l` and `clip_g`. It does not load tensor data, an
  encoder, or a model.
- At generation time, compatibility is revalidated against the two embedding
  databases owned by Forge's already loaded SDXL text encoders.
- Added a browser-to-Gradio preset bridge. The open Embedding UI now updates when the
  Forge preset changes, without Forge-core modifications.
- Only embeddings accepted by both SDXL text encoders are selectable. The first
  implementation is deliberately limited to SDXL-compatible dual-encoder families:
  SDXL, Pony, and Illustrious.
- Added folder-scoped Embedding selection, global and per-Embedding single/ranged
  prompt-attention weights, Start/End placement, and per-row positive/negative prompt
  targets (`0` = positive, `1` = negative).
- Added per-Embedding trigger discovery. A neighboring `.json` activation value is
  used when available; a missing/unusable JSON falls directly back to the filename
  stem. Forge's filename-derived technical Embedding token remains present while a
  distinct metadata trigger is inserted in addition; identical fallback values are
  not doubled.
- Positive or negative Hi-Res prompts receive the same per-cell Embedding injection.
- Embedding cells share the existing fixed-seed, disk-spooling, page-splitting,
  output-retention, model-unload, and recovery pipeline.
- Automated coverage now distinguishes an Embedding cell from the baseline and
  checks prompt/trigger composition, metadata precedence and fallback, table migration,
  shared range validation, SDXL dual-encoder gating, preset resolution, and real
  Safetensors header filtering.
- All 27 alpha.1 development tests passed with Forge's Python environment;
  Forge-Python syntax compilation and isolated Gradio UI construction also passed.
- Live Forge UI validation passed with the real local inventory: 267 Safetensors were
  scanned in about 1.3 seconds, 264 dual-CLIP Embeddings were shown for `xl`, the list
  cleared for `flux`, and it returned to 264 after switching back to `xl`. Live image
  generation remains outstanding.
- The real 264-item compatible inventory resolves 165 triggers from `.json` and 99
  from filename fallback. Four metadata triggers intentionally differ from their local
  filenames and therefore exercise the separate technical-token/activation-phrase path.

## Planned feature roadmap

1. **Per-LoRA matrix output**
   - Add output choices for one combined matrix plus one matrix per tested LoRA, or
     only the individual per-LoRA matrices without the combined matrix.
2. **LoRAs across multiple checkpoints**
   - Test one or more LoRAs against multiple selected checkpoints.
   - Intended matrix layout: LoRAs on the X axis and checkpoints on the Y axis.
3. **Embedding Test follow-up**
   - Manually validate and stabilize the new SDXL/Pony/Illustrious implementation,
     then revisit broader model-family support if Forge expands compatible textual
     inversion handling.
4. **Multiple images/seeds per tested LoRA**
   - Add incremental or random seed generation for several images per LoRA.
   - Offer separate per-seed matrices in addition to the aggregate comparison.

## Important technical decisions and invariants

### Fixed seed

`_force_batch_rng_seed()` rebuilds Forge's per-iteration RNG state so each cell
uses the first resolved seed instead of Forge's normal `seed + iteration` behavior.
Do not remove this reset when refactoring generation hooks.

### Disk-first processing

`postprocess_image_after_composite()` atomically saves each completed cell and
replaces large Forge/PIL references with a 1x1 placeholder. Matrix creation later
opens one source cell/row at a time. Disk usage may be large, but generation RAM
must not scale linearly with the number of cells.

### Recovery-safe Matrix-only mode

`_matrix_only_cleanup_decision()`, `_validate_matrix_pages()`, and
`_cleanup_spool_files()` form a safety boundary. Individual source files may be
permanently deleted only if all requested cells completed, generation was not
interrupted, and every matrix page exists and validates. On uncertainty or failure,
retain sources plus the manifest.

### Page splitting

`_estimate_page_peak()` rejects pages exceeding 65,000 pixels on either axis or
89,000,000 total pixels. The area ceiling stays below Pillow's default large-image
warning threshold because Gradio reopens returned Gallery files after matrix creation.
`_build_matrix_pages()` adds rows until the next row would cross a layout limit or,
if the experimental RAM system is later enabled, the calculated memory budget.
Splitting is expected behavior and not an OOM indication.

### Repeated reference

The baseline cell is converted to a one-column `reference-row` strip. It is passed
separately to every `_compose_matrix_page()` call, while normal LoRA rows keep the
configured column count. If reference generation is disabled, no reference row is
created or repeated.

### Model unload

`_unload_forge_model_for_matrix()` calls Forge's checkpoint unload path and clears
prompt/sampler references before allocating matrix canvases. A VRAM-only eviction
is insufficient because it may move weights into system RAM. Keep this option
enabled by default unless a regression is proven.

### RAM watchdog

The adaptive RAM implementation remains in the source, but
`RAM_WATCHDOG_ENABLED = False`. Its UI is visible but read-only. Do not re-enable it
without explicit cross-system testing and a deliberate release decision. Disk
spooling, model unload, dimension limits, and multi-page composition remain active.

### Branch and release layout

`develop` and `main` have intentionally parallel/cherry-picked histories rather than
a simple fast-forward relationship. Do not blindly merge one into the other.
Production code blobs must match, but `main` omits development-only tests and handoff
files. See `AGENTS.md` for the release procedure.

## Known bugs and open problems

### High priority

1. **Misleading startup warning / eager Forge-LoRA import**
   - Startup can log `[LoRA Tester] Warning: LoRA system not found`.
   - Cause: module-level import of `extensions_builtin.sd_forge_lora.networks`
     occurs before Forge has fully installed the built-in LoRA directory on the
     Python import path. This does not perform the filesystem LoRA scan.
   - Consequence: `LORA_AVAILABLE` remains false for the session, so the optional
     `networks.loaded_networks.clear()` step is skipped during model unload.
   - Intended fix: remove the eager import and resolve Forge's already-loaded
     top-level `networks` module lazily inside `_unload_forge_model_for_matrix()`.
     Validate expected attributes before clearing; do not add a new startup scan.

2. **Adaptive RAM watchdog is disabled**
   - Threshold behavior was too hardware-specific and caused premature stops after
     initial model/LoRA loading spikes.
   - The 2,743-cell run succeeded with the watchdog disabled and model unload enabled.
   - On 2026-08-09 the maintainer reported several days of extensive, stable use on
     their machine without the watchdog. This is strong single-system evidence, but
     not yet a basis for removing the existing disk/recovery safeguards or claiming
     cross-system stability.
   - A universal design must account for Windows commit/pagefile headroom, persistent
     model allocation, temporary loading spikes, and page-composition memory.

### Accepted beta limitations / backlog

3. **Matrix margin inconsistency**
   - Vertical row spacing works. Horizontal cell spacing currently applies only when
     `Draw Legend in Matrix Grid` is enabled.

4. **Grid date folders**
   - When model unload is enabled, saved grids use Forge's grid output root instead
     of Forge's configured date-subfolder layout.

5. **First Generate click may only commit a corrected table cell**
   - After correcting an invalid active Dataframe cell, the first Generate click can
     return focus to the table without starting. The second click starts normally.
   - This is a Gradio table-submit/focus issue; no generated data is lost.

6. **Cold-start invalid-input progress behavior**
   - On a cold model start, an invalid table value was once shown in the progress
     area and followed by delayed cleanup roughly 1–2 minutes later. Warm starts only
     showed the trace warning. Observe and reproduce before changing processing flow.

7. **No third trigger position**
   - Existing `Start` means before the user's prompt; `End` means after the injected
     LoRA tag. Backlog request: add `Before LoRA` to place trigger text immediately
     before `<lora:...>` while leaving the user's prompt first.

8. **Settings are not persisted**
   - Static UI settings reset between sessions. LoRA inventory, selected LoRAs,
     trigger values, and table rows are dynamic and should not be persisted blindly.
   - Backlog: persist safe static txt2img/img2img settings separately.

9. **Extreme one-row fallback edge case**
   - If one prebuilt row alone exceeds the 65,000-pixel limit, or if a future enabled
     RAM budget rejects even one row, the existing disk fallback returns that row
     directly. In that rare path the repeated reference is not composed into it.
   - Typical tested sizes do not hit this path. A future fix should reflow the source
     cells into fewer columns rather than returning an oversized row.

10. **No resume-after-restart workflow**
    - Manifests and sources preserve recovery data after interruption, but there is no
      UI to resume a multi-hour run after Forge or Windows restarts.

11. **Individual-image finalization can be optimized**
    - Retained images are first spooled under `tmp` and moved to the final run folder
      after generation. A future implementation could spool directly into the final
      folder while preserving atomic writes and recovery semantics.

## Not yet sufficiently tested

- A full 10,000-cell run. Current maximum validated run: 2,743 cells.
- Exact 65,000-pixel output with every Forge grid format, especially JPEG, and its
  display in Gradio/browser galleries.
- Manual visual confirmation that the reference row appears correctly on every page
  for multiple page counts, legends on/off, baseline on/off, and partial runs.
- Extreme multi-page img2img runs.
- Extreme runs with Matrix-only cleanup after full successful completion.
- Disk-full, permission-loss, path-length, antivirus-lock, and atomic-rename failures.
- Very wide rows: 10 columns with unusually large source images.
- Mixed or unexpected source image dimensions within one run.
- Low-RAM machines and systems with small/disabled Windows pagefiles.
- Long-run manual cancellation at several phases: generation, row-strip creation,
  page composition, image finalization, and cleanup.
- Upgrade testing from both `v0.1.0-beta.1` and a fresh clone of `v0.2.0-beta.1`.
- Compatibility with future Forge Neo/Gradio/Pillow changes.
- Embedding Test in live txt2img and img2img with SDXL, Pony, and Illustrious.
- Positive and negative Embedding targets with Hi-Res fix enabled and disabled.
- Mixed Embedding folder layouts, duplicate filename stems, incompatible SD1.x
  embeddings, model switching followed by refresh, and multi-page Embedding runs.

## Recommended next steps

1. Run short SDXL/Pony/Illustrious Embedding smoke tests in txt2img and img2img,
   covering positive/negative prompts, weight ranges, reference on/off, Hi-Res fix,
   model unload on/off, and both output-retention modes.
2. Fix the eager LoRA-system import and eliminate the misleading startup warning.
3. Run a short Forge smoke test after that fix: normal generation with the tester
   disabled, one baseline + two LoRAs, model unload on/off, txt2img and img2img.
4. Visually verify repeated references on at least a two-page matrix with legends
   both enabled and disabled.
5. Add tests for the visible over-limit block in `before_process()`, not only the
   pure 500/10,000 limit helper.
6. Test 65,000-pixel page encoding in every supported `grid_format`; reduce the
   ceiling if any encoder or gallery rejects it.
7. Fix the margin/date-folder beta limitations.
8. Design static settings persistence without persisting dynamic selections.
9. Add optional preflight information (planned cells, approximate rows/pages, and a
   conservative disk/time warning) without adding another user-editable cell limit.
10. Revisit adaptive RAM protection only after gathering measurements from different
   RAM sizes, model families, pagefiles, and loading strategies.
11. Consider manifest-based resume support for multi-hour Extreme Run jobs.

## Relevant files and functions

### `scripts/lora_tester.py`

- `RamMonitor`: dormant peak sampler used only if the release switch is enabled.
- `LoRaMetadataReader.get_lora_dir()`: derives Forge's configured LoRA directory.
- `LoRaMetadataReader.find_all_loras()`: recursive model and metadata inventory.
- `forge_neo_model_family.py`: portable preset normalization, model-family capability
  mapping, Embedding sidecar-trigger resolution, and header-only SDXL dual-CLIP
  Safetensors discovery.
- `EmbeddingMetadataReader.find_for_preset()`: creates the early preset-aware UI
  inventory without loading a model.
- `EmbeddingMetadataReader.find_compatible_embeddings()`: reuses Forge's loaded SDXL
  CLIP databases as the final authority and exposes true `clip_l` + `clip_g` entries.
- `LoRaTesterScript.ui()`: all txt2img/img2img controls and callbacks.
- `_parse_weight_spec()`: finite single/range parser, maximum 100 weights per item.
- `_resolve_embedding_target()`: validates per-row `0`/`1` prompt routing.
- `_compose_embedding_prompt()`: weighted positive/negative prompt injection.
- `_maximum_total_cases()`: chooses 500 normal or 10,000 Extreme limit.
- `_compose_lora_prompt()`: Start/End trigger placement and LoRA injection.
- `before_process()`: validates UI data, expands cases, creates run state/manifest.
- `_force_batch_rng_seed()`: enforces the shared resolved seed for every cell.
- `before_process_batch()`: applies per-cell prompts/LoRA tags and RAM-stop logic.
- `postprocess_image_after_composite()`: atomic per-cell disk spool and RAM release.
- `_unload_forge_model_for_matrix()`: checkpoint/text encoder/VAE unload.
- `_finalize_individual_images()`: moves retained source PNGs to final run folders.
- `_write_manifest()`: persistent recovery state.
- `_build_row_strips()`: creates labeled comparison/reference row PNGs on disk.
- `_estimate_page_peak()`: dimension and conservative memory estimate.
- `_build_matrix_pages()`: page packing and repeated-reference propagation.
- `_compose_matrix_page()`: creates/saves one final page.
- `_cleanup_spool_files()`: recovery-safe source/row/manifest cleanup.

### `javascript/lora_tester_dataframe.js`

- Makes editable Dataframe cells show a caret on one click.
- Places the caret at the start without selecting/deleting the cell value.
- Protects the read-only LoRA-name column.
- Handles Tab/Shift+Tab/Enter/arrow navigation and avoids focus reclaim loops.

### `tests/test_lora_tester_recovery.py` (`develop` only)

- Matrix-only cleanup/recovery decisions.
- Partial-run source retention and manifest preservation.
- Complete-run cleanup including the temporary reference row.
- Normal/Extreme cell limits.
- Reference-row inclusion in page estimates and every composed page.
- 65,000-pixel page boundary.
- Pillow/Gradio-safe 89,000,000-pixel page area and automatic area-based splitting.
- Embedding prompt/trigger composition, sidecar precedence and filename fallback,
  settings-table migration, dual-encoder compatibility gates, preset resolution,
  Safetensors header discovery, shared weight validation, baseline classification,
  and UI/callback argument order.

### Documentation and metadata

- `README.md`: public user documentation and known limitations.
- `CHANGELOG.md`: release history; current empty `[Unreleased]` section.
- `metadata.ini`: Forge extension description and AGPL identifier.
- `LICENSE`: complete AGPL-3.0-only license text.
- `AGENTS.md`: durable development/release/safety rules for future Codex chats.

## Verification commands

Run from `H:\DEVELOPMENT-sd-webui-forge-neo`:

```powershell
& '.\venv\Scripts\python.exe' -m unittest discover `
  -s '.\extensions\Forge-Neo-Lora-Tester\tests' -v
```

Run from the extension repository:

```powershell
git status --short --branch
git diff --check
git branch -vv
```

The automated tests intentionally avoid importing a live Forge processing stack.
They do not replace manual Forge UI/generation tests.
