# Forge Neo LoRA Tester – Project Status

Last updated: 2026-08-14

Repository: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester

Local development checkout: `H:\DEVELOPMENT-sd-webui-forge-neo\extensions\Forge-Neo-Lora-Tester`

This file is the handoff document for a new Codex chat. Read `AGENTS.md` first,
then this document, `README.md`, and the current Git diff before changing code.

## Current state

- Current public release: **v0.3.0-beta** (GitHub pre-release, 2026-08-14).
- Release tag: `v0.3.0-beta` -> `main` commit `d67569e`.
- Release preparation on `develop`: commit `f274768`; the feature snapshot immediately
  before it is `1a784f1`.
- Previous public release: **v0.2.0-beta.1**, tag to `main` commit `14c6dae`.
- The development installation must normally remain on branch `develop`.
- `develop`, `main`, and the annotated release tag were pushed atomically. The public
  repository and pre-release are available at the repository URL above.
- `main` intentionally contains only the public extension package. The test suite
  is intentionally retained on `develop` and omitted from `main`.
- License: `AGPL-3.0-only`; copyright holder: `vibecodingtoolmaker`.

## Release completion — 2026-08-14

- Public release URL:
  `https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester/releases/tag/v0.3.0-beta`.
- Remote `main` is `d67569efd04588bf4821d22bf65edddc36c9cc56`.
- Remote release-preparation `develop` commit is
  `f274768ff4c8bc2150df96075a3239b30a81eb74`; this status-only follow-up does not
  change the released product files.
- The annotated `v0.3.0-beta` tag resolves to the exact `main` commit above.
- All seven public product/documentation blobs match between `main` and `develop`.
  `AGENTS.md`, `PROJECT_STATUS.md`, and `tests/` remain development-only.
- The full 43-test suite passed on `develop`; Python production syntax, JavaScript
  syntax, Ruff fatal-error checks, and Git diff checks passed for the release tree.
- GitHub reports the repository as public and the release as a published prerelease.
  This repository has no GitHub Actions run for the release push.
- Manual generation remains open for final Gallery delivery, visible Embedding
  semantics, Hi-Res, img2img, every per-item/combined layout combination, both
  retention modes, forced multi-page splitting, and broader SDXL/Pony/Illustrious
  coverage. The release notes preserve this confidence boundary.

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
- Matrix pages split before either the 65,000-pixel axis ceiling or the dynamic
  89,000,000-pixel area ceiling is crossed, keeping saved pages safe for Pillow and
  Gradio to reopen.
- Individual images are retained in dedicated txt2img/img2img run directories when
  the safe output-retention mode is selected.
- Live Embedding inventory and preset switching work before checkpoint load. A real
  txt2img Embedding run completed sampling and saved its matrix pages; visual prompt
  semantics and the post-alpha.2 Gallery handoff still require focused confirmation.

## Day-end status — 2026-08-11

### Confirmed today

- `develop` code is synchronized with `origin/develop` at `df53e09`
  (`v0.3.0-alpha.2`); `main`, tags, and the public `v0.2.0-beta.1` release are
  intentionally unchanged.
- A real Embedding Test reached `Total progress: 9275/9275` in about 17 minutes and
  completed matrix construction and file output.
- The run then exposed a delivery-only failure when Gradio reopened a
  `3082 x 63904` page (196,952,128 pixels) through Pillow. The generated images and
  matrices themselves were retained.
- The failure was root-caused to total page area rather than Embedding injection or
  sampling. `v0.3.0-alpha.2` now splits pages before the safe area ceiling as well as
  the existing per-axis ceiling.
- The previous extreme LoRA test did not raise the hard error because its three-column
  page was narrower; the same page height therefore stayed below Pillow's hard limit.
  This was a latent matrix-layout issue, not an Embedding-only defect.

### Confidence boundary for the next session

- Proven in a live Forge run: Embedding discovery, selection, processing startup,
  complete sampling, matrix composition, and matrix file persistence.
- Still to confirm manually: post-alpha.2 Gallery delivery, visible Embedding effect,
  JSON trigger versus filename fallback, mixed `0`/`1` prompt targets, Hi-Res prompt
  injection, img2img, and broader SDXL/Pony/Illustrious coverage.

## Day-end handoff — 2026-08-13

### Exact Git state

- Worktree: `H:\DEVELOPMENT-sd-webui-forge-neo\extensions\Forge-Neo-Lora-Tester`.
- Branch: `develop`; `HEAD` and `origin/develop` are still `df53e09`
  (`Add v0.3.0-alpha.2 matrix page safeguards`).
- The complete 2026-08-13 change set is local and uncommitted. Modified files:
  `CHANGELOG.md`, `PROJECT_STATUS.md`, `README.md`,
  `javascript/lora_tester_dataframe.js`, `scripts/lora_tester.py`, `style.css`, and
  `tests/test_lora_tester_recovery.py`.
- No commit, tag, push, release, Forge-core edit, or change to `main` was made.

### Implemented locally

- Per-item grids are now optional and disabled by default. With them disabled, the
  former output path remains: retained source images according to the retention mode
  plus one required combined matrix. Enabling them unlocks the option to include or
  omit the combined overview.
- The combined matrix now offers Standard order plus aligned comparison layouts:
  LoRAs/Embeddings as rows (weights side-by-side) or as columns (weights underneath).
  Comparison layouts require identical weight sequences, start with a ten-item limit,
  and split the weight axis into aligned safe pages when necessary.
- UI capacity guidance follows the current Forge width/height, selection count,
  reference, legend, and margin settings. It reports safe per-page counts and a
  symmetric 0.5-step range hint; actual generated/Hi-Res cell dimensions remain the
  final authority. The guidance is hidden for Standard order and whenever the combined
  overview is disabled.
- Per-item grids ignore the Standard combined-matrix column setting and keep all safe
  weights for one LoRA/Embedding in a single horizontal row. Overwide rows split into
  additional safe one-row pages.
- Selection-driven Dataframes keep Gradio's compatible dynamic data model while hiding
  manual row insertion and filtering empty rows server-side. Scrolling commits and
  releases an active editor so the virtualized table no longer jumps back to its cell.
- Duplicate display names remain separate because grouping uses the selected relative
  model path, while filenames include an ordinal and a sanitized display label.
- The fixed-seed reference is repeated on every individual and combined page. All
  outputs share the existing 65,000-axis/89M-area limits and Matrix-only cleanup is
  allowed only after every requested output page validates.
- Automated coverage is currently 43 passing tests. Manual Forge Gallery validation
  is still required for legacy output, per-item/combined combinations, both comparison
  orientations, forced aligned splitting, Hi-Res, both retention modes, and
  txt2img/img2img.

### Root causes found and fixed

- The empty visible settings row came from Gradio's dynamic Dataframe placeholder,
  not from a real selected LoRA. Empty rows are now hidden in the browser and ignored
  by server-side normalization.
- Changing an empty table to `row_count=(0, "fixed")` made Forge stop at `Loading`.
  Forge's bundled Gradio 4.40 frontend expected row data and failed while slicing an
  undefined value. The compatible dynamic model is therefore deliberately retained;
  only its manual row-add UI is suppressed.
- The table scroll jump was caused by the virtualized Dataframe retaining the active
  editor and restoring its cell into view. A wheel action now commits and blurs that
  editor before scrolling continues.

### Verification and confidence boundary

- The full extension suite passes all 43 tests in the Forge venv.
- A stricter `python -S` run proves the recovery/UI tests that do not need Forge
  packages, but the complete discovery run is not dependency-free: it stops when
  `tests/test_model_family.py` imports `torch`. This is a test-portability follow-up,
  not a failure in the new matrix tests.
- Python syntax compilation, Ruff fatal-error checks, JavaScript syntax checking, and
  `git diff --check` pass. Git reports only the existing LF-to-CRLF conversion warning
  for `style.css`.
- A real Forge browser smoke test loaded the full UI, showed capacity guidance only
  for the two comparison orientations, hid it again in Standard order, and produced no
  LoRA Tester console error.
- A focused live Dataframe test scrolled from `scrollTop 977` to `327` and remained at
  `327` after 1.8 seconds; the former focus-driven return jump did not recur.
- Not yet proven by a complete live generation: Gallery order and visual layout for
  legacy output, per-item grids with combined overview on/off, both comparison
  orientations, forced safe splitting, reference on/off, Hi-Res, Matrix-only cleanup,
  both retention modes, and txt2img/img2img. Preserve this distinction in release
  notes; automated and browser UI checks do not prove the generation result.

### Tomorrow's primary task: one LoRA across multiple checkpoints

Build the first scaffold around exactly one selected LoRA and up to ten selected
checkpoints. The existing weight parser remains the shared axis, including ranges such
as `-5:5:0.5` (21 weights). Every checkpoint must receive the same prompt, resolved
seed, sampler, resolution, and exact weight sequence.

Required first-stage behavior:

1. Snapshot the complete original Forge model selection needed for reliable restore.
2. Preflight the LoRA/checkpoint pairs and show compatible, uncertain, or incompatible
   status before or at model load; never infer compatibility from filenames alone.
3. Load each checkpoint once, perform the authoritative runtime LoRA-key compatibility
   check, generate every weight for that checkpoint, and persist the completed row.
4. On incompatibility or generation failure, record and label the skipped/failed row
   without discarding already completed rows or their source images.
5. Build one horizontal full-resolution weight row per checkpoint and, when requested,
   a combined overview with checkpoints vertically and weights horizontally.
6. Restore the original Forge selection in a `finally` path even after interruption or
   failure; let Forge own the actual model loading and offload lifecycle.

Compatibility should be layered:

- Use LoRA embedded/sidecar metadata such as `ss_base_model_version`,
  `modelspec.architecture`, `ss_network_module`, or Civitai `baseModel` only as an
  early family hint.
- Classify checkpoint architecture from its Safetensors tensor names/shapes without
  decoding the weights. Embedded checkpoint metadata is too sparse to be the sole
  decision source.
- Once a checkpoint is loaded, reuse Forge's own LoRA key mapping as the final
  authority. A metadata-unknown pair that maps successfully is valid; a hard family
  mismatch or failed runtime mapping is skipped and retained in the manifest.
- Relevant read-only Forge references are `modules/sd_models.py` (checkpoint/header
  metadata), `backend/loader.py` (architecture and engine selection), and
  `extensions-builtin/sd_forge_lora/networks.py` (loaded-model key matching). Keep all
  new implementation inside this extension.

The target 10-checkpoint by 21-weight overview cannot generally remain full resolution:
at typical source sizes it exceeds the existing 89M-pixel Gallery boundary before
labels are added. Keep full-resolution per-checkpoint rows, then design a downscaled
overview (and later possibly tiled/zoomable output) without weakening the 65,000-axis,
89M-area, disk-first, or recovery guarantees. Multiple LoRAs across multiple
checkpoints is the next expansion after this one-LoRA scaffold is stable.

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

## Changes implemented for v0.3.0-alpha.2 (included in v0.3.0-beta)

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

## Changes implemented for v0.3.0-alpha.1 (included in v0.3.0-beta)

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
  cleared for `flux`, and it returned to 264 after switching back to `xl`.
- A live txt2img Embedding run subsequently completed sampling and saved matrix pages.
  Gallery preprocessing then exposed the page-area issue fixed in alpha.2; visual and
  prompt-semantic validation remains incomplete.
- The real 264-item compatible inventory resolves 165 triggers from `.json` and 99
  from filename fallback. Four metadata triggers intentionally differ from their local
  filenames and therefore exercise the separate technical-token/activation-phrase path.

## Planned feature roadmap

1. **LoRAs across multiple checkpoints**
   - First scaffold: exactly one LoRA, up to ten checkpoints, and one identical user
     weight sequence per checkpoint.
   - Load each checkpoint once and create one horizontal weight row per checkpoint.
   - Combined layout: checkpoints on the Y axis and weights on the X axis; retain
     full-resolution per-checkpoint rows even when the combined overview must be
     downscaled or split.
   - Add layered metadata/header/runtime compatibility checks, persistent failed-row
     status, and fail-safe restoration of the original Forge model selection.
   - Expand to multiple LoRAs only after the sequential model lifecycle is proven.
2. **Embedding Test follow-up**
   - Manually validate and stabilize the new SDXL/Pony/Illustrious implementation,
     then revisit broader model-family support if Forge expands compatible textual
     inversion handling.
3. **Multiple images/seeds per tested LoRA**
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

9. **Extreme one-row reflow is not implemented**
   - If one labeled row alone exceeds an active page limit, alpha.2 rejects it safely
     and retains recovery data instead of returning an unsafe image to Gradio.
   - A future enhancement could reflow that row into fewer columns and still produce a
     deliverable matrix page automatically.

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
- Post-`v0.3.0-beta` Gallery delivery of newly area-split pages from a real Embedding
  run.
- Visual and prompt-semantic confirmation of the Embedding Test in txt2img across
  SDXL, Pony, and Illustrious; one live SDXL-family run has completed through matrix
  persistence but not final Gallery delivery.
- Embedding Test in live img2img.
- Mixed positive and negative Embedding targets with Hi-Res fix enabled and disabled.
- Mixed Embedding folder layouts, duplicate filename stems, incompatible SD1.x
  embeddings, model switching followed by refresh, and multi-page Embedding runs.

## Recommended next steps

1. Before implementing cross-checkpoint generation, identify the exact Forge-owned
   reload API and processing boundary that can switch models sequentially without a
   second persistent model or recursive processing call. Add pure run-plan, family-
   classifier, manifest, and restore-path tests first.
2. Add the one-LoRA/up-to-ten-checkpoint selector and shared-weight run plan described
   in the 2026-08-13 handoff. Keep compatibility classification extension-local and
   make Forge's loaded-model key mapping authoritative.
3. Implement sequential checkpoint rows, immediate disk persistence, failed-row
   labeling, combined overview scaling/splitting, and unconditional original-model
   restoration. Verify one same-family pair before widening coverage.
4. Run a short txt2img test with three LoRAs and `0:5:0.5`: first in legacy output,
   then with LoRAs as rows and as columns. Confirm aligned weights and Gallery order.
5. Enable per-item grids, test the combined matrix on/off dependency, then force
   aligned and per-item multi-page splitting with reference on/off.
6. Repeat the output checks in img2img, with Hi-Res, both retention modes, and once
   with Embedding Test.
7. Run a short post-`v0.3.0-beta` txt2img smoke test with one Embedding
   whose JSON trigger differs from its filename and one filename-fallback Embedding.
   Route one row to positive (`0`) and one to negative (`1`), then confirm the visible
   effect, console prompt composition, saved matrices, and Gallery delivery.
8. Repeat with Hi-Res fix and deliberately force at least two matrix pages to verify
   area-driven splitting and the repeated reference in the live Gallery.
9. Extend the smoke test to img2img and representative SDXL, Pony, and Illustrious
   checkpoints, then cover model unload on/off and both output-retention modes.
10. Fix the eager LoRA-system import and eliminate the misleading startup warning.
11. Run a short Forge smoke test after that fix: normal generation with the tester
   disabled, one baseline + two LoRAs, model unload on/off, txt2img and img2img.
12. Visually verify repeated references on at least a two-page matrix with legends
   both enabled and disabled.
13. Add tests for the visible over-limit block in `before_process()`, not only the
   pure 500/10,000 limit helper.
14. Test 65,000-pixel page encoding in every supported `grid_format`; reduce the
   ceiling if any encoder or gallery rejects it.
15. Fix the margin/date-folder beta limitations.
16. Design static settings persistence without persisting dynamic selections.
17. Add optional preflight information (planned cells, approximate rows/pages, and a
   conservative disk/time warning) without adding another user-editable cell limit.
18. Revisit adaptive RAM protection only after gathering measurements from different
   RAM sizes, model families, pagefiles, and loading strategies.
19. Consider manifest-based resume support for multi-hour Extreme Run jobs.

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
- `_group_comparison_cells()`: groups weights by relative item path without merging
  duplicate display names.
- `_comparison_capacity()`: safe one-page aligned weight capacity from resolution,
  item count, legend, margin, and reference geometry.
- `_build_comparison_matrix_pages()`: aligned row/column layout and weight-axis parts.
- `_build_per_item_matrix_pages()`: horizontal per-item rows and safe width-based parts.
- `_build_requested_matrices()`: emits per-item pages first and the optional combined
  overview last.
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
- Commits and releases an active cell editor before table scrolling.

### `tests/test_lora_tester_recovery.py` (`develop` only)

- Matrix-only cleanup/recovery decisions.
- Partial-run source retention and manifest preservation.
- Complete-run cleanup including the temporary reference row.
- Normal/Extreme cell limits.
- Reference-row inclusion in page estimates and every composed page.
- 65,000-pixel page boundary.
- Pillow/Gradio-safe 89,000,000-pixel page area and automatic area-based splitting.
- Empty Dataframe-row filtering, optional per-item output, combined-output dependency,
  duplicate-label separation, horizontal per-item rows and safe splitting, aligned
  row/column layout, capacity hints, and output order.
- Embedding prompt/trigger composition, sidecar precedence and filename fallback,
  settings-table migration, dual-encoder compatibility gates, preset resolution,
  Safetensors header discovery, shared weight validation, baseline classification,
  and UI/callback argument order.

### Documentation and metadata

- `README.md`: public user documentation and known limitations.
- `CHANGELOG.md`: public release history and the current `[Unreleased]` section.
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
