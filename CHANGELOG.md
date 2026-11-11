# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added an optional sequential multiple-model runner for exactly one selected LoRA
  or for prompt-only comparison across up to 25 selected checkpoints normally and
  up to 150 checkpoints in Extreme Run Mode.
- Added an internal prompt-only/no-op LoRA case that reuses the fixed-seed,
  disk-spooling, matrix, and recovery paths without loading a fake LoRA file.
- Added an always-visible planned-run summary for item, cell, reference, checkpoint,
  generation, and Standard/Extreme limit counts.
- Added checkpoint-folder selection with optional recursive subfolder inclusion to
  the sequential multiple-model workflow, matching the LoRA/Embedding inventory UX.
- Added Forge Batch Count support to multiple-model runs. Each count creates a
  Forge-compatible seed/subseed round shared across every selected checkpoint and
  comparison cell, with independent seed-labeled matrix output.
- Added a shared-overview organization choice: model rows with seed-round columns by
  default, or one shared overview ordered by seed round with models grouped together.

### Changed

- Kept the existing current-model LoRA and Embedding workflows as the default and
  isolated multiple-model execution as an outer wrapper around complete Forge jobs.
- Fully unload the checkpoint, text encoder, VAE, processing references, and loaded
  LoRA registry after every model job, then restore the original Forge selection in
  a `finally` path.
- Stop before loading another checkpoint if the previous full unload cannot be
  confirmed; retain completed outputs and still attempt original-model restoration.
- Resolve the LoRA registry lazily during cleanup instead of importing it eagerly at
  extension startup.
- Combine completed per-model pages into one shared comparison grid instead of
  returning one gallery grid per checkpoint; retain safety-driven page splitting.
- Count Batch Count-expanded cells against the per-model Standard/Extreme limit before
  loading the first checkpoint. Keep Batch Size serialized to one for disk-first
  processing and recovery safety.
- Store exact per-page seed metadata when an overview page contains one seed. Mark
  mixed-seed pages with summary seed `-1` and retain their complete seed/subseed list
  in PNG information and the recovery manifest.
- Removed the disabled adaptive RAM watchdog, its read-only UI, background sampler,
  Windows commit checks, callback parameters, and dormant manifest fields.
- Replaced the nested Per-Item, Multiple models, and Advanced accordions with four
  workflow tabs: Selection, Models & Run Size, Grid, and Output & Safety.

### Fixed

- Render Seeds-first results as one labeled row per seed inside the shared overview:
  one left-hand seed label followed by the selected models as columns. Do not let the
  ordinary Matrix Columns setting mix two seeds into the same visual row. Safety
  limits can still split either organization into multiple pages.
- Honor Forge Batch Count in the existing current-model LoRA and Embedding paths by
  creating the same independently labeled seed/subseed rounds used by multi-model runs.
- Set Forge's multi-model progress total before the first nested checkpoint job so
  progress and ETA cover every model and seed round instead of exceeding 100 percent.
- Remove only tracked extension-owned per-model temporary grids after a complete run
  and validated shared overview; retain recovery sources on every partial path and
  preserve user-saved Forge grids.
- Inventory Forge's primary `lora_dir` and every additional `lora_dirs` root, while
  deduplicating repeated roots and overlapping files.

### Safety

- Persist per-model completion, failure, unload, and original-model restoration
  status in the recovery manifest. Model-family preflight remains intentionally
  deferred; Forge loading and generation errors are recorded per checkpoint.
- Preserve disk-first spooling, full pre-matrix model unload, fixed cell limits,
  65,000-pixel axis and 89,000,000-pixel page-area limits, multi-page composition,
  and recovery-safe Matrix-only cleanup after removing the watchdog.

## [0.3.0-beta] - 2026-08-14

### Added

- Added opt-in separate grids per tested LoRA or Embedding, in selection order.
- Added an output dependency: the combined matrix remains required while per-item
  grids are disabled, then becomes optional when per-item grids are enabled.
- Added aligned combined-matrix comparison layouts with LoRAs as rows or columns,
  dynamic safe-capacity guidance, 0.5-step symmetric range hints, an initial ten-item
  limit, and automatic aligned splitting across safe pages.
- Made every per-LoRA/per-Embedding grid a single horizontal weight row, independent
  of the Standard combined-matrix column setting, with safe one-row page splitting.
- Added an Embedding Test mode for textual inversion embeddings compatible with both
  text encoders of the loaded SDXL, Pony, or Illustrious checkpoint.
- Added folder selection, global and per-Embedding weight ranges, Start/End placement,
  positive/negative prompt targets, and matching Hi-Res prompt injection.
- Added reusable Forge UI-preset/model-family detection and header-only discovery of
  SDXL dual-CLIP Safetensors embeddings before a checkpoint is loaded.
- Added live preset synchronization: changing between `xl` and another Forge UI
  preset dynamically refreshes or clears the compatible Embedding inventory.
- Added Embedding trigger discovery from neighboring `.json` metadata with the
  filename stem as fallback, plus editable per-Embedding trigger cells and
  duplicate-safe prompt insertion.
- Added a per-Embedding `0`/`1` table column for routing each item independently to
  the positive or negative prompt, including its matching Hi-Res prompt.

### Changed

- Applied repeated-reference, automatic page splitting, Gallery-safe area limits,
  disk spooling, and Matrix-only recovery validation to every requested grid output.
- Generalized fixed-seed matrix, disk-spooling, model-unload, output-retention, and
  recovery handling so LoRA and Embedding comparisons share the same safety path.
- Kept the loaded checkpoint and its existing Forge embedding databases as the final
  generation-time compatibility authority; no additional text encoder is loaded.
- Replaced the single global Embedding prompt target with per-row targeting so one
  matrix run can contain both positive and negative Embeddings.

### Fixed

- Restored Forge UI startup on bundled Gradio 4.40: an empty fixed-row Dataframe
  caused the frontend to fail at `Loading`; selection-owned tables now retain the
  compatible dynamic data model while hiding manual row insertion.
- Released an active Dataframe cell editor on table scroll after committing its current
  value, preventing Gradio's virtualizer from pulling the view back to that cell.
- Hid comparison-capacity guidance while Standard order is active or the combined
  overview is disabled.
- Prevented Gradio from inserting editable blank rows into the selection-driven LoRA
  and Embedding settings tables, and discarded legacy empty rows server-side.
- Added an 89,000,000-pixel matrix-page area ceiling. Large valid grids now split
  before Gradio reopens them through Pillow, preventing a post-build
  `DecompressionBombError` while retaining the existing 65,000-pixel per-axis limit.

## [0.2.0-beta.1] - 2026-08-05

### Changed

- Enabled the fixed-seed reference image by default and repeat it on every matrix page.
- Raised the per-page image-dimension ceiling from 60,000 to 65,000 pixels while retaining automatic page splitting.

### Added

- Added an explicit Extreme Run Mode that raises the normal 500-cell limit to 10,000 cells with a long-running stability and disk-usage warning.

## [0.1.0-beta.1] - 2026-08-03

### Changed

- Temporarily disabled the adaptive RAM watchdog for the v0.1.0 release while its thresholds are evaluated further. Disk spooling and pre-matrix model unloading remain active.

### Fixed

- Made trigger positioning explicit: Start places triggers before the prompt, while End places them after the LoRA tag.
- Retained Matrix-only source images after an interrupted or otherwise incomplete generation, even when a valid partial matrix can still be created.
- Blocked incomplete or invalid per-LoRA weight settings with a visible explanation instead of silently falling back to global weights.
- Prevented the LoRA settings table from repeatedly reclaiming focus while the user scrolls or moves to another control.

### Added

- Initial public beta release for Forge Neo.
- Fixed-seed LoRA and weight-range comparisons in labeled matrix pages.
- Per-LoRA trigger-word and weight overrides.
- Folder-scoped LoRA selection with optional subfolder inclusion.
- Disk-spooled source images and recovery manifests.
- Adaptive matrix-page packing and an experimental RAM-watchdog implementation that
  remains disabled in this release.
- Optional checkpoint, text-encoder, and VAE unload before matrix creation.
- Recovery-safe matrix-only output mode.

[Unreleased]: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester/compare/v0.3.0-beta...HEAD
[0.3.0-beta]: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester/compare/v0.2.0-beta.1...v0.3.0-beta
[0.2.0-beta.1]: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester/compare/v0.1.0-beta.1...v0.2.0-beta.1
[0.1.0-beta.1]: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester/releases/tag/v0.1.0-beta.1
