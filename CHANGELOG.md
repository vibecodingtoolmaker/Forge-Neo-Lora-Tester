# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Current development snapshot: **v0.3.0-alpha.2** on `develop`. This snapshot is not
tagged or published as a GitHub release; `v0.2.0-beta.1` remains the public release.
The previous development snapshot, **v0.3.0-alpha.1**, is commit `7ad1cbc`.

### Added

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

- Generalized fixed-seed matrix, disk-spooling, model-unload, output-retention, and
  recovery handling so LoRA and Embedding comparisons share the same safety path.
- Kept the loaded checkpoint and its existing Forge embedding databases as the final
  generation-time compatibility authority; no additional text encoder is loaded.
- Replaced the single global Embedding prompt target with per-row targeting so one
  matrix run can contain both positive and negative Embeddings.

### Fixed

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

[Unreleased]: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester/compare/v0.2.0-beta.1...HEAD
[0.2.0-beta.1]: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester/compare/v0.1.0-beta.1...v0.2.0-beta.1
[0.1.0-beta.1]: https://github.com/vibecodingtoolmaker/Forge-Neo-Lora-Tester/releases/tag/v0.1.0-beta.1
