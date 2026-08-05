# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
