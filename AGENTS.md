# AGENTS.md

These instructions apply to the entire Forge Neo LoRA Tester repository.

## Start here

- Read `PROJECT_STATUS.md`, `README.md`, and `CHANGELOG.md` before changing code.
- Check `git status --short --branch`, `git branch -vv`, and the recent log.
- Work only inside this extension repository unless the user explicitly expands
  scope. The surrounding Forge checkout contains user models, outputs, settings,
  and unrelated local changes that must be preserved.
- The normal development branch is `develop`. Leave the development installation
  on `develop` after release work.

## Branch and release policy

- Implement and test changes on `develop` first.
- `develop` contains development-only tests and handoff documentation.
- `main` is the minimal public release package. Do not add `tests/`,
  `PROJECT_STATUS.md`, or `AGENTS.md` to `main` unless the maintainer explicitly
  changes this policy.
- `develop` and `main` intentionally use parallel/cherry-picked commits. Do not
  assume they can be fast-forwarded or blindly merged.
- For a release, synchronize both branches with their remote tips, commit/test on
  `develop`, then cherry-pick only public product/documentation changes to `main`.
  Resolve the development-test path as absent on `main`.
- Verify that `main:scripts/lora_tester.py` and
  `develop:scripts/lora_tester.py` have identical blob hashes before tagging.
- Tags and GitHub releases must point to `main`, not `develop`.
- Beta versions are GitHub pre-releases. Update the README status, dated changelog
  section, comparison links, and release notes together.
- Do not push branches, create tags, or publish/edit GitHub releases without explicit
  user authorization for that release action.

## Safety and architecture invariants

- Do not introduce hard-coded installation, model, output, user-profile, or drive
  paths. Derive paths from Forge processing objects, `shared.cmd_opts`,
  `shared.models_path`, or the extension/repository location.
- Do not add runtime network requests, telemetry, downloaded code execution, or
  automatic dependency installation. Metadata files are untrusted data.
- Preserve fixed-seed comparison: every baseline/LoRA/weight cell must use the first
  resolved seed, including Embedding/weight cells and when the user enters seed `-1`.
- Before model load, Embedding inventory may use Forge's selected UI preset and a
  header-only Safetensors scan. Generation-time validation and injection must reuse
  Forge's currently loaded SDXL text encoders and their compatibility-filtered
  embedding databases. Do not load a second text encoder or language model. Keep the
  initial Embedding Test scope to dual-encoder SDXL-compatible families (SDXL, Pony,
  and Illustrious).
- Preserve the filename stem as Forge's technical textual-inversion token. A distinct
  trigger from neighboring Embedding JSON metadata is additional prompt text; when no
  usable trigger exists, use the filename stem as the fallback without injecting it
  twice.
- Preserve per-Embedding prompt routing: table value `0` means positive and `1` means
  negative. Reject other values visibly, and apply each row's target to the matching
  normal and Hi-Res prompt without turning it into a run-wide setting.
- Preserve disk-first processing. Completed full-size cells must be atomically spooled
  and large Forge/PIL result references released promptly.
- Preserve Matrix-only recovery safety. Never delete source images unless all
  requested cells completed without interruption and every matrix page was written
  and validated. On ambiguity, retain sources and the recovery manifest.
- Preserve automatic multi-page composition and the 65,000-pixel per-axis ceiling
  plus the 89,000,000-pixel Pillow/Gradio-safe page-area ceiling unless
  encoder/gallery testing justifies a deliberate change.
- Normal runs are limited to 500 cells. The explicit Extreme Run Mode raises the
  fixed limit to 10,000; never silently truncate an over-limit request.
- Reference generation is enabled by default. When present, render the reference as
  a separate row on every successful matrix page without reordering comparison cells.
- Keep checkpoint/text-encoder/VAE unload enabled by default. Use Forge's full
  checkpoint unload path rather than a VRAM-only eviction.
- `RAM_WATCHDOG_ENABLED` must remain `False` until the maintainer explicitly approves
  reactivation after cross-system validation.
- The existing startup LoRA warning is an open eager-import bug. Fix it by lazy
  resolution at unload time; do not replace it with a new startup LoRA scan.

## Code and licensing

- Keep SPDX header `AGPL-3.0-only` and copyright
  `Copyright (C) 2026 vibecodingtoolmaker` in source/test files.
- Maintain compatibility with Forge Neo's public script/processing APIs and the
  installed Gradio/Pillow versions. Avoid adding dependencies when Forge or the
  Python standard library already provides the needed behavior.
- Preserve user changes and unrelated dirty files. Do not use destructive Git
  commands such as `git reset --hard` or discard edits without explicit approval.
- Use atomic writes for images/manifests and validate paths remain within
  Forge-derived output boundaries.

## Required verification

- Run the development tests with Forge's Python environment:

From the Forge root, run:

```powershell
& '.\venv\Scripts\python.exe' -m unittest discover `
  -s '.\extensions\Forge-Neo-Lora-Tester\tests' -v
```

- Run a syntax compilation with the same interpreter for
  `scripts/lora_tester.py`.
- Run `git diff --check` and inspect the complete relevant diff.
- For matrix/recovery changes, add or update a unit test on `develop`.
- Automated tests do not replace manual Forge tests. For behavior changes, report the
  exact txt2img/img2img, baseline, output-retention, model-unload, interruption, and
  multi-page cases that still require manual testing.
