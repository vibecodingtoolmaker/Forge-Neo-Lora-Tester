# LoRA Tester for Forge Neo

LoRA Tester compares multiple LoRAs and weight ranges with one prompt and one
fixed seed. It returns the results as one or more labeled, RAM-aware matrix pages.

> **Project status:** v0.2.0-beta.1 public beta. Extreme Run Mode completed a
> 2,743-cell stress test covering 211 LoRAs with 13 weights each. Known limitations
> remain; keep recoverable individual images enabled for important runs and please
> report reproducible issues.

## Welcome

Hello and welcome! I hope LoRA Tester makes it easier to compare large LoRA
collections, explore useful weight ranges, and keep the results understandable in a
single matrix. Feedback, careful testing, and constructive contributions are very
welcome.

## AI-assisted development transparency

LoRA Tester is created and maintained by **vibecodingtoolmaker** with substantial
AI-assisted development support from **OpenAI Codex**. Code, documentation, reviews,
and test ideas have been developed and refined through this collaborative workflow —
also known, with some affection, as *vibecoding*.

The project direction, feature decisions, hands-on Forge testing, final review, and
release responsibility remain with the human maintainer. Codex is a development tool,
not a runtime dependency: the installed extension does not contact OpenAI or send
prompts, images, model information, or other user data to Codex. This project is not
affiliated with or endorsed by OpenAI.

## Features

- Loads the LoRA list automatically on first activation
- Filters LoRAs by non-empty model folders, with optional subfolder inclusion
- Offers manual selection across the complete LoRA directory
- Reads trigger words from neighboring `.civitai.info` and `.json` metadata
- Supports per-LoRA trigger text and weight overrides
- Supports positive and negative weights and inclusive ranges such as `-3:3:0.5`
- Uses the first generated seed for every comparison cell
- Generates a fixed-seed reference image without a tested LoRA by default
- Repeats that reference image at the top of every matrix page
- Offers an explicit Extreme Run Mode for up to 10,000 matrix cells
- Writes completed cells to disk instead of retaining decoded images in RAM
- Builds labeled row strips on disk before composing the final matrix
- Retains the adaptive RAM-protection implementation behind a disabled release switch
- Can unload the checkpoint, text encoder, and VAE before matrix construction
- Keeps lossless individual source images by default
- Offers recovery-safe matrix-only cleanup after successful matrix validation
- Recreates Forge's configured temporary gallery directory if it is missing

## Requirements

- Stable Diffusion WebUI Forge - Neo
- A Forge environment that provides Gradio and Pillow

The extension does not install or download packages by itself. The optional
`psutil`-based RAM watchdog is disabled for the v0.2.0 release.

## Installation

1. Download or clone this repository into Forge Neo's `extensions` directory.
2. Keep the repository folder name as `Forge-Neo-Lora-Tester`.
3. Restart Forge Neo.

The **LoRA Tester** accordion is available in both txt2img and img2img.

## Usage

1. Open and enable **LoRA Tester**. The LoRA inventory loads on first activation.
2. Choose a folder or keep **Manual selection (all folders)**. Selecting a specific
   folder selects all LoRAs in that scope. Manual mode tests only explicitly
   selected LoRAs.
3. Use **Include subfolders** to control whether descendants of the chosen folder
   are included. Click **Refresh LoRA List** after adding or removing model files.
4. Enter one global weight or range.
5. Edit trigger words or weight overrides in the per-LoRA table if needed.
6. Configure the reference image, trigger position, label display, matrix columns,
   and margin. The reference is enabled by default and repeated on every page.
7. Choose an output-retention mode. Keeping individual images is the safe default.
8. Keep model unloading enabled so matrix creation can use the released RAM.
9. Enable **Extreme Run Mode** only for deliberately large jobs above 500 cells.
10. Generate normally. The gallery returns matrix pages rather than individual cells.

Generation is blocked while LoRA Tester is enabled and no LoRA is selected.
Disabling the accordion leaves normal Forge generation unchanged.

## Folder and table behavior

Only folders containing at least one LoRA are offered. Direct files appear before
nested folder groups. **Include subfolders** determines whether a folder selection
also contains its descendants.

The per-LoRA table contains these columns:

- **LoRA (do not edit):** read-only relative model path
- **Trigger words:** comma-separated manual value; blank falls back to metadata
- **Min / single weight:** one dedicated weight, or the start of a range
- **Max:** range end
- **Step:** range increment

Use the weight columns as follows:

- `0.8 | blank | blank` uses one dedicated weight.
- `-1 | 1 | 0.25` creates an inclusive range from -1 to 1.
- Three blank weight cells use the global specification.
- An incomplete combination blocks generation and identifies the affected LoRA in
  the UI and console. Correct the row, then start generation again.

One click opens an editable cell and places the caret at the beginning without
selecting or clearing its contents. Tab, Shift+Tab, Enter, and arrow-key cell
navigation activate the next editable cell. The LoRA identifier column rejects
typing, deletion, Enter, and double-click editing.

## Weight specifications

- Single positive value: `1.0`
- Single negative value: `-0.75`
- Inclusive ascending range: `-3:3:0.5`
- Inclusive descending range: `3:-3:0.5`

The syntax is `start:end:step`; direction is inferred automatically. One
specification is limited to 100 values. A normal run is limited to 500 matrix
cells. **Extreme Run Mode** raises this fixed limit to 10,000 cells and must be
enabled explicitly. Extreme runs can take many hours or days, consume very large
amounts of temporary disk space, and may expose long-duration Forge, driver,
model, extension, or system instability.

## Trigger metadata

The extension reads `trainedWords` from a neighboring `.civitai.info` file:

```json
{
  "trainedWords": ["trigger1", "trigger2"],
  "modelId": 12345,
  "baseModel": "SDXL 1.0"
}
```

Neighboring `.json` files may instead contain `activation text` or
`trigger_words`. Manual non-blank table text overrides metadata. A blank cell
falls back to metadata, including while Gradio is updating the table. Enter
`<none>` to suppress triggers explicitly for one LoRA. Missing metadata is not an
error.

**Trigger Word Position** is relative to the complete LoRA injection: **Start**
places trigger text before the user's prompt, while **End** places it after the
LoRA tag. The batch log reports the resolved trigger text and position.

## Fixed-seed comparison

Forge normally increments a seed across iterations. LoRA Tester rebuilds Forge's
per-iteration random-number generator before every cell so that all LoRA and weight
combinations use the seed of the first generated image. This preserves a meaningful
visual comparison.

## Adaptive RAM protection (disabled in v0.2.0)

The adaptive RAM watchdog is intentionally disabled for the v0.2.0 release while
its cross-system thresholds are evaluated further. Its checkbox and budget fields
are visible but read-only. No RAM-monitor thread is started, and the extension does
not stop between LoRA cells based on physical-memory or Windows-commit thresholds.

The implementation remains in the source behind one release switch so it can be
tested and re-enabled later without reintroducing separate UI and processing paths.

Disk spooling remains active: every completed cell is written to disk immediately.
The configured matrix column count remains the width of a logical row, and every
labeled row is saved separately before the final page is composed. The maximum safe
image-dimension fallback also remains active.

For v0.2.0, the primary memory-saving mechanism is the enabled-by-default model
unload described below. Users should still avoid matrix dimensions that exceed the
practical RAM and image-size limits of their system.

## Model unloading

The optional **Unload checkpoint, text encoder and VAE before building the matrix**
setting calls Forge's complete checkpoint-unload path. It releases sampler and
prompt-conditioning caches, then removes the diffusion engine that owns the model,
text encoder, and VAE. This avoids a VRAM-only eviction that could move weights into
system RAM.

The setting is enabled by default. Forge keeps the selected checkpoint configuration
and reloads it for the next generation, so the next job has the normal model-loading
delay.

## Output and recovery

Only matrix pages are returned to the Forge gallery. Pages are kept below 65,000
pixels on either axis and split automatically when necessary. When the default
reference generation is enabled, its fixed-seed image appears at the top of every
page. **Matrix + individual images**
keeps every final source cell as a lossless PNG with generation metadata. Files are
grouped per run and use names such as:

```text
0001_seed-123456_MyLora (weight 0.8).png
```

With Forge's default paths, files use this layout:

```text
output/Lora_tester/
├── tmp/<run-id>/
├── txt2img/<run-id>/
└── img2img/<run-id>/
```

Custom Forge output paths remain the controlling output boundary. Files are written
atomically below `tmp` and moved without a second encoding into the applicable
txt2img or img2img run folder when individual images are retained.

**CAUTION:** **Matrix only** permanently deletes individual LoRA test images only
after every requested cell has completed and every matrix page has been written and
validated. If generation is interrupted, matrix creation fails, a page is missing or
empty, or deletion fails, source files and a recovery manifest are retained. This also
applies when LoRA Tester can still build a valid partial matrix after an interruption.

If Forge's **Save grids** option is enabled, matrix pages are saved in Forge's normal
grid output directory and format. Otherwise, gallery pages remain in the run's
temporary directory. On matrix failure, available row strips or source cells are
returned as recovery output and the recovery location is logged.

## Troubleshooting

### The LoRA list is empty

- Close and reopen the accordion or click **Refresh LoRA List**.
- Confirm that files are below Forge's configured LoRA directory.
- Supported extensions are `.safetensors`, `.ckpt`, and `.pt`.

### Trigger words are missing

- Confirm **Insert Trigger Words** is enabled.
- Check the batch log for `triggers:` and the final injected prompt.
- Enter comma-separated text in the selected LoRA's table row.

### A weight range is rejected

- Use `start:end:step`, for example `-1:1:0.25`.
- The step must not be zero.
- Keep each range at or below 100 values.
- Per-LoRA settings accept either Min alone, all of Min/Max/Step, or three blank
  fields. Incomplete rows block generation and identify the affected LoRA in the UI
  and console instead of silently using another value.

## Known limitations in v0.2.0-beta.1

- Adaptive RAM protection is visible but intentionally disabled while universal
  thresholds are evaluated. Disk spooling and optional model unloading remain active.
- **Matrix Image Margin** controls spacing between matrix rows. Horizontal cell
  spacing currently applies only when **Draw Legend in Matrix Grid** is enabled.
- When model unloading is enabled, saved matrix grids currently use the Forge grid
  output root instead of Forge's date subfolder layout.
- After correcting an invalid value in an actively edited per-LoRA table cell, the
  first **Generate** click can commit the corrected cell and return focus to the table
  without starting generation. Click **Generate** again; the corrected run then starts
  normally. This is a Gradio table-editing usability issue and does not lose generated
  images or settings.

## Privacy and security

LoRA Tester performs no network requests and executes no downloaded metadata. It
reads model filenames and neighboring JSON metadata from Forge's configured LoRA
directory, then writes images only inside Forge-derived output boundaries. Treat
third-party model files as untrusted input and obtain them from sources you trust.

## License

Copyright (C) 2026 vibecodingtoolmaker.

LoRA Tester is free software licensed under the GNU Affero General Public License,
version 3 only (`AGPL-3.0-only`). See [`LICENSE`](LICENSE) for the complete terms.

## Attribution

This extension is independently implemented for
[Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic),
which is licensed under the GNU Affero General Public License version 3.

It integrates through Forge Neo's public extension and image-processing APIs,
including the script lifecycle, processing objects, grid helpers, and LoRA
extra-network syntax. Its labeled matrix presentation was inspired by Forge Neo's
X/Y/Z Plot user experience. Forge Neo's grid implementation is called through its
API and is not copied into this extension.
