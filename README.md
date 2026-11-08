# LoRA Tester for Forge Neo

LoRA Tester compares multiple LoRAs or SDXL-compatible textual inversion embeddings
and weight ranges with one prompt and one fixed seed. It returns the familiar combined
matrix by default and can optionally add one labeled, safety-bounded grid per tested
item.

> **Project status:** `v0.3.0-beta` public prerelease. Extreme Run Mode completed a
> 2,743-cell stress test covering 211 LoRAs with 13 weights each. Known limitations
> remain; keep recoverable individual images enabled for important runs and please
> report reproducible issues.

> **Beta confidence boundary:** Embedding Test supports dual-encoder SDXL, Pony, and
> Illustrious checkpoints. Live UI/preset validation and a txt2img run through matrix
> persistence have passed; visual prompt semantics, final Gallery delivery, Hi-Res,
> img2img, and the new per-item/comparison layouts still need broader live validation.

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
- Reads LoRA trigger words from neighboring `.json`/`.civitai.info` metadata and
  Embedding triggers from `.json` with a filename fallback
- Supports per-LoRA/per-Embedding trigger text and weight overrides
- Supports positive and negative weights and inclusive ranges such as `-3:3:0.5`
- Uses the first generated seed for every comparison cell
- Generates a fixed-seed reference image without a tested LoRA by default
- Repeats that reference image at the top of every matrix page
- Optionally creates a separate horizontal weight row for every tested LoRA or
  Embedding, split into further one-row pages only when a safety limit requires it
- Keeps the combined matrix mandatory in legacy output mode; with per-item grids it
  becomes an optional additional output
- Offers aligned comparison layouts with every LoRA in one row or one column
- Offers an explicit Extreme Run Mode for up to 10,000 matrix cells
- Organizes setup into four workflow tabs: Selection, Models & Run Size, Grid, and
  Output & Safety
- Shows an always-visible planned-run summary with comparison cells, reference,
  model count, total generations, and the active Standard/Extreme limits
- Writes completed cells to disk instead of retaining decoded images in RAM
- Builds labeled row strips on disk before composing the final matrix
- Can unload the checkpoint, text encoder, and VAE before matrix construction
- Keeps lossless individual source images by default
- Offers recovery-safe matrix-only cleanup after successful matrix validation
- Recreates Forge's configured temporary gallery directory if it is missing
- Uses Forge's selected `xl` UI preset to discover `clip_l` + `clip_g` Safetensors
  embeddings before a checkpoint is loaded, and refreshes the list on preset changes
- Revalidates selected embeddings against both text encoders of the loaded SDXL,
  Pony, or Illustrious checkpoint before generation
- Supports global/per-Embedding weight ranges, Start/End placement, and a per-row
  `0`/`1` choice for positive or negative injection, including matching Hi-Res prompts
- On the development branch, optionally runs exactly one LoRA or the unchanged prompt
  across up to 25 checkpoints normally or 150 in Extreme Run Mode, strictly
  sequentially with a complete unload between model jobs

## Requirements

- Stable Diffusion WebUI Forge - Neo
- A Forge environment that provides Gradio and Pillow

The extension does not install or download packages by itself.

## Installation

1. Download or clone this repository into Forge Neo's `extensions` directory.
2. Keep the repository folder name as `Forge-Neo-Lora-Tester`.
3. Restart Forge Neo.

The **LoRA / Embedding Tester** accordion is available in both txt2img and img2img.

## Usage

1. Open and enable **LoRA / Embedding Tester**. In **1. Selection**, choose LoRA or
   Embedding, then select the folder, tested items, global weights, and optional
   per-item overrides. The relevant inventory loads on first activation.
2. In **2. Models & Run Size**, keep **Current model only** for the established
   workflow or choose a sequential multiple-model mode and its checkpoints. Enable
   **Extreme Run Mode** only above 500 cells per model or 25 checkpoints.
3. In **3. Grid**, configure the repeated reference, legend, per-item grids,
   combined overview, layout, columns, and margin. The reference is enabled by
   default and repeated on every page.
4. In **4. Output & Safety**, choose whether to retain individual images and keep
   the recommended complete model unload enabled for matrix construction.
5. Check the always-visible **Planned run** line below the tabs. It reports the
   comparison/reference cells per model, selected model count, total generations,
   and active Standard or Extreme limits. Invalid or incomplete settings are shown
   there before generation.
6. Generate normally. Per-item pages are returned first; an enabled combined matrix
   follows them.

The comparison-capacity line uses the current Forge width/height, selected-item count,
legend, margin, and reference settings. It reports a safe maximum image count per item
and a symmetric `0.5`-step range such as `-5:5:0.5`. Hi-Res output is checked again
against the actual generated cell size and can split earlier.

Generation is blocked when the selected LoRA/Embedding mode is missing its required
item. Prompt-only multiple-model mode intentionally requires no LoRA. Disabling the
accordion leaves normal Forge generation unchanged.

### Multiple models (`Unreleased`, development branch)

The **2. Models & Run Size** tab is optional. Its default **Current model only** mode
preserves the existing tester behavior. Two additional modes are available:

- **One LoRA across selected models** requires **LoRA** under **What should be
  tested?**, exactly one selected LoRA, one shared weight or weight range, and up to
  25 selected checkpoints. Extreme Run Mode raises the checkpoint limit to 150.
- **Prompt only across selected models** does not require a LoRA. Internally it uses a
  no-op comparison case so the same fixed seed, disk spooling, matrix creation, and
  recovery path are reused without creating or loading an empty LoRA file.

Each selected checkpoint runs as a complete Forge processing job. After its pages are
persisted, LoRA Tester releases processing references and calls Forge's complete
checkpoint unload path before starting the next checkpoint. The originally selected
checkpoint and additional modules are restored in a `finally` path. Completed model
pages survive a later model failure or interruption, and a recovery manifest records
the per-model result. If a full unload fails, the sequence stops before another model
is loaded. After the sequential jobs finish, their labeled per-model pages are combined
into one shared comparison grid. The normal matrix dimension and pixel limits remain
active and split that overview only when a single safe page is impossible.

Model-family preflight is deliberately not part of this first implementation. Forge
therefore remains the authority: an incompatible checkpoint/LoRA combination may fail
during loading or generation and is reported as a failed model job. KREA and MOSS still
require focused live validation before this mode is release-qualified.

### Embedding Test (`v0.3.0-beta`)

1. Choose Forge UI Preset **xl**. This preset covers SDXL, Pony, and Illustrious.
2. Open and enable **LoRA / Embedding Tester**, then choose **Embedding** under
   **What should be tested?** in **1. Selection**.
3. The list is available before the checkpoint is loaded. It scans only Safetensors
   headers for both `clip_l` and `clip_g`; tensor data is not loaded. Click
   **Refresh Compatible Embeddings** after adding files.
4. Select a folder or individual embeddings and enter one global weight or an
   inclusive range. The per-Embedding table shows the discovered trigger words;
   trigger and Min/Max/Step overrides work like LoRA overrides.
5. Choose Start/End placement. In each Embedding row, keep **Negative prompt?** at
   `0` for positive injection or set it to `1` for negative injection. This allows
   positive and negative Embeddings in the same run.
6. Select the SDXL, Pony, or Illustrious checkpoint to test, configure the shared
   reference, matrix, retention, model-unload, and Extreme Run options, then generate
   normally. Forge's loaded dual-CLIP databases perform the final compatibility check.

Embedding weights use Forge's prompt-attention syntax, for example
`(MyEmbedding:0.8)`. Forge registers this technical token from the Embedding filename.
An activation phrase found in the sidecar metadata is inserted in addition to that
token; if no usable metadata exists, the filename without its extension is the
fallback and is not duplicated. The chosen token and trigger are also inserted into
the corresponding Hi-Res positive or negative prompt. SD1.x embeddings and model
families without both SDXL CLIP embedding databases are excluded in this first
implementation. Pre-load discovery intentionally ignores pickle-based `.pt` and
`.bin` files; after model load, Forge's own embedding databases remain authoritative.
No second encoder or model is loaded.

## Folder and table behavior

Only folders containing at least one LoRA are offered. Direct files appear before
nested folder groups. **Include subfolders** determines whether a folder selection
also contains its descendants.

Both settings tables contain the identifier, trigger, and Min/Max/Step weight columns:

- **LoRA/Embedding (do not edit):** read-only relative model path
- **Trigger words:** comma-separated manual value; blank falls back to metadata
- **Min / single weight:** one dedicated weight, or the start of a range
- **Max:** range end
- **Step:** range increment

The per-Embedding table additionally contains **Negative prompt?**: `0` injects that
row into the positive prompt and `1` injects it into the negative prompt. Any other
value blocks generation and identifies the affected Embedding.

Use the weight columns as follows:

- `0.8 | blank | blank` uses one dedicated weight.
- `-1 | 1 | 0.25` creates an inclusive range from -1 to 1.
- Three blank weight cells use the global specification.
- An incomplete combination blocks generation and identifies the affected LoRA in
  the UI and console. Correct the row, then start generation again.

One click opens an editable cell and places the caret at the beginning without
selecting or clearing its contents. Tab, Shift+Tab, Enter, and arrow-key cell
navigation activate the next editable cell. Scrolling the table commits the current
cell state and releases its editor so Gradio does not pull the scroll position back to
that cell. The LoRA identifier column rejects typing, deletion, Enter, and double-click
editing.

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
`trigger_words`. For Embeddings, the regular `.json` is checked; if it is missing or
contains no usable trigger, the Embedding filename without `.safetensors` is used.
The filename still remains the technical Forge textual-
inversion token. Manual non-blank table text overrides metadata. A blank cell falls
back to metadata, including while Gradio is updating the table. Enter `<none>` to
suppress an additional trigger explicitly; for Embeddings this never removes the
mandatory technical token. Missing metadata is not an error.

**Trigger Word Position** is relative to the complete LoRA injection: **Start**
places trigger text before the user's prompt, while **End** places it after the
LoRA tag. The batch log reports the resolved trigger text and position.

## Fixed-seed comparison

Forge normally increments a seed across iterations. LoRA Tester rebuilds Forge's
per-iteration random-number generator before every cell so that all LoRA/weight or
Embedding/weight combinations use the seed of the first generated image. This
preserves a meaningful visual comparison.

## Memory and page safeguards

Every completed cell is written to disk immediately, and large decoded Forge/Pillow
references are released before the next cells accumulate. Labeled row strips are also
saved separately before the final pages are composed. Matrix pages are split before
they exceed 65,000 pixels on either axis or 89,000,000 total pixels.

The former adaptive physical-RAM/Windows-commit watchdog and its UI have been removed.
Its thresholds were hardware-specific and it was already disabled in the public beta.
The enabled-by-default model unload described below remains the primary mechanism for
freeing model RAM before page composition. Users should still avoid jobs whose
practical disk, time, or memory requirements exceed their system.

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

Only matrix pages are returned to the Forge gallery. The combined matrix remains the
default and required output while per-item grids are disabled. Enabling per-item grids
makes the combined matrix optional. Any logical grid that exceeds a safety limit is
delivered as multiple pages. A per-item grid always lays out that item's weights in one
horizontal row and ignores the Standard combined matrix column setting; an overwide
row is split into additional safe one-row pages. Aligned comparison layouts preserve
the LoRA/weight axis and start a new aligned part before it becomes unsafe. Every page stays below 65,000
pixels on either axis and at or below 89,000,000 total pixels so Pillow/Gradio can
reopen it safely. When the default reference generation is enabled, its fixed-seed
image appears at the top of every requested page. **Matrix + individual images** keeps
every final source cell as a lossless PNG with generation metadata.
Files are grouped per run and use names such as:

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

**CAUTION:** **Matrix only** permanently deletes individual comparison images only
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

- For LoRAs, confirm **Insert Trigger Words** is enabled.
- For Embeddings, inspect the trigger cell: a blank cell uses JSON metadata or the
  filename fallback; `<none>` suppresses only an additional metadata trigger.
- Confirm that **Negative prompt?** is `0` or `1` for every selected Embedding.
- Check the batch log for `triggers:` and the final injected prompt.
- Enter comma-separated text in the selected item's table row.

### A weight range is rejected

- Use `start:end:step`, for example `-1:1:0.25`.
- The step must not be zero.
- Keep each range at or below 100 values.
- Per-LoRA settings accept either Min alone, all of Min/Max/Step, or three blank
  fields. Incomplete rows block generation and identify the affected LoRA in the UI
  and console instead of silently using another value.

## Known limitations in `v0.3.0-beta`

- Embedding Test is limited to dual-encoder SDXL-compatible model families (SDXL,
  Pony, and Illustrious). One live txt2img run completed sampling and matrix
  persistence; final Gallery delivery, visual prompt semantics, Hi-Res, img2img, and
  broader model-family validation are still in progress.
- Automated and browser UI checks cover the new per-item grids, optional combined
  overview, aligned row/column layouts, and safe splitting. A complete live-generation
  matrix covering every output/retention combination remains outstanding.
- Preset detection is a capability hint for early UI discovery. The loaded checkpoint
  is always the final authority when generation starts.

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
reads model filenames and neighboring JSON metadata from Forge's configured LoRA and
Embedding directories, then writes images only inside Forge-derived output boundaries.
Treat third-party model files as untrusted input and obtain them from sources you
trust.

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
