# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 vibecodingtoolmaker

"""
LoRA Tester Extension for Forge Neo
Allows testing multiple LoRAs with the same prompt, automatically inserting trigger words.
"""

import gc
import os
import json
import shutil
import threading
import time
import uuid
import weakref
import gradio as gr
from decimal import Decimal, InvalidOperation
from PIL import Image, ImageColor, ImageDraw, PngImagePlugin
from pathlib import Path
from typing import Dict

try:
    import psutil
except ImportError:
    psutil = None

from modules import scripts, shared, images
from modules.ui_components import InputAccordion, FormRow
from modules.processing import StableDiffusionProcessing, Processed, create_infotext

# Import LoRA system
try:
    from extensions_builtin.sd_forge_lora import networks
    LORA_AVAILABLE = True
except ImportError:
    try:
        from modules import extra_networks_lora as networks
        LORA_AVAILABLE = True
    except ImportError:
        LORA_AVAILABLE = False
        print("[LoRA Tester] Warning: LoRA system not found")


GIB = 1024 ** 3
MIB = 1024 ** 2


class RamMonitor:
    """Record short system-RAM peaks without blocking Forge's generation loop."""

    def __init__(self, interval_seconds=0.1):
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._interval_start = 0
        self._interval_minimum = 0
        self._interval_active = False
        self._thread = None

    @staticmethod
    def available():
        if psutil is None:
            return 0
        return int(psutil.virtual_memory().available)

    def start(self):
        if psutil is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="lora-tester-ram", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.wait(self.interval_seconds):
            value = self.available()
            with self._lock:
                if self._interval_active and value:
                    self._interval_minimum = min(self._interval_minimum, value)

    def begin_interval(self):
        value = self.available()
        with self._lock:
            self._interval_start = value
            self._interval_minimum = value
            self._interval_active = bool(value)
        return value

    def end_interval(self, detailed=False):
        value = self.available()
        with self._lock:
            if self._interval_active and value:
                self._interval_minimum = min(self._interval_minimum, value)
            start = self._interval_start
            minimum = self._interval_minimum
            self._interval_active = False

        total_drop = max(0, start - minimum)
        persistent_growth = max(0, start - value) if value else 0
        transient_drop = max(0, value - minimum) if value else 0
        if detailed:
            return {
                'start_bytes': start,
                'minimum_bytes': minimum,
                'end_bytes': value,
                'total_drop_bytes': total_drop,
                'persistent_growth_bytes': persistent_growth,
                'transient_drop_bytes': transient_drop,
            }
        return total_drop

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class LoRaMetadataReader:
    """Reads LoRA metadata from .civitai.info files"""

    @staticmethod
    def get_lora_dir() -> Path:
        """Get the LoRA models directory"""
        lora_path = Path(shared.cmd_opts.lora_dir) if hasattr(shared.cmd_opts, 'lora_dir') and shared.cmd_opts.lora_dir else Path(shared.models_path) / "Lora"
        return lora_path

    @staticmethod
    def find_all_loras() -> Dict[str, Dict]:
        """Find all LoRAs and their metadata"""
        lora_dir = LoRaMetadataReader.get_lora_dir()
        loras = {}

        if not lora_dir.exists():
            return loras

        # Search recursively for .safetensors and .ckpt files
        for ext in ['*.safetensors', '*.ckpt', '*.pt']:
            for lora_file in lora_dir.rglob(ext):
                lora_name = lora_file.stem  # Filename without extension
                relative_path = lora_file.relative_to(lora_dir)

                # Try to read metadata
                metadata = LoRaMetadataReader.read_lora_metadata(lora_file)

                # Get alias from ss_output_name or use filename
                alias = metadata.get('alias', lora_name)

                loras[str(relative_path)] = {
                    'name': lora_name,  # Base filename
                    'alias': alias,  # Display name or ss_output_name
                    'trigger_words': metadata.get('trigger_words', []),
                }

        return loras

    @staticmethod
    def read_lora_metadata(lora_file: Path) -> Dict:
        """Read metadata from safetensors, .civitai.info or .json file"""
        metadata = {
            'alias': None,
            'trigger_words': [],
        }

        # Try to read from safetensors metadata first (for ss_output_name)
        if lora_file.suffix.lower() == '.safetensors':
            try:
                from modules import sd_models
                safetensors_metadata = sd_models.read_metadata_from_safetensors(str(lora_file))

                # Get alias from ss_output_name
                if 'ss_output_name' in safetensors_metadata:
                    metadata['alias'] = safetensors_metadata['ss_output_name']

            except Exception:
                # Silently continue if safetensors reading fails
                pass

        # Try .civitai.info for trigger words and other info
        civitai_info = lora_file.with_suffix('.civitai.info')
        if civitai_info.exists():
            try:
                with open(civitai_info, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Extract trigger words
                    if 'trainedWords' in data and isinstance(data['trainedWords'], list):
                        metadata['trigger_words'] = data['trainedWords']

                    return metadata
            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"[LoRA Tester] Error reading {civitai_info}: {e}")

        # Try .json file
        json_file = lora_file.with_suffix('.json')
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Various possible formats
                    if 'activation text' in data:
                        metadata['trigger_words'] = [data['activation text']]
                    elif 'trigger_words' in data:
                        metadata['trigger_words'] = data['trigger_words']

                    return metadata
            except (json.JSONDecodeError, IOError) as e:
                print(f"[LoRA Tester] Error reading {json_file}: {e}")

        return metadata


class LoRaTesterScript(scripts.Script):
    """Main LoRA Tester script"""

    # Keep the accordion near the bottom of Forge's always-on UI section.
    sorting_priority = 1_000_000
    # Disabled for the v0.1.0 release while the adaptive thresholds are
    # evaluated further. Keep this as the single reactivation switch so the UI,
    # background sampler, and generation-stop logic cannot drift apart.
    RAM_WATCHDOG_ENABLED = False
    PROCESSING_STATE_ATTRIBUTE = "_lora_tester_state"
    MAX_WEIGHTS_PER_LORA = 100
    MAX_TOTAL_CASES = 500
    MAX_MATRIX_DIMENSION = 60_000
    PAGE_SAFETY_FACTOR = 1.35
    LABEL_FONT_DIVISOR = 36
    LABEL_MIN_FONT_SIZE = 14
    LABEL_MAX_LINES = 4
    ALL_LORA_FOLDERS = "__lora_tester_all_folders__"
    ROOT_LORA_FOLDER = "__lora_tester_root_folder__"
    KEEP_INDIVIDUAL_OUTPUT = "Matrix + individual images (Recommended)"
    MATRIX_ONLY_OUTPUT = (
        "Matrix only — permanently delete individual images after successful matrix creation"
    )

    def __init__(self):
        super().__init__()
        self.cached_loras = {}

    def title(self):
        return "LoRA Tester"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @staticmethod
    def _coerce_settings_rows(settings):
        """Normalize Gradio Dataframe values to a list of five-column rows."""
        if settings is None:
            return []
        if hasattr(settings, 'values'):
            settings = settings.values.tolist()
        elif isinstance(settings, dict):
            settings = settings.get('data', [])

        rows = []
        for row in settings or []:
            if not isinstance(row, (list, tuple)) or not row:
                continue
            normalized = list(row[:5])
            normalized.extend([""] * (5 - len(normalized)))
            rows.append(normalized)
        return rows

    @staticmethod
    def _format_weight(weight):
        value = Decimal(str(weight))
        if value == 0:
            return "0"
        return format(value.normalize(), 'f')

    @staticmethod
    def _cell_text(value):
        """Convert an editable Dataframe cell to text, treating empty values as blank."""
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"nan", "none", "<na>"} else text

    @classmethod
    def _resolve_trigger_words(cls, metadata_triggers, settings_value=None):
        """Resolve a per-LoRA trigger cell with metadata as the safe default."""
        trigger_text = cls._cell_text(settings_value)
        if trigger_text.casefold() == "<none>":
            return []
        if trigger_text:
            return [
                trigger.strip()
                for trigger in trigger_text.split(',')
                if trigger.strip()
            ]
        return list(metadata_triggers or [])

    @staticmethod
    def _compose_lora_prompt(
        prompt, lora_tag, trigger_words, use_trigger_words, trigger_position
    ):
        """Place trigger text explicitly before the prompt or after the LoRA tag."""
        prompt_with_lora = f"{str(prompt or '').rstrip()} {lora_tag}".strip()
        if not use_trigger_words or not trigger_words:
            return prompt_with_lora

        trigger_text = ", ".join(str(word).strip() for word in trigger_words if str(word).strip())
        if not trigger_text:
            return prompt_with_lora

        if str(trigger_position or "Start").casefold() == "end":
            return f"{prompt_with_lora}, {trigger_text}"
        return f"{trigger_text}, {prompt_with_lora}"

    @classmethod
    def _parse_weight_spec(cls, spec, description):
        """Parse either one weight or an inclusive start:end:step range."""
        text = "" if spec is None else str(spec).strip()
        if not text:
            return None

        try:
            parts = [Decimal(part.strip()) for part in text.split(':')]
        except (InvalidOperation, ValueError):
            print(f"[LoRA Tester] Invalid {description} weight specification: {text!r}")
            return None

        if not all(value.is_finite() for value in parts):
            print(f"[LoRA Tester] Non-finite {description} weight specification: {text!r}")
            return None

        if len(parts) == 1:
            return [float(parts[0])]
        if len(parts) != 3:
            print(
                f"[LoRA Tester] Invalid {description} weight specification: {text!r}. "
                "Use a number or start:end:step."
            )
            return None

        start, end, step = parts
        if step == 0:
            print(f"[LoRA Tester] Invalid {description} weight range: step must not be zero")
            return None

        direction = Decimal(1) if end >= start else Decimal(-1)
        step = abs(step) * direction
        values = []
        current = start
        in_range = lambda value: value <= end if direction > 0 else value >= end

        while in_range(current) and len(values) < cls.MAX_WEIGHTS_PER_LORA:
            values.append(float(current))
            current += step

        if in_range(current):
            print(
                f"[LoRA Tester] {description} weight range contains more than "
                f"{cls.MAX_WEIGHTS_PER_LORA} values and was rejected"
            )
            return None

        return values

    def _build_lora_settings_rows(self, selected_loras, current_settings=None):
        """Build editable trigger/weight rows while preserving user edits."""
        existing = {
            str(row[0]): row
            for row in self._coerce_settings_rows(current_settings)
            if row[0] not in (None, "")
        }

        rows = []
        for lora_path in selected_loras or []:
            if lora_path in existing:
                rows.append(existing[lora_path])
                continue

            info = self.cached_loras.get(lora_path, {})
            triggers = ", ".join(info.get('trigger_words', []))
            rows.append([lora_path, triggers, "", "", ""])

        return rows

    @staticmethod
    def _relative_lora_folder(lora_path):
        """Return a normalized relative parent folder, or an empty string for root."""
        parent = Path(str(lora_path)).parent
        return "" if parent == Path(".") else str(parent)

    def _lora_folder_choices(self):
        """Build labeled choices for every folder containing LoRAs recursively."""
        direct_counts = {}
        recursive_counts = {}

        for lora_path in self.cached_loras:
            folder = self._relative_lora_folder(lora_path)
            direct_counts[folder] = direct_counts.get(folder, 0) + 1

            parent = Path(folder) if folder else Path(".")
            while parent != Path("."):
                key = str(parent)
                recursive_counts[key] = recursive_counts.get(key, 0) + 1
                parent = parent.parent

        choices = [(
            f"Manual selection (all folders, {len(self.cached_loras)} LoRAs)",
            self.ALL_LORA_FOLDERS,
        )]
        root_count = direct_counts.get("", 0)
        if root_count:
            choices.append((f"Root folder ({root_count})", self.ROOT_LORA_FOLDER))

        for folder in sorted(recursive_counts, key=lambda value: value.casefold()):
            direct = direct_counts.get(folder, 0)
            recursive = recursive_counts[folder]
            if direct == recursive:
                label = f"{folder} ({direct})"
            else:
                label = f"{folder} ({direct} direct / {recursive} incl. subfolders)"
            choices.append((label, folder))

        return choices

    def _filtered_lora_choices(self, folder_filter, include_subfolders):
        """Return LoRA paths belonging to the selected relative folder scope."""
        if folder_filter in (None, "", self.ALL_LORA_FOLDERS):
            return sorted(self.cached_loras, key=str.casefold)

        wanted_folder = "" if folder_filter == self.ROOT_LORA_FOLDER else str(folder_filter)
        wanted_path = Path(wanted_folder) if wanted_folder else Path(".")
        choices = []
        for lora_path in self.cached_loras:
            folder = self._relative_lora_folder(lora_path)
            folder_path = Path(folder) if folder else Path(".")
            matches = folder_path == wanted_path
            if include_subfolders:
                matches = matches or wanted_path in folder_path.parents
            if matches:
                choices.append(lora_path)

        def folder_order(lora_path):
            relative = Path(str(lora_path)).relative_to(wanted_path)
            parts = relative.parts
            # Files directly in the selected folder come first. Descendants are
            # appended afterwards, grouped by their relative subfolder path.
            if len(parts) == 1:
                return (0, "", parts[0].casefold())
            subfolder = str(Path(*parts[:-1])).casefold()
            return (1, subfolder, parts[-1].casefold())

        return sorted(choices, key=folder_order)

    @staticmethod
    def _ensure_temp_directory():
        """Recreate Forge's configured gallery cache directory when missing."""
        temp_dir = getattr(shared.opts, 'temp_dir', "")
        if not temp_dir:
            return

        try:
            os.makedirs(temp_dir, exist_ok=True)
        except OSError as error:
            print(f"[LoRA Tester] Could not create temporary directory {temp_dir!r}: {error}")

    @staticmethod
    def _memory_snapshot():
        if psutil is None:
            return 0, 0
        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)

    @staticmethod
    def _windows_commit_snapshot():
        """Return the current Windows commit limit/headroom, or zeros elsewhere."""
        if os.name != "nt":
            return 0, 0

        try:
            import ctypes
            from ctypes import wintypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return 0, 0
            return int(status.ullTotalPageFile), int(status.ullAvailPageFile)
        except (AttributeError, OSError, ValueError):
            return 0, 0

    @classmethod
    def _resolve_ram_settings(cls, budget_choice, custom_budget_gb, minimum_free_gb):
        actual_total, _ = cls._memory_snapshot()
        configured_total = actual_total

        choice = str(budget_choice or "Automatic")
        if choice == "Custom":
            try:
                configured_total = int(float(custom_budget_gb) * GIB)
            except (TypeError, ValueError):
                configured_total = actual_total
        elif choice != "Automatic":
            try:
                configured_total = int(float(choice.split()[0]) * GIB)
            except (TypeError, ValueError):
                configured_total = actual_total

        if actual_total:
            configured_total = max(GIB, min(configured_total or actual_total, actual_total))
        else:
            configured_total = max(GIB, configured_total or 64 * GIB)

        try:
            minimum_free = int(float(minimum_free_gb) * GIB)
        except (TypeError, ValueError):
            minimum_free = 0
        if minimum_free <= 0:
            minimum_free = max(4 * GIB, int(configured_total * 0.08))

        return {
            'configured_total_bytes': configured_total,
            'minimum_free_bytes': minimum_free,
        }

    @staticmethod
    def _effective_available(state):
        actual_total, actual_available = LoRaTesterScript._memory_snapshot()
        if not actual_available:
            return 0
        configured_total = state.get('configured_total_bytes') or actual_total
        reserved_outside_budget = max(0, actual_total - configured_total)
        return max(0, actual_available - reserved_outside_budget)

    @classmethod
    def _unload_forge_model_for_matrix(cls, p, state):
        """Release Forge's complete diffusion engine before allocating matrix canvases."""

        if not state.get('unload_models_before_matrix'):
            return

        state['model_unload_attempted'] = True
        available_before = cls._effective_available(state)
        print(
            "[LoRA Tester] Unloading checkpoint, text encoder and VAE "
            "before matrix construction"
        )

        try:
            # processing.py normally performs this cleanup only after every script
            # postprocess callback has returned. Do it now so conditioning tensors
            # and sampler references cannot keep model components alive while the
            # matrix canvases are allocated. clear_prompt_cache() is intentional
            # even when Forge's persistent conditioning cache is enabled.
            try:
                p.close()
            finally:
                p.clear_prompt_cache()

            p.c = None
            p.uc = None
            p.sampler = None
            p.rng = None
            p.modified_noise = None
            if hasattr(p, 'hr_c'):
                p.hr_c = None
            if hasattr(p, 'hr_uc'):
                p.hr_uc = None
            for cache_name in ('cached_hr_c', 'cached_hr_uc'):
                if hasattr(type(p), cache_name):
                    setattr(type(p), cache_name, [None, None, None])

            # unload_all_models() alone primarily evicts GPU allocations and may
            # offload weights into system RAM. Forge's public checkpoint-unload
            # path also drops the global diffusion engine, which owns the UNet,
            # text encoder and VAE, and clears forge_hash so the next generation
            # reloads the selected checkpoint normally.
            from modules import devices, sd_models

            sd_models.unload_model_weights()
            if LORA_AVAILABLE and hasattr(networks, 'loaded_networks'):
                networks.loaded_networks.clear()
            gc.collect()
            devices.torch_gc()

            state['model_unloaded'] = True
            available_after = cls._effective_available(state)
            state['available_after_model_unload_bytes'] = available_after
            released = max(0, available_after - available_before)
            print(
                f"[LoRA Tester] Model unload complete: "
                f"{available_after / GIB:.1f} GB effective RAM available "
                f"(+{released / GIB:.1f} GB observed)"
            )
            cls._write_manifest(state, "model-unloaded")
        except Exception as error:
            state['model_unload_error'] = str(error)
            print(
                f"[LoRA Tester] Warning: model unload failed; "
                f"matrix construction will continue with RAM safeguards: {error}"
            )
            cls._write_manifest(state, "model-unload-failed")

    @staticmethod
    def _processing_output_mode(p):
        return "img2img" if hasattr(p, 'init_images') else "txt2img"

    @staticmethod
    def _lora_tester_output_root(p):
        """Follow Forge's configured output paths without embedding a machine path."""

        from modules.paths_internal import default_output_dir

        default_root = Path(default_output_dir)
        common_output = getattr(shared.opts, 'outdir_samples', "")
        if common_output:
            base = Path(common_output)
        else:
            sample_output = Path(p.outpath_samples)
            try:
                sample_output.resolve().relative_to(default_root.resolve())
                base = default_root
            except ValueError:
                # A mode-specific custom directory remains the user's output
                # boundary; keep LoRA Tester files inside it.
                base = sample_output

        return base / "Lora_tester"

    @classmethod
    def _create_spool_directory(cls, p, output_mode):
        output_root = cls._lora_tester_output_root(p)
        spool_root = output_root / "tmp"
        spool_root.mkdir(parents=True, exist_ok=True)
        session_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:10]}"
        session_dir = spool_root / session_name
        session_dir.mkdir()
        individual_output_dir = output_root / output_mode / session_name
        return session_dir, individual_output_dir

    @staticmethod
    def _safe_filename_part(value, limit=96):
        invalid = '<>:"/\\|?*'
        cleaned = ''.join(
            character if character not in invalid and ord(character) >= 32 else '_'
            for character in str(value or "image")
        )
        cleaned = ' '.join(cleaned.split()).strip(' .')
        return (cleaned or "image")[:limit].rstrip(' .')

    @classmethod
    def _finalize_individual_images(cls, state):
        """Move completed cells to their permanent run folder when requested."""

        if not state.get('keep_individual_images'):
            print(
                "[LoRA Tester] Matrix-only mode: individual images remain temporary "
                "until every matrix page has been created successfully"
            )
            return

        destination_dir = Path(state['individual_output_dir'])
        try:
            destination_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            destination_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            state['individual_finalize_error'] = str(error)
            print(
                f"[LoRA Tester] Warning: could not create individual-image output "
                f"directory. Source images are kept in the recovery folder: {error}"
            )
            cls._write_manifest(state, "individual-images-not-finalized")
            return

        moved = 0
        failed = 0
        for cell_index in sorted(state.get('cells', {})):
            cell = state['cells'][cell_index]
            source = Path(cell['path'])
            seed = state.get('fixed_seed')
            seed_text = str(seed) if seed is not None else "unknown"
            label = cls._safe_filename_part(cell.get('label'))
            filename = f"{cell_index + 1:04d}_seed-{seed_text}_{label}.png"
            destination = destination_dir / filename
            try:
                os.replace(source, destination)
            except OSError as error:
                cell['persistent'] = False
                cell['finalize_error'] = str(error)
                failed += 1
                print(
                    f"[LoRA Tester] Warning: could not finalize cell {cell_index + 1}; "
                    f"the temporary source is retained: {error}"
                )
                continue

            cell['path'] = str(destination)
            cell['persistent'] = True
            moved += 1

        state['individual_images_finalized'] = moved
        state['individual_finalize_failures'] = failed
        cls._write_manifest(state, "individual-images-finalized")
        print(
            f"[LoRA Tester] Kept {moved} individual image(s) in {destination_dir}"
        )

    @staticmethod
    def _cell_infotext(p, state, case_index):
        all_prompts = list(getattr(p, 'all_prompts', []) or [])
        effective_prompt = state.get('effective_prompts', {}).get(case_index)
        if effective_prompt is not None and case_index < len(all_prompts):
            all_prompts[case_index] = effective_prompt

        try:
            return create_infotext(
                p,
                all_prompts,
                p.all_seeds,
                p.all_subseeds,
                iteration=case_index,
                position_in_batch=0,
                all_negative_prompts=p.all_negative_prompts,
            )
        except Exception as error:
            label = state['cases'][case_index]['label']
            print(
                f"[LoRA Tester] Warning: full metadata for cell {case_index + 1} "
                f"could not be created: {error}"
            )
            return f"LoRA Tester cell: {label}"

    @staticmethod
    def _validate_matrix_pages(page_paths):
        if not page_paths:
            raise RuntimeError("matrix builder returned no pages")
        for page_path in page_paths:
            path = Path(page_path)
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"matrix page was not saved successfully: {path}")

    @staticmethod
    def _save_png_atomic(image, destination, parameters=None):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        pnginfo = None
        if parameters:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", str(parameters))

        try:
            image.save(
                temporary,
                format="PNG",
                compress_level=1,
                pnginfo=pnginfo,
            )
            os.replace(temporary, destination)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _write_manifest(state, status):
        manifest_path = state.get('manifest_path')
        if not manifest_path:
            return

        cells = [
            state['cells'][index]
            for index in sorted(state.get('cells', {}))
        ]
        payload = {
            'version': 1,
            'status': status,
            'columns': state.get('matrix_cols'),
            'margin': state.get('matrix_margin'),
            'fixed_seed': state.get('fixed_seed'),
            'cells': cells,
            'rows': state.get('rows', []),
            'ram_calibrated': state.get('ram_calibrated', False),
            'calibrated_available_bytes': state.get('calibrated_available_bytes', 0),
            'calibrated_commit_available_bytes': state.get(
                'calibrated_commit_available_bytes', 0
            ),
            'observed_peak_bytes': state.get('observed_peak_bytes', 0),
            'observed_transient_peak_bytes': state.get(
                'observed_transient_peak_bytes', 0
            ),
            'observed_persistent_growth_bytes': state.get(
                'observed_persistent_growth_bytes', 0
            ),
            'last_generation_memory': state.get('last_generation_memory'),
            'ram_stop_reason': state.get('ram_stop_reason'),
            'output_retention': state.get('output_retention'),
            'individual_output_dir': state.get('individual_output_dir'),
            'individual_images_finalized': state.get('individual_images_finalized', 0),
            'individual_finalize_error': state.get('individual_finalize_error'),
            'individual_finalize_failures': state.get('individual_finalize_failures', 0),
            'cleanup_incomplete': state.get('cleanup_incomplete', False),
            'model_unload_attempted': state.get('model_unload_attempted', False),
            'model_unloaded': state.get('model_unloaded', False),
            'model_unload_error': state.get('model_unload_error'),
            'available_after_model_unload_bytes': state.get(
                'available_after_model_unload_bytes',
                0,
            ),
        }
        manifest_path = Path(manifest_path)
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        try:
            with open(temporary, 'w', encoding='utf-8') as manifest_file:
                json.dump(payload, manifest_file, ensure_ascii=False, indent=2)
            os.replace(temporary, manifest_path)
        except OSError as error:
            print(f"[LoRA Tester] Could not update spool manifest: {error}")
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _safe_unlink(path):
        if not path:
            return True
        try:
            Path(path).unlink(missing_ok=True)
            return True
        except OSError as error:
            print(f"[LoRA Tester] Could not remove temporary file {path!r}: {error}")
            return False

    def _move_spool_hooks_last(self, p):
        """Preserve other extensions by running destructive spool callbacks last."""

        runner = getattr(p, 'scripts', None)
        if runner is None or not hasattr(runner, 'ordered_callbacks'):
            return
        for method_name in (
            "process",
            "before_process_batch",
            "postprocess_image_after_composite",
            "postprocess",
        ):
            category = f"script_{method_name}"
            try:
                callbacks = list(runner.ordered_callbacks(method_name))
                own_callback = next(
                    (callback for callback in callbacks if callback.callback is self),
                    None,
                )
                if own_callback is None or callbacks[-1] is own_callback:
                    continue
                callbacks.remove(own_callback)
                callbacks.append(own_callback)
                script_count = len(runner.list_scripts_for_method(method_name))
                runner.callback_map[category] = (script_count, callbacks)
            except Exception as error:
                print(f"[LoRA Tester] Could not move {method_name} last: {error}")

    def ui(self, is_img2img):
        """Create the UI for LoRA Tester"""

        with InputAccordion(False, label="LoRA Tester", elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester") as lora_tester_enabled:

            with FormRow():
                refresh_loras_btn = gr.Button("Refresh LoRA List", variant="secondary", size="sm")

            with FormRow():
                lora_folder = gr.Dropdown(
                    choices=[("Manual selection (all folders)", self.ALL_LORA_FOLDERS)],
                    value=self.ALL_LORA_FOLDERS,
                    label="LoRA Folder",
                    info=(
                        "Selecting a specific folder loads all LoRAs in that scope into "
                        "the selection and per-LoRA table. Manual selection shows all LoRAs "
                        "but only tests those selected explicitly."
                    ),
                    elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_folder"
                )
                include_subfolders = gr.Checkbox(
                    value=True,
                    label="Include subfolders",
                    info="Also include LoRAs stored below the selected folder.",
                    elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_include_subfolders"
                )

            with FormRow():
                lora_selection = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="Select LoRAs to Test",
                    elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_selection",
                    info="Select one or more LoRAs to test with the current prompt"
                )

            with FormRow():
                global_weight_spec = gr.Textbox(
                    value="1.0",
                    label="Global LoRA Weight(s)",
                    placeholder="1.0 or -3:3:0.5",
                    info="Use one value or an inclusive start:end:step range. Per-LoRA overrides are available below.",
                    elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_weights"
                )

            with gr.Accordion("Per-LoRA Settings", open=True):
                with FormRow():
                    use_trigger_words = gr.Checkbox(
                        value=True,
                        label="Insert Trigger Words",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_use_triggers"
                    )
                    trigger_position = gr.Radio(
                        choices=["Start", "End"],
                        value="Start",
                        label="Trigger Word Position",
                        info="Start = before the prompt. End = after the LoRA tag.",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_trigger_pos"
                    )

                gr.HTML(
                    value=(
                        "<p style='color: #888; font-size: 0.9em;'>"
                        "Examples: <code>0.8 | blank | blank</code> = one dedicated weight; "
                        "<code>-1 | 1 | 0.25</code> = range from -1 to 1. "
                        "Leave Min, Max, and Step blank to use the global setting. "
                        "A blank trigger cell uses metadata; enter <code>&lt;none&gt;</code> "
                        "to explicitly disable triggers for one LoRA.</p>"
                    )
                )

                lora_settings = gr.Dataframe(
                    value=[],
                    headers=[
                        "LoRA (do not edit)",
                        "Trigger words (comma-separated)",
                        "Min / single weight",
                        "Max",
                        "Step",
                    ],
                    datatype=["str", "str", "str", "str", "str"],
                    type="array",
                    row_count=(0, "dynamic"),
                    col_count=(5, "fixed"),
                    interactive=True,
                    wrap=True,
                    height=300,
                    column_widths=["30%", "34%", "14%", "11%", "11%"],
                    label="Selected LoRA Settings",
                    elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_settings"
                )

            with gr.Accordion("Advanced Options", open=False):
                with FormRow():
                    save_original = gr.Checkbox(
                        value=False,
                        label="Also generate without any LoRA (baseline)",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_baseline"
                    )

                    draw_legend = gr.Checkbox(
                        value=True,
                        label="Draw legend in matrix grid (LoRA names + trigger words)",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_draw_legend"
                    )

                with FormRow():
                    matrix_cols = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=3,
                        label="Matrix Columns",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_matrix_cols"
                    )

                    matrix_margin = gr.Slider(
                        minimum=0,
                        maximum=50,
                        step=1,
                        value=5,
                        label="Matrix Image Margin (px)",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_matrix_margin"
                    )

                output_retention = gr.Radio(
                    choices=[
                        self.KEEP_INDIVIDUAL_OUTPUT,
                        self.MATRIX_ONLY_OUTPUT,
                    ],
                    value=self.KEEP_INDIVIDUAL_OUTPUT,
                    label="Output retention",
                    info=(
                        "CAUTION: Matrix-only mode permanently deletes all individual "
                        "LoRA test images, but only after every matrix page has been "
                        "created successfully. If matrix creation fails, the individual "
                        "images are kept for recovery."
                    ),
                    elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_output_retention"
                )

                unload_models_before_matrix = gr.Checkbox(
                    value=True,
                    label="Unload checkpoint, text encoder and VAE before building the matrix",
                    info=(
                        "Frees Forge's model RAM/VRAM for larger matrix pages. "
                        "The selected checkpoint is reloaded automatically for the next generation."
                    ),
                    elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_unload_models"
                )

                with gr.Accordion("Adaptive RAM Protection", open=True):
                    gr.HTML(
                        value=(
                            "<p style='color: #d99a35; font-size: 0.9em;'>"
                            "Temporarily disabled for the v0.1.0 release while the RAM thresholds "
                            "are evaluated further. Disk spooling and the checkpoint/text encoder/VAE "
                            "unload before matrix creation remain active.</p>"
                        )
                    )

                    adaptive_ram = gr.Checkbox(
                        value=False,
                        interactive=False,
                        label="Enable adaptive RAM protection (disabled in v0.1.0)",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_adaptive_ram"
                    )

                    with FormRow():
                        ram_budget = gr.Dropdown(
                            choices=["Automatic", "16 GB", "32 GB", "64 GB", "92 GB", "128 GB", "Custom"],
                            value="Automatic",
                            label="RAM Budget",
                            info="Automatic uses detected physical RAM. A smaller budget reserves RAM for other applications.",
                            interactive=False,
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_ram_budget"
                        )

                        custom_ram_gb = gr.Number(
                            value=64,
                            minimum=1,
                            precision=1,
                            label="Custom Budget (GB)",
                            info="Used only when RAM Budget is Custom.",
                            interactive=False,
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_custom_ram"
                        )

                        minimum_free_ram_gb = gr.Number(
                            value=0,
                            minimum=0,
                            precision=1,
                            label="Minimum Free RAM (GB)",
                            info=(
                                "0 = automatic: max(4 GB, 8% of the selected RAM budget). "
                                "On Windows the same minimum is also required as commit headroom."
                            ),
                            interactive=False,
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_min_free_ram"
                        )

            info_text = gr.HTML(
                value=(
                    "<p style='color: #888; font-size: 0.9em;'>"
                    "The extension returns one or more labeled matrix pages. Each LoRA/weight combination creates one matrix cell. "
                    "Trigger words and optional weight overrides can be edited in the table above.</p>"
                )
            )

        # Event handlers
        def update_lora_list(folder_filter, include_children, selected_loras, current_settings):
            """Refresh folders/LoRAs and preserve selections that remain in scope."""
            self.cached_loras = LoRaMetadataReader.find_all_loras()
            folder_choices = self._lora_folder_choices()
            folder_values = {choice[1] for choice in folder_choices}
            if folder_filter not in folder_values:
                folder_filter = self.ALL_LORA_FOLDERS

            choices = self._filtered_lora_choices(folder_filter, include_children)
            allowed_loras = set(choices)
            selected_loras = [
                lora for lora in (selected_loras or []) if lora in allowed_loras
            ]
            rows = self._build_lora_settings_rows(selected_loras, current_settings)
            return (
                gr.update(choices=folder_choices, value=folder_filter),
                gr.update(choices=choices, value=selected_loras),
                rows,
            )

        def load_loras_on_activation(
            enabled, folder_filter, include_children, selected_loras, current_settings
        ):
            """Populate the list the first time the accordion is enabled."""
            if not enabled:
                return gr.update(), gr.update(), current_settings
            if not self.cached_loras:
                return update_lora_list(
                    folder_filter, include_children, selected_loras, current_settings
                )

            folder_choices = self._lora_folder_choices()
            folder_values = {choice[1] for choice in folder_choices}
            if folder_filter not in folder_values:
                folder_filter = self.ALL_LORA_FOLDERS

            choices = self._filtered_lora_choices(folder_filter, include_children)
            allowed_loras = set(choices)
            selected_loras = [
                lora for lora in (selected_loras or []) if lora in allowed_loras
            ]
            rows = self._build_lora_settings_rows(selected_loras, current_settings)
            return (
                gr.update(choices=folder_choices, value=folder_filter),
                gr.update(choices=choices, value=selected_loras),
                rows,
            )

        def apply_folder_filter(
            folder_filter, include_children, selected_loras, current_settings
        ):
            """Load a user-selected folder scope without rescanning model metadata."""
            choices = self._filtered_lora_choices(folder_filter, include_children)
            if folder_filter in (None, "", self.ALL_LORA_FOLDERS):
                allowed_loras = set(choices)
                selected_loras = [
                    lora for lora in (selected_loras or []) if lora in allowed_loras
                ]
            else:
                selected_loras = list(choices)
            rows = self._build_lora_settings_rows(selected_loras, current_settings)
            return gr.update(choices=choices, value=selected_loras), rows

        def update_lora_settings(selected_loras, current_settings):
            return self._build_lora_settings_rows(selected_loras, current_settings)

        # Wire up event handlers
        lora_tester_enabled.change(
            fn=load_loras_on_activation,
            inputs=[
                lora_tester_enabled,
                lora_folder,
                include_subfolders,
                lora_selection,
                lora_settings,
            ],
            outputs=[lora_folder, lora_selection, lora_settings]
        )

        refresh_loras_btn.click(
            fn=update_lora_list,
            inputs=[lora_folder, include_subfolders, lora_selection, lora_settings],
            outputs=[lora_folder, lora_selection, lora_settings]
        )

        lora_folder.input(
            fn=apply_folder_filter,
            inputs=[lora_folder, include_subfolders, lora_selection, lora_settings],
            outputs=[lora_selection, lora_settings]
        )

        include_subfolders.input(
            fn=apply_folder_filter,
            inputs=[lora_folder, include_subfolders, lora_selection, lora_settings],
            outputs=[lora_selection, lora_settings]
        )

        lora_selection.change(
            fn=update_lora_settings,
            inputs=[lora_selection, lora_settings],
            outputs=[lora_settings]
        )

        # Return all components that will be passed to processing callbacks.
        return [
            lora_tester_enabled,
            lora_selection,
            global_weight_spec,
            use_trigger_words,
            trigger_position,
            lora_settings,
            save_original,
            draw_legend,
            matrix_cols,
            matrix_margin,
            output_retention,
            unload_models_before_matrix,
            adaptive_ram,
            ram_budget,
            custom_ram_gb,
            minimum_free_ram_gb
        ]

    def before_process(self, p: StableDiffusionProcessing,
                       lora_tester_enabled, lora_selection, global_weight_spec,
                       use_trigger_words, trigger_position, lora_settings,
                       save_original, draw_legend, matrix_cols, matrix_margin,
                       output_retention,
                       unload_models_before_matrix,
                       adaptive_ram, ram_budget, custom_ram_gb, minimum_free_ram_gb):
        """
        Prepare the current generation before Forge expands prompts and seeds.

        The state is stored on ``p`` instead of the long-lived Script instance. This
        prevents a previously enabled run from leaking into later disabled runs.
        """

        setattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)

        if not lora_tester_enabled:
            return

        if not lora_selection:
            message = (
                "LoRA Tester is enabled, but no LoRA is selected. "
                "Select at least one LoRA or disable LoRA Tester before generating."
            )
            print(f"[LoRA Tester] Generation blocked: {message}")
            try:
                gr.Warning(message)
            except Exception:
                pass
            shared.state.textinfo = message
            shared.state.interrupted = True
            p.do_not_save_grid = True
            return

        self._move_spool_hooks_last(p)
        self._ensure_temp_directory()

        if not self.cached_loras:
            self.cached_loras = LoRaMetadataReader.find_all_loras()

        global_weights = self._parse_weight_spec(global_weight_spec, "global") or [1.0]
        settings_by_lora = {
            str(row[0]): row
            for row in self._coerce_settings_rows(lora_settings)
            if row[0] not in (None, "")
        }

        cases = []
        if save_original:
            cases.append({
                'label': "Baseline (No LoRA)",
                'lora_tag_name': None,
                'trigger_words': [],
                'weight': None,
            })

        for lora_path in lora_selection:
            lora_info = self.cached_loras.get(lora_path)
            if lora_info is None:
                print(f"[LoRA Tester] Warning: LoRA {lora_path} not found in cache")
                continue

            lora_name = lora_info['name']
            lora_alias = lora_info.get('alias')
            display_name = lora_alias or lora_name
            settings_row = settings_by_lora.get(lora_path)
            metadata_triggers = list(lora_info.get('trigger_words', []))

            if settings_row is None:
                trigger_words = self._resolve_trigger_words(metadata_triggers)
                weights = global_weights
            else:
                # Gradio can transiently submit an empty Dataframe cell while
                # folder-selection outputs are settling. Preserve metadata as
                # the default; <none> remains an explicit per-LoRA opt-out.
                trigger_words = self._resolve_trigger_words(
                    metadata_triggers,
                    settings_row[1],
                )

                minimum = self._cell_text(settings_row[2])
                maximum = self._cell_text(settings_row[3])
                step = self._cell_text(settings_row[4])

                if not minimum and not maximum and not step:
                    weights = global_weights
                elif minimum and not maximum and not step:
                    weights = self._parse_weight_spec(minimum, lora_path) or global_weights
                elif minimum and maximum and step:
                    weight_spec = f"{minimum}:{maximum}:{step}"
                    weights = self._parse_weight_spec(weight_spec, lora_path) or global_weights
                else:
                    print(
                        f"[LoRA Tester] Incomplete weight override for {lora_path!r}; "
                        "fill Min, Max, and Step, or only Min for one weight. Using global weights."
                    )
                    weights = global_weights

            for weight in weights:
                if len(cases) >= self.MAX_TOTAL_CASES:
                    print(
                        f"[LoRA Tester] Reached the safety limit of {self.MAX_TOTAL_CASES} "
                        "matrix cells; remaining weights were skipped"
                    )
                    break

                formatted_weight = self._format_weight(weight)
                cases.append({
                    'label': f"{display_name} (weight {formatted_weight})",
                    'lora_tag_name': lora_alias or lora_name,
                    'trigger_words': trigger_words,
                    'weight': formatted_weight,
                })

            if len(cases) >= self.MAX_TOTAL_CASES:
                break

        if not cases:
            return

        output_mode = self._processing_output_mode(p)
        try:
            session_dir, individual_output_dir = self._create_spool_directory(
                p,
                output_mode,
            )
        except OSError as error:
            print(f"[LoRA Tester] Could not create disk spool directory: {error}")
            return

        ram_settings = self._resolve_ram_settings(
            ram_budget,
            custom_ram_gb,
            minimum_free_ram_gb,
        )
        ram_protection_active = bool(
            self.RAM_WATCHDOG_ENABLED and adaptive_ram and psutil is not None
        )
        ram_monitor = RamMonitor() if ram_protection_active else None
        if ram_monitor is not None:
            ram_monitor.start()
            try:
                ram_finalizer = weakref.finalize(p, ram_monitor.stop)
            except TypeError:
                ram_finalizer = None
        else:
            ram_finalizer = None

        calibration_case_index = next(
            index
            for index, case in enumerate(cases)
            if case.get('lora_tag_name') is not None
        )

        state = {
            'cases': cases,
            'use_trigger_words': use_trigger_words,
            'trigger_position': trigger_position,
            'draw_legend': draw_legend,
            'matrix_cols': int(matrix_cols),
            'matrix_margin': int(matrix_margin),
            'output_retention': output_retention,
            'keep_individual_images': output_retention != self.MATRIX_ONLY_OUTPUT,
            'output_mode': output_mode,
            'individual_output_dir': str(individual_output_dir),
            'individual_images_finalized': 0,
            'individual_finalize_failures': 0,
            'effective_prompts': {},
            'unload_models_before_matrix': bool(unload_models_before_matrix),
            'model_unload_attempted': False,
            'model_unloaded': False,
            'adaptive_ram': ram_protection_active,
            'session_dir': str(session_dir),
            'manifest_path': str(session_dir / "manifest.json"),
            'cells': {},
            'rows': [],
            'ram_monitor': ram_monitor,
            'ram_finalizer': ram_finalizer,
            'fixed_seed': None,
            'fixed_subseed': None,
            'observed_peak_bytes': 0,
            'observed_transient_peak_bytes': 0,
            'observed_persistent_growth_bytes': 0,
            'last_generation_memory': None,
            'maximum_row_peak_bytes': 0,
            'ram_calibration_case_index': calibration_case_index,
            'ram_calibrated': False,
            'calibrated_available_bytes': 0,
            'calibrated_commit_available_bytes': 0,
            'ram_stop_reason': None,
            **ram_settings,
        }
        setattr(p, self.PROCESSING_STATE_ATTRIBUTE, state)
        self._write_manifest(state, "generating")

        # This hook runs before setup_prompts(), so Forge creates matching prompt,
        # seed and subseed arrays for every LoRA case.
        p.n_iter = len(cases)
        p.batch_size = 1

        # The extension returns its own matrix and therefore suppresses Forge's
        # automatic result grid and sample files. Final cells are spooled by the
        # extension before Forge stores only tiny placeholders.
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        if hasattr(p, 'return_mask'):
            p.return_mask = False
        if hasattr(p, 'return_mask_composite'):
            p.return_mask_composite = False

        print(f"[LoRA Tester] Enabled with {len(lora_selection)} LoRAs")
        print(f"[LoRA Tester] Will generate {len(cases)} matrix cells total")
        if state['adaptive_ram']:
            print(
                f"[LoRA Tester] Adaptive RAM: budget {state['configured_total_bytes'] / GIB:.1f} GB, "
                f"minimum free {state['minimum_free_bytes'] / GIB:.1f} GB"
            )
            commit_total, commit_available = self._windows_commit_snapshot()
            if commit_available:
                print(
                    f"[LoRA Tester] Windows commit headroom: "
                    f"{commit_available / GIB:.1f} GB available of "
                    f"{commit_total / GIB:.1f} GB total"
                )
        elif not self.RAM_WATCHDOG_ENABLED:
            print(
                "[LoRA Tester] RAM watchdog disabled for v0.1.0; "
                "disk spooling and pre-matrix model unload remain active"
            )
        elif adaptive_ram and psutil is None:
            print("[LoRA Tester] Warning: psutil unavailable; adaptive RAM protection is disabled")

    def process(self, p: StableDiffusionProcessing, *args):
        """Use one fixed seed so LoRA and weight comparisons are meaningful."""
        if not args or not args[0]:
            return

        state = getattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)
        if not state:
            return

        if p.all_seeds:
            state['fixed_seed'] = int(p.all_seeds[0])
            p.all_seeds = [state['fixed_seed']] * len(p.all_prompts)
        if p.all_subseeds:
            state['fixed_subseed'] = int(p.all_subseeds[0])
            p.all_subseeds = [state['fixed_subseed']] * len(p.all_prompts)

        self._write_manifest(state, "seed-fixed")
        if state.get('fixed_seed') is not None:
            print(f"[LoRA Tester] Fixed comparison seed: {state['fixed_seed']}")

    @staticmethod
    def _force_batch_rng_seed(p, state, batch_number, seeds, subseeds):
        """Replace Forge's already-created per-iteration RNG with the first seed."""

        if state.get('fixed_seed') is None:
            source_seeds = getattr(p, 'all_seeds', None) or seeds
            if not source_seeds:
                return
            state['fixed_seed'] = int(source_seeds[0])

        fixed_seed = state['fixed_seed']
        if seeds is None:
            seeds = []
        seeds[:] = [fixed_seed] * len(seeds)
        p.seeds = seeds

        fixed_subseed = state.get('fixed_subseed')
        if fixed_subseed is None:
            source_subseeds = getattr(p, 'all_subseeds', None) or subseeds
            if source_subseeds:
                fixed_subseed = int(source_subseeds[0])
                state['fixed_subseed'] = fixed_subseed
        if subseeds is None:
            subseeds = []
        if fixed_subseed is not None:
            subseeds[:] = [fixed_subseed] * len(subseeds)
        p.subseeds = subseeds

        start = batch_number * int(getattr(p, 'batch_size', 1))
        end = start + len(seeds)
        if getattr(p, 'all_seeds', None) is not None:
            p.all_seeds[start:end] = seeds
        if fixed_subseed is not None and getattr(p, 'all_subseeds', None) is not None:
            p.all_subseeds[start:start + len(subseeds)] = subseeds

        current_rng = getattr(p, 'rng', None)
        if current_rng is None:
            return

        from modules import rng as forge_rng
        p.rng = forge_rng.ImageRNG(
            current_rng.shape,
            seeds,
            subseeds=subseeds,
            subseed_strength=getattr(
                current_rng,
                'subseed_strength',
                getattr(p, 'subseed_strength', 0.0),
            ),
            seed_resize_from_h=getattr(
                current_rng,
                'seed_resize_from_h',
                getattr(p, 'seed_resize_from_h', 0),
            ),
            seed_resize_from_w=getattr(
                current_rng,
                'seed_resize_from_w',
                getattr(p, 'seed_resize_from_w', 0),
            ),
        )

    def before_process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
        """
        Modify prompts before each batch - this is where we inject LoRA tags
        """

        # The first script argument is the InputAccordion's enabled state. Check it
        # as an additional guard even though inactive runs do not receive state.
        if not args or not args[0]:
            return

        state = getattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)
        if not state:
            return

        # Extract batch_number and prompts from kwargs
        batch_number = kwargs.get('batch_number', 0)
        prompts = kwargs.get('prompts', [])
        seeds = kwargs.get('seeds', getattr(p, 'seeds', []))
        subseeds = kwargs.get('subseeds', getattr(p, 'subseeds', []))

        if not prompts:
            return

        if batch_number >= len(state['cases']):
            return

        # Forge constructs p.rng immediately before this hook. Replacing only
        # all_seeds in process() is therefore not sufficient if another script or
        # model path changes the per-iteration lists afterwards. Rebuild the RNG
        # here, before the sampler consumes its first noise tensor.
        self._force_batch_rng_seed(p, state, batch_number, seeds, subseeds)

        calibration_case_index = state.get('ram_calibration_case_index', 0)
        if state.get('adaptive_ram') and batch_number > calibration_case_index:
            effective_available = self._effective_available(state)
            commit_total, commit_available = self._windows_commit_snapshot()
            if not state.get('ram_calibrated'):
                state['ram_calibrated'] = True
                state['calibrated_available_bytes'] = effective_available
                state['calibrated_commit_available_bytes'] = commit_available
                self._write_manifest(state, "ram-calibrated")
                interval = state.get('last_generation_memory') or {}
                commit_text = (
                    f"; Windows commit headroom {commit_available / GIB:.1f} GB "
                    f"of {commit_total / GIB:.1f} GB total"
                    if commit_available
                    else ""
                )
                print(
                    f"[LoRA Tester] RAM after first LoRA generation: "
                    f"{effective_available / GIB:.1f} GB effective available; "
                    f"persistent physical change "
                    f"{interval.get('persistent_growth_bytes', 0) / GIB:.1f} GB; "
                    f"temporary physical drop "
                    f"{state.get('observed_transient_peak_bytes', 0) / GIB:.1f} GB"
                    f"{commit_text}"
                )

            minimum_free = state['minimum_free_bytes']
            shortages = []
            if effective_available and effective_available < minimum_free:
                shortages.append(
                    f"{effective_available / GIB:.1f} GB effective physical RAM available"
                )
            if commit_available and commit_available < minimum_free:
                shortages.append(
                    f"{commit_available / GIB:.1f} GB Windows commit headroom"
                )

            if shortages:
                reason = (
                    f"RAM safety stop before cell {batch_number + 1}: "
                    f"{' and '.join(shortages)}; "
                    f"minimum required {minimum_free / GIB:.1f} GB"
                )
                state['ram_stop_reason'] = reason
                self._write_manifest(state, "ram-safety-stop")
                print(f"[LoRA Tester] {reason}")
                prompts.clear()
                shared.state.interrupted = True
                return

        monitor = state.get('ram_monitor')
        if monitor is not None:
            monitor.begin_interval()

        case = state['cases'][batch_number]
        lora_tag_name = case['lora_tag_name']
        if lora_tag_name is None:
            state['effective_prompts'][batch_number] = prompts[0]
            print(f"[LoRA Tester] Batch {batch_number}: Baseline (no LoRA)")
            return

        # IMPORTANT: Forge Neo LoRA system uses JUST THE FILENAME, not the full path!
        # The system internally scans all directories and indexes by basename
        # So <lora:myLora:1.0> works even if file is in 02_Illustrious/myLora.safetensors
        trigger_words = case['trigger_words']
        trigger_position = str(state.get('trigger_position') or "Start")
        lora_tag = f"<lora:{lora_tag_name}:{case['weight']}>"
        if not state['use_trigger_words']:
            trigger_summary = "disabled globally"
        elif trigger_words:
            trigger_summary = ", ".join(trigger_words)
        else:
            trigger_summary = "none"
        print(
            f"[LoRA Tester] Batch {batch_number}: {case['label']} -> "
            f"Using tag: <lora:{lora_tag_name}:...>; triggers: {trigger_summary}; "
            f"position: {trigger_position}"
        )

        # Start with Forge's current per-batch prompt so styles and other normal
        # prompt processing are preserved.
        for i in range(len(prompts)):
            prompts[i] = self._compose_lora_prompt(
                prompts[i],
                lora_tag,
                trigger_words,
                state['use_trigger_words'],
                trigger_position,
            )

        state['effective_prompts'][batch_number] = prompts[0]
        print(f"[LoRA Tester] Injected prompt: {prompts[0]}")

    def postprocess_image_after_composite(self, p: StableDiffusionProcessing, pp, *args):
        """Spool one final cell to disk, then release Forge's large PIL references."""

        if not args or not args[0]:
            return

        state = getattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)
        if not state:
            return

        case_index = int(getattr(p, 'iteration', len(state.get('cells', {}))))
        if case_index >= len(state['cases']):
            return

        original = pp.image
        case = state['cases'][case_index]
        cell_path = Path(state['session_dir']) / f"cell-{case_index:04d}.png"
        saved = False

        try:
            cell_infotext = self._cell_infotext(p, state, case_index)
            self._save_png_atomic(
                original,
                cell_path,
                parameters=cell_infotext,
            )
            state['cells'][case_index] = {
                'index': case_index,
                'path': str(cell_path),
                'label': case['label'],
                'trigger_words': list(case.get('trigger_words', [])),
                'weight': case.get('weight'),
                'width': int(original.width),
                'height': int(original.height),
                'persistent': False,
            }
            self._write_manifest(state, "generating")
            saved = True
        except Exception as error:
            print(f"[LoRA Tester] Could not spool cell {case_index + 1}: {error}")
        finally:
            monitor = state.get('ram_monitor')
            if monitor is not None:
                interval = monitor.end_interval(detailed=True)
                interval_peak = interval['total_drop_bytes']
                transient_peak = interval['transient_drop_bytes']
                persistent_growth = interval['persistent_growth_bytes']
                state['last_generation_memory'] = interval
                state['observed_peak_bytes'] = max(
                    state.get('observed_peak_bytes', 0),
                    interval_peak,
                )
                state['observed_transient_peak_bytes'] = max(
                    state.get('observed_transient_peak_bytes', 0),
                    transient_peak,
                )
                state['observed_persistent_growth_bytes'] = max(
                    state.get('observed_persistent_growth_bytes', 0),
                    persistent_growth,
                )

        if not saved:
            return

        # processing.py has already appended the original to pixels_after_sampling
        # and will append pp.image to output_images after this hook. Replace both
        # references so the completed cell no longer accumulates in system RAM.
        placeholder = Image.new("RGB", (1, 1), (0, 0, 0))
        if getattr(p, 'pixels_after_sampling', None):
            p.pixels_after_sampling[-1] = placeholder
        pp.image = placeholder
        # Forge otherwise retains one decoded latent per iteration even though
        # LoRA Tester needs only the final disk-spooled pixels.
        if getattr(p, 'latents_after_sampling', None):
            p.latents_after_sampling.clear()

        try:
            original.close()
        except Exception:
            pass

        if (case_index + 1) % max(1, state['matrix_cols']) == 0:
            gc.collect()

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed,
                   lora_tester_enabled, lora_selection, global_weight_spec,
                   use_trigger_words, trigger_position, lora_settings,
                   save_original, draw_legend, matrix_cols, matrix_margin,
                   output_retention,
                   unload_models_before_matrix,
                   adaptive_ram, ram_budget, custom_ram_gb, minimum_free_ram_gb):
        """Build RAM-aware matrix pages from the cells spooled during generation."""

        state = getattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)
        if not lora_tester_enabled or not state:
            return

        monitor = state.get('ram_monitor')
        completed = False
        print(f"[LoRA Tester] Post-processing {len(state.get('cells', {}))} spooled cells")

        try:
            self._spool_fallback_images(p, state, processed)
            cells = [state['cells'][index] for index in sorted(state['cells'])]
            if not cells:
                processed.images = []
                processed.extra_images = []
                if hasattr(p, 'extra_result_images'):
                    p.extra_result_images = []
                processed.infotexts = []
                self._write_manifest(state, "complete-no-cells")
                self._cleanup_spool_files(state, [])
                completed = True
                print("[LoRA Tester] No completed cells are available for a matrix")
                return

            self._finalize_individual_images(state)

            first_image = processed.index_of_first_image
            grid_info = (
                processed.infotexts[first_image]
                if first_image < len(processed.infotexts)
                else "LoRA Tester Matrix Grid"
            )

            self._unload_forge_model_for_matrix(p, state)

            print(
                f"[LoRA Tester] Creating row strips: {len(cells)} cells, "
                f"{state['matrix_cols']} configured columns"
            )
            rows = self._build_row_strips(state, cells)
            page_paths = self._build_matrix_pages(state, rows, p, processed, grid_info)
            self._validate_matrix_pages(page_paths)

            processed.images = [str(path) for path in page_paths]
            processed.extra_images = []
            if hasattr(p, 'extra_result_images'):
                p.extra_result_images = []
            processed.infotexts = [grid_info] * len(page_paths)
            # Every gallery entry is a matrix/grid page; there are no samples
            # after this index for Forge's single-image actions to target.
            processed.index_of_first_image = len(page_paths)
            if state.get('ram_stop_reason'):
                processed.comments += f"\nLoRA Tester: {state['ram_stop_reason']}"

            self._write_manifest(state, "complete")
            self._cleanup_spool_files(state, page_paths)
            completed = True
            print(
                f"[LoRA Tester] Matrix creation complete: {len(page_paths)} page(s), "
                f"{len(cells)} cell(s)"
            )
        except Exception as error:
            self._write_manifest(state, "matrix-error")
            fallback_paths = [
                row['path']
                for row in state.get('rows', [])
                if Path(row['path']).exists()
            ]
            if not fallback_paths:
                fallback_paths = [
                    cell['path']
                    for cell in state.get('cells', {}).values()
                    if cell.get('path') and Path(cell['path']).exists()
                ]
            processed.images = fallback_paths
            processed.infotexts = ["LoRA Tester recovery output"] * len(fallback_paths)
            processed.index_of_first_image = 0
            print(
                f"[LoRA Tester] Matrix creation failed: {error}. "
                f"Temporary recovery files remain in {state['session_dir']}"
            )
        finally:
            if monitor is not None:
                monitor.stop()
            ram_finalizer = state.get('ram_finalizer')
            if ram_finalizer is not None and ram_finalizer.alive:
                ram_finalizer.detach()
            self._ensure_temp_directory()
            setattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)
            if not completed:
                gc.collect()

    def _spool_fallback_images(self, p, state, processed):
        """Persist any cell that could not be captured by the per-image hook."""

        first_image = processed.index_of_first_image
        result_images = processed.images[first_image:first_image + len(state['cases'])]
        for case_index, image in enumerate(result_images):
            if case_index in state['cells']:
                continue
            if not isinstance(image, Image.Image) or image.size == (1, 1):
                continue

            case = state['cases'][case_index]
            cell_path = Path(state['session_dir']) / f"cell-{case_index:04d}.png"
            cell_infotext = self._cell_infotext(p, state, case_index)
            self._save_png_atomic(
                image,
                cell_path,
                parameters=cell_infotext,
            )
            state['cells'][case_index] = {
                'index': case_index,
                'path': str(cell_path),
                'label': case['label'],
                'trigger_words': list(case.get('trigger_words', [])),
                'weight': case.get('weight'),
                'width': int(image.width),
                'height': int(image.height),
                'persistent': False,
            }
            self._write_manifest(state, "generating")

    @staticmethod
    def _cell_annotation(cell):
        label = cell.get('label', '')
        triggers = cell.get('trigger_words', [])
        if not triggers:
            return label
        trigger_text = ", ".join(triggers[:3])
        if len(triggers) > 3:
            trigger_text += "..."
        return f"{label}\n{trigger_text}"

    @staticmethod
    def _text_width(drawing, text, font):
        bbox = drawing.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    @classmethod
    def _compact_label_lines(cls, drawing, text, font, maximum_width):
        """Wrap a label by measured pixels and cap excessive vertical growth."""
        if not str(text).strip():
            return []

        def split_token(token):
            chunks = []
            current = ""
            for character in token:
                candidate = current + character
                if current and cls._text_width(drawing, candidate, font) > maximum_width:
                    chunks.append(current)
                    current = character
                else:
                    current = candidate
            if current:
                chunks.append(current)
            return chunks or [""]

        lines = []
        for paragraph in str(text).splitlines():
            words = paragraph.split()
            if not words:
                continue

            current = ""
            for word in words:
                parts = (
                    split_token(word)
                    if cls._text_width(drawing, word, font) > maximum_width
                    else [word]
                )
                for part in parts:
                    candidate = f"{current} {part}".strip()
                    if current and cls._text_width(drawing, candidate, font) > maximum_width:
                        lines.append(current)
                        current = part
                    else:
                        current = candidate
            if current:
                lines.append(current)

        if len(lines) <= cls.LABEL_MAX_LINES:
            return lines

        lines = lines[:cls.LABEL_MAX_LINES]
        final_line = lines[-1].rstrip()
        while final_line and cls._text_width(drawing, final_line + "...", font) > maximum_width:
            final_line = final_line[:-1].rstrip()
        lines[-1] = (final_line + "...") if final_line else "..."
        return lines

    @classmethod
    def _draw_compact_cell_labels(
        cls, row_grid, row_labels, cell_width, cell_height, margin, background
    ):
        """Add compact centered labels above a single matrix row."""
        columns = len(row_labels)
        font_size = max(
            cls.LABEL_MIN_FONT_SIZE,
            (int(cell_width) + int(cell_height)) // cls.LABEL_FONT_DIVISOR,
        )
        font = images.get_font(font_size)
        calculator = ImageDraw.Draw(row_grid)
        horizontal_padding = max(8, font_size // 2)
        maximum_width = max(1, int(cell_width) - horizontal_padding * 2)
        label_lines = [
            cls._compact_label_lines(calculator, label, font, maximum_width)
            for label in row_labels
        ]

        sample_bbox = calculator.textbbox((0, 0), "Ag", font=font)
        line_height = max(1, sample_bbox[3] - sample_bbox[1])
        line_spacing = max(2, font_size // 5)
        block_heights = [
            len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
            for lines in label_lines
        ]
        vertical_padding = max(6, font_size // 3)
        header_height = max(block_heights, default=0) + vertical_padding * 2

        result = Image.new(
            "RGB",
            (
                row_grid.width + max(0, columns - 1) * int(margin),
                row_grid.height + header_height,
            ),
            background,
        )
        for column_index in range(columns):
            source_x = column_index * int(cell_width)
            target_x = column_index * (int(cell_width) + int(margin))
            cell = row_grid.crop(
                (source_x, 0, source_x + int(cell_width), int(cell_height))
            )
            result.paste(cell, (target_x, header_height))
            cell.close()

        text_color = ImageColor.getcolor(shared.opts.grid_text_active_color, "RGB")
        drawing = ImageDraw.Draw(result)
        for column_index, lines in enumerate(label_lines):
            if not lines:
                continue
            block_height = block_heights[column_index]
            draw_y = (header_height - block_height) // 2
            center_x = (
                column_index * (int(cell_width) + int(margin)) + int(cell_width) / 2
            )
            for line in lines:
                bbox = drawing.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                drawing.text(
                    (center_x - text_width / 2 - bbox[0], draw_y - bbox[1]),
                    line,
                    font=font,
                    fill=text_color,
                )
                draw_y += line_height + line_spacing

        return result

    def _build_row_strips(self, state, cells):
        """Create labeled row files while decoding only one source cell at a time."""

        columns = max(1, min(int(state['matrix_cols']), len(cells)))
        cell_width = max(int(cell['width']) for cell in cells)
        cell_height = max(int(cell['height']) for cell in cells)
        background = ImageColor.getcolor(shared.opts.grid_background_color, "RGB")
        monitor = state.get('ram_monitor')
        rows = []

        for row_index, start in enumerate(range(0, len(cells), columns)):
            group = cells[start:start + columns]
            row_grid = None
            annotated_row = None
            row_record = None
            if monitor is not None:
                monitor.begin_interval()

            try:
                row_grid = Image.new("RGB", (cell_width * columns, cell_height), background)
                row_labels = []
                for column_index in range(columns):
                    if column_index >= len(group):
                        row_labels.append("")
                        continue

                    cell = group[column_index]
                    row_labels.append(self._cell_annotation(cell))
                    with Image.open(cell['path']) as source:
                        source.load()
                        converted = source if source.mode == "RGB" else source.convert("RGB")
                        x = column_index * cell_width + (cell_width - converted.width) // 2
                        y = (cell_height - converted.height) // 2
                        row_grid.paste(converted, (x, y))
                        if converted is not source:
                            converted.close()

                if state['draw_legend']:
                    annotated_row = self._draw_compact_cell_labels(
                        row_grid,
                        row_labels,
                        cell_width,
                        cell_height,
                        state['matrix_margin'],
                        background,
                    )
                else:
                    annotated_row = row_grid

                row_path = Path(state['session_dir']) / f"row-{row_index:04d}.png"
                self._save_png_atomic(annotated_row, row_path)
                row_record = {
                    'index': row_index,
                    'path': str(row_path),
                    'width': int(annotated_row.width),
                    'height': int(annotated_row.height),
                    'cell_indices': [cell['index'] for cell in group],
                    'peak_bytes': 0,
                }
                rows.append(row_record)
            finally:
                row_peak = monitor.end_interval() if monitor is not None else 0
                state['maximum_row_peak_bytes'] = max(
                    state.get('maximum_row_peak_bytes', 0),
                    row_peak,
                )
                if row_record is not None:
                    row_record['peak_bytes'] = row_peak
                if annotated_row is not None and annotated_row is not row_grid:
                    annotated_row.close()
                if row_grid is not None:
                    row_grid.close()

            state['rows'] = rows
            self._write_manifest(state, "row-strips")

        # Keep every source cell until all matrix pages have been written and
        # validated. Matrix-only deletion is deliberately deferred to the final
        # cleanup; a page-composition failure must leave recoverable originals.
        self._write_manifest(state, "row-strips-ready")
        gc.collect()
        return rows

    def _estimate_page_peak(self, state, rows):
        width = max(row['width'] for row in rows)
        height = sum(row['height'] for row in rows)
        height += state['matrix_margin'] * max(0, len(rows) - 1)
        if width > self.MAX_MATRIX_DIMENSION or height > self.MAX_MATRIX_DIMENSION:
            return width, height, float('inf')

        # Four bytes per page pixel is deliberately conservative for an RGB PIL
        # canvas. Add the measured/theoretical working row and PNG encoding room.
        canvas_bytes = width * height * 4
        largest_row_bytes = max(row['width'] * row['height'] * 4 for row in rows)
        working_row = max(
            state.get('maximum_row_peak_bytes', 0),
            largest_row_bytes * 2,
        )
        encoding_reserve = max(64 * MIB, int(canvas_bytes * 0.20))
        estimated_peak = int(canvas_bytes * self.PAGE_SAFETY_FACTOR + working_row + encoding_reserve)
        return width, height, estimated_peak

    def _build_matrix_pages(self, state, rows, p, processed, grid_info):
        page_paths = []
        row_offset = 0

        while row_offset < len(rows):
            effective_available = self._effective_available(state)
            page_budget = max(0, effective_available - state['minimum_free_bytes'])
            selected_rows = []

            for row in rows[row_offset:]:
                candidate = selected_rows + [row]
                _, _, estimated_peak = self._estimate_page_peak(state, candidate)
                dimensions_safe = estimated_peak != float('inf')
                fits_ram = dimensions_safe and (
                    not state['adaptive_ram']
                    or not effective_available
                    or estimated_peak <= page_budget
                )
                if fits_ram:
                    selected_rows = candidate
                    continue
                break

            page_number = len(page_paths) + 1
            if not selected_rows:
                # The already-labeled row file is itself a valid one-row matrix.
                # Returning its path requires no additional decoded-image canvas.
                row = rows[row_offset]
                fallback_path = Path(row['path'])
                if getattr(shared.opts, 'grid_save', False):
                    output_dir = Path(p.outpath_grids)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    session_name = Path(state['session_dir']).name
                    output_path = output_dir / (
                        f"lora_tester_grid-{session_name}-page-{page_number:03d}.png"
                    )
                    shutil.copy2(fallback_path, output_path)
                    fallback_path = output_path
                page_paths.append(fallback_path)
                row_offset += 1
                print(
                    f"[LoRA Tester] Matrix page {page_number}: one-row disk fallback "
                    f"because the calculated RAM reserve would be crossed"
                )
                continue

            width, height, estimated_peak = self._estimate_page_peak(state, selected_rows)
            print(
                f"[LoRA Tester] Matrix page {page_number}: {len(selected_rows)} row(s), "
                f"estimated peak {estimated_peak / MIB:.0f} MB"
            )
            page_path = self._compose_matrix_page(
                state,
                selected_rows,
                width,
                height,
                page_number,
                p,
                processed,
                grid_info,
            )
            page_paths.append(Path(page_path))
            row_offset += len(selected_rows)
            gc.collect()

        return page_paths

    def _compose_matrix_page(self, state, rows, width, height, page_number,
                             p, processed, grid_info):
        background = ImageColor.getcolor(shared.opts.grid_background_color, "RGB")
        page = Image.new("RGB", (width, height), background)
        try:
            y = 0
            for row in rows:
                with Image.open(row['path']) as row_image:
                    row_image.load()
                    converted = row_image if row_image.mode == "RGB" else row_image.convert("RGB")
                    x = (width - converted.width) // 2
                    page.paste(converted, (x, y))
                    y += converted.height + state['matrix_margin']
                    if converted is not row_image:
                        converted.close()

            if getattr(shared.opts, 'grid_save', False):
                seed = processed.all_seeds[0] if processed.all_seeds else -1
                prompt = processed.all_prompts[0] if processed.all_prompts else ""
                save_overrides = {}
                if state.get('model_unloaded'):
                    # Avoid evaluating filename tokens such as [model_name] after
                    # Forge's global model object has intentionally been replaced
                    # with FakeInitialModel. The session id keeps names unique.
                    session_name = Path(state['session_dir']).name
                    save_overrides = {
                        'forced_filename': (
                            f"lora_tester_grid-{session_name}-page-{page_number:03d}"
                        ),
                        'save_to_dirs': False,
                    }
                page_path, _ = images.save_image(
                    page,
                    p.outpath_grids,
                    "lora_tester_grid",
                    seed=seed,
                    prompt=prompt,
                    extension=shared.opts.grid_format,
                    info=grid_info,
                    grid=True,
                    p=p,
                    suffix=f"-page-{page_number:03d}",
                    **save_overrides,
                )
                return page_path

            page_path = Path(state['session_dir']) / f"matrix-page-{page_number:03d}.png"
            parameters = grid_info if getattr(shared.opts, 'enable_pnginfo', False) else None
            self._save_png_atomic(page, page_path, parameters=parameters)
            return str(page_path)
        finally:
            page.close()

    def _cleanup_spool_files(self, state, page_paths):
        protected = {str(Path(path).resolve()) for path in page_paths}
        cleanup_complete = True
        if not state.get('keep_individual_images'):
            deleted = 0
            for cell in state.get('cells', {}).values():
                if self._safe_unlink(cell.get('path')):
                    deleted += 1
                else:
                    cleanup_complete = False
            if cleanup_complete:
                print(
                    f"[LoRA Tester] Matrix-only cleanup: deleted {deleted} individual "
                    "image(s) after matrix validation"
                )
            else:
                print(
                    "[LoRA Tester] Warning: matrix pages are safe, but one or more "
                    "individual images could not be deleted"
                )
        for row in state.get('rows', []):
            row_path = row.get('path')
            if row_path and str(Path(row_path).resolve()) not in protected:
                cleanup_complete = self._safe_unlink(row_path) and cleanup_complete

        if cleanup_complete:
            self._safe_unlink(state.get('manifest_path'))
        else:
            state['cleanup_incomplete'] = True
            self._write_manifest(state, "cleanup-incomplete")

        session_dir = Path(state['session_dir'])
        try:
            session_dir.rmdir()
        except OSError:
            # Unsaved gallery pages intentionally remain in Forge's temp tree.
            pass

