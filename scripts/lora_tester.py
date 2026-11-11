# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 vibecodingtoolmaker

"""
LoRA and Embedding Tester Extension for Forge Neo.

Allows fixed-seed matrix testing of multiple LoRAs or SDXL-compatible textual
inversion embeddings with one prompt.
"""

import copy
import gc
import html
import os
import json
import shutil
import time
import uuid
import gradio as gr
from decimal import Decimal, InvalidOperation
from PIL import Image, ImageColor, ImageDraw, PngImagePlugin
from pathlib import Path
from typing import Dict

from modules import scripts, shared, images
from modules.ui_components import InputAccordion, FormRow
from modules.processing import StableDiffusionProcessing, Processed, create_infotext
from forge_neo_model_family import (
    detect_model_family,
    read_embedding_trigger_words,
    resolve_forge_preset,
    scan_sdxl_dual_clip_embeddings,
)

MIB = 1024 ** 2


class LoRaMetadataReader:
    """Reads LoRA metadata from .civitai.info files"""

    @staticmethod
    def get_lora_dir() -> Path:
        """Return Forge's primary LoRA models directory."""

        configured = getattr(shared.cmd_opts, 'lora_dir', None)
        return (
            Path(configured)
            if configured
            else Path(shared.models_path) / "Lora"
        )

    @staticmethod
    def get_lora_dirs() -> list[Path]:
        """Return Forge's primary and additional LoRA roots without duplicates."""

        configured = [
            LoRaMetadataReader.get_lora_dir(),
            *(
                Path(path)
                for path in (getattr(shared.cmd_opts, 'lora_dirs', []) or [])
                if path
            ),
        ]
        roots = []
        seen = set()
        for root in configured:
            try:
                identity = os.path.normcase(str(root.resolve()))
            except OSError:
                identity = os.path.normcase(str(root.absolute()))
            if identity in seen:
                continue
            seen.add(identity)
            roots.append(root)
        return roots

    @staticmethod
    def find_all_loras() -> Dict[str, Dict]:
        """Find all LoRAs and their metadata"""
        loras = {}
        seen_files = set()

        for root_index, lora_dir in enumerate(
            LoRaMetadataReader.get_lora_dirs()
        ):
            if not lora_dir.exists():
                continue
            root_label = lora_dir.name or f"root-{root_index + 1}"

            # Match Forge's supported LoRA file extensions and recursive roots.
            for ext in ['*.safetensors', '*.ckpt', '*.pt']:
                for lora_file in lora_dir.rglob(ext):
                    try:
                        resolved_file = lora_file.resolve()
                    except OSError:
                        resolved_file = lora_file.absolute()
                    file_identity = os.path.normcase(str(resolved_file))
                    if file_identity in seen_files:
                        continue
                    seen_files.add(file_identity)

                    lora_name = lora_file.stem
                    relative_path = lora_file.relative_to(lora_dir)
                    if root_index == 0:
                        inventory_path = relative_path
                    else:
                        inventory_path = (
                            Path(
                                f"[Additional root {root_index}: {root_label}]"
                            )
                            / relative_path
                        )
                    inventory_key = str(inventory_path)
                    collision_index = 2
                    while inventory_key in loras:
                        inventory_path = (
                            Path(
                                f"[Additional root {root_index}: {root_label} "
                                f"#{collision_index}]"
                            )
                            / relative_path
                        )
                        inventory_key = str(inventory_path)
                        collision_index += 1

                    metadata = LoRaMetadataReader.read_lora_metadata(lora_file)
                    alias = metadata.get('alias', lora_name)
                    loras[inventory_key] = {
                        'name': lora_name,
                        'alias': alias,
                        'trigger_words': metadata.get('trigger_words', []),
                        'source_path': str(resolved_file),
                        'source_root': str(lora_dir),
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


class EmbeddingMetadataReader:
    """Discover early candidates and validate against loaded SDXL text encoders."""

    _disk_inventory_cache = {}

    @staticmethod
    def is_dual_encoder_embedding(embedding):
        """Reject single-tensor concepts that cannot feed both SDXL CLIP models."""

        vectors = getattr(embedding, "vec", None)
        return isinstance(vectors, dict) and {"clip_l", "clip_g"}.issubset(vectors)

    @staticmethod
    def support_error(model) -> str | None:
        if model is None:
            return (
                "No checkpoint is loaded. Load an SDXL, Pony, or Illustrious "
                "checkpoint, then refresh the embedding list."
            )
        if not getattr(model, "is_sdxl", False):
            return (
                "Embedding Test currently supports only SDXL-compatible model "
                "families (SDXL, Pony, and Illustrious)."
            )

        engines = (
            getattr(model, "text_processing_engine_l", None),
            getattr(model, "text_processing_engine_g", None),
        )
        if any(getattr(engine, "embeddings", None) is None for engine in engines):
            return (
                "The loaded checkpoint does not expose both SDXL text-encoder "
                "embedding databases."
            )
        return None

    @classmethod
    def find_compatible_embeddings(cls, refresh=False, model=None):
        """Return the intersection loaded by Forge's existing SDXL CLIP engines."""

        if model is None:
            model = getattr(shared, "sd_model", None)
        error = cls.support_error(model)
        if error:
            return {}, error

        databases = (
            model.text_processing_engine_l.embeddings,
            model.text_processing_engine_g.embeddings,
        )
        if refresh:
            for database in databases:
                database.load_textual_inversion_embeddings()

        compatible_names = set(databases[0].word_embeddings).intersection(
            databases[1].word_embeddings
        )
        embedding_root = Path(shared.cmd_opts.embeddings_dir).resolve()
        embeddings = {}
        for name in sorted(compatible_names, key=str.casefold):
            embedding_l = databases[0].word_embeddings[name]
            embedding_g = databases[1].word_embeddings[name]
            if not all(
                cls.is_dual_encoder_embedding(embedding)
                for embedding in (embedding_l, embedding_g)
            ):
                continue
            filename = getattr(embedding_l, "filename", None) or getattr(
                embedding_g, "filename", None
            )
            if not filename:
                continue

            path = Path(filename).resolve()
            try:
                relative_path = path.relative_to(embedding_root)
            except ValueError:
                # Forge should only expose files from its configured embedding
                # directories. Keep this extension inside that same boundary.
                continue

            embeddings[str(relative_path)] = {
                "name": str(name),
                "vectors": int(getattr(embedding_l, "vectors", 0) or 0),
                "shape": int(getattr(embedding_l, "shape", 0) or 0),
            }
            trigger_words, trigger_source = read_embedding_trigger_words(path)
            embeddings[str(relative_path)].update(
                {
                    "trigger_words": trigger_words,
                    "trigger_source": trigger_source,
                }
            )

        if embeddings:
            status = (
                f"Found {len(embeddings)} embedding(s) compatible with both "
                "text encoders of the loaded SDXL-family checkpoint."
            )
        else:
            status = (
                "No embeddings compatible with both text encoders of the loaded "
                "SDXL-family checkpoint were found."
            )
        return embeddings, status

    @classmethod
    def find_for_preset(cls, preset=None, refresh=False, model=None):
        """Build an early UI inventory from the preset, without loading a model."""

        family = detect_model_family(preset, shared.opts)
        if not family.supports_sdxl_dual_clip_embeddings:
            selected = f"'{family.preset}'" if family.preset else "unknown"
            return {}, (
                "Embedding Test currently supports only the Forge UI Preset 'xl' "
                "(SDXL, Pony, and Illustrious). The selected preset is "
                f"{selected}."
            )

        if model is None:
            model = getattr(shared, "sd_model", None)
        if cls.support_error(model) is None:
            return cls.find_compatible_embeddings(refresh=refresh, model=model)

        embedding_root = Path(shared.cmd_opts.embeddings_dir).resolve()
        cache_key = str(embedding_root)
        if refresh or cache_key not in cls._disk_inventory_cache:
            cls._disk_inventory_cache[cache_key] = scan_sdxl_dual_clip_embeddings(
                embedding_root
            )
        inventory = cls._disk_inventory_cache[cache_key]
        embeddings = dict(inventory.embeddings)

        if embeddings:
            status = (
                f"Found {len(embeddings)} SDXL dual-CLIP embedding(s) from "
                f"Safetensors headers using UI Preset 'xl'. Forge will validate "
                "the selection against the loaded checkpoint when generation starts."
            )
        else:
            status = (
                "UI Preset 'xl' is selected, but no Safetensors embedding containing "
                "both clip_l and clip_g was found."
            )
        if inventory.unreadable_files:
            status += (
                f" {inventory.unreadable_files} Safetensors file(s) could not be read."
            )
        return embeddings, status


class LoRaTesterScript(scripts.Script):
    """Main LoRA Tester script"""

    # Keep the accordion near the bottom of Forge's always-on UI section.
    sorting_priority = 1_000_000
    PROCESSING_STATE_ATTRIBUTE = "_lora_tester_state"
    MAX_WEIGHTS_PER_ITEM = 100
    MAX_TOTAL_CASES = 500
    MAX_EXTREME_TOTAL_CASES = 10_000
    MAX_MATRIX_DIMENSION = 65_000
    # Pillow warns above 89,478,485 pixels by default and Gradio reopens every
    # returned Gallery image through Pillow. Stay slightly below that warning
    # threshold so a valid matrix cannot fail only while Gradio serializes it.
    MAX_MATRIX_PIXELS = min(89_000_000, int(Image.MAX_IMAGE_PIXELS or 89_000_000))
    PAGE_SAFETY_FACTOR = 1.35
    LABEL_FONT_DIVISOR = 36
    LABEL_MIN_FONT_SIZE = 14
    LABEL_MAX_LINES = 4
    ALL_LORA_FOLDERS = "__lora_tester_all_folders__"
    ROOT_LORA_FOLDER = "__lora_tester_root_folder__"
    ALL_EMBEDDING_FOLDERS = "__lora_tester_all_embedding_folders__"
    ROOT_EMBEDDING_FOLDER = "__lora_tester_root_embedding_folder__"
    ALL_CHECKPOINT_FOLDERS = "__lora_tester_all_checkpoint_folders__"
    ROOT_CHECKPOINT_FOLDER = "__lora_tester_root_checkpoint_folder__"
    TEST_TYPE_LORA = "LoRA"
    TEST_TYPE_EMBEDDING = "Embedding"
    TEST_TYPE_PROMPT = "Prompt only (internal)"
    MULTI_MODEL_CURRENT = "Current model only (existing behavior)"
    MULTI_MODEL_LORA = "One LoRA across selected models"
    MULTI_MODEL_PROMPT = "Prompt only across selected models"
    MULTI_OVERVIEW_MODELS_FIRST = "Models first — seed rounds as columns"
    MULTI_OVERVIEW_SEEDS_FIRST = "Seeds first — models grouped by seed round"
    MAX_MULTI_MODEL_CHECKPOINTS = 25
    MAX_EXTREME_MULTI_MODEL_CHECKPOINTS = 150
    PROMPT_ONLY_SENTINEL = "__lora_tester_prompt_only__"
    BATCH_MODEL_LABEL_ATTRIBUTE = "_lora_tester_batch_model_label"
    BATCH_SEED_ROUNDS_ATTRIBUTE = "_lora_tester_batch_seed_rounds"
    MATRIX_LAYOUT_STANDARD = "Standard order"
    MATRIX_LAYOUT_COMPARE_HORIZONTAL = "Compare: LoRAs as rows"
    MATRIX_LAYOUT_COMPARE_VERTICAL = "Compare: LoRAs as columns"
    MAX_COMPARISON_ITEMS = 10
    DEFAULT_COMPARISON_STEP = Decimal("0.5")
    EMBEDDING_TARGET_POSITIVE = "Positive prompt"
    EMBEDDING_TARGET_NEGATIVE = "Negative prompt"
    KEEP_INDIVIDUAL_OUTPUT = "Matrix + individual images (Recommended)"
    MATRIX_ONLY_OUTPUT = (
        "Matrix only — permanently delete individual images after successful matrix creation"
    )

    def __init__(self):
        super().__init__()
        self.cached_loras = {}
        self.cached_embeddings = {}

    def title(self):
        return "LoRA / Embedding Tester"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @classmethod
    def _multi_model_request_error(
        cls,
        mode,
        checkpoint_selection,
        test_type,
        lora_selection,
        extreme_run_mode=False,
    ):
        """Validate only the explicit first-stage multi-model contract."""

        if mode == cls.MULTI_MODEL_CURRENT:
            return None
        if mode not in {cls.MULTI_MODEL_LORA, cls.MULTI_MODEL_PROMPT}:
            return "Unknown multiple-model mode. Select a supported mode and try again."

        checkpoints = list(checkpoint_selection or [])
        if not checkpoints:
            return "Select at least one checkpoint for the multiple-model run."
        checkpoint_limit = (
            cls.MAX_EXTREME_MULTI_MODEL_CHECKPOINTS
            if extreme_run_mode
            else cls.MAX_MULTI_MODEL_CHECKPOINTS
        )
        if len(checkpoints) > checkpoint_limit:
            extreme_hint = (
                ""
                if extreme_run_mode
                else (
                    " Enable Extreme Run Mode for up to "
                    f"{cls.MAX_EXTREME_MULTI_MODEL_CHECKPOINTS} checkpoints."
                )
            )
            return (
                "Multiple-model runs currently support at most "
                f"{checkpoint_limit} checkpoints.{extreme_hint}"
            )

        if mode == cls.MULTI_MODEL_LORA:
            if test_type != cls.TEST_TYPE_LORA:
                return (
                    "One-LoRA/multiple-model mode requires LoRA under "
                    "'What should be tested?'. Switch the selection or choose "
                    "Prompt-only mode."
                )
            selected_loras = list(lora_selection or [])
            if len(selected_loras) != 1:
                return (
                    "One-LoRA/multiple-model mode requires exactly one selected LoRA."
                )
        return None

    @classmethod
    def _multi_model_cells_per_seed(
        cls,
        mode,
        lora_selection,
        global_weight_spec,
        lora_settings,
        include_reference,
    ):
        """Return logical cells per model for one Forge Batch Count step."""

        if mode == cls.MULTI_MODEL_PROMPT:
            return 1, None
        selected_loras = list(lora_selection or [])
        if len(selected_loras) != 1:
            return 0, "Multiple-model LoRA mode requires exactly one selected LoRA."

        global_text = cls._cell_text(global_weight_spec)
        global_weights = (
            cls._parse_weight_spec(global_text, "global", report_errors=False)
            if global_text
            else [1.0]
        )
        if global_weights is None:
            return 0, (
                "The global LoRA weight is invalid. Enter one number or an inclusive "
                "Min:Max:Step range."
            )
        settings = {
            str(row[0]): row
            for row in cls._coerce_settings_rows(lora_settings)
            if row[0] not in (None, "")
        }
        selected = str(selected_loras[0])
        weights, weight_error = cls._resolve_item_weights(
            settings.get(selected),
            2,
            global_weights,
            selected,
            report_errors=False,
        )
        if weight_error:
            return 0, weight_error
        return len(weights) + (1 if include_reference else 0), None

    @staticmethod
    def _batch_seed_rounds(
        base_seed,
        base_subseed,
        batch_count,
        subseed_strength=0.0,
    ):
        """Mirror Forge's Batch Count seed/subseed progression."""

        count = max(1, int(batch_count))
        seed = int(base_seed)
        subseed = int(base_subseed)
        uses_variations = float(subseed_strength or 0.0) != 0.0
        rounds = []
        for index in range(count):
            round_seed = seed if uses_variations else seed + index
            round_subseed = subseed + index
            label = f"Seed {round_seed}"
            if uses_variations:
                label += f" / variation {round_subseed}"
            rounds.append(
                {
                    "index": index,
                    "seed": round_seed,
                    "subseed": round_subseed,
                    "label": label,
                }
            )
        return rounds

    @staticmethod
    def _expand_cases_for_seed_rounds(
        cases,
        seed_rounds,
        include_seed_label=True,
    ):
        """Repeat one logical comparison plan for every requested seed round."""

        rounds = list(seed_rounds or [])
        if not rounds:
            return list(cases or [])
        show_seed = len(rounds) > 1
        expanded = []
        for round_info in rounds:
            for base_index, case in enumerate(cases or []):
                expanded_case = dict(case)
                expanded_case["base_case_index"] = base_index
                expanded_case["seed_round"] = int(round_info["index"])
                expanded_case["seed"] = int(round_info["seed"])
                expanded_case["subseed"] = int(round_info["subseed"])
                if show_seed and include_seed_label:
                    expanded_case["label"] = (
                        f"{case['label']} — {round_info['label']}"
                    )
                expanded.append(expanded_case)
        return expanded

    @classmethod
    def _prepare_multi_model_job_arguments(cls, arguments, mode):
        """Disable the outer runner while preserving the normal inner tester path."""

        job_arguments = list(arguments)
        if len(job_arguments) < 24:
            raise ValueError("Incomplete LoRA Tester script arguments for model job")

        # The multi-model components are deliberately appended to the stable
        # existing callback contract. Disable mode/selection in the nested processing
        # job so that the unchanged single-model tester executes exactly once; the
        # overview preference remains available only to the outer coordinator.
        job_arguments[-3] = cls.MULTI_MODEL_CURRENT
        job_arguments[-2] = []

        if mode == cls.MULTI_MODEL_PROMPT:
            job_arguments[1] = cls.TEST_TYPE_PROMPT
            job_arguments[2] = [cls.PROMPT_ONLY_SENTINEL]
            # A prompt-only job is itself the no-op comparison cell. Generating the
            # ordinary reference as well would create the same image twice.
            job_arguments[11] = False
        return job_arguments

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
            if not any(
                str(value or "").strip().casefold()
                not in {"", "nan", "none", "<na>"}
                for value in normalized
            ):
                continue
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
        """Resolve a per-item trigger cell with metadata as the safe default."""
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

    @classmethod
    def _resolve_embedding_target(cls, settings_value=None):
        """Map a per-Embedding 0/1 marker to its prompt target."""

        target_text = cls._cell_text(settings_value) or "0"
        try:
            target_value = Decimal(target_text)
        except InvalidOperation:
            target_value = None

        if target_value == 0:
            return cls.EMBEDDING_TARGET_POSITIVE, None
        if target_value == 1:
            return cls.EMBEDDING_TARGET_NEGATIVE, None
        return None, (
            "The Embedding prompt target must be 0 for the positive prompt or "
            f"1 for the negative prompt. Invalid value: {target_text!r}."
        )

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

    @staticmethod
    def _compose_embedding_prompt(
        prompt, embedding_name, trigger_words, weight, position
    ):
        """Insert the Forge Embedding token and its optional metadata trigger."""

        prompt = str(prompt or "").strip()
        embedding_name = str(embedding_name or "")
        if not embedding_name.strip():
            return prompt

        weighted_embedding = f"({embedding_name}:{weight})"
        distinct_triggers = []
        for trigger in trigger_words or []:
            raw_trigger = str(trigger)
            if raw_trigger.casefold() == embedding_name.casefold():
                continue
            trigger = raw_trigger.strip()
            if not trigger:
                continue
            if trigger.casefold() in {
                existing.casefold() for existing in distinct_triggers
            }:
                continue
            distinct_triggers.append(trigger)

        if str(position or "Start").casefold() == "end":
            embedding_parts = [weighted_embedding, *distinct_triggers]
        else:
            embedding_parts = [*distinct_triggers, weighted_embedding]
        embedding_text = ", ".join(embedding_parts)
        if not prompt:
            return embedding_text
        if str(position or "Start").casefold() == "end":
            return f"{prompt}, {embedding_text}"
        return f"{embedding_text}, {prompt}"

    @staticmethod
    def _case_is_baseline(case):
        """Recognize new typed cases and recovery data from older runs."""

        kind = case.get("kind")
        if kind is not None:
            return kind == "baseline"
        return (
            case.get("lora_tag_name") is None
            and case.get("embedding_name") is None
        )

    @classmethod
    def _parse_weight_spec(cls, spec, description, report_errors=True):
        """Parse either one weight or an inclusive start:end:step range."""
        text = "" if spec is None else str(spec).strip()
        if not text:
            return None

        try:
            parts = [Decimal(part.strip()) for part in text.split(':')]
        except (InvalidOperation, ValueError):
            if report_errors:
                print(
                    f"[LoRA Tester] Invalid {description} weight specification: {text!r}"
                )
            return None

        if not all(value.is_finite() for value in parts):
            if report_errors:
                print(
                    f"[LoRA Tester] Non-finite {description} weight specification: {text!r}"
                )
            return None

        if len(parts) == 1:
            return [float(parts[0])]
        if len(parts) != 3:
            if report_errors:
                print(
                    f"[LoRA Tester] Invalid {description} weight specification: {text!r}. "
                    "Use a number or start:end:step."
                )
            return None

        start, end, step = parts
        if step == 0:
            if report_errors:
                print(
                    f"[LoRA Tester] Invalid {description} weight range: step must not be zero"
                )
            return None

        direction = Decimal(1) if end >= start else Decimal(-1)
        step = abs(step) * direction
        values = []
        current = start
        in_range = lambda value: value <= end if direction > 0 else value >= end

        while in_range(current) and len(values) < cls.MAX_WEIGHTS_PER_ITEM:
            values.append(float(current))
            current += step

        if in_range(current):
            if report_errors:
                print(
                    f"[LoRA Tester] {description} weight range contains more than "
                    f"{cls.MAX_WEIGHTS_PER_ITEM} values and was rejected"
                )
            return None

        return values

    @classmethod
    def _maximum_total_cases(cls, extreme_run_mode):
        """Return the explicit normal or opt-in extreme run limit."""

        return (
            cls.MAX_EXTREME_TOTAL_CASES
            if extreme_run_mode
            else cls.MAX_TOTAL_CASES
        )

    @staticmethod
    def _resolve_matrix_outputs(per_item_grids, combined_overview):
        """Keep the combined matrix mandatory until per-item grids are requested."""

        per_item_grids = bool(per_item_grids)
        return per_item_grids, (
            bool(combined_overview) if per_item_grids else True
        )

    @classmethod
    def _run_summary_html(
        cls,
        test_type,
        selected_loras,
        global_lora_weights,
        lora_settings,
        selected_embeddings,
        global_embedding_weights,
        embedding_settings,
        include_reference,
        multi_model_mode,
        selected_checkpoints,
        extreme_run_mode,
    ):
        """Describe the planned run without mutating or validating Forge state."""

        mode = multi_model_mode or cls.MULTI_MODEL_CURRENT
        checkpoints = list(selected_checkpoints or [])
        model_count = 1 if mode == cls.MULTI_MODEL_CURRENT else len(checkpoints)
        case_limit = cls._maximum_total_cases(extreme_run_mode)
        model_limit = (
            cls.MAX_EXTREME_MULTI_MODEL_CHECKPOINTS
            if extreme_run_mode
            else cls.MAX_MULTI_MODEL_CHECKPOINTS
        )
        limit_label = "Extreme" if extreme_run_mode else "Standard"

        def summary(message, warning=False):
            classes = "lora-tester-run-summary"
            if warning:
                classes += " lora-tester-run-summary-warning"
            return f"<div class='{classes}'>{message}</div>"

        if mode != cls.MULTI_MODEL_CURRENT and not checkpoints:
            return summary(
                "<b>Planned run:</b> Select at least one checkpoint in "
                "<b>2. Models &amp; Run Size</b>.",
                warning=True,
            )

        if mode == cls.MULTI_MODEL_PROMPT:
            over_limit = model_count > model_limit
            message = (
                f"<b>Planned run:</b> Prompt only &middot; {model_count} model(s) "
                f"&times; Forge Batch Count &middot; {limit_label} limit: "
                f"{model_limit} models. Exact generations are resolved when "
                "generation starts."
            )
            if over_limit:
                message += " &middot; <b>Reduce the model selection</b>"
            return summary(message, warning=over_limit)

        embedding_mode = test_type == cls.TEST_TYPE_EMBEDDING
        selected_items = list(
            (selected_embeddings if embedding_mode else selected_loras) or []
        )
        item_name = "Embedding" if embedding_mode else "LoRA"
        if not selected_items:
            return summary(
                f"<b>Planned run:</b> Select at least one {item_name} in "
                "<b>1. Selection</b>.",
                warning=True,
            )
        if mode == cls.MULTI_MODEL_LORA and (
            embedding_mode or len(selected_items) != 1
        ):
            return summary(
                "<b>Planned run:</b> Multiple-model LoRA mode requires exactly one "
                "selected LoRA.",
                warning=True,
            )

        global_spec = (
            global_embedding_weights if embedding_mode else global_lora_weights
        )
        global_text = cls._cell_text(global_spec)
        global_weights = (
            cls._parse_weight_spec(global_text, "global", report_errors=False)
            if global_text
            else [1.0]
        )
        if global_weights is None:
            return summary(
                "<b>Planned run:</b> Correct the global weight specification in "
                "<b>1. Selection</b>.",
                warning=True,
            )

        if embedding_mode:
            settings = {
                str(row[0]): row
                for row in cls._coerce_embedding_settings_rows(embedding_settings)
                if row[0] not in (None, "")
            }
            weight_offset = 3
        else:
            settings = {
                str(row[0]): row
                for row in cls._coerce_settings_rows(lora_settings)
                if row[0] not in (None, "")
            }
            weight_offset = 2

        comparison_cells = 0
        for item in selected_items:
            weights, weight_error = cls._resolve_item_weights(
                settings.get(str(item)),
                weight_offset,
                global_weights,
                str(item),
                report_errors=False,
            )
            if weight_error:
                return summary(
                    f"<b>Planned run:</b> Correct the incomplete or invalid per-{item_name} "
                    "weight settings in <b>1. Selection</b>.",
                    warning=True,
                )
            comparison_cells += len(weights)

        reference_cells = 1 if include_reference else 0
        cells_per_model = comparison_cells + reference_cells
        total_generations = cells_per_model * model_count
        over_case_limit = cells_per_model > case_limit
        over_model_limit = mode != cls.MULTI_MODEL_CURRENT and model_count > model_limit
        reference_text = (
            " + 1 reference" if reference_cells else " &middot; reference disabled"
        )
        model_text = (
            "current model"
            if mode == cls.MULTI_MODEL_CURRENT
            else f"{model_count} models"
        )
        message = (
            f"<b>Planned run:</b> {len(selected_items)} {item_name}(s) &middot; "
            f"{comparison_cells} comparison cell(s){reference_text} &middot; {model_text} "
            f"= <b>{total_generations} base generation(s)</b> "
            f"&times; Forge Batch Count "
            f"&middot; {limit_label} limits: "
            f"{case_limit:,} cells/model"
        )
        if mode != cls.MULTI_MODEL_CURRENT:
            message += f", {model_limit} models"
        message += ". Exact generations are resolved when generation starts"
        if over_case_limit or over_model_limit:
            message += " &middot; <b>Reduce the run or change Extreme Run Mode</b>"
        return summary(message, warning=over_case_limit or over_model_limit)

    @classmethod
    def _comparison_capacity_visible(
        cls,
        comparison_layout,
        per_item_grids,
        combined_overview,
    ):
        """Show comparison guidance only for an active comparison overview."""

        _, combined_enabled = cls._resolve_matrix_outputs(
            per_item_grids,
            combined_overview,
        )
        return combined_enabled and comparison_layout in {
            cls.MATRIX_LAYOUT_COMPARE_HORIZONTAL,
            cls.MATRIX_LAYOUT_COMPARE_VERTICAL,
        }

    @classmethod
    def _estimated_label_header_height(cls, cell_width, cell_height):
        """Conservatively mirror the compact legend's maximum four-line header."""

        font_size = max(
            cls.LABEL_MIN_FONT_SIZE,
            (int(cell_width) + int(cell_height)) // cls.LABEL_FONT_DIVISOR,
        )
        line_spacing = max(2, font_size // 5)
        vertical_padding = max(6, font_size // 3)
        return (
            cls.LABEL_MAX_LINES * font_size
            + (cls.LABEL_MAX_LINES - 1) * line_spacing
            + vertical_padding * 2
        )

    @classmethod
    def _comparison_capacity(
        cls,
        cell_width,
        cell_height,
        item_count,
        orientation,
        margin=5,
        draw_legend=True,
        include_reference=True,
        reference_width=None,
        reference_height=None,
    ):
        """Return the maximum equally aligned weight slots on one safe page."""

        try:
            cell_width = max(1, int(cell_width))
            cell_height = max(1, int(cell_height))
            item_count = int(item_count)
            margin = max(0, int(margin))
        except (TypeError, ValueError, OverflowError):
            return 0
        if item_count < 1 or item_count > cls.MAX_COMPARISON_ITEMS:
            return 0

        label_height = (
            cls._estimated_label_header_height(cell_width, cell_height)
            if draw_legend
            else 0
        )
        labeled_height = cell_height + label_height
        horizontal_gap = margin if draw_legend else 0
        if include_reference:
            reference_width = max(1, int(reference_width or cell_width))
            reference_height = max(1, int(reference_height or labeled_height))
        else:
            reference_width = 0
            reference_height = 0

        capacity = 0
        for slot_count in range(1, cls.MAX_WEIGHTS_PER_ITEM + 1):
            if orientation == cls.MATRIX_LAYOUT_COMPARE_HORIZONTAL:
                content_width = (
                    slot_count * cell_width
                    + max(0, slot_count - 1) * horizontal_gap
                )
                content_height = (
                    item_count * labeled_height
                    + max(0, item_count - 1) * margin
                )
            elif orientation == cls.MATRIX_LAYOUT_COMPARE_VERTICAL:
                content_width = (
                    item_count * cell_width
                    + max(0, item_count - 1) * horizontal_gap
                )
                content_height = (
                    slot_count * labeled_height
                    + max(0, slot_count - 1) * margin
                )
            else:
                return 0

            page_width = max(content_width, reference_width)
            page_height = content_height
            if include_reference:
                page_height += reference_height + margin
            if (
                page_width > cls.MAX_MATRIX_DIMENSION
                or page_height > cls.MAX_MATRIX_DIMENSION
                or page_width * page_height > cls.MAX_MATRIX_PIXELS
            ):
                break
            capacity = slot_count
        return capacity

    @classmethod
    def _comparison_range_hint(cls, capacity):
        """Describe the largest symmetric 0.5-step range fitting one page."""

        capacity = max(0, int(capacity or 0))
        symmetric_count = capacity if capacity % 2 else capacity - 1
        if symmetric_count < 1:
            return "none"
        if symmetric_count == 1:
            return "0"
        endpoint = cls.DEFAULT_COMPARISON_STEP * ((symmetric_count - 1) // 2)
        endpoint_text = cls._format_weight(endpoint)
        return f"-{endpoint_text}:{endpoint_text}:{cls._format_weight(cls.DEFAULT_COMPARISON_STEP)}"

    @classmethod
    def _comparison_capacity_html(
        cls,
        test_type,
        selected_loras,
        selected_embeddings,
        width,
        height,
        margin,
        draw_legend,
        include_reference,
    ):
        selected = (
            selected_embeddings
            if test_type == cls.TEST_TYPE_EMBEDDING
            else selected_loras
        ) or []
        item_count = len(selected)
        try:
            width = max(1, int(float(width)))
            height = max(1, int(float(height)))
        except (TypeError, ValueError, OverflowError):
            width, height = 0, 0

        if not item_count:
            detail = "Select at least one LoRA or Embedding to calculate capacity."
        elif item_count > cls.MAX_COMPARISON_ITEMS:
            detail = (
                f"Comparison layouts support at most {cls.MAX_COMPARISON_ITEMS} "
                f"items; {item_count} are currently selected."
            )
        elif not width or not height:
            detail = "The current Forge width and height are not available yet."
        else:
            horizontal = cls._comparison_capacity(
                width,
                height,
                item_count,
                cls.MATRIX_LAYOUT_COMPARE_HORIZONTAL,
                margin=margin,
                draw_legend=draw_legend,
                include_reference=include_reference,
            )
            vertical = cls._comparison_capacity(
                width,
                height,
                item_count,
                cls.MATRIX_LAYOUT_COMPARE_VERTICAL,
                margin=margin,
                draw_legend=draw_legend,
                include_reference=include_reference,
            )
            detail = (
                f"<b>Rows:</b> safe max {horizontal} image(s) per item/page; "
                f"symmetric 0.5 range: <code>{html.escape(cls._comparison_range_hint(horizontal))}</code>"
                " &nbsp; | &nbsp; "
                f"<b>Columns:</b> safe max {vertical} image(s) per item/page; "
                f"symmetric 0.5 range: <code>{html.escape(cls._comparison_range_hint(vertical))}</code>"
            )
        return (
            "<p class='lora-tester-comparison-capacity'>"
            f"{detail}</p>"
        )

    @classmethod
    def _resolve_item_weights(
        cls,
        settings_row,
        weight_offset,
        global_weights,
        item,
        report_errors=True,
    ):
        """Resolve one optional Min/Max/Step override without silent fallback."""

        if settings_row is None:
            return global_weights, None

        minimum = cls._cell_text(settings_row[weight_offset])
        maximum = cls._cell_text(settings_row[weight_offset + 1])
        step = cls._cell_text(settings_row[weight_offset + 2])
        if not minimum and not maximum and not step:
            return global_weights, None
        if minimum and not maximum and not step:
            weights = cls._parse_weight_spec(
                minimum,
                item,
                report_errors=report_errors,
            )
            if weights is None:
                return None, (
                    f"The individual weight for '{item}' is invalid. Enter one "
                    "number in Min / single weight, or leave all three weight "
                    "fields blank to use the global value."
                )
            return weights, None
        if minimum and maximum and step:
            weights = cls._parse_weight_spec(
                f"{minimum}:{maximum}:{step}",
                item,
                report_errors=report_errors,
            )
            if weights is None:
                return None, (
                    f"The individual weight range for '{item}' is invalid. Use "
                    "finite numbers and a non-zero Step value."
                )
            return weights, None
        return None, (
            f"The individual weight settings for '{item}' are incomplete. Enter "
            "only Min / single weight for one value, fill Min, Max, and Step for "
            "a range, or leave all three fields blank to use the global value."
        )

    @staticmethod
    def _block_generation(p, message):
        """Stop a LoRA Tester run while keeping the reason visible to the user."""

        print(f"[LoRA Tester] Generation blocked: {message}")
        try:
            gr.Warning(message)
        except Exception:
            pass
        shared.state.textinfo = message
        shared.state.interrupted = True
        p.do_not_save_grid = True

    @staticmethod
    def _matrix_only_cleanup_decision(state, completed_count, interrupted):
        """Allow destructive cleanup only after every requested cell completed."""

        expected_count = len(state.get('cases', []))
        if completed_count != expected_count:
            return False, (
                f"generation completed {completed_count} of {expected_count} requested cells"
            )
        if interrupted:
            return False, "generation was interrupted"
        return True, None

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
    def _coerce_embedding_settings_rows(settings):
        """Normalize Gradio Embedding Dataframe values to six-column rows."""

        if settings is None:
            return []
        if hasattr(settings, "values"):
            settings = settings.values.tolist()
        elif isinstance(settings, dict):
            settings = settings.get("data", [])

        rows = []
        for row in settings or []:
            if not isinstance(row, (list, tuple)) or not row:
                continue
            if len(row) == 4:
                # Migrate rows from the initial Embedding UI, which had no
                # editable trigger column: path, min, max, step.
                normalized = [row[0], "", "0", *row[1:4]]
            elif len(row) == 5:
                # Migrate the trigger-aware layout that predated per-item prompt
                # targets: path, trigger, min, max, step.
                normalized = [row[0], row[1], "0", *row[2:5]]
            else:
                normalized = list(row[:6])
                normalized.extend([""] * (6 - len(normalized)))
            if not any(
                str(value or "").strip().casefold()
                not in {"", "nan", "none", "<na>"}
                for value in normalized
            ):
                continue
            rows.append(normalized)
        return rows

    def _build_embedding_settings_rows(
        self, selected_embeddings, current_settings=None
    ):
        """Build editable trigger/weight rows while preserving user edits."""

        existing = {
            str(row[0]): row
            for row in self._coerce_embedding_settings_rows(current_settings)
            if row[0] not in (None, "")
        }
        rows = []
        for embedding_path in selected_embeddings or []:
            if embedding_path in existing:
                rows.append(existing[embedding_path])
                continue
            info = self.cached_embeddings.get(embedding_path, {})
            triggers = ", ".join(info.get("trigger_words", []))
            rows.append([embedding_path, triggers, "0", "", "", ""])
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
    def _relative_embedding_folder(embedding_path):
        """Return a normalized relative embedding folder, or root."""

        parent = Path(str(embedding_path)).parent
        return "" if parent == Path(".") else str(parent)

    def _embedding_folder_choices(self):
        """Build labeled choices for folders containing compatible embeddings."""

        direct_counts = {}
        recursive_counts = {}
        for embedding_path in self.cached_embeddings:
            folder = self._relative_embedding_folder(embedding_path)
            direct_counts[folder] = direct_counts.get(folder, 0) + 1

            parent = Path(folder) if folder else Path(".")
            while parent != Path("."):
                key = str(parent)
                recursive_counts[key] = recursive_counts.get(key, 0) + 1
                parent = parent.parent

        choices = [(
            f"Manual selection (all folders, {len(self.cached_embeddings)} embeddings)",
            self.ALL_EMBEDDING_FOLDERS,
        )]
        root_count = direct_counts.get("", 0)
        if root_count:
            choices.append((f"Root folder ({root_count})", self.ROOT_EMBEDDING_FOLDER))

        for folder in sorted(recursive_counts, key=lambda value: value.casefold()):
            direct = direct_counts.get(folder, 0)
            recursive = recursive_counts[folder]
            if direct == recursive:
                label = f"{folder} ({direct})"
            else:
                label = f"{folder} ({direct} direct / {recursive} incl. subfolders)"
            choices.append((label, folder))
        return choices

    def _filtered_embedding_choices(self, folder_filter, include_subfolders):
        """Return embeddings belonging to the selected relative folder scope."""

        if folder_filter in (None, "", self.ALL_EMBEDDING_FOLDERS):
            return sorted(self.cached_embeddings, key=str.casefold)

        wanted_folder = (
            "" if folder_filter == self.ROOT_EMBEDDING_FOLDER else str(folder_filter)
        )
        wanted_path = Path(wanted_folder) if wanted_folder else Path(".")
        choices = []
        for embedding_path in self.cached_embeddings:
            folder = self._relative_embedding_folder(embedding_path)
            folder_path = Path(folder) if folder else Path(".")
            matches = folder_path == wanted_path
            if include_subfolders:
                matches = matches or wanted_path in folder_path.parents
            if matches:
                choices.append(embedding_path)

        def folder_order(embedding_path):
            relative = Path(str(embedding_path)).relative_to(wanted_path)
            parts = relative.parts
            if len(parts) == 1:
                return (0, "", parts[0].casefold())
            subfolder = str(Path(*parts[:-1])).casefold()
            return (1, subfolder, parts[-1].casefold())

        return sorted(choices, key=folder_order)

    @staticmethod
    def _checkpoint_choices(refresh=False):
        """Return stable Forge checkpoint aliases without loading any model."""

        from modules import sd_models

        if refresh:
            sd_models.list_models()
        checkpoint_infos = sorted(
            sd_models.checkpoints_list.values(),
            key=lambda info: str(info.title).casefold(),
        )
        return [(info.title, info.name) for info in checkpoint_infos]

    @staticmethod
    def _relative_checkpoint_folder(checkpoint_name):
        """Return the relative Forge checkpoint folder, or root."""

        parent = Path(str(checkpoint_name)).parent
        return "" if parent == Path(".") else str(parent)

    @classmethod
    def _checkpoint_folder_choices(cls, checkpoint_choices=None):
        """Build labeled choices for folders containing Forge checkpoints."""

        checkpoint_choices = (
            cls._checkpoint_choices()
            if checkpoint_choices is None
            else list(checkpoint_choices)
        )
        direct_counts = {}
        recursive_counts = {}
        for _, checkpoint_name in checkpoint_choices:
            folder = cls._relative_checkpoint_folder(checkpoint_name)
            direct_counts[folder] = direct_counts.get(folder, 0) + 1

            parent = Path(folder) if folder else Path(".")
            while parent != Path("."):
                key = str(parent)
                recursive_counts[key] = recursive_counts.get(key, 0) + 1
                parent = parent.parent

        choices = [(
            f"Manual selection (all folders, {len(checkpoint_choices)} checkpoints)",
            cls.ALL_CHECKPOINT_FOLDERS,
        )]
        root_count = direct_counts.get("", 0)
        if root_count:
            choices.append((f"Root folder ({root_count})", cls.ROOT_CHECKPOINT_FOLDER))

        for folder in sorted(recursive_counts, key=lambda value: value.casefold()):
            direct = direct_counts.get(folder, 0)
            recursive = recursive_counts[folder]
            if direct == recursive:
                label = f"{folder} ({direct})"
            else:
                label = (
                    f"{folder} ({direct} direct / {recursive} incl. subfolders)"
                )
            choices.append((label, folder))
        return choices

    @classmethod
    def _filtered_checkpoint_choices(
        cls,
        folder_filter,
        include_subfolders,
        checkpoint_choices=None,
    ):
        """Return Forge checkpoints belonging to the selected folder scope."""

        checkpoint_choices = (
            cls._checkpoint_choices()
            if checkpoint_choices is None
            else list(checkpoint_choices)
        )
        if folder_filter in (None, "", cls.ALL_CHECKPOINT_FOLDERS):
            return checkpoint_choices

        wanted_folder = (
            "" if folder_filter == cls.ROOT_CHECKPOINT_FOLDER else str(folder_filter)
        )
        wanted_path = Path(wanted_folder) if wanted_folder else Path(".")
        choices = []
        for label, checkpoint_name in checkpoint_choices:
            folder = cls._relative_checkpoint_folder(checkpoint_name)
            folder_path = Path(folder) if folder else Path(".")
            matches = folder_path == wanted_path
            if include_subfolders:
                matches = matches or wanted_path in folder_path.parents
            if matches:
                choices.append((label, checkpoint_name))

        def folder_order(choice):
            relative = Path(str(choice[1])).relative_to(wanted_path)
            parts = relative.parts
            if len(parts) == 1:
                return (0, "", parts[0].casefold())
            subfolder = str(Path(*parts[:-1])).casefold()
            return (1, subfolder, parts[-1].casefold())

        return sorted(choices, key=folder_order)

    @staticmethod
    def _resolve_checkpoint_jobs(checkpoint_selection):
        """Resolve UI aliases to Forge-owned checkpoint records at run time."""

        from modules import sd_models

        jobs = []
        missing = []
        for selected in checkpoint_selection or []:
            checkpoint_info = sd_models.get_closet_checkpoint_match(str(selected))
            if checkpoint_info is None:
                missing.append(str(selected))
                continue
            jobs.append(
                {
                    "checkpoint": checkpoint_info.name,
                    "title": checkpoint_info.title,
                    "label": checkpoint_info.name_for_extra,
                }
            )
        return jobs, missing

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
    def _release_processing_references(p):
        """Drop per-job sampler and conditioning references before model unload."""

        if p is None:
            return
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

    @staticmethod
    def _clear_loaded_lora_registry():
        """Resolve Forge's LoRA registry lazily, only when cleanup needs it."""

        try:
            from extensions_builtin.sd_forge_lora import networks
        except ImportError:
            try:
                from modules import extra_networks_lora as networks
            except ImportError:
                return

        loaded_networks = getattr(networks, 'loaded_networks', None)
        if loaded_networks is not None:
            loaded_networks.clear()

    @classmethod
    def _full_forge_model_unload(cls, p=None):
        """Use Forge's complete checkpoint unload, not a VRAM-only eviction."""

        cls._release_processing_references(p)
        from modules import devices, sd_models

        sd_models.unload_model_weights()
        cls._clear_loaded_lora_registry()
        gc.collect()
        devices.torch_gc()

    @classmethod
    def _unload_forge_model_for_matrix(cls, p, state):
        """Release Forge's complete diffusion engine before allocating matrix canvases."""

        if not state.get('unload_models_before_matrix'):
            return

        state['model_unload_attempted'] = True
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
            # unload_all_models() alone primarily evicts GPU allocations and may
            # offload weights into system RAM. Forge's public checkpoint-unload
            # path also drops the global diffusion engine, which owns the UNet,
            # text encoder and VAE, and clears forge_hash so the next generation
            # reloads the selected checkpoint normally.
            cls._full_forge_model_unload(p)

            state['model_unloaded'] = True
            print("[LoRA Tester] Model unload complete")
            cls._write_manifest(state, "model-unloaded")
        except Exception as error:
            state['model_unload_error'] = str(error)
            print(
                f"[LoRA Tester] Warning: model unload failed; "
                f"matrix construction will continue with static page limits: {error}"
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
            seed = cell.get('seed', state.get('fixed_seed'))
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

        all_negative_prompts = list(
            getattr(p, 'all_negative_prompts', []) or []
        )
        effective_negative_prompt = state.get(
            'effective_negative_prompts', {}
        ).get(case_index)
        if (
            effective_negative_prompt is not None
            and case_index < len(all_negative_prompts)
        ):
            all_negative_prompts[case_index] = effective_negative_prompt

        try:
            return create_infotext(
                p,
                all_prompts,
                p.all_seeds,
                p.all_subseeds,
                iteration=case_index,
                position_in_batch=0,
                all_negative_prompts=all_negative_prompts,
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
            'output_retention': state.get('output_retention'),
            'create_per_item_grids': state.get('create_per_item_grids', False),
            'create_combined_overview': state.get('create_combined_overview', True),
            'comparison_layout': state.get(
                'comparison_layout',
                "Standard order",
            ),
            'matrix_outputs': state.get('matrix_outputs', []),
            'extreme_run_mode': state.get('extreme_run_mode', False),
            'case_limit': state.get('case_limit'),
            'requested_batch_count': state.get('requested_batch_count', 1),
            'requested_batch_size': state.get('requested_batch_size', 1),
            'cells_per_seed_round': state.get('cells_per_seed_round'),
            'requested_cells_per_model': state.get('requested_cells_per_model'),
            'requested_generations': state.get('requested_generations'),
            'progress_job_count': state.get('progress_job_count'),
            'seed_rounds': state.get('seed_rounds', []),
            'individual_output_dir': state.get('individual_output_dir'),
            'individual_images_finalized': state.get('individual_images_finalized', 0),
            'individual_finalize_error': state.get('individual_finalize_error'),
            'individual_finalize_failures': state.get('individual_finalize_failures', 0),
            'expected_cell_count': len(state.get('cases', [])),
            'completed_cell_count': len(cells),
            'reference_row': state.get('reference_row'),
            'reference_rows': state.get('reference_rows', []),
            'generation_interrupted': state.get('generation_interrupted', False),
            'matrix_only_delete_allowed': state.get(
                'matrix_only_delete_allowed',
                False,
            ),
            'matrix_only_retention_reason': state.get(
                'matrix_only_retention_reason'
            ),
            'recovery_retained': state.get('recovery_retained', False),
            'cleanup_incomplete': state.get('cleanup_incomplete', False),
            'model_unload_attempted': state.get('model_unload_attempted', False),
            'model_unloaded': state.get('model_unloaded', False),
            'model_unload_error': state.get('model_unload_error'),
            'multi_model': state.get('multi_model', False),
            'multi_model_mode': state.get('multi_model_mode'),
            'multi_model_overview_order': state.get(
                'multi_model_overview_order'
            ),
            'multi_model_page_records': state.get('multi_model_page_records', []),
            'multi_model_temporary_pages': state.get(
                'multi_model_temporary_pages',
                [],
            ),
            'multi_model_source_cleanup_complete': state.get(
                'multi_model_source_cleanup_complete'
            ),
            'multi_model_overview_metadata': state.get(
                'multi_model_overview_metadata',
                [],
            ),
            'model_jobs': state.get('model_jobs', []),
            'model_results': state.get('model_results', []),
            'original_checkpoint': state.get('original_checkpoint'),
            'original_model_restored': state.get(
                'original_model_restored',
                False,
            ),
            'original_model_restore_error': state.get(
                'original_model_restore_error'
            ),
            'multi_model_overview_pages': state.get(
                'multi_model_overview_pages',
                [],
            ),
            'multi_model_overview_error': state.get('multi_model_overview_error'),
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

        with InputAccordion(False, label="LoRA / Embedding Tester", elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester") as lora_tester_enabled:
            with gr.Tabs(
                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_tabs"
            ):
                with gr.Tab("1. Selection"):
                    test_type = gr.Radio(
                        choices=[self.TEST_TYPE_LORA, self.TEST_TYPE_EMBEDDING],
                        value=self.TEST_TYPE_LORA,
                        label="What should be tested?",
                        info=(
                            "Embedding Test currently supports SDXL, Pony, and "
                            "Illustrious checkpoints with dual-encoder SDXL embeddings."
                        ),
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_type",
                    )

                    with gr.Group(visible=True) as lora_controls:
                        with FormRow():
                            refresh_loras_btn = gr.Button(
                                "Refresh LoRA List", variant="secondary", size="sm"
                            )
                        with FormRow():
                            lora_folder = gr.Dropdown(
                                choices=[(
                                    "Manual selection (all folders)",
                                    self.ALL_LORA_FOLDERS,
                                )],
                                value=self.ALL_LORA_FOLDERS,
                                label="LoRA Folder",
                                info=(
                                    "A folder selection loads every LoRA in that scope. "
                                    "Manual selection tests only explicitly selected LoRAs."
                                ),
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_folder",
                            )
                            include_subfolders = gr.Checkbox(
                                value=True,
                                label="Include subfolders",
                                info="Also include LoRAs below the selected folder.",
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_include_subfolders",
                            )
                        lora_selection = gr.Dropdown(
                            choices=[],
                            multiselect=True,
                            label="Select LoRAs to Test",
                            info="Select one or more LoRAs to test with the current prompt.",
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_selection",
                        )
                        global_weight_spec = gr.Textbox(
                            value="1.0",
                            label="Global LoRA Weight(s)",
                            placeholder="1.0 or -3:3:0.5",
                            info=(
                                "Use one value or an inclusive start:end:step range. "
                                "The table below can override individual LoRAs."
                            ),
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_weights",
                        )
                        with FormRow():
                            use_trigger_words = gr.Checkbox(
                                value=True,
                                label="Insert Trigger Words",
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_use_triggers",
                            )
                            trigger_position = gr.Radio(
                                choices=["Start", "End"],
                                value="Start",
                                label="Trigger Word Position",
                                info="Start = before the prompt. End = after the LoRA tag.",
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_trigger_pos",
                            )
                        gr.HTML(
                            value=(
                                "<p class='lora-tester-section-note'>"
                                "Per-LoRA overrides: <code>0.8 | blank | blank</code> "
                                "uses one weight; <code>-1 | 1 | 0.25</code> uses a range. "
                                "Blank weight cells use the global setting. Blank triggers "
                                "use metadata; <code>&lt;none&gt;</code> disables them.</p>"
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
                            visible=False,
                            wrap=True,
                            height=300,
                            column_widths=["30%", "34%", "14%", "11%", "11%"],
                            label="Selected LoRA Settings",
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_settings",
                        )

                    with gr.Group(visible=False) as embedding_controls:
                        preset_bridge = gr.Textbox(
                            value=resolve_forge_preset(forge_options=shared.opts),
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_preset_bridge",
                            elem_classes=["lora-tester-preset-bridge"],
                        )
                        preset_refresh_bridge = gr.Button(
                            "Sync Forge preset",
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_preset_refresh",
                            elem_classes=["lora-tester-preset-bridge"],
                        )
                        with FormRow():
                            refresh_embeddings_btn = gr.Button(
                                "Refresh Compatible Embeddings",
                                variant="secondary",
                                size="sm",
                            )
                        embedding_status = gr.HTML(
                            value=(
                                "<p class='lora-tester-section-note'>Choose Forge UI "
                                "Preset <code>xl</code> to discover compatible Safetensors "
                                "embeddings before loading a checkpoint.</p>"
                            )
                        )
                        with FormRow():
                            embedding_folder = gr.Dropdown(
                                choices=[(
                                    "Manual selection (all folders)",
                                    self.ALL_EMBEDDING_FOLDERS,
                                )],
                                value=self.ALL_EMBEDDING_FOLDERS,
                                label="Embedding Folder",
                                info=(
                                    "Safetensors headers are filtered for clip_l + clip_g; "
                                    "the loaded checkpoint validates them at runtime."
                                ),
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_embedding_folder",
                            )
                            include_embedding_subfolders = gr.Checkbox(
                                value=True,
                                label="Include subfolders",
                                info="Also include compatible embeddings below the folder.",
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_embedding_subfolders",
                            )
                        embedding_selection = gr.Dropdown(
                            choices=[],
                            multiselect=True,
                            label="Select Embeddings to Test",
                            info="Select one or more compatible textual inversion embeddings.",
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_embedding_selection",
                        )
                        with FormRow():
                            global_embedding_weight_spec = gr.Textbox(
                                value="1.0",
                                label="Global Embedding Weight(s)",
                                placeholder="1.0 or 0.5:1.5:0.25",
                                info=(
                                    "Weights use Forge prompt attention. The table below "
                                    "can override individual embeddings."
                                ),
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_embedding_weights",
                            )
                            embedding_position = gr.Radio(
                                choices=["Start", "End"],
                                value="Start",
                                label="Embedding Position",
                                info="Insert the weighted embedding before or after the prompt.",
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_embedding_position",
                            )
                        gr.HTML(
                            value=(
                                "<p class='lora-tester-section-note'>Blank triggers use "
                                "JSON metadata or the filename; <code>&lt;none&gt;</code> "
                                "suppresses the additional trigger. Set Negative prompt? "
                                "to <code>0</code> for positive or <code>1</code> for "
                                "negative injection.</p>"
                            )
                        )
                        embedding_settings = gr.Dataframe(
                            value=[],
                            headers=[
                                "Embedding (do not edit)",
                                "Trigger words (comma-separated)",
                                "Negative prompt? (0 = Positive, 1 = Negative)",
                                "Min / single weight",
                                "Max",
                                "Step",
                            ],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            type="array",
                            row_count=(0, "dynamic"),
                            col_count=(6, "fixed"),
                            interactive=True,
                            visible=False,
                            wrap=True,
                            height=300,
                            column_widths=["25%", "28%", "19%", "11%", "9%", "8%"],
                            label="Selected Embedding Settings",
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_embedding_settings",
                        )

                with gr.Tab("2. Models & Run Size"):
                    multi_model_mode = gr.Radio(
                        choices=[
                            self.MULTI_MODEL_CURRENT,
                            self.MULTI_MODEL_LORA,
                            self.MULTI_MODEL_PROMPT,
                        ],
                        value=self.MULTI_MODEL_CURRENT,
                        label="Which models should be used?",
                        info=(
                            "Multiple-model modes run complete Forge jobs sequentially "
                            "and fully unload each checkpoint before loading the next."
                        ),
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_model_mode",
                    )
                    with gr.Group(visible=False) as multi_model_controls:
                        checkpoint_choices = self._checkpoint_choices()
                        with FormRow():
                            refresh_checkpoints_btn = gr.Button(
                                "Refresh Checkpoint List",
                                variant="secondary",
                                size="sm",
                            )
                        with FormRow():
                            checkpoint_folder = gr.Dropdown(
                                choices=self._checkpoint_folder_choices(
                                    checkpoint_choices
                                ),
                                value=self.ALL_CHECKPOINT_FOLDERS,
                                label="Checkpoint Folder",
                                info=(
                                    "A folder selection loads every checkpoint in that "
                                    "scope. Manual selection tests only explicitly "
                                    "selected checkpoints."
                                ),
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_checkpoint_folder",
                            )
                            include_checkpoint_subfolders = gr.Checkbox(
                                value=True,
                                label="Include subfolders",
                                info=(
                                    "Also include checkpoints below the selected folder."
                                ),
                                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_include_checkpoint_subfolders",
                            )
                        checkpoint_selection = gr.Dropdown(
                            choices=self._filtered_checkpoint_choices(
                                self.ALL_CHECKPOINT_FOLDERS,
                                True,
                                checkpoint_choices,
                            ),
                            multiselect=True,
                            label="Select Models / Checkpoints",
                            info=(
                                "Standard mode supports 25 checkpoints; Extreme Run "
                                "Mode supports 150. Forge remains the compatibility authority."
                            ),
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_checkpoint_selection",
                        )
                        multi_model_overview_order = gr.Radio(
                            choices=[
                                self.MULTI_OVERVIEW_MODELS_FIRST,
                                self.MULTI_OVERVIEW_SEEDS_FIRST,
                            ],
                            value=self.MULTI_OVERVIEW_MODELS_FIRST,
                            label="Shared overview organization",
                            info=(
                                "Both choices create one shared overview unless a "
                                "matrix safety limit requires multiple pages. Models "
                                "first groups seed rounds by model; Seeds first creates "
                                "one labeled row per seed with models as columns."
                            ),
                            elem_id=(
                                f"{'img2img' if is_img2img else 'txt2img'}_"
                                "lora_tester_multi_overview_order"
                            ),
                        )
                        gr.HTML(
                            value=(
                                "<p class='lora-tester-section-note'>Prompt-only loads no "
                                "LoRA file but reuses the same fixed-seed, disk-spooling, "
                                "matrix, and recovery path.</p>"
                            )
                        )
                    extreme_run_mode = gr.Checkbox(
                        value=False,
                        label=(
                            "Extreme Run Mode (up to 10,000 matrix cells / "
                            "150 checkpoints)"
                        ),
                        info=(
                            "CAUTION: Extreme runs can take many hours or days and use "
                            "very large amounts of disk space. Standard limits are 500 "
                            "cells per model and 25 checkpoints."
                        ),
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_extreme_run",
                    )

                with gr.Tab("3. Grid"):
                    with FormRow():
                        save_original = gr.Checkbox(
                            value=True,
                            label="Generate reference image without tested item (Recommended)",
                            info=(
                                "The fixed-seed reference is repeated at the top of every "
                                "matrix page."
                            ),
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_baseline",
                        )
                        draw_legend = gr.Checkbox(
                            value=True,
                            label="Draw legend (names, weights, and triggers)",
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_draw_legend",
                        )
                    per_item_grids = gr.Checkbox(
                        value=False,
                        label="Create one grid per tested LoRA or Embedding",
                        info="Disabled returns the familiar single combined matrix.",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_per_item_grids",
                    )
                    combined_overview = gr.Checkbox(
                        value=True,
                        interactive=False,
                        label="Also create the combined overview",
                        info=(
                            "Available with per-item grids. Without them, the combined "
                            "matrix remains the required primary output."
                        ),
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_combined_overview",
                    )
                    comparison_width_bridge = gr.Number(
                        value=1024,
                        precision=0,
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_comparison_width",
                        elem_classes=["lora-tester-layout-bridge"],
                    )
                    comparison_height_bridge = gr.Number(
                        value=1024,
                        precision=0,
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_comparison_height",
                        elem_classes=["lora-tester-layout-bridge"],
                    )
                    comparison_refresh_bridge = gr.Button(
                        "Refresh comparison capacity",
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_comparison_refresh",
                        elem_classes=["lora-tester-layout-bridge"],
                    )
                    comparison_capacity = gr.HTML(
                        value=self._comparison_capacity_html(
                            self.TEST_TYPE_LORA, [], [], 1024, 1024, 5, True, True
                        ),
                        visible=False,
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_comparison_capacity",
                    )
                    comparison_layout = gr.Radio(
                        choices=[
                            self.MATRIX_LAYOUT_STANDARD,
                            self.MATRIX_LAYOUT_COMPARE_HORIZONTAL,
                            self.MATRIX_LAYOUT_COMPARE_VERTICAL,
                        ],
                        value=self.MATRIX_LAYOUT_STANDARD,
                        label="Combined matrix layout",
                        info=(
                            "Comparison rows/columns require identical weights for every "
                            "item and support at most 10 items."
                        ),
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_comparison_layout",
                    )
                    with FormRow():
                        matrix_cols = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=3,
                            label="Standard combined matrix columns",
                            info=(
                                "Used only by the Standard order combined overview. "
                                "Per-item grids keep an item's weights together."
                            ),
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_matrix_cols",
                        )
                        matrix_margin = gr.Slider(
                            minimum=0,
                            maximum=50,
                            step=1,
                            value=5,
                            label="Matrix Image Margin (px)",
                            elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_matrix_margin",
                        )

                with gr.Tab("4. Output & Safety"):
                    output_retention = gr.Radio(
                        choices=[self.KEEP_INDIVIDUAL_OUTPUT, self.MATRIX_ONLY_OUTPUT],
                        value=self.KEEP_INDIVIDUAL_OUTPUT,
                        label="Output retention",
                        info=(
                            "CAUTION: Matrix-only deletes individual comparison images "
                            "only after every matrix page was created and validated. "
                            "Failures retain the sources for recovery."
                        ),
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_output_retention",
                    )
                    unload_models_before_matrix = gr.Checkbox(
                        value=True,
                        label=(
                            "Unload checkpoint, text encoder and VAE before building "
                            "the matrix (Recommended)"
                        ),
                        info=(
                            "Frees Forge model RAM/VRAM for larger matrix pages. The "
                            "selected checkpoint reloads automatically for the next job."
                        ),
                        elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_unload_models",
                    )
                    gr.HTML(
                        value=(
                            "<div class='lora-tester-safety-note'><b>Active safeguards:</b> "
                            "completed cells are spooled to disk, model state can be fully "
                            "unloaded before composition, pages stay below the fixed axis "
                            "and area limits, and Matrix-only sources are deleted only after "
                            "successful validation.</div>"
                        )
                    )

            run_summary = gr.HTML(
                value=self._run_summary_html(
                    self.TEST_TYPE_LORA,
                    [],
                    "1.0",
                    [],
                    [],
                    "1.0",
                    [],
                    True,
                    self.MULTI_MODEL_CURRENT,
                    [],
                    False,
                ),
                elem_id=f"{'img2img' if is_img2img else 'txt2img'}_lora_tester_run_summary",
            )

        # Event handlers
        def settings_table_update(rows):
            """Show selection-owned rows and hide the empty table state."""

            rows = list(rows or [])
            return gr.update(
                value=rows,
                visible=bool(rows),
            )

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
                settings_table_update(rows),
            )

        def load_loras_on_activation(
            enabled, folder_filter, include_children, selected_loras, current_settings
        ):
            """Populate the list the first time the accordion is enabled."""
            if not enabled:
                return gr.update(), gr.update(), gr.update()
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
                settings_table_update(rows),
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
            return (
                gr.update(choices=choices, value=selected_loras),
                settings_table_update(rows),
            )

        def update_lora_settings(selected_loras, current_settings):
            rows = self._build_lora_settings_rows(selected_loras, current_settings)
            return settings_table_update(rows)

        def format_embedding_status(message, is_error=False):
            color = "#d99a35" if is_error else "#888"
            return (
                f"<p style='color: {color}; font-size: 0.9em;'>"
                f"{html.escape(message)}</p>"
            )

        def update_embedding_list(
            folder_filter,
            include_children,
            selected_embeddings,
            current_settings,
            preset,
            refresh=True,
        ):
            """Use the UI preset early and the loaded dual-CLIP DB when available."""

            self.cached_embeddings, status = (
                EmbeddingMetadataReader.find_for_preset(
                    preset=preset,
                    refresh=refresh,
                )
            )
            folder_choices = self._embedding_folder_choices()
            folder_values = {choice[1] for choice in folder_choices}
            if folder_filter not in folder_values:
                folder_filter = self.ALL_EMBEDDING_FOLDERS

            choices = self._filtered_embedding_choices(
                folder_filter, include_children
            )
            allowed_embeddings = set(choices)
            selected_embeddings = [
                embedding
                for embedding in (selected_embeddings or [])
                if embedding in allowed_embeddings
            ]
            rows = self._build_embedding_settings_rows(
                selected_embeddings, current_settings
            )
            is_error = not self.cached_embeddings
            return (
                gr.update(choices=folder_choices, value=folder_filter),
                gr.update(choices=choices, value=selected_embeddings),
                settings_table_update(rows),
                format_embedding_status(status, is_error=is_error),
            )

        def load_embeddings_on_activation(
            enabled,
            folder_filter,
            include_children,
            selected_embeddings,
            current_settings,
            preset,
        ):
            if not enabled:
                return gr.update(), gr.update(), gr.update(), gr.update()
            return update_embedding_list(
                folder_filter,
                include_children,
                selected_embeddings,
                current_settings,
                preset,
                refresh=False,
            )

        def update_embeddings_for_selected_type(
            selected_type,
            preset,
            folder_filter,
            include_children,
            selected_embeddings,
            current_settings,
        ):
            if selected_type != self.TEST_TYPE_EMBEDDING:
                return gr.update(), gr.update(), gr.update(), gr.update()
            return update_embedding_list(
                folder_filter,
                include_children,
                selected_embeddings,
                current_settings,
                preset,
                refresh=False,
            )

        def apply_embedding_folder_filter(
            folder_filter, include_children, selected_embeddings, current_settings
        ):
            choices = self._filtered_embedding_choices(
                folder_filter, include_children
            )
            if folder_filter in (None, "", self.ALL_EMBEDDING_FOLDERS):
                allowed_embeddings = set(choices)
                selected_embeddings = [
                    embedding
                    for embedding in (selected_embeddings or [])
                    if embedding in allowed_embeddings
                ]
            else:
                selected_embeddings = list(choices)
            rows = self._build_embedding_settings_rows(
                selected_embeddings, current_settings
            )
            return (
                gr.update(choices=choices, value=selected_embeddings),
                settings_table_update(rows),
            )

        def update_embedding_settings(selected_embeddings, current_settings):
            rows = self._build_embedding_settings_rows(
                selected_embeddings, current_settings
            )
            return settings_table_update(rows)

        def toggle_test_type(selected_type):
            embedding_mode = selected_type == self.TEST_TYPE_EMBEDDING
            return (
                gr.update(visible=not embedding_mode),
                gr.update(visible=embedding_mode),
            )

        def toggle_multi_model_controls(selected_mode):
            return gr.update(visible=selected_mode != self.MULTI_MODEL_CURRENT)

        def update_run_summary(
            selected_type,
            selected_loras,
            lora_weights,
            selected_lora_settings,
            selected_embeddings,
            embedding_weights,
            selected_embedding_settings,
            reference_enabled,
            selected_model_mode,
            selected_models,
            extreme_enabled,
        ):
            return self._run_summary_html(
                selected_type,
                selected_loras,
                lora_weights,
                selected_lora_settings,
                selected_embeddings,
                embedding_weights,
                selected_embedding_settings,
                reference_enabled,
                selected_model_mode,
                selected_models,
                extreme_enabled,
            )

        def update_checkpoint_list(
            folder_filter, include_children, selected_checkpoints
        ):
            all_choices = self._checkpoint_choices(refresh=True)
            folder_choices = self._checkpoint_folder_choices(all_choices)
            folder_values = {choice[1] for choice in folder_choices}
            if folder_filter not in folder_values:
                folder_filter = self.ALL_CHECKPOINT_FOLDERS

            choices = self._filtered_checkpoint_choices(
                folder_filter,
                include_children,
                all_choices,
            )
            valid_values = {value for _, value in choices}
            if folder_filter in (None, "", self.ALL_CHECKPOINT_FOLDERS):
                selected_checkpoints = [
                    checkpoint
                    for checkpoint in (selected_checkpoints or [])
                    if checkpoint in valid_values
                ]
            else:
                selected_checkpoints = [value for _, value in choices]
            return (
                gr.update(choices=folder_choices, value=folder_filter),
                gr.update(choices=choices, value=selected_checkpoints),
            )

        def apply_checkpoint_folder_filter(
            folder_filter, include_children, selected_checkpoints
        ):
            choices = self._filtered_checkpoint_choices(
                folder_filter,
                include_children,
            )
            valid_values = {value for _, value in choices}
            if folder_filter in (None, "", self.ALL_CHECKPOINT_FOLDERS):
                selected_checkpoints = [
                    checkpoint
                    for checkpoint in (selected_checkpoints or [])
                    if checkpoint in valid_values
                ]
            else:
                selected_checkpoints = [value for _, value in choices]
            return gr.update(choices=choices, value=selected_checkpoints)

        def update_output_controls(create_per_item, create_combined):
            create_per_item, combined_enabled = self._resolve_matrix_outputs(
                create_per_item,
                create_combined,
            )
            return (
                gr.update(value=combined_enabled, interactive=create_per_item),
                gr.update(interactive=combined_enabled),
            )

        def update_comparison_interactivity(create_per_item, create_combined):
            _, combined_enabled = self._resolve_matrix_outputs(
                create_per_item,
                create_combined,
            )
            return gr.update(
                interactive=combined_enabled
            )

        def update_comparison_capacity(
            selected_type,
            selected_loras,
            selected_embeddings,
            width,
            height,
            margin,
            legend,
            reference,
        ):
            return self._comparison_capacity_html(
                selected_type,
                selected_loras,
                selected_embeddings,
                width,
                height,
                margin,
                legend,
                reference,
            )

        def update_comparison_capacity_visibility(
            selected_layout,
            create_per_item,
            create_combined,
        ):
            return gr.update(
                visible=self._comparison_capacity_visible(
                    selected_layout,
                    create_per_item,
                    create_combined,
                )
            )

        # Wire up event handlers
        test_type_event = test_type.change(
            fn=toggle_test_type,
            inputs=[test_type],
            outputs=[lora_controls, embedding_controls],
        )
        test_type_event.then(
            fn=update_embeddings_for_selected_type,
            inputs=[
                test_type,
                preset_bridge,
                embedding_folder,
                include_embedding_subfolders,
                embedding_selection,
                embedding_settings,
            ],
            outputs=[
                embedding_folder,
                embedding_selection,
                embedding_settings,
                embedding_status,
            ],
        )

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

        lora_tester_enabled.change(
            fn=load_embeddings_on_activation,
            inputs=[
                lora_tester_enabled,
                embedding_folder,
                include_embedding_subfolders,
                embedding_selection,
                embedding_settings,
                preset_bridge,
            ],
            outputs=[
                embedding_folder,
                embedding_selection,
                embedding_settings,
                embedding_status,
            ],
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

        refresh_embeddings_btn.click(
            fn=update_embedding_list,
            inputs=[
                embedding_folder,
                include_embedding_subfolders,
                embedding_selection,
                embedding_settings,
                preset_bridge,
            ],
            outputs=[
                embedding_folder,
                embedding_selection,
                embedding_settings,
                embedding_status,
            ],
        )

        preset_refresh_bridge.click(
            fn=update_embeddings_for_selected_type,
            inputs=[
                test_type,
                preset_bridge,
                embedding_folder,
                include_embedding_subfolders,
                embedding_selection,
                embedding_settings,
            ],
            outputs=[
                embedding_folder,
                embedding_selection,
                embedding_settings,
                embedding_status,
            ],
            queue=False,
            show_progress=False,
        )

        embedding_folder.input(
            fn=apply_embedding_folder_filter,
            inputs=[
                embedding_folder,
                include_embedding_subfolders,
                embedding_selection,
                embedding_settings,
            ],
            outputs=[embedding_selection, embedding_settings],
        )

        include_embedding_subfolders.input(
            fn=apply_embedding_folder_filter,
            inputs=[
                embedding_folder,
                include_embedding_subfolders,
                embedding_selection,
                embedding_settings,
            ],
            outputs=[embedding_selection, embedding_settings],
        )

        embedding_selection.change(
            fn=update_embedding_settings,
            inputs=[embedding_selection, embedding_settings],
            outputs=[embedding_settings],
        )

        multi_model_mode.change(
            fn=toggle_multi_model_controls,
            inputs=[multi_model_mode],
            outputs=[multi_model_controls],
            queue=False,
        )

        refresh_checkpoints_btn.click(
            fn=update_checkpoint_list,
            inputs=[
                checkpoint_folder,
                include_checkpoint_subfolders,
                checkpoint_selection,
            ],
            outputs=[checkpoint_folder, checkpoint_selection],
            show_progress=False,
        )

        checkpoint_folder.input(
            fn=apply_checkpoint_folder_filter,
            inputs=[
                checkpoint_folder,
                include_checkpoint_subfolders,
                checkpoint_selection,
            ],
            outputs=[checkpoint_selection],
        )

        include_checkpoint_subfolders.input(
            fn=apply_checkpoint_folder_filter,
            inputs=[
                checkpoint_folder,
                include_checkpoint_subfolders,
                checkpoint_selection,
            ],
            outputs=[checkpoint_selection],
        )

        per_item_grids.change(
            fn=update_output_controls,
            inputs=[per_item_grids, combined_overview],
            outputs=[combined_overview, comparison_layout],
            queue=False,
        )
        combined_overview.change(
            fn=update_comparison_interactivity,
            inputs=[per_item_grids, combined_overview],
            outputs=[comparison_layout],
            queue=False,
        )

        capacity_visibility_inputs = [
            comparison_layout,
            per_item_grids,
            combined_overview,
        ]
        for component in (
            comparison_layout,
            per_item_grids,
            combined_overview,
        ):
            component.change(
                fn=update_comparison_capacity_visibility,
                inputs=capacity_visibility_inputs,
                outputs=[comparison_capacity],
                queue=False,
                show_progress=False,
            )

        capacity_inputs = [
            test_type,
            lora_selection,
            embedding_selection,
            comparison_width_bridge,
            comparison_height_bridge,
            matrix_margin,
            draw_legend,
            save_original,
        ]
        for component in (
            test_type,
            lora_selection,
            embedding_selection,
            matrix_margin,
            draw_legend,
            save_original,
        ):
            component.change(
                fn=update_comparison_capacity,
                inputs=capacity_inputs,
                outputs=[comparison_capacity],
                queue=False,
                show_progress=False,
            )
        comparison_refresh_bridge.click(
            fn=update_comparison_capacity,
            inputs=capacity_inputs,
            outputs=[comparison_capacity],
            queue=False,
            show_progress=False,
        )

        run_summary_inputs = [
            test_type,
            lora_selection,
            global_weight_spec,
            lora_settings,
            embedding_selection,
            global_embedding_weight_spec,
            embedding_settings,
            save_original,
            multi_model_mode,
            checkpoint_selection,
            extreme_run_mode,
        ]
        for component in run_summary_inputs:
            component.change(
                fn=update_run_summary,
                inputs=run_summary_inputs,
                outputs=[run_summary],
                queue=False,
                show_progress=False,
            )

        # Return all components that will be passed to processing callbacks.
        return [
            lora_tester_enabled,
            test_type,
            lora_selection,
            global_weight_spec,
            use_trigger_words,
            trigger_position,
            lora_settings,
            embedding_selection,
            global_embedding_weight_spec,
            embedding_position,
            embedding_settings,
            save_original,
            per_item_grids,
            combined_overview,
            comparison_layout,
            extreme_run_mode,
            draw_legend,
            matrix_cols,
            matrix_margin,
            output_retention,
            unload_models_before_matrix,
            multi_model_mode,
            checkpoint_selection,
            multi_model_overview_order,
        ]

    def before_process(self, p: StableDiffusionProcessing,
                       lora_tester_enabled, test_type,
                       lora_selection, global_weight_spec,
                       use_trigger_words, trigger_position, lora_settings,
                       embedding_selection, global_embedding_weight_spec,
                       embedding_position, embedding_settings,
                       save_original, per_item_grids, combined_overview,
                       comparison_layout, extreme_run_mode, draw_legend,
                       matrix_cols, matrix_margin,
                       output_retention,
                       unload_models_before_matrix,
                       multi_model_mode=None, checkpoint_selection=None,
                       multi_model_overview_order=None):
        """
        Prepare the current generation before Forge expands prompts and seeds.

        The state is stored on ``p`` instead of the long-lived Script instance. This
        prevents a previously enabled run from leaking into later disabled runs.
        """

        setattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)

        if not lora_tester_enabled:
            return

        multi_model_mode = multi_model_mode or self.MULTI_MODEL_CURRENT
        multi_model_overview_order = (
            multi_model_overview_order or self.MULTI_OVERVIEW_MODELS_FIRST
        )
        if multi_model_overview_order not in {
            self.MULTI_OVERVIEW_MODELS_FIRST,
            self.MULTI_OVERVIEW_SEEDS_FIRST,
        }:
            multi_model_overview_order = self.MULTI_OVERVIEW_MODELS_FIRST
        multi_model_error = self._multi_model_request_error(
            multi_model_mode,
            checkpoint_selection,
            test_type,
            lora_selection,
            extreme_run_mode,
        )
        if multi_model_error:
            self._block_generation(p, multi_model_error)
            return

        if multi_model_mode != self.MULTI_MODEL_CURRENT:
            requested_batch_count = max(1, int(getattr(p, 'n_iter', 1) or 1))
            requested_batch_size = max(1, int(getattr(p, 'batch_size', 1) or 1))
            cells_per_seed_round, cell_plan_error = self._multi_model_cells_per_seed(
                multi_model_mode,
                lora_selection,
                global_weight_spec,
                lora_settings,
                save_original,
            )
            if cell_plan_error:
                self._block_generation(p, cell_plan_error)
                return
            requested_cells_per_model = cells_per_seed_round * requested_batch_count
            case_limit = self._maximum_total_cases(extreme_run_mode)
            if requested_cells_per_model > case_limit:
                mode_hint = (
                    "Reduce the weight range or Forge Batch Count."
                    if extreme_run_mode
                    else (
                        "Reduce the weight range or Forge Batch Count, or enable "
                        "Extreme Run Mode."
                    )
                )
                self._block_generation(
                    p,
                    f"Forge Batch Count expands this multiple-model run to "
                    f"{requested_cells_per_model:,} cells per checkpoint, above the "
                    f"{case_limit:,}-cell limit. {mode_hint}",
                )
                return
            model_jobs, missing_checkpoints = self._resolve_checkpoint_jobs(
                checkpoint_selection
            )
            if missing_checkpoints:
                self._block_generation(
                    p,
                    "One or more selected checkpoints are no longer available. "
                    "Refresh the checkpoint list. First missing item: "
                    f"'{missing_checkpoints[0]}'.",
                )
                return

            self._ensure_temp_directory()
            output_mode = self._processing_output_mode(p)
            try:
                session_dir, individual_output_dir = self._create_spool_directory(
                    p,
                    output_mode,
                )
            except OSError as error:
                self._block_generation(
                    p,
                    f"Could not create the multiple-model spool directory: {error}",
                )
                return

            state = {
                'multi_model': True,
                'multi_model_mode': multi_model_mode,
                'model_jobs': model_jobs,
                'model_results': [],
                'multi_model_pages': [],
                'multi_model_infotexts': [],
                'multi_model_prompts': [],
                'multi_model_negative_prompts': [],
                'multi_model_seeds': [],
                'multi_model_subseeds': [],
                'multi_model_page_records': [],
                'multi_model_temporary_pages': [],
                'multi_model_overview_order': multi_model_overview_order,
                'multi_model_overview_metadata': [],
                'requested_batch_count': requested_batch_count,
                'requested_batch_size': requested_batch_size,
                'cells_per_seed_round': cells_per_seed_round,
                'requested_cells_per_model': requested_cells_per_model,
                'requested_generations': requested_cells_per_model * len(model_jobs),
                'seed_rounds': [],
                'original_checkpoint': None,
                'original_model_restored': False,
                'cases': [
                    {
                        'kind': 'model',
                        'label': job['label'],
                        'checkpoint': job['checkpoint'],
                    }
                    for job in model_jobs
                ],
                'cells': {},
                'rows': [],
                'matrix_outputs': [],
                'matrix_cols': int(matrix_cols),
                'matrix_margin': int(matrix_margin),
                'draw_legend': bool(draw_legend),
                'output_retention': output_retention,
                'output_mode': output_mode,
                'individual_output_dir': str(individual_output_dir),
                'session_dir': str(session_dir),
                'manifest_path': str(session_dir / "manifest.json"),
                'fixed_seed': None,
                'fixed_subseed': None,
                'model_unload_attempted': False,
                'model_unloaded': False,
            }
            setattr(p, self.PROCESSING_STATE_ATTRIBUTE, state)
            self._write_manifest(state, "models-pending")

            # The outer Forge job is only the coordinator. Every selected model is
            # executed by a fresh nested process_images() job in process(), which
            # lets Forge recalculate family-specific latent shapes, samplers, text
            # encoders and VAEs instead of switching inside one initialized loop.
            # Run our coordinator callbacks after every other always-on script so
            # none of them can restore an iteration that would sample the shell job.
            self._move_spool_hooks_last(p)
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.do_not_save_samples = True
            print(
                f"[LoRA Tester] Multiple-model mode enabled with "
                f"{len(model_jobs)} sequential checkpoint job(s), "
                f"Forge Batch Count {requested_batch_count}, "
                f"{requested_cells_per_model} cell(s) per checkpoint, "
                f"{requested_cells_per_model * len(model_jobs)} total generation(s)"
            )
            if requested_batch_size != 1:
                print(
                    "[LoRA Tester] Forge Batch Size remains serialized to 1 for "
                    "disk-spooling and recovery safety; Batch Count is preserved."
                )
            return

        embedding_mode = test_type == self.TEST_TYPE_EMBEDDING
        prompt_only_mode = test_type == self.TEST_TYPE_PROMPT
        if prompt_only_mode:
            selected_items = [self.PROMPT_ONLY_SENTINEL]
            item_label = "Prompt"
        else:
            selected_items = embedding_selection if embedding_mode else lora_selection
            item_label = "Embedding" if embedding_mode else "LoRA"
        if not selected_items:
            self._block_generation(
                p,
                f"{item_label} Tester is enabled, but no {item_label} is selected. "
                f"Select at least one {item_label} or disable the tester before generating.",
            )
            return

        self._move_spool_hooks_last(p)
        self._ensure_temp_directory()

        if embedding_mode:
            support_error = EmbeddingMetadataReader.support_error(
                getattr(p, "sd_model", None)
            )
            if support_error:
                self._block_generation(p, support_error)
                return
            current_embeddings, _ = (
                EmbeddingMetadataReader.find_compatible_embeddings(
                    refresh=False,
                    model=getattr(p, "sd_model", None),
                )
            )
            self.cached_embeddings = current_embeddings
            missing_embeddings = [
                item for item in selected_items if item not in current_embeddings
            ]
            if missing_embeddings:
                self._block_generation(
                    p,
                    "One or more selected embeddings are not compatible with the "
                    "currently loaded SDXL-family checkpoint. Refresh the compatible "
                    f"embedding list. First unavailable item: '{missing_embeddings[0]}'.",
                )
                return
        elif not prompt_only_mode and not self.cached_loras:
            self.cached_loras = LoRaMetadataReader.find_all_loras()

        if prompt_only_mode:
            global_weights = [None]
        else:
            selected_global_spec = (
                global_embedding_weight_spec if embedding_mode else global_weight_spec
            )
            global_weight_text = self._cell_text(selected_global_spec)
            if global_weight_text:
                global_weights = self._parse_weight_spec(global_weight_text, "global")
                if global_weights is None:
                    self._block_generation(
                        p,
                        "The global weight is invalid. Enter one number or an inclusive "
                        "Min:Max:Step range (for example -1:1:0.25).",
                    )
                    return
            else:
                global_weights = [1.0]

        settings_by_lora = {
            str(row[0]): row
            for row in self._coerce_settings_rows(lora_settings)
            if row[0] not in (None, "")
        }
        settings_by_embedding = {
            str(row[0]): row
            for row in self._coerce_embedding_settings_rows(embedding_settings)
            if row[0] not in (None, "")
        }

        case_limit = self._maximum_total_cases(extreme_run_mode)
        cases = []
        batch_model_label = getattr(p, self.BATCH_MODEL_LABEL_ATTRIBUTE, None)
        if save_original:
            baseline_label = f"Baseline (No {item_label})"
            if batch_model_label:
                baseline_label = f"{batch_model_label} — {baseline_label}"
            cases.append({
                'kind': "baseline",
                'label': baseline_label,
                'item_key': None,
                'item_label': None,
                'lora_tag_name': None,
                'embedding_name': None,
                'trigger_words': [],
                'weight': None,
            })

        for item_path in selected_items:
            if prompt_only_mode:
                display_name = batch_model_label or "Prompt only"
                embedding_name = None
                embedding_target = None
                lora_tag_name = None
                trigger_words = []
                weights = [None]
                weight_error = None
            elif embedding_mode:
                embedding_info = self.cached_embeddings.get(item_path)
                if embedding_info is None:
                    print(
                        f"[LoRA Tester] Warning: Embedding {item_path} not found in cache"
                    )
                    continue
                display_name = embedding_info["name"]
                embedding_name = embedding_info["name"]
                settings_row = settings_by_embedding.get(item_path)
                metadata_triggers = list(embedding_info.get("trigger_words", []))
                if settings_row is None:
                    trigger_words = self._resolve_trigger_words(metadata_triggers)
                else:
                    trigger_words = self._resolve_trigger_words(
                        metadata_triggers,
                        settings_row[1],
                    )
                embedding_target, target_error = self._resolve_embedding_target(
                    settings_row[2] if settings_row is not None else None
                )
                if target_error:
                    self._block_generation(
                        p,
                        f"{target_error} Embedding: '{item_path}'.",
                    )
                    return
                weights, weight_error = self._resolve_item_weights(
                    settings_row, 3, global_weights, item_path
                )
                lora_tag_name = None
            else:
                lora_path = item_path
                lora_info = self.cached_loras.get(lora_path)
                if lora_info is None:
                    print(f"[LoRA Tester] Warning: LoRA {lora_path} not found in cache")
                    continue

                lora_name = lora_info['name']
                lora_alias = lora_info.get('alias')
                display_name = lora_alias or lora_name
                embedding_name = None
                embedding_target = None
                lora_tag_name = lora_alias or lora_name
                settings_row = settings_by_lora.get(lora_path)
                metadata_triggers = list(lora_info.get('trigger_words', []))

                if settings_row is None:
                    trigger_words = self._resolve_trigger_words(metadata_triggers)
                else:
                    # Gradio can transiently submit an empty Dataframe cell while
                    # folder-selection outputs are settling. Preserve metadata as
                    # the default; <none> remains an explicit per-LoRA opt-out.
                    trigger_words = self._resolve_trigger_words(
                        metadata_triggers,
                        settings_row[1],
                    )
                weights, weight_error = self._resolve_item_weights(
                    settings_row, 2, global_weights, lora_path
                )

            if weight_error:
                self._block_generation(p, weight_error)
                return

            for weight in weights:
                if len(cases) >= case_limit:
                    mode_hint = (
                        "Reduce the selection or weight ranges."
                        if extreme_run_mode
                        else "Reduce the run or enable Extreme Run Mode."
                    )
                    self._block_generation(
                        p,
                        f"The requested run contains more than {case_limit:,} matrix "
                        f"cells. {mode_hint}",
                    )
                    return

                formatted_weight = (
                    None if prompt_only_mode else self._format_weight(weight)
                )
                if prompt_only_mode:
                    case_label = display_name
                    case_item_label = display_name
                else:
                    case_item_label = display_name
                    if batch_model_label:
                        case_item_label = f"{batch_model_label} — {display_name}"
                    case_label = f"{case_item_label} (weight {formatted_weight})"
                cases.append({
                    'kind': (
                        "prompt"
                        if prompt_only_mode
                        else ("embedding" if embedding_mode else "lora")
                    ),
                    'label': case_label,
                    'item_key': item_path,
                    'item_label': case_item_label,
                    'lora_tag_name': lora_tag_name,
                    'embedding_name': embedding_name,
                    'embedding_target': embedding_target,
                    'trigger_words': trigger_words,
                    'weight': formatted_weight,
                })

        comparison_cases = [case for case in cases if not self._case_is_baseline(case)]
        if not comparison_cases:
            self._block_generation(
                p,
                f"No selected {item_label} could be prepared for generation. "
                f"Refresh the {item_label} list and try again.",
            )
            return

        if comparison_layout not in (
            self.MATRIX_LAYOUT_STANDARD,
            self.MATRIX_LAYOUT_COMPARE_HORIZONTAL,
            self.MATRIX_LAYOUT_COMPARE_VERTICAL,
        ):
            comparison_layout = self.MATRIX_LAYOUT_STANDARD
        per_item_grids, combined_requested = self._resolve_matrix_outputs(
            per_item_grids,
            combined_overview,
        )
        if combined_requested and comparison_layout != self.MATRIX_LAYOUT_STANDARD:
            groups = self._group_comparison_cells(comparison_cases)
            comparison_error = self._comparison_group_error(groups)
            if comparison_error:
                self._block_generation(p, comparison_error)
                return

        seed_rounds = list(
            getattr(p, self.BATCH_SEED_ROUNDS_ATTRIBUTE, None) or []
        )
        requested_batch_count = (
            len(seed_rounds)
            if seed_rounds
            else max(1, int(getattr(p, 'n_iter', 1) or 1))
        )
        requested_batch_size = max(1, int(getattr(p, 'batch_size', 1) or 1))
        if not seed_rounds and requested_batch_count > 1:
            from modules.processing import get_fixed_seed

            # Resolve random seeds once before Forge expands the comparison cases.
            # This preserves Forge's seed progression while allowing every case in
            # one Batch Count round to share the same seed.
            requested_seed = getattr(p, 'seed', -1)
            requested_subseed = getattr(p, 'subseed', -1)
            if isinstance(requested_seed, list):
                requested_seed = requested_seed[0] if requested_seed else -1
            if isinstance(requested_subseed, list):
                requested_subseed = requested_subseed[0] if requested_subseed else -1
            p.seed = int(get_fixed_seed(requested_seed))
            p.subseed = int(get_fixed_seed(requested_subseed))
            seed_rounds = self._batch_seed_rounds(
                p.seed,
                p.subseed,
                requested_batch_count,
                getattr(p, 'subseed_strength', 0.0),
            )
        cells_per_seed_round = len(cases)
        if seed_rounds:
            expanded_case_count = len(cases) * len(seed_rounds)
            if expanded_case_count > case_limit:
                self._block_generation(
                    p,
                    f"Forge Batch Count expands this run to {expanded_case_count:,} "
                    f"cells, above the {case_limit:,}-cell limit.",
                )
                return
            cases = self._expand_cases_for_seed_rounds(
                cases,
                seed_rounds,
                include_seed_label=not (
                    bool(batch_model_label)
                    and multi_model_overview_order == self.MULTI_OVERVIEW_SEEDS_FIRST
                ),
            )

        output_mode = self._processing_output_mode(p)
        try:
            session_dir, individual_output_dir = self._create_spool_directory(
                p,
                output_mode,
            )
        except OSError as error:
            print(f"[LoRA Tester] Could not create disk spool directory: {error}")
            return

        state = {
            'cases': cases,
            'test_type': test_type,
            'extreme_run_mode': bool(extreme_run_mode),
            'case_limit': case_limit,
            'use_trigger_words': use_trigger_words,
            'trigger_position': trigger_position,
            'draw_legend': draw_legend,
            'create_per_item_grids': per_item_grids,
            'create_combined_overview': combined_requested,
            'comparison_layout': comparison_layout,
            'matrix_cols': int(matrix_cols),
            'matrix_margin': int(matrix_margin),
            'output_retention': output_retention,
            'keep_individual_images': output_retention != self.MATRIX_ONLY_OUTPUT,
            'output_mode': output_mode,
            'individual_output_dir': str(individual_output_dir),
            'individual_images_finalized': 0,
            'individual_finalize_failures': 0,
            'effective_prompts': {},
            'effective_negative_prompts': {},
            'embedding_position': embedding_position,
            'unload_models_before_matrix': bool(unload_models_before_matrix),
            'model_unload_attempted': False,
            'model_unloaded': False,
            'session_dir': str(session_dir),
            'manifest_path': str(session_dir / "manifest.json"),
            'cells': {},
            'rows': [],
            'matrix_outputs': [],
            'reference_row': None,
            'reference_rows': [],
            'fixed_seed': None,
            'fixed_subseed': None,
            'requested_batch_count': requested_batch_count,
            'requested_batch_size': requested_batch_size,
            'cells_per_seed_round': cells_per_seed_round,
            'requested_cells_per_model': len(cases),
            'requested_generations': len(cases),
            'seed_round_count': len(seed_rounds) if seed_rounds else 1,
            'seed_rounds': seed_rounds,
        }
        setattr(p, self.PROCESSING_STATE_ATTRIBUTE, state)
        self._write_manifest(state, "generating")

        # This hook runs before setup_prompts(), so Forge creates matching prompt,
        # seed and subseed arrays for every comparison case.
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

        print(
            f"[LoRA Tester] {item_label} mode enabled with "
            f"{len(selected_items)} selected item(s)"
        )
        print(f"[LoRA Tester] Will generate {len(cases)} matrix cells total")
        if requested_batch_size != 1:
            print(
                "[LoRA Tester] Forge Batch Size remains serialized to 1 for "
                "disk-spooling and recovery safety; Batch Count is preserved."
            )

    @staticmethod
    def _snapshot_forge_model_state():
        """Capture Forge's selected model inputs before sequential overrides."""

        from modules import sd_models

        loading_parameters = dict(sd_models.model_data.forge_loading_parameters)
        if isinstance(loading_parameters.get('additional_modules'), list):
            loading_parameters['additional_modules'] = list(
                loading_parameters['additional_modules']
            )
        return {
            'checkpoint': getattr(shared.opts, 'sd_model_checkpoint', None),
            'additional_modules': list(
                getattr(shared.opts, 'forge_additional_modules', []) or []
            ),
            'loading_parameters': loading_parameters,
        }

    @classmethod
    def _restore_forge_model_state(cls, p, snapshot):
        """Restore the exact original Forge selection and reload it for the outer job."""

        from modules import processing as forge_processing, sd_models
        from modules.sysinfo import set_config

        original_settings = {
            'sd_model_checkpoint': snapshot.get('checkpoint'),
            'forge_additional_modules': snapshot.get('additional_modules', []),
        }
        set_config(
            original_settings,
            is_api=True,
            run_callbacks=False,
            save_config=False,
        )
        sd_models.model_data.forge_loading_parameters = dict(
            snapshot.get('loading_parameters') or {}
        )
        forge_processing.need_global_unload = True
        forge_processing.manage_model_and_prompt_cache(p)

    def _persist_multi_model_outputs(self, state, job, processed):
        """Normalize nested Gallery results to validated disk paths."""

        pages = []
        infotexts = []
        source_infotexts = list(getattr(processed, 'infotexts', []) or [])
        fallback_infotext = source_infotexts[0] if source_infotexts else ""
        for output_index, output in enumerate(getattr(processed, 'images', []) or []):
            if isinstance(output, Image.Image):
                destination = Path(state['session_dir']) / (
                    f"model-{len(state.get('model_results', [])):02d}-"
                    f"page-{output_index:03d}.png"
                )
                parameters = (
                    source_infotexts[output_index]
                    if output_index < len(source_infotexts)
                    else fallback_infotext
                )
                self._save_png_atomic(output, destination, parameters=parameters)
                output_path = destination
            else:
                output_path = Path(str(output))

            if not output_path.exists() or output_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"Model '{job['label']}' returned a missing or empty output page: "
                    f"{output_path}"
                )
            if self._is_multi_model_temporary_page(state, output_path):
                temporary_pages = state.setdefault(
                    'multi_model_temporary_pages',
                    [],
                )
                normalized_path = str(output_path.resolve())
                if normalized_path not in temporary_pages:
                    temporary_pages.append(normalized_path)
            pages.append(str(output_path))
            infotexts.append(
                source_infotexts[output_index]
                if output_index < len(source_infotexts)
                else fallback_infotext
            )
        return pages, infotexts

    @staticmethod
    def _is_multi_model_temporary_page(state, page_path):
        """Recognize only extension-owned matrix pages below its spool root."""

        try:
            path = Path(page_path).resolve()
            spool_root = Path(state['session_dir']).resolve().parent
            relative = path.relative_to(spool_root)
        except (KeyError, OSError, ValueError):
            return False
        return (
            len(relative.parts) >= 2
            and path.name.startswith(("matrix-", "model-"))
        )

    def _cleanup_multi_model_sources(self, state, protected_pages):
        """Delete tracked temporary source grids after a validated final overview."""

        protected = {
            str(Path(path).resolve())
            for path in protected_pages or []
        }
        spool_root = Path(state['session_dir']).resolve().parent
        cleanup_complete = True
        deleted = 0
        source_parents = set()
        for source_path in state.get('multi_model_temporary_pages', []):
            source = Path(source_path)
            try:
                resolved = source.resolve()
                resolved.relative_to(spool_root)
            except (OSError, ValueError):
                cleanup_complete = False
                print(
                    "[LoRA Tester] Refusing to remove an untrusted multi-model "
                    f"source path: {source_path!r}"
                )
                continue
            if str(resolved) in protected:
                continue
            source_parents.add(resolved.parent)
            if self._safe_unlink(resolved):
                deleted += 1
            else:
                cleanup_complete = False

        for parent in sorted(
            source_parents,
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            if parent == spool_root:
                continue
            try:
                parent.rmdir()
            except OSError:
                pass

        state['multi_model_source_cleanup_complete'] = cleanup_complete
        if deleted:
            print(
                f"[LoRA Tester] Removed {deleted} temporary per-model page(s) "
                "after validating the shared overview"
            )
        return cleanup_complete

    @classmethod
    def _multi_model_page_geometry(cls, page_records, requested_columns, margin):
        """Return the uniform grid geometry for persisted per-model pages."""

        if not page_records:
            return {
                'columns': 0,
                'rows': 0,
                'cell_width': 0,
                'cell_height': 0,
                'width': 0,
                'height': 0,
            }

        columns = max(1, min(int(requested_columns), len(page_records)))
        rows = (len(page_records) + columns - 1) // columns
        cell_width = max(int(record['width']) for record in page_records)
        cell_height = max(int(record['height']) for record in page_records)
        margin = max(0, int(margin))
        return {
            'columns': columns,
            'rows': rows,
            'cell_width': cell_width,
            'cell_height': cell_height,
            'width': cell_width * columns + margin * max(0, columns - 1),
            'height': cell_height * rows + margin * max(0, rows - 1),
        }

    @classmethod
    def _split_multi_model_overview_records(
        cls,
        page_records,
        requested_columns,
        margin,
    ):
        """Keep model pages together unless the matrix limits require a split."""

        chunks = []
        current = []

        def is_safe(records):
            geometry = cls._multi_model_page_geometry(
                records,
                requested_columns,
                margin,
            )
            return (
                geometry['width'] <= cls.MAX_MATRIX_DIMENSION
                and geometry['height'] <= cls.MAX_MATRIX_DIMENSION
                and geometry['width'] * geometry['height'] <= cls.MAX_MATRIX_PIXELS
            )

        for record in page_records:
            candidate = current + [record]
            if is_safe(candidate):
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = [record]
            if not is_safe(current):
                raise RuntimeError(
                    f"Per-model page {record.get('path')!r} exceeds the matrix limits"
                )

        if current:
            chunks.append(current)
        return chunks

    def _build_multi_model_overview_pages(
        self,
        p,
        processed,
        state,
        source_pages,
        source_infotexts,
    ):
        """Compose persisted per-model grids into shared comparison pages."""

        records = []
        page_metadata = list(state.get('multi_model_page_records', []))
        for page_index, page_path in enumerate(source_pages):
            path = Path(page_path)
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing per-model page: {path}")
            with Image.open(path) as source:
                width, height = source.size
            record = (
                dict(page_metadata[page_index])
                if page_index < len(page_metadata)
                else {}
            )
            record.update(
                {
                    'index': page_index,
                    'path': str(path),
                    'width': int(width),
                    'height': int(height),
                }
            )
            records.append(record)

        overview_order = state.get(
            'multi_model_overview_order',
            self.MULTI_OVERVIEW_MODELS_FIRST,
        )
        seeds_first = overview_order == self.MULTI_OVERVIEW_SEEDS_FIRST

        def integer_order(value, fallback):
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        if seeds_first:
            records.sort(
                key=lambda record: (
                    integer_order(record.get('seed_round'), 1_000_000),
                    integer_order(record.get('model_index'), record['index']),
                    integer_order(record.get('model_page_index'), record['index']),
                )
            )
        else:
            records.sort(
                key=lambda record: (
                    integer_order(record.get('model_index'), record['index']),
                    integer_order(record.get('seed_round'), 1_000_000),
                    integer_order(record.get('model_page_index'), record['index']),
                )
            )

        requested_columns = max(1, int(state.get('matrix_cols', 1)))
        seed_rounds = list(state.get('seed_rounds', []))
        if not seeds_first and len(seed_rounds) > 1 and records:
            pages_per_model_round = {}
            for record in records:
                key = (record.get('model_index'), record.get('seed_round'))
                pages_per_model_round[key] = pages_per_model_round.get(key, 0) + 1
            model_indexes = {record.get('model_index') for record in records}
            expected_rounds = {int(round_info['index']) for round_info in seed_rounds}
            one_page_per_round = all(
                pages_per_model_round.get((model_index, round_index)) == 1
                for model_index in model_indexes
                for round_index in expected_rounds
            )
            if one_page_per_round:
                # Preserve the maintainer's preferred comparison: one model per
                # row, with that model's seed rounds occupying the columns.
                requested_columns = len(seed_rounds)

        background = ImageColor.getcolor(shared.opts.grid_background_color, "RGB")
        margin = max(0, int(state.get('matrix_margin', 0)))
        seed_label_font = None
        seed_label_width = 0
        seed_label_line_spacing = 0

        def unique_seed_round(page_records):
            chunk_rounds = list(
                dict.fromkeys(
                    record.get('seed_round')
                    for record in page_records
                    if record.get('seed_round') is not None
                )
            )
            return chunk_rounds[0] if len(chunk_rounds) == 1 else None

        def seed_row_geometry(rows):
            page_records = [
                record
                for row_info in rows
                for record in row_info['records']
            ]
            columns = max((len(row_info['records']) for row_info in rows), default=0)
            cell_width = max(
                (int(record['width']) for record in page_records),
                default=0,
            )
            cell_height = max(
                (int(record['height']) for record in page_records),
                default=0,
            )
            image_width = (
                cell_width * columns + margin * max(0, columns - 1)
            )
            return {
                'columns': columns,
                'rows': len(rows),
                'cell_width': cell_width,
                'cell_height': cell_height,
                'seed_label_width': seed_label_width,
                'width': seed_label_width + margin + image_width,
                'height': (
                    cell_height * len(rows) + margin * max(0, len(rows) - 1)
                ),
            }

        def geometry_is_safe(geometry):
            return (
                geometry['width'] <= self.MAX_MATRIX_DIMENSION
                and geometry['height'] <= self.MAX_MATRIX_DIMENSION
                and geometry['width'] * geometry['height'] <= self.MAX_MATRIX_PIXELS
            )

        chunk_records = []
        has_seed_metadata = any(
            record.get('seed_round') is not None for record in records
        )
        if seeds_first and has_seed_metadata:
            round_indexes = list(
                dict.fromkeys(record.get('seed_round') for record in records)
            )
            seed_rows = []
            for round_index in round_indexes:
                round_records = [
                    record
                    for record in records
                    if record.get('seed_round') == round_index
                ]
                matching_round = next(
                    (
                        round_info
                        for round_info in seed_rounds
                        if int(round_info['index']) == int(round_index)
                    ),
                    None,
                )
                if matching_round is not None:
                    label = str(matching_round['label'])
                elif round_records and round_records[0].get('seed') is not None:
                    label = f"Seed {int(round_records[0]['seed'])}"
                else:
                    label = f"Seed round {integer_order(round_index, 0) + 1}"
                seed_rows.append(
                    {
                        'seed_round': round_index,
                        'label': label,
                        'records': round_records,
                    }
                )

            cell_width = max(int(record['width']) for record in records)
            cell_height = max(int(record['height']) for record in records)
            font_size = max(
                self.LABEL_MIN_FONT_SIZE,
                (cell_width + cell_height) // self.LABEL_FONT_DIVISOR,
            )
            seed_label_font = images.get_font(font_size)
            seed_label_line_spacing = max(2, font_size // 5)
            label_padding = max(8, font_size // 2)
            scratch = Image.new("RGB", (1, 1), background)
            try:
                calculator = ImageDraw.Draw(scratch)
                maximum_label_width = 0
                for row_info in seed_rows:
                    row_info['label_lines'] = str(row_info['label']).replace(
                        " / ",
                        "\n",
                    ).splitlines()
                    for line in row_info['label_lines']:
                        bbox = calculator.textbbox((0, 0), line, font=seed_label_font)
                        maximum_label_width = max(
                            maximum_label_width,
                            bbox[2] - bbox[0],
                        )
                seed_label_width = maximum_label_width + label_padding * 2
            finally:
                scratch.close()

            maximum_row_columns = max(
                (len(row_info['records']) for row_info in seed_rows),
                default=0,
            )
            safe_row_columns = 0
            for column_count in range(1, maximum_row_columns + 1):
                candidate_width = (
                    seed_label_width
                    + margin
                    + cell_width * column_count
                    + margin * max(0, column_count - 1)
                )
                if (
                    candidate_width <= self.MAX_MATRIX_DIMENSION
                    and cell_height <= self.MAX_MATRIX_DIMENSION
                    and candidate_width * cell_height <= self.MAX_MATRIX_PIXELS
                ):
                    safe_row_columns = column_count
                else:
                    break
            if safe_row_columns <= 0:
                raise RuntimeError(
                    "A seed-first model row exceeds the matrix limits even with "
                    "one model page"
                )

            row_segments = []
            for row_info in seed_rows:
                for start in range(0, len(row_info['records']), safe_row_columns):
                    row_segments.append(
                        {
                            **row_info,
                            'records': row_info['records'][
                                start:start + safe_row_columns
                            ],
                        }
                    )

            page_rows = []
            for row_info in row_segments:
                candidate_rows = page_rows + [row_info]
                if geometry_is_safe(seed_row_geometry(candidate_rows)):
                    page_rows = candidate_rows
                    continue
                if page_rows:
                    page_records = [
                        record
                        for page_row in page_rows
                        for record in page_row['records']
                    ]
                    chunk_records.append(
                        {
                            'seed_round': unique_seed_round(page_records),
                            'records': page_records,
                            'seed_rows': page_rows,
                        }
                    )
                page_rows = [row_info]
                if not geometry_is_safe(seed_row_geometry(page_rows)):
                    raise RuntimeError(
                        "A seed-first model row exceeds the matrix limits"
                    )
            if page_rows:
                page_records = [
                    record
                    for page_row in page_rows
                    for record in page_row['records']
                ]
                chunk_records.append(
                    {
                        'seed_round': unique_seed_round(page_records),
                        'records': page_records,
                        'seed_rows': page_rows,
                    }
                )
        else:
            for chunk in self._split_multi_model_overview_records(
                records,
                requested_columns,
                margin,
            ):
                chunk_records.append(
                    {
                        'seed_round': unique_seed_round(chunk),
                        'records': chunk,
                    }
                )

        model_labels = [
            result.get('label') or result.get('title') or result.get('checkpoint')
            for result in state.get('model_results', [])
            if result.get('status') == 'complete'
        ]
        model_summary = ", ".join(str(label) for label in model_labels if label)
        base_infotext = source_infotexts[0] if source_infotexts else ""

        page_paths = []
        page_infotexts = []
        state['multi_model_overview_metadata'] = []
        for page_number, chunk_info in enumerate(chunk_records, start=1):
            chunk = chunk_info['records']
            seed_rows = chunk_info.get('seed_rows')
            geometry = (
                seed_row_geometry(seed_rows)
                if seed_rows is not None
                else self._multi_model_page_geometry(
                    chunk,
                    requested_columns,
                    margin,
                )
            )
            page_seeds = list(dict.fromkeys(
                int(record['seed'])
                for record in chunk
                if record.get('seed') is not None
            ))
            page_subseeds = list(dict.fromkeys(
                int(record['subseed'])
                for record in chunk
                if record.get('subseed') is not None
            ))
            page_seed = page_seeds[0] if len(page_seeds) == 1 else -1
            page_subseed = page_subseeds[0] if len(page_subseeds) == 1 else -1
            source_index = integer_order(chunk[0].get('index'), -1)
            if 0 <= source_index < len(source_infotexts):
                page_info = source_infotexts[source_index]
            else:
                page_info = base_infotext
            page_info = page_info or "LoRA Tester multi-model comparison"
            if model_summary:
                page_info += f"\nLoRA Tester models: {model_summary}"
            page_info += f"\nLoRA Tester overview: {overview_order}"
            if chunk_info['seed_round'] is not None:
                matching_round = next(
                    (
                        round_info
                        for round_info in seed_rounds
                        if int(round_info['index'])
                        == int(chunk_info['seed_round'])
                    ),
                    None,
                )
                if matching_round is not None:
                    page_info += f"\nLoRA Tester seed round: {matching_round['label']}"
            if page_seeds:
                page_info += "\nLoRA Tester page seeds: " + ", ".join(
                    str(seed) for seed in page_seeds
                )
                if page_seed == -1:
                    page_info += "\nLoRA Tester page seed metadata: mixed (-1)"
            if page_subseeds:
                page_info += "\nLoRA Tester page subseeds: " + ", ".join(
                    str(subseed) for subseed in page_subseeds
                )

            canvas = Image.new(
                "RGB",
                (geometry['width'], geometry['height']),
                background,
            )
            try:
                if seed_rows is not None:
                    text_color = ImageColor.getcolor(
                        getattr(shared.opts, 'grid_text_active_color', '#000000'),
                        "RGB",
                    )
                    drawing = ImageDraw.Draw(canvas)
                    for row, row_info in enumerate(seed_rows):
                        row_y = row * (geometry['cell_height'] + margin)
                        label_lines = row_info.get('label_lines') or [
                            str(row_info['label'])
                        ]
                        line_boxes = [
                            drawing.textbbox((0, 0), line, font=seed_label_font)
                            for line in label_lines
                        ]
                        line_heights = [
                            max(1, bbox[3] - bbox[1]) for bbox in line_boxes
                        ]
                        block_height = sum(line_heights) + seed_label_line_spacing * max(
                            0,
                            len(label_lines) - 1,
                        )
                        draw_y = row_y + (geometry['cell_height'] - block_height) // 2
                        for line, bbox, line_height in zip(
                            label_lines,
                            line_boxes,
                            line_heights,
                        ):
                            text_width = bbox[2] - bbox[0]
                            drawing.text(
                                (
                                    (seed_label_width - text_width) / 2 - bbox[0],
                                    draw_y - bbox[1],
                                ),
                                line,
                                font=seed_label_font,
                                fill=text_color,
                            )
                            draw_y += line_height + seed_label_line_spacing

                        for column, record in enumerate(row_info['records']):
                            with Image.open(record['path']) as source:
                                source.load()
                                converted = (
                                    source
                                    if source.mode == "RGB"
                                    else source.convert("RGB")
                                )
                                x = (
                                    seed_label_width
                                    + margin
                                    + column * (geometry['cell_width'] + margin)
                                    + (geometry['cell_width'] - converted.width) // 2
                                )
                                y = (
                                    row_y
                                    + (geometry['cell_height'] - converted.height) // 2
                                )
                                canvas.paste(converted, (x, y))
                                if converted is not source:
                                    converted.close()
                else:
                    for chunk_index, record in enumerate(chunk):
                        column = chunk_index % geometry['columns']
                        row = chunk_index // geometry['columns']
                        with Image.open(record['path']) as source:
                            source.load()
                            converted = (
                                source
                                if source.mode == "RGB"
                                else source.convert("RGB")
                            )
                            x = (
                                column * (geometry['cell_width'] + margin)
                                + (geometry['cell_width'] - converted.width) // 2
                            )
                            y = (
                                row * (geometry['cell_height'] + margin)
                                + (geometry['cell_height'] - converted.height) // 2
                            )
                            canvas.paste(converted, (x, y))
                            if converted is not source:
                                converted.close()

                if getattr(shared.opts, 'grid_save', False):
                    prompt_values = list(getattr(processed, 'all_prompts', []) or [])
                    session_name = Path(state['session_dir']).name
                    page_path, _ = images.save_image(
                        canvas,
                        p.outpath_grids,
                        "lora_tester_multi_model_overview",
                        seed=page_seed,
                        prompt=(
                            prompt_values[0]
                            if prompt_values
                            else str(p.prompt)
                        ),
                        extension=shared.opts.grid_format,
                        info=page_info,
                        grid=True,
                        p=p,
                        suffix=f"-page-{page_number:03d}",
                        forced_filename=(
                            f"lora_tester_grid-{session_name}-multi-model-overview-"
                            f"page-{page_number:03d}"
                        ),
                        save_to_dirs=False,
                    )
                else:
                    page_path = Path(state['session_dir']) / (
                        f"matrix-multi-model-overview-page-{page_number:03d}.png"
                    )
                    parameters = (
                        page_info
                        if getattr(shared.opts, 'enable_pnginfo', False)
                        else None
                    )
                    self._save_png_atomic(canvas, page_path, parameters=parameters)
                page_paths.append(str(page_path))
                page_infotexts.append(page_info)
                state['multi_model_overview_metadata'].append(
                    {
                        'path': str(page_path),
                        'seed_round': chunk_info['seed_round'],
                        'seed': page_seed,
                        'subseed': page_subseed,
                        'seeds': page_seeds,
                        'subseeds': page_subseeds,
                        'source_page_count': len(chunk),
                        'layout': 'seed-rows' if seed_rows is not None else 'grid',
                        'columns': geometry['columns'],
                        'rows': geometry['rows'],
                        'seed_label_width': geometry.get('seed_label_width', 0),
                    }
                )
            finally:
                canvas.close()

        self._validate_matrix_pages(page_paths)
        print(
            f"[LoRA Tester] Combined {len(source_pages)} per-model page(s) into "
            f"{len(page_paths)} shared overview page(s) using {overview_order}"
        )
        return page_paths, page_infotexts

    def _run_multi_model_jobs(self, p, state, arguments):
        """Run the normal single-model tester once per checkpoint, strictly serially."""

        from modules import processing as forge_processing

        snapshot = self._snapshot_forge_model_state()
        state['original_checkpoint'] = snapshot.get('checkpoint')
        fixed_seed = int(p.all_seeds[0]) if p.all_seeds else int(p.seed)
        fixed_subseed = (
            int(p.all_subseeds[0]) if p.all_subseeds else int(p.subseed)
        )
        state['fixed_seed'] = fixed_seed
        state['fixed_subseed'] = fixed_subseed
        seed_rounds = self._batch_seed_rounds(
            fixed_seed,
            fixed_subseed,
            state.get('requested_batch_count', 1),
            getattr(p, 'subseed_strength', 0.0),
        )
        state['seed_rounds'] = seed_rounds
        total_progress_jobs = max(1, int(state.get('requested_generations', 1)))
        state['progress_job_count'] = total_progress_jobs
        # The nested process_images() calls share Forge's one outer State. Seed the
        # complete job total before the first nested job so job_no can advance across
        # every checkpoint without the progress bar exceeding 100 percent.
        shared.state.job_count = total_progress_jobs
        self._write_manifest(state, "models-running")

        try:
            for model_index, job in enumerate(state['model_jobs']):
                if shared.state.interrupted or shared.state.stopping_generation:
                    break

                shared.state.job = (
                    f"LoRA Tester model {model_index + 1} of "
                    f"{len(state['model_jobs'])}: {job['label']}"
                )
                result = {
                    'checkpoint': job['checkpoint'],
                    'title': job['title'],
                    'label': job['label'],
                    'status': 'running',
                    'pages': [],
                }
                unload_succeeded = False
                job_p = copy.copy(p)
                job_p.styles = list(getattr(p, 'styles', []) or [])
                job_p.override_settings = dict(
                    getattr(p, 'override_settings', {}) or {}
                )
                job_p.override_settings['sd_model_checkpoint'] = job['checkpoint']
                job_p.override_settings_restore_afterwards = True
                job_p.n_iter = len(seed_rounds)
                job_p.batch_size = 1
                job_p.seed = fixed_seed
                job_p.subseed = fixed_subseed
                setattr(job_p, self.BATCH_MODEL_LABEL_ATTRIBUTE, job['label'])
                setattr(job_p, self.BATCH_SEED_ROUNDS_ATTRIBUTE, seed_rounds)

                nested_script_args = list(p.script_args)
                nested_arguments = self._prepare_multi_model_job_arguments(
                    arguments,
                    state['multi_model_mode'],
                )
                nested_script_args[self.args_from:self.args_to] = nested_arguments
                job_p.script_args = nested_script_args

                try:
                    print(
                        f"[LoRA Tester] Loading model {model_index + 1}/"
                        f"{len(state['model_jobs'])}: {job['title']}"
                    )
                    nested_processed = forge_processing.process_images(job_p)
                    pages, infotexts = self._persist_multi_model_outputs(
                        state,
                        job,
                        nested_processed,
                    )
                    if not pages:
                        raise RuntimeError(
                            f"Model '{job['label']}' completed without output pages"
                        )

                    state['multi_model_pages'].extend(pages)
                    state['multi_model_infotexts'].extend(infotexts)
                    page_count = len(pages)
                    source_prompts = list(
                        getattr(nested_processed, 'all_prompts', []) or []
                    )
                    source_negative_prompts = list(
                        getattr(nested_processed, 'all_negative_prompts', []) or []
                    )
                    source_seeds = list(
                        getattr(nested_processed, 'all_seeds', []) or []
                    )
                    source_subseeds = list(
                        getattr(nested_processed, 'all_subseeds', []) or []
                    )
                    page_prompts = (
                        source_prompts[:page_count]
                        if len(source_prompts) >= page_count
                        else [source_prompts[0] if source_prompts else str(p.prompt)]
                        * page_count
                    )
                    page_negative_prompts = (
                        source_negative_prompts[:page_count]
                        if len(source_negative_prompts) >= page_count
                        else [
                            source_negative_prompts[0]
                            if source_negative_prompts
                            else str(p.negative_prompt)
                        ]
                        * page_count
                    )
                    page_seeds = (
                        source_seeds[:page_count]
                        if len(source_seeds) >= page_count
                        else [source_seeds[0] if source_seeds else fixed_seed]
                        * page_count
                    )
                    page_subseeds = (
                        source_subseeds[:page_count]
                        if len(source_subseeds) >= page_count
                        else [
                            source_subseeds[0]
                            if source_subseeds
                            else fixed_subseed
                        ]
                        * page_count
                    )
                    state['multi_model_prompts'].extend(page_prompts)
                    state['multi_model_negative_prompts'].extend(
                        page_negative_prompts
                    )
                    state['multi_model_seeds'].extend(page_seeds)
                    state['multi_model_subseeds'].extend(page_subseeds)

                    for page_index, page_path in enumerate(pages):
                        page_seed = int(page_seeds[page_index])
                        page_subseed = int(page_subseeds[page_index])
                        matching_round = next(
                            (
                                round_info
                                for round_info in seed_rounds
                                if int(round_info['seed']) == page_seed
                                and int(round_info['subseed']) == page_subseed
                            ),
                            None,
                        )
                        state.setdefault('multi_model_page_records', []).append(
                            {
                                'path': str(page_path),
                                'infotext': infotexts[page_index],
                                'model_index': model_index,
                                'model_label': job['label'],
                                'model_page_index': page_index,
                                'seed_round': (
                                    int(matching_round['index'])
                                    if matching_round is not None
                                    else None
                                ),
                                'seed': page_seed,
                                'subseed': page_subseed,
                                'prompt': page_prompts[page_index],
                                'negative_prompt': page_negative_prompts[page_index],
                            }
                        )
                    result['status'] = 'complete'
                    result['pages'] = pages
                except Exception as error:
                    result['status'] = 'failed'
                    result['error'] = str(error)
                    print(
                        f"[LoRA Tester] Model job failed for {job['title']}: {error}"
                    )
                finally:
                    state['model_unload_attempted'] = True
                    try:
                        self._full_forge_model_unload(job_p)
                        state['model_unloaded'] = True
                        result['unloaded'] = True
                        unload_succeeded = True
                    except Exception as unload_error:
                        state['model_unloaded'] = False
                        result['unloaded'] = False
                        result['unload_error'] = str(unload_error)
                        if result['status'] == 'complete':
                            result['status'] = 'unload-failed'
                        print(
                            f"[LoRA Tester] Warning: complete unload failed after "
                            f"{job['title']}: {unload_error}"
                        )

                state['model_results'].append(result)
                self._write_manifest(state, "models-running")
                if not unload_succeeded:
                    print(
                        "[LoRA Tester] Stopping the multiple-model run because "
                        "the previous checkpoint could not be confirmed as unloaded"
                    )
                    break
        finally:
            try:
                self._restore_forge_model_state(p, snapshot)
                state['original_model_restored'] = True
                print(
                    "[LoRA Tester] Restored the original Forge model selection: "
                    f"{state.get('original_checkpoint')}"
                )
            except Exception as restore_error:
                state['original_model_restore_error'] = str(restore_error)
                print(
                    "[LoRA Tester] ERROR: could not restore the original Forge "
                    f"model selection: {restore_error}"
                )
            finally:
                # The outer processing object is only a result-delivery shell. Its
                # already-expanded prompt/seed arrays remain valid for Processed,
                # while zero iterations prevent an accidental extra generation.
                p.n_iter = 0
                p.batch_size = 1
                self._write_manifest(state, "models-finished")

    def process(self, p: StableDiffusionProcessing, *args):
        """Use one fixed seed so every item/weight comparison is meaningful."""
        if not args or not args[0]:
            return

        state = getattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)
        if not state:
            return

        if state.get('multi_model'):
            self._run_multi_model_jobs(p, state, args)
            return

        explicit_seeds = [case.get('seed') for case in state.get('cases', [])]
        if explicit_seeds and all(seed is not None for seed in explicit_seeds):
            state['fixed_seed'] = int(explicit_seeds[0])
            p.all_seeds = [int(seed) for seed in explicit_seeds]
        elif p.all_seeds:
            state['fixed_seed'] = int(p.all_seeds[0])
            p.all_seeds = [state['fixed_seed']] * len(p.all_prompts)

        explicit_subseeds = [
            case.get('subseed') for case in state.get('cases', [])
        ]
        if explicit_subseeds and all(
            subseed is not None for subseed in explicit_subseeds
        ):
            state['fixed_subseed'] = int(explicit_subseeds[0])
            p.all_subseeds = [int(subseed) for subseed in explicit_subseeds]
        elif p.all_subseeds:
            state['fixed_subseed'] = int(p.all_subseeds[0])
            p.all_subseeds = [state['fixed_subseed']] * len(p.all_prompts)

        self._write_manifest(state, "seed-fixed")
        if int(state.get('seed_round_count', 1)) > 1:
            seed_labels = ", ".join(
                str(round_info.get('label'))
                for round_info in state.get('seed_rounds', [])
            )
            print(f"[LoRA Tester] Fixed comparison seed rounds: {seed_labels}")
        elif state.get('fixed_seed') is not None:
            print(f"[LoRA Tester] Fixed comparison seed: {state['fixed_seed']}")

    @staticmethod
    def _force_batch_rng_seed(p, state, batch_number, seeds, subseeds):
        """Replace Forge's per-iteration RNG with the current comparison seed."""

        if state.get('fixed_seed') is None:
            source_seeds = getattr(p, 'all_seeds', None) or seeds
            if not source_seeds:
                return
            state['fixed_seed'] = int(source_seeds[0])

        case = (
            state.get('cases', [])[batch_number]
            if batch_number < len(state.get('cases', []))
            else {}
        )
        fixed_seed = int(case.get('seed', state['fixed_seed']))
        if seeds is None:
            seeds = []
        seeds[:] = [fixed_seed] * len(seeds)
        p.seeds = seeds

        fixed_subseed = case.get('subseed', state.get('fixed_subseed'))
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

    @staticmethod
    def _suppress_multi_model_coordinator_batch(state, prompts):
        """Prevent the result-delivery shell from entering Forge's sampler."""

        if not state.get('multi_model'):
            return False
        if prompts is not None:
            prompts.clear()
        state['outer_sampling_suppressed'] = True
        return True

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

        # p.n_iter is set to zero after the nested model jobs, but another
        # always-on script may still change it. An empty prompt list is Forge's
        # supported per-batch stop path and guarantees that the outer coordinator
        # only delivers the pages already produced by the nested jobs.
        if self._suppress_multi_model_coordinator_batch(state, prompts):
            return

        if not prompts:
            return

        if batch_number >= len(state['cases']):
            return

        # Forge constructs p.rng immediately before this hook. Replacing only
        # all_seeds in process() is therefore not sufficient if another script or
        # model path changes the per-iteration lists afterwards. Rebuild the RNG
        # here, before the sampler consumes its first noise tensor.
        self._force_batch_rng_seed(p, state, batch_number, seeds, subseeds)

        case = state['cases'][batch_number]
        if self._case_is_baseline(case):
            state['effective_prompts'][batch_number] = prompts[0]
            negative_prompts = getattr(p, 'negative_prompts', []) or []
            if negative_prompts:
                state['effective_negative_prompts'][batch_number] = negative_prompts[0]
            print(
                f"[LoRA Tester] Batch {batch_number}: "
                f"Baseline (no {state.get('test_type', 'LoRA')})"
            )
            return

        if case.get('kind') == "prompt":
            state['effective_prompts'][batch_number] = prompts[0]
            negative_prompts = getattr(p, 'negative_prompts', []) or []
            if negative_prompts:
                state['effective_negative_prompts'][batch_number] = negative_prompts[0]
            print(
                f"[LoRA Tester] Batch {batch_number}: {case['label']} -> "
                "Prompt only (no LoRA injection)"
            )
            return

        if case.get('kind') == "embedding":
            target_negative = (
                case.get('embedding_target') == self.EMBEDDING_TARGET_NEGATIVE
            )
            target_prompts = (
                getattr(p, 'negative_prompts', []) if target_negative else prompts
            )
            if not target_prompts:
                self._block_generation(
                    p,
                    "The selected embedding prompt target is unavailable for this batch.",
                )
                prompts.clear()
                return

            for index in range(len(target_prompts)):
                target_prompts[index] = self._compose_embedding_prompt(
                    target_prompts[index],
                    case['embedding_name'],
                    case.get('trigger_words', []),
                    case['weight'],
                    state.get('embedding_position'),
                )

            start = batch_number * int(getattr(p, 'batch_size', 1))
            end = start + len(target_prompts)
            all_prompt_attribute = (
                'all_negative_prompts' if target_negative else 'all_prompts'
            )
            all_target_prompts = getattr(p, all_prompt_attribute, None)
            if all_target_prompts is not None:
                all_target_prompts[start:end] = target_prompts

            high_resolution_attribute = (
                'all_hr_negative_prompts' if target_negative else 'all_hr_prompts'
            )
            high_resolution_prompts = getattr(
                p, high_resolution_attribute, None
            )
            if high_resolution_prompts is not None:
                for index in range(start, min(end, len(high_resolution_prompts))):
                    high_resolution_prompts[index] = self._compose_embedding_prompt(
                        high_resolution_prompts[index],
                        case['embedding_name'],
                        case.get('trigger_words', []),
                        case['weight'],
                        state.get('embedding_position'),
                    )

            if target_negative:
                state['effective_prompts'][batch_number] = prompts[0]
                state['effective_negative_prompts'][batch_number] = target_prompts[0]
            else:
                state['effective_prompts'][batch_number] = target_prompts[0]
                negative_prompts = getattr(p, 'negative_prompts', []) or []
                if negative_prompts:
                    state['effective_negative_prompts'][batch_number] = (
                        negative_prompts[0]
                    )

            print(
                f"[LoRA Tester] Batch {batch_number}: {case['label']} -> "
                f"Embedding '{case['embedding_name']}' in "
                f"{'negative' if target_negative else 'positive'} prompt; "
                f"triggers: {case.get('trigger_words', [])}; "
                f"position: {state.get('embedding_position', 'Start')}"
            )
            print(f"[LoRA Tester] Injected prompt: {target_prompts[0]}")
            return

        lora_tag_name = case['lora_tag_name']

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
                'item_key': case.get('item_key'),
                'item_label': case.get('item_label'),
                'trigger_words': list(case.get('trigger_words', [])),
                'weight': case.get('weight'),
                'kind': case.get('kind'),
                'seed_round': case.get('seed_round', 0),
                'seed': case.get('seed'),
                'subseed': case.get('subseed'),
                'is_baseline': self._case_is_baseline(case),
                'width': int(original.width),
                'height': int(original.height),
                'persistent': False,
            }
            self._write_manifest(state, "generating")
            saved = True
        except Exception as error:
            print(f"[LoRA Tester] Could not spool cell {case_index + 1}: {error}")

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

    def _postprocess_multi_model(self, p, processed, state):
        """Combine persisted per-model pages and deliver them through the outer job."""

        source_pages = list(state.get('multi_model_pages', []))
        source_infotexts = list(state.get('multi_model_infotexts', []))
        pages = source_pages
        infotexts = source_infotexts
        if source_pages:
            try:
                pages, infotexts = self._build_multi_model_overview_pages(
                    p,
                    processed,
                    state,
                    source_pages,
                    source_infotexts,
                )
                state['multi_model_overview_pages'] = list(pages)
            except Exception as overview_error:
                state['multi_model_overview_error'] = str(overview_error)
                processed.comments += (
                    "\nLoRA Tester: the shared multi-model grid could not be created; "
                    "the completed per-model grids were retained."
                )
                print(
                    "[LoRA Tester] Warning: shared multi-model grid creation failed: "
                    f"{overview_error}"
                )
        if len(infotexts) < len(pages):
            infotexts.extend(
                [infotexts[0] if infotexts else ""]
                * (len(pages) - len(infotexts))
            )

        processed.images = pages
        processed.extra_images = []
        if hasattr(p, 'extra_result_images'):
            p.extra_result_images = []
        processed.infotexts = infotexts
        processed.index_of_first_image = len(pages)
        prompt_values = list(state.get('multi_model_prompts', []))
        negative_values = list(state.get('multi_model_negative_prompts', []))
        seed_values = list(state.get('multi_model_seeds', []))
        subseed_values = list(state.get('multi_model_subseeds', []))
        overview_metadata = list(
            state.get('multi_model_overview_metadata', [])
        )

        def page_values(values, fallback):
            if len(values) >= len(pages):
                return values[:len(pages)]
            return [values[0] if values else fallback] * len(pages)

        processed.all_prompts = page_values(prompt_values, str(p.prompt))
        processed.all_negative_prompts = page_values(
            negative_values,
            str(p.negative_prompt),
        )
        if len(overview_metadata) == len(pages):
            processed.all_seeds = [
                int(metadata.get('seed', -1)) for metadata in overview_metadata
            ]
            processed.all_subseeds = [
                int(metadata.get('subseed', -1)) for metadata in overview_metadata
            ]
        else:
            processed.all_seeds = page_values(
                seed_values,
                state.get('fixed_seed', -1),
            )
            processed.all_subseeds = page_values(
                subseed_values,
                state.get('fixed_subseed', -1),
            )

        failures = [
            result
            for result in state.get('model_results', [])
            if result.get('status') != 'complete'
        ]
        interrupted = bool(
            shared.state.interrupted or shared.state.stopping_generation
        )
        restore_error = state.get('original_model_restore_error')
        overview_error = state.get('multi_model_overview_error')
        if failures:
            processed.comments += (
                f"\nLoRA Tester: {len(failures)} model job(s) failed; "
                "completed model pages were retained."
            )
        if interrupted:
            processed.comments += (
                "\nLoRA Tester: the multiple-model run was interrupted; "
                "completed model pages were retained."
            )
        if restore_error:
            processed.comments += (
                "\nLoRA Tester: restoring the original Forge model failed: "
                f"{restore_error}"
            )

        complete_count = sum(
            result.get('status') == 'complete'
            for result in state.get('model_results', [])
        )
        all_complete = (
            complete_count == len(state.get('model_jobs', []))
            and not failures
            and not interrupted
            and state.get('original_model_restored')
            and not overview_error
        )
        final_status = (
            "multi-model-complete" if all_complete else "multi-model-partial"
        )
        self._write_manifest(state, final_status)

        manifest_path = state.get('manifest_path')
        session_dir = Path(state['session_dir'])
        if all_complete:
            cleanup_complete = self._cleanup_multi_model_sources(state, pages)
            if cleanup_complete:
                self._safe_unlink(manifest_path)
                try:
                    session_dir.rmdir()
                except OSError:
                    pass
            else:
                state['cleanup_incomplete'] = True
                self._write_manifest(state, "multi-model-cleanup-incomplete")
                processed.comments += (
                    "\nLoRA Tester: the shared overview is complete, but one or more "
                    "temporary per-model pages could not be removed."
                )
        else:
            print(
                f"[LoRA Tester] Multiple-model recovery manifest retained in "
                f"{session_dir}"
            )

        print(
            f"[LoRA Tester] Multiple-model run delivered {len(pages)} page(s) "
            f"from {complete_count}/{len(state.get('model_jobs', []))} model(s)"
        )
        self._ensure_temp_directory()
        setattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)
        gc.collect()

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed,
                   lora_tester_enabled, test_type,
                   lora_selection, global_weight_spec,
                   use_trigger_words, trigger_position, lora_settings,
                   embedding_selection, global_embedding_weight_spec,
                   embedding_position, embedding_settings,
                   save_original, per_item_grids, combined_overview,
                   comparison_layout, extreme_run_mode, draw_legend,
                   matrix_cols, matrix_margin,
                   output_retention,
                   unload_models_before_matrix,
                   multi_model_mode=None, checkpoint_selection=None,
                   multi_model_overview_order=None):
        """Build safety-bounded matrix pages from disk-spooled generation cells."""

        state = getattr(p, self.PROCESSING_STATE_ATTRIBUTE, None)
        if not lora_tester_enabled or not state:
            return

        if state.get('multi_model'):
            self._postprocess_multi_model(p, processed, state)
            return

        completed = False
        print(f"[LoRA Tester] Post-processing {len(state.get('cells', {}))} spooled cells")

        try:
            self._spool_fallback_images(p, state, processed)
            cells = [state['cells'][index] for index in sorted(state['cells'])]
            generation_interrupted = bool(
                getattr(shared.state, 'interrupted', False)
            )
            delete_allowed, retention_reason = self._matrix_only_cleanup_decision(
                state,
                len(cells),
                generation_interrupted,
            )
            state['generation_interrupted'] = generation_interrupted
            state['matrix_only_delete_allowed'] = delete_allowed
            state['matrix_only_retention_reason'] = retention_reason
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

            if int(state.get('seed_round_count', 1)) > 1:
                print(
                    f"[LoRA Tester] Preparing independent matrices for "
                    f"{state['seed_round_count']} Forge Batch Count seed rounds"
                )
                page_paths, page_seeds, page_subseeds = (
                    self._build_seed_round_matrix_pages(
                        state,
                        cells,
                        p,
                        processed,
                        grid_info,
                    )
                )
                processed.all_seeds = page_seeds
                processed.all_subseeds = page_subseeds
            else:
                reference_cell = next(
                    (cell for cell in cells if cell.get('is_baseline')),
                    None,
                )
                comparison_cells = [
                    cell for cell in cells if cell is not reference_cell
                ]
                reference_row = None
                if reference_cell is not None:
                    reference_rows = self._build_row_strips(
                        state,
                        [reference_cell],
                        columns=1,
                        filename_prefix="reference-row",
                        track_as_matrix_rows=False,
                    )
                    reference_row = reference_rows[0]
                    state['reference_row'] = reference_row

                print(
                    f"[LoRA Tester] Preparing requested matrices for "
                    f"{len(self._group_comparison_cells(comparison_cells))} tested item(s), "
                    f"{state['matrix_cols']} standard-overview columns; reference image: "
                    f"{'repeated on every page' if reference_row else 'disabled'}"
                )
                page_paths = self._build_requested_matrices(
                    state,
                    comparison_cells,
                    p,
                    processed,
                    grid_info,
                    reference_row=reference_row,
                )
            self._validate_matrix_pages(page_paths)

            processed.images = [str(path) for path in page_paths]
            processed.extra_images = []
            if hasattr(p, 'extra_result_images'):
                p.extra_result_images = []
            processed.infotexts = [grid_info] * len(page_paths)
            # Every gallery entry is a matrix/grid page; there are no samples
            # after this index for Forge's single-image actions to target.
            processed.index_of_first_image = len(page_paths)
            if retention_reason and not state.get('keep_individual_images'):
                processed.comments += (
                    "\nLoRA Tester recovery: individual source images were retained "
                    f"because {retention_reason}."
                )

            self._write_manifest(state, "complete")
            self._cleanup_spool_files(state, page_paths)
            completed = True
            print(
                f"[LoRA Tester] Matrix creation complete: {len(page_paths)} page(s), "
                f"{len(cells)} cell(s)"
            )
        except Exception as error:
            self._write_manifest(state, "matrix-error")
            reference_rows = list(state.get('reference_rows', []))
            if not reference_rows and state.get('reference_row'):
                reference_rows = [state['reference_row']]
            fallback_paths = [
                row['path']
                for row in reference_rows
                if row.get('path') and Path(row['path']).exists()
            ]
            fallback_paths.extend(
                [
                    row['path']
                    for row in state.get('rows', [])
                    if Path(row['path']).exists()
                ]
            )
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
                'item_key': case.get('item_key'),
                'item_label': case.get('item_label'),
                'trigger_words': list(case.get('trigger_words', [])),
                'weight': case.get('weight'),
                'kind': case.get('kind'),
                'seed_round': case.get('seed_round', 0),
                'seed': case.get('seed'),
                'subseed': case.get('subseed'),
                'is_baseline': self._case_is_baseline(case),
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
    def _group_comparison_cells(cells):
        """Keep tested items in generation order while collecting their weights."""

        groups = []
        groups_by_key = {}
        for cell in cells or []:
            if (
                cell.get('is_baseline')
                or cell.get('kind') == "baseline"
                or ('item_key' in cell and cell.get('item_key') is None)
            ):
                continue
            item_key = cell.get('item_key')
            if item_key is None:
                item_key = f"legacy-cell-{cell.get('index', len(groups))}"
            item_key = str(item_key)
            group = groups_by_key.get(item_key)
            if group is None:
                group = {
                    'item_key': item_key,
                    'item_label': str(cell.get('item_label') or cell.get('label') or item_key),
                    'cells': [],
                }
                groups_by_key[item_key] = group
                groups.append(group)
            group['cells'].append(cell)
        return groups

    @classmethod
    def _comparison_group_error(cls, groups):
        if len(groups) > cls.MAX_COMPARISON_ITEMS:
            return (
                f"Comparison matrix layouts support at most "
                f"{cls.MAX_COMPARISON_ITEMS} LoRAs or Embeddings; "
                f"{len(groups)} were prepared. Use Standard order or reduce the selection."
            )
        if not groups:
            return "No comparison items are available for the selected matrix layout."

        expected_weights = tuple(
            str(cell.get('weight')) for cell in groups[0]['cells']
        )
        for group in groups[1:]:
            weights = tuple(str(cell.get('weight')) for cell in group['cells'])
            if weights != expected_weights:
                return (
                    "Comparison matrix layouts require the same weight values in the "
                    "same order for every tested item. Remove per-item range differences "
                    "or use Standard order."
                )
        return None

    @classmethod
    def _matrix_output_slug(cls, test_type, item_label, ordinal):
        kind = cls._safe_filename_part(str(test_type or "item").casefold(), limit=16)
        label = cls._safe_filename_part(item_label, limit=48)
        label = '-'.join(label.split())
        return f"{kind}-{int(ordinal):03d}-{label}"

    def _build_comparison_matrix_pages(
        self,
        state,
        groups,
        p,
        processed,
        grid_info,
        reference_row=None,
        output_slug_prefix=None,
    ):
        """Build aligned comparison pages without splitting an item/weight axis."""

        error = self._comparison_group_error(groups)
        if error:
            raise RuntimeError(error)
        orientation = state.get('comparison_layout', self.MATRIX_LAYOUT_STANDARD)
        all_cells = [cell for group in groups for cell in group['cells']]
        cell_width = max(int(cell['width']) for cell in all_cells)
        cell_height = max(int(cell['height']) for cell in all_cells)
        capacity = self._comparison_capacity(
            cell_width,
            cell_height,
            len(groups),
            orientation,
            margin=state.get('matrix_margin', 0),
            draw_legend=state.get('draw_legend', True),
            include_reference=reference_row is not None,
            reference_width=(reference_row or {}).get('width'),
            reference_height=(reference_row or {}).get('height'),
        )
        if capacity < 1:
            raise RuntimeError(
                "the selected comparison orientation cannot fit one aligned weight "
                "slot inside the safe Gallery page limits at the generated resolution"
            )

        total_weights = len(groups[0]['cells'])
        slug_prefix = (
            f"{str(output_slug_prefix).strip('-')}-" if output_slug_prefix else ""
        )
        page_paths = []
        for part_number, start in enumerate(range(0, total_weights, capacity), start=1):
            end = min(total_weights, start + capacity)
            rows = []
            if orientation == self.MATRIX_LAYOUT_COMPARE_HORIZONTAL:
                for item_number, group in enumerate(groups, start=1):
                    rows.extend(
                        self._build_row_strips(
                            state,
                            group['cells'][start:end],
                            columns=end - start,
                            filename_prefix=(
                                f"{slug_prefix}comparison-rows-part-{part_number:03d}-"
                                f"item-{item_number:02d}"
                            ),
                        )
                    )
                orientation_text = "LoRAs as rows"
            else:
                for weight_index in range(start, end):
                    rows.extend(
                        self._build_row_strips(
                            state,
                            [group['cells'][weight_index] for group in groups],
                            columns=len(groups),
                            filename_prefix=(
                                f"{slug_prefix}comparison-columns-part-{part_number:03d}-"
                                f"weight-{weight_index:03d}"
                            ),
                        )
                    )
                orientation_text = "LoRAs as columns"

            _, _, estimated_peak = self._estimate_page_peak(
                state,
                rows,
                reference_row,
            )
            if estimated_peak == float('inf'):
                raise RuntimeError(
                    "an aligned comparison part exceeded the safe Gallery page limits "
                    "after its real labels were measured; reduce the range or source "
                    "resolution"
                )
            slug = f"{slug_prefix}combined-comparison-part-{part_number:03d}"
            pages = self._build_matrix_pages(
                state,
                rows,
                p,
                processed,
                grid_info,
                reference_row=reference_row,
                output_slug=slug,
                output_description=(
                    f"combined comparison ({orientation_text}, weights "
                    f"{start + 1}-{end} of {total_weights})"
                ),
            )
            if len(pages) != 1:
                raise RuntimeError(
                    "an aligned comparison part could not be kept on one safe page; "
                    "recovery files were retained instead of returning a misleading "
                    "partial comparison"
                )
            page_paths.extend(pages)
        return page_paths

    def _build_per_item_matrix_pages(
        self,
        state,
        group,
        p,
        processed,
        grid_info,
        reference_row=None,
        output_slug="item",
    ):
        """Build one horizontal weight row per item and split only when required."""

        cells = list(group.get('cells') or [])
        if not cells:
            return []
        cell_width = max(int(cell['width']) for cell in cells)
        cell_height = max(int(cell['height']) for cell in cells)
        capacity = self._comparison_capacity(
            cell_width,
            cell_height,
            1,
            self.MATRIX_LAYOUT_COMPARE_HORIZONTAL,
            margin=state.get('matrix_margin', 0),
            draw_legend=state.get('draw_legend', True),
            include_reference=reference_row is not None,
            reference_width=(reference_row or {}).get('width'),
            reference_height=(reference_row or {}).get('height'),
        )
        if capacity < 1:
            raise RuntimeError(
                "the per-item grid cannot fit one image inside the safe Gallery "
                "page limits at the generated resolution"
            )

        total_parts = (len(cells) + capacity - 1) // capacity
        page_paths = []
        for part_number, start in enumerate(range(0, len(cells), capacity), start=1):
            part = cells[start:start + capacity]
            part_slug = (
                output_slug
                if total_parts == 1
                else f"{output_slug}-part-{part_number:03d}"
            )
            rows = self._build_row_strips(
                state,
                part,
                columns=len(part),
                filename_prefix=f"{part_slug}-row",
            )
            if len(rows) != 1:
                raise RuntimeError(
                    "a per-item grid part could not be kept in one horizontal row"
                )
            _, _, estimated_peak = self._estimate_page_peak(
                state,
                rows,
                reference_row,
            )
            if estimated_peak == float('inf'):
                raise RuntimeError(
                    "a horizontal per-item grid exceeded the safe Gallery page limits "
                    "after its real labels were measured; reduce the range or source "
                    "resolution"
                )
            pages = self._build_matrix_pages(
                state,
                rows,
                p,
                processed,
                grid_info,
                reference_row=reference_row,
                output_slug=part_slug,
                output_description=(
                    f"individual grid for {group['item_label']} "
                    f"(horizontal part {part_number} of {total_parts})"
                ),
            )
            if len(pages) != 1:
                raise RuntimeError(
                    "a horizontal per-item grid part could not be kept on one safe page; "
                    "recovery files were retained"
                )
            page_paths.extend(pages)
        return page_paths

    def _build_requested_matrices(
        self,
        state,
        comparison_cells,
        p,
        processed,
        grid_info,
        reference_row=None,
        output_slug_prefix=None,
    ):
        """Build per-item grids first and append the optional combined overview."""

        page_paths = []
        state['matrix_outputs'] = []
        groups = self._group_comparison_cells(comparison_cells)
        slug_prefix = f"{str(output_slug_prefix).strip('-')}-" if output_slug_prefix else ""

        def record_pages(kind, slug, paths, item_key=None, item_label=None):
            for page_number, path in enumerate(paths, start=1):
                state['matrix_outputs'].append(
                    {
                        'kind': kind,
                        'slug': slug,
                        'item_key': item_key,
                        'item_label': item_label,
                        'page': page_number,
                        'path': str(path),
                    }
                )
            page_paths.extend(paths)
            self._write_manifest(state, "matrix-pages")

        if state.get('create_per_item_grids', False):
            for ordinal, group in enumerate(groups, start=1):
                slug = self._matrix_output_slug(
                    state.get('test_type'),
                    group['item_label'],
                    ordinal,
                )
                slug = f"{slug_prefix}{slug}"
                item_pages = self._build_per_item_matrix_pages(
                    state,
                    group,
                    p,
                    processed,
                    grid_info,
                    reference_row=reference_row,
                    output_slug=slug,
                )
                record_pages(
                    "item",
                    slug,
                    item_pages,
                    item_key=group['item_key'],
                    item_label=group['item_label'],
                )

        if state.get('create_combined_overview', True):
            layout = state.get('comparison_layout', self.MATRIX_LAYOUT_STANDARD)
            if layout == self.MATRIX_LAYOUT_STANDARD:
                combined_rows = self._build_row_strips(
                    state,
                    comparison_cells,
                    filename_prefix=f"{slug_prefix}combined-row",
                )
                combined_pages = self._build_matrix_pages(
                    state,
                    combined_rows,
                    p,
                    processed,
                    grid_info,
                    reference_row=reference_row,
                    output_slug=f"{slug_prefix}combined-overview",
                    output_description="combined overview",
                )
            else:
                combined_pages = self._build_comparison_matrix_pages(
                    state,
                    groups,
                    p,
                    processed,
                    grid_info,
                    reference_row=reference_row,
                    output_slug_prefix=output_slug_prefix,
                )
            record_pages(
                "combined",
                f"{slug_prefix}combined-overview",
                combined_pages,
            )
        elif groups:
            print("[LoRA Tester] Combined overview disabled; returning individual grids only")

        if not page_paths and reference_row is not None:
            reference_pages = self._build_matrix_pages(
                state,
                [],
                p,
                processed,
                grid_info,
                reference_row=reference_row,
                output_slug=f"{slug_prefix}reference-recovery",
                output_description="reference recovery",
            )
            record_pages(
                "reference",
                f"{slug_prefix}reference-recovery",
                reference_pages,
            )

        return page_paths

    def _build_seed_round_matrix_pages(
        self,
        state,
        cells,
        p,
        processed,
        grid_info,
    ):
        """Build independent matrix output for each preserved Forge Batch Count seed."""

        page_paths = []
        page_seeds = []
        page_subseeds = []
        state['rows'] = []
        state['matrix_outputs'] = []
        state['reference_rows'] = []

        for round_info in state.get('seed_rounds', []):
            round_index = int(round_info['index'])
            round_cells = [
                cell
                for cell in cells
                if int(cell.get('seed_round', 0)) == round_index
            ]
            if not round_cells:
                continue

            round_state = dict(state)
            round_state['rows'] = []
            round_state['matrix_outputs'] = []
            reference_cell = next(
                (cell for cell in round_cells if cell.get('is_baseline')),
                None,
            )
            comparison_cells = [
                cell for cell in round_cells if cell is not reference_cell
            ]
            prefix = (
                f"seed-round-{round_index + 1:03d}-"
                f"seed-{int(round_info['seed'])}"
            )
            reference_row = None
            if reference_cell is not None:
                reference_rows = self._build_row_strips(
                    round_state,
                    [reference_cell],
                    columns=1,
                    filename_prefix=f"{prefix}-reference-row",
                    track_as_matrix_rows=False,
                )
                reference_row = reference_rows[0]
                round_state['reference_row'] = reference_row
                state['reference_rows'].append(reference_row)
                if state.get('reference_row') is None:
                    state['reference_row'] = reference_row

            round_pages = self._build_requested_matrices(
                round_state,
                comparison_cells,
                p,
                processed,
                grid_info,
                reference_row=reference_row,
                output_slug_prefix=prefix,
            )
            for output in round_state.get('matrix_outputs', []):
                output['seed_round'] = round_index
                output['seed'] = int(round_info['seed'])
                output['subseed'] = int(round_info['subseed'])
            state['rows'].extend(round_state.get('rows', []))
            state['matrix_outputs'].extend(round_state.get('matrix_outputs', []))
            page_paths.extend(round_pages)
            page_seeds.extend([int(round_info['seed'])] * len(round_pages))
            page_subseeds.extend([int(round_info['subseed'])] * len(round_pages))
            self._write_manifest(state, "matrix-pages")

        return page_paths, page_seeds, page_subseeds

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

    def _build_row_strips(
        self,
        state,
        cells,
        columns=None,
        filename_prefix="row",
        track_as_matrix_rows=True,
    ):
        """Create labeled row files while decoding only one source cell at a time."""

        if not cells:
            return []
        requested_columns = state['matrix_cols'] if columns is None else columns
        columns = max(1, min(int(requested_columns), len(cells)))
        cell_width = max(int(cell['width']) for cell in cells)
        cell_height = max(int(cell['height']) for cell in cells)
        background = ImageColor.getcolor(shared.opts.grid_background_color, "RGB")
        rows = []

        for row_index, start in enumerate(range(0, len(cells), columns)):
            group = cells[start:start + columns]
            row_grid = None
            annotated_row = None

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

                row_path = Path(state['session_dir']) / (
                    f"{filename_prefix}-{row_index:04d}.png"
                )
                self._save_png_atomic(annotated_row, row_path)
                row_record = {
                    'index': row_index,
                    'path': str(row_path),
                    'width': int(annotated_row.width),
                    'height': int(annotated_row.height),
                    'cell_indices': [cell['index'] for cell in group],
                }
                rows.append(row_record)
                if track_as_matrix_rows:
                    state.setdefault('rows', []).append(row_record)
            finally:
                if annotated_row is not None and annotated_row is not row_grid:
                    annotated_row.close()
                if row_grid is not None:
                    row_grid.close()

            if track_as_matrix_rows:
                self._write_manifest(state, "row-strips")

        # Keep every source cell until all matrix pages have been written and
        # validated. Matrix-only deletion is deliberately deferred to the final
        # cleanup; a page-composition failure must leave recoverable originals.
        if track_as_matrix_rows:
            self._write_manifest(state, "row-strips-ready")
        gc.collect()
        return rows

    def _estimate_page_peak(self, state, rows, reference_row=None):
        page_rows = ([reference_row] if reference_row is not None else []) + list(rows)
        if not page_rows:
            return 0, 0, 0
        width = max(row['width'] for row in page_rows)
        height = sum(row['height'] for row in page_rows)
        height += state['matrix_margin'] * max(0, len(page_rows) - 1)
        pixel_count = width * height
        if (
            width > self.MAX_MATRIX_DIMENSION
            or height > self.MAX_MATRIX_DIMENSION
            or pixel_count > self.MAX_MATRIX_PIXELS
        ):
            return width, height, float('inf')

        # Four bytes per page pixel is deliberately conservative for an RGB PIL
        # canvas. Add theoretical working-row and PNG encoding room. This estimate
        # is informational; fixed dimension and area limits determine page safety.
        canvas_bytes = width * height * 4
        largest_row_bytes = max(
            row['width'] * row['height'] * 4 for row in page_rows
        )
        working_row = largest_row_bytes * 2
        encoding_reserve = max(64 * MIB, int(canvas_bytes * 0.20))
        estimated_peak = int(canvas_bytes * self.PAGE_SAFETY_FACTOR + working_row + encoding_reserve)
        return width, height, estimated_peak

    def _build_matrix_pages(
        self,
        state,
        rows,
        p,
        processed,
        grid_info,
        reference_row=None,
        output_slug="combined-overview",
        output_description="combined overview",
    ):
        page_paths = []
        row_offset = 0

        if not rows and reference_row is not None:
            width, height, estimated_peak = self._estimate_page_peak(
                state, [], reference_row
            )
            print(
                f"[LoRA Tester] {output_description}, page 1: reference image only, "
                f"estimated peak {estimated_peak / MIB:.0f} MB"
            )
            page_path = self._compose_matrix_page(
                state,
                [],
                width,
                height,
                1,
                p,
                processed,
                grid_info,
                reference_row=reference_row,
                output_slug=output_slug,
            )
            return [Path(page_path)]

        while row_offset < len(rows):
            selected_rows = []

            for row in rows[row_offset:]:
                candidate = selected_rows + [row]
                _, _, estimated_peak = self._estimate_page_peak(
                    state, candidate, reference_row
                )
                if estimated_peak != float('inf'):
                    selected_rows = candidate
                    continue
                break

            page_number = len(page_paths) + 1
            if not selected_rows:
                # The already-labeled row file is itself a valid one-row matrix.
                # Returning its path requires no additional decoded-image canvas.
                row = rows[row_offset]
                row_width, row_height, row_peak = self._estimate_page_peak(
                    state, [row], None
                )
                if row_peak == float('inf'):
                    raise RuntimeError(
                        "one labeled matrix row exceeds the safe Gallery page limits "
                        f"({row_width}x{row_height}, {row_width * row_height:,} pixels); "
                        "reduce Matrix Columns or the source image resolution"
                    )
                fallback_path = Path(row['path'])
                if getattr(shared.opts, 'grid_save', False):
                    output_dir = Path(p.outpath_grids)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    session_name = Path(state['session_dir']).name
                    output_path = output_dir / (
                        f"lora_tester_grid-{session_name}-{output_slug}-"
                        f"page-{page_number:03d}.png"
                    )
                    shutil.copy2(fallback_path, output_path)
                    fallback_path = output_path
                page_paths.append(fallback_path)
                row_offset += 1
                print(
                    f"[LoRA Tester] {output_description}, page {page_number}: "
                    "one-row disk fallback "
                    "because the repeated reference would cross a page limit"
                )
                continue

            width, height, estimated_peak = self._estimate_page_peak(
                state, selected_rows, reference_row
            )
            print(
                f"[LoRA Tester] {output_description}, page {page_number}: "
                f"{len(selected_rows)} row(s), "
                f"{width}x{height} ({width * height:,} pixels), "
                f"estimated peak {estimated_peak / MIB:.0f} MB"
                f"{' including repeated reference' if reference_row else ''}"
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
                reference_row=reference_row,
                output_slug=output_slug,
            )
            page_paths.append(Path(page_path))
            row_offset += len(selected_rows)
            gc.collect()

        return page_paths

    def _compose_matrix_page(
        self,
        state,
        rows,
        width,
        height,
        page_number,
        p,
        processed,
        grid_info,
        reference_row=None,
        output_slug="combined-overview",
    ):
        background = ImageColor.getcolor(shared.opts.grid_background_color, "RGB")
        page = Image.new("RGB", (width, height), background)
        try:
            page_rows = (
                ([reference_row] if reference_row is not None else []) + list(rows)
            )
            y = 0
            for row_index, row in enumerate(page_rows):
                with Image.open(row['path']) as row_image:
                    row_image.load()
                    converted = row_image if row_image.mode == "RGB" else row_image.convert("RGB")
                    x = (width - converted.width) // 2
                    page.paste(converted, (x, y))
                    y += converted.height
                    if row_index + 1 < len(page_rows):
                        y += state['matrix_margin']
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
                            f"lora_tester_grid-{session_name}-{output_slug}-"
                            f"page-{page_number:03d}"
                        ),
                        'save_to_dirs': False,
                    }
                page_path, _ = images.save_image(
                    page,
                    p.outpath_grids,
                    f"lora_tester_{output_slug}",
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

            page_path = Path(state['session_dir']) / (
                f"matrix-{output_slug}-page-{page_number:03d}.png"
            )
            parameters = grid_info if getattr(shared.opts, 'enable_pnginfo', False) else None
            self._save_png_atomic(page, page_path, parameters=parameters)
            return str(page_path)
        finally:
            page.close()

    def _cleanup_spool_files(self, state, page_paths):
        protected = {str(Path(path).resolve()) for path in page_paths}
        cleanup_complete = True
        if not state.get('keep_individual_images'):
            delete_allowed = bool(state.get('matrix_only_delete_allowed', False))
            cells = list(state.get('cells', {}).values())
            if delete_allowed:
                deleted = 0
                for cell in cells:
                    if self._safe_unlink(cell.get('path')):
                        deleted += 1
                    else:
                        cleanup_complete = False
            elif cells:
                state['recovery_retained'] = True
                cleanup_complete = False
                reason = state.get('matrix_only_retention_reason') or "the run was incomplete"
                print(
                    f"[LoRA Tester] Matrix-only recovery: retained {len(cells)} source "
                    f"image(s) in {state['session_dir']} because {reason}"
                )
            if delete_allowed and cleanup_complete:
                print(
                    f"[LoRA Tester] Matrix-only cleanup: deleted {deleted} individual "
                    "image(s) after complete generation and matrix validation"
                )
            elif delete_allowed:
                print(
                    "[LoRA Tester] Warning: matrix pages are safe, but one or more "
                    "individual images could not be deleted"
                )
        for row in state.get('rows', []):
            row_path = row.get('path')
            if row_path and str(Path(row_path).resolve()) not in protected:
                cleanup_complete = self._safe_unlink(row_path) and cleanup_complete

        reference_rows = list(state.get('reference_rows', []))
        if not reference_rows and state.get('reference_row'):
            reference_rows = [state['reference_row']]
        cleaned_reference_paths = set()
        for reference_row in reference_rows:
            reference_path = reference_row.get('path')
            if not reference_path:
                continue
            resolved = str(Path(reference_path).resolve())
            if resolved in protected or resolved in cleaned_reference_paths:
                continue
            cleaned_reference_paths.add(resolved)
            cleanup_complete = self._safe_unlink(reference_path) and cleanup_complete

        if state.get('recovery_retained'):
            self._write_manifest(state, "recovery-retained")
        elif cleanup_complete:
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
