# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 vibecodingtoolmaker

import ast
import copy
import gc
import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal, InvalidOperation
from pathlib import Path
from PIL import Image, ImageColor, PngImagePlugin
from types import ModuleType, SimpleNamespace
from unittest import mock


SOURCE_PATH = Path(__file__).parents[1] / "scripts" / "lora_tester.py"
JAVASCRIPT_PATH = (
    Path(__file__).parents[1] / "javascript" / "lora_tester_dataframe.js"
)


def load_class_methods(class_name, *method_names):
    """Load selected class methods without importing Forge-specific modules."""

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    script_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    requested = set(method_names)
    functions = []
    for node in script_class.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in requested
        ):
            node.decorator_list = []
            functions.append(node)

    missing = requested.difference(node.name for node in functions)
    if missing:
        raise AssertionError(f"Methods not found: {sorted(missing)}")

    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {
        "copy": copy,
        "Image": Image,
        "ImageColor": ImageColor,
        "PngImagePlugin": PngImagePlugin,
        "Path": Path,
        "json": json,
        "os": os,
        "print": print,
        "MIB": 1024**2,
        "gc": gc,
        "Decimal": Decimal,
        "InvalidOperation": InvalidOperation,
        "StableDiffusionProcessing": object,
        "shared": SimpleNamespace(
            cmd_opts=SimpleNamespace(embeddings_dir="."),
            sd_model=None,
        ),
        "read_embedding_trigger_words": lambda path: ([Path(path).stem], "filename"),
    }
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return {name: namespace[name] for name in requested}


def load_methods(*method_names):
    return load_class_methods("LoRaTesterScript", *method_names)


class RecoveryHarness:
    pass


METHODS = load_methods(
    "_case_is_baseline",
    "_build_embedding_settings_rows",
    "_cell_text",
    "_format_weight",
    "_coerce_settings_rows",
    "_coerce_embedding_settings_rows",
    "_compose_embedding_prompt",
    "_maximum_total_cases",
    "_resolve_matrix_outputs",
    "_run_summary_html",
    "_comparison_capacity_visible",
    "_estimated_label_header_height",
    "_comparison_capacity",
    "_comparison_range_hint",
    "_estimate_page_peak",
    "_build_matrix_pages",
    "_build_comparison_matrix_pages",
    "_build_per_item_matrix_pages",
    "_build_requested_matrices",
    "_group_comparison_cells",
    "_comparison_group_error",
    "_matrix_output_slug",
    "_matrix_only_cleanup_decision",
    "_parse_weight_spec",
    "_resolve_embedding_target",
    "_resolve_trigger_words",
    "_resolve_item_weights",
    "_multi_model_request_error",
    "_build_multi_model_overview_pages",
    "_multi_model_page_geometry",
    "_prepare_multi_model_job_arguments",
    "_postprocess_multi_model",
    "_run_multi_model_jobs",
    "_split_multi_model_overview_records",
    "_suppress_multi_model_coordinator_batch",
    "before_process_batch",
    "_release_processing_references",
    "_restore_forge_model_state",
    "_save_png_atomic",
    "_safe_filename_part",
    "_safe_unlink",
    "_validate_matrix_pages",
    "_write_manifest",
    "_cleanup_spool_files",
)
RecoveryHarness._matrix_only_cleanup_decision = staticmethod(
    METHODS["_matrix_only_cleanup_decision"]
)
RecoveryHarness._safe_unlink = staticmethod(METHODS["_safe_unlink"])
RecoveryHarness._write_manifest = staticmethod(METHODS["_write_manifest"])
RecoveryHarness._cleanup_spool_files = METHODS["_cleanup_spool_files"]
RecoveryHarness._maximum_total_cases = classmethod(METHODS["_maximum_total_cases"])
RecoveryHarness._resolve_matrix_outputs = staticmethod(
    METHODS["_resolve_matrix_outputs"]
)
RecoveryHarness._run_summary_html = classmethod(METHODS["_run_summary_html"])
RecoveryHarness._comparison_capacity_visible = classmethod(
    METHODS["_comparison_capacity_visible"]
)
RecoveryHarness._cell_text = staticmethod(METHODS["_cell_text"])
RecoveryHarness._format_weight = staticmethod(METHODS["_format_weight"])
RecoveryHarness._coerce_settings_rows = staticmethod(METHODS["_coerce_settings_rows"])
RecoveryHarness._coerce_embedding_settings_rows = staticmethod(
    METHODS["_coerce_embedding_settings_rows"]
)
RecoveryHarness._parse_weight_spec = classmethod(METHODS["_parse_weight_spec"])
RecoveryHarness._resolve_embedding_target = classmethod(
    METHODS["_resolve_embedding_target"]
)
RecoveryHarness._resolve_item_weights = classmethod(METHODS["_resolve_item_weights"])
RecoveryHarness._multi_model_request_error = classmethod(
    METHODS["_multi_model_request_error"]
)
RecoveryHarness._build_multi_model_overview_pages = METHODS[
    "_build_multi_model_overview_pages"
]
RecoveryHarness._multi_model_page_geometry = classmethod(
    METHODS["_multi_model_page_geometry"]
)
RecoveryHarness._prepare_multi_model_job_arguments = classmethod(
    METHODS["_prepare_multi_model_job_arguments"]
)
RecoveryHarness._postprocess_multi_model = METHODS["_postprocess_multi_model"]
RecoveryHarness._run_multi_model_jobs = METHODS["_run_multi_model_jobs"]
RecoveryHarness._split_multi_model_overview_records = classmethod(
    METHODS["_split_multi_model_overview_records"]
)
RecoveryHarness._suppress_multi_model_coordinator_batch = staticmethod(
    METHODS["_suppress_multi_model_coordinator_batch"]
)
RecoveryHarness.before_process_batch = METHODS["before_process_batch"]
RecoveryHarness._release_processing_references = staticmethod(
    METHODS["_release_processing_references"]
)
RecoveryHarness._restore_forge_model_state = classmethod(
    METHODS["_restore_forge_model_state"]
)
RecoveryHarness._save_png_atomic = staticmethod(METHODS["_save_png_atomic"])
RecoveryHarness._validate_matrix_pages = staticmethod(
    METHODS["_validate_matrix_pages"]
)
RecoveryHarness._compose_embedding_prompt = staticmethod(
    METHODS["_compose_embedding_prompt"]
)
RecoveryHarness._resolve_trigger_words = classmethod(METHODS["_resolve_trigger_words"])
RecoveryHarness._case_is_baseline = staticmethod(METHODS["_case_is_baseline"])
RecoveryHarness._build_embedding_settings_rows = METHODS[
    "_build_embedding_settings_rows"
]
RecoveryHarness._estimate_page_peak = METHODS["_estimate_page_peak"]
RecoveryHarness._build_matrix_pages = METHODS["_build_matrix_pages"]
RecoveryHarness._build_comparison_matrix_pages = METHODS[
    "_build_comparison_matrix_pages"
]
RecoveryHarness._build_per_item_matrix_pages = METHODS[
    "_build_per_item_matrix_pages"
]
RecoveryHarness._build_requested_matrices = METHODS["_build_requested_matrices"]
RecoveryHarness._group_comparison_cells = staticmethod(
    METHODS["_group_comparison_cells"]
)
RecoveryHarness._safe_filename_part = staticmethod(METHODS["_safe_filename_part"])
RecoveryHarness._matrix_output_slug = classmethod(METHODS["_matrix_output_slug"])
RecoveryHarness._comparison_group_error = classmethod(
    METHODS["_comparison_group_error"]
)
RecoveryHarness._estimated_label_header_height = classmethod(
    METHODS["_estimated_label_header_height"]
)
RecoveryHarness._comparison_capacity = classmethod(METHODS["_comparison_capacity"])
RecoveryHarness._comparison_range_hint = classmethod(
    METHODS["_comparison_range_hint"]
)
RecoveryHarness.MAX_TOTAL_CASES = 500
RecoveryHarness.MAX_EXTREME_TOTAL_CASES = 10_000
RecoveryHarness.MAX_WEIGHTS_PER_ITEM = 100
RecoveryHarness.MAX_MATRIX_DIMENSION = 65_000
RecoveryHarness.MAX_MATRIX_PIXELS = 89_000_000
RecoveryHarness.PAGE_SAFETY_FACTOR = 1.35
RecoveryHarness.LABEL_FONT_DIVISOR = 36
RecoveryHarness.LABEL_MIN_FONT_SIZE = 14
RecoveryHarness.LABEL_MAX_LINES = 4
RecoveryHarness.MATRIX_LAYOUT_STANDARD = "Standard order"
RecoveryHarness.MATRIX_LAYOUT_COMPARE_HORIZONTAL = "Compare: LoRAs as rows"
RecoveryHarness.MATRIX_LAYOUT_COMPARE_VERTICAL = "Compare: LoRAs as columns"
RecoveryHarness.MAX_COMPARISON_ITEMS = 10
RecoveryHarness.DEFAULT_COMPARISON_STEP = Decimal("0.5")
RecoveryHarness.EMBEDDING_TARGET_POSITIVE = "Positive prompt"
RecoveryHarness.EMBEDDING_TARGET_NEGATIVE = "Negative prompt"
RecoveryHarness.TEST_TYPE_LORA = "LoRA"
RecoveryHarness.TEST_TYPE_EMBEDDING = "Embedding"
RecoveryHarness.TEST_TYPE_PROMPT = "Prompt only (internal)"
RecoveryHarness.MULTI_MODEL_CURRENT = "Current model only (existing behavior)"
RecoveryHarness.MULTI_MODEL_LORA = "One LoRA across selected models"
RecoveryHarness.MULTI_MODEL_PROMPT = "Prompt only across selected models"
RecoveryHarness.MAX_MULTI_MODEL_CHECKPOINTS = 25
RecoveryHarness.MAX_EXTREME_MULTI_MODEL_CHECKPOINTS = 150
RecoveryHarness.PROMPT_ONLY_SENTINEL = "__lora_tester_prompt_only__"
RecoveryHarness.BATCH_MODEL_LABEL_ATTRIBUTE = "_lora_tester_batch_model_label"
RecoveryHarness.PROCESSING_STATE_ATTRIBUTE = "_lora_tester_state"


class MultiModelPlanTests(unittest.TestCase):
    def setUp(self):
        self.harness = RecoveryHarness()

    def test_existing_single_model_mode_keeps_selection_contract_unchanged(self):
        error = self.harness._multi_model_request_error(
            self.harness.MULTI_MODEL_CURRENT,
            [],
            self.harness.TEST_TYPE_EMBEDDING,
            [],
        )

        self.assertIsNone(error)

    def test_one_lora_mode_requires_models_and_exactly_one_lora(self):
        self.assertIn(
            "at least one checkpoint",
            self.harness._multi_model_request_error(
                self.harness.MULTI_MODEL_LORA,
                [],
                self.harness.TEST_TYPE_LORA,
                ["one.safetensors"],
            ),
        )
        self.assertIn(
            "exactly one selected LoRA",
            self.harness._multi_model_request_error(
                self.harness.MULTI_MODEL_LORA,
                ["model-a.safetensors"],
                self.harness.TEST_TYPE_LORA,
                ["one.safetensors", "two.safetensors"],
            ),
        )
        self.assertIsNone(
            self.harness._multi_model_request_error(
                self.harness.MULTI_MODEL_LORA,
                ["model-a.safetensors", "model-b.safetensors"],
                self.harness.TEST_TYPE_LORA,
                ["one.safetensors"],
            )
        )

    def test_prompt_only_mode_needs_no_real_lora(self):
        self.assertIsNone(
            self.harness._multi_model_request_error(
                self.harness.MULTI_MODEL_PROMPT,
                ["model-a.safetensors"],
                self.harness.TEST_TYPE_EMBEDDING,
                [],
            )
        )

    def test_standard_multi_model_limit_is_twenty_five(self):
        self.assertIsNone(
            self.harness._multi_model_request_error(
                self.harness.MULTI_MODEL_PROMPT,
                [f"model-{index}" for index in range(25)],
                self.harness.TEST_TYPE_LORA,
                [],
            )
        )
        error = self.harness._multi_model_request_error(
            self.harness.MULTI_MODEL_PROMPT,
            [f"model-{index}" for index in range(26)],
            self.harness.TEST_TYPE_LORA,
            [],
        )

        self.assertIn("at most 25 checkpoints", error)
        self.assertIn("Extreme Run Mode", error)
        self.assertIn("150 checkpoints", error)

    def test_extreme_multi_model_limit_is_one_hundred_fifty(self):
        self.assertIsNone(
            self.harness._multi_model_request_error(
                self.harness.MULTI_MODEL_PROMPT,
                [f"model-{index}" for index in range(150)],
                self.harness.TEST_TYPE_LORA,
                [],
                extreme_run_mode=True,
            )
        )
        error = self.harness._multi_model_request_error(
            self.harness.MULTI_MODEL_PROMPT,
            [f"model-{index}" for index in range(151)],
            self.harness.TEST_TYPE_LORA,
            [],
            extreme_run_mode=True,
        )

        self.assertIn("at most 150 checkpoints", error)

    def test_safe_model_pages_share_one_overview_grid(self):
        records = [
            {"path": "model-a.png", "width": 1024, "height": 1112},
            {"path": "model-b.png", "width": 1024, "height": 1112},
        ]

        chunks = self.harness._split_multi_model_overview_records(
            records,
            requested_columns=3,
            margin=8,
        )
        geometry = self.harness._multi_model_page_geometry(
            chunks[0],
            requested_columns=3,
            margin=8,
        )

        self.assertEqual(chunks, [records])
        self.assertEqual(geometry["columns"], 2)
        self.assertEqual(geometry["rows"], 1)
        self.assertEqual(geometry["width"], 2056)
        self.assertEqual(geometry["height"], 1112)

    def test_multi_model_overview_splits_only_at_matrix_safety_limit(self):
        records = [
            {"path": f"model-{index}.png", "width": 30_000, "height": 1_000}
            for index in range(3)
        ]

        chunks = self.harness._split_multi_model_overview_records(
            records,
            requested_columns=3,
            margin=8,
        )

        self.assertEqual([len(chunk) for chunk in chunks], [2, 1])

    def test_multi_model_overview_writes_both_model_pages_to_one_png(self):
        shared_stub = METHODS["_build_multi_model_overview_pages"].__globals__["shared"]
        shared_stub.opts = SimpleNamespace(
            grid_background_color="#000000",
            grid_save=False,
            enable_pnginfo=True,
        )

        with tempfile.TemporaryDirectory() as temporary_root:
            temporary_root = Path(temporary_root)
            first = temporary_root / "model-a.png"
            second = temporary_root / "model-b.png"
            Image.new("RGB", (64, 32), (255, 0, 0)).save(first)
            Image.new("RGB", (64, 32), (0, 0, 255)).save(second)
            state = {
                "session_dir": str(temporary_root),
                "matrix_cols": 3,
                "matrix_margin": 8,
                "fixed_seed": 42,
                "model_results": [
                    {"label": "A", "status": "complete"},
                    {"label": "B", "status": "complete"},
                ],
            }
            processing = SimpleNamespace(
                prompt="prompt",
                outpath_grids=str(temporary_root),
            )
            processed = SimpleNamespace(all_seeds=[42], all_prompts=["prompt"])

            pages, infotexts = self.harness._build_multi_model_overview_pages(
                processing,
                processed,
                state,
                [str(first), str(second)],
                ["grid info A", "grid info B"],
            )

            self.assertEqual(len(pages), 1)
            self.assertEqual(len(infotexts), 1)
            with Image.open(pages[0]) as overview:
                self.assertEqual(overview.size, (136, 32))
                self.assertEqual(overview.getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(overview.getpixel((72, 0)), (0, 0, 255))

    def test_nested_prompt_job_uses_noop_case_and_disables_recursion(self):
        arguments = [None] * 23
        arguments[0] = True
        arguments[1] = self.harness.TEST_TYPE_LORA
        arguments[2] = ["real-lora.safetensors"]
        arguments[11] = True
        arguments[-2] = self.harness.MULTI_MODEL_PROMPT
        arguments[-1] = ["model-a.safetensors"]

        nested = self.harness._prepare_multi_model_job_arguments(
            arguments,
            self.harness.MULTI_MODEL_PROMPT,
        )

        self.assertEqual(nested[1], self.harness.TEST_TYPE_PROMPT)
        self.assertEqual(nested[2], [self.harness.PROMPT_ONLY_SENTINEL])
        self.assertFalse(nested[11])
        self.assertEqual(nested[-2], self.harness.MULTI_MODEL_CURRENT)
        self.assertEqual(nested[-1], [])
        self.assertEqual(arguments[-2], self.harness.MULTI_MODEL_PROMPT)

    def test_prompt_case_is_not_mistaken_for_reference_baseline(self):
        self.assertFalse(
            self.harness._case_is_baseline(
                {
                    "kind": "prompt",
                    "lora_tag_name": None,
                    "embedding_name": None,
                }
            )
        )

    def test_outer_coordinator_batch_is_stopped_before_case_injection(self):
        state = {
            "multi_model": True,
            # Coordinator cases deliberately do not carry LoRA injection fields.
            "cases": [{"kind": "model", "label": "Model A"}],
        }
        processing = SimpleNamespace(_lora_tester_state=state)
        prompts = ["prompt that must not be sampled"]

        self.harness.before_process_batch(
            processing,
            True,
            batch_number=0,
            prompts=prompts,
            seeds=[42],
            subseeds=[7],
        )

        self.assertEqual(prompts, [])
        self.assertTrue(state["outer_sampling_suppressed"])

    def test_outer_job_delivers_one_combined_grid_for_multiple_models(self):
        shared_stub = METHODS["_postprocess_multi_model"].__globals__["shared"]
        shared_stub.state = SimpleNamespace(
            interrupted=False,
            stopping_generation=False,
        )
        self.harness._build_multi_model_overview_pages = (
            lambda p, processed, state, pages, infotexts: (
                ["combined-grid.png"],
                ["combined grid info"],
            )
        )
        self.harness._write_manifest = lambda state, status: None
        self.harness._safe_unlink = lambda path: True
        self.harness._ensure_temp_directory = lambda: None

        with tempfile.TemporaryDirectory() as temporary_root:
            state = {
                "multi_model_pages": ["model-a.png", "model-b.png"],
                "multi_model_infotexts": ["A", "B"],
                "multi_model_prompts": ["prompt", "prompt"],
                "multi_model_negative_prompts": ["negative", "negative"],
                "multi_model_seeds": [42, 42],
                "multi_model_subseeds": [7, 7],
                "model_jobs": [{"label": "A"}, {"label": "B"}],
                "model_results": [
                    {"label": "A", "status": "complete"},
                    {"label": "B", "status": "complete"},
                ],
                "original_model_restored": True,
                "session_dir": temporary_root,
                "manifest_path": None,
            }
            processing = SimpleNamespace(
                _lora_tester_state=state,
                prompt="prompt",
                negative_prompt="negative",
                extra_result_images=["old"],
            )
            processed = SimpleNamespace(
                images=[],
                extra_images=["old"],
                infotexts=[],
                all_prompts=[],
                all_negative_prompts=[],
                all_seeds=[],
                all_subseeds=[],
                comments="",
                index_of_first_image=0,
            )

            self.harness._postprocess_multi_model(processing, processed, state)

        self.assertEqual(processed.images, ["combined-grid.png"])
        self.assertEqual(processed.infotexts, ["combined grid info"])
        self.assertEqual(processed.all_prompts, ["prompt"])
        self.assertEqual(processed.all_seeds, [42])
        self.assertEqual(processing.extra_result_images, [])

    def test_multi_model_manifest_records_jobs_results_and_restore_status(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            manifest = Path(temporary_root) / "manifest.json"
            state = {
                "manifest_path": str(manifest),
                "multi_model": True,
                "multi_model_mode": self.harness.MULTI_MODEL_PROMPT,
                "model_jobs": [{"label": "A", "checkpoint": "a.safetensors"}],
                "model_results": [
                    {
                        "label": "A",
                        "checkpoint": "a.safetensors",
                        "status": "complete",
                        "pages": ["a.png"],
                    }
                ],
                "original_checkpoint": "original.safetensors",
                "original_model_restored": True,
                "cases": [{}],
                "cells": {},
            }

            self.harness._write_manifest(state, "multi-model-complete")
            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertTrue(payload["multi_model"])
        self.assertEqual(payload["model_jobs"][0]["label"], "A")
        self.assertEqual(payload["model_results"][0]["status"], "complete")
        self.assertTrue(payload["original_model_restored"])

    def test_model_boundary_releases_processing_references(self):
        class ProcessingStub:
            cached_hr_c = [1, 2, 3]
            cached_hr_uc = [1, 2, 3]

            def __init__(self):
                self.closed = False
                self.prompt_cache_cleared = False
                self.c = object()
                self.uc = object()
                self.sampler = object()
                self.rng = object()
                self.modified_noise = object()
                self.hr_c = object()
                self.hr_uc = object()

            def close(self):
                self.closed = True

            def clear_prompt_cache(self):
                self.prompt_cache_cleared = True

        processing = ProcessingStub()

        self.harness._release_processing_references(processing)

        self.assertTrue(processing.closed)
        self.assertTrue(processing.prompt_cache_cleared)
        for attribute in (
            "c",
            "uc",
            "sampler",
            "rng",
            "modified_noise",
            "hr_c",
            "hr_uc",
        ):
            self.assertIsNone(getattr(processing, attribute))
        self.assertEqual(ProcessingStub.cached_hr_c, [None, None, None])
        self.assertEqual(ProcessingStub.cached_hr_uc, [None, None, None])

    def test_restore_path_reinstates_selection_and_loading_parameters(self):
        events = []
        fake_processing = SimpleNamespace(need_global_unload=False)

        def manage_model_and_prompt_cache(p):
            events.append(("manage", p, fake_processing.need_global_unload))

        fake_processing.manage_model_and_prompt_cache = manage_model_and_prompt_cache
        fake_sd_models = SimpleNamespace(
            model_data=SimpleNamespace(forge_loading_parameters={"changed": True})
        )
        fake_modules = ModuleType("modules")
        fake_modules.processing = fake_processing
        fake_modules.sd_models = fake_sd_models
        fake_sysinfo = ModuleType("modules.sysinfo")

        def set_config(settings, **kwargs):
            events.append(("settings", settings, kwargs))

        fake_sysinfo.set_config = set_config
        snapshot = {
            "checkpoint": "original.safetensors",
            "additional_modules": ["vae.safetensors"],
            "loading_parameters": {
                "checkpoint_info": "original-info",
                "additional_modules": ["vae.safetensors"],
                "unet_storage_dtype": "bf16",
            },
        }
        processing = object()

        with mock.patch.dict(
            sys.modules,
            {
                "modules": fake_modules,
                "modules.sysinfo": fake_sysinfo,
            },
        ):
            self.harness._restore_forge_model_state(processing, snapshot)

        self.assertEqual(
            events[0][1],
            {
                "sd_model_checkpoint": "original.safetensors",
                "forge_additional_modules": ["vae.safetensors"],
            },
        )
        self.assertEqual(
            fake_sd_models.model_data.forge_loading_parameters,
            snapshot["loading_parameters"],
        )
        self.assertEqual(events[1], ("manage", processing, True))

    def test_model_jobs_unload_between_loads_then_restore_original_selection(self):
        events = []
        shared_stub = METHODS["_run_multi_model_jobs"].__globals__["shared"]
        shared_stub.state = SimpleNamespace(
            interrupted=False,
            stopping_generation=False,
            job="",
        )
        fake_processing = ModuleType("modules.processing")

        def process_images(job_p):
            checkpoint = job_p.override_settings["sd_model_checkpoint"]
            events.append(("run", checkpoint))
            return SimpleNamespace(
                images=[f"{checkpoint}.png"],
                infotexts=[checkpoint],
                all_prompts=["prompt"],
                all_negative_prompts=["negative"],
                all_seeds=[42],
                all_subseeds=[7],
            )

        fake_processing.process_images = process_images
        fake_modules = ModuleType("modules")
        fake_modules.processing = fake_processing

        self.harness.args_from = 0
        self.harness.args_to = 23
        self.harness._snapshot_forge_model_state = lambda: {
            "checkpoint": "original.safetensors"
        }
        self.harness._persist_multi_model_outputs = (
            lambda state, job, processed: (
                list(processed.images),
                list(processed.infotexts),
            )
        )
        self.harness._full_forge_model_unload = lambda job_p: events.append(
            ("unload", job_p.override_settings["sd_model_checkpoint"])
        )
        self.harness._restore_forge_model_state = lambda p, snapshot: events.append(
            ("restore", snapshot["checkpoint"])
        )
        self.harness._write_manifest = lambda state, status: None

        processing = SimpleNamespace(
            all_seeds=[42],
            seed=42,
            all_subseeds=[7],
            subseed=7,
            styles=[],
            override_settings={},
            script_args=[None] * 23,
            prompt="prompt",
            negative_prompt="negative",
            n_iter=1,
            batch_size=1,
        )
        state = {
            "multi_model_mode": self.harness.MULTI_MODEL_PROMPT,
            "model_jobs": [
                {
                    "checkpoint": "a.safetensors",
                    "title": "A",
                    "label": "A",
                },
                {
                    "checkpoint": "b.safetensors",
                    "title": "B",
                    "label": "B",
                },
            ],
            "model_results": [],
            "multi_model_pages": [],
            "multi_model_infotexts": [],
            "multi_model_prompts": [],
            "multi_model_negative_prompts": [],
            "multi_model_seeds": [],
            "multi_model_subseeds": [],
        }
        arguments = [None] * 23

        with mock.patch.dict(sys.modules, {"modules": fake_modules}):
            self.harness._run_multi_model_jobs(
                processing,
                state,
                arguments,
            )

        self.assertEqual(
            events,
            [
                ("run", "a.safetensors"),
                ("unload", "a.safetensors"),
                ("run", "b.safetensors"),
                ("unload", "b.safetensors"),
                ("restore", "original.safetensors"),
            ],
        )
        self.assertTrue(state["original_model_restored"])
        self.assertEqual([result["status"] for result in state["model_results"]], ["complete", "complete"])
        self.assertEqual(processing.n_iter, 0)

    def test_failed_unload_stops_before_next_model_and_still_restores(self):
        events = []
        shared_stub = METHODS["_run_multi_model_jobs"].__globals__["shared"]
        shared_stub.state = SimpleNamespace(
            interrupted=False,
            stopping_generation=False,
            job="",
        )
        fake_processing = ModuleType("modules.processing")

        def process_images(job_p):
            checkpoint = job_p.override_settings["sd_model_checkpoint"]
            events.append(("run", checkpoint))
            return SimpleNamespace(
                images=[f"{checkpoint}.png"],
                infotexts=[checkpoint],
                all_prompts=["prompt"],
                all_negative_prompts=["negative"],
                all_seeds=[42],
                all_subseeds=[7],
            )

        fake_processing.process_images = process_images
        fake_modules = ModuleType("modules")
        fake_modules.processing = fake_processing

        self.harness.args_from = 0
        self.harness.args_to = 23
        self.harness._snapshot_forge_model_state = lambda: {
            "checkpoint": "original.safetensors"
        }
        self.harness._persist_multi_model_outputs = (
            lambda state, job, processed: (
                list(processed.images),
                list(processed.infotexts),
            )
        )

        def fail_unload(job_p):
            checkpoint = job_p.override_settings["sd_model_checkpoint"]
            events.append(("unload", checkpoint))
            raise RuntimeError("unload failed")

        self.harness._full_forge_model_unload = fail_unload
        self.harness._restore_forge_model_state = lambda p, snapshot: events.append(
            ("restore", snapshot["checkpoint"])
        )
        self.harness._write_manifest = lambda state, status: None

        processing = SimpleNamespace(
            all_seeds=[42],
            seed=42,
            all_subseeds=[7],
            subseed=7,
            styles=[],
            override_settings={},
            script_args=[None] * 23,
            prompt="prompt",
            negative_prompt="negative",
            n_iter=1,
            batch_size=1,
        )
        state = {
            "multi_model_mode": self.harness.MULTI_MODEL_PROMPT,
            "model_jobs": [
                {
                    "checkpoint": "a.safetensors",
                    "title": "A",
                    "label": "A",
                },
                {
                    "checkpoint": "b.safetensors",
                    "title": "B",
                    "label": "B",
                },
            ],
            "model_results": [],
            "multi_model_pages": [],
            "multi_model_infotexts": [],
            "multi_model_prompts": [],
            "multi_model_negative_prompts": [],
            "multi_model_seeds": [],
            "multi_model_subseeds": [],
        }

        with mock.patch.dict(sys.modules, {"modules": fake_modules}):
            self.harness._run_multi_model_jobs(
                processing,
                state,
                [None] * 23,
            )

        self.assertEqual(
            events,
            [
                ("run", "a.safetensors"),
                ("unload", "a.safetensors"),
                ("restore", "original.safetensors"),
            ],
        )
        self.assertEqual(state["model_results"][0]["status"], "unload-failed")
        self.assertFalse(state["model_results"][0]["unloaded"])
        self.assertTrue(state["original_model_restored"])


class RunSummaryTests(unittest.TestCase):
    def setUp(self):
        self.harness = RecoveryHarness()

    def summary(self, **overrides):
        values = {
            "test_type": self.harness.TEST_TYPE_LORA,
            "selected_loras": ["a.safetensors", "b.safetensors"],
            "global_lora_weights": "0:1:0.5",
            "lora_settings": [],
            "selected_embeddings": [],
            "global_embedding_weights": "1.0",
            "embedding_settings": [],
            "include_reference": True,
            "multi_model_mode": self.harness.MULTI_MODEL_CURRENT,
            "selected_checkpoints": [],
            "extreme_run_mode": False,
        }
        values.update(overrides)
        return self.harness._run_summary_html(**values)

    def test_current_model_summary_counts_weights_and_reference(self):
        summary = self.summary()

        self.assertIn("6 comparison cell(s) + 1 reference", summary)
        self.assertIn("current model", summary)
        self.assertIn("7 generation(s)", summary)
        self.assertIn("500 cells/model", summary)

    def test_per_item_overrides_are_included_in_summary(self):
        summary = self.summary(
            selected_loras=["a.safetensors", "b.safetensors"],
            global_lora_weights="1.0",
            lora_settings=[
                ["a.safetensors", "", "0", "1", "1"],
                ["b.safetensors", "", "", "", ""],
            ],
            include_reference=False,
        )

        self.assertIn("3 comparison cell(s)", summary)
        self.assertIn("reference disabled", summary)
        self.assertIn("3 generation(s)", summary)

    def test_prompt_only_summary_counts_selected_models(self):
        summary = self.summary(
            multi_model_mode=self.harness.MULTI_MODEL_PROMPT,
            selected_checkpoints=["a", "b", "c", "d"],
        )

        self.assertIn("Prompt only", summary)
        self.assertIn("4 model(s)", summary)
        self.assertIn("4 generation(s)", summary)
        self.assertIn("25 models", summary)

    def test_invalid_weight_preview_warns_without_starting_a_run(self):
        summary = self.summary(global_lora_weights="not-a-weight")

        self.assertIn("lora-tester-run-summary-warning", summary)
        self.assertIn("Correct the global weight specification", summary)


class MatrixOnlyRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.harness = RecoveryHarness()
        self.harness.cached_embeddings = {}

    def test_cleanup_decision_requires_every_cell_and_no_interrupt(self):
        state = {"cases": [{}, {}, {}]}

        self.assertEqual(
            self.harness._matrix_only_cleanup_decision(state, 3, False),
            (True, None),
        )
        allowed, reason = self.harness._matrix_only_cleanup_decision(state, 2, False)
        self.assertFalse(allowed)
        self.assertIn("2 of 3", reason)
        self.assertEqual(
            self.harness._matrix_only_cleanup_decision(state, 3, True),
            (False, "generation was interrupted"),
        )

    def test_partial_run_retains_sources_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            session = Path(temporary_root) / "session"
            session.mkdir()
            source = session / "cell-0000.png"
            source.write_bytes(b"source")
            row = session / "row-0000.png"
            row.write_bytes(b"row")
            manifest = session / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            state = {
                "session_dir": str(session),
                "manifest_path": str(manifest),
                "cases": [{}, {}],
                "cells": {0: {"path": str(source), "label": "first"}},
                "rows": [{"path": str(row)}],
                "keep_individual_images": False,
                "matrix_only_delete_allowed": False,
                "matrix_only_retention_reason": "generation completed 1 of 2 requested cells",
            }

            self.harness._cleanup_spool_files(state, [])

            self.assertTrue(source.exists())
            self.assertFalse(row.exists())
            self.assertTrue(manifest.exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "recovery-retained")
            self.assertTrue(payload["recovery_retained"])

    def test_complete_run_deletes_matrix_only_sources(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            session = Path(temporary_root) / "session"
            session.mkdir()
            source = session / "cell-0000.png"
            source.write_bytes(b"source")
            reference_row = session / "reference-row-0000.png"
            reference_row.write_bytes(b"reference")
            manifest = session / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            state = {
                "session_dir": str(session),
                "manifest_path": str(manifest),
                "cases": [{}],
                "cells": {0: {"path": str(source), "label": "first"}},
                "rows": [],
                "reference_row": {"path": str(reference_row)},
                "keep_individual_images": False,
                "matrix_only_delete_allowed": True,
            }

            self.harness._cleanup_spool_files(state, [])

            self.assertFalse(source.exists())
            self.assertFalse(reference_row.exists())
            self.assertFalse(manifest.exists())
            self.assertFalse(session.exists())


class ExtremeRunAndMatrixLayoutTests(unittest.TestCase):
    def setUp(self):
        self.harness = RecoveryHarness()

    def test_extreme_run_is_an_explicit_ten_thousand_cell_opt_in(self):
        self.assertEqual(self.harness._maximum_total_cases(False), 500)
        self.assertEqual(self.harness._maximum_total_cases(True), 10_000)

    def test_reference_row_is_included_in_every_page_estimate(self):
        state = {"matrix_margin": 5}
        reference_row = {"width": 832, "height": 1_300}
        rows = [
            {"width": 3_328, "height": 1_400},
            {"width": 3_328, "height": 1_400},
        ]

        width, height, estimated_peak = self.harness._estimate_page_peak(
            state,
            rows,
            reference_row,
        )

        self.assertEqual(width, 3_328)
        self.assertEqual(height, 1_300 + 1_400 + 1_400 + 10)
        self.assertGreater(estimated_peak, 0)

    def test_sixty_five_thousand_pixel_page_limit_is_enforced(self):
        state = {"matrix_margin": 5}
        reference_row = {"width": 832, "height": 1_300}
        rows = [{"width": 3_328, "height": 64_000}]

        _, height, estimated_peak = self.harness._estimate_page_peak(
            state,
            rows,
            reference_row,
        )

        self.assertGreater(height, self.harness.MAX_MATRIX_DIMENSION)
        self.assertEqual(estimated_peak, float("inf"))

    def test_pillow_gallery_pixel_area_limit_is_enforced(self):
        state = {"matrix_margin": 0}
        rows = [{"width": 3_082, "height": 63_904}]

        width, height, estimated_peak = self.harness._estimate_page_peak(
            state,
            rows,
        )

        self.assertLess(width, self.harness.MAX_MATRIX_DIMENSION)
        self.assertLess(height, self.harness.MAX_MATRIX_DIMENSION)
        self.assertGreater(width * height, self.harness.MAX_MATRIX_PIXELS)
        self.assertEqual(estimated_peak, float("inf"))

    def test_reference_row_is_passed_to_every_composed_page(self):
        state = {
            "matrix_margin": 0,
        }
        reference_row = {"path": "reference.png", "width": 832, "height": 1_000}
        rows = [
            {"path": f"row-{index}.png", "width": 1_000, "height": 30_000}
            for index in range(5)
        ]
        composed_references = []

        def compose_page(
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
            composed_references.append(reference_row)
            return f"matrix-page-{page_number}.png"

        self.harness._compose_matrix_page = compose_page
        page_paths = self.harness._build_matrix_pages(
            state,
            rows,
            None,
            None,
            "test",
            reference_row,
        )

        self.assertEqual(len(page_paths), 3)
        self.assertEqual(composed_references, [reference_row] * 3)

    def test_pixel_area_limit_splits_pages_before_gradio_reopens_them(self):
        state = {
            "matrix_margin": 0,
        }
        reference_row = {"path": "reference.png", "width": 3_082, "height": 1_000}
        rows = [
            {"path": f"row-{index}.png", "width": 3_082, "height": 10_000}
            for index in range(5)
        ]
        composed_pages = []

        def compose_page(
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
            composed_pages.append((len(rows), width, height, reference_row))
            return f"matrix-page-{page_number}.png"

        self.harness._compose_matrix_page = compose_page
        page_paths = self.harness._build_matrix_pages(
            state,
            rows,
            None,
            None,
            "test",
            reference_row,
        )

        self.assertEqual(len(page_paths), 3)
        self.assertEqual([page[0] for page in composed_pages], [2, 2, 1])
        self.assertTrue(
            all(
                width * height <= self.harness.MAX_MATRIX_PIXELS
                for _, width, height, _ in composed_pages
            )
        )
        self.assertEqual(
            [page[3] for page in composed_pages],
            [reference_row] * 3,
        )


class PerItemMatrixTests(unittest.TestCase):
    def setUp(self):
        self.harness = RecoveryHarness()

    @staticmethod
    def _comparison_cell(index, key, label, weight):
        return {
            "index": index,
            "item_key": key,
            "item_label": label,
            "label": f"{label} (weight {weight})",
            "weight": weight,
            "is_baseline": False,
            "width": 832,
            "height": 1216,
        }

    def test_cells_are_grouped_by_item_key_without_merging_duplicate_labels(self):
        cells = [
            self._comparison_cell(1, "folder-a/shared.safetensors", "Shared", "0.5"),
            self._comparison_cell(2, "folder-a/shared.safetensors", "Shared", "1"),
            self._comparison_cell(3, "folder-b/shared.safetensors", "Shared", "0.5"),
        ]

        groups = self.harness._group_comparison_cells(cells)

        self.assertEqual(
            [group["item_key"] for group in groups],
            ["folder-a/shared.safetensors", "folder-b/shared.safetensors"],
        )
        self.assertEqual([len(group["cells"]) for group in groups], [2, 1])

    def test_empty_dataframe_rows_are_discarded(self):
        self.assertEqual(
            self.harness._coerce_settings_rows(
                [
                    ["", "", "", "", ""],
                    ["real.safetensors", "trigger", "", "", ""],
                ]
            ),
            [["real.safetensors", "trigger", "", "", ""]],
        )

    def test_combined_matrix_is_required_until_per_item_grids_are_enabled(self):
        self.assertEqual(
            self.harness._resolve_matrix_outputs(False, False),
            (False, True),
        )
        self.assertEqual(
            self.harness._resolve_matrix_outputs(True, False),
            (True, False),
        )
        self.assertEqual(
            self.harness._resolve_matrix_outputs(True, True),
            (True, True),
        )

    def test_individual_grids_precede_the_default_combined_overview(self):
        cells = [
            self._comparison_cell(1, "first.safetensors", "First LoRA", "0.5"),
            self._comparison_cell(2, "first.safetensors", "First LoRA", "1"),
            self._comparison_cell(3, "second.safetensors", "Second LoRA", "1"),
        ]
        state = {
            "test_type": "LoRA",
            "create_per_item_grids": True,
            "create_combined_overview": True,
            "comparison_layout": self.harness.MATRIX_LAYOUT_STANDARD,
            "matrix_outputs": [],
            "matrix_cols": 1,
            "matrix_margin": 5,
            "draw_legend": True,
            "rows": [],
        }
        row_calls = []
        page_calls = []
        reference_row = {"path": "reference.png", "width": 1, "height": 1}

        def build_rows(
            state,
            selected_cells,
            columns=None,
            filename_prefix="row",
            track_as_matrix_rows=True,
        ):
            row_calls.append(
                (
                    filename_prefix,
                    [cell["index"] for cell in selected_cells],
                    columns,
                )
            )
            return [{"path": f"{filename_prefix}.png", "width": 1, "height": 1}]

        def build_pages(
            state,
            rows,
            p,
            processed,
            grid_info,
            reference_row=None,
            output_slug="combined-overview",
            output_description="combined overview",
        ):
            page_calls.append((output_slug, output_description, reference_row))
            return [Path(f"{output_slug}-page.png")]

        self.harness._build_row_strips = build_rows
        self.harness._build_matrix_pages = build_pages

        pages = self.harness._build_requested_matrices(
            state,
            cells,
            None,
            None,
            "test info",
            reference_row=reference_row,
        )

        self.assertEqual(
            [path.name for path in pages],
            [
                "lora-001-First-LoRA-page.png",
                "lora-002-Second-LoRA-page.png",
                "combined-overview-page.png",
            ],
        )
        self.assertEqual(
            row_calls,
            [
                ("lora-001-First-LoRA-row", [1, 2], 2),
                ("lora-002-Second-LoRA-row", [3], 1),
                ("combined-row", [1, 2, 3], None),
            ],
        )
        self.assertTrue(all(call[2] is reference_row for call in page_calls))
        self.assertEqual(
            [output["kind"] for output in state["matrix_outputs"]],
            ["item", "item", "combined"],
        )

    def test_combined_overview_can_be_disabled(self):
        cells = [self._comparison_cell(1, "only.safetensors", "Only", "1")]
        state = {
            "test_type": "LoRA",
            "create_per_item_grids": True,
            "create_combined_overview": False,
            "comparison_layout": self.harness.MATRIX_LAYOUT_STANDARD,
            "matrix_outputs": [],
            "matrix_cols": 1,
            "matrix_margin": 5,
            "draw_legend": True,
            "rows": [],
        }
        row_prefixes = []

        def build_rows(
            state,
            selected_cells,
            columns=None,
            filename_prefix="row",
            track_as_matrix_rows=True,
        ):
            row_prefixes.append(filename_prefix)
            return [{"path": "row.png", "width": 1, "height": 1}]

        def build_pages(
            state,
            rows,
            p,
            processed,
            grid_info,
            reference_row=None,
            output_slug="combined-overview",
            output_description="combined overview",
        ):
            return [Path(f"{output_slug}-page.png")]

        self.harness._build_row_strips = build_rows
        self.harness._build_matrix_pages = build_pages

        pages = self.harness._build_requested_matrices(
            state,
            cells,
            None,
            None,
            "test info",
        )

        self.assertEqual([path.name for path in pages], ["lora-001-Only-page.png"])
        self.assertEqual(row_prefixes, ["lora-001-Only-row"])
        self.assertEqual(
            [output["kind"] for output in state["matrix_outputs"]],
            ["item"],
        )

    def test_per_item_grid_splits_only_into_safe_horizontal_rows(self):
        cells = [
            self._comparison_cell(index, "only.safetensors", "Only", str(index))
            for index in range(1, 6)
        ]
        group = self.harness._group_comparison_cells(cells)[0]
        state = {
            "matrix_margin": 5,
            "draw_legend": True,
            "rows": [],
        }
        row_calls = []
        page_slugs = []

        def build_rows(
            state,
            selected_cells,
            columns=None,
            filename_prefix="row",
            track_as_matrix_rows=True,
        ):
            row_calls.append(
                ([cell["index"] for cell in selected_cells], columns, filename_prefix)
            )
            return [{"path": f"{filename_prefix}.png", "width": 100, "height": 100}]

        def build_pages(
            state,
            rows,
            p,
            processed,
            grid_info,
            reference_row=None,
            output_slug="combined-overview",
            output_description="combined overview",
        ):
            page_slugs.append(output_slug)
            return [Path(f"{output_slug}.png")]

        self.harness._comparison_capacity = lambda *args, **kwargs: 2
        self.harness._build_row_strips = build_rows
        self.harness._build_matrix_pages = build_pages

        pages = self.harness._build_per_item_matrix_pages(
            state,
            group,
            None,
            None,
            "test info",
            output_slug="lora-001-Only",
        )

        self.assertEqual(
            row_calls,
            [
                ([1, 2], 2, "lora-001-Only-part-001-row"),
                ([3, 4], 2, "lora-001-Only-part-002-row"),
                ([5], 1, "lora-001-Only-part-003-row"),
            ],
        )
        self.assertEqual(
            page_slugs,
            [
                "lora-001-Only-part-001",
                "lora-001-Only-part-002",
                "lora-001-Only-part-003",
            ],
        )
        self.assertEqual(len(pages), 3)

    def test_per_item_grids_are_off_by_default_for_legacy_output(self):
        cells = [self._comparison_cell(1, "only.safetensors", "Only", "1")]
        state = {
            "test_type": "LoRA",
            "create_per_item_grids": False,
            "create_combined_overview": True,
            "comparison_layout": self.harness.MATRIX_LAYOUT_STANDARD,
            "matrix_outputs": [],
        }
        row_prefixes = []

        def build_rows(
            state,
            selected_cells,
            columns=None,
            filename_prefix="row",
            track_as_matrix_rows=True,
        ):
            row_prefixes.append(filename_prefix)
            return [{"path": "row.png", "width": 1, "height": 1}]

        def build_pages(
            state,
            rows,
            p,
            processed,
            grid_info,
            reference_row=None,
            output_slug="combined-overview",
            output_description="combined overview",
        ):
            return [Path(f"{output_slug}-page.png")]

        self.harness._build_row_strips = build_rows
        self.harness._build_matrix_pages = build_pages

        pages = self.harness._build_requested_matrices(
            state,
            cells,
            None,
            None,
            "test info",
        )

        self.assertEqual([path.name for path in pages], ["combined-overview-page.png"])
        self.assertEqual(row_prefixes, ["combined-row"])
        self.assertEqual(
            [output["kind"] for output in state["matrix_outputs"]],
            ["combined"],
        )


class ComparisonLayoutTests(unittest.TestCase):
    def setUp(self):
        self.harness = RecoveryHarness()

    @staticmethod
    def _cell(index, key, weight):
        return {
            "index": index,
            "item_key": key,
            "item_label": key,
            "label": f"{key} ({weight})",
            "weight": weight,
            "is_baseline": False,
            "width": 832,
            "height": 1216,
        }

    def _groups(self):
        cells = [
            self._cell(1, "A", "0"),
            self._cell(2, "A", "0.5"),
            self._cell(3, "A", "1"),
            self._cell(4, "B", "0"),
            self._cell(5, "B", "0.5"),
            self._cell(6, "B", "1"),
        ]
        return self.harness._group_comparison_cells(cells)

    def _build_with_layout(self, layout):
        state = {
            "comparison_layout": layout,
            "matrix_margin": 5,
            "draw_legend": True,
            "rows": [],
        }
        row_calls = []

        def build_rows(
            state,
            selected_cells,
            columns=None,
            filename_prefix="row",
            track_as_matrix_rows=True,
        ):
            row_calls.append([cell["index"] for cell in selected_cells])
            return [{"path": f"{filename_prefix}.png", "width": 100, "height": 100}]

        def build_pages(
            state,
            rows,
            p,
            processed,
            grid_info,
            reference_row=None,
            output_slug="combined-overview",
            output_description="combined overview",
        ):
            return [Path(f"{output_slug}.png")]

        self.harness._build_row_strips = build_rows
        self.harness._build_matrix_pages = build_pages
        pages = self.harness._build_comparison_matrix_pages(
            state,
            self._groups(),
            None,
            None,
            "test info",
        )
        return row_calls, pages

    def test_horizontal_comparison_keeps_each_lora_in_one_row(self):
        rows, pages = self._build_with_layout(
            self.harness.MATRIX_LAYOUT_COMPARE_HORIZONTAL
        )
        self.assertEqual(rows, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(len(pages), 1)

    def test_vertical_comparison_keeps_each_lora_in_one_column(self):
        rows, pages = self._build_with_layout(
            self.harness.MATRIX_LAYOUT_COMPARE_VERTICAL
        )
        self.assertEqual(rows, [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(len(pages), 1)

    def test_weight_axis_splits_at_the_same_boundary_for_every_lora(self):
        state = {
            "comparison_layout": self.harness.MATRIX_LAYOUT_COMPARE_HORIZONTAL,
            "matrix_margin": 5,
            "draw_legend": True,
            "rows": [],
        }
        row_calls = []

        def build_rows(
            state,
            selected_cells,
            columns=None,
            filename_prefix="row",
            track_as_matrix_rows=True,
        ):
            row_calls.append([cell["index"] for cell in selected_cells])
            return [{"path": f"{filename_prefix}.png", "width": 100, "height": 100}]

        def build_pages(
            state,
            rows,
            p,
            processed,
            grid_info,
            reference_row=None,
            output_slug="combined-overview",
            output_description="combined overview",
        ):
            return [Path(f"{output_slug}.png")]

        self.harness._comparison_capacity = lambda *args, **kwargs: 2
        self.harness._build_row_strips = build_rows
        self.harness._build_matrix_pages = build_pages

        pages = self.harness._build_comparison_matrix_pages(
            state,
            self._groups(),
            None,
            None,
            "test info",
        )

        self.assertEqual(row_calls, [[1, 2], [4, 5], [3], [6]])
        self.assertEqual(len(pages), 2)

    def test_comparison_requires_identical_weight_sequences(self):
        groups = self._groups()
        groups[1]["cells"][-1]["weight"] = "1.5"
        error = self.harness._comparison_group_error(groups)
        self.assertIn("same weight values", error)

    def test_capacity_reports_a_symmetric_half_step_hint(self):
        capacity = self.harness._comparison_capacity(
            832,
            1216,
            3,
            self.harness.MATRIX_LAYOUT_COMPARE_HORIZONTAL,
            margin=5,
            draw_legend=True,
            include_reference=True,
        )
        self.assertGreater(capacity, 1)
        self.assertLessEqual(capacity, self.harness.MAX_WEIGHTS_PER_ITEM)
        hint = self.harness._comparison_range_hint(capacity)
        self.assertRegex(hint, r"^-\d+(?:\.\d+)?:\d+(?:\.\d+)?:0\.5$")

        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("Estimate: {width or '?'}", source)
        self.assertNotIn("Hi-Res output is revalidated", source)

    def test_capacity_guidance_is_hidden_for_standard_or_disabled_overview(self):
        self.assertFalse(
            self.harness._comparison_capacity_visible(
                self.harness.MATRIX_LAYOUT_STANDARD,
                False,
                True,
            )
        )
        self.assertTrue(
            self.harness._comparison_capacity_visible(
                self.harness.MATRIX_LAYOUT_COMPARE_HORIZONTAL,
                False,
                True,
            )
        )
        self.assertFalse(
            self.harness._comparison_capacity_visible(
                self.harness.MATRIX_LAYOUT_COMPARE_VERTICAL,
                True,
                False,
            )
        )


class EmbeddingModeTests(unittest.TestCase):
    def setUp(self):
        self.harness = RecoveryHarness()

    def test_embedding_prompt_supports_position_and_prompt_attention_weight(self):
        self.assertEqual(
            self.harness._compose_embedding_prompt(
                "portrait, studio light",
                "MyEmbedding",
                ["portraitStyle"],
                "0.8",
                "Start",
            ),
            "portraitStyle, (MyEmbedding:0.8), portrait, studio light",
        )
        self.assertEqual(
            self.harness._compose_embedding_prompt(
                "low quality",
                "EasyNegativeXL",
                ["bad hands"],
                "1.2",
                "End",
            ),
            "low quality, (EasyNegativeXL:1.2), bad hands",
        )

    def test_filename_fallback_does_not_duplicate_the_embedding_token(self):
        self.assertEqual(
            self.harness._compose_embedding_prompt(
                "", "MyEmbedding", ["MyEmbedding"], "1", "Start"
            ),
            "(MyEmbedding:1)",
        )

    def test_technical_embedding_name_preserves_filename_whitespace(self):
        self.assertEqual(
            self.harness._compose_embedding_prompt(
                "", "Embedding  ", ["Embedding  "], "1", "Start"
            ),
            "(Embedding  :1)",
        )
        self.assertEqual(
            self.harness._compose_embedding_prompt(
                "", "Embedding  ", ["Embedding"], "1", "Start"
            ),
            "Embedding, (Embedding  :1)",
        )

    def test_embedding_weight_rows_share_lora_range_validation(self):
        global_weights = [1.0]
        weights, error = self.harness._resolve_item_weights(
            ["embedding.safetensors", "trigger", "0", "0.5", "1.0", "0.25"],
            3,
            global_weights,
            "embedding.safetensors",
        )
        self.assertIsNone(error)
        self.assertEqual(weights, [0.5, 0.75, 1.0])

        weights, error = self.harness._resolve_item_weights(
            ["embedding.safetensors", "trigger", "1", "", "", ""],
            3,
            global_weights,
            "embedding.safetensors",
        )
        self.assertIsNone(error)
        self.assertIs(weights, global_weights)

    def test_old_four_column_embedding_rows_are_migrated(self):
        self.assertEqual(
            self.harness._coerce_embedding_settings_rows(
                [["embedding.safetensors", "0.5", "1.0", "0.25"]]
            ),
            [["embedding.safetensors", "", "0", "0.5", "1.0", "0.25"]],
        )
        self.assertEqual(
            self.harness._coerce_embedding_settings_rows(
                [["embedding.safetensors", "trigger", "0.5", "1.0", "0.25"]]
            ),
            [["embedding.safetensors", "trigger", "0", "0.5", "1.0", "0.25"]],
        )

    def test_embedding_settings_table_is_prefilled_from_metadata(self):
        self.harness.cached_embeddings = {
            "Embedding.safetensors": {"trigger_words": ["first", "second"]}
        }

        self.assertEqual(
            self.harness._build_embedding_settings_rows(["Embedding.safetensors"]),
            [["Embedding.safetensors", "first, second", "0", "", "", ""]],
        )

    def test_embedding_prompt_target_accepts_only_zero_or_one(self):
        self.assertEqual(
            self.harness._resolve_embedding_target("0"),
            (self.harness.EMBEDDING_TARGET_POSITIVE, None),
        )
        self.assertEqual(
            self.harness._resolve_embedding_target("1"),
            (self.harness.EMBEDDING_TARGET_NEGATIVE, None),
        )
        target, error = self.harness._resolve_embedding_target("2")
        self.assertIsNone(target)
        self.assertIn("must be 0", error)

    def test_none_suppresses_optional_metadata_trigger(self):
        self.assertEqual(
            self.harness._resolve_trigger_words(["metadataTrigger"], "<none>"),
            [],
        )

    def test_embedding_case_is_not_mistaken_for_reference(self):
        self.assertTrue(self.harness._case_is_baseline({"kind": "baseline"}))
        self.assertFalse(
            self.harness._case_is_baseline(
                {
                    "kind": "embedding",
                    "lora_tag_name": None,
                    "embedding_name": "MyEmbedding",
                }
            )
        )


class EmbeddingCompatibilityTests(unittest.TestCase):
    def test_inventory_intersects_both_forge_databases_and_excludes_single_tensors(
        self,
    ):
        methods = load_class_methods(
            "EmbeddingMetadataReader",
            "find_compatible_embeddings",
            "is_dual_encoder_embedding",
            "support_error",
        )

        class ReaderHarness:
            pass

        ReaderHarness.support_error = staticmethod(methods["support_error"])
        ReaderHarness.is_dual_encoder_embedding = staticmethod(
            methods["is_dual_encoder_embedding"]
        )
        ReaderHarness.find_compatible_embeddings = classmethod(
            methods["find_compatible_embeddings"]
        )

        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            shared_stub = methods["find_compatible_embeddings"].__globals__["shared"]
            shared_stub.cmd_opts.embeddings_dir = str(root)

            def embedding(name, dual=True):
                vectors = {"clip_l": object(), "clip_g": object()} if dual else object()
                return SimpleNamespace(
                    vec=vectors,
                    filename=str(root / "XL" / f"{name}.safetensors"),
                    vectors=4,
                    shape=2048,
                )

            compatible_l = embedding("Compatible")
            compatible_g = embedding("Compatible")
            databases = [
                SimpleNamespace(
                    word_embeddings={
                        "Compatible": compatible_l,
                        "SingleTensor": embedding("SingleTensor", dual=False),
                        "OnlyClipL": embedding("OnlyClipL"),
                    }
                ),
                SimpleNamespace(
                    word_embeddings={
                        "Compatible": compatible_g,
                        "SingleTensor": embedding("SingleTensor", dual=False),
                    }
                ),
            ]
            model = SimpleNamespace(
                is_sdxl=True,
                text_processing_engine_l=SimpleNamespace(embeddings=databases[0]),
                text_processing_engine_g=SimpleNamespace(embeddings=databases[1]),
            )

            inventory, status = ReaderHarness.find_compatible_embeddings(
                refresh=False,
                model=model,
            )

        self.assertEqual(list(inventory), [str(Path("XL") / "Compatible.safetensors")])
        self.assertEqual(inventory[next(iter(inventory))]["name"], "Compatible")
        self.assertEqual(
            inventory[next(iter(inventory))]["trigger_words"], ["Compatible"]
        )
        self.assertEqual(inventory[next(iter(inventory))]["trigger_source"], "filename")
        self.assertIn("Found 1 embedding", status)

    def test_only_true_clip_l_and_clip_g_embeddings_pass_the_filter(self):
        method = load_class_methods(
            "EmbeddingMetadataReader", "is_dual_encoder_embedding"
        )["is_dual_encoder_embedding"]

        dual = type("Embedding", (), {"vec": {"clip_l": 1, "clip_g": 2}})()
        single = type("Embedding", (), {"vec": object()})()
        incomplete = type("Embedding", (), {"vec": {"clip_l": 1}})()

        self.assertTrue(method(dual))
        self.assertFalse(method(single))
        self.assertFalse(method(incomplete))

    def test_only_dual_encoder_sdxl_models_are_accepted(self):
        method = load_class_methods("EmbeddingMetadataReader", "support_error")[
            "support_error"
        ]

        class Model:
            pass

        self.assertIn("No checkpoint", method(None))

        model = Model()
        model.is_sdxl = False
        self.assertIn("SDXL-compatible", method(model))

        model.is_sdxl = True
        model.text_processing_engine_l = type("Engine", (), {"embeddings": object()})()
        model.text_processing_engine_g = type("Engine", (), {"embeddings": object()})()
        self.assertIsNone(method(model))


class UiCallbackContractTests(unittest.TestCase):
    def test_ui_uses_four_workflow_tabs_without_nested_accordions(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")

        for tab_label in (
            "1. Selection",
            "2. Models & Run Size",
            "3. Grid",
            "4. Output & Safety",
        ):
            self.assertIn(f'with gr.Tab("{tab_label}"):', source)
        self.assertNotIn("with gr.Accordion(", source)

    def test_removed_ram_watchdog_is_absent_from_runtime_source(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")

        for removed_symbol in (
            "RamMonitor",
            "RAM_WATCHDOG_ENABLED",
            "psutil",
            "adaptive_ram",
            "ram_budget",
            "minimum_free_ram_gb",
        ):
            self.assertNotIn(removed_symbol, source)

    def test_ui_component_order_matches_processing_callback_signatures(self):
        tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
        script_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LoRaTesterScript"
        )
        methods = {
            node.name: node
            for node in script_class.body
            if isinstance(node, ast.FunctionDef)
        }

        ui_return = next(
            node
            for node in reversed(methods["ui"].body)
            if isinstance(node, ast.Return)
        )
        ui_names = [element.id for element in ui_return.value.elts]
        before_names = [
            argument.arg for argument in methods["before_process"].args.args[2:]
        ]
        post_names = [argument.arg for argument in methods["postprocess"].args.args[3:]]

        self.assertEqual(before_names, ui_names)
        self.assertEqual(post_names, ui_names)

    def test_selection_driven_dataframes_hide_gradio_dynamic_blank_rows(self):
        tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
        script_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LoRaTesterScript"
        )
        ui_method = next(
            node
            for node in script_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "ui"
        )
        dataframes = [
            node
            for node in ast.walk(ui_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Dataframe"
        ]
        self.assertEqual(len(dataframes), 2)
        for dataframe in dataframes:
            row_count = next(
                keyword.value
                for keyword in dataframe.keywords
                if keyword.arg == "row_count"
            )
            self.assertEqual(row_count.elts[0].value, 0)
            self.assertEqual(row_count.elts[1].value, "dynamic")

        javascript = JAVASCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("hideDynamicBlankRows(root)", javascript)
        self.assertIn("hideRowInsertionControls(root)", javascript)
        self.assertIn("releaseEditorForScroll", javascript)
        self.assertIn('root.addEventListener("wheel"', javascript)
        self.assertIn("editor.blur()", javascript)
        self.assertIn("lora-tester-empty-dataframe-row", javascript)


if __name__ == "__main__":
    unittest.main()
