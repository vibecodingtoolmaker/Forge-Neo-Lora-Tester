# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 vibecodingtoolmaker

import ast
import gc
import json
import os
import tempfile
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).parents[1] / "scripts" / "lora_tester.py"


def load_methods(*method_names):
    """Load selected class methods without importing Forge-specific modules."""

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    script_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LoRaTesterScript"
    )
    requested = set(method_names)
    functions = []
    for node in script_class.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in requested:
            node.decorator_list = []
            functions.append(node)

    missing = requested.difference(node.name for node in functions)
    if missing:
        raise AssertionError(f"Methods not found: {sorted(missing)}")

    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {
        "Path": Path,
        "json": json,
        "os": os,
        "print": print,
        "MIB": 1024 ** 2,
        "gc": gc,
    }
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return {name: namespace[name] for name in requested}


class RecoveryHarness:
    pass


METHODS = load_methods(
    "_maximum_total_cases",
    "_estimate_page_peak",
    "_build_matrix_pages",
    "_matrix_only_cleanup_decision",
    "_safe_unlink",
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
RecoveryHarness._estimate_page_peak = METHODS["_estimate_page_peak"]
RecoveryHarness._build_matrix_pages = METHODS["_build_matrix_pages"]
RecoveryHarness._effective_available = staticmethod(lambda state: 0)
RecoveryHarness.MAX_TOTAL_CASES = 500
RecoveryHarness.MAX_EXTREME_TOTAL_CASES = 10_000
RecoveryHarness.MAX_MATRIX_DIMENSION = 65_000
RecoveryHarness.PAGE_SAFETY_FACTOR = 1.35


class MatrixOnlyRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.harness = RecoveryHarness()

    def test_cleanup_decision_requires_every_cell_and_no_interrupt(self):
        state = {"cases": [{}, {}, {}], "ram_stop_reason": None}

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
        state = {"matrix_margin": 5, "maximum_row_peak_bytes": 0}
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
        state = {"matrix_margin": 5, "maximum_row_peak_bytes": 0}
        reference_row = {"width": 832, "height": 1_300}
        rows = [{"width": 3_328, "height": 64_000}]

        _, height, estimated_peak = self.harness._estimate_page_peak(
            state,
            rows,
            reference_row,
        )

        self.assertGreater(height, self.harness.MAX_MATRIX_DIMENSION)
        self.assertEqual(estimated_peak, float("inf"))

    def test_reference_row_is_passed_to_every_composed_page(self):
        state = {
            "matrix_margin": 0,
            "maximum_row_peak_bytes": 0,
            "adaptive_ram": False,
            "minimum_free_bytes": 0,
        }
        reference_row = {"path": "reference.png", "width": 832, "height": 1_000}
        rows = [
            {"path": f"row-{index}.png", "width": 3_328, "height": 30_000}
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


if __name__ == "__main__":
    unittest.main()
