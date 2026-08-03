# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 vibecodingtoolmaker

import ast
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
    namespace = {"Path": Path, "json": json, "os": os, "print": print}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return {name: namespace[name] for name in requested}


class RecoveryHarness:
    pass


METHODS = load_methods(
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
            manifest = session / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            state = {
                "session_dir": str(session),
                "manifest_path": str(manifest),
                "cases": [{}],
                "cells": {0: {"path": str(source), "label": "first"}},
                "rows": [],
                "keep_individual_images": False,
                "matrix_only_delete_allowed": True,
            }

            self.harness._cleanup_spool_files(state, [])

            self.assertFalse(source.exists())
            self.assertFalse(manifest.exists())
            self.assertFalse(session.exists())


if __name__ == "__main__":
    unittest.main()
