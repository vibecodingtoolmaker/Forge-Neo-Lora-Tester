# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 vibecodingtoolmaker

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file


MODULE_PATH = Path(__file__).parents[1] / "forge_neo_model_family.py"
SPEC = importlib.util.spec_from_file_location(
    "forge_neo_model_family_test", MODULE_PATH
)
MODEL_FAMILY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL_FAMILY)


class ForgeModelFamilyTests(unittest.TestCase):
    def test_explicit_ui_value_wins_over_saved_option(self):
        family = MODEL_FAMILY.detect_model_family(
            "XL",
            SimpleNamespace(forge_preset="flux"),
        )

        self.assertEqual(family.preset, "xl")
        self.assertTrue(family.supports_sdxl_dual_clip_embeddings)
        self.assertIn("Pony", family.label)

    def test_saved_option_is_used_without_explicit_value(self):
        family = MODEL_FAMILY.detect_model_family(
            forge_options=SimpleNamespace(forge_preset="qwen")
        )

        self.assertEqual(family.preset, "qwen")
        self.assertFalse(family.supports_sdxl_dual_clip_embeddings)

    def test_unknown_future_preset_is_safe_by_default(self):
        family = MODEL_FAMILY.detect_model_family("future-architecture")

        self.assertEqual(family.preset, "future-architecture")
        self.assertFalse(family.supports_sdxl_dual_clip_embeddings)

    def test_header_scan_accepts_only_clip_l_plus_clip_g(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            compatible = root / "XL" / "Compatible.safetensors"
            incompatible = root / "SD15.safetensors"
            unreadable = root / "Broken.safetensors"
            compatible.parent.mkdir()
            save_file(
                {
                    "clip_l": torch.zeros((4, 768)),
                    "clip_g": torch.zeros((4, 1280)),
                },
                str(compatible),
            )
            compatible.with_suffix(".json").write_text(
                json.dumps({"activation text": "portraitStyle"}),
                encoding="utf-8",
            )
            save_file({"emb_params": torch.zeros((4, 768))}, str(incompatible))
            unreadable.write_bytes(b"not a Safetensors file")

            inventory = MODEL_FAMILY.scan_sdxl_dual_clip_embeddings(root)

        relative = str(Path("XL") / "Compatible.safetensors")
        self.assertEqual(list(inventory.embeddings), [relative])
        self.assertEqual(inventory.embeddings[relative]["name"], "Compatible")
        self.assertEqual(inventory.embeddings[relative]["vectors"], 4)
        self.assertEqual(inventory.embeddings[relative]["shape"], 2048)
        self.assertEqual(
            inventory.embeddings[relative]["trigger_words"], ["portraitStyle"]
        )
        self.assertEqual(
            inventory.embeddings[relative]["trigger_source"], "Compatible.json"
        )
        self.assertEqual(inventory.scanned_files, 3)
        self.assertEqual(inventory.unreadable_files, 1)

    def test_embedding_trigger_prefers_json_activation_text(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            embedding = Path(temporary_root) / "DifferentFilename.safetensors"
            embedding.touch()
            embedding.with_suffix(".json").write_text(
                json.dumps({"activation text": "actualTrigger"}),
                encoding="utf-8",
            )
            embedding.with_suffix(".civitai.info").write_text(
                json.dumps({"trainedWords": ["secondaryTrigger"]}),
                encoding="utf-8",
            )

            triggers, source = MODEL_FAMILY.read_embedding_trigger_words(embedding)

        self.assertEqual(triggers, ["actualTrigger"])
        self.assertEqual(source, "DifferentFilename.json")

    def test_embedding_trigger_uses_filename_when_json_has_no_trigger(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            embedding = Path(temporary_root) / "Embedding.safetensors"
            embedding.touch()
            embedding.with_suffix(".json").write_text("{}", encoding="utf-8")
            embedding.with_suffix(".civitai.info").write_text(
                json.dumps({"trainedWords": ["first", "second"]}),
                encoding="utf-8",
            )

            triggers, source = MODEL_FAMILY.read_embedding_trigger_words(embedding)

        self.assertEqual(triggers, ["Embedding"])
        self.assertEqual(source, "filename")

    def test_embedding_trigger_falls_back_to_filename_without_extension(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            embedding = Path(temporary_root) / "Fallback Name.safetensors"
            embedding.touch()
            embedding.with_suffix(".json").write_text("not-json", encoding="utf-8")

            triggers, source = MODEL_FAMILY.read_embedding_trigger_words(embedding)

        self.assertEqual(triggers, ["Fallback Name"])
        self.assertEqual(source, "filename")


if __name__ == "__main__":
    unittest.main()
