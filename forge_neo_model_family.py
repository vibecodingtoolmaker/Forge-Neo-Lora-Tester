# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 vibecodingtoolmaker

"""Small, reusable Forge Neo UI-preset and model-family helpers.

The selected Forge UI preset is available before a checkpoint is loaded.  Extensions
can therefore use it as an early capability hint, while retaining runtime model
validation for operations that must be exact.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForgeModelFamily:
    """Capabilities associated with one Forge Neo UI preset."""

    preset: str
    label: str
    supports_sdxl_dual_clip_embeddings: bool = False


@dataclass
class EmbeddingHeaderInventory:
    """Header-only textual-inversion inventory; no tensor data is loaded."""

    embeddings: dict[str, dict]
    scanned_files: int = 0
    unreadable_files: int = 0


_MODEL_FAMILIES = {
    "sd": ForgeModelFamily("sd", "Stable Diffusion 1.x"),
    "xl": ForgeModelFamily(
        "xl",
        "SDXL family (SDXL, Pony, Illustrious)",
        supports_sdxl_dual_clip_embeddings=True,
    ),
    "flux": ForgeModelFamily("flux", "FLUX.1"),
    "klein": ForgeModelFamily("klein", "FLUX.2 Klein"),
    "qwen": ForgeModelFamily("qwen", "Qwen-Image"),
    "lumina": ForgeModelFamily("lumina", "Lumina-Image-2.0"),
    "zit": ForgeModelFamily("zit", "Z-Image-Turbo"),
    "wan": ForgeModelFamily("wan", "Wan 2.2"),
    "anima": ForgeModelFamily("anima", "Anima"),
    "ernie": ForgeModelFamily("ernie", "ERNIE-Image"),
    "pid": ForgeModelFamily("pid", "PiD"),
    "krea": ForgeModelFamily("krea", "KREA 2"),
}

_TRIGGER_FIELDS = (
    "activation text",
    "activation_text",
    "trigger_words",
    "triggerWords",
    "trainedWords",
)


def normalize_forge_preset(value) -> str:
    """Normalize a dropdown string or ``PresetArch`` enum member."""

    if value is None:
        return ""
    enum_name = getattr(value, "name", None)
    if enum_name:
        value = enum_name
    return str(value).strip().casefold()


def resolve_forge_preset(explicit_preset=None, forge_options=None) -> str:
    """Resolve the current preset, preferring an explicit browser/UI value.

    ``forge_options`` can be supplied by another extension for easy unit testing.
    When omitted, the import of Forge's shared options remains lazy so this module
    can also be imported by lightweight tooling outside a running Forge process.
    """

    preset = normalize_forge_preset(explicit_preset)
    if preset:
        return preset

    if forge_options is None:
        try:
            from modules import shared

            forge_options = shared.opts
        except (ImportError, AttributeError):
            forge_options = None

    return normalize_forge_preset(getattr(forge_options, "forge_preset", None))


def detect_model_family(explicit_preset=None, forge_options=None) -> ForgeModelFamily:
    """Map Forge's selected UI preset to a reusable capability description."""

    preset = resolve_forge_preset(explicit_preset, forge_options)
    if preset in _MODEL_FAMILIES:
        return _MODEL_FAMILIES[preset]
    label = f"Unknown Forge preset ({preset})" if preset else "Unknown Forge preset"
    return ForgeModelFamily(preset, label)


def _normalize_trigger_words(value) -> list[str]:
    """Normalize the common scalar/list trigger formats used by sidecar files."""

    values = value if isinstance(value, (list, tuple)) else [value]
    triggers = []
    for item in values:
        if item is None or isinstance(item, (dict, list, tuple)):
            continue
        trigger = str(item).strip()
        if trigger and trigger not in triggers:
            triggers.append(trigger)
    return triggers


def read_embedding_trigger_words(embedding_path) -> tuple[list[str], str]:
    """Read an Embedding trigger from its JSON sidecar or use its filename.

    Civitai Helper writes the explicit ``activation text`` to the regular
    ``.json`` sidecar. Unlike LoRA metadata resolution, an absent or unusable
    Embedding JSON intentionally falls straight back to the filename stem.
    """

    path = Path(embedding_path)
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            with sidecar.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            for field in _TRIGGER_FIELDS:
                triggers = _normalize_trigger_words(data.get(field))
                if triggers:
                    return triggers, sidecar.name

    return [path.stem], "filename"


def scan_sdxl_dual_clip_embeddings(embedding_root) -> EmbeddingHeaderInventory:
    """Find Safetensors files containing both SDXL CLIP tensors.

    ``safe_open().keys()`` and ``get_slice().get_shape()`` read only the
    Safetensors header.  The function intentionally ignores pickle-based ``.pt``
    and ``.bin`` files; Forge's loaded embedding database remains the authority for
    those formats.
    """

    root = Path(embedding_root).resolve()
    inventory = EmbeddingHeaderInventory(embeddings={})
    if not root.is_dir():
        return inventory

    from safetensors import SafetensorError, safe_open

    for candidate in sorted(root.rglob("*"), key=lambda path: str(path).casefold()):
        if not candidate.is_file():
            continue
        if candidate.suffix.casefold() not in {".safetensors", ".sft"}:
            continue
        if ".preview." in candidate.name.casefold():
            continue

        inventory.scanned_files += 1
        try:
            path = candidate.resolve()
            relative_path = path.relative_to(root)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if not {"clip_l", "clip_g"}.issubset(keys):
                    continue
                clip_l_shape = tuple(handle.get_slice("clip_l").get_shape())
                clip_g_shape = tuple(handle.get_slice("clip_g").get_shape())
        except (OSError, RuntimeError, SafetensorError, ValueError):
            inventory.unreadable_files += 1
            continue

        vectors = clip_g_shape[0] if clip_g_shape else 0
        combined_shape = (
            clip_l_shape[-1] + clip_g_shape[-1] if clip_l_shape and clip_g_shape else 0
        )
        inventory.embeddings[str(relative_path)] = {
            "name": path.stem,
            "vectors": int(vectors),
            "shape": int(combined_shape),
            "source": "safetensors-header",
        }
        trigger_words, trigger_source = read_embedding_trigger_words(path)
        inventory.embeddings[str(relative_path)].update(
            {
                "trigger_words": trigger_words,
                "trigger_source": trigger_source,
            }
        )

    return inventory
