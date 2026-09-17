"""The exact Korean EVAI backbone; no trainer or runtime may select another model."""

import json
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.hashing import compute_file_sha256
from common.model_storage import MODEL_MARKER, model_storage_lock, read_model_marker

MODEL_ID = "LiquidAI/LFM2.5-230M-Base"
MODEL_REVISION = "9d2be5519834990d30996f878b6771cccbd24f2c"
WEIGHTS_SHA256 = "e91eb22c0aeae0bcbea8ade56f5cfe3cf91bca0c34e859adacae8f4445416fe6"
CONTEXT_LENGTH = 32768
DATASET_VERSION = "2.1.0"
TRAINING_LANGUAGES = {"ko": "kr", "en": "en", "zh_tw": "zh_tw"}
MAX_MODEL_BYTES = 700_000_000
TRAINING_CONTRACT_FILE = MODEL_MARKER
BEHAVIOR_FIELDS = ("interpretation", "emotion", "intention", "decision", "action")


def validate_model_config(config: dict[str, Any]) -> None:
    expected = {
        "model_type": "lfm2",
        "hidden_size": 1024,
        "num_hidden_layers": 14,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 65536,
        "tie_word_embeddings": True,
        "conv_L_cache": 3,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise EvaiError(f"{MODEL_ID} requires {key}={value}, got {config.get(key)}")
    layers = config.get("layer_types", [])
    if layers != [
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    ]:
        raise EvaiError("Backbone must retain the exact 8 LIV / 6 GQA layer order")


def _verify_weights(path: Path) -> str:
    actual = compute_file_sha256(path)
    if actual != WEIGHTS_SHA256:
        raise EvaiError(f"Backbone checksum differs from {MODEL_ID}@{MODEL_REVISION}: {actual}")
    return actual


def verify_backbone(directory: Path) -> str:
    with model_storage_lock():
        validate_model_config(json.loads((directory / "config.json").read_text(encoding="utf-8")))
        path = directory / "model.safetensors"
        contract = read_training_contract(directory)
        if contract is None:
            return _verify_weights(path)
        actual = compute_file_sha256(path)
        if actual != contract["weights_sha256"]:
            raise EvaiError("Trained model weights differ from their training provenance")
        return actual


def read_training_contract(directory: Path) -> dict[str, Any] | None:
    contract = read_model_marker(directory)
    if contract is None:
        return None
    if (
        contract.get("base_model") != MODEL_ID
        or contract.get("origin_sha256") != WEIGHTS_SHA256
        or contract.get("training_mode") != "full_sft"
        or not contract.get("spirits")
        or not contract.get("weights_sha256")
    ):
        raise EvaiError("Invalid single-model training provenance")
    return contract
