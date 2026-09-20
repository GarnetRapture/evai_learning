"""The exact Korean EVAI backbone; no trainer or runtime may select another model."""

import json
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.hashing import compute_file_sha256
from common.model_storage import MODEL_MARKER, model_storage_lock, read_model_marker

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHTS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
CONTEXT_LENGTH = 32768
DATASET_VERSION = "2.1.0"
TRAINING_LANGUAGES = {"ko": "kr", "en": "en", "zh_tw": "zh_tw"}
MAX_MODEL_BYTES = 1_600_000_000
TRAINING_CONTRACT_FILE = MODEL_MARKER
BEHAVIOR_FIELDS = ("interpretation", "emotion", "intention", "decision", "action")


def validate_model_config(config: dict[str, Any]) -> None:
    expected = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise EvaiError(f"{MODEL_ID} requires {key}={value}, got {config.get(key)}")
    if config.get("max_position_embeddings", 0) < CONTEXT_LENGTH:
        raise EvaiError(f"Backbone must reach the {CONTEXT_LENGTH} token training range")


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
