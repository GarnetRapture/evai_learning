"""The exact Korean EVAI backbone; no trainer or runtime may select another model."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.errors import EvaiError

MODEL_ID = "LiquidAI/LFM2.5-230M-Base"
MODEL_REVISION = "9d2be5519834990d30996f878b6771cccbd24f2c"
WEIGHTS_SHA256 = "e91eb22c0aeae0bcbea8ade56f5cfe3cf91bca0c34e859adacae8f4445416fe6"
CONTEXT_LENGTH = 32768
DATASET_VERSION = "2.0.0"
TRAINING_LANGUAGES = {"ko": "kr", "en": "en", "zh_tw": "zh_tw"}
ADAPTER_FORMAT = "evai.lora.curriculum.v2"
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


@lru_cache(maxsize=4)
def _verify_weights(path: str, size: int, modified_ns: int) -> str:
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != WEIGHTS_SHA256:
        raise EvaiError(f"Backbone checksum differs from {MODEL_ID}@{MODEL_REVISION}: {actual}")
    return actual


def verify_backbone(directory: Path) -> str:
    validate_model_config(json.loads((directory / "config.json").read_text(encoding="utf-8")))
    path = directory / "model.safetensors"
    stat = path.stat()
    return _verify_weights(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
