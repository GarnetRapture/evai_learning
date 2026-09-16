from pathlib import Path
from typing import Any

from common.paths import MODEL_DIR


def load_tokenizer(model_dir: Path = MODEL_DIR) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def load_causal_lm(
    model_dir: Path = MODEL_DIR, device: str | None = None, dtype: Any = None
) -> Any:
    from transformers import AutoModelForCausalLM

    options: dict[str, Any] = {"local_files_only": True}
    if dtype is not None:
        options["dtype"] = dtype
    if device is not None:
        options["device_map"] = device
    return AutoModelForCausalLM.from_pretrained(str(model_dir), **options)


def load_model_and_tokenizer(
    model_dir: Path = MODEL_DIR, device: str | None = None
) -> tuple[Any, Any]:
    return load_causal_lm(model_dir, device), load_tokenizer(model_dir)
