from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from common.errors import EvaiError
from common.model_contract import verify_backbone
from common.model_storage import model_storage_lock
from common.paths import MODEL_DIR
from inference.cuda_operators import bind_cuda_operators


def load_tokenizer(model_dir: Path = MODEL_DIR) -> Any:
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def load_causal_lm(
    model_dir: Path = MODEL_DIR, device: str | None = None, dtype: Any = None,
    *, expected_sha: str | None = None,
) -> Any:
    options: dict[str, Any] = {"local_files_only": True}
    options["attn_implementation"] = "sdpa"
    if dtype is not None:
        options["dtype"] = dtype
    if device is not None:
        options["device_map"] = device
    with model_storage_lock():
        digest = verify_backbone(model_dir)
        if expected_sha is not None and digest != expected_sha:
            raise EvaiError("The model changed while preparing its training update")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **options)
    bind_cuda_operators(model)
    return model


def load_model_and_tokenizer(
    model_dir: Path = MODEL_DIR, device: str | None = None
) -> tuple[Any, Any]:
    return load_causal_lm(model_dir, device), load_tokenizer(model_dir)
