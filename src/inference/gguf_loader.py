"""Load the one exported GGUF into the fixed PC model without external engines."""

import json
from typing import Any

import gguf
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from common.errors import EvaiError
from common.hashing import compute_file_sha256
from common.model_contract import MODEL_ID, read_training_contract, validate_model_config
from common.model_storage import model_storage_lock
from common.paths import GGUF_MODEL_FILE, MODEL_DIR
from inference.cuda_operators import bind_cuda_operators

DEQUANTIZE_ROWS = 256


def load_gguf_model(device: str) -> Any:
    with model_storage_lock():
        return _load_gguf_model(device)


def _load_gguf_model(device: str) -> Any:
    contract = read_training_contract(MODEL_DIR)
    if contract is None:
        raise EvaiError("GGUF runtime requires the single model's training provenance")
    metadata = json.loads(GGUF_MODEL_FILE.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        metadata["base_model"] != MODEL_ID
        or metadata["source_sha256"] != contract["weights_sha256"]
        or metadata["sha256"] != compute_file_sha256(GGUF_MODEL_FILE)
    ):
        raise EvaiError("GGUF does not match the current trained model")
    config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    validate_model_config(config.to_dict())
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.requires_grad_(False)
    model.eval()
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LFM2, config.num_hidden_layers)
    parameters = {}
    for name, parameter in model.named_parameters():
        target = names.get_name(name, try_suffixes=(".weight",))
        if target is None or target in parameters:
            raise EvaiError(f"Ambiguous GGUF model parameter mapping: {name}")
        parameters[target] = parameter
    reader = gguf.GGUFReader(GGUF_MODEL_FILE)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if set(tensors) != set(parameters):
        raise EvaiError("GGUF tensors do not match the fixed model's complete parameter set")
    for name, parameter in parameters.items():
        tensor = tensors[name]
        expected = tuple(parameter.shape)
        stored = tuple(int(size) for size in reversed(tensor.shape))
        if len(expected) == 3 and expected[1] == 1:
            expected = (expected[0], expected[2])
        if stored != expected:
            raise EvaiError(f"GGUF tensor shape differs from the fixed model: {name}")
        for start in range(0, parameter.shape[0], DEQUANTIZE_ROWS):
            end = min(start + DEQUANTIZE_ROWS, parameter.shape[0])
            decoded = np.array(
                gguf.dequantize(tensor.data[start:end], tensor.tensor_type), copy=True
            )
            parameter[start:end].copy_(
                torch.from_numpy(decoded).reshape(parameter[start:end].shape)
            )
    bind_cuda_operators(model)
    return model.to(device)
