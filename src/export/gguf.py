"""Streaming GGUF conversion of the single trained 230M model."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
from safetensors import safe_open

from common.errors import EvaiError
from common.hashing import compute_file_sha256
from common.model_contract import CONTEXT_LENGTH, MODEL_ID, read_training_contract, verify_backbone
from common.paths import GGUF_DIR, GGUF_MODEL_FILE, MODEL_DIR
from inference.model_loader import load_tokenizer


@dataclass(frozen=True)
class TensorExport:
    source_name: str
    target_name: str
    shape: tuple[int, ...]
    quantization: gguf.GGMLQuantizationType
    convolution: bool = False


def _tensor_plan(path: Path) -> list[TensorExport]:
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LFM2, 14)
    plans = []
    with safe_open(path, framework="pt", device="cpu") as source:
        for key in source.keys():
            name = key
            target = names.get_name(name, try_suffixes=(".weight",))
            if target is None:
                raise EvaiError(f"LFM2 tensor has no GGUF mapping: {name}")
            shape = tuple(source.get_slice(key).get_shape())
            convolution = ".conv.conv.weight" in name
            if convolution:
                shape = (shape[0], shape[2])
            quantization = (
                gguf.GGMLQuantizationType.F32
                if len(shape) == 1 or convolution
                else gguf.GGMLQuantizationType.Q8_0
            )
            plans.append(TensorExport(key, target, shape, quantization, convolution))
    return plans


def _write_tensors(writer: Any, path: Path, plans: list[TensorExport]) -> None:
    for plan in plans:
        block, size = gguf.GGML_QUANT_SIZES[plan.quantization]
        byte_shape = (*plan.shape[:-1], plan.shape[-1] // block * size)
        writer.add_tensor_info(
            plan.target_name,
            byte_shape,
            np.dtype(np.uint8),
            math.prod(plan.shape) // block * size,
            raw_dtype=plan.quantization,
        )
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        with safe_open(path, framework="pt", device="cpu") as source:
            for plan in plans:
                tensor = source.get_tensor(plan.source_name)
                if plan.convolution:
                    tensor = tensor.squeeze(1)
                array = tensor.to(dtype=torch.float32).numpy()
                encoded = gguf.quantize(array, plan.quantization)
                writer.write_tensor_data(encoded)
                del encoded, array, tensor
    finally:
        writer.close()


def _add_tokenizer(writer: Any, config: dict[str, Any]) -> None:
    raw = json.loads((MODEL_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    by_id = {index: text for text, index in raw["model"]["vocab"].items()}
    added = {token["id"]: token for token in raw["added_tokens"]}
    tokens, types = [], []
    for index in range(config["vocab_size"]):
        if index in added:
            entry = added[index]
            tokens.append(entry["content"])
            types.append(
                gguf.TokenType.CONTROL if entry["special"] else gguf.TokenType.USER_DEFINED
            )
        elif index in by_id:
            tokens.append(by_id[index])
            types.append(gguf.TokenType.NORMAL)
        else:
            tokens.append(f"[PAD{index}]")
            types.append(gguf.TokenType.UNUSED)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("lfm2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges([" ".join(pair) for pair in raw["model"]["merges"]])
    writer.add_bos_token_id(config["bos_token_id"])
    writer.add_eos_token_id(config["eos_token_id"])
    writer.add_pad_token_id(config["pad_token_id"])
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    writer.add_chat_template(load_tokenizer().chat_template)


def export_model_gguf() -> Path:
    if read_training_contract(MODEL_DIR) is None:
        raise EvaiError("Train the single model before GGUF conversion")
    source_sha = verify_backbone(MODEL_DIR)
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    output = GGUF_MODEL_FILE
    metadata = output.with_suffix(".json")
    if output.exists() and metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["source_sha256"] == source_sha:
            if compute_file_sha256(output) != saved["sha256"]:
                raise EvaiError(f"Existing GGUF is corrupted: {output}")
            return output
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    source = MODEL_DIR / "model.safetensors"
    plans = _tensor_plan(source)
    writer = gguf.GGUFWriter(output, "lfm2")
    writer.add_name(MODEL_ID)
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_context_length(CONTEXT_LENGTH)
    writer.add_embedding_length(config["hidden_size"])
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_feed_forward_length(config["block_ff_dim"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(
        [
            config["num_key_value_heads"] if kind == "full_attention" else 0
            for kind in config["layer_types"]
        ]
    )
    writer.add_layer_norm_rms_eps(config["norm_eps"])
    writer.add_shortconv_l_cache(config["conv_L_cache"])
    writer.add_rope_dimension_count(config["hidden_size"] // config["num_attention_heads"])
    writer.add_rope_freq_base(config["rope_parameters"]["rope_theta"])
    writer.add_vocab_size(config["vocab_size"])
    _add_tokenizer(writer, config)
    _write_tensors(writer, source, plans)
    metadata.write_text(
        json.dumps(
            {
                "base_model": MODEL_ID,
                "source_sha256": source_sha,
                "sha256": compute_file_sha256(output),
                "size_bytes": output.stat().st_size,
                "quantization": "Q8_0",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
