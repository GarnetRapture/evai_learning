"""One frozen backbone and one independently trained selected spirit delta."""

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from common.device import select_torch_device
from common.errors import EvaiError
from common.model_contract import ADAPTER_FORMAT, MODEL_ID, WEIGHTS_SHA256
from common.paths import ADAPTERS_DIR
from inference.generation import (
    DEFAULT_GENERATION_SETTINGS,
    GenerationSettings,
    generate_from_messages,
)
from inference.model_loader import load_causal_lm, load_tokenizer

LORA_LINEAR_SUFFIXES = ("in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "w1", "w2", "w3")


def measured_lora_targets(model: Any) -> list[str]:
    from common.model_contract import validate_model_config

    validate_model_config(model.config.to_dict())
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
    ]
    if not targets or any(name.rsplit(".", 1)[-1] not in LORA_LINEAR_SUFFIXES for name in targets):
        raise EvaiError("Unrecognized linear layout in locked LFM2.5 backbone")
    return targets


def read_adapter_contract(path: Path, persona_id: str) -> dict[str, Any]:
    marker = path / "evai_adapter.json"
    if not marker.is_file():
        raise EvaiError(f"Adapter has no corrected curriculum provenance: {path}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if (
        metadata.get("format") != ADAPTER_FORMAT
        or metadata.get("spirit") != persona_id
        or metadata.get("base_model") != MODEL_ID
        or metadata.get("base_weights_sha256") != WEIGHTS_SHA256
    ):
        raise EvaiError(f"Adapter violates fixed base/persona contract: {path}")
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        config.get("peft_type") != "LORA"
        or config.get("bias") != "none"
        or config.get("modules_to_save")
        or config.get("use_dora")
    ):
        raise EvaiError(f"Adapter must contain only independent LoRA weights: {path}")
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(name.rsplit(".", 1)[-1] not in LORA_LINEAR_SUFFIXES for name in targets)
    ):
        raise EvaiError(f"Adapter changes unsupported backbone parameters: {path}")
    return metadata


class SpiritRuntime:
    """Serialize selection and generation, retaining at most one adapter in GPU memory."""

    def __init__(self, persona_ids: list[str], adapters_root: Path | None = None) -> None:
        from peft import PeftModel

        if not persona_ids or len(set(persona_ids)) != len(persona_ids):
            raise EvaiError("Runtime requires unique registered spirit identifiers")
        self.adapters_root = adapters_root if adapters_root is not None else ADAPTERS_DIR
        self._known = list(persona_ids)
        self._lock = threading.RLock()
        first = persona_ids[0]
        for slug in persona_ids:
            read_adapter_contract(self.adapters_root / slug, slug)
        self.device = select_torch_device()
        self.tokenizer = load_tokenizer()
        base = load_causal_lm(device=self.device, dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(
            base,
            str(self.adapters_root / first),
            adapter_name=first,
            is_trainable=False,
        )
        self.active_persona = first
        self._freeze()

    def _freeze(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def persona_ids(self) -> list[str]:
        return list(self._known)

    def ensure_adapter(self, persona_id: str) -> None:
        with self._lock:
            if persona_id == self.active_persona:
                return
            read_adapter_contract(self.adapters_root / persona_id, persona_id)
            if persona_id not in self._known:
                self._known.append(persona_id)

    def activate(self, persona_id: str) -> None:
        with self._lock:
            self.ensure_adapter(persona_id)
            if persona_id != self.active_persona:
                previous = self.active_persona
                self.model.load_adapter(
                    str(self.adapters_root / persona_id),
                    adapter_name=persona_id,
                    is_trainable=False,
                )
                self.model.set_adapter(persona_id)
                self.model.delete_adapter(previous)
                self.active_persona = persona_id
                self._freeze()
            if self.model.active_adapters != [persona_id]:
                raise EvaiError("Runtime must activate exactly one selected spirit")

    @contextmanager
    def base_only(self) -> Iterator[None]:
        with self._lock:
            try:
                with self.model.disable_adapter():
                    yield
            finally:
                self._freeze()

    def token_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        with self._lock, torch.inference_mode():
            logits = self.model(input_ids=input_ids.to(self.device)).logits[0].float()
            return torch.log_softmax(logits, dim=-1).cpu()

    def generate(
        self,
        user_message: str,
        settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS,
    ) -> str:
        from spirit_dataset.runtime_prompt import build_chat_messages, load_spirit_prompt_source

        with self._lock:
            messages = build_chat_messages(
                load_spirit_prompt_source(self.active_persona),
                user_message,
                1,
            )
            return self.reply(self.active_persona, messages, settings)

    def reply(
        self,
        persona_id: str,
        messages: list[dict[str, str]],
        settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS,
    ) -> str:
        with self._lock:
            self.activate(persona_id)
            return generate_from_messages(self.model, self.tokenizer, messages, settings)
