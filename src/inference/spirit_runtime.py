"""One jointly trained model; spirit selection changes input, never weights."""

import threading
from typing import Any

import torch

from common.device import select_torch_device
from common.errors import EvaiError
from common.hashing import compute_file_sha256
from common.model_contract import read_training_contract
from common.paths import MODEL_DIR
from inference.generation import (
    DEFAULT_GENERATION_SETTINGS,
    GenerationSettings,
    generate_from_messages,
)
from inference.gguf_loader import load_gguf_model
from inference.model_loader import load_causal_lm, load_tokenizer
from inference.spirit_session import SpiritSession
from spirit_dataset.roster import roster_slugs
from spirit_dataset.runtime_prompt import (
    SpiritPromptSource,
    load_spirit_prompt_source,
    spirit_file_path,
    spirit_selection_header,
)


class SpiritRuntime:
    def __init__(self, persona_ids: list[str], *, use_gguf: bool = False) -> None:
        known = set(roster_slugs())
        if not persona_ids or len(persona_ids) != len(set(persona_ids)):
            raise EvaiError("Runtime requires unique registered spirit IDs")
        if not set(persona_ids) <= known:
            raise EvaiError("Runtime contains an unknown spirit ID")
        contract = read_training_contract(MODEL_DIR)
        if contract is None:
            raise EvaiError("The single model has not been trained yet")
        if not set(persona_ids) <= set(contract["spirits"]):
            raise EvaiError("Selected spirit was not included in the model's training")
        self.contract: dict[str, Any] = contract
        self._known = tuple(persona_ids)
        self._lock = threading.RLock()
        self.device = select_torch_device()
        self.tokenizer = load_tokenizer()
        self.model: Any = (
            load_gguf_model(self.device)
            if use_gguf
            else load_causal_lm(device=self.device, dtype=torch.bfloat16)
        )
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def persona_ids(self) -> list[str]:
        return list(self._known)

    def open_session(
        self, spirit_id: str, language: str = "ko", love_level: int = 1
    ) -> SpiritSession:
        return SpiritSession(self, spirit_id, language, love_level)

    def prompt_source(self, spirit_id: str, language: str = "ko") -> SpiritPromptSource:
        if spirit_id not in self._known:
            raise EvaiError(f"Spirit is not registered in this runtime: {spirit_id}")
        if (
            compute_file_sha256(spirit_file_path(spirit_id))
            != self.contract["profile_sha256"][spirit_id]
        ):
            raise EvaiError(f"Spirit profile changed after model training: {spirit_id}")
        return load_spirit_prompt_source(spirit_id, language)

    def reply(
        self,
        persona_id: str,
        messages: list[dict[str, str]],
        settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS,
    ) -> str:
        if persona_id not in self._known:
            raise EvaiError(f"Spirit is not registered in this runtime: {persona_id}")
        if (
            not messages
            or messages[0]["role"] != "system"
            or not messages[0]["content"].startswith(spirit_selection_header(persona_id))
        ):
            raise EvaiError("Conversation identity differs from the platform-selected spirit")
        with self._lock:
            return generate_from_messages(self.model, self.tokenizer, messages, settings)
