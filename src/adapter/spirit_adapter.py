from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from common.device import select_torch_device
from common.errors import EvaiError
from common.paths import ADAPTERS_DIR, MERGED_DIR, MODEL_DIR
from inference.generation import DEFAULT_GENERATION_SETTINGS, GenerationSettings, generate_reply
from inference.model_loader import load_causal_lm, load_tokenizer
from sft_dataset.storage import read_split_conversations, sft_split_path

LORA_LINEAR_SUFFIXES: tuple[str, ...] = (
    "in_proj",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "w1",
    "w2",
    "w3",
)
EMBEDDING_MODULE = "embed_tokens"
TIED_HEAD_MODULE = "lm_head"
FULL_COPY_SUFFIXES: tuple[str, ...] = (
    "conv.conv",
    "operator_norm",
    "ffn_norm",
    "embedding_norm",
    "q_layernorm",
    "k_layernorm",
)


@dataclass(frozen=True)
class LowRankFactor:
    lora_a: torch.Tensor
    lora_b: torch.Tensor


@dataclass(frozen=True)
class RankFidelity:
    rank: int
    mean_kl_to_full: float
    top1_agreement_with_full: float
    retained_effect: float


@dataclass(frozen=True)
class FidelityReport:
    persona_id: str
    token_count: int
    base_mean_kl_to_full: float
    base_top1_agreement_with_full: float
    ranks: list[RankFidelity]

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "token_count": self.token_count,
            "base_mean_kl_to_full": self.base_mean_kl_to_full,
            "base_top1_agreement_with_full": self.base_top1_agreement_with_full,
            "ranks": [
                {
                    "rank": r.rank,
                    "mean_kl_to_full": r.mean_kl_to_full,
                    "top1_agreement_with_full": r.top1_agreement_with_full,
                    "retained_effect": r.retained_effect,
                }
                for r in self.ranks
            ],
        }


def _module_name(parameter_key: str) -> str:
    return parameter_key.rsplit(".", 1)[0]


def _is_lora_linear(parameter_key: str) -> bool:
    leaf = _module_name(parameter_key).rsplit(".", 1)[-1]
    return parameter_key.endswith(".weight") and leaf in LORA_LINEAR_SUFFIXES


def _is_embedding(parameter_key: str) -> bool:
    return _module_name(parameter_key).endswith(EMBEDDING_MODULE)


def _is_full_copy(parameter_key: str) -> bool:
    module = _module_name(parameter_key)
    return any(module.endswith(suffix) for suffix in FULL_COPY_SUFFIXES)


def full_finetune_path(persona_id: str) -> Path:
    return MERGED_DIR / persona_id / "model.safetensors"


def load_base_and_full_state(
    persona_id: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    full_path = full_finetune_path(persona_id)
    if not full_path.exists():
        raise EvaiError(
            f"Full fine-tuned checkpoint not found for persona '{persona_id}': {full_path}"
        )
    base = load_file(str(MODEL_DIR / "model.safetensors"))
    full = load_file(str(full_path))
    unclassified = [
        k for k in base if not (_is_lora_linear(k) or _is_embedding(k) or _is_full_copy(k))
    ]
    if unclassified:
        raise EvaiError(f"Parameters not covered by the adapter layout: {unclassified}")
    if set(base) != set(full):
        raise EvaiError(f"Parameter key mismatch between base and '{persona_id}' checkpoint")
    return base, full


def factorize_delta(delta: torch.Tensor, rank: int) -> LowRankFactor:
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    effective = min(rank, s.numel())
    root = s[:effective].sqrt()
    return LowRankFactor(lora_a=root[:, None] * vh[:effective], lora_b=u[:, :effective] * root)


def compute_low_rank_factors(
    base: dict[str, torch.Tensor],
    full: dict[str, torch.Tensor],
    rank: int,
    device: str,
) -> dict[str, LowRankFactor]:
    factors: dict[str, LowRankFactor] = {}
    for key in sorted(base):
        if not (_is_lora_linear(key) or _is_embedding(key)):
            continue
        delta = full[key].to(device, torch.float32) - base[key].to(device, torch.float32)
        matrix = delta.T if _is_embedding(key) else delta
        factor = factorize_delta(matrix, rank)
        factors[key] = LowRankFactor(lora_a=factor.lora_a.cpu(), lora_b=factor.lora_b.cpu())
    return factors


def reconstruct_state(
    base: dict[str, torch.Tensor],
    full: dict[str, torch.Tensor],
    factors: dict[str, LowRankFactor],
    rank: int,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, base_weight in base.items():
        if key in factors:
            factor = factors[key]
            approx = factor.lora_b[:, :rank] @ factor.lora_a[:rank]
            if _is_embedding(key):
                approx = approx.T
            state[key] = base_weight.float() + approx
        elif _is_full_copy(key):
            state[key] = full[key].float()
        else:
            state[key] = base_weight.float()
    return state


def _read_split_messages(persona_id: str, split: str) -> list[list[dict[str, str]]]:
    path = sft_split_path(persona_id, split)
    return read_split_conversations(path) if path.exists() else []


def _persona_eval_messages(persona_id: str, max_records: int) -> list[list[dict[str, str]]]:
    held_out = [
        *_read_split_messages(persona_id, "validation"),
        *_read_split_messages(persona_id, "test"),
    ]
    source = held_out if held_out else _read_split_messages(persona_id, "train")
    return source[:max_records]


def _token_log_probs(model: Any, batches: list[torch.Tensor]) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for input_ids in batches:
            logits = model(input_ids=input_ids).logits[0].float()
            outputs.append(torch.log_softmax(logits, dim=-1).cpu())
    return outputs


def _compare(
    reference: list[torch.Tensor], candidate: list[torch.Tensor]
) -> tuple[float, float, int]:
    total_kl = 0.0
    agree = 0
    tokens = 0
    for ref, cand in zip(reference, candidate, strict=True):
        p = ref.exp()
        total_kl += (p * (ref - cand)).sum().item()
        agree += (ref.argmax(-1) == cand.argmax(-1)).sum().item()
        tokens += ref.shape[0]
    return total_kl / tokens, agree / tokens, tokens


def measure_rank_fidelity(
    persona_id: str,
    ranks: list[int],
    max_records: int = 16,
) -> FidelityReport:
    device = select_torch_device()
    base, full = load_base_and_full_state(persona_id)
    factors = compute_low_rank_factors(base, full, max(ranks), device)

    tokenizer = load_tokenizer()
    conversations = _persona_eval_messages(persona_id, max_records)
    if not conversations:
        raise EvaiError(f"No SFT records available to measure fidelity for '{persona_id}'")
    batches = [
        tokenizer.apply_chat_template(c, return_tensors="pt", return_dict=True)["input_ids"].to(
            device
        )
        for c in conversations
    ]

    model = load_causal_lm(device=device, dtype=torch.float32)
    model.eval()
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise EvaiError("Expected lm_head to share storage with embed_tokens (tie_word_embeddings)")

    def load_into_model(state: dict[str, torch.Tensor]) -> None:
        result = model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [k for k in result.missing_keys if k != "lm_head.weight"]
        if unexpected or missing:
            raise EvaiError(f"State load mismatch: unexpected={unexpected} missing={missing}")

    load_into_model({k: v.float() for k, v in full.items()})
    full_lp = _token_log_probs(model, batches)

    load_into_model({k: v.float() for k, v in base.items()})
    base_kl, base_agree, token_count = _compare(full_lp, _token_log_probs(model, batches))

    results: list[RankFidelity] = []
    for rank in sorted(ranks):
        load_into_model(reconstruct_state(base, full, factors, rank))
        kl, agree, _ = _compare(full_lp, _token_log_probs(model, batches))
        retained = 1.0 - kl / base_kl if base_kl > 0 else 1.0
        results.append(RankFidelity(rank, kl, agree, retained))

    return FidelityReport(persona_id, token_count, base_kl, base_agree, results)


@dataclass(frozen=True)
class RuntimeFidelity:
    persona_id: str
    token_count: int
    mean_kl_to_full: float
    top1_agreement_with_full: float


def verify_runtime_fidelity(
    persona_ids: list[str], max_records: int = 16
) -> list[RuntimeFidelity]:
    device = select_torch_device()
    tokenizer = load_tokenizer()
    batches_by_persona: dict[str, list[torch.Tensor]] = {}
    full_lp_by_persona: dict[str, list[torch.Tensor]] = {}
    for persona_id in persona_ids:
        conversations = _persona_eval_messages(persona_id, max_records)
        batches = [
            tokenizer.apply_chat_template(c, return_tensors="pt", return_dict=True)["input_ids"]
            for c in conversations
        ]
        batches_by_persona[persona_id] = batches
        full_model = load_causal_lm(MERGED_DIR / persona_id, device, torch.bfloat16)
        full_model.eval()
        full_lp_by_persona[persona_id] = _token_log_probs(
            full_model, [b.to(device) for b in batches]
        )
        del full_model

    runtime = SpiritRuntime(persona_ids)
    results: list[RuntimeFidelity] = []
    for persona_id in persona_ids:
        runtime.activate(persona_id)
        candidate = [runtime.token_log_probs(b) for b in batches_by_persona[persona_id]]
        kl, agree, tokens = _compare(full_lp_by_persona[persona_id], candidate)
        results.append(RuntimeFidelity(persona_id, tokens, kl, agree))
    return results


def load_untied_base_model(dtype: torch.dtype) -> Any:
    model = load_causal_lm(dtype=dtype)
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if not isinstance(embed, torch.nn.Embedding) or not isinstance(head, torch.nn.Linear):
        raise EvaiError("Base model input/output embeddings are not Embedding/Linear modules")
    if head.weight.data_ptr() != embed.weight.data_ptr():
        raise EvaiError("Base model is expected to load with tied input/output embeddings")
    head.weight = torch.nn.Parameter(embed.weight.detach().clone())
    model.config.tie_word_embeddings = False
    if head.weight.data_ptr() == embed.weight.data_ptr():
        raise EvaiError("Failed to untie lm_head from embed_tokens")
    return model


def build_spirit_lora_config(rank: int) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=[*LORA_LINEAR_SUFFIXES, EMBEDDING_MODULE, TIED_HEAD_MODULE],
        modules_to_save=list(FULL_COPY_SUFFIXES),
        task_type="CAUSAL_LM",
    )


def extract_spirit_adapter(persona_id: str, rank: int, output_root: Path | None = None) -> Path:
    from peft import get_peft_model
    from peft.utils.other import ModulesToSaveWrapper

    device = select_torch_device()
    base, full = load_base_and_full_state(persona_id)
    factors = compute_low_rank_factors(base, full, rank, device)

    model = load_untied_base_model(torch.bfloat16)
    peft_model = get_peft_model(model, build_spirit_lora_config(rank), adapter_name=persona_id)

    embedding_key = next(k for k in base if _is_embedding(k))
    embedding_factor = factors[embedding_key]
    tied_head_assigned = False

    assigned: set[str] = set()
    for name, module in peft_model.named_modules():
        relative = name.removeprefix("base_model.model.")
        weight_key = f"{relative}.weight"
        has_lora = hasattr(module, "lora_A") and persona_id in module.lora_A
        if relative == TIED_HEAD_MODULE and has_lora:
            module.lora_A[persona_id].weight.data.copy_(embedding_factor.lora_b.T)
            module.lora_B[persona_id].weight.data.copy_(embedding_factor.lora_a.T)
            tied_head_assigned = True
        elif isinstance(module, ModulesToSaveWrapper):
            saved_weight = module.modules_to_save[persona_id].get_parameter("weight")
            saved_weight.data.copy_(full[weight_key])
            assigned.add(weight_key)
        elif hasattr(module, "lora_embedding_A") and persona_id in module.lora_embedding_A:
            factor = factors[weight_key]
            module.lora_embedding_A[persona_id].data.copy_(factor.lora_a)
            module.lora_embedding_B[persona_id].data.copy_(factor.lora_b)
            assigned.add(weight_key)
        elif has_lora and weight_key in factors:
            factor = factors[weight_key]
            module.lora_A[persona_id].weight.data.copy_(factor.lora_a)
            module.lora_B[persona_id].weight.data.copy_(factor.lora_b)
            assigned.add(weight_key)

    missing = sorted(set(base) - assigned)
    if missing:
        raise EvaiError(
            f"Adapter extraction left parameters unassigned for '{persona_id}': {missing}"
        )
    if not tied_head_assigned:
        raise EvaiError(f"lm_head LoRA was not created for '{persona_id}'")

    inner = peft_model.base_model.model
    head_delta = inner.get_submodule(TIED_HEAD_MODULE).get_delta_weight(persona_id).float()
    embed_delta = inner.get_submodule(_module_name(embedding_key)).get_delta_weight(persona_id)
    tie_error = ((head_delta - embed_delta.float()).norm() / embed_delta.float().norm()).item()
    if not tie_error < 1e-3:
        raise EvaiError(
            f"lm_head delta diverges from embedding delta for '{persona_id}': {tie_error}"
        )

    root = output_root if output_root is not None else ADAPTERS_DIR
    root.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(
        str(root), selected_adapters=[persona_id], save_embedding_layers=False
    )
    return root / persona_id


class SpiritRuntime:
    def __init__(self, persona_ids: list[str], adapters_root: Path | None = None) -> None:
        from peft import PeftModel

        if not persona_ids:
            raise EvaiError("SpiritRuntime requires at least one persona adapter")
        root = adapters_root if adapters_root is not None else ADAPTERS_DIR
        device = select_torch_device()
        self.tokenizer = load_tokenizer()
        base = load_untied_base_model(torch.bfloat16)
        first, *rest = persona_ids
        self.model = PeftModel.from_pretrained(base, str(root / first), adapter_name=first)
        for persona_id in rest:
            self.model.load_adapter(str(root / persona_id), adapter_name=persona_id)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.active_persona: str = first

    @property
    def persona_ids(self) -> list[str]:
        return list(self.model.peft_config.keys())

    def activate(self, persona_id: str) -> None:
        if persona_id not in self.model.peft_config:
            raise EvaiError(f"Spirit adapter '{persona_id}' is not loaded in this runtime")
        self.model.set_adapter(persona_id)
        self.active_persona = persona_id

    def token_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(input_ids=input_ids.to(self.device)).logits[0].float()
        return torch.log_softmax(logits, dim=-1).cpu()

    def generate(
        self, user_message: str, settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS
    ) -> str:
        return generate_reply(self.model, self.tokenizer, user_message, settings)
