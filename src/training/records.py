"""Shared training records; no model allocation or I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from spirit_dataset.records import TrainingTask

if TYPE_CHECKING:
    import torch

IGNORE_INDEX = -100
SHUFFLE_POOL_BATCHES = 16


@dataclass(frozen=True)
class TokenizedExample:
    record_id: str
    prompt: list[dict[str, str]]
    reference: str
    input_ids: list[int]
    labels: list[int]
    task: str = TrainingTask.PERSONA_SPEECH.value
    language: str = "ko"


@dataclass(frozen=True)
class EncodedRecord:
    spirit_id: str
    token_offset: int
    token_count: int
    prompt_tokens: int


@dataclass(frozen=True)
class TrainingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_positions: torch.Tensor
    target_ids: torch.Tensor

    @property
    def target_count(self) -> int:
        return self.target_ids.numel()


@dataclass(frozen=True)
class TrainingStep:
    micro_batches: tuple[TrainingBatch, ...]

    @property
    def target_count(self) -> int:
        return sum(batch.target_count for batch in self.micro_batches)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class SampleGeneration:
    record_id: str
    user: str
    reference: str
    generated: str
    language: str = "ko"


@dataclass
class TrainingReport:
    config: dict[str, Any]
    model_dir: str
    input_weights_sha256: str
    trainable_parameters: int
    train_examples: int
    validation_examples: int
    test_examples: int
    dataset_provenance: dict[str, Any]
    over_length_excluded: dict[str, int]
    partition_moves: dict[str, int]
    epochs: list[EpochResult] = field(default_factory=list)
    optimizer_steps: int = 0
    test_loss: float | None = None
    seconds: float = 0.0
    pytorch_peak_allocated_gib: float = 0.0
    token_storage_bytes: int = 0
    preparation_seconds: float = 0.0
    data_wait_seconds: float = 0.0
    train_micro_batches: int = 0
    quality_approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
