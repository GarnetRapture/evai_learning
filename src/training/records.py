"""Shared training records; no model allocation or I/O."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spirit_dataset.records import TrainingTask

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
class RecordLocation:
    path: Path
    offset: int
    spirit_id: str
    token_count: int
    target_tokens: int


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    validation_loss: float | None


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
    peak_memory_gib: float = 0.0
    quality_approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
