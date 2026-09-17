"""Training records and result types; no model allocation or I/O."""

from dataclasses import asdict, dataclass, field
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
class PreparedSpirit:
    train: list[TokenizedExample]
    validation: list[TokenizedExample]
    test: list[TokenizedExample]
    over_length: list[str]
    source_analysis_examples: int
    dataset_provenance: dict[str, Any]
    training_tasks: dict[str, int] = field(default_factory=dict)
    supervised_tokens_by_task: dict[str, int] = field(default_factory=dict)


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
class SpiritLoraReport:
    slug: str
    config: dict[str, Any]
    adapter_dir: str
    train_examples: int
    validation_examples: int
    over_length_excluded: list[str]
    trainable_parameters: int
    optimizer_steps: int
    base_validation_loss: float | None
    epochs: list[EpochResult] = field(default_factory=list)
    samples: list[SampleGeneration] = field(default_factory=list)
    seconds: float = 0.0
    peak_memory_gib: float = 0.0
    rank_trials: list[dict[str, Any]] = field(default_factory=list)
    rank_selection_metric: str = ""
    test_loss: float | None = None
    source_analysis_examples: int = 0
    target_contract: str = "self_memory+behavior_judgment+persona_speech"
    training_tasks: dict[str, int] = field(default_factory=dict)
    supervised_tokens_by_task: dict[str, int] = field(default_factory=dict)
    dataset_provenance: dict[str, Any] = field(default_factory=dict)
    quality_approved: bool = False
    phase_seconds: dict[str, float] = field(
        default_factory=lambda: {
            "prepare": 0.0,
            "train": 0.0,
            "validation": 0.0,
            "samples": 0.0,
            "save": 0.0,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
