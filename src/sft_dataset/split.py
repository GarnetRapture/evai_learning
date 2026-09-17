import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class HasContent(Protocol):
    @property
    def content(self) -> str: ...


class HasCompletionText(Protocol):
    @property
    def completion(self) -> Sequence[HasContent]: ...


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    seed: int = 42

    def validate(self) -> bool:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        return (
            all(0.0 <= ratio <= 1.0 for ratio in (
                self.train_ratio, self.val_ratio, self.test_ratio,
            ))
            and abs(total - 1.0) < 1e-6
        )


@dataclass(frozen=True)
class DatasetSplit[T]:
    train: list[T]
    validation: list[T]
    test: list[T]

    @property
    def total_count(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


def _completion_group_key(record: HasCompletionText) -> str:
    return "␟".join(turn.content for turn in record.completion)


def leakage_safe_split[T: HasCompletionText](
    records: list[T], config: SplitConfig
) -> DatasetSplit[T]:
    if not config.validate():
        raise ValueError(
            f"Split ratios must each be in 0..1 and sum to 1.0: got "
            f"{config.train_ratio} + {config.val_ratio} + {config.test_ratio}"
        )

    groups: dict[str, list[T]] = {}
    for record in records:
        key = _completion_group_key(record)
        groups.setdefault(key, []).append(record)

    ordered_keys = sorted(groups.keys())
    rng = random.Random(config.seed)
    rng.shuffle(ordered_keys)

    total = len(records)
    train_cutoff = round(total * config.train_ratio)
    val_cutoff = train_cutoff + round(total * config.val_ratio)

    train: list[T] = []
    validation: list[T] = []
    test: list[T] = []
    running_count = 0

    for key in ordered_keys:
        group = groups[key]
        destination = train
        if running_count >= val_cutoff:
            destination = test
        elif running_count >= train_cutoff:
            destination = validation
        destination.extend(group)
        running_count += len(group)

    return DatasetSplit(train=train, validation=validation, test=test)
