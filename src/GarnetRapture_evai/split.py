"""Deterministic, leakage-safe dataset splitting.

Splitting groups records by their normalized completion text before
shuffling, so that an identical canonical utterance (e.g. a greeting or a
speech_pattern line repeated verbatim) always lands entirely inside one of
train/validation/test. Without this grouping, the same completion text could
appear in both train and test, letting the model "recognize" a held-out
example it was actually trained on rather than generalizing to it.
"""

import random
from dataclasses import dataclass
from typing import Protocol


class HasCompletionText(Protocol):
    """Structural type for any record exposing normalized completion text for grouping."""

    completion: list


@dataclass(frozen=True)
class SplitConfig:
    """Dataset splitting ratios and seed configuration."""

    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    seed: int = 42

    def validate(self) -> bool:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        return abs(total - 1.0) < 1e-6


@dataclass(frozen=True)
class DatasetSplit[T]:
    """Partitioned dataset containing train, validation, and test collections."""

    train: list[T]
    validation: list[T]
    test: list[T]

    @property
    def total_count(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


def _completion_group_key(record: HasCompletionText) -> str:
    """Build a grouping key from a record's completion turns so identical
    canonical utterances are never split across train/validation/test."""
    return "␟".join(turn.content for turn in record.completion)


def leakage_safe_split[T](
    records: list[T], config: SplitConfig
) -> DatasetSplit[T]:
    """Deterministically partition records into train/validation/test by completion-text group.

    Records sharing identical completion text are grouped and kept together
    in the same split, then groups are seed-shuffled and assigned to splits
    by cumulative record count against the configured ratios. Raises
    ValueError if `config.validate()` fails.
    """
    if not config.validate():
        raise ValueError(
            f"Split ratios must sum to 1.0: got "
            f"{config.train_ratio} + {config.val_ratio} + {config.test_ratio}"
        )

    groups: dict[str, list[T]] = {}
    for record in records:
        key = _completion_group_key(record)  # type: ignore[arg-type]
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
