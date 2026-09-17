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
            all(
                0.0 <= ratio <= 1.0
                for ratio in (
                    self.train_ratio,
                    self.val_ratio,
                    self.test_ratio,
                )
            )
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


def connected_groups[T: HasCompletionText](records: list[T]) -> list[list[T]]:
    """Transitive closure of events, paired tasks and identical target text."""
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[str, int] = {}
    for index, record in enumerate(records):
        keys: list[str] = list(getattr(record, "event_keys", ()))
        origin = getattr(record, "origin_id", None)
        if origin:
            keys.append(f"origin:{origin}")
        identifier = getattr(record, "id", None)
        if identifier:
            keys.append(f"origin:{identifier}")
        answer = _completion_group_key(record)
        if answer:
            keys.append(f"answer:{answer}")
        for key in keys:
            previous = owners.setdefault(key, index)
            parents[find(index)] = find(previous)
    groups: dict[int, list[T]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    return list(groups.values())


def leakage_safe_split[T: HasCompletionText](
    records: list[T], config: SplitConfig
) -> DatasetSplit[T]:
    if not config.validate():
        raise ValueError(
            f"Split ratios must each be in 0..1 and sum to 1.0: got "
            f"{config.train_ratio} + {config.val_ratio} + {config.test_ratio}"
        )

    groups = connected_groups(records)
    groups.sort(key=lambda group: min(_completion_group_key(record) for record in group))
    rng = random.Random(config.seed)
    rng.shuffle(groups)

    total = len(records)
    partitions: list[list[T]] = [[], [], []]
    ratios = (config.train_ratio, config.val_ratio, config.test_ratio)
    targets = [total * ratio for ratio in ratios]

    # Required sparse supervision is assigned by whole connected component.
    reviewed = next(
        (
            group
            for group in groups
            if any(
                getattr(getattr(record, "judgment", None), "is_complete", False) for record in group
            )
        ),
        None,
    )
    if reviewed is not None and config.train_ratio > 0:
        groups.remove(reviewed)
        groups.insert(0, reviewed)
    for index, group in enumerate(groups):
        empty = [i for i, part in enumerate(partitions) if not part and ratios[i] > 0]
        if index == 0 and reviewed is not None and config.train_ratio > 0:
            destination = 0
        elif len(groups) - index <= len(empty):
            destination = empty[0]
        else:
            destination = max(
                (i for i, ratio in enumerate(ratios) if ratio > 0),
                key=lambda i: targets[i] - len(partitions[i]),
            )
        partitions[destination].extend(group)
    return DatasetSplit(train=partitions[0], validation=partitions[1], test=partitions[2])
