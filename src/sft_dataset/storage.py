import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from common.paths import DATASETS_DIR


class SerializableRecord(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


def persona_dataset_dir(persona_id: str) -> Path:
    return DATASETS_DIR / persona_id


def sft_split_path(persona_id: str, split: str) -> Path:
    return persona_dataset_dir(persona_id) / f"{split}.jsonl"


def write_records_jsonl(path: Path, rows: Iterable[SerializableRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False))
            handle.write("\n")


def read_split_conversations(path: Path) -> list[list[dict[str, str]]]:
    conversations: list[list[dict[str, str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            turns = [*record["prompt"], *record["completion"]]
            conversations.append([{"role": t["role"], "content": t["content"]} for t in turns])
    return conversations
