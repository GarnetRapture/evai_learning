"""Bounded parallel CPU tokenization with a private tokenizer per worker."""

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import local
from typing import Any

from training.data import prepare_records
from training.records import TOKENIZATION_CHUNK_RECORDS, PreparedRecord

_worker = local()


def _initialize(tokenizer: Any) -> None:
    _worker.tokenizer = deepcopy(tokenizer)


def _chunks(
    path: Path, max_length: int, spirit_id: str
) -> Iterator[tuple[list[dict[str, Any]], int, str]]:
    with path.open(encoding="utf-8") as handle:
        records = []
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) == TOKENIZATION_CHUNK_RECORDS:
                yield records, max_length, spirit_id
                records = []
        if records:
            yield records, max_length, spirit_id


def _encode(job: tuple[list[dict[str, Any]], int, str]) -> list[PreparedRecord]:
    records, max_length, spirit_id = job
    return prepare_records(_worker.tokenizer, records, max_length, spirit_id=spirit_id)


class ParallelTokenizer:
    def __init__(self, tokenizer: Any, workers: int) -> None:
        self._workers = workers
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="training-tokenizer",
            initializer=_initialize,
            initargs=(tokenizer,),
        )

    def records(self, path: Path, max_length: int, spirit_id: str) -> Iterator[PreparedRecord]:
        for chunk in self._executor.map(
            _encode,
            _chunks(path, max_length, spirit_id),
            buffersize=self._workers,
        ):
            yield from chunk

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
