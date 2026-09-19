"""Bounded parallel CPU tokenization with a private tokenizer per worker."""

import json
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from training.data import prepare_records
from training.records import TOKENIZATION_CHUNK_RECORDS, PreparedRecord

_worker_tokenizer: Any = None
TOKENIZER_PARALLELISM_VARIABLE = "TOKENIZERS_PARALLELISM"
RAYON_THREADS_VARIABLE = "RAYON_NUM_THREADS"


def _initialize(tokenizer: Any) -> None:
    global _worker_tokenizer
    os.environ[TOKENIZER_PARALLELISM_VARIABLE] = "false"
    os.environ[RAYON_THREADS_VARIABLE] = "1"
    _worker_tokenizer = tokenizer


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


def _stream_chunks(
    records: Iterable[dict[str, Any]], max_length: int, spirit_id: str
) -> Iterator[tuple[list[dict[str, Any]], int, str]]:
    chunk: list[dict[str, Any]] = []
    for record in records:
        chunk.append(record)
        if len(chunk) == TOKENIZATION_CHUNK_RECORDS:
            yield chunk, max_length, spirit_id
            chunk = []
    if chunk:
        yield chunk, max_length, spirit_id


def _encode(job: tuple[list[dict[str, Any]], int, str]) -> list[PreparedRecord]:
    records, max_length, spirit_id = job
    return prepare_records(_worker_tokenizer, records, max_length, spirit_id=spirit_id)


class ParallelTokenizer:
    def __init__(self, tokenizer: Any, workers: int) -> None:
        self._workers = workers
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
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

    def stream(
        self, records: Iterable[dict[str, Any]], max_length: int, context_id: str
    ) -> Iterator[PreparedRecord]:
        for chunk in self._executor.map(
            _encode,
            _stream_chunks(records, max_length, context_id),
            buffersize=self._workers,
        ):
            yield from chunk

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
