"""One CPU worker prepares at most one upcoming optimizer step in pinned memory."""

import random
from collections.abc import Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from training.corpus import TrainingCorpus
from training.data import collate
from training.records import EncodedRecord, TrainingStep


def _micro_batches(
    records: list[EncodedRecord], max_records: int, max_tokens: int
) -> Iterator[list[EncodedRecord]]:
    batch: list[EncodedRecord] = []
    width = 0
    for record in records:
        next_width = max(width, record.token_count)
        if batch and (len(batch) == max_records or (len(batch) + 1) * next_width > max_tokens):
            yield batch
            batch = []
            width = 0
        batch.append(record)
        width = max(width, record.token_count)
    if batch:
        yield batch


def _steps(
    corpus: TrainingCorpus,
    split: str,
    batch_size: int,
    micro_batch_size: int,
    micro_batch_tokens: int,
    pad_token_id: int,
    rng: random.Random | None,
    skip_batches: int,
) -> Generator[TrainingStep]:
    for records in corpus.batches(split, batch_size, rng, skip_batches):
        yield TrainingStep(
            tuple(
                collate(batch, corpus.token_ids, pad_token_id)
                for batch in _micro_batches(records, micro_batch_size, micro_batch_tokens)
            ),
            tuple(record.fingerprint for record in records),
        )


@contextmanager
def prefetched_steps(
    corpus: TrainingCorpus,
    split: str,
    batch_size: int,
    micro_batch_size: int,
    micro_batch_tokens: int,
    pad_token_id: int,
    rng: random.Random | None = None,
    skip_batches: int = 0,
) -> Iterator[Iterator[TrainingStep]]:
    producer = _steps(
        corpus,
        split,
        batch_size,
        micro_batch_size,
        micro_batch_tokens,
        pad_token_id,
        rng,
        skip_batches,
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-data")

    def consume() -> Generator[TrainingStep]:
        pending = executor.submit(next, producer, None)
        while (step := pending.result()) is not None:
            pending = executor.submit(next, producer, None)
            yield step

    consumer = consume()
    try:
        yield consumer
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        consumer.close()
        producer.close()
