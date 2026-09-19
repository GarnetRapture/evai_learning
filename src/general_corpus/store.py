from collections import Counter
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from common.errors import EvaiError
from common.paths import GENERAL_CORPUS_FILE
from general_corpus.sources import GeneralConversation, general_conversations

WRITE_BATCH_ROWS = 50000
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 9
SCHEMA = pa.schema(
    [
        ("conversation_id", pa.string()),
        ("source", pa.dictionary(pa.int8(), pa.string())),
        ("language", pa.dictionary(pa.int8(), pa.string())),
        ("split", pa.dictionary(pa.int8(), pa.string())),
        ("first_role", pa.dictionary(pa.int8(), pa.string())),
        ("turns", pa.list_(pa.string())),
        ("judgment", pa.string()),
    ]
)


def _batch(conversations: list[GeneralConversation]) -> pa.RecordBatch:
    columns = {name: [getattr(item, name) for item in conversations] for name in SCHEMA.names}
    columns["turns"] = [list(item.turns) for item in conversations]
    return pa.RecordBatch.from_pydict(columns, schema=SCHEMA)


def write_general_corpus() -> Counter[str]:
    GENERAL_CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    pending: list[GeneralConversation] = []
    with pq.ParquetWriter(
        GENERAL_CORPUS_FILE,
        SCHEMA,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
    ) as writer:
        for conversation in general_conversations():
            pending.append(conversation)
            counts[f"{conversation.source}:{conversation.language}:{conversation.split}"] += 1
            if len(pending) == WRITE_BATCH_ROWS:
                writer.write_batch(_batch(pending))
                pending = []
        if pending:
            writer.write_batch(_batch(pending))
    return counts


def read_general_corpus() -> Iterator[GeneralConversation]:
    if not GENERAL_CORPUS_FILE.is_file():
        raise EvaiError(f"General corpus not built: {GENERAL_CORPUS_FILE}")
    for batch in pq.ParquetFile(GENERAL_CORPUS_FILE).iter_batches(batch_size=WRITE_BATCH_ROWS):
        for row in batch.to_pylist():
            yield GeneralConversation(**{**row, "turns": tuple(row["turns"])})
