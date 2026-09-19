import gzip
import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow.parquet as pq

from common.paths import EXTERNAL_DATA_DIR
from general_corpus.chinese_script import to_taiwan_traditional

SODA_DIR = EXTERNAL_DATA_DIR / "soda"
KOR_EMPATHETIC_DIR = EXTERNAL_DATA_DIR / "KorEmpatheticDialogues"
TINY_CHAT_FILE = EXTERNAL_DATA_DIR / "tiny-multiturn-chat-ko" / "dataset_shuffled.jsonl"
LCCC_DIR = EXTERNAL_DATA_DIR / "lccc_data"
TW_ARENA_FILE = EXTERNAL_DATA_DIR / "tw_chatbot_arena" / "all" / "train-00000-of-00001.parquet"
SOURCE_FILE_SPLITS: tuple[str, ...] = ("train", "valid", "test")
HELD_OUT_BUCKETS = 1000
VALIDATION_BUCKET = 0
TEST_BUCKET = 1
PARQUET_BATCH_ROWS = 65536
SODA_RELATION_FIELDS: dict[str, str] = {
    "xAttr": "interpretation",
    "xReact": "emotion",
    "xIntent": "intention",
    "xWant": "intention",
    "xNeed": "decision",
    "xEffect": "action",
}
EMPATHETIC_EMOTIONS_KO: dict[str, str] = {
    "afraid": "두려움",
    "angry": "분노",
    "annoyed": "짜증",
    "anticipating": "기대",
    "anxious": "불안",
    "apprehensive": "걱정",
    "ashamed": "수치심",
    "caring": "보살핌",
    "confident": "자신감",
    "content": "만족",
    "devastated": "절망",
    "disappointed": "실망",
    "disgusted": "혐오",
    "embarrassed": "당혹",
    "excited": "신남",
    "faithful": "신뢰",
    "furious": "격분",
    "grateful": "감사",
    "guilty": "죄책감",
    "hopeful": "희망",
    "impressed": "감탄",
    "jealous": "질투",
    "joyful": "기쁨",
    "lonely": "외로움",
    "nostalgic": "그리움",
    "prepared": "준비됨",
    "proud": "자부심",
    "sad": "슬픔",
    "sentimental": "감상",
    "surprised": "놀람",
    "terrified": "공포",
    "trusting": "믿음",
}
TINY_CHAT_OPENING = "대화 시작"
MASKED_ENTITY_PATTERN = re.compile(r"@[^@\s]{1,20}@")
ASSISTANT_SELF_REFERENCE_PATTERN = re.compile(r"챗봇|인공지능|AI")
CJK_SPACING_PATTERN = re.compile(r"(?<=[^\x00-\x7F])\s+(?=[^\x00-\x7F])")


@dataclass(frozen=True)
class GeneralConversation:
    conversation_id: str
    source: str
    language: str
    split: str
    first_role: str
    turns: tuple[str, ...]
    judgment: str | None = None


def held_out_split(conversation_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(conversation_id.encode()).digest()[:4], "big")
    bucket %= HELD_OUT_BUCKETS
    if bucket == VALIDATION_BUCKET:
        return "validation"
    if bucket == TEST_BUCKET:
        return "test"
    return "train"


def soda_conversations() -> Iterator[GeneralConversation]:
    for source_split in SOURCE_FILE_SPLITS:
        table = pq.ParquetFile(SODA_DIR / f"{source_split}.parquet")
        offset = 0
        for batch in table.iter_batches(
            batch_size=PARQUET_BATCH_ROWS, columns=["relation", "tail", "narrative", "dialogue"]
        ):
            for index, row in enumerate(batch.to_pylist()):
                identifier = f"soda:{source_split}:{offset + index}"
                split = held_out_split(identifier)
                dialogue = tuple(str(turn).strip() for turn in row["dialogue"] if str(turn).strip())
                if len(dialogue) >= 2:
                    yield GeneralConversation(identifier, "soda", "en", split, "user", dialogue)
                field = SODA_RELATION_FIELDS.get(str(row["relation"]))
                if field and row["narrative"] and row["tail"]:
                    yield GeneralConversation(
                        f"{identifier}:judgment",
                        "soda",
                        "en",
                        split,
                        "user",
                        (str(row["narrative"]).strip(),),
                        json.dumps({field: str(row["tail"]).strip()}, ensure_ascii=False),
                    )
            offset += batch.num_rows


def kor_empathetic_conversations() -> Iterator[GeneralConversation]:
    for source_split in SOURCE_FILE_SPLITS:
        rows = json.loads((KOR_EMPATHETIC_DIR / f"{source_split}.json").read_text(encoding="utf-8"))
        for row in rows:
            identifier = f"kor_empathetic:{source_split}:{row['dialogue_id']}"
            split = held_out_split(identifier)
            utterances = sorted(row["dialogue"], key=lambda item: item["utter_idx"])
            turns = tuple(str(item["utter"]).strip() for item in utterances if item["utter"])
            if len(turns) >= 2:
                yield GeneralConversation(identifier, "kor_empathetic", "ko", split, "user", turns)
            emotion = EMPATHETIC_EMOTIONS_KO.get(str(row["emotion"]))
            if emotion and row["situation"]:
                yield GeneralConversation(
                    f"{identifier}:judgment",
                    "kor_empathetic",
                    "ko",
                    split,
                    "user",
                    (str(row["situation"]).strip(),),
                    json.dumps({"emotion": emotion}, ensure_ascii=False),
                )


def tiny_chat_conversations() -> Iterator[GeneralConversation]:
    with TINY_CHAT_FILE.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            texts = [str(row["context"]).strip(), str(row["prompt"]).strip(), str(row["answer"])]
            if any(MASKED_ENTITY_PATTERN.search(text) for text in texts):
                continue
            if ASSISTANT_SELF_REFERENCE_PATTERN.search(texts[2]):
                continue
            identifier = f"tiny_chat:{index}"
            if texts[0] and texts[0] != TINY_CHAT_OPENING:
                turns = (texts[0], texts[1], texts[2].strip())
                first_role = "assistant"
            else:
                turns = (texts[1], texts[2].strip())
                first_role = "user"
            if all(turns):
                yield GeneralConversation(
                    identifier, "tiny_chat", "ko", held_out_split(identifier), first_role, turns
                )


def lccc_conversations() -> Iterator[GeneralConversation]:
    for source_split in SOURCE_FILE_SPLITS:
        path = LCCC_DIR / f"lccc_base_{source_split}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                turns = tuple(
                    to_taiwan_traditional(CJK_SPACING_PATTERN.sub("", str(turn)).strip())
                    for turn in json.loads(line)
                )
                identifier = f"lccc:{source_split}:{index}"
                if len(turns) >= 2 and all(turns):
                    yield GeneralConversation(
                        identifier, "lccc", "zh_tw", held_out_split(identifier), "user", turns
                    )


def tw_arena_conversations() -> Iterator[GeneralConversation]:
    rows = pq.read_table(TW_ARENA_FILE, columns=["question_id", "chosen"]).to_pylist()
    for row in rows:
        chosen = row["chosen"] or []
        if not chosen or chosen[0]["role"] != "user":
            continue
        turns = tuple(to_taiwan_traditional(str(turn["content"]).strip()) for turn in chosen)
        identifier = f"tw_arena:{row['question_id']}"
        if len(turns) >= 2 and all(turns):
            yield GeneralConversation(
                identifier, "tw_arena", "zh_tw", held_out_split(identifier), "user", turns
            )


SOURCE_READERS = (
    soda_conversations,
    kor_empathetic_conversations,
    tiny_chat_conversations,
    lccc_conversations,
    tw_arena_conversations,
)


def general_conversations() -> Iterator[GeneralConversation]:
    for reader in SOURCE_READERS:
        yield from reader()
