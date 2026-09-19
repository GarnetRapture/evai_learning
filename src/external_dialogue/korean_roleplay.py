import re

import pyarrow.parquet as pq

from common.paths import KOREAN_ROLEPLAY_DIR
from external_dialogue.source import (
    SegmentKind,
    SourceConversation,
    SourceDataset,
    SourceNames,
    SourceTurn,
    TurnSegment,
)

EXA_SUBSET = "exa-data"
PARQUET_FILE_NAME = "train-00000-of-00001.parquet"
KOREAN_ROLEPLAY_SOURCE = SourceDataset(
    name="huggingface-KREW/korean-role-playing",
    license="Apache-2.0",
    id_prefix="krew",
    subset=EXA_SUBSET,
    path=KOREAN_ROLEPLAY_DIR / EXA_SUBSET / PARQUET_FILE_NAME,
)
KOREAN_ROLEPLAY_NAMES = SourceNames(character=("엑사",), user=("{유저}",))
STAGE_DIRECTION_PATTERN = re.compile(r"\*([^*]*)\*")
SPEECH_QUOTES = '"“”'
SPIRIT_DIALOGUE_TOPICS: tuple[str, ...] = (
    "일상 대화를",
    "하루 일과를 서로 나누는 대화를",
    "오늘의 계획이나 루틴을 정리하는 대화",
    "정서적으로 위로가 필요한 상황을",
    "기분을 묻고 위로하는 대화를",
    "보고 싶다고 말하는 대화를",
    "질투 섞인 장난을 주고받는 대화를",
    "사랑을 밀당하는 대화를",
    "잘 자/잘 일어나 인사하는 대화를",
    "데이트 메뉴를 고르는 대화를",
    "농담을",
    "철학적인 질문을 던지는 대화",
    "공부나 자기계발을 도와주는 대화",
    "학술적인 대화를",
    "챗봇에게 물어볼만한 질의응답을",
    "미래 진로에 대한 고민을",
)


def parse_roleplay_turn(role: str, content: str) -> SourceTurn:
    segments: list[TurnSegment] = []
    position = 0
    for match in STAGE_DIRECTION_PATTERN.finditer(content):
        _append_speech(segments, content[position : match.start()])
        action = match.group(1).strip()
        if action:
            segments.append(TurnSegment(SegmentKind.ACTION, action))
        position = match.end()
    _append_speech(segments, content[position:])
    return SourceTurn(role, tuple(segments))


def _append_speech(segments: list[TurnSegment], text: str) -> None:
    speech = text.strip().strip(SPEECH_QUOTES).strip()
    if speech:
        segments.append(TurnSegment(SegmentKind.SPEECH, speech))


def load_korean_roleplay_conversations() -> list[SourceConversation]:
    rows = pq.read_table(KOREAN_ROLEPLAY_SOURCE.require_path()).to_pylist()
    return [
        SourceConversation(
            dataset=KOREAN_ROLEPLAY_SOURCE,
            row=index,
            topic=topic,
            setting="",
            names=KOREAN_ROLEPLAY_NAMES,
            turns=tuple(parse_roleplay_turn(turn["role"], turn["content"]) for turn in row["text"]),
        )
        for index, row in enumerate(rows)
        if (topic := str(row.get("topic") or "")) in SPIRIT_DIALOGUE_TOPICS
    ]
