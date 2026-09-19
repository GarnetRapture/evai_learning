import json
import re
from collections import Counter
from enum import StrEnum

from common.paths import KOREAN_ADULT_ROLEPLAY_FILE
from external_dialogue.source import (
    SegmentKind,
    SourceConversation,
    SourceDataset,
    SourceNames,
    SourceTurn,
    TurnSegment,
)

KOREAN_ADULT_ROLEPLAY_SOURCE = SourceDataset(
    name="kuuhaku06/RP",
    license="unspecified",
    id_prefix="adult_rp",
    subset="rp_ko",
    path=KOREAN_ADULT_ROLEPLAY_FILE,
)
SPEAKER_PATTERN = re.compile(r"^\s*([^「(（\n]{1,24}?)\s*「")
TURN_PART_PATTERN = re.compile(r"「([^」]*)(?:」|$)|[(（]([^)）]*)[)）]|([^「(（]+)")
QUOTE_PART_PATTERN = re.compile(r"[(（]([^)）]*)[)）]|([^(（]+)")
HONORIFIC_SUFFIXES: tuple[str, ...] = ("선생님", "선배", "씨", "님", "군", "양", "짱", "쨩", "상")
MIN_NAME_PART_LENGTH = 2
SCENE_HEADER_MARKER = "장면"
SYSTEM_ROLE = "system"
SPEAKER_ROLES: tuple[str, ...] = ("assistant", "user")


class CharacterGender(StrEnum):
    FEMALE = "female"
    MALE = "male"
    UNKNOWN = "unknown"


CHARACTER_HEADER_MARKERS: tuple[str, ...] = ("캐릭터", "인물")
USER_HEADER_MARKER = "유저"
ASSISTANT_HEADER_MARKER = "당신"
ROLE_LINE_LOOKAHEAD = 2
GENDER_TAG_PATTERN = re.compile(
    r"성별\s*[:：]\s*(?:(여성|여자)|(남성|남자))|[(（,，]\s*(?:(여성|여자)|(남성|남자))"
)
GENDER_WORD_PATTERN = re.compile(
    r"(?:(여성|여자|아가씨|여왕|공주|왕녀|메이드|유녀|기녀|창부|마녀|아내|여신|서큐버스"
    r"|여교사|여의사|여기사|누나|언니|미녀|숙녀)"
    r"|(남성|남자|청년|왕자|아저씨|남편|신사|무사|미남|오빠))(?!을|를|에게|의|에)"
)


def _role_line(lines: list[str], index: int) -> str:
    return next(
        (line for line in lines[index + 1 : index + 1 + ROLE_LINE_LOOKAHEAD] if line.strip()), ""
    )


def _line_gender(line: str) -> CharacterGender:
    tag = GENDER_TAG_PATTERN.search(line)
    if tag is not None:
        return CharacterGender.FEMALE if tag.group(1) or tag.group(3) else CharacterGender.MALE
    word = GENDER_WORD_PATTERN.search(line)
    if word is None:
        return CharacterGender.UNKNOWN
    return CharacterGender.FEMALE if word.group(1) else CharacterGender.MALE


def setting_genders(setting: str) -> dict[str, CharacterGender]:
    lines = setting.split("\n")
    roles: dict[str, str] = {}
    for index, line in enumerate(lines):
        if not any(marker in line for marker in CHARACTER_HEADER_MARKERS):
            continue
        if USER_HEADER_MARKER in line:
            roles.setdefault("user", _role_line(lines, index))
        elif ASSISTANT_HEADER_MARKER in line:
            roles.setdefault("assistant", _role_line(lines, index))
    return {role: _line_gender(roles.get(role, "")) for role in SPEAKER_ROLES}


def parse_adult_roleplay_turn(role: str, content: str) -> tuple[str | None, SourceTurn]:
    speaker_match = SPEAKER_PATTERN.match(content)
    speaker = speaker_match.group(1).strip() if speaker_match else None
    body = content[speaker_match.end() - 1 :] if speaker_match else content
    narrated = "「" in body
    segments: list[TurnSegment] = []
    for quote, action, plain in TURN_PART_PATTERN.findall(body):
        if quote:
            segments.extend(_quote_segments(quote))
        elif action.strip():
            segments.append(TurnSegment(SegmentKind.ACTION, action.strip()))
        elif plain.strip():
            kind = SegmentKind.ACTION if narrated else SegmentKind.SPEECH
            segments.append(TurnSegment(kind, plain.strip()))
    return speaker, SourceTurn(role, tuple(segments))


def _quote_segments(quote: str) -> list[TurnSegment]:
    segments: list[TurnSegment] = []
    for action, speech in QUOTE_PART_PATTERN.findall(quote):
        if action.strip():
            segments.append(TurnSegment(SegmentKind.ACTION, action.strip()))
        elif speech.strip():
            segments.append(TurnSegment(SegmentKind.SPEECH, speech.strip()))
    return segments


def name_forms(name: str | None) -> tuple[str, ...]:
    if not name:
        return ()
    bases = {name, *(part for part in name.split() if len(part) >= MIN_NAME_PART_LENGTH)}
    forms = {base + suffix for base in bases for suffix in HONORIFIC_SUFFIXES} | bases
    return tuple(sorted(forms, key=lambda form: (-len(form), form)))


def scene_description(setting: str) -> str:
    lines = setting.split("\n")
    for index, line in enumerate(lines[:-1]):
        if SCENE_HEADER_MARKER in line:
            return lines[index + 1].strip()
    return ""


def load_korean_adult_roleplay_conversations() -> list[SourceConversation]:
    conversations: list[SourceConversation] = []
    with KOREAN_ADULT_ROLEPLAY_SOURCE.require_path().open(encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            setting = "\n".join(
                message["content"] for message in messages if message["role"] == SYSTEM_ROLE
            )
            parsed = [
                parse_adult_roleplay_turn(message["role"], message["content"])
                for message in messages
                if message["role"] in SPEAKER_ROLES
            ]
            speakers = {
                role: Counter(speaker for speaker, turn in parsed if turn.role == role and speaker)
                for role in SPEAKER_ROLES
            }
            conversations.append(
                SourceConversation(
                    dataset=KOREAN_ADULT_ROLEPLAY_SOURCE,
                    row=row,
                    topic=scene_description(setting),
                    setting=setting,
                    names=SourceNames(
                        character=name_forms(_most_common(speakers["assistant"])),
                        user=name_forms(_most_common(speakers["user"])),
                    ),
                    turns=tuple(turn for _, turn in parsed),
                )
            )
    return conversations


def _most_common(counter: Counter[str]) -> str | None:
    return counter.most_common(1)[0][0] if counter else None
