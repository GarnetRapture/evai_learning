import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from common.messages import SFTRecordTurn
from external_dialogue.hangul import has_final, vocative_particle
from external_dialogue.scenario_policy import scenario_classification
from external_dialogue.source import SegmentKind, SourceConversation, SourceNames, SourceTurn
from external_dialogue.speech_style import (
    SpeechLevel,
    SpeechProfile,
    classify_level,
    sentence_endings,
)
from external_dialogue.style_transfer import clean_roleplay_text, convert_speech
from sft_dataset.dialogue import TurnClassification
from spirit_dataset.language import render
from spirit_dataset.memory import MIN_LOVE_LEVEL, compose_identity_prompt
from spirit_dataset.profile import SpiritProfile
from spirit_dataset.records import (
    CONVERSATION_RECORD_PREFIX,
    ExclusionReason,
    JudgmentTrace,
    MemoryEvidence,
    SourceClass,
    SourceKind,
    SourceReference,
    SpiritExclusionRecord,
    SpiritTrainingRecord,
)
from spirit_dataset.script import EARLIER_EXCHANGE_LIMIT
from spirit_dataset.situations import EVERTALK_OPENING_SITUATION

NAME_PATTERN_CACHE_SIZE = 16384
GRAMMATICAL_FOLLOWERS = "인이입으로에께처보만까부들야여나과와랑도을를은는의가다든일예라안"
CONCEPT_FOLLOWERS = GRAMMATICAL_FOLLOWERS + "해합하시전후용중안식풍류"
PLURAL_PRONOUN_ENDINGS = "들희"
COPULA_DROPPING_FOLLOWERS: tuple[str, ...] = ("다", "든", "라")
COPULA_STEM = "이"
ACTION_FORMAT = "({})"
SELF_REFERENCE = "나"
PREVIEW_LENGTH = 80
REAL_WORLD_CONTEXT = "real_world_context"
EMPTY_CONTEXT = "empty_context"
SPEECH_STYLE_CONVERSION = "speech_style_conversion"
PARTNER_ADDRESSES: tuple[str, ...] = ("우리 자기", "자기야", "오빠")
SPIRIT_ONLY_MARKERS: tuple[str, ...] = (
    "AI",
    "인공지능",
    "챗봇",
    "코딩",
    "남자친구",
    "여자친구",
    "여러분",
    "엄마",
    "아빠",
    "학교",
    "회사",
)
REAL_WORLD_MARKERS: tuple[str, ...] = (
    "인스타",
    "넷플릭스",
    "디즈니",
    "유튜브",
    "틱톡",
    "페이스북",
    "트위터",
    "카톡",
    "카카오",
    "스타벅스",
    "아이폰",
    "갤럭시",
    "서울",
    "부산",
    "제주",
    "한국",
    "미국",
    "일본",
    "중국",
    "톰 크루즈",
    "겨울왕국",
    "아시안게임",
    "올림픽",
    "Let It Go",
)
PARTICLE_PATTERN = r"(이랑|은|는|이|가|을|를|랑|와|과|의|도|한테|에게|께)?"
PARTICLE_CLASSES: dict[str, tuple[str, str]] = {
    "은": ("은", "는"),
    "는": ("은", "는"),
    "이": ("이", "가"),
    "가": ("이", "가"),
    "을": ("을", "를"),
    "를": ("을", "를"),
    "이랑": ("이랑", "랑"),
    "랑": ("이랑", "랑"),
    "와": ("과", "와"),
    "과": ("과", "와"),
}
HISTORY_TURN_LIMIT = (EARLIER_EXCHANGE_LIMIT + 1) * 2
EXTERNAL_ID_PREFIX = f"{CONVERSATION_RECORD_PREFIX}external:"
EXTERNAL_TABLE_PREFIX = "korean_roleplay:"


def called_name(spirit_name: str) -> str:
    return spirit_name.split("(", 1)[0].strip()


def compile_names(sources: Iterable[str]) -> re.Pattern[str] | None:
    ordered = sorted(set(sources), key=lambda source: (-len(source), source))
    if not ordered:
        return None
    return re.compile(f"(?:{'|'.join(map(re.escape, ordered))}){PARTICLE_PATTERN}")


def compile_terms(
    sources: Iterable[str], followers: str = GRAMMATICAL_FOLLOWERS
) -> re.Pattern[str] | None:
    ordered = sorted(set(sources), key=lambda source: (-len(source), source))
    if not ordered:
        return None
    return re.compile(
        f"(?:(?<![가-힣])|(?<=[{PLURAL_PRONOUN_ENDINGS}]))"
        f"(?:{'|'.join(map(re.escape, ordered))}){PARTICLE_PATTERN}"
        f"(?=[^가-힣]|$|[{followers}])"
    )


@lru_cache(maxsize=NAME_PATTERN_CACHE_SIZE)
def cached_names(sources: tuple[str, ...]) -> re.Pattern[str] | None:
    return compile_names(sources)


def substitute_names(text: str, pattern: re.Pattern[str] | None, target: str) -> str:
    if pattern is None:
        return text

    def substitute(match: re.Match[str]) -> str:
        particle = match.group(1) or ""
        if particle in PARTICLE_CLASSES:
            after_final, after_vowel = PARTICLE_CLASSES[particle]
            particle = after_final if has_final(target) else after_vowel
        elif (
            not particle
            and has_final(target)
            and match.string.startswith(COPULA_DROPPING_FOLLOWERS, match.end())
        ):
            particle = COPULA_STEM
        return target + particle

    return pattern.sub(substitute, text)


def replace_with_particle(text: str, source: str, target: str) -> str:
    return substitute_names(text, cached_names((source,)), target)


def mentions_real_world(text: str) -> bool:
    return any(marker in text for marker in REAL_WORLD_MARKERS)


@dataclass(frozen=True)
class ConvertedTurn:
    turn: SFTRecordTurn
    trainable: bool
    source_index: int


def _replace_names(text: str, sources: tuple[str, ...], target: str) -> str:
    return substitute_names(text, cached_names(sources), target)


def _render(kind: SegmentKind, text: str) -> str:
    return text if kind is SegmentKind.SPEECH else ACTION_FORMAT.format(text)


def convert_user_turn(turn: SourceTurn, spirit_name: str, names: SourceNames) -> str | None:
    name = called_name(spirit_name)
    parts: list[str] = []
    for segment in turn.segments:
        if mentions_real_world(segment.text):
            return None
        text = segment.text
        for source in names.character:
            text = text.replace(source + vocative_particle(source), name + vocative_particle(name))
        text = _replace_names(text, names.character, name)
        text = clean_roleplay_text(_replace_names(text, names.user, SELF_REFERENCE))
        if text:
            parts.append(_render(segment.kind, text))
    return " ".join(parts) or None


def speech_fits_level(text: str, level: SpeechLevel) -> bool:
    return level is not SpeechLevel.CASUAL or all(
        classify_level(sentence) is SpeechLevel.CASUAL for sentence in sentence_endings(text)
    )


def convert_spirit_turn(
    turn: SourceTurn, profile: SpeechProfile, spirit_name: str, names: SourceNames
) -> tuple[str, bool]:
    parts: list[str] = []
    trainable = True
    for segment in turn.segments:
        if mentions_real_world(segment.text) or any(
            marker in segment.text for marker in SPIRIT_ONLY_MARKERS
        ):
            trainable = False
        text = _replace_names(segment.text, names.user, profile.address)
        text = _replace_names(text, names.character, called_name(spirit_name))
        text = _replace_names(text, PARTNER_ADDRESSES, profile.address)
        if segment.kind is SegmentKind.SPEECH:
            if not speech_fits_level(text, profile.level):
                trainable = False
            converted = convert_speech(text, profile) if trainable else None
            if converted is None:
                trainable = False
                converted = clean_roleplay_text(text)
        else:
            converted = clean_roleplay_text(text)
        if converted:
            parts.append(_render(segment.kind, converted))
    content = " ".join(parts)
    return content, trainable and bool(content)


def _external_reference(table: str, keys: tuple[int, ...]) -> SourceReference:
    return SourceReference(
        SourceKind.EXTERNAL_DIALOGUE, table, keys, source_class=SourceClass.DERIVED_SPEECH
    )


def convert_conversation(
    conversation: SourceConversation, profile: SpeechProfile, spirit_name: str, table: str
) -> tuple[list[ConvertedTurn], list[SpiritExclusionRecord]]:
    turns: list[ConvertedTurn] = []
    exclusions: list[SpiritExclusionRecord] = []
    for index, turn in enumerate(conversation.turns):
        if turn.role == "user":
            text = convert_user_turn(turn, spirit_name, conversation.names)
            if text is None:
                remaining = sum(1 for later in conversation.turns[index:] if later.role != "user")
                detail = (
                    REAL_WORLD_CONTEXT
                    if any(mentions_real_world(segment.text) for segment in turn.segments)
                    else EMPTY_CONTEXT
                )
                exclusions.append(
                    SpiritExclusionRecord(
                        reason=ExclusionReason.EXTERNAL_CONTEXT_BREAK,
                        detail=f"{detail}:{remaining}",
                        source=_external_reference(table, (conversation.row, index)),
                        text_preview=turn.raw_text[:PREVIEW_LENGTH],
                    )
                )
                break
            turns.append(ConvertedTurn(SFTRecordTurn("user", text), False, index))
            continue
        text, trainable = convert_spirit_turn(turn, profile, spirit_name, conversation.names)
        if not text:
            continue
        if not trainable:
            exclusions.append(
                SpiritExclusionRecord(
                    reason=ExclusionReason.UNCONVERTED_SPEECH,
                    detail=SPEECH_STYLE_CONVERSION,
                    source=_external_reference(table, (conversation.row, index)),
                    text_preview=text[:PREVIEW_LENGTH],
                )
            )
        turns.append(ConvertedTurn(SFTRecordTurn("assistant", text), trainable, index))
    while turns and turns[-1].turn.role != "assistant":
        turns.pop()
    return turns, exclusions


def external_dialogue_records(
    profile: SpiritProfile,
    speech: SpeechProfile,
    conversations: list[SourceConversation],
) -> tuple[list[SpiritTrainingRecord], list[SpiritExclusionRecord]]:
    system = SFTRecordTurn(
        "system",
        compose_identity_prompt(profile.identity_memory, MIN_LOVE_LEVEL, language=profile.language),
    )
    opening = ConvertedTurn(
        SFTRecordTurn("user", render(EVERTALK_OPENING_SITUATION, profile.language)), False, -1
    )
    records: list[SpiritTrainingRecord] = []
    exclusions: list[SpiritExclusionRecord] = []
    for conversation in conversations:
        table = f"{EXTERNAL_TABLE_PREFIX}{conversation.subset}"
        classification = scenario_classification(conversation.setting)
        if classification is not TurnClassification.ACCEPTED:
            exclusions.append(
                SpiritExclusionRecord(
                    reason=ExclusionReason.CONTENT_CLASSIFICATION,
                    detail=classification.value,
                    source=_external_reference(table, (conversation.row,)),
                    text_preview=conversation.topic[:PREVIEW_LENGTH],
                )
            )
            continue
        converted, conversation_exclusions = convert_conversation(
            conversation, speech, profile.identity.name, table
        )
        exclusions.extend(conversation_exclusions)
        if converted and converted[0].turn.role == "assistant":
            converted.insert(0, opening)
        turns = [item.turn for item in converted]
        for index, item in enumerate(converted):
            if item.turn.role != "assistant" or not item.trainable:
                continue
            turn = item.turn
            history = turns[max(0, index - HISTORY_TURN_LIMIT) : index]
            if history and history[0].role != "user":
                history = history[1:]
            records.append(
                SpiritTrainingRecord(
                    id=f"{EXTERNAL_ID_PREFIX}{conversation.subset}:{conversation.row}:{index}",
                    source=_external_reference(table, (conversation.row, index)),
                    love_level=MIN_LOVE_LEVEL,
                    judgment=JudgmentTrace(history[-1].content),
                    prompt=[system, *history],
                    completion=[turn],
                    source_class=SourceClass.DERIVED_SPEECH,
                    evidence=(
                        MemoryEvidence(
                            SourceClass.GENERAL_KNOWLEDGE,
                            f"{conversation.provenance()}",
                        ),
                    ),
                    event_keys=(f"{table}:{conversation.row}",),
                    origin_id=f"{EXTERNAL_ID_PREFIX}{conversation.subset}:{conversation.row}",
                    language="ko",
                )
            )
    return records, exclusions
