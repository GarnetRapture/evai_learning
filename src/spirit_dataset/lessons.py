"""Authored knowledge expression grounded in each spirit's own source speech."""

import json
from dataclasses import replace
from functools import lru_cache
from typing import Any

from common.errors import EvaiError
from common.messages import SFTRecordTurn
from common.paths import SPIRIT_LESSONS_FILE
from spirit_dataset.language import LANGUAGE_CODES, render
from spirit_dataset.memory import MIN_LOVE_LEVEL, compose_identity_prompt
from spirit_dataset.profile import SpiritProfile
from spirit_dataset.records import (
    CONVERSATION_RECORD_PREFIX,
    DIALOGUE_LESSON_PREFIX,
    JudgmentTrace,
    MemoryEvidence,
    SourceClass,
    SourceKind,
    SourceReference,
    SpiritTrainingRecord,
    TrainingTask,
)
from spirit_dataset.situations import LOBBY_TYPE_SITUATIONS

INTRODUCTION_QUESTIONS: dict[str, tuple[str, ...]] = {
    "ko": ("너는 누구야?", "너 자신을 소개해 줄래?", "네 이야기를 직접 듣고 싶어."),
    "en": ("Who are you?", "Please introduce yourself.", "Tell me about yourself."),
    "zh_tw": ("妳是誰？", "可以介紹妳自己嗎？", "我想聽妳親自介紹自己。"),
}
AFFECTION_QUESTIONS: dict[str, tuple[str, ...]] = {
    "ko": ("지금 나한테 해 주고 싶은 말이 있어?", "나에게 품은 마음을 네 말로 들려줘."),
    "en": ("Is there something you want to tell me?", "Tell me how you feel about me."),
    "zh_tw": ("妳現在有什麼想對我說的嗎？", "用妳自己的話告訴我，妳對我的心意。"),
}


def canonical_conversation_lessons(
    profile: SpiritProfile, canonical_records: list[SpiritTrainingRecord]
) -> list[SpiritTrainingRecord]:
    """Connect unmodified self-introductions and affection to the Savior's speech."""
    language = LANGUAGE_CODES[profile.language]
    affection_situations = {
        render(LOBBY_TYPE_SITUATIONS[kind], profile.language)
        for kind in ("Love1", "Love2", "Love3")
    }
    lessons = []
    for record in canonical_records:
        if record.task is not TrainingTask.PERSONA_SPEECH:
            continue
        if record.source.table == "HeroDesc.introduction":
            questions = INTRODUCTION_QUESTIONS[language]
        elif (
            record.source.kind is SourceKind.LOBBY
            and record.judgment.situation in affection_situations
        ):
            questions = AFFECTION_QUESTIONS[language]
        else:
            continue
        lessons.extend(
            replace(
                record,
                id=f"{CONVERSATION_RECORD_PREFIX}{record.id}:{index}",
                origin_id=record.origin_id or record.id,
                prompt=[record.prompt[0], SFTRecordTurn("user", question)],
                judgment=JudgmentTrace(question),
                source_class=SourceClass.DERIVED_SPEECH,
                evidence=(
                    *record.evidence,
                    MemoryEvidence(
                        SourceClass.CANON_DIALOGUE,
                        f"{record.source.table}:{list(record.source.keys)}",
                    ),
                ),
            )
            for index, question in enumerate(questions)
        )
    return lessons


@lru_cache(maxsize=1)
def lesson_specification() -> dict[str, Any]:
    return json.loads(SPIRIT_LESSONS_FILE.read_text(encoding="utf-8"))


def persona_lessons(
    profile: SpiritProfile, canonical_records: list[SpiritTrainingRecord]
) -> list[SpiritTrainingRecord]:
    specification = lesson_specification()
    authored = specification["spirits"].get(str(profile.identity.hero_no))
    if authored is None:
        return []  # No authored material exists for this identity; never borrow another's.
    style = authored["style_source"]
    matching = [
        record
        for record in canonical_records
        if record.source.table == style["table"] and list(record.source.keys) == style["keys"]
    ]
    if len(matching) != 1:
        raise EvaiError(f"Lesson style evidence is not unique for {profile.identity.slug}")
    source = matching[0]
    if profile.language == "kr" and source.completion[0].content != style["quote"]:
        raise EvaiError(f"Lesson style evidence changed for {profile.identity.slug}")
    language = LANGUAGE_CODES[profile.language]
    prefix = (
        DIALOGUE_LESSON_PREFIX if authored.get("conversation_extension") else "persona_lesson:"
    )
    expressions = {
        topic: answers[language].format(name=profile.identity.name)
        for topic, answers in authored["answers"].items()
    }
    expressions["name"] = profile.identity.name
    lessons = []
    for index, topic in enumerate(authored["answers"]):
        knowledge = specification["topics"][topic]
        questions = [
            knowledge["questions"][language],
            *knowledge.get("paraphrases", {}).get(language, []),
        ]
        for variant, question in enumerate(questions):
            lessons.append(
                SpiritTrainingRecord(
                    id=f"{prefix}{topic}:{variant}",
                    source=SourceReference(
                        SourceKind.PERSONA_LESSON,
                        "authored_lessons",
                        (profile.identity.hero_no, index, variant),
                        source_class=SourceClass.DERIVED_SPEECH,
                    ),
                    love_level=MIN_LOVE_LEVEL,
                    judgment=JudgmentTrace(question),
                    prompt=[
                        SFTRecordTurn(
                            "system",
                            compose_identity_prompt(
                                profile.identity_memory, MIN_LOVE_LEVEL, language=profile.language
                            ),
                        ),
                        *[
                            SFTRecordTurn(
                                turn["role"],
                                turn["content"].format_map(expressions),
                            )
                            for turn in knowledge["history"][language]
                        ],
                        SFTRecordTurn("user", question),
                    ],
                    completion=[SFTRecordTurn("assistant", expressions[topic])],
                    source_class=SourceClass.DERIVED_SPEECH,
                    evidence=(
                        MemoryEvidence(
                            SourceClass.CANON_DIALOGUE, f"{style['table']}:{style['keys']}"
                        ),
                        MemoryEvidence(SourceClass(knowledge["source_class"]), knowledge["fact"]),
                    ),
                    # Authored expressions and all translations stay with their style source.
                    event_keys=source.event_keys,
                    origin_id=source.id,
                    language=language,
                )
            )
    lessons.extend(_conversation_lessons(profile, authored, source))
    return lessons


def _conversation_lessons(
    profile: SpiritProfile, authored: dict[str, Any], source: SpiritTrainingRecord
) -> list[SpiritTrainingRecord]:
    language = LANGUAGE_CODES[profile.language]
    system = SFTRecordTurn(
        "system",
        compose_identity_prompt(profile.identity_memory, MIN_LOVE_LEVEL, language=profile.language),
    )
    lessons: list[SpiritTrainingRecord] = []
    identifiers: set[str] = set()
    for conversation_index, conversation in enumerate(authored.get("conversations", ())):
        identifier = conversation["id"]
        if identifier in identifiers:
            raise EvaiError(
                f"Duplicate authored conversation: {profile.identity.slug}:{identifier}"
            )
        identifiers.add(identifier)
        turns = [
            SFTRecordTurn(turn["role"], turn["content"].format(name=profile.identity.name))
            for turn in conversation["turns"][language]
        ]
        if len(turns) < 2 or len(turns) % 2 or any(
            turn.role != ("user" if index % 2 == 0 else "assistant") or not turn.content.strip()
            for index, turn in enumerate(turns)
        ):
            raise EvaiError(
                f"Invalid authored conversation turns: {profile.identity.slug}:{identifier}"
            )
        for index in range(1, len(turns), 2):
            lessons.append(
                SpiritTrainingRecord(
                    id=f"{DIALOGUE_LESSON_PREFIX}conversation:{identifier}:{index // 2}",
                    source=SourceReference(
                        SourceKind.PERSONA_LESSON,
                        "authored_conversations",
                        (profile.identity.hero_no, conversation_index, index // 2),
                        source_class=SourceClass.DERIVED_SPEECH,
                    ),
                    love_level=MIN_LOVE_LEVEL,
                    judgment=JudgmentTrace(turns[index - 1].content),
                    prompt=[system, *turns[:index]],
                    completion=[turns[index]],
                    source_class=SourceClass.DERIVED_SPEECH,
                    evidence=(
                        MemoryEvidence(
                            SourceClass.CANON_DIALOGUE,
                            f"{source.source.table}:{list(source.source.keys)}",
                        ),
                    ),
                    event_keys=source.event_keys,
                    origin_id=source.id,
                    language=language,
                )
            )
    return lessons
