"""Authored knowledge expression grounded in each spirit's own source speech."""

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.hashing import stable_index
from common.messages import SFTRecordTurn
from common.paths import SPIRIT_LESSON_EXTENSIONS_DIR, SPIRIT_LESSONS_FILE
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
from spirit_dataset.situations import (
    CANONICAL_SITUATION_TOPICS,
    LOBBY_TYPE_SITUATIONS,
    OUTING_AFFECTION_GROUP_ROLES,
    OUTING_AFFECTION_TOPIC,
    OUTING_GROUP_ROLE_MODULUS,
)

EXTENSION_TABLE_PREFIX = "lesson_extension:"
AUTHORED_QUESTION_VARIANT = 0
LESSON_LANGUAGES: tuple[str, ...] = ("ko", "en", "zh_tw")
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
    situation_topics = {
        render(situation, profile.language): topic
        for situation, topic in CANONICAL_SITUATION_TOPICS.items()
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
        elif record.judgment.situation in situation_topics:
            topics = [situation_topics[record.judgment.situation]]
            if (
                record.source.kind is SourceKind.TRIP
                and record.source.keys[0] % OUTING_GROUP_ROLE_MODULUS
                in OUTING_AFFECTION_GROUP_ROLES
            ):
                topics.append(OUTING_AFFECTION_TOPIC)
            questions = tuple(
                question for topic in topics for question in topic_questions(topic, language)
            )
        else:
            continue
        source_identity = f"{record.source.table}:{list(record.source.keys)}"
        speech = "\n".join(turn.content for turn in record.completion)
        index = stable_index(speech, len(questions))
        question = questions[index]
        lessons.append(
            replace(
                record,
                id=f"{CONVERSATION_RECORD_PREFIX}{record.id}:{index}",
                origin_id=record.origin_id or record.id,
                prompt=[record.prompt[0], SFTRecordTurn("user", question)],
                judgment=JudgmentTrace(question),
                source_class=SourceClass.DERIVED_SPEECH,
                evidence=(
                    *record.evidence,
                    MemoryEvidence(SourceClass.CANON_DIALOGUE, source_identity),
                ),
            )
        )
    return lessons


@dataclass(frozen=True)
class LessonSpecification:
    name: str
    lesson_table: str
    conversation_table: str
    topics: dict[str, Any]
    spirits: dict[str, Any]

    @property
    def id_scope(self) -> str:
        return f"{self.name}:" if self.name else ""


@lru_cache(maxsize=1)
def lesson_specification() -> dict[str, Any]:
    return json.loads(SPIRIT_LESSONS_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def lesson_specifications() -> tuple[LessonSpecification, ...]:
    base = lesson_specification()
    specifications = [
        LessonSpecification(
            "", "authored_lessons", "authored_conversations", base["topics"], base["spirits"]
        )
    ]
    for path in lesson_extension_files():
        extension = json.loads(path.read_text(encoding="utf-8"))
        own_topics = extension.get("topics", {})
        problems = extension_format_problems(extension, base["topics"])
        if problems:
            raise EvaiError(
                f"Lesson extension {path.name} does not follow the curriculum format:\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )
        specifications.append(
            LessonSpecification(
                path.stem,
                f"{EXTENSION_TABLE_PREFIX}{path.stem}",
                f"{EXTENSION_TABLE_PREFIX}{path.stem}:conversations",
                {**base["topics"], **own_topics},
                extension.get("spirits", {}),
            )
        )
    return tuple(specifications)


def extension_format_problems(
    extension: dict[str, Any], base_topics: dict[str, Any]
) -> list[str]:
    problems: list[str] = []
    own_topics = extension.get("topics", {})
    for topic in sorted(set(own_topics) & set(base_topics)):
        problems.append(f"topic '{topic}' redefines a base topic")
    topics = {**base_topics, **own_topics}
    for topic, knowledge in own_topics.items():
        for field in ("source_class", "fact", "questions", "history"):
            if field not in knowledge:
                problems.append(f"topic '{topic}' lacks '{field}'")
        history = knowledge.get("history")
        questions = knowledge.get("questions")
        for language in LESSON_LANGUAGES:
            if not isinstance(history, dict) or not isinstance(history.get(language), list):
                problems.append(f"topic '{topic}' history needs a list for '{language}'")
            if not isinstance(questions, dict) or not questions.get(language):
                problems.append(f"topic '{topic}' questions need '{language}'")
    for hero_no, authored in extension.get("spirits", {}).items():
        style = authored.get("style_source")
        if not isinstance(style, dict) or not {"table", "keys", "quote"} <= set(style):
            problems.append(f"spirit {hero_no} style_source needs table, keys and quote")
        answers = authored.get("answers")
        if not isinstance(answers, dict):
            problems.append(f"spirit {hero_no} answers must map topics to languages")
            continue
        for topic, texts in answers.items():
            if topic not in topics:
                problems.append(f"spirit {hero_no} answers unknown topic '{topic}'")
            elif not isinstance(texts, dict) or any(
                not texts.get(language) for language in LESSON_LANGUAGES
            ):
                problems.append(f"spirit {hero_no} topic '{topic}' needs ko, en and zh_tw")
        conversations = authored.get("conversations", [])
        if not isinstance(conversations, list) or any(
            not isinstance(item, dict) or not {"id", "turns"} <= set(item)
            for item in conversations
        ):
            problems.append(f"spirit {hero_no} conversations must be a list of id/turns")
    return problems


def lesson_extension_files() -> list[Path]:
    if not SPIRIT_LESSON_EXTENSIONS_DIR.is_dir():
        return []
    return sorted(SPIRIT_LESSON_EXTENSIONS_DIR.glob("*.json"))


def topic_questions(
    topic: str, language: str, topics: dict[str, Any] | None = None
) -> tuple[str, ...]:
    knowledge = (topics if topics is not None else lesson_specification()["topics"])[topic]
    return (
        knowledge["questions"][language],
        *knowledge.get("paraphrases", {}).get(language, []),
    )


def persona_lessons(
    profile: SpiritProfile, canonical_records: list[SpiritTrainingRecord]
) -> list[SpiritTrainingRecord]:
    return [
        lesson
        for specification in lesson_specifications()
        for lesson in _specification_lessons(profile, canonical_records, specification)
    ]


def _specification_lessons(
    profile: SpiritProfile,
    canonical_records: list[SpiritTrainingRecord],
    specification: LessonSpecification,
) -> list[SpiritTrainingRecord]:
    authored = specification.spirits.get(str(profile.identity.hero_no))
    if authored is None:
        return []
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
    ) + specification.id_scope
    expressions = {
        topic: answers[language].format(name=profile.identity.name)
        for topic, answers in authored["answers"].items()
    }
    expressions["name"] = profile.identity.name
    lessons = []
    for index, topic in enumerate(authored["answers"]):
        knowledge = specification.topics[topic]
        question = knowledge["questions"][language]
        lessons.append(
            SpiritTrainingRecord(
                id=f"{prefix}{topic}:{AUTHORED_QUESTION_VARIANT}",
                source=SourceReference(
                    SourceKind.PERSONA_LESSON,
                    specification.lesson_table,
                    (profile.identity.hero_no, index, AUTHORED_QUESTION_VARIANT),
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
                event_keys=source.event_keys,
                origin_id=source.id,
                language=language,
            )
        )
    lessons.extend(_conversation_lessons(profile, authored, source, specification))
    return lessons


def _conversation_lessons(
    profile: SpiritProfile,
    authored: dict[str, Any],
    source: SpiritTrainingRecord,
    specification: LessonSpecification,
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
                    id=(
                        f"{DIALOGUE_LESSON_PREFIX}{specification.id_scope}"
                        f"conversation:{identifier}:{index // 2}"
                    ),
                    source=SourceReference(
                        SourceKind.PERSONA_LESSON,
                        specification.conversation_table,
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
