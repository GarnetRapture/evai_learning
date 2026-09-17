"""Authored knowledge expression grounded in each spirit's own source speech."""

import json
from functools import lru_cache
from typing import Any

from common.errors import EvaiError
from common.messages import SFTRecordTurn
from common.paths import SPIRIT_LESSONS_FILE
from spirit_dataset.language import LANGUAGE_CODES
from spirit_dataset.memory import MIN_LOVE_LEVEL, compose_identity_prompt
from spirit_dataset.profile import SpiritProfile
from spirit_dataset.records import (
    JudgmentTrace,
    MemoryEvidence,
    SourceClass,
    SourceKind,
    SourceReference,
    SpiritTrainingRecord,
)


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
                    id=f"persona_lesson:{topic}:{variant}",
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
    return lessons
