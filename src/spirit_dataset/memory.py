import json
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.paths import SPIRIT_MEMORY_DIR
from game_data.database import open_tbl_database
from game_data.story import BOND_STORY_TYPES, MAIN_STORY_TYPE, StoryRepository
from spirit_dataset.language import render
from spirit_dataset.records import MEMORY_LINE_MAX_LENGTH, MemoryEvidence, SelfMemory, SourceClass
from spirit_dataset.roster import SpiritIdentity
from spirit_dataset.situations import LOVE_LEVEL_MEMORY

ALTERNATE_ENDING_AFFINITIES: frozenset[int] = frozenset({39, 81})
MIN_LOVE_LEVEL = 1
MAX_LOVE_LEVEL = 40


@dataclass(frozen=True)
class PastMemory:
    text: str
    story_no: int
    story_type: int
    love_level_min: int
    talk_keys: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_class": (
                "shared_world" if self.story_type == MAIN_STORY_TYPE else "personal_episode"
            ),
            **SelfMemory(
                self.text,
                (
                    MemoryEvidence(SourceClass.CANON_STORY, f"StoryInfo.No={self.story_no}"),
                    *(
                        MemoryEvidence(SourceClass.CANON_DIALOGUE, f"Talk.No={key}")
                        for key in self.talk_keys
                    ),
                ),
            ).to_dict(),
            "story_no": self.story_no,
            "story_type": self.story_type,
            "love_level_min": self.love_level_min,
            "talk_keys": list(self.talk_keys),
        }


def spirit_memory_path(slug: str) -> Path:
    return SPIRIT_MEMORY_DIR / f"{slug}.jsonl"


class PastMemoryRepository:
    def __init__(self, story_repository: StoryRepository) -> None:
        self._story_repository = story_repository
        with (
            closing(open_tbl_database("story")) as story,
            closing(open_tbl_database("evertalk")) as evertalk,
        ):
            messenger_levels: dict[int, int] = {
                row["GroupNo"]: row["LoveLevel"]
                for row in evertalk.execute("SELECT GroupNo, LoveLevel FROM EverTalk")
            }
            self._stories: dict[int, tuple[int, int, int, int, int]] = {
                row["No"]: (
                    row["StoryType"],
                    row["Act"],
                    row["EndingAffinity"],
                    messenger_levels.get(row["MessengerGroup"], MIN_LOVE_LEVEL),
                    row["TalkGroup"],
                )
                for row in story.execute(
                    "SELECT No, StoryType, Act, EndingAffinity, MessengerGroup, "
                    "TalkGroup FROM StoryInfo"
                )
            }

    def validate(self, identity: SpiritIdentity, raw: dict[str, Any], location: str) -> PastMemory:
        try:
            text = str(raw["text"]).strip()
            story_no = int(raw["story_no"])
            love_level_min = int(raw.get("love_level_min", MIN_LOVE_LEVEL))
            talk_keys = tuple(int(key) for key in raw.get("talk_keys", ()))
        except (KeyError, TypeError, ValueError) as err:
            raise EvaiError(f"{location}: invalid memory row ({err})") from err
        if not text or len(text) > MEMORY_LINE_MAX_LENGTH:
            raise EvaiError(
                f"{location}: memory text must be 1..{MEMORY_LINE_MAX_LENGTH} chars: {text!r}"
            )
        if not MIN_LOVE_LEVEL <= love_level_min <= MAX_LOVE_LEVEL:
            raise EvaiError(f"{location}: love_level_min out of range: {love_level_min}")
        story = self._stories.get(story_no)
        if story is None:
            raise EvaiError(f"{location}: StoryInfo.No {story_no} does not exist")
        story_type, act, ending_affinity, unlock_level, talk_group = story
        love_level_min = max(love_level_min, unlock_level)
        if story_type not in {*BOND_STORY_TYPES, MAIN_STORY_TYPE}:
            raise EvaiError(
                f"{location}: story {story_no} is StoryType {story_type}; "
                "memory requires an owned bond story or an evidenced main-story experience"
            )
        if story_type in BOND_STORY_TYPES and act != identity.hero_no:
            raise EvaiError(
                f"{location}: bond story {story_no} belongs to hero {act}, not {identity.hero_no}"
            )
        if ending_affinity in ALTERNATE_ENDING_AFFINITIES:
            raise EvaiError(f"{location}: bond story {story_no} is an alternate ending branch")
        if story_type == MAIN_STORY_TYPE or talk_keys:
            lines = self._story_repository.lines(talk_group)
            available_keys = {line.talk_no for line in lines}
            if not talk_keys or not set(talk_keys) <= available_keys:
                raise EvaiError(f"{location}: memory needs visible Talk evidence from {story_no}")
        return PastMemory(
            text=text,
            story_no=story_no,
            story_type=story_type,
            love_level_min=love_level_min,
            talk_keys=talk_keys,
        )

    def load(self, identity: SpiritIdentity) -> list[PastMemory]:
        path = spirit_memory_path(identity.slug)
        if not path.exists():
            return []
        memories: list[PastMemory] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                location = f"{path.name}:{line_number}"
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as err:
                    raise EvaiError(f"{location}: invalid JSON ({err.msg})") from err
                memories.append(self.validate(identity, raw, location))
        return memories


def compose_system_memory(
    identity_memory: tuple[str, ...],
    past_memories: list[PastMemory],
    love_level: int | None,
) -> tuple[str, ...]:
    factual = identity_memory
    if love_level is not None:
        factual = (*factual, LOVE_LEVEL_MEMORY.format(level=love_level))
    return (*factual, *select_episodic_memory(past_memories, love_level))


def compose_identity_prompt(
    identity_memory: tuple[str, ...],
    love_level: int | None,
    memory_target: str | None = None,
    language: str = "kr",
) -> str:
    from spirit_dataset.profile import core_identity_memory

    # The same owner composes training, evaluation and chat identity.
    # Self-memory exercises must not receive their own answer verbatim.
    identity = tuple(
        line for line in core_identity_memory(identity_memory) if line != memory_target
    )
    return "\n".join(
        (
            *identity,
            render(
                LOVE_LEVEL_MEMORY,
                language,
                level=love_level or MIN_LOVE_LEVEL,
            ),
        )
    )


def past_memory_from_dict(raw: dict[str, Any]) -> PastMemory:
    return PastMemory(
        text=str(raw["text"]),
        story_no=int(raw["story_no"]),
        story_type=int(raw["story_type"]),
        love_level_min=int(raw["love_level_min"]),
        talk_keys=tuple(int(key) for key in raw.get("talk_keys", ())),
    )


def select_episodic_memory(
    past_memories: list[PastMemory], love_level: int | None
) -> tuple[str, ...]:
    level = love_level if love_level is not None else MIN_LOVE_LEVEL
    return tuple(
        dict.fromkeys(memory.text for memory in past_memories if memory.love_level_min <= level)
    )
