"""Canonical event aliases shared by speech, translations and derived memories."""

import re

from game_data.story import MAIN_STORY_TYPE, StoryEpisode
from spirit_dataset.records import MemoryEvidence, SourceReference


class EventIndex:
    def __init__(self, episodes: list[StoryEpisode]) -> None:
        self._aliases: dict[str, tuple[str, ...]] = {}
        for episode in episodes:
            keys = [f"StoryInfo:{episode.story_no}", f"Talk:{episode.talk_group}"]
            if episode.messenger_group:
                keys.append(f"EverTalkDesc:{episode.messenger_group}")
            if episode.story_type == MAIN_STORY_TYPE:
                keys.append(f"main_story:{episode.chapter}-{episode.episode}")
            for key in keys:
                self._aliases[key] = tuple(keys)

    def keys(
        self,
        source: SourceReference,
        evidence: tuple[MemoryEvidence, ...],
    ) -> tuple[str, ...]:
        keys: set[str] = set()
        if source.story_no is not None:
            keys.add(f"StoryInfo:{source.story_no}")
        if source.table != "self_memory":
            table = "HeroDesc" if source.table.startswith("HeroDesc.") else source.table
            keys.add(f"{table}:{source.keys[0]}")
        for item in evidence:
            reference = item.reference
            keys.add(f"memory:{reference}")
            if reference.startswith("main_story "):
                keys.update(f"main_story:{part}" for part in re.findall(r"\d+-\d+", reference))
            match = re.match(r"HeroDesc.HeroNo=(\d+)", reference)
            if match:
                keys.add(f"HeroDesc:{match[1]}")
        for key in tuple(keys):
            keys.update(self._aliases.get(key, ()))
        return tuple(sorted(keys))
