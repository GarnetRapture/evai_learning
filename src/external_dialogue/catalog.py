from external_dialogue.korean_roleplay import (
    KOREAN_ROLEPLAY_SOURCE,
    load_korean_roleplay_conversations,
)
from external_dialogue.patterns import (
    INTIMACY_PATTERN_SOURCE,
    load_intimacy_pattern_conversations,
)
from external_dialogue.source import SourceConversation, SourceDataset

EXTERNAL_DIALOGUE_SOURCES: tuple[SourceDataset, ...] = (
    KOREAN_ROLEPLAY_SOURCE,
    INTIMACY_PATTERN_SOURCE,
)


def load_external_conversations() -> list[SourceConversation]:
    return [
        *load_korean_roleplay_conversations(),
        *load_intimacy_pattern_conversations(),
    ]


def external_dialogue_hashes() -> dict[str, str]:
    return {source.name: source.sha256() for source in EXTERNAL_DIALOGUE_SOURCES}
