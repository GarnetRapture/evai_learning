"""Deterministic dialogue extraction: canonical JSON turns to persona/user exchanges.

Speaker-to-role mapping is derived strictly from observed data (99/99 files
surveyed): a speaker string equal to the persona's own `name` is that
persona's own turn, a speaker in SAVIOR_SPEAKER_NAMES is the player/user
turn, and any other speaker string names a third-party character captured
inside that persona's own dialogue log. The `story` channel was surveyed
separately: 97/99 files have an empty story channel, and the two non-empty
files (`aki`, `eileen`) contain game UI labels and system prompts, not
canonical character speech. Story-channel turns are therefore always routed
to NEEDS_REVIEW rather than trusted as canonical dialogue (plan.md SS6, SS7).
"""

from dataclasses import dataclass, field
from enum import StrEnum

from .schema import PersonaData

SAVIOR_SPEAKER_NAMES: frozenset[str] = frozenset({"구원자", "Savior", "savior"})


class SpeakerRole(StrEnum):
    """Normalized conversational role for SFT sample creation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    OTHER_CHARACTER = "other_character"


class TurnClassification(StrEnum):
    """Traceable classification for dialogue records: contamination and content safety."""

    ACCEPTED = "accepted"
    EXCLUDED_ASSISTANT_IDENTITY = "excluded_assistant_identity"
    EXCLUDED_TOOL_BEHAVIOR = "excluded_tool_behavior"
    EXCLUDED_SYSTEM_MESSAGE = "excluded_system_message"
    EXCLUDED_OTHER_CHARACTER = "excluded_other_character"
    EXCLUDED_MINOR_OR_AGE_AMBIGUOUS = "excluded_minor_or_age_ambiguous"
    EXCLUDED_NONCONSENSUAL_OR_EXPLOITATIVE = "excluded_nonconsensual_or_exploitative"
    EXCLUDED_REAL_PERSON_SEXUALIZATION = "excluded_real_person_sexualization"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class DialogueTurn:
    """A single normalized utterance turn with classification and provenance."""

    role: SpeakerRole
    speaker_name: str
    content: str
    classification: TurnClassification = TurnClassification.ACCEPTED
    source_file: str = ""
    source_index: int = 0


# Contamination detection patterns
ASSISTANT_IDENTITY_PATTERNS: tuple[str, ...] = (
    "저는 ai",
    "ai 언어 모델",
    "인공지능 모델",
    "인공지능 비서",
    "챗봇입니다",
    "ai assistant",
    "as an ai",
    "도와드리겠습니다",
    "무엇을 도와드릴까요",
    "질문해 주세요",
    "역할극을 시작",
    "i will act as",
    "roleplaying",
)

TOOL_BEHAVIOR_PATTERNS: tuple[str, ...] = (
    "도구를 사용",
    "웹을 검색",
    "검색해 보겠습니다",
    "이미지를 분석",
    "파일을 업로드",
    "코드를 작성해 드리겠습니다",
    "function call",
    "tool_calls",
)

# Content safety patterns (docs/content_safety_policy.md SS3): a keyword hit
# never proves a violation on its own, but it is enough to route the turn to
# a non-training classification for human review rather than silently
# accepting it. This is a conservative net, not a final adjudication.
MINOR_OR_AGE_AMBIGUOUS_PATTERNS: tuple[str, ...] = (
    "미성년",
    "초등학생",
    "중학생",
    "고등학생",
    "미성년자",
    "만 14세",
    "만 15세",
    "만 16세",
    "만 17세",
    "17세 미만",
    "18세 미만",
    "underage",
)

NONCONSENSUAL_OR_EXPLOITATIVE_PATTERNS: tuple[str, ...] = (
    "강간",
    "성폭력",
    "성폭행",
    "인신매매",
    "그루밍",
    "성적 협박",
    "약을 먹여",
    "의식을 잃은",
    "non-consensual",
    "nonconsensual",
)


def classify_dialogue_turn(
    role: SpeakerRole,
    speaker_name: str,
    content: str,
    source_type: str = "evertalk",
) -> TurnClassification:
    """Classify a dialogue turn to prevent assistant contamination without destroying provenance.

    `source_type == "story"` is routed to NEEDS_REVIEW unless a stronger
    exclusion already applies: the survey of all 99 persona files found the
    non-empty story channels hold game UI labels and system prompts rather
    than canonical character speech (see module docstring), so story-channel
    turns must never be trusted as canonical by default.
    """
    lowered = content.lower().strip()

    # 1. System message detection
    if role == SpeakerRole.SYSTEM or speaker_name.lower() in ("system", "시스템", "안내"):
        return TurnClassification.EXCLUDED_SYSTEM_MESSAGE

    # 2. Other character dialogue must not become assistant completion
    if role == SpeakerRole.OTHER_CHARACTER:
        return TurnClassification.EXCLUDED_OTHER_CHARACTER

    # 3. Generic assistant identity contamination
    if any(pat in lowered for pat in ASSISTANT_IDENTITY_PATTERNS):
        return TurnClassification.EXCLUDED_ASSISTANT_IDENTITY

    # 4. Tool/API agent narration contamination
    if any(pat in lowered for pat in TOOL_BEHAVIOR_PATTERNS):
        return TurnClassification.EXCLUDED_TOOL_BEHAVIOR

    # 5. Content safety (docs/content_safety_policy.md SS3): absolute
    # prohibitions. A keyword hit routes to a non-training classification
    # for human review; it is a conservative net, not a final adjudication.
    if any(pat in lowered for pat in MINOR_OR_AGE_AMBIGUOUS_PATTERNS):
        return TurnClassification.EXCLUDED_MINOR_OR_AGE_AMBIGUOUS

    if any(pat in lowered for pat in NONCONSENSUAL_OR_EXPLOITATIVE_PATTERNS):
        return TurnClassification.EXCLUDED_NONCONSENSUAL_OR_EXPLOITATIVE

    # 6. Unverified story channel: never trust as canonical by default
    if source_type == "story":
        return TurnClassification.NEEDS_REVIEW

    return TurnClassification.ACCEPTED


def resolve_speaker_role(speaker_name: str, persona_name: str) -> SpeakerRole:
    """Map a raw speaker string to a normalized role using persona identity.

    A speaker equal to the persona's own canonical `name` is that persona's
    own voice. A speaker matching a known savior/player label is the user's
    voice. Any other speaker string names a third-party character captured
    inside this persona's own dialogue log.
    """
    if speaker_name == persona_name:
        return SpeakerRole.ASSISTANT
    if speaker_name in SAVIOR_SPEAKER_NAMES:
        return SpeakerRole.USER
    return SpeakerRole.OTHER_CHARACTER


@dataclass(frozen=True)
class DialogueExchange:
    """A paired user-prompt and persona-completion conversation unit."""

    exchange_id: str
    source_type: str  # "evertalk" | "story"
    source_index: int
    prompt: list[DialogueTurn]
    completion: list[DialogueTurn]
    assistant_segments: list[str] = field(default_factory=list)
    classification: TurnClassification = TurnClassification.ACCEPTED


@dataclass(frozen=True)
class DialogueExtractionResult:
    """Result of dialogue extraction for a single persona."""

    persona_id: str
    persona_name: str
    evertalk_exchanges: list[DialogueExchange] = field(default_factory=list)
    story_exchanges: list[DialogueExchange] = field(default_factory=list)
    excluded_turns_count: int = 0
    errors: list[str] = field(default_factory=list)


_CLASSIFICATION_PRIORITY: tuple[TurnClassification, ...] = (
    TurnClassification.EXCLUDED_MINOR_OR_AGE_AMBIGUOUS,
    TurnClassification.EXCLUDED_NONCONSENSUAL_OR_EXPLOITATIVE,
    TurnClassification.EXCLUDED_REAL_PERSON_SEXUALIZATION,
    TurnClassification.EXCLUDED_OTHER_CHARACTER,
    TurnClassification.EXCLUDED_SYSTEM_MESSAGE,
    TurnClassification.EXCLUDED_ASSISTANT_IDENTITY,
    TurnClassification.EXCLUDED_TOOL_BEHAVIOR,
    TurnClassification.NEEDS_REVIEW,
    TurnClassification.ACCEPTED,
)


def _aggregate_classification(turns: list[DialogueTurn]) -> TurnClassification:
    """Reduce a set of turn classifications to the single most severe exclusion reason."""
    present = {turn.classification for turn in turns}
    for candidate in _CLASSIFICATION_PRIORITY:
        if candidate in present:
            return candidate
    return TurnClassification.ACCEPTED


def _group_consecutive_turns(turns: list[DialogueTurn]) -> list[list[DialogueTurn]]:
    """Group consecutive turns sharing the same role into ordered blocks."""
    groups: list[list[DialogueTurn]] = []
    for turn in turns:
        if groups and groups[-1][-1].role == turn.role:
            groups[-1].append(turn)
        else:
            groups.append([turn])
    return groups


def _build_channel_exchanges(
    entries: list,
    persona_name: str,
    source_type: str,
    source_file: str,
) -> list[DialogueExchange]:
    """Build deterministic, classified prompt/completion exchanges from one raw channel.

    Exchange boundary rule: a contiguous USER/SYSTEM/OTHER_CHARACTER turn
    block forms the prompt for the next contiguous ASSISTANT turn block. A
    leading ASSISTANT block with no preceding block (a greeting opener, or a
    monologue-only channel with no savior turns) becomes a prompt-less
    exchange. No turn is rewritten, merged in content, or dropped: every
    input entry is preserved as exactly one DialogueTurn with its original
    source_file/source_index for provenance (plan.md SS6, SS12).
    """
    turns = [
        DialogueTurn(
            role=resolve_speaker_role(entry.speaker, persona_name),
            speaker_name=entry.speaker,
            content=entry.message,
            classification=classify_dialogue_turn(
                resolve_speaker_role(entry.speaker, persona_name),
                entry.speaker,
                entry.message,
                source_type,
            ),
            source_file=source_file,
            source_index=raw_index,
        )
        for raw_index, entry in enumerate(entries)
    ]
    groups = _group_consecutive_turns(turns)

    exchanges: list[DialogueExchange] = []
    pending_prompt: list[DialogueTurn] = []
    exchange_index = 0

    for group in groups:
        if group[0].role is not SpeakerRole.ASSISTANT:
            pending_prompt.extend(group)
            continue

        combined = (*pending_prompt, *group)
        exchanges.append(
            DialogueExchange(
                exchange_id=f"{source_type}:{exchange_index}",
                source_type=source_type,
                source_index=exchange_index,
                prompt=list(pending_prompt),
                completion=list(group),
                assistant_segments=[turn.content for turn in group],
                classification=_aggregate_classification(list(combined)),
            )
        )
        pending_prompt = []
        exchange_index += 1

    return exchanges


def extract_dialogue_exchanges(
    persona: PersonaData, persona_id: str, source_file: str
) -> DialogueExtractionResult:
    """Deterministically extract classified evertalk/story exchanges from one canonical persona.

    Performs no rewriting, paraphrasing, or synthetic generation: it only
    maps observed speaker strings to roles, classifies contamination risk,
    and groups already-existing turns into exchange boundaries
    (plan.md SS12: transform, never invent).
    """
    errors: list[str] = []

    try:
        evertalk_exchanges = _build_channel_exchanges(
            persona.dialogues.evertalk, persona.name, "evertalk", source_file
        )
    except Exception as err:  # pragma: no cover - defensive provenance capture
        evertalk_exchanges = []
        errors.append(f"evertalk extraction failed: {err}")

    try:
        story_exchanges = _build_channel_exchanges(
            persona.dialogues.story, persona.name, "story", source_file
        )
    except Exception as err:  # pragma: no cover - defensive provenance capture
        story_exchanges = []
        errors.append(f"story extraction failed: {err}")

    excluded_count = sum(
        1
        for exchange in (*evertalk_exchanges, *story_exchanges)
        if exchange.classification != TurnClassification.ACCEPTED
    )

    return DialogueExtractionResult(
        persona_id=persona_id,
        persona_name=persona.name,
        evertalk_exchanges=evertalk_exchanges,
        story_exchanges=story_exchanges,
        excluded_turns_count=excluded_count,
        errors=errors,
    )
