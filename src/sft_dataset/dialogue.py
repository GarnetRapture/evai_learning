from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from persona.schema import DialogueEntry, PersonaData

SAVIOR_SPEAKER_NAMES: frozenset[str] = frozenset({"구원자", "Savior", "savior"})
SYSTEM_SPEAKER_NAMES: frozenset[str] = frozenset({"system", "시스템", "안내"})


class SpeakerRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    OTHER_CHARACTER = "other_character"


class TurnClassification(StrEnum):
    ACCEPTED = "accepted"
    EXCLUDED_ASSISTANT_IDENTITY = "excluded_assistant_identity"
    EXCLUDED_TOOL_BEHAVIOR = "excluded_tool_behavior"
    EXCLUDED_SYSTEM_MESSAGE = "excluded_system_message"
    EXCLUDED_OTHER_CHARACTER = "excluded_other_character"
    EXCLUDED_MINOR_OR_AGE_AMBIGUOUS = "excluded_minor_or_age_ambiguous"
    EXCLUDED_NONCONSENSUAL_OR_EXPLOITATIVE = "excluded_nonconsensual_or_exploitative"
    EXCLUDED_REAL_PERSON_SEXUALIZATION = "excluded_real_person_sexualization"
    NEEDS_REVIEW = "needs_review"
    EXCLUDED_UNANSWERED_CONTEXT = "excluded_unanswered_context"


@dataclass(frozen=True)
class DialogueTurn:
    role: SpeakerRole
    speaker_name: str
    content: str
    classification: TurnClassification = TurnClassification.ACCEPTED
    source_file: str = ""
    source_index: int = 0


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
    lowered = content.lower().strip()

    if role == SpeakerRole.SYSTEM or speaker_name.lower() in SYSTEM_SPEAKER_NAMES:
        return TurnClassification.EXCLUDED_SYSTEM_MESSAGE

    if role == SpeakerRole.OTHER_CHARACTER:
        return TurnClassification.EXCLUDED_OTHER_CHARACTER

    if any(pat in lowered for pat in ASSISTANT_IDENTITY_PATTERNS):
        return TurnClassification.EXCLUDED_ASSISTANT_IDENTITY

    if any(pat in lowered for pat in TOOL_BEHAVIOR_PATTERNS):
        return TurnClassification.EXCLUDED_TOOL_BEHAVIOR

    if any(pat in lowered for pat in MINOR_OR_AGE_AMBIGUOUS_PATTERNS):
        return TurnClassification.EXCLUDED_MINOR_OR_AGE_AMBIGUOUS

    if any(pat in lowered for pat in NONCONSENSUAL_OR_EXPLOITATIVE_PATTERNS):
        return TurnClassification.EXCLUDED_NONCONSENSUAL_OR_EXPLOITATIVE

    if source_type == "story":
        return TurnClassification.NEEDS_REVIEW

    return TurnClassification.ACCEPTED


def resolve_speaker_role(speaker_name: str, persona_name: str) -> SpeakerRole:
    if speaker_name == persona_name:
        return SpeakerRole.ASSISTANT
    if speaker_name in SAVIOR_SPEAKER_NAMES:
        return SpeakerRole.USER
    return SpeakerRole.OTHER_CHARACTER


@dataclass(frozen=True)
class DialogueExchange:
    exchange_id: str
    source_type: str
    source_index: int
    prompt: list[DialogueTurn]
    completion: list[DialogueTurn]
    assistant_segments: list[str] = field(default_factory=list)
    classification: TurnClassification = TurnClassification.ACCEPTED


@dataclass(frozen=True)
class DialogueExtractionResult:
    persona_id: str
    persona_name: str
    evertalk_exchanges: list[DialogueExchange] = field(default_factory=list)
    story_exchanges: list[DialogueExchange] = field(default_factory=list)
    excluded_turns_count: int = 0
    errors: list[str] = field(default_factory=list)


CLASSIFICATION_PRIORITY: tuple[TurnClassification, ...] = (
    TurnClassification.EXCLUDED_UNANSWERED_CONTEXT,
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


def aggregate_classification(turns: Sequence[DialogueTurn]) -> TurnClassification:
    present = {turn.classification for turn in turns}
    for candidate in CLASSIFICATION_PRIORITY:
        if candidate in present:
            return candidate
    return TurnClassification.ACCEPTED


def group_consecutive_turns(turns: Sequence[DialogueTurn]) -> list[list[DialogueTurn]]:
    groups: list[list[DialogueTurn]] = []
    for turn in turns:
        if groups and groups[-1][-1].role == turn.role:
            groups[-1].append(turn)
        else:
            groups.append([turn])
    return groups


def build_channel_exchanges(
    entries: Sequence[DialogueEntry],
    persona_name: str,
    source_type: str,
    source_file: str,
) -> list[DialogueExchange]:
    turns: list[DialogueTurn] = []
    for raw_index, entry in enumerate(entries):
        role = resolve_speaker_role(entry.speaker, persona_name)
        turns.append(
            DialogueTurn(
                role=role,
                speaker_name=entry.speaker,
                content=entry.message,
                classification=classify_dialogue_turn(
                    role, entry.speaker, entry.message, source_type
                ),
                source_file=source_file,
                source_index=raw_index,
            )
        )

    exchanges: list[DialogueExchange] = []
    pending_prompt: list[DialogueTurn] = []

    for group in group_consecutive_turns(turns):
        if group[0].role is not SpeakerRole.ASSISTANT:
            pending_prompt.extend(group)
            continue

        exchange_index = len(exchanges)
        exchanges.append(
            DialogueExchange(
                exchange_id=f"{source_type}:{exchange_index}",
                source_type=source_type,
                source_index=exchange_index,
                prompt=list(pending_prompt),
                completion=list(group),
                assistant_segments=[turn.content for turn in group],
                classification=aggregate_classification([*pending_prompt, *group]),
            )
        )
        pending_prompt = []

    if pending_prompt:
        exchanges.append(
            DialogueExchange(
                exchange_id=f"{source_type}:{len(exchanges)}",
                source_type=source_type,
                source_index=len(exchanges),
                prompt=pending_prompt,
                completion=[],
                classification=TurnClassification.EXCLUDED_UNANSWERED_CONTEXT,
            )
        )
    return exchanges


def extract_dialogue_exchanges(
    persona: PersonaData, persona_id: str, source_file: str
) -> DialogueExtractionResult:
    errors: list[str] = []

    try:
        evertalk_exchanges = build_channel_exchanges(
            persona.dialogues.evertalk, persona.name, "evertalk", source_file
        )
    except Exception as err:
        evertalk_exchanges = []
        errors.append(f"evertalk extraction failed: {err}")

    try:
        story_exchanges = build_channel_exchanges(
            persona.dialogues.story, persona.name, "story", source_file
        )
    except Exception as err:
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
