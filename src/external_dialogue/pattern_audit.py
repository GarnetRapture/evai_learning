import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.paths import INTIMACY_PATTERNS_FILE, KOREAN_ADULT_ROLEPLAY_FILE, REPORTS_DIR
from external_dialogue.adult_roleplay import load_korean_adult_roleplay_conversations
from external_dialogue.conversion import (
    CONCEPT_FOLLOWERS,
    REAL_WORLD_MARKERS,
    SPIRIT_ONLY_MARKERS,
    compile_terms,
)
from external_dialogue.patterns import (
    ADDRESS_FORM_PATTERN,
    FEMALE_ANATOMY,
    GENERAL_PEJORATIVE_PATTERN,
    KANA_PATTERN,
    MALE_ANATOMY,
    ORDINAL_STEMS,
    ROLE_TITLES,
    SELF_FEMALE_ANATOMY_PATTERN,
    SELF_MALE_ANATOMY_PATTERN,
    WORD_PATTERN,
    WORLD_COMPOUND_CORRECTIONS,
    WORLD_CONCEPT_CORRECTIONS,
    WORLD_TERM_CORRECTIONS,
    canon_address_forms,
    canon_word_stems,
    word_stem,
)
from external_dialogue.scenario_policy import (
    DIALOGUE_MINOR_PATTERN,
    DIALOGUE_NONCONSENSUAL_PATTERN,
    DIALOGUE_REVIEW_PATTERN,
    SCHOOL_SCENARIO_PATTERN,
)
from external_dialogue.speech_style import SpeechLevel
from spirit_dataset.situations import INTIMACY_SITUATION_TOPICS

AUDIT_REPORT_FILE = REPORTS_DIR / "dialogue_pattern_audit.json"
SAMPLES_PER_CHECK = 5
SAMPLE_TEXT_LENGTH = 160
SOURCE_ROLES: tuple[str, ...] = ("user", "assistant")
EXPECTED_FIRST_ROLE = "user"
EXPECTED_LAST_ROLE = "assistant"
SLOT_TOKENS: frozenset[str] = frozenset({"{spirit}", "{savior}"})
UNKNOWN_SLOT_PATTERN = re.compile(r"\{[^{}]*\}")
SLOT_HONORIFIC_PATTERN = re.compile(r"\{(?:spirit|savior)\}(?:님|씨|군|짱|쨩)(?![가-힣])")
DOUBLED_SLOT_PATTERN = re.compile(r"\{(?:spirit|savior)\}\s*\{(?:spirit|savior)\}")
SPIRIT_MALE_PATTERN = re.compile(rf"\{{spirit\}}의 ?(?:{MALE_ANATOMY})")
SAVIOR_FEMALE_PATTERN = re.compile(rf"\{{savior\}}의 ?(?:{FEMALE_ANATOMY})")
SPIRIT_SELF_NAMING_PATTERN = re.compile(r"(?:^|[.!?…♡♥]\s*)\{spirit\}\s*[,!?…]")
SAVIOR_SELF_NAMING_PATTERN = re.compile(r"(?:^|[.!?…♡♥]\s*)\{savior\}\s*[,!?…]")
SPELLED_OUT_GLOSS_PATTERN = re.compile(r"\((?:안|때문에|벌써|이제)\)")
DROPPED_COPULA_PATTERN = re.compile(r"정령(?:다|든|라)(?![가-힣])")
CONTEXT_DEPENDENT_PATTERN = re.compile(r"(?<![가-힣])(?:짐승|공포|노예|억지로|습격|조교|강제)")


@dataclass(frozen=True)
class PatternTurn:
    pattern_id: str
    row: int
    turn_index: int
    role: str
    text: str
    segments: tuple[dict[str, str], ...]


@dataclass
class AuditCheck:
    name: str
    description: str
    matcher: Callable[[PatternTurn], str | None]
    count: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


def _first_match(pattern: re.Pattern[str] | None, text: str) -> str | None:
    if pattern is None:
        return None
    match = pattern.search(text)
    return match.group(0) if match else None


def _first_marker(markers: tuple[str, ...], text: str) -> str | None:
    lowered = text.lower()
    return next((marker for marker in markers if marker.lower() in lowered), None)


def source_speaker_names() -> frozenset[str]:
    names: set[str] = set()
    for conversation in load_korean_adult_roleplay_conversations():
        names.update(
            form
            for form in (*conversation.names.character, *conversation.names.user)
            if not ADDRESS_FORM_PATTERN.fullmatch(form)
        )
    return frozenset(names)


CORRECTION_TABLES = (
    *WORLD_COMPOUND_CORRECTIONS,
    *WORLD_TERM_CORRECTIONS,
    *WORLD_CONCEPT_CORRECTIONS,
)


def correction_sources() -> tuple[str, ...]:
    return tuple(source for _, sources in CORRECTION_TABLES for source in sources) + ROLE_TITLES


def correction_targets() -> re.Pattern[str]:
    targets = sorted(
        {target for target, _ in CORRECTION_TABLES if target not in SLOT_TOKENS},
        key=len,
        reverse=True,
    )
    return re.compile("|".join(map(re.escape, targets)))


def _segment_texts(turn: PatternTurn) -> list[str]:
    return [segment["text"] for segment in turn.segments]


def build_checks(
    speaker_names: frozenset[str], canon_stems: frozenset[str], canon_forms: frozenset[str]
) -> list[AuditCheck]:
    residual_sources = compile_terms(correction_sources(), CONCEPT_FOLLOWERS)
    corrected_targets = correction_targets()
    foreign_names = frozenset(name for name in speaker_names if name not in canon_stems)

    def correction_source(turn: PatternTurn) -> str | None:
        return _first_match(residual_sources, corrected_targets.sub(" ", turn.text))

    def speaker_name(turn: PatternTurn) -> str | None:
        for word in WORD_PATTERN.findall(turn.text):
            if word in foreign_names or word_stem(word) in foreign_names:
                return word
        return None

    def foreign_honorific(turn: PatternTurn) -> str | None:
        for match in ADDRESS_FORM_PATTERN.finditer(turn.text):
            if match.group(2) in {"씨", "짱", "쨩"} and match.group(0) not in canon_forms:
                return match.group(0)
        return None

    def gender(turn: PatternTurn) -> str | None:
        own = SELF_MALE_ANATOMY_PATTERN if turn.role == "assistant" else SELF_FEMALE_ANATOMY_PATTERN
        for text in _segment_texts(turn):
            found = (
                _first_match(own, text)
                or _first_match(SPIRIT_MALE_PATTERN, text)
                or _first_match(SAVIOR_FEMALE_PATTERN, text)
            )
            if found:
                return found
        return None

    def slot_integrity(turn: PatternTurn) -> str | None:
        for text in _segment_texts(turn):
            unknown = next(
                (token for token in UNKNOWN_SLOT_PATTERN.findall(text) if token not in SLOT_TOKENS),
                None,
            )
            found = (
                unknown
                or _first_match(SLOT_HONORIFIC_PATTERN, text)
                or _first_match(DOUBLED_SLOT_PATTERN, text)
            )
            if found:
                return found
        return None

    def role_slot(turn: PatternTurn) -> str | None:
        pattern = (
            SPIRIT_SELF_NAMING_PATTERN if turn.role == "assistant" else SAVIOR_SELF_NAMING_PATTERN
        )
        return _first_match(pattern, turn.text)

    def minor(turn: PatternTurn) -> str | None:
        return _first_match(SCHOOL_SCENARIO_PATTERN, turn.text) or _first_match(
            DIALOGUE_MINOR_PATTERN, turn.text
        )

    def translation(turn: PatternTurn) -> str | None:
        for match in GENERAL_PEJORATIVE_PATTERN.finditer(turn.text):
            if match.group(1) not in ORDINAL_STEMS:
                return match.group(0)
        return _first_match(SPELLED_OUT_GLOSS_PATTERN, turn.text) or _first_match(
            DROPPED_COPULA_PATTERN, turn.text
        )

    return [
        AuditCheck("gender", "정령=여성·구원자=남성 모순", gender),
        AuditCheck("speaker_name", "원천 화자 이름 잔여", speaker_name),
        AuditCheck("foreign_honorific", "씨·짱·쨩 이름 호칭 잔여", foreign_honorific),
        AuditCheck("correction_source", "교정표 원천어·직함 잔여", correction_source),
        AuditCheck(
            "real_world",
            "현실 세계·정령 금지 표지",
            lambda turn: _first_marker(REAL_WORLD_MARKERS + SPIRIT_ONLY_MARKERS, turn.text),
        ),
        AuditCheck("kana", "일본어 문자", lambda turn: _first_match(KANA_PATTERN, turn.text)),
        AuditCheck("slot_integrity", "슬롯 무결성", slot_integrity),
        AuditCheck("role_slot", "화자가 자기 슬롯을 부름", role_slot),
        AuditCheck("minor_marker", "연령 불명확 표지", minor),
        AuditCheck(
            "nonconsent_marker",
            "비동의·검토 표지",
            lambda turn: _first_match(DIALOGUE_NONCONSENSUAL_PATTERN, turn.text)
            or _first_match(DIALOGUE_REVIEW_PATTERN, turn.text),
        ),
        AuditCheck(
            "context_marker",
            "문맥 의존 표지(참고)",
            lambda turn: _first_match(CONTEXT_DEPENDENT_PATTERN, turn.text),
        ),
        AuditCheck("translation_artifact", "번역 흔적·조사 누락", translation),
    ]


def _pattern_turns(pattern: dict[str, Any]) -> list[PatternTurn]:
    source = pattern["source"]
    return [
        PatternTurn(
            pattern_id=pattern["id"],
            row=int(source["row"]),
            turn_index=int(source["turn"]) + offset,
            role=turn["role"],
            text=" ".join(segment["text"] for segment in turn["segments"]),
            segments=tuple(turn["segments"]),
        )
        for offset, turn in enumerate(pattern["turns"])
    ]


def _structure_problem(pattern: dict[str, Any]) -> str | None:
    turns = pattern["turns"]
    roles = [turn["role"] for turn in turns]
    if not turns or roles[0] != EXPECTED_FIRST_ROLE or roles[-1] != EXPECTED_LAST_ROLE:
        return f"roles={roles[:1]}..{roles[-1:]}"
    if any(role not in SOURCE_ROLES for role in roles):
        return "unknown_role"
    if any(not segment["text"].strip() for turn in turns for segment in turn["segments"]):
        return "empty_segment"
    if any(not turn["segments"] for turn in turns):
        return "empty_turn"
    return None


def _field_problem(pattern: dict[str, Any]) -> str | None:
    situation = pattern.get("situation")
    if situation not in INTIMACY_SITUATION_TOPICS:
        return f"situation={situation}"
    if pattern.get("topic") != INTIMACY_SITUATION_TOPICS[situation]:
        return f"topic={pattern.get('topic')}"
    if pattern.get("level") not in {level.value for level in SpeechLevel}:
        return f"level={pattern.get('level')}"
    return None


def _source_originals(requests: set[tuple[int, int]]) -> dict[tuple[int, int], str]:
    rows = {row for row, _ in requests}
    originals: dict[tuple[int, int], str] = {}
    with KOREAN_ADULT_ROLEPLAY_FILE.open(encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if row not in rows:
                continue
            messages = [
                message["content"]
                for message in json.loads(line)["messages"]
                if message["role"] in SOURCE_ROLES
            ]
            for requested_row, turn_index in requests:
                if requested_row == row and turn_index < len(messages):
                    originals[(row, turn_index)] = messages[turn_index]
    return originals


def audit_patterns(path: Path = INTIMACY_PATTERNS_FILE) -> dict[str, Any]:
    checks = build_checks(source_speaker_names(), canon_word_stems(), canon_address_forms())
    structure = AuditCheck("structure", "패턴 구조", lambda _: None)
    fields = AuditCheck("fields", "상황·주제·말투 필드", lambda _: None)
    totals: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pattern = json.loads(line)
            totals["patterns"] += 1
            pattern_problems = (
                (structure, _structure_problem(pattern)),
                (fields, _field_problem(pattern)),
            )
            for check, problem in pattern_problems:
                if problem is not None:
                    check.count += 1
                    if len(check.samples) < SAMPLES_PER_CHECK:
                        check.samples.append({"id": pattern["id"], "match": problem})
            for turn in _pattern_turns(pattern):
                totals[f"{turn.role}_turns"] += 1
                for check in checks:
                    match = check.matcher(turn)
                    if match is None:
                        continue
                    check.count += 1
                    if len(check.samples) < SAMPLES_PER_CHECK:
                        check.samples.append(
                            {
                                "id": turn.pattern_id,
                                "row": turn.row,
                                "turn": turn.turn_index,
                                "role": turn.role,
                                "match": match,
                                "pattern": turn.text[:SAMPLE_TEXT_LENGTH],
                            }
                        )
    all_checks = [*checks, structure, fields]
    originals = _source_originals(
        {
            (sample["row"], sample["turn"])
            for check in all_checks
            for sample in check.samples
            if "row" in sample
        }
    )
    for check in all_checks:
        for sample in check.samples:
            if "row" in sample:
                sample["original"] = originals.get((sample["row"], sample["turn"]), "")[
                    :SAMPLE_TEXT_LENGTH
                ]
    report = {
        "file": str(path),
        "totals": dict(totals),
        "checks": [
            {
                "name": check.name,
                "description": check.description,
                "count": check.count,
                "samples": check.samples,
            }
            for check in all_checks
        ],
    }
    AUDIT_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
