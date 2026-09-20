import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.paths import INTIMACY_PATTERNS_FILE
from external_dialogue.adult_roleplay import (
    KOREAN_ADULT_ROLEPLAY_SOURCE,
    CharacterGender,
    load_korean_adult_roleplay_conversations,
    setting_genders,
)
from external_dialogue.conversion import (
    CONCEPT_FOLLOWERS,
    GRAMMATICAL_FOLLOWERS,
    SPIRIT_ONLY_MARKERS,
    compile_terms,
    mentions_real_world,
    substitute_names,
)
from external_dialogue.scenario_policy import dialogue_classification, scenario_classification
from external_dialogue.source import (
    SegmentKind,
    SourceConversation,
    SourceDataset,
    SourceNames,
    SourceTurn,
    TurnSegment,
)
from external_dialogue.speech_style import SpeechLevel, classify_level, sentence_endings
from sft_dataset.dialogue import TurnClassification
from sft_dataset.storage import sft_split_path
from spirit_dataset.records import SourceClass
from spirit_dataset.roster import roster_slugs
from spirit_dataset.situations import (
    BATH_TOGETHER_SITUATION,
    INTIMACY_SITUATION_TOPICS,
    LOBBY_TYPE_SITUATIONS,
    NIGHT_TOGETHER_SITUATION,
    TRIP_OPENING_SITUATION,
)

SPIRIT_SLOT = "{spirit}"
SAVIOR_SLOT = "{savior}"
PATTERN_NAMES = SourceNames(character=(SPIRIT_SLOT,), user=(SAVIOR_SLOT,))
INTIMACY_PATTERN_SOURCE = SourceDataset(
    name=f"{KOREAN_ADULT_ROLEPLAY_SOURCE.name}:intimacy_patterns",
    license=KOREAN_ADULT_ROLEPLAY_SOURCE.license,
    id_prefix="pattern",
    subset="intimacy_ko",
    path=INTIMACY_PATTERNS_FILE,
)
ROLE_ADDRESSES: tuple[str, ...] = (
    "주인님",
    "선생님",
    "선배님",
    "선배",
    "용사님",
    "손님",
    "공주님",
    "아가씨",
    "도련님",
)
SETTING_HEADER_WORDS: tuple[str, ...] = (
    "설정",
    "장면",
    "캐릭터",
    "인물",
    "톤",
    "세계",
    "롤 플레이",
    "대화",
    "회화",
    "유저",
    "당신",
)
GENERIC_SETTING_WORDS: frozenset[str] = frozenset(
    {
        "여성", "남성", "여자", "남자", "인간", "소유자", "가진다",
        "있다", "있는", "하는", "되는", "대한", "위해",
    }
)
PARTICLE_SUFFIXES: tuple[str, ...] = (
    "에게서", "에서", "에게", "한테", "으로", "로", "은", "는",
    "이", "가", "을", "를", "의", "와", "과", "도", "만",
)
MIN_WORD_LENGTH = 2
WORD_PATTERN = re.compile(r"[가-힣]{2,}")
KANA_PATTERN = re.compile(r"[぀-ヿ]")
ADDRESS_FORM_PATTERN = re.compile(
    r"(?<![가-힣{}])([가-힣]{1,5})(씨|님|짱|쨩|군)"
    r"(?=(?:이랑|에게|한테|이|가|은|는|을|를|의|도|랑|와|과|께|야|아|여)?(?![가-힣]))"
)
NAME_EVIDENCE_SUFFIXES: frozenset[str] = frozenset({"씨", "님", "짱", "쨩"})
VOCATIVE_FOLLOW_PATTERN = re.compile(r"\s*(?:[,!?…~♡♥.]|$)")
MIN_VOCATIVE_USES = 3
STAGING_SUFFIX = ".staging"
MALE_ANATOMY = "자지|페니스|육봉|거근|남근|고환|불알"
FEMALE_ANATOMY = "보지|유방|자궁|질내|질구|클리토리스|음핵|젖꼭지|유두|애액"
SELF_REFERENCES = "나의|내|저의|제"
SLOT_ANATOMY_JOINER = "의 ?"
SELF_MALE_ANATOMY_PATTERN = re.compile(f"(?:{SELF_REFERENCES}) ?(?:{MALE_ANATOMY})")
SELF_FEMALE_ANATOMY_PATTERN = re.compile(f"(?:{SELF_REFERENCES}) ?(?:{FEMALE_ANATOMY})")
ROLE_SLOTS: dict[str, str] = {"assistant": SPIRIT_SLOT, "user": SAVIOR_SLOT}
GENDER_CONTRADICTIONS: dict[str, re.Pattern[str]] = {
    "assistant": SELF_MALE_ANATOMY_PATTERN,
    "user": SELF_FEMALE_ANATOMY_PATTERN,
}
COUNTERPART_ROLE: dict[str, str] = {"assistant": "user", "user": "assistant"}
SWAP_MARKER = "{swapping}"
GENDER_SAME = "gender_same"
GENDER_SWAPPED = "gender_swapped"
GENDER_KEPT = "gender_kept"
SELF_POSSESSIVE = "나의 "
ANATOMY_CONSISTENCY_RULES: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "assistant": (
        (re.compile(rf"\{{savior\}}의 ?(?=(?:{FEMALE_ANATOMY}))"), SELF_POSSESSIVE),
        (re.compile(rf"\{{spirit\}}의 ?(?=(?:{MALE_ANATOMY}))"), "{savior}의 "),
    ),
    "user": (
        (re.compile(rf"\{{spirit\}}의 ?(?=(?:{MALE_ANATOMY}))"), SELF_POSSESSIVE),
        (re.compile(rf"\{{savior\}}의 ?(?=(?:{FEMALE_ANATOMY}))"), "{spirit}의 "),
    ),
}
SCENE_SITUATION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (NIGHT_TOGETHER_SITUATION, ("첫날밤", "침실", "침대", "잠자리", "신혼", "호텔", "밤")),
    (BATH_TOGETHER_SITUATION, ("목욕", "욕실", "욕조", "온천", "샤워")),
    (TRIP_OPENING_SITUATION, ("데이트", "외출", "공원", "해변", "바다", "축제", "여행", "산책")),
    (LOBBY_TYPE_SITUATIONS["Greeting"], ("방문", "찾아", "첫 만남", "처음 만나", "재회")),
    (LOBBY_TYPE_SITUATIONS["Love1"], ("고백", "연인", "사랑")),
)
DEFAULT_TEMPLATE_SITUATION = LOBBY_TYPE_SITUATIONS["Normal"]
SPIRIT_KIND = "정령"
ARTIFICIAL_SPIRIT = "인공 정령"
DUPLICATE_SLOT_PATTERN = re.compile(r"(\{spirit\}|\{savior\})(?:\s*\1)+")
SLOT_LOOKAHEAD = r"\{(?:spirit|savior)\}"
NAME_FOLLOWERS = GRAMMATICAL_FOLLOWERS + "아"
SELF_REFERENCE_WORD = "나"
SELF_NAMING_PATTERNS: dict[str, re.Pattern[str]] = {
    "assistant": re.compile(r"(^|[.!?…♡♥~]\s*)\{spirit\}(?=\s*,)"),
    "user": re.compile(r"(^|[.!?…♡♥~]\s*)\{savior\}(?=\s*,)"),
}
LATIN_WORD_PATTERN = re.compile(r"[A-Za-z]{3,}")
SLOT_TOKEN_PATTERN = re.compile(SLOT_LOOKAHEAD)
GATE_MONSTER = "마물"
EDEN = "에덴"
ARK = "방주"
WORLD_TERM_CORRECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (SAVIOR_SLOT, ("인간 남자", "인간 남성", "인간", "지구인", "인류", "용사", "왕자")),
    (ARTIFICIAL_SPIRIT, ("인공지능",)),
    (
        SPIRIT_KIND,
        (
            "마족", "마왕", "엘프", "서큐버스", "흡혈귀", "뱀파이어", "요괴", "구미호",
            "이성인", "외계인", "우주인", "안드로이드", "사이보그", "인조인간", "로봇", "수인",
            "요정", "마녀", "악마", "여신", "인어", "드래곤", "공주", "왕녀",
        ),
    ),
    (GATE_MONSTER, ("몬스터", "슬라임", "오크", "고블린")),
    (EDEN, ("지구",)),
    (ARK, ("우주선",)),
)
CASTLE = "성"
WORLD_COMPOUND_CORRECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (CASTLE, ("마왕성",)),
    ("치료대", ("수술대",)),
    ("치료실", ("수술실",)),
    ("마도 실험대", ("실험대",)),
    ("마도 실험실", ("실험실",)),
    ("마도 실험체", ("실험체",)),
    ("마도 연구실", ("연구실",)),
    ("마도 연구소", ("연구소",)),
    ("마도 공학자", ("과학자",)),
    ("마도 공학적", ("과학적",)),
    ("집무실", ("부장실", "사장실", "회장실")),
    ("술", ("일본술",)),
    ("말", ("일본어",)),
    ("차", ("카페라테",)),
)
WORLD_CONCEPT_CORRECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("옷", ("기모노", "유카타")),
    ("하얀 옷", ("백의",)),
    ("집무실", ("오피스", "사무실", "회사")),
    ("보고서", ("서류", "자료", "기획서")),
    ("기사단", ("부서",)),
    ("집무", ("업무", "잔업", "야근")),
    ("마도 공학", ("과학",)),
    ("마도 연구", ("연구",)),
    ("마도 실험", ("실험",)),
    ("마나", ("유전자",)),
    ("치료", ("수술", "진찰", "진료")),
    ("치료실", ("병원", "진료소", "클리닉", "의무실")),
    ("솔레이 왕국", ("왕국",)),
    ("솔레이 왕궁", ("왕궁", "궁전")),
    ("아케나인", ("도시",)),
    ("엘나스", ("도쿄",)),
    ("아르카디아", ("일본",)),
    (EDEN, ("화성", "혹성")),
    (ARK, ("우주 스테이션", "스테이션")),
    ("성소", ("교회", "신전")),
    ("유적", ("던전", "미궁")),
    ("훈련장", ("체육관",)),
    ("객실", ("호텔", "여관", "모텔")),
    ("찻집", ("카페",)),
    ("가게", ("편의점",)),
    ("집", ("맨션", "아파트")),
    ("주점", ("유곽", "클럽")),
)
ROLE_TITLES: tuple[str, ...] = (
    "과장", "부장", "회장", "사장", "상사", "박사", "간호사", "점장", "함장",
    "교수", "주임", "팀장", "실장", "대리",
)
SLOT_SUFFIX_TITLES: tuple[str, ...] = (
    "훈", "공", "경", "공주", "왕녀", "왕자", "여왕", "마왕", "용사",
)
ORDINAL_STEMS: frozenset[str] = frozenset(
    {"첫", "둘", "셋", "넷", "다섯", "여섯", "일곱", "몇", "통"}
)
GENERAL_PEJORATIVE_PATTERN = re.compile(r"([가-힣]+)째(?=[!?.…,~♡♥\s]|$)")
TITLE_HONORIFICS: tuple[str, ...] = ("님", "씨")
BARE_NAME_SUFFIXES: frozenset[str] = frozenset({"씨", "군", "짱", "쨩", "님"})
BARE_NAME_MIN_SUFFIXED_SHARE = 0.5
TITLE_AFTER_SLOT_PATTERN = re.compile(
    r"(\{spirit\}|\{savior\})\s?(?:"
    + "|".join(sorted((*ROLE_TITLES, *SLOT_SUFFIX_TITLES), key=len, reverse=True))
    + rf")(?:님|씨)?(?=[^가-힣]|$|[{GRAMMATICAL_FOLLOWERS}])"
)
PEJORATIVE_ARTIFACT = "째"
PEJORATIVE_REPLACEMENT = "녀석"
DATASET_SPLITS: tuple[str, ...] = ("train", "validation", "test")
CANON_LANGUAGE = "ko"
NON_CASUAL_MAJORITY = 0.5


@dataclass(frozen=True)
class DialoguePattern:
    pattern_id: str
    source: dict[str, Any]
    situation: str
    level: SpeechLevel
    turns: tuple[SourceTurn, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.pattern_id,
            "source": self.source,
            "situation": self.situation,
            "topic": INTIMACY_SITUATION_TOPICS[self.situation],
            "level": self.level.value,
            "turns": [
                {
                    "role": turn.role,
                    "segments": [
                        {"kind": segment.kind.value, "text": segment.text}
                        for segment in turn.segments
                    ],
                }
                for turn in self.turns
            ],
        }


def template_situation(scene: str) -> str:
    for situation, keywords in SCENE_SITUATION_RULES:
        if any(keyword in scene for keyword in keywords):
            return situation
    return DEFAULT_TEMPLATE_SITUATION


def word_stem(word: str) -> str:
    for particle in PARTICLE_SUFFIXES:
        if word.endswith(particle) and len(word) - len(particle) >= MIN_WORD_LENGTH:
            return word[: -len(particle)]
    return word


def setting_vocabulary(setting: str) -> frozenset[str]:
    content = [
        line
        for line in setting.split("\n")
        if not any(header in line for header in SETTING_HEADER_WORDS)
    ]
    words = {word_stem(word) for line in content for word in WORD_PATTERN.findall(line)}
    return frozenset(words - GENERIC_SETTING_WORDS)


@dataclass(frozen=True)
class SlotPatterns:
    spirit: re.Pattern[str] | None
    savior: re.Pattern[str] | None


def slot_patterns(names: SourceNames, foreign_addresses: Iterable[str] = ()) -> SlotPatterns:
    return SlotPatterns(
        spirit=compile_terms(names.character, NAME_FOLLOWERS),
        savior=compile_terms((*names.user, *foreign_addresses), NAME_FOLLOWERS),
    )


def slot_turn(turn: SourceTurn, patterns: SlotPatterns) -> SourceTurn:
    return SourceTurn(
        turn.role,
        tuple(
            TurnSegment(
                segment.kind,
                substitute_names(
                    substitute_names(segment.text, patterns.spirit, SPIRIT_SLOT),
                    patterns.savior,
                    SAVIOR_SLOT,
                ),
            )
            for segment in turn.segments
        ),
    )


def slot_conversation(
    conversation: SourceConversation, foreign_addresses: Iterable[str] = ()
) -> SourceConversation:
    patterns = slot_patterns(conversation.names, foreign_addresses)
    return replace(
        conversation, turns=tuple(slot_turn(turn, patterns) for turn in conversation.turns)
    )


WORLD_TERM_PATTERNS: tuple[tuple[str, re.Pattern[str] | None], ...] = (
    *((target, compile_terms(terms)) for target, terms in WORLD_COMPOUND_CORRECTIONS),
    *((target, compile_terms(terms)) for target, terms in WORLD_TERM_CORRECTIONS),
    *(
        (target, compile_terms(terms, CONCEPT_FOLLOWERS))
        for target, terms in WORLD_CONCEPT_CORRECTIONS
    ),
)
ROLE_ADDRESS_PATTERN = compile_terms(ROLE_ADDRESSES, NAME_FOLLOWERS)
ROLE_TITLE_PATTERN = compile_terms(
    title + honorific for title in ROLE_TITLES for honorific in ("", *TITLE_HONORIFICS)
)
PEJORATIVE_ARTIFACT_PATTERN = re.compile(
    "(?:(?<![가-힣])|(?<=[들희]))("
    + "|".join(
        re.escape(term)
        for _, terms in WORLD_TERM_CORRECTIONS
        for term in sorted(terms, key=len, reverse=True)
    )
    + f"){PEJORATIVE_ARTIFACT}(?![가-힣])"
)


def _pejorative_replacement(match: re.Match[str]) -> str:
    stem = match.group(1)
    if stem in ORDINAL_STEMS:
        return match.group(0)
    return f"{stem} {PEJORATIVE_REPLACEMENT}"


def correct_world_terms(conversation: SourceConversation) -> SourceConversation:
    turns = []
    for turn in conversation.turns:
        segments = []
        for segment in turn.segments:
            text = PEJORATIVE_ARTIFACT_PATTERN.sub(rf"\1 {PEJORATIVE_REPLACEMENT}", segment.text)
            text = GENERAL_PEJORATIVE_PATTERN.sub(_pejorative_replacement, text)
            for target, pattern in WORLD_TERM_PATTERNS:
                text = substitute_names(text, pattern, target)
            segments.append(TurnSegment(segment.kind, text))
        turns.append(SourceTurn(turn.role, tuple(segments)))
    return replace(conversation, turns=tuple(turns))


def address_forms(text: str) -> list[tuple[str, str]]:
    return ADDRESS_FORM_PATTERN.findall(text)


def canon_dialogue_lines() -> Iterator[str]:
    for slug in roster_slugs():
        for split in DATASET_SPLITS:
            path = sft_split_path(slug, split)
            if not path.is_file():
                raise EvaiError(f"Spirit dataset split not built: {path}")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if (
                        record.get("language") == CANON_LANGUAGE
                        and record.get("source_class") == SourceClass.CANON_DIALOGUE.value
                    ):
                        yield record["completion"][0]["content"]


def canon_address_forms() -> frozenset[str]:
    return frozenset(
        base + suffix for line in canon_dialogue_lines() for base, suffix in address_forms(line)
    )


def canon_word_stems() -> frozenset[str]:
    return frozenset(
        word_stem(word) for line in canon_dialogue_lines() for word in WORD_PATTERN.findall(line)
    )


def foreign_address_forms(
    slotted_conversations: Iterable[SourceConversation], canon_forms: frozenset[str]
) -> tuple[str, ...]:
    found: set[tuple[str, str]] = set()
    vocative_uses: Counter[str] = Counter()
    setting_words: set[str] = set()
    for conversation in slotted_conversations:
        setting_words.update(WORD_PATTERN.findall(conversation.setting))
        for turn in conversation.turns:
            for segment in turn.segments:
                for match in ADDRESS_FORM_PATTERN.finditer(segment.text):
                    found.add((match.group(1), match.group(2)))
                    if VOCATIVE_FOLLOW_PATTERN.match(segment.text, match.end()):
                        vocative_uses[match.group(0)] += 1
    named_bases = {
        base
        for base, suffix in found
        if suffix in NAME_EVIDENCE_SUFFIXES or base in setting_words
    }
    forms = {
        base + suffix
        for base, suffix in found
        if (base + suffix not in canon_forms or vocative_uses[base + suffix] >= MIN_VOCATIVE_USES)
        and base in named_bases
    }
    return tuple(sorted(forms, key=lambda form: (-len(form), form)))


def foreign_bare_names(
    slotted_conversations: Iterable[SourceConversation], canon_forms: frozenset[str]
) -> tuple[str, ...]:
    conversations = list(slotted_conversations)
    suffixed: Counter[str] = Counter()
    for conversation in conversations:
        for turn in conversation.turns:
            for segment in turn.segments:
                for match in ADDRESS_FORM_PATTERN.finditer(segment.text):
                    if (
                        match.group(2) in BARE_NAME_SUFFIXES
                        and len(match.group(1)) > 1
                        and match.group(0) not in canon_forms
                        and match.group(1) not in ROLE_TITLES
                    ):
                        suffixed[match.group(1)] += 1
    candidates = compile_terms(suffixed)
    bare: Counter[str] = Counter()
    if candidates is not None:
        for conversation in conversations:
            for turn in conversation.turns:
                for segment in turn.segments:
                    for match in candidates.finditer(segment.text):
                        bare[match.group(0)[: len(match.group(0)) - len(match.group(1) or "")]] += 1
    names = {
        base
        for base, count in suffixed.items()
        if count >= BARE_NAME_MIN_SUFFIXED_SHARE * bare[base]
    }
    return tuple(sorted(names, key=lambda name: (-len(name), name)))


@dataclass(frozen=True)
class RoleCorrection:
    foreign: re.Pattern[str] | None
    bare_names: re.Pattern[str] | None
    surnames: re.Pattern[str] | None


def role_correction(
    foreign_addresses: Iterable[str] = (), bare_names: Iterable[str] = ()
) -> RoleCorrection:
    names = sorted(set(bare_names), key=lambda name: (-len(name), name))
    return RoleCorrection(
        foreign=compile_terms(foreign_addresses, NAME_FOLLOWERS),
        bare_names=compile_terms(names),
        surnames=(
            re.compile(
                f"(?<![가-힣])(?:{'|'.join(map(re.escape, names))})\\s+(?={SLOT_LOOKAHEAD})"
            )
            if names
            else None
        ),
    )


def correct_roles(
    conversation: SourceConversation, correction: RoleCorrection
) -> SourceConversation:
    turns = []
    for turn in conversation.turns:
        counterpart = SPIRIT_SLOT if turn.role == "user" else SAVIOR_SLOT
        segments = []
        for segment in turn.segments:
            text = substitute_names(segment.text, ROLE_ADDRESS_PATTERN, counterpart)
            text = substitute_names(text, correction.foreign, counterpart)
            if correction.surnames is not None:
                text = correction.surnames.sub("", text)
            text = substitute_names(text, correction.bare_names, counterpart)
            text = TITLE_AFTER_SLOT_PATTERN.sub(r"\1", text)
            text = substitute_names(text, ROLE_TITLE_PATTERN, counterpart)
            text = DUPLICATE_SLOT_PATTERN.sub(r"\1", text)
            text = SELF_NAMING_PATTERNS[turn.role].sub(rf"\1{SELF_REFERENCE_WORD}", text)
            segments.append(TurnSegment(segment.kind, text))
        turns.append(SourceTurn(turn.role, tuple(segments)))
    return replace(conversation, turns=tuple(turns))


def is_common_turn(turn: SourceTurn, vocabulary: frozenset[str]) -> bool:
    contradiction = GENDER_CONTRADICTIONS[turn.role]
    for segment in turn.segments:
        if (
            KANA_PATTERN.search(segment.text)
            or LATIN_WORD_PATTERN.search(SLOT_TOKEN_PATTERN.sub("", segment.text))
            or contradiction.search(segment.text)
            or mentions_real_world(segment.text)
            or any(marker in segment.text for marker in SPIRIT_ONLY_MARKERS)
        ):
            return False
        if {word_stem(word) for word in WORD_PATTERN.findall(segment.text)} & vocabulary:
            return False
    return bool(turn.segments)


def speech_level(turns: Iterable[SourceTurn]) -> SpeechLevel:
    levels = Counter(
        classify_level(sentence)
        for turn in turns
        if turn.role == "assistant"
        for segment in turn.segments
        if segment.kind is SegmentKind.SPEECH
        for sentence in sentence_endings(segment.text)
    )
    total = sum(levels.values())
    formal = [(count, level) for level, count in levels.items() if level is not SpeechLevel.CASUAL]
    if total and formal and sum(count for count, _ in formal) / total >= NON_CASUAL_MAJORITY:
        return max(formal)[1]
    return SpeechLevel.CASUAL


PatternRun = tuple[int, list[SourceTurn]]


def _pattern_runs(turns: list[SourceTurn], common: list[bool]) -> list[PatternRun]:
    runs: list[PatternRun] = []
    start: int | None = None
    for index, (turn, is_common) in enumerate([*zip(turns, common, strict=True), (None, False)]):
        if is_common and start is None and turn is not None and turn.role == "user":
            start = index
        elif not is_common and start is not None:
            run = turns[start:index]
            while run and run[-1].role != "assistant":
                run.pop()
            if any(item.role == "assistant" for item in run):
                runs.append((start, run))
            start = None
    return runs


def scenario_decision(conversation: SourceConversation) -> TurnClassification:
    setting = scenario_classification(conversation.setting)
    if setting is not TurnClassification.ACCEPTED:
        return setting
    return dialogue_classification(
        segment.text for turn in conversation.turns for segment in turn.segments
    )


def is_accepted_scenario(conversation: SourceConversation) -> bool:
    return scenario_decision(conversation) is TurnClassification.ACCEPTED


def template_conversation(conversation: SourceConversation) -> SourceConversation | None:
    return conversation if is_accepted_scenario(conversation) else None


def _anatomy_references(text: str, slot: str) -> tuple[int, int]:
    return (
        len(re.findall(f"{re.escape(slot)}{SLOT_ANATOMY_JOINER}(?:{MALE_ANATOMY})", text)),
        len(re.findall(f"{re.escape(slot)}{SLOT_ANATOMY_JOINER}(?:{FEMALE_ANATOMY})", text)),
    )


def dialogue_genders(slotted: SourceConversation) -> dict[str, CharacterGender]:
    votes = {role: Counter[str]() for role in ROLE_SLOTS}
    for turn in slotted.turns:
        text = " ".join(segment.text for segment in turn.segments)
        votes[turn.role][CharacterGender.MALE] += len(SELF_MALE_ANATOMY_PATTERN.findall(text))
        votes[turn.role][CharacterGender.FEMALE] += len(SELF_FEMALE_ANATOMY_PATTERN.findall(text))
        partner = COUNTERPART_ROLE[turn.role]
        male, female = _anatomy_references(text, ROLE_SLOTS[partner])
        votes[partner][CharacterGender.MALE] += male
        votes[partner][CharacterGender.FEMALE] += female
    return {
        role: (
            CharacterGender.UNKNOWN
            if counter[CharacterGender.MALE] == counter[CharacterGender.FEMALE]
            else CharacterGender.MALE
            if counter[CharacterGender.MALE] > counter[CharacterGender.FEMALE]
            else CharacterGender.FEMALE
        )
        for role, counter in votes.items()
    }


def resolved_gender(setting: CharacterGender, dialogue: CharacterGender) -> CharacterGender:
    if setting is CharacterGender.UNKNOWN:
        return dialogue
    if dialogue is CharacterGender.UNKNOWN or dialogue is setting:
        return setting
    return CharacterGender.UNKNOWN


def _swap_slots(text: str) -> str:
    return text.replace(SPIRIT_SLOT, SWAP_MARKER).replace(SAVIOR_SLOT, SPIRIT_SLOT).replace(
        SWAP_MARKER, SAVIOR_SLOT
    )


def align_genders(slotted: SourceConversation) -> tuple[SourceConversation | None, str]:
    from_setting = setting_genders(slotted.setting)
    from_dialogue = dialogue_genders(slotted)
    spirit, partner = (
        resolved_gender(from_setting[role], from_dialogue[role]) for role in ("assistant", "user")
    )
    if spirit is partner and spirit is not CharacterGender.UNKNOWN:
        return None, GENDER_SAME
    if spirit is CharacterGender.MALE or partner is CharacterGender.FEMALE:
        return (
            replace(
                slotted,
                turns=tuple(
                    SourceTurn(
                        COUNTERPART_ROLE[turn.role],
                        tuple(
                            TurnSegment(segment.kind, _swap_slots(segment.text))
                            for segment in turn.segments
                        ),
                    )
                    for turn in slotted.turns
                ),
            ),
            GENDER_SWAPPED,
        )
    return slotted, GENDER_KEPT


def correct_anatomy(conversation: SourceConversation) -> SourceConversation:
    turns = []
    for turn in conversation.turns:
        rules = ANATOMY_CONSISTENCY_RULES[turn.role]
        segments = []
        for segment in turn.segments:
            text = segment.text
            for pattern, replacement in rules:
                text = pattern.sub(replacement, text)
            segments.append(TurnSegment(segment.kind, text))
        turns.append(SourceTurn(turn.role, tuple(segments)))
    return replace(conversation, turns=tuple(turns))


def pattern_provenance(conversation: SourceConversation, start: int) -> dict[str, Any]:
    provenance = conversation.provenance()
    provenance.pop("topic")
    return {**provenance, "turn": start}


def slotted_patterns(slotted: SourceConversation) -> list[DialoguePattern]:
    vocabulary = setting_vocabulary(slotted.setting)
    turns = list(slotted.turns)
    common = [is_common_turn(turn, vocabulary) for turn in turns]
    return [
        DialoguePattern(
            pattern_id=f"{slotted.conversation_id}:{start}",
            source=pattern_provenance(slotted, start),
            situation=template_situation(slotted.topic),
            level=speech_level(run),
            turns=tuple(run),
        )
        for start, run in _pattern_runs(turns, common)
    ]


def corrected_conversation(
    slotted: SourceConversation, correction: RoleCorrection
) -> SourceConversation:
    return correct_anatomy(correct_world_terms(correct_roles(slotted, correction)))


def conversation_patterns(
    conversation: SourceConversation,
    foreign_addresses: Iterable[str] = (),
    bare_names: Iterable[str] = (),
) -> list[DialoguePattern]:
    accepted = template_conversation(conversation)
    if accepted is None:
        return []
    aligned, _ = align_genders(slot_conversation(accepted))
    if aligned is None:
        return []
    return slotted_patterns(
        corrected_conversation(aligned, role_correction(foreign_addresses, bare_names))
    )


def build_intimacy_patterns(path: Path = INTIMACY_PATTERNS_FILE) -> Counter[str]:
    stats: Counter[str] = Counter()
    conversations = load_korean_adult_roleplay_conversations()
    stats["source_conversations"] = len(conversations)
    slotted: list[SourceConversation] = []
    for conversation in conversations:
        policy = scenario_decision(conversation)
        if policy is not TurnClassification.ACCEPTED:
            stats[f"policy:{policy.value}"] += 1
            continue
        aligned, decision = align_genders(slot_conversation(conversation))
        stats[decision] += 1
        if aligned is not None:
            slotted.append(aligned)
    canon_forms = canon_address_forms()
    foreign_addresses = foreign_address_forms(slotted, canon_forms)
    stats["foreign_address_forms"] = len(foreign_addresses)
    bare_names = foreign_bare_names(slotted, canon_forms)
    stats["foreign_bare_names"] = len(bare_names)
    correction = role_correction(foreign_addresses, bare_names)
    staging = path.with_name(f"{path.name}{STAGING_SUFFIX}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("w", encoding="utf-8", newline="\n") as handle:
        for conversation in slotted:
            patterns = slotted_patterns(corrected_conversation(conversation, correction))
            stats["pattern_conversations"] += bool(patterns)
            for pattern in patterns:
                stats["patterns"] += 1
                stats[f"level:{pattern.level.value}"] += 1
                stats["spirit_turns"] += sum(turn.role == "assistant" for turn in pattern.turns)
                handle.write(json.dumps(pattern.to_dict(), ensure_ascii=False) + "\n")
    os.replace(staging, path)
    return stats


def load_intimacy_pattern_conversations() -> list[SourceConversation]:
    conversations: list[SourceConversation] = []
    with INTIMACY_PATTERN_SOURCE.require_path().open(encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            pattern = json.loads(line)
            conversations.append(
                SourceConversation(
                    dataset=INTIMACY_PATTERN_SOURCE,
                    row=row,
                    topic=pattern["situation"],
                    setting="",
                    names=PATTERN_NAMES,
                    turns=tuple(
                        SourceTurn(
                            turn["role"],
                            tuple(
                                TurnSegment(SegmentKind(segment["kind"]), segment["text"])
                                for segment in turn["segments"]
                            ),
                        )
                        for turn in pattern["turns"]
                    ),
                )
            )
    return conversations
