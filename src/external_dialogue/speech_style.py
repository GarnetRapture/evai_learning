import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class SpeechLevel(StrEnum):
    CASUAL = "casual"
    POLITE = "polite"
    FORMAL = "formal"
    ARCHAIC = "archaic"
    MILITARY = "military"


SENTENCE_PATTERN = re.compile(r"[^.!?…~♪♡♥★]+[.!?…~♪♡♥★]*")
TRAILING_MARKS = ".!?…~♪♡♥★ )\"'"
MILITARY_ENDINGS = "슴다|슴까|임다|임까|심다|심까|십쇼|겠슴|했슴|았슴|었슴"
FORMAL_ENDINGS = "니다|니까|십시오|옵니다|사옵|시지요"
ARCHAIC_ENDINGS = "하오|이오|구려|시오|겠소|했소|었소|았소|이라오|라오|느냐|거라|노라"
LEVEL_PATTERNS: tuple[tuple[SpeechLevel, re.Pattern[str]], ...] = (
    (SpeechLevel.MILITARY, re.compile(rf"({MILITARY_ENDINGS})$")),
    (SpeechLevel.FORMAL, re.compile(rf"({FORMAL_ENDINGS})$")),
    (SpeechLevel.ARCHAIC, re.compile(rf"({ARCHAIC_ENDINGS})$")),
    (SpeechLevel.POLITE, re.compile(r"(요|죠)$")),
)
ADDRESS_TERMS: tuple[str, ...] = (
    "구원자님",
    "구원자",
    "프로듀서님",
    "프로듀서",
    "왕자님",
    "달링",
    "마스터",
    "그대",
    "당신",
    "인간",
)
FIRST_PERSON_TERMS: tuple[str, ...] = ("소생", "소녀", "짐", "저", "나")
FIRST_PERSON_PATTERNS: dict[str, re.Pattern[str]] = {
    term: re.compile(rf"(?<![가-힣]){term}(?:는|도|를|랑|의|가|한테|에게|만|은|이)?(?![가-힣])")
    for term in FIRST_PERSON_TERMS
}
SECOND_PERSON_PATTERN = re.compile(
    r"(?<![가-힣])(너|네가|넌|널|너도|너랑|너를|너한테|너는)(?![가-힣])"
)
MIN_LEVEL_SHARE = 0.2
MIN_HONORIFIC_SHARE = 0.3
MIN_ADDRESSED_SENTENCES = 20


@dataclass(frozen=True)
class SpeechProfile:
    level: SpeechLevel
    address: str
    first_person: str
    uses_second_person: bool
    level_counts: dict[str, int]


def sentence_endings(text: str) -> list[str]:
    return [
        stripped
        for sentence in SENTENCE_PATTERN.findall(text)
        if (stripped := sentence.strip().rstrip(TRAILING_MARKS))
    ]


def classify_level(sentence: str) -> SpeechLevel:
    for level, pattern in LEVEL_PATTERNS:
        if pattern.search(sentence):
            return level
    return SpeechLevel.CASUAL


def extract_speech_profile(lines: Iterable[str]) -> SpeechProfile:
    all_levels: Counter[SpeechLevel] = Counter()
    addressed_levels: Counter[SpeechLevel] = Counter()
    addresses: Counter[str] = Counter()
    first_persons: Counter[str] = Counter()
    second_person = 0
    for line in lines:
        addressed = any(term in line for term in ADDRESS_TERMS)
        for sentence in sentence_endings(line):
            level = classify_level(sentence)
            all_levels[level] += 1
            if addressed:
                addressed_levels[level] += 1
        remaining = line
        for term in ADDRESS_TERMS:
            count = remaining.count(term)
            if count:
                addresses[term] += count
                remaining = remaining.replace(term, " ")
        for term, pattern in FIRST_PERSON_PATTERNS.items():
            first_persons[term] += len(pattern.findall(line))
        second_person += len(SECOND_PERSON_PATTERN.findall(line))
    levels = (
        addressed_levels
        if sum(addressed_levels.values()) >= MIN_ADDRESSED_SENTENCES
        else all_levels
    )
    total = sum(levels.values())
    formal_levels = [
        level
        for level in (SpeechLevel.MILITARY, SpeechLevel.ARCHAIC, SpeechLevel.FORMAL)
        if total and levels[level] / total >= MIN_LEVEL_SHARE
    ]
    honorific = total - levels[SpeechLevel.CASUAL] - levels[SpeechLevel.ARCHAIC]
    if formal_levels:
        level = max(formal_levels, key=lambda item: levels[item])
    elif total and honorific / total >= MIN_HONORIFIC_SHARE:
        level = SpeechLevel.POLITE
    else:
        level = SpeechLevel.CASUAL
    address = addresses.most_common(1)[0][0] if addresses else "구원자"
    honorific_first_person = next(
        (
            term
            for term, count in first_persons.most_common()
            if count > 0 and term != "나"
        ),
        "저",
    )
    first_person = (
        next((term for term, count in first_persons.most_common() if count > 0), "나")
        if level in (SpeechLevel.CASUAL, SpeechLevel.ARCHAIC)
        else honorific_first_person
    )
    return SpeechProfile(
        level=level,
        address=address,
        first_person=first_person,
        uses_second_person=second_person > addresses[address] if addresses else second_person > 0,
        level_counts={item.value: levels[item] for item in SpeechLevel},
    )
