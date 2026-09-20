import re
from collections.abc import Iterable

from sft_dataset.dialogue import (
    MINOR_OR_AGE_AMBIGUOUS_PATTERNS,
    NONCONSENSUAL_OR_EXPLOITATIVE_PATTERNS,
    TurnClassification,
)

ADULT_AGE = 18
AGE_PATTERN = re.compile(r"(\d+)\s*(?:살|세)(?![가-힣])")
SCHOOL_SCENARIO_PATTERN = re.compile(r"(?<!대)학생|수업|교실|학교|학원|방과 ?후|체육 창고|담임")
MINOR_SCENARIO_MARKERS: tuple[str, ...] = (
    *MINOR_OR_AGE_AMBIGUOUS_PATTERNS,
    "소년",
    "소녀",
    "어린",
    "로리",
    "쇼타",
    "아동",
    "유치원",
    "초등",
    "중학",
    "고등학",
    "교복",
    "여학생",
    "남학생",
)
NONCONSENSUAL_SCENARIO_MARKERS: tuple[str, ...] = (
    *NONCONSENSUAL_OR_EXPLOITATIVE_PATTERNS,
    "강제",
    "억지로",
    "습격",
    "납치",
    "감금",
    "최면",
    "세뇌",
    "협박",
    "능욕",
    "조교",
    "노예",
    "수면",
    "치한",
    "윤간",
    "공포",
)
REVIEW_SCENARIO_MARKERS: tuple[str, ...] = (
    "근친",
    "친동생",
    "친누나",
    "친오빠",
    "친언니",
    "여동생",
    "남동생",
    "의붓",
    "계모",
    "계부",
    "촉수",
    "짐승",
    "수간",
)


DIALOGUE_MINOR_PATTERN = re.compile(
    r"(?<![가-힣])(?:소년|소녀|어린애|로리|쇼타|아동|초등학생|중학생|고등학생|여학생|남학생|교복|교실|담임"
    r"|클래스메이트|반 친구|위원장|학생회)"
)
DIALOGUE_NONCONSENSUAL_PATTERN = re.compile(
    r"(?<![가-힣])(?:강간|성폭행|성폭력|윤간|협박|의식을 잃은|약을 먹여|수면제)"
)
DIALOGUE_REVIEW_PATTERN = re.compile(r"(?<![가-힣])(?:촉수|수간|근친)")


def dialogue_classification(texts: Iterable[str]) -> TurnClassification:
    joined = "\n".join(texts)
    if DIALOGUE_MINOR_PATTERN.search(joined) or SCHOOL_SCENARIO_PATTERN.search(joined):
        return TurnClassification.EXCLUDED_MINOR_OR_AGE_AMBIGUOUS
    if DIALOGUE_NONCONSENSUAL_PATTERN.search(joined):
        return TurnClassification.EXCLUDED_NONCONSENSUAL_OR_EXPLOITATIVE
    if DIALOGUE_REVIEW_PATTERN.search(joined):
        return TurnClassification.NEEDS_REVIEW
    return TurnClassification.ACCEPTED


def scenario_classification(setting: str) -> TurnClassification:
    lowered = setting.lower()
    if (
        any(int(age) < ADULT_AGE for age in AGE_PATTERN.findall(setting))
        or any(marker in lowered for marker in MINOR_SCENARIO_MARKERS)
        or SCHOOL_SCENARIO_PATTERN.search(setting)
    ):
        return TurnClassification.EXCLUDED_MINOR_OR_AGE_AMBIGUOUS
    if any(marker in lowered for marker in NONCONSENSUAL_SCENARIO_MARKERS):
        return TurnClassification.EXCLUDED_NONCONSENSUAL_OR_EXPLOITATIVE
    if any(marker in lowered for marker in REVIEW_SCENARIO_MARKERS):
        return TurnClassification.NEEDS_REVIEW
    return TurnClassification.ACCEPTED
