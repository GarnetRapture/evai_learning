"""Negative identity signals for rank selection; never a semantic quality approval."""

import re

IDENTITY_QUESTIONS = {
    "kr": "네 이름과 네가 어떤 존재인지 말해 줘. 너와 대화하는 나는 누구야?",
    "en": "What is your name and what kind of being are you? Who am I to you?",
    "zh_tw": "你叫什麼名字，是什麼樣的存在？與你交談的我又是誰？",
}
ROLE_REVERSALS = (
    re.compile(
        r"(?:나는|난|저는|전|내가|제가)\s*구원자(?:님)?"
        r"\s*(?:입니다|이다|이야|예요|이에요|야|다|이고|로서|[.!?]|$)"
    ),
    re.compile(r"\bI(?:\s+am|'m|’m)\s+(?:an?\s+|the\s+)?Savior\b(?!['’])", re.IGNORECASE),
    re.compile(r"我(?:就是|是|身為)\s*(?:一[名位個])?救贖者(?!的)"),
)

# Fit signals for authored facts already in training, explicitly not held-out quality scores.
KNOWN_FACT_TERMS: dict[str, dict[str, tuple[tuple[str, ...], ...]]] = {
    "addition": {
        "ko": (("다섯", "5"),),
        "en": (("five", "5"),),
        "zh_tw": (("五", "5"),),
    },
    "blue_sky": {
        "ko": (("공기", "분자"), ("파란", "파랑", "푸른"), ("흩", "산란")),
        "en": (("air", "molecule"), ("blue",), ("scatter",)),
        "zh_tw": (("空氣", "分子"), ("藍",), ("散",)),
    },
}


def known_fact_failure(response: str, topic: str, language: str) -> bool:
    folded = response.casefold()
    return any(
        not any(term in folded for term in alternatives)
        for alternatives in KNOWN_FACT_TERMS[topic][language]
    )


def identity_failure(response: str, own_name: str) -> bool:
    return own_name.casefold() not in response.casefold() or any(
        pattern.search(response) for pattern in ROLE_REVERSALS
    )
