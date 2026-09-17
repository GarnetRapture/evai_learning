import re

from sft_dataset.normalize import normalize_text

CONTROL_TAG_NAMES: tuple[str, ...] = (
    "font",
    "sound",
    "effect",
    "camera",
    "transition",
    "motionway",
    "display",
)
CONTROL_TAG_PATTERN = re.compile(r"<(?:" + "|".join(CONTROL_TAG_NAMES) + r")(?::[^<>]*)?>")
TIMING_CODE_PATTERN = re.compile(r"(?:[ \t]*@-?\d+(?:\.\d+)?)+[ \t]*$", re.MULTILINE)
WHITESPACE_RUN_PATTERN = re.compile(r"\s+")


def clean_game_text(text: str | None) -> str:
    if text is None:
        return ""
    without_tags = CONTROL_TAG_PATTERN.sub("", normalize_text(text))
    without_codes = TIMING_CODE_PATTERN.sub("", without_tags)
    return WHITESPACE_RUN_PATTERN.sub(" ", without_codes).strip()


def is_quoted_utterance(text: str) -> bool:
    return len(text) >= 2 and text[0] == '"' and text[-1] == '"'


def unquote_utterance(text: str) -> str:
    return text[1:-1].strip() if is_quoted_utterance(text) else text
