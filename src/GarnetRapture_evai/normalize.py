"""Conservative text normalization for persona utterances and dialogue entries."""

import unicodedata


def is_empty_text(text: str | None) -> bool:
    """Check if text is None, empty, or consists only of whitespace."""
    if text is None:
        return True
    return len(text.strip()) == 0


def normalize_text(text: str | None) -> str:
    """Conservatively normalize text while strictly preserving character speech style.

    Allowed transformations:
    - Unicode NFC normalization
    - Line ending normalization (\\r\\n and \\r to \\n)
    - Leading/trailing whitespace stripping

    Strictly forbidden transformations:
    - Spelling / grammar alterations
    - Honorific or politeness conversion
    - Punctuation alteration or stripping (e.g. '...', '…', '!!', '~~')
    - Paraphrasing or tone normalization

    Returns empty string if input is None.
    """
    if text is None:
        return ""

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Line ending normalization
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Strip leading and trailing whitespace
    return normalized.strip()
