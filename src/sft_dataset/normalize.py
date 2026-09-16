import unicodedata


def is_empty_text(text: str | None) -> bool:
    if text is None:
        return True
    return len(text.strip()) == 0


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()
