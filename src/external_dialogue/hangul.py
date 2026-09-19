HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3
FINAL_COUNT = 28
FINAL_NONE = 0
FINAL_RIEUL = 8
FINAL_MIEUM = 16
FINAL_BIEUP = 17


def is_hangul_syllable(character: str) -> bool:
    return len(character) == 1 and HANGUL_BASE <= ord(character) <= HANGUL_LAST


def final_consonant(character: str) -> int:
    if not is_hangul_syllable(character):
        return FINAL_NONE
    return (ord(character) - HANGUL_BASE) % FINAL_COUNT


def has_final(word: str) -> bool:
    return bool(word) and final_consonant(word[-1]) != FINAL_NONE


def with_final(character: str, final: int) -> str:
    offset = ord(character) - HANGUL_BASE
    return chr(HANGUL_BASE + offset - offset % FINAL_COUNT + final)


def topic_particle(word: str) -> str:
    return "은" if has_final(word) else "는"


def subject_particle(word: str) -> str:
    return "이" if has_final(word) else "가"


def object_particle(word: str) -> str:
    return "을" if has_final(word) else "를"


def with_particle(word: str) -> str:
    return "이랑" if has_final(word) else "랑"


def vocative_particle(word: str) -> str:
    return "아" if has_final(word) else "야"


def copula_polite(word: str) -> str:
    return "이에요" if has_final(word) else "예요"
