import re

from external_dialogue.hangul import (
    FINAL_BIEUP,
    FINAL_MIEUM,
    FINAL_NONE,
    FINAL_RIEUL,
    final_consonant,
    has_final,
    object_particle,
    subject_particle,
    topic_particle,
    with_final,
    with_particle,
)
from external_dialogue.speech_style import SpeechLevel, SpeechProfile

STAGE_DIRECTION_PATTERN = re.compile(r"\*[^*]*\*")
EMOTICON_PATTERN = re.compile(r"(ㅋ|ㅎ|ㅠ|ㅜ){2,}")
SPACE_PATTERN = re.compile(r"\s+")
SENTENCE_PATTERN = re.compile(r"([^.!?~…]+)([.!?~…]*)")
BOUNDARY_BEFORE = r"(?<![가-힣A-Za-z0-9])"
BOUNDARY_AFTER = r"(?![가-힣A-Za-z0-9])"
INTERJECTIONS: frozenset[str] = frozenset(
    {
        "진짜", "정말", "대박", "우와", "와", "헐", "음", "흠", "오", "아", "어머",
        "히히", "헤헤", "하하", "후후", "에이", "앗", "엥", "흐음", "음음", "그치",
    }
)
POLITE_EXACT: dict[str, str] = {
    "안녕": "안녕하세요",
    "응": "네",
    "어": "네",
    "아니": "아니요",
    "그래": "그래요",
    "맞아": "맞아요",
    "좋아": "좋아요",
    "고마워": "고마워요",
    "미안": "미안해요",
    "알겠어": "알겠어요",
    "그렇지": "그렇죠",
    "그치": "그렇죠",
    "미안해": "미안해요",
    "괜찮아": "괜찮아요",
    "당연하지": "당연하죠",
    "물론이지": "물론이죠",
    "알았어": "알았어요",
}
CLAUSE_SEPARATOR = ", "
CASUAL_VOCATIVES: frozenset[str] = frozenset({"야", "얘", "야야"})
REDUPLICATION_PATTERN = re.compile(r"^(.+)\1$")
SUGGESTION_POLITE: dict[str, str] = {
    "보자": "봐요",
    "가자": "가요",
    "하자": "해요",
    "먹자": "먹어요",
    "놀자": "놀아요",
    "쉬자": "쉬어요",
    "찾자": "찾아요",
    "만나자": "만나요",
    "걷자": "걸어요",
    "자자": "자요",
    "나가자": "나가요",
    "시작하자": "시작해요",
}
PLAIN_DA_POLITE: dict[str, str] = {
    "겠다": "겠어요",
    "좋다": "좋아요",
    "싶다": "싶어요",
    "있다": "있어요",
    "없다": "없어요",
    "같다": "같아요",
    "맞다": "맞아요",
    "된다": "돼요",
    "한다": "해요",
    "많다": "많아요",
    "크다": "커요",
    "예쁘다": "예뻐요",
    "귀엽다": "귀여워요",
    "재밌다": "재밌어요",
    "멋지다": "멋져요",
    "부럽다": "부러워요",
    "기쁘다": "기뻐요",
    "고맙다": "고마워요",
    "반갑다": "반가워요",
}
POLITE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("구나", "군요"),
    ("더라", "더라고요"),
    ("라구", "라고요"),
    ("다구", "다고요"),
    ("거든", "거든요"),
    ("잖아", "잖아요"),
    ("는데", "는데요"),
    ("은데", "은데요"),
    ("던데", "던데요"),
    ("니까", "니까요"),
    ("라고", "라고요"),
    ("다고", "다고요"),
    ("을걸", "을걸요"),
    ("할걸", "할걸요"),
    ("군", "군요"),
    ("네", "네요"),
    ("지", "죠"),
    ("게", "게요"),
    ("까", "까요"),
    ("대", "대요"),
    ("래", "래요"),
)
VERB_FINAL_SYLLABLES = frozenset("어아여와워해돼봐줘가서내개새채배케세제려져쳐뻐퍼껴겨텨펴혀")
FORMAL_CONTRACTED: dict[str, str] = {
    "해요": "합니다",
    "돼요": "됩니다",
    "봐요": "봅니다",
    "줘요": "줍니다",
    "가요": "갑니다",
    "와요": "옵니다",
    "몰라요": "모릅니다",
    "그래요": "그렇습니다",
    "어때요": "어떻습니까",
    "기뻐요": "기쁩니다",
    "예뻐요": "예쁩니다",
    "커요": "큽니다",
    "멋져요": "멋집니다",
    "알아요": "압니다",
    "살아요": "삽니다",
    "놀아요": "놉니다",
    "만나요": "만납니다",
    "쉬어요": "쉽니다",
    "자요": "잡니다",
    "나가요": "나갑니다",
}
FORMAL_KEEP_SUFFIXES: tuple[str, ...] = (
    "네요", "군요", "잖아요", "거든요", "는데요", "은데요", "던데요", "니까요", "게요",
    "까요", "더라고요", "라고요", "다고요", "대요", "래요", "걸요", "지요",
)
ARCHAIC_CONTRACTED: dict[str, str] = {
    "해요": "하오",
    "돼요": "되오",
    "봐요": "보오",
    "줘요": "주오",
    "가요": "가오",
    "그래요": "그렇소",
    "어때요": "어떻소",
    "몰라요": "모르오",
    "알아요": "아오",
    "만나요": "만나오",
    "쉬어요": "쉬오",
    "자요": "자오",
}
ARCHAIC_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("네요", "구려"),
    ("군요", "구려"),
    ("잖아요", "잖소"),
)
FIRST_PERSON_FORMS: tuple[tuple[str, str], ...] = (
    ("나는", "{p}{topic}"),
    ("난", "{short_topic}"),
    ("나도", "{p}도"),
    ("나를", "{p}{object}"),
    ("나랑", "{p}{with}"),
    ("내가", "{subject_form}"),
    ("나의", "{p}의"),
    ("나한테", "{p}한테"),
    ("나만", "{p}만"),
    ("나처럼", "{p}처럼"),
    ("나보다", "{p}보다"),
    ("나와", "{p}{with_formal}"),
    ("나에게", "{p}에게"),
    ("나까지", "{p}까지"),
    ("내", "{possessive}"),
    ("나", "{p}"),
)
SECOND_PERSON_FORMS: tuple[tuple[str, str], ...] = (
    ("너는", "{a}{topic}"),
    ("넌", "{a}{topic}"),
    ("너도", "{a}도"),
    ("너를", "{a}{object}"),
    ("널", "{a}{object}"),
    ("너랑", "{a}{with}"),
    ("네가", "{a}{subject}"),
    ("니가", "{a}{subject}"),
    ("너한테", "{a}한테"),
    ("너에게", "{a}에게"),
    ("너의", "{a}의"),
    ("너", "{a}"),
)


def clean_roleplay_text(text: str) -> str:
    without_actions = STAGE_DIRECTION_PATTERN.sub(" ", text).replace('"', " ")
    return SPACE_PATTERN.sub(" ", EMOTICON_PATTERN.sub("", without_actions)).strip()


def _replace_words(text: str, forms: tuple[tuple[str, str], ...], values: dict[str, str]) -> str:
    for source, template in forms:
        text = re.sub(
            rf"{BOUNDARY_BEFORE}{source}{BOUNDARY_AFTER}", template.format(**values), text
        )
    return text


def first_person_values(term: str) -> dict[str, str]:
    if term == "저":
        return {
            "p": "저", "topic": "는", "short_topic": "전", "object": "를", "with": "랑",
            "with_formal": "와", "subject_form": "제가", "possessive": "제",
        }
    return {
        "p": term,
        "topic": topic_particle(term),
        "short_topic": term + topic_particle(term),
        "object": object_particle(term),
        "with": with_particle(term),
        "with_formal": "과" if has_final(term) else "와",
        "subject_form": term + subject_particle(term),
        "possessive": f"{term}의",
    }


def second_person_values(address: str) -> dict[str, str]:
    return {
        "a": address,
        "topic": topic_particle(address),
        "object": object_particle(address),
        "with": with_particle(address),
        "subject": subject_particle(address),
    }


def adapt_persons(text: str, profile: SpeechProfile) -> str:
    if profile.first_person != "나":
        text = _replace_words(text, FIRST_PERSON_FORMS, first_person_values(profile.first_person))
    if not (profile.uses_second_person and profile.level is SpeechLevel.CASUAL):
        text = _replace_words(text, SECOND_PERSON_FORMS, second_person_values(profile.address))
    return text


def to_polite(core: str) -> str | None:
    if core.endswith(("요", "죠", "니다", "니까")):
        return core
    last_word = core.rsplit(" ", 1)[-1]
    head = core[: len(core) - len(last_word)]
    repeated = REDUPLICATION_PATTERN.match(last_word)
    if repeated and repeated.group(1) in POLITE_EXACT:
        return head + POLITE_EXACT[repeated.group(1)]
    if core in POLITE_EXACT:
        return POLITE_EXACT[core]
    if last_word in POLITE_EXACT:
        return head + POLITE_EXACT[last_word]
    if core in INTERJECTIONS:
        return core
    if last_word in SUGGESTION_POLITE:
        return head + SUGGESTION_POLITE[last_word]
    for source, target in PLAIN_DA_POLITE.items():
        if last_word.endswith(source):
            return core[: len(core) - len(source)] + target
    if last_word.endswith(("었다", "았다", "였다", "했다", "됐다", "겠다")):
        return core[:-1] + "어요"
    if last_word.endswith("야") and len(last_word) > 1:
        stem = core[:-1]
        if stem.endswith("이") and has_final(stem[:-1]):
            return stem[:-1] + "이에요"
        return stem + ("이에요" if has_final(stem) else "예요")
    for source, target in POLITE_SUFFIXES:
        if last_word.endswith(source) and len(last_word) > len(source) - 1:
            return core[: len(core) - len(source)] + target
    last = core[-1]
    if last in VERB_FINAL_SYLLABLES and final_consonant(last) == FINAL_NONE:
        return core + "요"
    return None


def to_formal(polite: str, question: bool) -> str | None:
    for source, target in FORMAL_CONTRACTED.items():
        if polite.endswith(source):
            formal = polite[: len(polite) - len(source)] + target
            return _formal_question(formal, question)
    if polite.endswith("죠"):
        return polite[:-1] + "지요"
    if polite.endswith(("이에요", "예요")):
        stem = polite[:-3] if polite.endswith("이에요") else polite[:-2]
        if stem.endswith("거"):
            stem = stem[:-1] + "것"
        return _formal_question(stem + "입니다", question)
    if polite.endswith(FORMAL_KEEP_SUFFIXES):
        return polite
    if polite.endswith("해요"):
        return _formal_question(polite[:-2] + "합니다", question)
    if polite.endswith("워요") and len(polite) > 2:
        stem = polite[:-2]
        return _formal_question(stem[:-1] + with_final(stem[-1], FINAL_BIEUP) + "습니다", question)
    if polite.endswith(("어요", "아요", "여요")) and len(polite) > 2:
        stem = polite[:-2]
        final = final_consonant(stem[-1])
        if final not in (FINAL_NONE, FINAL_RIEUL):
            return _formal_question(stem + "습니다", question)
        return None
    if polite.endswith(("니다", "니까")):
        return polite
    return None


def _formal_question(formal: str, question: bool) -> str:
    if question and formal.endswith("니다"):
        return formal[:-2] + "니까"
    return formal


def to_military(formal: str) -> str | None:
    for ending in ("니다", "니까"):
        if formal.endswith(ending) and len(formal) > 2:
            syllable = formal[-3]
            if final_consonant(syllable) == FINAL_BIEUP:
                return formal[:-3] + with_final(syllable, FINAL_MIEUM) + ending[-1]
    return None


def to_archaic(polite: str) -> str | None:
    for source, target in ARCHAIC_CONTRACTED.items():
        if polite.endswith(source):
            return polite[: len(polite) - len(source)] + target
    for source, target in ARCHAIC_SUFFIXES:
        if polite.endswith(source):
            return polite[: len(polite) - len(source)] + target
    if polite.endswith(("이에요", "예요")):
        stem = polite[:-3] if polite.endswith("이에요") else polite[:-2]
        if stem.endswith("거"):
            stem = stem[:-1] + "것"
        return stem + "이오"
    if polite.endswith("워요") and len(polite) > 2:
        stem = polite[:-2]
        return stem[:-1] + with_final(stem[-1], FINAL_BIEUP) + "소"
    if polite.endswith(("어요", "아요", "여요")) and len(polite) > 2:
        stem = polite[:-2]
        if final_consonant(stem[-1]) not in (FINAL_NONE, FINAL_RIEUL):
            return stem + "소"
    return None


def convert_leading_clauses(core: str, level: SpeechLevel) -> str | None:
    clauses = core.split(CLAUSE_SEPARATOR)
    converted: list[str] = []
    for clause in clauses[:-1]:
        stripped = clause.strip()
        if stripped in CASUAL_VOCATIVES:
            continue
        if stripped in POLITE_EXACT:
            ending = convert_ending(stripped, "", level)
            if ending is None:
                return None
            converted.append(ending)
        else:
            converted.append(clause)
    return CLAUSE_SEPARATOR.join([*converted, clauses[-1]])


def convert_ending(core: str, marks: str, level: SpeechLevel) -> str | None:
    if level is SpeechLevel.CASUAL:
        return core
    if core in INTERJECTIONS:
        return core
    polite = to_polite(core)
    if polite is None:
        return None
    if level is SpeechLevel.POLITE:
        return polite
    if level is SpeechLevel.ARCHAIC:
        return to_archaic(polite)
    formal = to_formal(polite, "?" in marks)
    if level is SpeechLevel.FORMAL or formal is None:
        return formal
    return to_military(formal)


def convert_speech(text: str, profile: SpeechProfile) -> str | None:
    adapted = adapt_persons(clean_roleplay_text(text), profile)
    converted: list[str] = []
    for body, marks in SENTENCE_PATTERN.findall(adapted):
        stripped = body.strip()
        if not stripped:
            continue
        core = convert_leading_clauses(stripped, profile.level)
        if core is None:
            return None
        ending = (
            core
            if core.endswith(profile.address)
            else convert_ending(core, marks, profile.level)
        )
        if ending is None:
            return None
        converted.append(ending + marks)
    result = " ".join(converted).strip()
    return result or None
