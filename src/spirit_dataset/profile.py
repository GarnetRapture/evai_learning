from dataclasses import dataclass, field, replace
from typing import Any

from common.errors import EvaiError
from game_data.database import open_tbl_database
from game_data.localization import StringResolver
from game_data.references import StringTableReferences
from spirit_dataset.curriculum import learns_narrative
from spirit_dataset.language import render
from spirit_dataset.records import MEMORY_LINE_MAX_LENGTH, MemoryEvidence, SelfMemory, SourceClass
from spirit_dataset.roster import UNRELEASED_SPIRIT_HERO_NOS, SpiritIdentity
from spirit_dataset.text_cleaning import clean_game_text, is_localization_placeholder
from spirit_dataset.world import shared_world_memories

HERO_DESC_SHIFTED_COLUMNS: dict[str, str] = {
    "hobby": "UnionSno",
    "speciality": "HobbySno",
    "like": "SpecialitySno",
    "dislike": "LikeSno",
    "introduction": "DislikeSno",
    "union": "CvJpSno",
}
HERO_DESC_ALIGNED_COLUMNS: dict[str, str] = {
    "nickname": "NickNameSno",
    "greeting": "MentSno",
    "contract_line": "GachaMentSno",
    "constellation": "ConstellationSno",
}
PROFILE_LIST_SEPARATOR = ","

PROJECT_CONTRACT_MEMORY: tuple[tuple[str, str], ...] = (
    ("나와 대화하는 구원자는 성인 남성", "나는 어떤 존재야?"),
    ("나는 구원자에게 연애 감정을 품고 있어", "나를 어떤 마음으로 대하고 있어?"),
)
ANIMA_IDENTITY = "나는 무기에서 태어난 여성 정령 아니마"


def core_identity_memory(identity_memory: tuple[str, ...]) -> tuple[str, ...]:
    contract_text = {
        render(text, language)
        for text, _ in PROJECT_CONTRACT_MEMORY
        for language in ("kr", "en", "zh_tw")
    }
    contract_text.update(render(ANIMA_IDENTITY, language) for language in ("kr", "en", "zh_tw"))
    return tuple(
        dict.fromkeys(
            (
                *identity_memory[:1],
                *(text for text in identity_memory if text in contract_text),
            )
        )
    )


SHARED_WORLD_MEMORY: tuple[tuple[str, str, str], ...] = (
    ("나는 유물에 깃든 영혼인 정령", "main_story 1-2", "넌 어떤 존재야?"),
    *((text, "user_contract 2026-09-17", cue) for text, cue in PROJECT_CONTRACT_MEMORY),
    ("구원자는 과거에서 소환된 인간", "main_story 1-2", "나는 어디에서 왔지?"),
    ("구원자는 정령과 계약하는 정령술사", "main_story 0-2", "내가 정령과 계약할 수 있어?"),
    ("에덴은 인간이 사라진 정령들의 낙원", "main_story 0-2", "네가 살아온 에덴은 어떤 곳이야?"),
    ("에덴의 대륙 이름은 아르카디아", "main_story 0-3", "우리가 사는 대륙 이름이 뭐야?"),
    ("정령은 죽으면 정령석으로 돌아가 잠든다", "main_story 1-2; 5-3", "정령은 죽으면 어떻게 돼?"),
    ("긴 잠에서 깨면 옛 기억이 흐려진다", "main_story 1-1; 1-2", "긴 잠에서 깨도 기억이 선명해?"),
    ("하늘에 게이트가 열려 마물이 쏟아진다", "main_story 0-3", "게이트에서는 무슨 일이 일어나?"),
    ("일곱 나라가 정령 연합군을 만들었다", "main_story 1-2", "정령 연합군은 누가 만들었어?"),
    ("유리아는 솔레이 왕국의 여왕", "main_story 1-2", "유리아는 어떤 위치에 있어?"),
    ("메피스토펠레스는 방주의 인공 정령", "main_story 1-2", "메피스토펠레스는 어떤 정령이야?"),
    ("구원자는 아케나인의 영주", "main_story 1-3; 1-4", "내가 영주로 있는 곳이 어디지?"),
    ("천사형과 악마형 정령은 드물고 강하다", "main_story 6-11; 6-12", "천사형과 악마형은 흔해?"),
    ("계약한 구원자와는 인연의 끈이 이어진다", "main_story 8-1; 8-10", "우리 계약은 무슨 의미야?"),
)
WORLD_SELF_REFERENCES: dict[int, tuple[str, str]] = {
    1010: ("메피스토펠레스는 방주의 인공 정령", "나는 방주의 인공 정령"),
    5030: ("유리아는 솔레이 왕국의 여왕", "나는 솔레이 왕국의 여왕"),
}
PROFILE_MEMORY_CUES: dict[str, str] = {
    "nickname": "네 이명은 뭐야?",
    "race": "너는 어떤 유형의 정령이야?",
    "union": "너는 어디에 소속되어 있어?",
    "constellation": "네 별자리가 뭐야?",
    "hobby": "네 취미를 알려줘.",
    "speciality": "네 특기는 뭐야?",
    "like": "네가 좋아하는 것을 말해줘.",
    "dislike": "네가 싫어하는 것을 말해줘.",
}
PROFILE_MEMORY_TEMPLATES: dict[str, str] = {
    "nickname": "내 이명은 {value}",
    "race": "나는 {value} 정령",
    "union": "내 소속은 {value}",
    "constellation": "내 별자리는 {value}",
    "hobby": "내 취미는 {value}",
    "speciality": "내 특기는 {value}",
    "like": "좋아하는 것은 {value}",
    "dislike": "싫어하는 것은 {value}",
}
PROFILE_LIST_FIELDS: frozenset[str] = frozenset({"hobby", "speciality", "like", "dislike"})
OTHER_SPIRIT_UNION_TEMPLATE = "{name}의 소속은 {value}"
OTHER_SPIRIT_RACE_TEMPLATE = "{name}의 유형은 {value}"
OTHER_SPIRIT_CUE = "{name}에 대해 알아?"


@dataclass(frozen=True)
class OtherSpiritFact:
    hero_no: int
    name: str
    value: str
    template: str
    reference: str


@dataclass(frozen=True)
class RejectedMemoryLine:
    text: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "reason": self.reason}


@dataclass(frozen=True)
class SpiritComment:
    about_hero_no: int
    about_name: str | None
    writer_hero_no: int
    writer_name: str | None
    text: str
    comment_no: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment_no": self.comment_no,
            "about_hero_no": self.about_hero_no,
            "about_name": self.about_name,
            "writer_hero_no": self.writer_hero_no,
            "writer_name": self.writer_name,
            "text": self.text,
        }


@dataclass(frozen=True)
class SpiritProfile:
    identity: SpiritIdentity
    fields: dict[str, str]
    comments_about: list[SpiritComment]
    comments_written: list[SpiritComment]
    identity_memory: tuple[str, ...]
    rejected_memory: list[RejectedMemoryLine] = field(default_factory=list)
    self_memory: tuple[SelfMemory, ...] = ()
    language: str = "kr"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hero_no": self.identity.hero_no,
            "grade_sno": int(self.identity.grade),
            "is_variant": self.identity.is_variant,
            "language": self.language,
            "slug": self.identity.slug,
            "name": self.identity.name,
            "name_en": self.identity.name_en,
            "legacy_persona_file": (
                None
                if self.identity.legacy_persona_file is None
                else self.identity.legacy_persona_file.name
            ),
            "fields": self.fields,
            "identity_memory": list(self.identity_memory),
            "self_memory": [memory.to_dict() for memory in self.self_memory],
            "rejected_memory": [line.to_dict() for line in self.rejected_memory],
            "comments_about": [comment.to_dict() for comment in self.comments_about],
            "comments_written": [comment.to_dict() for comment in self.comments_written],
        }


def accept_memory_line(text: str, accepted: list[str], rejected: list[RejectedMemoryLine]) -> bool:
    if len(text) > MEMORY_LINE_MAX_LENGTH:
        rejected.append(RejectedMemoryLine(text=text, reason="exceeds_length"))
        return False
    elif text not in accepted:
        accepted.append(text)
    return True


class SpiritProfileRepository:
    def __init__(self, resolver: StringResolver, references: StringTableReferences) -> None:
        self._resolver = resolver
        self._references = references
        self._hero = open_tbl_database("hero")
        self._roster: tuple[SpiritIdentity, ...] = ()
        self._roster_fact_cache: tuple[OtherSpiritFact, ...] | None = None

    def close(self) -> None:
        self._hero.close()

    def bind_roster(self, spirits: list[SpiritIdentity]) -> None:
        self._roster = tuple(spirits)
        self._roster_fact_cache = None

    def _roster_facts(self) -> tuple[OtherSpiritFact, ...]:
        if self._roster_fact_cache is None:
            union_column = HERO_DESC_SHIFTED_COLUMNS["union"]
            facts: list[OtherSpiritFact] = []
            for spirit in self._roster:
                hero = self._hero.execute(
                    "SELECT NameSno, RaceSno FROM Hero WHERE No = ?", (spirit.hero_no,)
                ).fetchone()
                desc = self._hero.execute(
                    f"SELECT {union_column} FROM HeroDesc WHERE HeroNo = ?", (spirit.hero_no,)
                ).fetchone()
                name = self._resolve("Hero", "NameSno", hero["NameSno"])
                union = (
                    None
                    if desc is None
                    else self._resolve("HeroDesc", union_column, desc[union_column])
                )
                if name is None:
                    raise EvaiError(f"Missing identity name for Hero {spirit.hero_no}")
                if union is not None:
                    facts.append(
                        OtherSpiritFact(
                            spirit.hero_no,
                            name,
                            union,
                            OTHER_SPIRIT_UNION_TEMPLATE,
                            f"HeroDesc.HeroNo={spirit.hero_no}; HeroDesc.{union_column}",
                        )
                    )
                    continue
                race = self._resolve("Hero", "RaceSno", hero["RaceSno"])
                if race is not None:
                    facts.append(
                        OtherSpiritFact(
                            spirit.hero_no,
                            name,
                            race,
                            OTHER_SPIRIT_RACE_TEMPLATE,
                            f"Hero.No={spirit.hero_no}; Hero.RaceSno",
                        )
                    )
            self._roster_fact_cache = tuple(facts)
        return self._roster_fact_cache

    def _resolve(self, table: str, column: str, sno: int | None) -> str | None:
        text = self._resolver.resolve_current(self._references.string_table(table, column), sno)
        cleaned = clean_game_text(text)
        return cleaned or None

    def _hero_name(self, hero_no: int) -> str | None:
        row = self._hero.execute("SELECT NameSno FROM Hero WHERE No = ?", (hero_no,)).fetchone()
        return None if row is None else self._resolve("Hero", "NameSno", row["NameSno"])

    def _comments(self, column: str, hero_no: int) -> list[SpiritComment]:
        comments: list[SpiritComment] = []
        for row in self._hero.execute(
            f"SELECT No, HeroNo, CommentHeroNo, CommentDesc FROM HeroComment "
            f"WHERE Hide = 0 AND {column} = ? ORDER BY No",
            (hero_no,),
        ):
            text = clean_game_text(
                self._resolver.resolve_current("StringCharacter", row["CommentDesc"])
            )
            if not text:
                continue
            comments.append(
                SpiritComment(
                    about_hero_no=row["HeroNo"],
                    about_name=self._hero_name(row["HeroNo"]),
                    writer_hero_no=row["CommentHeroNo"],
                    writer_name=self._hero_name(row["CommentHeroNo"]),
                    text=text,
                    comment_no=row["No"],
                )
            )
        return comments

    def load(self, identity: SpiritIdentity) -> SpiritProfile:
        desc = self._hero.execute(
            "SELECT * FROM HeroDesc WHERE HeroNo = ?", (identity.hero_no,)
        ).fetchone()
        hero_row = self._hero.execute(
            "SELECT RaceSno, NameSno FROM Hero WHERE No = ?", (identity.hero_no,)
        ).fetchone()
        if hero_row is None or (
            desc is None and identity.hero_no not in UNRELEASED_SPIRIT_HERO_NOS
        ):
            raise EvaiError(f"Hero {identity.hero_no} has no HeroDesc/Hero row")
        language = self._resolver.language
        name = self._resolver.resolve_current(
            self._references.string_table("Hero", "NameSno"),
            hero_row["NameSno"],
        )
        if not name:
            raise EvaiError(f"Missing {language} identity name for Hero {identity.hero_no}")
        identity = replace(identity, name=name)
        fields: dict[str, str] = {}
        desc_columns = {**HERO_DESC_SHIFTED_COLUMNS, **HERO_DESC_ALIGNED_COLUMNS}
        for field_name, column in desc_columns.items():
            if desc is None:
                break
            value = self._resolve("HeroDesc", column, desc[column])
            if value is not None and not is_localization_placeholder(value):
                fields[field_name] = value
        race = self._resolve("Hero", "RaceSno", hero_row["RaceSno"])
        if race is not None:
            fields["race"] = race

        accepted: list[str] = []
        rejected: list[RejectedMemoryLine] = []
        self_memory: list[SelfMemory] = []

        def add_memory(text: str, source_class: SourceClass, reference: str, cue: str) -> None:
            if accept_memory_line(text, accepted, rejected):
                self_memory.append(
                    SelfMemory(
                        text,
                        (MemoryEvidence(source_class, reference),),
                        cue,
                    )
                )

        add_memory(
            render("나는 {name}", language, name=identity.name),
            SourceClass.CANON_TBL,
            f"Hero.No={identity.hero_no}; Hero.NameSno",
            render("네 이름이 뭐야?", language),
        )
        if not learns_narrative(identity.grade):
            add_memory(
                render(ANIMA_IDENTITY, language),
                SourceClass.PROJECT_CONTRACT,
                "user_contract common_rare_anima",
                render("넌 어떤 존재야?", language),
            )
        for field_name, template in PROFILE_MEMORY_TEMPLATES.items():
            value = fields.get(field_name)
            if value is None:
                continue
            values = (
                [item.strip() for item in value.split(PROFILE_LIST_SEPARATOR) if item.strip()]
                if field_name in PROFILE_LIST_FIELDS
                else [value]
            )
            for item in values:
                reference = (
                    f"Hero.No={identity.hero_no}; Hero.RaceSno"
                    if field_name == "race"
                    else f"HeroDesc.HeroNo={identity.hero_no}; HeroDesc.{desc_columns[field_name]}"
                )
                add_memory(
                    render(template, language, value=item),
                    SourceClass.CANON_TBL,
                    reference,
                    render(PROFILE_MEMORY_CUES[field_name], language),
                )
        for text, source, cue in SHARED_WORLD_MEMORY:
            source_class = (
                SourceClass.PROJECT_CONTRACT
                if source.startswith("user_contract")
                else SourceClass.CANON_STORY
            )
            own_reference = WORLD_SELF_REFERENCES.get(identity.hero_no)
            if own_reference is not None and text == own_reference[0]:
                text = own_reference[1]
            add_memory(render(text, language), source_class, source, render(cue, language))

        for memory in shared_world_memories(language):
            if len(memory.text) > MEMORY_LINE_MAX_LENGTH:
                raise EvaiError(f"Shared world fact exceeds memory contract: {memory.text}")
            if memory.text not in accepted:
                accepted.append(memory.text)
                self_memory.append(memory)
        for other in self._roster_facts():
            if other.hero_no == identity.hero_no:
                continue
            add_memory(
                render(other.template, language, name=other.name, value=other.value),
                SourceClass.CANON_TBL,
                other.reference,
                render(OTHER_SPIRIT_CUE, language, name=other.name),
            )

        return SpiritProfile(
            identity=identity,
            fields=fields,
            comments_about=self._comments("HeroNo", identity.hero_no),
            comments_written=self._comments("CommentHeroNo", identity.hero_no),
            identity_memory=tuple(accepted),
            rejected_memory=rejected,
            self_memory=tuple(self_memory),
            language=language,
        )
