import json
import re
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from common.errors import EvaiError
from common.paths import DATASETS_DIR, ROSTER_FILE_NAME
from game_data.database import open_tbl_database
from game_data.localization import StringResolver
from game_data.references import StringTableReferences
from spirit_dataset.curriculum import SpiritGrade

SLUG_INVALID_CHARACTER_PATTERN = re.compile(r"[^a-z0-9]+")
UNRELEASED_SPIRIT_HERO_NOS: frozenset[int] = frozenset({2060, 3020, 5050})


def roster_slugs() -> list[str]:
    path = DATASETS_DIR / ROSTER_FILE_NAME
    if not path.is_file():
        raise EvaiError(f"Spirit roster not found: {path}. Run `build-dataset` first.")
    roster = json.loads(path.read_text(encoding="utf-8"))
    return [str(spirit["slug"]) for spirit in roster["spirits"]]


@dataclass(frozen=True)
class SpiritIdentity:
    hero_no: int
    slug: str
    name: str
    name_en: str | None
    legacy_persona_file: Path | None = None
    grade: SpiritGrade = SpiritGrade.EPIC
    is_variant: bool = False


@dataclass(frozen=True)
class SpiritRoster:
    spirits: list[SpiritIdentity]
    unmatched_legacy_files: list[Path] = field(default_factory=list)


def slug_from_english_name(name_en: str) -> str:
    return SLUG_INVALID_CHARACTER_PATTERN.sub("_", name_en.lower()).strip("_")


def load_spirit_roster(resolver: StringResolver, references: StringTableReferences) -> SpiritRoster:
    name_table = references.string_table("Hero", "NameSno")
    spirits: list[SpiritIdentity] = []
    used_slugs: dict[str, int] = {}
    with closing(open_tbl_database("hero")) as hero:
        unreleased = sorted(UNRELEASED_SPIRIT_HERO_NOS)
        rows = hero.execute(
            "SELECT h.No, h.NameSno, h.GradeSno FROM Hero h WHERE h.IsCollectable = 1 "
            "AND (EXISTS (SELECT 1 FROM HeroDesc d WHERE d.HeroNo = h.No) "
            f"OR h.No IN ({','.join('?' for _ in unreleased)})) ORDER BY h.No",
            unreleased,
        ).fetchall()
    with closing(open_tbl_database("story")) as story:
        variants = {
            row["Act"]
            for row in story.execute("SELECT DISTINCT Act FROM StoryInfo WHERE StoryType = 10")
        }
    for row in rows:
        name = resolver.resolve_kr(name_table, row["NameSno"])
        name_en = resolver.resolve_text(name_table, row["NameSno"], "en")
        if not name:
            raise EvaiError(f"Hero {row['No']} has no Korean name")
        if not name_en:
            raise EvaiError(f"Hero {row['No']} has no canonical English name")
        slug = slug_from_english_name(name_en)
        if slug in used_slugs:
            raise EvaiError(f"Slug '{slug}' collides for heroes {used_slugs[slug]} and {row['No']}")
        used_slugs[slug] = row["No"]
        spirits.append(
            SpiritIdentity(
                hero_no=row["No"],
                slug=slug,
                name=name,
                name_en=name_en,
                grade=SpiritGrade(row["GradeSno"]),
                is_variant=row["No"] in variants,
            )
        )
    return SpiritRoster(spirits=spirits)
