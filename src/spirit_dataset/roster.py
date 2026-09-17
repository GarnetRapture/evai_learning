import re
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from common.errors import EvaiError
from game_data.database import open_tbl_database
from game_data.localization import StringResolver
from game_data.references import StringTableReferences
from persona.loader import discover_persona_files, load_persona_file

SLUG_INVALID_CHARACTER_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class SpiritIdentity:
    hero_no: int
    slug: str
    name: str
    name_en: str | None
    legacy_persona_file: Path | None


@dataclass(frozen=True)
class SpiritRoster:
    spirits: list[SpiritIdentity]
    unmatched_legacy_files: list[Path]


def slug_from_english_name(name_en: str) -> str:
    return SLUG_INVALID_CHARACTER_PATTERN.sub("_", name_en.lower()).strip("_")


def legacy_persona_names() -> dict[str, Path]:
    names: dict[str, Path] = {}
    for file_path in discover_persona_files():
        persona = load_persona_file(file_path)
        if persona.name in names:
            raise EvaiError(f"Duplicate legacy persona name '{persona.name}': {file_path}")
        names[persona.name] = file_path
    return names


def load_spirit_roster(
    resolver: StringResolver, references: StringTableReferences
) -> SpiritRoster:
    name_table = references.string_table("Hero", "NameSno")
    legacy_by_name = legacy_persona_names()
    matched_files: set[Path] = set()
    spirits: list[SpiritIdentity] = []
    used_slugs: dict[str, int] = {}
    with closing(open_tbl_database("hero")) as hero:
        rows = hero.execute(
            "SELECT h.No, h.NameSno FROM Hero h WHERE h.IsCollectable = 1 "
            "AND EXISTS (SELECT 1 FROM HeroDesc d WHERE d.HeroNo = h.No) ORDER BY h.No"
        ).fetchall()
    for row in rows:
        name = resolver.resolve_kr(name_table, row["NameSno"])
        name_en = resolver.resolve_text(name_table, row["NameSno"], "en")
        if not name:
            raise EvaiError(f"Hero {row['No']} has no Korean name")
        legacy_file = legacy_by_name.get(name)
        if legacy_file is not None:
            slug = legacy_file.stem
            matched_files.add(legacy_file)
        elif name_en:
            slug = slug_from_english_name(name_en)
        else:
            raise EvaiError(f"Hero {row['No']} has neither legacy persona nor English name")
        if slug in used_slugs:
            raise EvaiError(f"Slug '{slug}' collides for heroes {used_slugs[slug]} and {row['No']}")
        used_slugs[slug] = row["No"]
        spirits.append(
            SpiritIdentity(
                hero_no=row["No"],
                slug=slug,
                name=name,
                name_en=name_en,
                legacy_persona_file=legacy_file,
            )
        )
    unmatched = sorted(set(legacy_by_name.values()) - matched_files)
    return SpiritRoster(spirits=spirits, unmatched_legacy_files=unmatched)
