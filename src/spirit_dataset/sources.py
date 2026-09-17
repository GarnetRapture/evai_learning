from collections import defaultdict
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace

from game_data.database import open_tbl_database
from game_data.localization import StringResolver
from game_data.references import StringTableReferences
from game_data.story import BOND_STORY_TYPES, MAIN_STORY_TYPE, StoryLine, StoryRepository
from spirit_dataset.curriculum import owns_story
from spirit_dataset.events import EventIndex
from spirit_dataset.language import render
from spirit_dataset.memory import ALTERNATE_ENDING_AFFINITIES
from spirit_dataset.profile import SpiritProfile
from spirit_dataset.records import ExclusionReason, SourceKind, SourceReference
from spirit_dataset.roster import SpiritIdentity
from spirit_dataset.script import (
    SPEECH_UI_TYPES,
    ChoiceLayout,
    ScriptExchange,
    ScriptLine,
    ScriptWalker,
)
from spirit_dataset.situations import (
    BUBBLE_BATTLE_COLUMNS,
    BUBBLE_COLUMN_EMOTIONS,
    BUBBLE_COLUMN_SITUATIONS,
    BUBBLE_STRING_TABLE,
    EVERTALK_OPENING_SITUATION,
    HERO_COMMENT_SITUATION,
    HERO_DESC_SITUATIONS,
    LOBBY_GAME_FEATURE_TYPES,
    LOBBY_TYPE_SITUATIONS,
    TOWN_LOST_ITEM_OPENING_SITUATION,
    TRIP_KEYWORD_STRING_TABLE,
    TRIP_OPENING_SITUATION,
    TRIP_PERSONAL_KEYWORD_SITUATION,
    TRIP_SHARED_KEYWORD_SITUATION,
)
from spirit_dataset.text_cleaning import clean_game_text

TOWN_LOST_ITEM_GROUP_COLUMNS: tuple[str, ...] = (
    "GroupStart",
    "GroupDoing",
    "GroupEnd",
    "GroupTrip",
)


@dataclass(frozen=True)
class CanonicalExchange:
    source: SourceReference
    situation: str
    user_items: tuple[str, ...]
    previous_spirit_text: str | None
    spirit_text: str
    love_level: int | None
    emotion: str | None
    previous_user_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSkip:
    source: SourceReference
    reason: ExclusionReason
    detail: str
    text: str


@dataclass(frozen=True)
class SpiritSourceMaterial:
    exchanges: list[CanonicalExchange]
    skips: list[SourceSkip]


class SpiritSourceReader:
    def __init__(
        self,
        resolver: StringResolver,
        references: StringTableReferences,
        story_repository: StoryRepository,
    ) -> None:
        self._resolver = resolver
        self._references = references
        self._story = story_repository
        with ExitStack() as resources:
            self._load_sources(resolver, resources)
        story_types = (MAIN_STORY_TYPE, *sorted(BOND_STORY_TYPES))
        self._episodes = [
            episode
            for story_type in story_types
            for episode in self._story.episodes(story_type)
            if episode.ending_affinity not in ALTERNATE_ENDING_AFFINITIES
        ]
        self._episode_lines = {
            group: self._script_lines(lines)
            for group, lines in self._story.episode_lines(story_types).items()
        }
        self.events = EventIndex(self._episodes)
        self._speaker_groups: dict[int, set[int]] = defaultdict(set)
        self._speaker_name_groups: dict[str, set[int]] = defaultdict(set)
        for group_no, lines in self._episode_lines.items():
            for line in lines:
                if line.ui_type in SPEECH_UI_TYPES:
                    self._speaker_groups[line.speaker_no].add(group_no)
                    if line.speaker_name:
                        self._speaker_name_groups[line.speaker_name].add(group_no)

    def _load_sources(self, resolver: StringResolver, resources: ExitStack) -> None:
        evertalk = resources.enter_context(closing(open_tbl_database("evertalk")))
        self._evertalk_episodes: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in evertalk.execute(
            "SELECT HeroNo, GroupNo, LoveLevel FROM EverTalk WHERE Hide = 0 ORDER BY HeroNo, No"
        ):
            self._evertalk_episodes[row["HeroNo"]].append((row["GroupNo"], row["LoveLevel"]))
        self._evertalk_levels = {
            group_no: love_level
            for episodes in self._evertalk_episodes.values()
            for group_no, love_level in episodes
        }
        self._evertalk_lines: dict[int, list[ScriptLine]] = defaultdict(list)
        for row in evertalk.execute(
            "SELECT No, GroupNo, TalkIndex, UiType, ChoiceGroup, SpeakerNo FROM EverTalkDesc "
            "WHERE Hide = 0 ORDER BY GroupNo, TalkIndex, No"
        ):
            self._evertalk_lines[row["GroupNo"]].append(
                ScriptLine(
                    key=row["No"],
                    talk_index=row["TalkIndex"],
                    ui_type=row["UiType"].lower(),
                    choice_group=row["ChoiceGroup"],
                    speaker_no=row["SpeakerNo"],
                    speaker_name=self._story.speaker_name(row["SpeakerNo"]),
                    text=clean_game_text(resolver.resolve_current("StringEverTalk", row["No"])),
                )
            )
        lobby = resources.enter_context(closing(open_tbl_database("lobby")))
        self._lobby_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in lobby.execute("SELECT * FROM LobbyAnimation ORDER BY HeroNo, No"):
            self._lobby_rows[row["HeroNo"]].append(dict(row))
        story = resources.enter_context(closing(open_tbl_database("story")))
        self._bubbles: dict[int, dict[str, object]] = {
            row["HeroNo"]: dict(row) for row in story.execute("SELECT * FROM TalkBubble")
        }
        trip = resources.enter_context(closing(open_tbl_database("trip")))
        self._trip_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in trip.execute("SELECT * FROM TripHero ORDER BY HeroNo, No"):
            self._trip_rows[row["HeroNo"]].append(dict(row))
        self._trip_keywords: dict[int, tuple[int, str]] = {}
        for row in trip.execute("SELECT No, HeroNo, KeywordString FROM TripKeyword"):
            keyword = clean_game_text(
                resolver.resolve_current(TRIP_KEYWORD_STRING_TABLE, row["KeywordString"])
            )
            if keyword:
                self._trip_keywords[row["No"]] = (row["HeroNo"], keyword)
        town = resources.enter_context(closing(open_tbl_database("town")))
        self._town_groups: dict[int, list[int]] = defaultdict(list)
        for row in town.execute("SELECT * FROM TownLostItem ORDER BY HeroNo, No"):
            for column in TOWN_LOST_ITEM_GROUP_COLUMNS:
                group_no = row[column]
                if group_no and group_no not in self._town_groups[row["HeroNo"]]:
                    self._town_groups[row["HeroNo"]].append(group_no)

    def _walker(self, identity: SpiritIdentity, layout: ChoiceLayout) -> ScriptWalker:
        return ScriptWalker(
            lambda line: line.speaker_no == identity.hero_no or line.speaker_name == identity.name,
            layout,
        )

    def _context(self, template: str, **values: object) -> str:
        return render(template, self._resolver.language, **values)

    def _story_lines(self, group_no: int) -> list[ScriptLine]:
        return self._script_lines(self._story.lines(group_no))

    @staticmethod
    def _script_lines(lines: list[StoryLine]) -> list[ScriptLine]:
        return [
            ScriptLine(
                key=line.talk_no,
                talk_index=line.talk_index,
                ui_type=line.ui_type.lower(),
                choice_group=line.choice_group,
                speaker_no=line.speaker_no,
                speaker_name=line.speaker_name,
                text=clean_game_text(line.text),
            )
            for line in lines
        ]

    def _story_exchanges(
        self, identity: SpiritIdentity, skips: list[SourceSkip]
    ) -> list[CanonicalExchange]:
        groups = self._speaker_groups.get(identity.hero_no, set()) | (
            self._speaker_name_groups.get(identity.name, set())
        )
        walker = self._walker(identity, ChoiceLayout.GROUPED_BY_CHOICE_GROUP)
        exchanges: list[CanonicalExchange] = []
        for episode in self._episodes:
            if episode.talk_group not in groups:
                continue
            if not owns_story(
                identity.hero_no, identity.is_variant, episode.story_type, episode.act
            ):
                continue
            lines = self._episode_lines[episode.talk_group]
            group_skips: list[SourceSkip] = []
            parsed = self._parse_group(
                walker,
                lines,
                SourceKind.STORY,
                "Talk",
                episode.talk_group,
                group_skips,
            )
            situation = clean_game_text(episode.title)
            contextualized = []
            for exchange in parsed:
                if not exchange.user_items and not situation:
                    group_skips.append(
                        SourceSkip(
                            SourceReference(
                                SourceKind.STORY,
                                "Talk",
                                (
                                    episode.talk_group,
                                    *exchange.keys,
                                ),
                            ),
                            ExclusionReason.UNVERIFIED_TRIGGER,
                            "Story opening has no resolved situation",
                            exchange.spirit_text,
                        )
                    )
                else:
                    contextualized.append(exchange)
            for exchange in self._from_script(
                contextualized,
                SourceKind.STORY,
                "Talk",
                episode.talk_group,
                situation,
                self._evertalk_levels.get(episode.messenger_group),
            ):
                exchanges.append(
                    replace(
                        exchange,
                        source=replace(exchange.source, story_no=episode.story_no),
                    )
                )
            skips.extend(
                replace(
                    skip,
                    source=replace(skip.source, story_no=episode.story_no),
                )
                for skip in group_skips
            )
        return exchanges

    @staticmethod
    def _from_script(
        exchanges: list[ScriptExchange],
        kind: SourceKind,
        table: str,
        group_no: int,
        situation: str,
        love_level: int | None,
    ) -> list[CanonicalExchange]:
        return [
            CanonicalExchange(
                source=SourceReference(kind=kind, table=table, keys=(group_no, *exchange.keys)),
                situation=situation,
                user_items=exchange.user_items,
                previous_spirit_text=exchange.previous_spirit_text,
                previous_user_items=exchange.previous_user_items,
                spirit_text=exchange.spirit_text,
                love_level=love_level,
                emotion=None,
            )
            for exchange in exchanges
        ]

    def _parse_group(
        self,
        walker: ScriptWalker,
        lines: list[ScriptLine],
        kind: SourceKind,
        table: str,
        group_no: int,
        skips: list[SourceSkip],
    ) -> list[ScriptExchange]:
        parsed = walker.parse(lines)
        for line in parsed.unconverted:
            skips.append(
                SourceSkip(
                    SourceReference(kind, table, (group_no, line.key)),
                    ExclusionReason.EMPTY_TEXT
                    if not line.text
                    else ExclusionReason.UNVERIFIED_TRIGGER,
                    f"Unconverted source UI type: {line.ui_type}",
                    line.text,
                )
            )
        for pending in parsed.unanswered:
            skips.append(
                SourceSkip(
                    SourceReference(kind, table, (group_no, *pending.keys)),
                    ExclusionReason.UNANSWERED_CONTEXT,
                    "Source context has no following spirit response",
                    "\n".join(pending.items),
                )
            )
        return parsed.exchanges

    def _evertalk(
        self, identity: SpiritIdentity, skips: list[SourceSkip]
    ) -> list[CanonicalExchange]:
        walker = self._walker(identity, ChoiceLayout.EACH_ROW_IS_OPTION)
        exchanges: list[CanonicalExchange] = []
        for group_no, love_level in self._evertalk_episodes.get(identity.hero_no, []):
            exchanges.extend(
                self._from_script(
                    self._parse_group(
                        walker,
                        self._evertalk_lines.get(group_no, []),
                        SourceKind.EVERTALK,
                        "EverTalkDesc",
                        group_no,
                        skips,
                    ),
                    SourceKind.EVERTALK,
                    "EverTalkDesc",
                    group_no,
                    self._context(EVERTALK_OPENING_SITUATION),
                    love_level,
                )
            )
        return exchanges

    def _trip(self, identity: SpiritIdentity, skips: list[SourceSkip]) -> list[CanonicalExchange]:
        walker = self._walker(identity, ChoiceLayout.GROUPED_BY_CHOICE_GROUP)
        exchanges: list[CanonicalExchange] = []
        for row in self._trip_rows.get(identity.hero_no, []):
            keyword_entry = self._trip_keywords.get(int(str(row["KeywordNo"])))
            if keyword_entry is None:
                situation = self._context(TRIP_OPENING_SITUATION)
            elif keyword_entry[0] == identity.hero_no:
                situation = self._context(TRIP_PERSONAL_KEYWORD_SITUATION, keyword=keyword_entry[1])
            else:
                situation = self._context(TRIP_SHARED_KEYWORD_SITUATION, keyword=keyword_entry[1])
            group_no = int(str(row["KeywordTalk"]))
            exchanges.extend(
                self._from_script(
                    self._parse_group(
                        walker,
                        self._story_lines(group_no),
                        SourceKind.TRIP,
                        "Talk",
                        group_no,
                        skips,
                    ),
                    SourceKind.TRIP,
                    "Talk",
                    group_no,
                    situation,
                    None,
                )
            )
        return exchanges

    def _town(self, identity: SpiritIdentity, skips: list[SourceSkip]) -> list[CanonicalExchange]:
        walker = self._walker(identity, ChoiceLayout.GROUPED_BY_CHOICE_GROUP)
        exchanges: list[CanonicalExchange] = []
        for group_no in self._town_groups.get(identity.hero_no, []):
            exchanges.extend(
                self._from_script(
                    self._parse_group(
                        walker,
                        self._story_lines(group_no),
                        SourceKind.TOWN_LOST_ITEM,
                        "Talk",
                        group_no,
                        skips,
                    ),
                    SourceKind.TOWN_LOST_ITEM,
                    "Talk",
                    group_no,
                    self._context(TOWN_LOST_ITEM_OPENING_SITUATION),
                    None,
                )
            )
        return exchanges

    def _single(
        self,
        kind: SourceKind,
        table: str,
        keys: tuple[int, ...],
        situation: str,
        text: str,
        emotion: str | None,
    ) -> CanonicalExchange:
        return CanonicalExchange(
            source=SourceReference(kind=kind, table=table, keys=keys),
            situation=situation,
            user_items=(),
            previous_spirit_text=None,
            spirit_text=text,
            love_level=None,
            emotion=emotion,
        )

    def _lobby(self, identity: SpiritIdentity, skips: list[SourceSkip]) -> list[CanonicalExchange]:
        table = self._references.string_table("LobbyAnimation", "TextSno")
        exchanges: list[CanonicalExchange] = []
        for row in self._lobby_rows.get(identity.hero_no, []):
            lobby_type = str(row["Type"])
            source = SourceReference(
                kind=SourceKind.LOBBY, table="LobbyAnimation", keys=(int(str(row["No"])),)
            )
            text = clean_game_text(self._resolver.resolve_current(table, int(str(row["TextSno"]))))
            if not text:
                skips.append(SourceSkip(source, ExclusionReason.EMPTY_TEXT, lobby_type, ""))
                continue
            if lobby_type in LOBBY_GAME_FEATURE_TYPES:
                skips.append(
                    SourceSkip(source, ExclusionReason.GAME_FEATURE_GUIDE, lobby_type, text)
                )
                continue
            template = LOBBY_TYPE_SITUATIONS.get(lobby_type)
            if template is None:
                skips.append(
                    SourceSkip(source, ExclusionReason.UNVERIFIED_TRIGGER, lobby_type, text)
                )
                continue
            exchanges.append(
                self._single(
                    SourceKind.LOBBY,
                    "LobbyAnimation",
                    source.keys,
                    self._context(template, month=row["Mm"], day=row["Dd"]),
                    text,
                    str(row["Emotion"]) or None,
                )
            )
        return exchanges

    def _bubble(self, identity: SpiritIdentity, skips: list[SourceSkip]) -> list[CanonicalExchange]:
        row = self._bubbles.get(identity.hero_no)
        if row is None:
            return []
        exchanges: list[CanonicalExchange] = []
        for column, value in row.items():
            if column == "HeroNo" or not isinstance(value, int) or value == 0:
                continue
            source = SourceReference(
                kind=SourceKind.BUBBLE, table=f"TalkBubble.{column}", keys=(identity.hero_no, value)
            )
            text = clean_game_text(self._resolver.resolve_current(BUBBLE_STRING_TABLE, value))
            if text.startswith(f"말풍선_{identity.name}") or text in (
                f"아르바이트 시작_{identity.name}",
                f"아르바이트 종료_{identity.name}",
            ):
                skips.append(
                    SourceSkip(
                        source,
                        ExclusionReason.UNRESOLVED_STRING,
                        "Localization placeholder",
                        text,
                    )
                )
                continue
            if column in BUBBLE_BATTLE_COLUMNS:
                skips.append(SourceSkip(source, ExclusionReason.GAME_FEATURE_GUIDE, column, text))
                continue
            situation = BUBBLE_COLUMN_SITUATIONS.get(column)
            if situation is None:
                skips.append(SourceSkip(source, ExclusionReason.UNVERIFIED_TRIGGER, column, text))
                continue
            if not text:
                skips.append(SourceSkip(source, ExclusionReason.UNRESOLVED_STRING, column, ""))
                continue
            exchanges.append(
                self._single(
                    SourceKind.BUBBLE,
                    source.table,
                    source.keys,
                    self._context(situation),
                    text,
                    BUBBLE_COLUMN_EMOTIONS.get(column),
                )
            )
        return exchanges

    def _profile(self, profile: SpiritProfile) -> list[CanonicalExchange]:
        exchanges: list[CanonicalExchange] = []
        for field_name, situation in HERO_DESC_SITUATIONS.items():
            text = profile.fields.get(field_name)
            if text:
                exchanges.append(
                    self._single(
                        SourceKind.HERO_DESC,
                        f"HeroDesc.{field_name}",
                        (profile.identity.hero_no,),
                        self._context(situation),
                        text,
                        None,
                    )
                )
        for comment in profile.comments_written:
            if comment.about_name is None:
                continue
            exchanges.append(
                self._single(
                    SourceKind.HERO_COMMENT,
                    "HeroComment",
                    (comment.comment_no,),
                    self._context(HERO_COMMENT_SITUATION, name=comment.about_name),
                    comment.text,
                    None,
                )
            )
        return exchanges

    def read(self, profile: SpiritProfile) -> SpiritSourceMaterial:
        identity = profile.identity
        skips: list[SourceSkip] = []
        exchanges = [
            *self._profile(profile),
            *self._lobby(identity, skips),
            *self._bubble(identity, skips),
            *self._evertalk(identity, skips),
            *self._trip(identity, skips),
            *self._town(identity, skips),
            *self._story_exchanges(identity, skips),
        ]
        return SpiritSourceMaterial(exchanges=exchanges, skips=skips)
