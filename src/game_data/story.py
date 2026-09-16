import sqlite3
from dataclasses import dataclass

from game_data.database import open_tbl_database
from game_data.localization import StringResolver


@dataclass(frozen=True)
class StoryEpisode:
    story_no: int
    story_type: int
    act: int
    chapter: int
    episode: int
    title: str | None
    talk_group: int
    messenger_group: int
    ending_affinity: int


@dataclass(frozen=True)
class StoryLine:
    talk_no: int
    group_no: int
    talk_index: int
    talk_type: int
    ui_type: str
    speaker_no: int
    speaker_name: str | None
    choice_group: int
    love_level: int
    hero_no: int
    text: str | None


class StoryRepository:
    def __init__(
        self,
        resolver: StringResolver,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._resolver = resolver
        self._connection = connection if connection is not None else open_tbl_database("story")
        self._actor_names: dict[int, str | None] = {
            row["No"]: resolver.resolve_kr("StringCharacter", row["NameSno"])
            for row in self._connection.execute("SELECT No, NameSno FROM TalkActor")
        }

    def speaker_name(self, speaker_no: int) -> str | None:
        return self._actor_names.get(speaker_no)

    def episodes(self, story_type: int) -> list[StoryEpisode]:
        rows = self._connection.execute(
            "SELECT No, StoryType, Act, Chapter, Episode, EpisodeNameSno, TalkGroup, "
            "MessengerGroup, EndingAffinity FROM StoryInfo WHERE StoryType = ? "
            "ORDER BY Act, Chapter, Episode, No",
            (story_type,),
        ).fetchall()
        return [
            StoryEpisode(
                story_no=row["No"],
                story_type=row["StoryType"],
                act=row["Act"],
                chapter=row["Chapter"],
                episode=row["Episode"],
                title=self._resolver.resolve_kr("StringTalk", row["EpisodeNameSno"]),
                talk_group=row["TalkGroup"],
                messenger_group=row["MessengerGroup"],
                ending_affinity=row["EndingAffinity"],
            )
            for row in rows
        ]

    def lines(self, group_no: int) -> list[StoryLine]:
        rows = self._connection.execute(
            "SELECT No, GroupNo, TalkIndex, TalkType, UiType, SpeakerNo, ChoiceGroup, "
            "LoveLevel, HeroNo, Hide FROM Talk WHERE GroupNo = ? AND Hide = 0 "
            "ORDER BY TalkIndex, No",
            (group_no,),
        ).fetchall()
        return [
            StoryLine(
                talk_no=row["No"],
                group_no=row["GroupNo"],
                talk_index=row["TalkIndex"],
                talk_type=row["TalkType"],
                ui_type=row["UiType"],
                speaker_no=row["SpeakerNo"],
                speaker_name=self.speaker_name(row["SpeakerNo"]),
                choice_group=row["ChoiceGroup"],
                love_level=row["LoveLevel"],
                hero_no=row["HeroNo"],
                text=self._resolver.resolve_kr("StringTalk", row["No"]),
            )
            for row in rows
        ]
