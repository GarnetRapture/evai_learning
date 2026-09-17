import sqlite3
from compression import zstd
from dataclasses import dataclass
from typing import Literal

from game_data.database import open_tbl_database

type SourceLanguage = Literal["kr", "en", "ja", "zh_tw", "zh_cn"]

LANGUAGE_COLUMNS: dict[SourceLanguage, str] = {
    "kr": "pool_id_kr", "en": "pool_id_en", "ja": "pool_id_ja",
    "zh_tw": "pool_id_zh_tw", "zh_cn": "pool_id_zh_cn",
}


@dataclass(frozen=True)
class LocalizedText:
    kr: str | None
    en: str | None
    ja: str | None
    zh_tw: str | None
    zh_cn: str | None


class StringResolver:
    def __init__(self, connection: sqlite3.Connection | None = None) -> None:
        self._owns_connection = connection is None
        self._connection = connection if connection is not None else open_tbl_database(
            "localization"
        )
        try:
            self._dictionaries: dict[int, zstd.ZstdDict] = {
                row["dictionary_id"]: zstd.ZstdDict(row["dictionary_blob"])
                for row in self._connection.execute(
                    "SELECT dictionary_id, dictionary_blob FROM compression_dictionary"
                )
            }
        except BaseException:
            self.close()
            raise
        self._pool_cache: dict[int, str | None] = {}
        self._label_cache: dict[
            tuple[str, int], dict[SourceLanguage, int | None] | None
        ] = {}

    def close(self) -> None:
        if self._owns_connection:
            self._connection.close()

    def _pool_text(self, pool_id: int | None) -> str | None:
        if pool_id is None:
            return None
        if pool_id in self._pool_cache:
            return self._pool_cache[pool_id]
        row = self._connection.execute(
            "SELECT dictionary_id, is_compressed, payload FROM string_pool WHERE pool_id = ?",
            (pool_id,),
        ).fetchone()
        if row is None:
            self._pool_cache[pool_id] = None
            return None
        payload = row["payload"]
        if row["is_compressed"]:
            raw = zstd.decompress(payload, zstd_dict=self._dictionaries[row["dictionary_id"]])
            text = raw.decode("utf-8")
        else:
            text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        self._pool_cache[pool_id] = text
        return text

    def _label_pool_ids(
        self, string_table: str, sno: int | None,
    ) -> dict[SourceLanguage, int | None] | None:
        if sno is None or sno == 0:
            return None
        key = (string_table, sno)
        if key in self._label_cache:
            return self._label_cache[key]
        row = self._connection.execute(
            "SELECT pool_id_kr, pool_id_en, pool_id_ja, pool_id_zh_tw, pool_id_zh_cn "
            "FROM string_label WHERE source_table = ? AND sno = ?",
            key,
        ).fetchone()
        pool_ids: dict[SourceLanguage, int | None] | None = (
            None if row is None
            else {language: row[column] for language, column in LANGUAGE_COLUMNS.items()}
        )
        self._label_cache[key] = pool_ids
        return pool_ids

    def resolve(self, string_table: str, sno: int | None) -> LocalizedText | None:
        pool_ids = self._label_pool_ids(string_table, sno)
        if pool_ids is None:
            return None
        return LocalizedText(
            kr=self._pool_text(pool_ids["kr"]), en=self._pool_text(pool_ids["en"]),
            ja=self._pool_text(pool_ids["ja"]), zh_tw=self._pool_text(pool_ids["zh_tw"]),
            zh_cn=self._pool_text(pool_ids["zh_cn"]),
        )

    def resolve_text(
        self, string_table: str, sno: int | None, language: SourceLanguage,
    ) -> str | None:
        pool_ids = self._label_pool_ids(string_table, sno)
        return self._pool_text(pool_ids[language]) if pool_ids is not None else None

    def resolve_kr(self, string_table: str, sno: int | None) -> str | None:
        return self.resolve_text(string_table, sno, "kr")
