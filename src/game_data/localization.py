import sqlite3
from compression import zstd
from dataclasses import dataclass

from game_data.database import open_tbl_database


@dataclass(frozen=True)
class LocalizedText:
    kr: str | None
    en: str | None
    ja: str | None
    zh_tw: str | None
    zh_cn: str | None


class StringResolver:
    def __init__(self, connection: sqlite3.Connection | None = None) -> None:
        self._connection = connection if connection is not None else open_tbl_database(
            "localization"
        )
        self._dictionaries: dict[int, zstd.ZstdDict] = {
            row["dictionary_id"]: zstd.ZstdDict(row["dictionary_blob"])
            for row in self._connection.execute(
                "SELECT dictionary_id, dictionary_blob FROM compression_dictionary"
            )
        }
        self._pool_cache: dict[int, str] = {}
        self._label_cache: dict[tuple[str, int], LocalizedText | None] = {}

    def _pool_text(self, pool_id: int | None) -> str | None:
        if pool_id is None:
            return None
        cached = self._pool_cache.get(pool_id)
        if cached is not None:
            return cached
        row = self._connection.execute(
            "SELECT dictionary_id, is_compressed, payload FROM string_pool WHERE pool_id = ?",
            (pool_id,),
        ).fetchone()
        if row is None:
            return None
        payload = row["payload"]
        if row["is_compressed"]:
            raw = zstd.decompress(payload, zstd_dict=self._dictionaries[row["dictionary_id"]])
            text = raw.decode("utf-8")
        else:
            text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        self._pool_cache[pool_id] = text
        return text

    def resolve(self, string_table: str, sno: int | None) -> LocalizedText | None:
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
        text = (
            None
            if row is None
            else LocalizedText(
                kr=self._pool_text(row["pool_id_kr"]),
                en=self._pool_text(row["pool_id_en"]),
                ja=self._pool_text(row["pool_id_ja"]),
                zh_tw=self._pool_text(row["pool_id_zh_tw"]),
                zh_cn=self._pool_text(row["pool_id_zh_cn"]),
            )
        )
        self._label_cache[key] = text
        return text

    def resolve_kr(self, string_table: str, sno: int | None) -> str | None:
        text = self.resolve(string_table, sno)
        return text.kr if text is not None else None
