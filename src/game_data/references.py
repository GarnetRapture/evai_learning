import sqlite3

from common.errors import EvaiError
from game_data.database import open_tbl_database


class StringTableReferences:
    def __init__(self, connection: sqlite3.Connection | None = None) -> None:
        source = connection if connection is not None else open_tbl_database("meta")
        targets: dict[tuple[str, str], list[str]] = {}
        try:
            for row in source.execute(
                "SELECT from_table, from_column, target_string_table FROM sno_reference_map "
                "WHERE is_primary_target = 1"
            ):
                targets.setdefault((row["from_table"], row["from_column"]), []).append(
                    row["target_string_table"]
                )
        finally:
            if connection is None:
                source.close()
        self._targets = targets

    def string_table(self, table: str, column: str) -> str:
        candidates = self._targets.get((table, column), [])
        if len(candidates) != 1:
            raise EvaiError(
                f"String table for {table}.{column} is not uniquely resolved: {candidates}"
            )
        return candidates[0]
