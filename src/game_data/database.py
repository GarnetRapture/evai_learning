import sqlite3
from pathlib import Path

from common.errors import EvaiError
from common.paths import TBL_DIR


def tbl_database_path(database_name: str) -> Path:
    return TBL_DIR / f"{database_name}.db"


def open_tbl_database(database_name: str) -> sqlite3.Connection:
    path = tbl_database_path(database_name)
    if not path.exists():
        raise EvaiError(f"TBL database not found: {path}")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection
