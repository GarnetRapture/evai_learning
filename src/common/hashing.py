"""Streamed file digests cached by file identity and modification state."""

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=256)
def _file_digest(path: str, size: int, modified_ns: int, changed_ns: int) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_index(identity: str, modulus: int) -> int:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def compute_file_sha256(file_path: Path) -> str:
    path = file_path.resolve()
    stat = path.stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
