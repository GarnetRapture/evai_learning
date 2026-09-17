"""Commit one model and its provenance together through a recoverable journal."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, local
from typing import Any

from common.errors import EvaiError
from common.file_lock import exclusive_file_lock
from common.hashing import compute_file_sha256
from common.paths import MODEL_DIR, SCRATCH_DIR

MODEL_STAGE = SCRATCH_DIR / "final-model"
MODEL_MARKER = "evai_training.json"


class _LockState(local):
    held: bool = False


_THREAD_LOCK = RLock()
_LOCK_STATE = _LockState()


@contextmanager
def model_storage_lock() -> Iterator[None]:
    """Serialize threads/processes; nested reads retain the outer transaction lock."""
    with _THREAD_LOCK:
        if _LOCK_STATE.held:
            yield
            return
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(SCRATCH_DIR / "model-storage.lock"):
            _LOCK_STATE.held = True
            try:
                yield
            finally:
                _LOCK_STATE.held = False


def commit_pending_model(verified_digest: str | None = None) -> None:
    """Caller holds the storage lock; the staged marker is the commit journal."""
    marker = MODEL_STAGE / MODEL_MARKER
    if not marker.is_file():
        return
    contract = json.loads(marker.read_text(encoding="utf-8"))
    staged = MODEL_STAGE / "model.safetensors"
    target = MODEL_DIR / "model.safetensors"
    candidate = staged if staged.is_file() else target
    digest = verified_digest if verified_digest is not None else compute_file_sha256(candidate)
    if digest != contract["weights_sha256"]:
        raise EvaiError("Pending model update does not match its provenance journal")
    if staged.is_file():
        os.replace(staged, target)
    os.replace(marker, MODEL_DIR / MODEL_MARKER)


def read_model_marker(directory: Path) -> dict[str, Any] | None:
    with model_storage_lock():
        if directory.resolve() == MODEL_DIR.resolve():
            commit_pending_model()
        marker = directory / MODEL_MARKER
        return json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
