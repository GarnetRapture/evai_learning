"""Provenance tracking and manifest records for generated datasets and artifacts."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .split import DatasetSplit


def compute_file_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file deterministically in chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    """Provenance manifest recording sources, hashes, and configuration."""

    persona_id: str
    persona_name: str
    source_json_path: str
    source_json_sha256: str
    dataset_version: str
    train_count: int
    validation_count: int
    test_count: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "source_json_path": self.source_json_path,
            "source_json_sha256": self.source_json_sha256,
            "dataset_version": self.dataset_version,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "test_count": self.test_count,
            "created_at": self.created_at,
        }


def build_dataset_manifest(
    persona_id: str,
    persona_name: str,
    source_json_path: Path,
    dataset_version: str,
    split: "DatasetSplit[Any]",
) -> DatasetManifest:
    """Build a provenance manifest for one persona's generated dataset split."""
    return DatasetManifest(
        persona_id=persona_id,
        persona_name=persona_name,
        source_json_path=str(source_json_path),
        source_json_sha256=compute_file_sha256(source_json_path),
        dataset_version=dataset_version,
        train_count=len(split.train),
        validation_count=len(split.validation),
        test_count=len(split.test),
        created_at=datetime.now(UTC).isoformat(),
    )
