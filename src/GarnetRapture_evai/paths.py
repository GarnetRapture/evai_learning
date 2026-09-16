"""Authoritative project path definitions for GarnetRapture_evai."""

from pathlib import Path

# Resolve project root from this file's location
# (src/GarnetRapture_evai/paths.py -> GarnetRapture_evai -> src -> root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Canonical primary directories
DATA_DIR: Path = PROJECT_ROOT / "data"
MODEL_DIR: Path = PROJECT_ROOT / "models" / "lfm2-230m"
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"

# Canonical artifact subdirectories
DATASETS_DIR: Path = ARTIFACT_DIR / "datasets"
ADAPTERS_DIR: Path = ARTIFACT_DIR / "adapters"
MERGED_DIR: Path = ARTIFACT_DIR / "merged"
GGUF_DIR: Path = ARTIFACT_DIR / "gguf"
REPORTS_DIR: Path = ARTIFACT_DIR / "reports"


def ensure_artifact_directories() -> None:
    """Ensure that all canonical artifact directories exist on disk."""
    for directory in (
        ARTIFACT_DIR,
        DATASETS_DIR,
        ADAPTERS_DIR,
        MERGED_DIR,
        GGUF_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
