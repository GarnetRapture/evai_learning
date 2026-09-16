from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
TBL_DIR: Path = DATA_DIR / "tbl"
MODEL_DIR: Path = PROJECT_ROOT / "models" / "lfm2-230m"
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"

DATASETS_DIR: Path = ARTIFACT_DIR / "datasets"
ADAPTERS_DIR: Path = ARTIFACT_DIR / "adapters"
MERGED_DIR: Path = ARTIFACT_DIR / "merged"
GGUF_DIR: Path = ARTIFACT_DIR / "gguf"
REPORTS_DIR: Path = ARTIFACT_DIR / "reports"


def ensure_artifact_directories() -> None:
    for directory in (
        ARTIFACT_DIR,
        DATASETS_DIR,
        ADAPTERS_DIR,
        MERGED_DIR,
        GGUF_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
