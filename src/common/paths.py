from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
TBL_DIR: Path = DATA_DIR / "tbl"
SPIRIT_MEMORY_DIR: Path = DATA_DIR / "spirit_memory"
SPIRIT_JUDGMENT_DIR: Path = DATA_DIR / "spirit_judgment"
MODEL_DIR: Path = PROJECT_ROOT / "models" / "lfm2-230m"
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"

DATASETS_DIR: Path = ARTIFACT_DIR / "datasets"
ADAPTERS_DIR: Path = ARTIFACT_DIR / "adapters"
MERGED_DIR: Path = ARTIFACT_DIR / "merged"
GGUF_DIR: Path = ARTIFACT_DIR / "gguf"
REPORTS_DIR: Path = ARTIFACT_DIR / "reports"
SCRATCH_DIR: Path = PROJECT_ROOT / "tmp-codex"


def spirit_adapter_dir(slug: str) -> Path:
    return ADAPTERS_DIR / slug


def spirit_training_report_path(slug: str) -> Path:
    return REPORTS_DIR / f"{slug}_spirit_lora.json"


def ensure_artifact_directories() -> None:
    for directory in (
        ARTIFACT_DIR,
        DATASETS_DIR,
        ADAPTERS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ROSTER_FILE_NAME = "spirit_roster.json"
SPIRIT_FILE_NAME = "spirit.json"
