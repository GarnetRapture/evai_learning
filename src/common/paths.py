import os
from pathlib import Path


def _project_root() -> Path:
    configured = os.environ.get("EVAI_PROJECT_ROOT")
    if configured is None:
        return Path(__file__).resolve().parent.parent.parent
    root = Path(configured)
    if not root.is_absolute():
        raise ValueError("EVAI_PROJECT_ROOT must be an absolute deployment root")
    return root.resolve()


PROJECT_ROOT: Path = _project_root()

DATA_DIR: Path = PROJECT_ROOT / "data"
TBL_DIR: Path = DATA_DIR / "tbl"
SPIRIT_MEMORY_DIR: Path = DATA_DIR / "spirit_memory"
SPIRIT_JUDGMENT_DIR: Path = DATA_DIR / "spirit_judgment"
SPIRIT_LESSONS_FILE: Path = DATA_DIR / "spirit_lessons" / "curriculum.json"
MODEL_DIR: Path = PROJECT_ROOT / "models" / "lfm2-230m"
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"

DATASETS_DIR: Path = ARTIFACT_DIR / "datasets"
GGUF_DIR: Path = ARTIFACT_DIR / "gguf"
GGUF_MODEL_FILE: Path = GGUF_DIR / "evai-230m-Q8_0.gguf"
REPORTS_DIR: Path = ARTIFACT_DIR / "reports"
SCRATCH_DIR: Path = PROJECT_ROOT / "tmp-codex"


def ensure_artifact_directories() -> None:
    for directory in (
        ARTIFACT_DIR,
        DATASETS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ROSTER_FILE_NAME = "spirit_roster.json"
SPIRIT_FILE_NAME = "spirit.json"
