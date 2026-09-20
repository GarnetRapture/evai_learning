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
TBL_DIR: Path = PROJECT_ROOT / "docs" / "tbl"
SPIRIT_MEMORY_DIR: Path = DATA_DIR / "spirit_memory"
SPIRIT_JUDGMENT_DIR: Path = DATA_DIR / "spirit_judgment"
SPIRIT_LESSONS_FILE: Path = DATA_DIR / "spirit_lessons" / "curriculum.json"
SPIRIT_LESSON_EXTENSIONS_DIR: Path = DATA_DIR / "spirit_lessons" / "extensions"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
KOREAN_ROLEPLAY_DIR: Path = EXTERNAL_DATA_DIR / "korean-role-playing"
KOREAN_ADULT_ROLEPLAY_FILE: Path = EXTERNAL_DATA_DIR / "korean-adult-roleplay" / "RP_KO.jsonl"
DIALOGUE_PATTERNS_DIR: Path = DATA_DIR / "dialogue_patterns"
INTIMACY_PATTERNS_FILE: Path = DIALOGUE_PATTERNS_DIR / "intimacy_ko.jsonl"
GENERAL_CORPUS_FILE: Path = DATA_DIR / "general_corpus" / "general_corpus.parquet"
MODEL_DIR: Path = PROJECT_ROOT / "models" / "qwen3-0.6b"
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
