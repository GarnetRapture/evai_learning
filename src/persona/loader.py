import json
from pathlib import Path

from common.errors import PersonaParseError
from common.paths import DATA_DIR
from persona.schema import PersonaData, validate_persona_dict


def discover_persona_files(data_dir: Path | None = None) -> list[Path]:
    target_dir = data_dir if data_dir is not None else DATA_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        return []
    files = [p for p in target_dir.glob("*.json") if p.is_file()]
    return sorted(files, key=lambda p: p.name)


def load_persona_file(file_path: Path) -> PersonaData:
    content = file_path.read_text(encoding="utf-8")
    try:
        raw_data = json.loads(content)
    except json.JSONDecodeError as err:
        raise PersonaParseError(
            source_file=file_path,
            line=err.lineno,
            column=err.colno,
            parser_message=err.msg,
        ) from err

    return validate_persona_dict(raw_data, source_file=file_path)


def load_all_personas(data_dir: Path | None = None) -> list[tuple[Path, PersonaData]]:
    return [(path, load_persona_file(path)) for path in discover_persona_files(data_dir)]


def inspect_persona_files(
    data_dir: Path | None = None,
) -> tuple[list[tuple[Path, PersonaData]], list[tuple[Path, Exception]]]:
    valid_personas: list[tuple[Path, PersonaData]] = []
    errors: list[tuple[Path, Exception]] = []

    for file_path in discover_persona_files(data_dir):
        try:
            valid_personas.append((file_path, load_persona_file(file_path)))
        except Exception as err:
            errors.append((file_path, err))

    return valid_personas, errors
