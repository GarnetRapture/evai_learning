"""Deterministic, non-recursive loader for data/*.json persona files."""

import json
from pathlib import Path

from .errors import PersonaParseError
from .paths import DATA_DIR
from .schema import PersonaData, validate_persona_dict


def discover_persona_files(data_dir: Path | None = None) -> list[Path]:
    """Discover all canonical persona JSON files non-recursively.

    Returns deterministic file list sorted by filename.
    """
    target_dir = data_dir if data_dir is not None else DATA_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        return []
    # Strict non-recursive discovery of *.json files directly under target_dir
    files = [p for p in target_dir.glob("*.json") if p.is_file()]
    return sorted(files, key=lambda p: p.name)


def load_persona_file(file_path: Path) -> PersonaData:
    """Load and validate a single persona JSON file.

    Raises:
        PersonaParseError: If JSON syntax is invalid.
        PersonaSchemaError: If required schema invariants are violated.
        FileNotFoundError: If the file does not exist.
    """
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
    """Discover and load all persona files, raising on the first error encountered."""
    files = discover_persona_files(data_dir)
    results: list[tuple[Path, PersonaData]] = []
    for file_path in files:
        persona = load_persona_file(file_path)
        results.append((file_path, persona))
    return results


def inspect_persona_files(
    data_dir: Path | None = None,
) -> tuple[list[tuple[Path, PersonaData]], list[tuple[Path, Exception]]]:
    """Inspect all persona files, separating valid personas from errors without crashing."""
    files = discover_persona_files(data_dir)
    valid_personas: list[tuple[Path, PersonaData]] = []
    errors: list[tuple[Path, Exception]] = []

    for file_path in files:
        try:
            persona = load_persona_file(file_path)
            valid_personas.append((file_path, persona))
        except Exception as err:
            errors.append((file_path, err))

    return valid_personas, errors
