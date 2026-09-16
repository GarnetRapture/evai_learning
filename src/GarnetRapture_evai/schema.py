"""Persona JSON schema and validation models for fff."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import PersonaSchemaError


class DialogueEntry(BaseModel):
    """Single utterance entry within dialogues (story or evertalk)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    speaker: str
    message: str


class Dialogues(BaseModel):
    """Container for story and evertalk dialogues."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    story: list[DialogueEntry] = Field(default_factory=list)
    evertalk: list[DialogueEntry] = Field(default_factory=list)


class PersonaComment(BaseModel):
    """External observation or remark made about the persona by another character."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    writer: str
    comment: str


class Profile(BaseModel):
    """Persona demographic and biographical profile."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    nick_name: str | None = None
    constellation: str | None = None
    union: str | None = None
    birthday: str | None = None
    height: int | float | None = None
    weight: int | float | None = None
    cv_ko: str | None = None
    cv_jp: str | None = None
    like: list[str] = Field(default_factory=list)
    dislike: list[str] = Field(default_factory=list)
    hobby: list[str] = Field(default_factory=list)
    speciality: list[str] = Field(default_factory=list)


class Personality(BaseModel):
    """Persona canonical personality description and greeting."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    description: str | None = None
    greeting: str | None = None


class PersonaData(BaseModel):
    """Canonical persona model derived from data/*.json files."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str
    name_en: str
    grade: str
    race: str
    class_: str = Field(default="", alias="class")
    sub_class: str = ""
    stat: str = ""

    profile: Profile = Field(default_factory=Profile)
    personality: Personality = Field(default_factory=Personality)
    speech_patterns: list[str] = Field(default_factory=list)
    comments: list[PersonaComment] = Field(default_factory=list)
    dialogues: Dialogues = Field(default_factory=Dialogues)


def validate_persona_dict(raw: dict[str, Any], source_file: Path) -> PersonaData:
    """Validate a raw dictionary against the PersonaData schema.

    Raises:
        PersonaSchemaError: If any required field is missing or has an invalid type.
    """
    try:
        return PersonaData.model_validate(raw)
    except ValidationError as err:
        # Extract the first error location and detail for clean diagnostic reporting
        first_err = err.errors()[0]
        field_path = " -> ".join(str(loc) for loc in first_err.get("loc", ()))
        expected = first_err.get("type", "valid structure")
        actual_issue = first_err.get("msg", str(first_err))
        raise PersonaSchemaError(
            source_file=source_file,
            field_path=field_path,
            expected_shape=expected,
            actual_issue=actual_issue,
        ) from err
