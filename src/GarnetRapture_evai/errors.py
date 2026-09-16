"""Project-specific exception hierarchy for fff."""

from pathlib import Path


class FFFError(Exception):
    """Base exception for all fff errors."""


class ConfigurationError(FFFError):
    """Raised when project configuration is missing, invalid, or cannot be parsed."""

    def __init__(self, message: str, config_path: Path | None = None) -> None:
        self.config_path = config_path
        if config_path:
            super().__init__(f"[{config_path}] {message}")
        else:
            super().__init__(message)


class PersonaParseError(FFFError):
    """Raised when a persona JSON file cannot be decoded as valid JSON."""

    def __init__(
        self,
        source_file: Path,
        line: int,
        column: int,
        parser_message: str,
    ) -> None:
        self.source_file = source_file
        self.line = line
        self.column = column
        self.parser_message = parser_message
        super().__init__(
            f"Failed to parse JSON in {source_file} at line {line}, col {column}: {parser_message}"
        )


class PersonaSchemaError(FFFError):
    """Raised when a persona JSON does not conform to the required structural invariants."""

    def __init__(
        self,
        source_file: Path,
        field_path: str,
        expected_shape: str,
        actual_issue: str,
    ) -> None:
        self.source_file = source_file
        self.field_path = field_path
        self.expected_shape = expected_shape
        self.actual_issue = actual_issue
        super().__init__(
            f"Schema violation in {source_file} at '{field_path}': "
            f"expected {expected_shape}, got {actual_issue}"
        )


class ModelDirectoryError(FFFError):
    """Raised when local base model directory or required model assets are invalid or missing."""

    def __init__(self, model_path: Path, reason: str) -> None:
        self.model_path = model_path
        self.reason = reason
        super().__init__(f"Model asset check failed at {model_path}: {reason}")


class EnvironmentValidationError(FFFError):
    """Raised when runtime environment requirements (e.g. PyTorch, CUDA, libraries) are not met."""

    def __init__(self, reason: str, details: dict[str, str] | None = None) -> None:
        self.reason = reason
        self.details = details or {}
        super().__init__(f"Environment validation failed: {reason}")
