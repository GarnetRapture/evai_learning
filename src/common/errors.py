from pathlib import Path


class EvaiError(Exception):
    pass


class ConfigurationError(EvaiError):
    def __init__(self, message: str, config_path: Path | None = None) -> None:
        self.config_path = config_path
        if config_path:
            super().__init__(f"[{config_path}] {message}")
        else:
            super().__init__(message)


class PersonaParseError(EvaiError):
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


class PersonaSchemaError(EvaiError):
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
