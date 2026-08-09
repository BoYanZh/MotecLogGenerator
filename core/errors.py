"""Domain exceptions for log parsing and conversion."""


class LogParseError(Exception):
    """Raised when a telemetry log cannot be parsed."""


class InvalidLogFormatError(LogParseError):
    """Raised when the file does not match the expected format structure."""


class MissingDependencyError(LogParseError):
    """Raised when an optional runtime dependency is unavailable (e.g. cantools)."""
