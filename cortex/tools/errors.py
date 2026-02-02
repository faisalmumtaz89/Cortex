"""Tooling error types."""


class ToolError(Exception):
    """Base error for tool execution failures."""


class ValidationError(ToolError):
    """Raised when tool arguments or inputs are invalid."""
