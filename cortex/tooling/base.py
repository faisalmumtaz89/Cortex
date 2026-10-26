"""Base abstractions for tooling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from cortex.tooling.types import ToolResult, ToolSpec, TurnInterrupt


@dataclass
class ToolContext:
    """Execution context for builtin tools."""

    repo_root: Path
    session_id: str
    call_id: str
    # Set when the user interrupts the turn; long-running tools stop on it.
    interrupt: TurnInterrupt = field(default_factory=TurnInterrupt)


class ToolExecutionError(RuntimeError):
    """Tool execution-level error."""


ToolArgsT = TypeVar("ToolArgsT", bound=BaseModel)


class BaseTool(Generic[ToolArgsT]):
    """Base class for validated tooling."""

    name: str = ""
    description: str = ""
    permission: str = ""
    args_model: Type[ToolArgsT]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
            permission=self.permission,
        )

    def validate_args(self, arguments: Dict[str, Any]) -> ToolArgsT:
        try:
            return self.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            raise ToolExecutionError(f"invalid arguments for {self.name}: {exc}") from exc

    def execute(self, *, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Validate and execute tool call."""
        parsed = self.validate_args(arguments)
        return self.run(parsed, context)

    def run(self, parsed_args: ToolArgsT, context: ToolContext) -> ToolResult:
        """Implement in subclasses."""
        raise NotImplementedError

    def preview(self, parsed_args: ToolArgsT, context: ToolContext) -> Optional[Dict[str, Any]]:
        """The diff this call would make, computed without changing anything,
        for its permission prompt; None when there is none to show."""
        return None
