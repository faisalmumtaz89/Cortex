"""Tool registry and execution entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.builtin import ApplyPatchTool, BashTool, ListDirTool, ReadFileTool, SearchTool
from cortex.tooling.types import ToolCall, ToolExecutionState, ToolResult, ToolSpec

_TOOL_PROFILE_ORDER = {
    "off": 0,
    "read_only": 1,
    "patch": 2,
    "full": 3,
}


class ToolRegistry:
    """Profile-aware tool registry."""

    def __init__(self, *, repo_root: Path, profile: str = "off"):
        self.repo_root = repo_root.resolve()
        self.profile = profile if profile in _TOOL_PROFILE_ORDER else "off"

        self._tools: Dict[str, BaseTool] = {}
        self._register_defaults()

    def _enabled_for_profile(self, required_profile: str) -> bool:
        return _TOOL_PROFILE_ORDER[self.profile] >= _TOOL_PROFILE_ORDER[required_profile]

    def _register(self, tool: BaseTool, required_profile: str) -> None:
        if self._enabled_for_profile(required_profile):
            self._tools[tool.name] = tool

    def _register_defaults(self) -> None:
        self._register(ListDirTool(), "read_only")
        self._register(ReadFileTool(), "read_only")
        self._register(SearchTool(), "read_only")
        self._register(ApplyPatchTool(), "patch")
        self._register(BashTool(), "full")

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def specs(self) -> List[ToolSpec]:
        return [self._tools[name].spec() for name in self.names()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def permission_for(self, name: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return name
        return tool.permission

    def execute(self, *, call: ToolCall, session_id: str) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                id=call.id,
                name=call.name,
                state=ToolExecutionState.ERROR,
                ok=False,
                error=f"unknown tool: {call.name}",
            )

        context = ToolContext(
            repo_root=self.repo_root,
            session_id=session_id,
            call_id=call.id,
        )

        try:
            result = tool.execute(arguments=call.arguments, context=context)
        except ToolExecutionError as exc:
            return ToolResult(
                id=call.id,
                name=call.name,
                state=ToolExecutionState.ERROR,
                ok=False,
                error=str(exc),
            )
        except Exception as exc:  # pragma: no cover - final safety net
            return ToolResult(
                id=call.id,
                name=call.name,
                state=ToolExecutionState.ERROR,
                ok=False,
                error=f"tool execution failed: {exc}",
            )

        return result
