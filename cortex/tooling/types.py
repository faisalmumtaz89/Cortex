"""Core tooling runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class ToolExecutionState(str, Enum):
    """Lifecycle state for a tool invocation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class PermissionAction(str, Enum):
    """Permission action for rule evaluation."""

    ASK = "ask"
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRule:
    """Permission rule with wildcard path pattern."""

    permission: str
    pattern: str
    action: PermissionAction


@dataclass(frozen=True)
class ToolSpec:
    """Tool contract exposed to model providers."""

    name: str
    description: str
    parameters: Dict[str, Any]
    permission: str


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Normalized tool result payload."""

    id: str
    name: str
    state: ToolExecutionState
    ok: bool
    output: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextDeltaEvent:
    type: Literal["text_delta"] = "text_delta"
    delta: str = ""


@dataclass(frozen=True)
class ReasoningDeltaEvent:
    type: Literal["reasoning_delta"] = "reasoning_delta"
    delta: str = ""


@dataclass(frozen=True)
class ToolCallEvent:
    type: Literal["tool_call"] = "tool_call"
    call: ToolCall = field(default_factory=lambda: ToolCall(id="", name="", arguments={}))


@dataclass(frozen=True)
class ToolResultEvent:
    type: Literal["tool_result"] = "tool_result"
    result: ToolResult = field(
        default_factory=lambda: ToolResult(
            id="",
            name="",
            state=ToolExecutionState.PENDING,
            ok=False,
        )
    )


@dataclass(frozen=True)
class ErrorEvent:
    type: Literal["error"] = "error"
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinishEvent:
    type: Literal["finish"] = "finish"
    reason: str = "stop"


ModelEvent = Union[
    TextDeltaEvent,
    ReasoningDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    ErrorEvent,
    FinishEvent,
]


@dataclass
class AssistantTurnResult:
    """Structured result for one assistant turn."""

    text: str
    parts: List[Dict[str, Any]] = field(default_factory=list)
    token_count: int = 0
    elapsed_seconds: float = 0.0
    first_token_latency_seconds: Optional[float] = None
