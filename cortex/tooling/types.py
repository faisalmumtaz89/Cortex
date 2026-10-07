"""Core tooling runtime types."""

from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Literal, Optional, Union


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
    # Response-side identity proof, attached by the client that actually
    # answered: {client_kind, reported_model, response_id, endpoint}. The
    # orchestrator verifies it against the requested target after every turn.
    provenance: Optional[Dict[str, Any]] = None


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
    # Verified per-turn provenance: set only after the response-side identity
    # was checked against the requested target (see tooling/provenance.py).
    provenance: Optional[Dict[str, Any]] = None
    provenance_verified: bool = False
    served_model_label: Optional[str] = None  # e.g. "openai:gpt-5.1"


class ReplyCutOffError(RuntimeError):
    """The model's reply stopped at a limit before it finished. The reply was
    verified, so what the turn produced (``parts``) stays in the conversation."""

    def __init__(self, message: str, parts: List[Dict[str, Any]]) -> None:
        super().__init__(message)
        self.parts = parts


class TurnInterruptedError(RuntimeError):
    """The user interrupted the running turn."""


class TurnInterrupt:
    """The interrupt signal of one turn.

    set() also shuts down the network connection of the model response the
    turn is reading (see watching()), so a read blocked while the model is
    still working ends at once instead of when the reply arrives.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._set = False
        self._stream: Any = None

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        with self._lock:
            self._set = True
            stream = self._stream
        if stream is not None:
            _shut_down(stream)

    @contextmanager
    def watching(self, stream: Any) -> Iterator[None]:
        """Read an SDK response stream inside this block; set() shuts it down.

        The stream is forgotten when the block ends, because its pooled
        connection may then carry the turn's next request. The block raises
        TurnInterruptedError if the turn was interrupted while reading.
        """
        with self._lock:
            self._stream = stream
            already_set = self._set
        if already_set:
            _shut_down(stream)
        try:
            yield
        except Exception as exc:
            if self._set:
                raise TurnInterruptedError() from exc
            raise
        finally:
            with self._lock:
                self._stream = None
        self.raise_if_set()

    def raise_if_set(self) -> None:
        if self._set:
            raise TurnInterruptedError()


def _shut_down(stream: Any) -> None:
    """Shut down the socket under an SDK response stream. Closing the stream
    alone does not wake a read blocked in another thread; shutting down the
    socket does."""
    raw = getattr(stream, "_raw_stream", stream)
    network_stream = raw.response.extensions.get("network_stream")
    sock = network_stream.get_extra_info("socket") if network_stream is not None else None
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
