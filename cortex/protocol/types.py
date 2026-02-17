"""Core protocol types for Cortex worker/frontend communication."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "1.0.0"

JsonRpcId = Union[str, int]

EventType = Literal[
    "session.status",
    "message.updated",
    "message.part.updated",
    "permission.asked",
    "permission.replied",
    "session.error",
    "system.notice",
]

ToolExecutionState = Literal["pending", "running", "completed", "error"]


class RpcRequest(BaseModel):
    """Incoming JSON-RPC 2.0 request."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId
    method: str = Field(min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)


class RpcError(BaseModel):
    """JSON-RPC error object."""

    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


class RpcSuccessResponse(BaseModel):
    """JSON-RPC success response."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId
    result: Dict[str, Any]


class RpcErrorResponse(BaseModel):
    """JSON-RPC error response."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[JsonRpcId] = None
    error: RpcError


class EventEnvelope(BaseModel):
    """Protocol event envelope emitted by the worker."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    seq: int = Field(ge=1)
    ts_ms: int = Field(ge=0)
    event_type: EventType
    payload: Dict[str, Any] = Field(default_factory=dict)


class EventFrame(BaseModel):
    """Wire frame for emitted events."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["event"] = "event"
    params: EventEnvelope
