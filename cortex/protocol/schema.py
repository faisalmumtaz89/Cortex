"""Method schemas for Cortex JSON-RPC protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from cortex.cloud.types import CloudProvider
from cortex.protocol.types import PROTOCOL_VERSION


class ActiveTargetInput(BaseModel):
    """Serialized active model target from frontend."""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(pattern=r"^(local|cloud)$")
    local_model: Optional[str] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None


class HandshakeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = Field(min_length=1)
    client_name: Optional[str] = None


class HandshakeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = PROTOCOL_VERSION
    server_name: str = "cortex-worker"
    supported_profiles: List[str] = ["off", "read_only", "patch", "full"]
    features: Dict[str, bool] = {"events": True, "permissions": True, "tooling": True}


class SessionCreateOrResumeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: Optional[str] = None
    title: Optional[str] = None


class SessionCreateOrResumeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    conversation_id: str
    restored: bool = False
    active_model_label: str


class SessionSubmitUserInputParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    active_target: Optional[ActiveTargetInput] = None
    stop_sequences: Optional[List[str]] = None


class SessionSubmitUserInputResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    assistant_text: str
    token_count: int
    elapsed_seconds: float


class SessionInterruptParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)


class PermissionReplyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    reply: str = Field(pattern=r"^(allow_once|allow_always|reject)$")


class CommandExecuteParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    command: str = Field(min_length=1)


class ModelSelectParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = Field(pattern=r"^(local|cloud)$")
    local_model: Optional[str] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None


class ModelDeleteLocalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)


class CloudAuthStatusParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)

    def provider_enum(self) -> CloudProvider:
        return CloudProvider.from_value(self.provider)


class CloudAuthSaveKeyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    api_key: str = Field(min_length=1)

    def provider_enum(self) -> CloudProvider:
        return CloudProvider.from_value(self.provider)


class CloudAuthDeleteKeyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)

    def provider_enum(self) -> CloudProvider:
        return CloudProvider.from_value(self.provider)


class ModelListParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


METHOD_PARAM_MODELS: Dict[str, Type[BaseModel]] = {
    "app.handshake": HandshakeParams,
    "session.create_or_resume": SessionCreateOrResumeParams,
    "session.submit_user_input": SessionSubmitUserInputParams,
    "session.interrupt": SessionInterruptParams,
    "permission.reply": PermissionReplyParams,
    "command.execute": CommandExecuteParams,
    "model.list": ModelListParams,
    "model.select": ModelSelectParams,
    "model.delete_local": ModelDeleteLocalParams,
    "cloud.auth.status": CloudAuthStatusParams,
    "cloud.auth.save_key": CloudAuthSaveKeyParams,
    "cloud.auth.delete_key": CloudAuthDeleteKeyParams,
}


def parse_method_params(method: str, params: Dict[str, Any]) -> BaseModel:
    """Validate params for a method and return typed model."""
    model = METHOD_PARAM_MODELS.get(method)
    if model is None:
        raise ValueError(f"Unsupported RPC method: {method}")
    return model.model_validate(params)
