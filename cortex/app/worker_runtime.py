"""Worker runtime assembly for Cortex JSON-RPC mode."""

from __future__ import annotations

import json
import logging
import uuid
from io import TextIOBase
from typing import Any, Callable, Dict, TypeVar

from pydantic import BaseModel

from cortex.app.command_service import CommandService
from cortex.app.model_service import ModelService
from cortex.app.permission_service import PermissionService
from cortex.app.session_service import SessionService, _WorkerToolingBridge
from cortex.cloud import CloudCredentialStore, CloudModelCatalog, CloudRouter
from cortex.cloud.types import ActiveModelTarget, CloudProvider
from cortex.model_downloader import ModelDownloader
from cortex.protocol.events import EventEmitter
from cortex.protocol.rpc_server import RpcMethodError, StdioJsonRpcServer
from cortex.protocol.schema import (
    CloudAuthDeleteKeyParams,
    CloudAuthSaveKeyParams,
    CloudAuthStatusParams,
    CommandExecuteParams,
    HandshakeParams,
    ModelDeleteLocalParams,
    ModelListParams,
    ModelSelectParams,
    PermissionReplyParams,
    SessionCreateOrResumeParams,
    SessionInterruptParams,
    SessionSubmitUserInputParams,
)
from cortex.protocol.types import PROTOCOL_VERSION, EventType
from cortex.template_registry import TemplateRegistry
from cortex.tooling.orchestrator import ToolingOrchestrator

logger = logging.getLogger(__name__)

ParamsModelT = TypeVar("ParamsModelT", bound=BaseModel)


class _SilentConsole:
    """Suppress console output in worker mode."""

    def print(self, *args: object, **kwargs: object) -> None:
        return


class WorkerRuntime:
    """Create and run the worker-side JSON-RPC runtime."""

    def __init__(
        self,
        *,
        config,
        model_manager,
        gpu_validator,
        inference_engine,
        conversation_manager,
        rpc_stdin: TextIOBase | None = None,
        rpc_stdout: TextIOBase | None = None,
    ) -> None:
        self.config = config
        self.model_manager = model_manager
        self.inference_engine = inference_engine
        self.conversation_manager = conversation_manager

        self.rpc_server = StdioJsonRpcServer(stdin=rpc_stdin, stdout=rpc_stdout)
        self.event_emitter = EventEmitter(send=self.rpc_server.send_raw)

        self.credential_store = CloudCredentialStore()
        self.cloud_catalog = CloudModelCatalog()
        self.cloud_router = CloudRouter(config, self.credential_store)

        permission_timeout = int(getattr(config.tools, "tools_idle_timeout_seconds", 45) or 45)
        self.permission_service = PermissionService(timeout_seconds=permission_timeout)

        # Keep any optional template diagnostics away from worker transport output.
        self.template_registry = TemplateRegistry(console=_SilentConsole())

        self.tooling_bridge = _WorkerToolingBridge(
            config=config,
            inference_engine=inference_engine,
            model_manager=model_manager,
            conversation_manager=conversation_manager,
            template_registry=self.template_registry,
            cloud_router=self.cloud_router,
            permission_service=self.permission_service,
        )
        self.tooling_orchestrator = ToolingOrchestrator(cli=self.tooling_bridge)

        self.model_service = ModelService(
            config=config,
            model_manager=model_manager,
            gpu_validator=gpu_validator,
            cloud_router=self.cloud_router,
            credential_store=self.credential_store,
            cloud_catalog=self.cloud_catalog,
        )
        self.session_service = SessionService(
            config=config,
            conversation_manager=conversation_manager,
            tooling_orchestrator=self.tooling_orchestrator,
            model_service=self.model_service,
            tooling_bridge=self.tooling_bridge,
        )
        self.command_service = CommandService(
            model_service=self.model_service,
            clear_session=self.session_service.clear_session,
            save_session=self.session_service.save_session,
            model_downloader=ModelDownloader(config.model.model_path),
            template_registry=self.template_registry,
            inference_engine=self.inference_engine,
        )
        self._startup_notices: list[str] = self._restore_startup_target()
        self._startup_notices_emitted = False

        self._register_methods()

    def _restore_startup_target(self) -> list[str]:
        """Restore the previously active model target from persisted state."""
        notices: list[str] = []
        last_backend = str(self.config.get_state_value("last_used_backend", "local") or "local").strip().lower()

        if last_backend == "cloud" and getattr(self.config.cloud, "cloud_enabled", True):
            provider = str(self.config.get_state_value("last_used_cloud_provider", "") or "").strip()
            model_id = str(self.config.get_state_value("last_used_cloud_model", "") or "").strip()
            if provider and model_id:
                try:
                    result = self.model_service.select_cloud_model(provider=provider, model_id=model_id)
                except Exception as exc:  # pragma: no cover - defensive path
                    notices.append(f"Failed to restore cloud model: {exc}")
                else:
                    if bool(result.get("ok")):
                        notices.append(f"Restored cloud model: {provider}:{model_id}")
                    else:
                        notices.append(str(result.get("message", "Failed to restore cloud model.")))
                return notices

        local_model = str(self.config.model.last_used_model or self.config.model.default_model or "").strip()
        if local_model:
            result = self.model_service.select_local_model(local_model)
            if bool(result.get("ok")):
                notices.append(f"Restored local model: {self.model_service.get_active_model_label()}")
            else:
                notices.append(str(result.get("message", f"Failed to restore local model: {local_model}")))
                self.model_service.active_target = ActiveModelTarget.local(model_name=None)
            return notices

        cloud_router = getattr(self, "cloud_router", None)
        cloud_catalog = getattr(self, "cloud_catalog", None)
        if getattr(self.config.cloud, "cloud_enabled", True) and cloud_router is not None and cloud_catalog is not None:
            preferred_providers: list[str] = []
            last_provider = str(self.config.get_state_value("last_used_cloud_provider", "") or "").strip().lower()
            if last_provider:
                preferred_providers.append(last_provider)
            preferred_providers.extend(["openai", "anthropic"])

            seen: set[str] = set()
            for provider_name in preferred_providers:
                if provider_name in seen:
                    continue
                seen.add(provider_name)

                try:
                    provider_enum = CloudProvider.from_value(provider_name)
                except Exception:
                    continue

                try:
                    is_auth, _source = cloud_router.get_auth_status(provider_enum)
                except Exception:
                    continue
                if not is_auth:
                    continue

                default_key = f"cloud_default_{provider_enum.value}_model"
                configured_default = str(getattr(self.config.cloud, default_key, "") or "").strip()
                model_id = configured_default

                if not model_id:
                    for ref in cloud_catalog.list_models():
                        if ref.provider == provider_enum:
                            model_id = ref.model_id
                            break
                if not model_id:
                    continue

                try:
                    result = self.model_service.select_cloud_model(provider=provider_enum.value, model_id=model_id)
                except Exception as exc:  # pragma: no cover - defensive path
                    notices.append(f"Failed to restore cloud model: {exc}")
                    continue

                if bool(result.get("ok")):
                    notices.append(f"Restored cloud model: {provider_enum.value}:{model_id}")
                    return notices
                notices.append(str(result.get("message", f"Failed to restore cloud model: {provider_enum.value}:{model_id}")))

        self.model_service.active_target = ActiveModelTarget.local(model_name=None)
        notices.append("No model loaded. Use /model to select local or cloud.")
        return notices

    def _emit_event(self, *, session_id: str, event_type: EventType, payload: Dict[str, object]) -> None:
        envelope = self.event_emitter.emit(session_id=session_id, event_type=event_type, payload=payload)
        logger.debug(
            "event emitted session_id=%s seq=%s event_type=%s",
            envelope.session_id,
            envelope.seq,
            envelope.event_type,
        )

    def _register_typed_method(
        self,
        *,
        name: str,
        params_model: type[ParamsModelT],
        handler: Callable[[ParamsModelT], Any],
    ) -> None:
        """Register an RPC method while preserving handler parameter typing."""

        def _wrapper(params: BaseModel) -> Any:
            if not isinstance(params, params_model):
                raise RpcMethodError(
                    code=-32602,
                    message="Invalid params model",
                    data={"method": name, "expected": params_model.__name__, "received": type(params).__name__},
                )
            return handler(params)

        self.rpc_server.register(name, _wrapper)

    def _register_methods(self) -> None:
        self._register_typed_method(name="app.handshake", params_model=HandshakeParams, handler=self._rpc_handshake)
        self._register_typed_method(
            name="session.create_or_resume",
            params_model=SessionCreateOrResumeParams,
            handler=self._rpc_session_create_or_resume,
        )
        self._register_typed_method(
            name="session.submit_user_input",
            params_model=SessionSubmitUserInputParams,
            handler=self._rpc_session_submit_user_input,
        )
        self._register_typed_method(
            name="session.interrupt",
            params_model=SessionInterruptParams,
            handler=self._rpc_session_interrupt,
        )
        self._register_typed_method(
            name="permission.reply",
            params_model=PermissionReplyParams,
            handler=self._rpc_permission_reply,
        )
        self._register_typed_method(
            name="command.execute",
            params_model=CommandExecuteParams,
            handler=self._rpc_command_execute,
        )
        self._register_typed_method(name="model.list", params_model=ModelListParams, handler=self._rpc_model_list)
        self._register_typed_method(name="model.select", params_model=ModelSelectParams, handler=self._rpc_model_select)
        self._register_typed_method(
            name="model.delete_local",
            params_model=ModelDeleteLocalParams,
            handler=self._rpc_model_delete_local,
        )
        self._register_typed_method(
            name="cloud.auth.status",
            params_model=CloudAuthStatusParams,
            handler=self._rpc_cloud_auth_status,
        )
        self._register_typed_method(
            name="cloud.auth.save_key",
            params_model=CloudAuthSaveKeyParams,
            handler=self._rpc_cloud_auth_save_key,
        )
        self._register_typed_method(
            name="cloud.auth.delete_key",
            params_model=CloudAuthDeleteKeyParams,
            handler=self._rpc_cloud_auth_delete_key,
        )

    def _rpc_handshake(self, params: HandshakeParams):
        if params.protocol_version != PROTOCOL_VERSION:
            raise RpcMethodError(
                code=-32000,
                message="Protocol version mismatch",
                data={
                    "expected": PROTOCOL_VERSION,
                    "received": params.protocol_version,
                },
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "server_name": "cortex-worker",
            "supported_profiles": ["off", "read_only", "patch", "full"],
            "features": {
                "events": True,
                "permissions": True,
                "tooling": True,
                "worker_mode": True,
            },
        }

    def _rpc_session_create_or_resume(self, params: SessionCreateOrResumeParams):
        session_id = uuid.uuid4().hex
        result = self.session_service.create_or_resume(
            session_id=session_id,
            conversation_id=params.conversation_id,
        )
        self._emit_event(
            session_id=session_id,
            event_type="system.notice",
            payload={"message": "Session ready"},
        )
        if not self._startup_notices_emitted:
            self._startup_notices_emitted = True
            for message in self._startup_notices:
                self._emit_event(
                    session_id=session_id,
                    event_type="system.notice",
                    payload={"message": message},
                )
        return result

    def _rpc_session_submit_user_input(self, params: SessionSubmitUserInputParams):
        return self.session_service.submit_user_input(
            session_id=params.session_id,
            user_input=params.user_input,
            active_target_input=params.active_target,
            stop_sequences=params.stop_sequences,
            emit_event=self._emit_event,
        )

    def _rpc_session_interrupt(self, params: SessionInterruptParams):
        cancel = getattr(self.inference_engine, "cancel_generation", None)
        if callable(cancel):
            cancel()
        self._emit_event(
            session_id=params.session_id,
            event_type="session.status",
            payload={"status": "idle", "interrupted": True},
        )
        return {"ok": True}

    def _rpc_permission_reply(self, params: PermissionReplyParams):
        accepted = self.permission_service.reply(
            session_id=params.session_id,
            request_id=params.request_id,
            reply=params.reply,
        )
        if not accepted:
            raise RpcMethodError(
                code=-32001,
                message="Permission request not found or already resolved",
                data={"request_id": params.request_id},
            )
        return {"ok": True}

    def _rpc_command_execute(self, params: CommandExecuteParams):
        result = self.command_service.execute(session_id=params.session_id, command=params.command)

        notice_message = None
        if isinstance(result.get("message"), str) and result["message"]:
            notice_message = str(result["message"])
        elif "status" in result:
            notice_message = json.dumps(result["status"], ensure_ascii=True)
        elif "path" in result:
            notice_message = f"Saved conversation: {result['path']}"
        elif "auth" in result:
            notice_message = json.dumps(result["auth"], ensure_ascii=True)

        if notice_message:
            self._emit_event(
                session_id=params.session_id,
                event_type="system.notice",
                payload={"message": notice_message},
            )

        return result

    def _rpc_model_list(self, _params: ModelListParams):
        return self.model_service.list_models()

    def _rpc_model_select(self, params: ModelSelectParams):
        if params.backend == "cloud":
            if not params.provider or not params.model_id:
                raise RpcMethodError(
                    code=-32602,
                    message="Cloud model selection requires provider and model_id",
                )
            return self.model_service.select_cloud_model(provider=params.provider, model_id=params.model_id)
        if not params.local_model:
            raise RpcMethodError(code=-32602, message="Local model selection requires local_model")
        return self.model_service.select_local_model(params.local_model)

    def _rpc_model_delete_local(self, params: ModelDeleteLocalParams):
        return self.model_service.delete_local_model(params.model_name)

    def _rpc_cloud_auth_status(self, params: CloudAuthStatusParams):
        return self.model_service.auth_status(params.provider_enum())

    def _rpc_cloud_auth_save_key(self, params: CloudAuthSaveKeyParams):
        return self.model_service.auth_save_key(params.provider_enum(), params.api_key)

    def _rpc_cloud_auth_delete_key(self, params: CloudAuthDeleteKeyParams):
        return self.model_service.auth_delete_key(params.provider_enum())

    def run(self) -> None:
        logger.info("Starting Cortex worker RPC server protocol=%s", PROTOCOL_VERSION)
        self.rpc_server.run_forever()

    def __del__(self) -> None:
        return
