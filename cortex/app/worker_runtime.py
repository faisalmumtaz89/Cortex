"""Worker runtime assembly for Cortex JSON-RPC mode."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import uuid
from io import TextIOBase
from typing import Any, Callable, Dict, TypeVar

from pydantic import BaseModel

from cortex.app.command_service import CommandService
from cortex.app.model_service import ModelService
from cortex.app.permission_service import PermissionService
from cortex.app.session_service import SessionService, _WorkerToolingBridge
from cortex.app.update_service import UpdateService
from cortex.cloud import CloudCredentialStore, CloudModelCatalog, CloudRouter
from cortex.cloud.types import ActiveModelTarget, CloudProvider
from cortex.protocol.events import EventEmitter
from cortex.protocol.rpc_server import RpcMethodError, StdioJsonRpcServer
from cortex.protocol.schema import (
    CloudAuthDeleteKeyParams,
    CloudAuthSaveKeyParams,
    CloudAuthStatusParams,
    CommandExecuteParams,
    HandshakeParams,
    ModelListParams,
    ModelSelectParams,
    PermissionReplyParams,
    SessionCreateOrResumeParams,
    SessionInterruptParams,
    SessionSubmitUserInputParams,
)
from cortex.protocol.types import PROTOCOL_VERSION, EventType
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
        conversation_manager,
        rpc_stdin: TextIOBase | None = None,
        rpc_stdout: TextIOBase | None = None,
    ) -> None:
        self.config = config
        self.conversation_manager = conversation_manager

        self.rpc_server = StdioJsonRpcServer(stdin=rpc_stdin, stdout=rpc_stdout)
        self.event_emitter = EventEmitter(send=self.rpc_server.send_raw)

        self.credential_store = CloudCredentialStore()
        self.cloud_catalog = CloudModelCatalog()
        self.cloud_router = CloudRouter(config, self.credential_store)

        permission_timeout = int(getattr(config.tools, "tools_idle_timeout_seconds", 45) or 45)
        self.permission_service = PermissionService(timeout_seconds=permission_timeout)

        self.tooling_bridge = _WorkerToolingBridge(
            config=config,
            conversation_manager=conversation_manager,
            cloud_router=self.cloud_router,
            permission_service=self.permission_service,
        )
        self.tooling_orchestrator = ToolingOrchestrator(cli=self.tooling_bridge)

        self.model_service = ModelService(
            config=config,
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
        self.update_service = UpdateService()
        self.command_service = CommandService(
            model_service=self.model_service,
            clear_session=self.session_service.clear_session,
            save_session=self.session_service.save_session,
            update_service=self.update_service,
        )
        # Background self-update (the wheel download and install take a
        # while; the command returns immediately and this tracks the update).
        self._update_task_lock = threading.Lock()
        self._active_update_thread: threading.Thread | None = None
        # Daily update check: the daemon thread computes a notice; the first
        # session emits it exactly once (whichever side finishes second does
        # the emit — see _emit_update_notice_if_ready).
        self._update_notice_lock = threading.Lock()
        self._pending_update_notice: str | None = None
        self._update_notice_session_id: str | None = None
        self._update_notice_emitted = False
        self._startup_notices: list[str] = self._restore_startup_target()
        if os.environ.get("CORTEX_SCRIPTED_MODEL", "").strip():
            # A scripted override answers EVERY turn with canned output — make
            # that impossible to miss in a real session.
            self._startup_notices.append(
                "! SCRIPTED MODEL ACTIVE — responses are canned (CORTEX_SCRIPTED_MODEL is set)"
            )
        self._startup_notices_emitted = False

        self._register_methods()

        if self._auto_update_check_enabled():
            # Fire-and-forget daily release check (opt-out via the
            # auto_update_check config key). Daemon thread: startup must
            # never wait on the network, and check failures stay silent.
            threading.Thread(
                target=self._run_startup_update_check,
                name="cortex-update-check",
                daemon=True,
            ).start()

    def _auto_update_check_enabled(self) -> bool:
        # Missing attribute (stub configs in tests) counts as DISABLED: the
        # check may only run when a real config explicitly carries the key —
        # a fake must never be able to trigger a network probe.
        return bool(getattr(getattr(self.config, "system", None), "auto_update_check", False))

    def _run_startup_update_check(self) -> None:
        """Compute the update notice off-thread and hand it to the notice
        gate. Never raises; never blocks startup."""
        try:
            notice = self.update_service.startup_notice()
        except Exception:  # pragma: no cover - defensive path
            logger.debug("startup update check failed", exc_info=True)
            return
        if not notice:
            return
        with self._update_notice_lock:
            self._pending_update_notice = notice
        self._emit_update_notice_if_ready()

    def _emit_update_notice_if_ready(self) -> None:
        """Emit the update notice exactly once, as soon as BOTH a session and
        the check result exist (either side may finish first)."""
        with self._update_notice_lock:
            notice = self._pending_update_notice
            session_id = self._update_notice_session_id
            if not notice or not session_id or self._update_notice_emitted:
                return
            self._update_notice_emitted = True
        # origin marks this as OUT-OF-BAND for the frontend: it is emitted
        # exactly once, asynchronously — the store must transcribe it even if
        # a slash command happens to be in flight (command notices are
        # dropped as duplicates of the command result; this one is not).
        self._emit_event(
            session_id=session_id,
            event_type="system.notice",
            payload={"message": notice, "origin": "update-check"},
        )

    def _restore_startup_target(self) -> list[str]:
        """Restore the previously active model target from persisted state."""
        notices: list[str] = []
        last_provider = str(
            self.config.get_state_value("last_used_cloud_provider", "") or ""
        ).strip()
        last_model = str(self.config.get_state_value("last_used_cloud_model", "") or "").strip()

        if last_provider and last_model:
            try:
                result = self.model_service.select_cloud_model(
                    provider=last_provider, model_id=last_model
                )
            except Exception as exc:  # pragma: no cover - defensive path
                notices.append(f"Failed to restore model: {exc}")
            else:
                if bool(result.get("ok")):
                    notices.append(f"Restored {last_provider}:{last_model}")
                    return notices
                notices.append(str(result.get("message", "Failed to restore model.")))

        cloud_router = getattr(self, "cloud_router", None)
        cloud_catalog = getattr(self, "cloud_catalog", None)
        if cloud_router is not None and cloud_catalog is not None:
            preferred_providers: list[str] = []
            if last_provider:
                preferred_providers.append(last_provider.lower())
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
                    result = self.model_service.select_cloud_model(
                        provider=provider_enum.value, model_id=model_id
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    notices.append(f"Failed to restore model: {exc}")
                    continue

                if bool(result.get("ok")):
                    notices.append(f"Restored {provider_enum.value}:{model_id}")
                    return notices
                notices.append(
                    str(
                        result.get(
                            "message",
                            f"Failed to restore model: {provider_enum.value}:{model_id}",
                        )
                    )
                )

        self.model_service.active_target = ActiveModelTarget()
        self._set_runtime_state_if_supported("last_used_cloud_provider", "")
        self._set_runtime_state_if_supported("last_used_cloud_model", "")
        notices.append("No model loaded. Pick one with /model, or add a key with /login.")
        return notices

    def _set_runtime_state_if_supported(self, key: str, value: object) -> None:
        setter = getattr(self.config, "set_state_value", None)
        if callable(setter):
            try:
                setter(key, value)
            except Exception:  # pragma: no cover - defensive path
                logger.debug("Failed to persist runtime state key=%s", key, exc_info=True)

    def _emit_event(
        self, *, session_id: str, event_type: EventType, payload: Dict[str, object]
    ) -> None:
        envelope = self.event_emitter.emit(
            session_id=session_id, event_type=event_type, payload=payload
        )
        logger.debug(
            "event emitted session_id=%s seq=%s event_type=%s",
            envelope.session_id,
            envelope.seq,
            envelope.event_type,
        )

    @staticmethod
    def _notice_message_for_result(result: Dict[str, object]) -> str | None:
        if isinstance(result.get("message"), str) and result["message"]:
            return str(result["message"])
        if "status" in result:
            return json.dumps(result["status"], ensure_ascii=True)
        if "path" in result:
            return f"Saved conversation: {result['path']}"
        if "auth" in result:
            return json.dumps(result["auth"], ensure_ascii=True)
        return None

    # ---- /update background operations -----------------------------------

    def _update_in_progress(self) -> bool:
        with self._update_task_lock:
            return (
                self._active_update_thread is not None and self._active_update_thread.is_alive()
            )

    def _run_background_update(self, *, session_id: str) -> None:
        """Run the self-update off the RPC thread as a live system progress
        message (kind "engine-update"): one message, installer output lines
        as phases, resolved in place by the final ready/failed frame."""
        progress_message_id = f"engine-update:{uuid.uuid4().hex}"
        headline = "Updating Cortex…"

        def _emit(*, content: str, phase: str, final: bool) -> None:
            self._emit_event(
                session_id=session_id,
                event_type="message.updated",
                payload={
                    "message_id": progress_message_id,
                    "role": "system",
                    "content": content,
                    "final": final,
                    "progress": {
                        "kind": "engine-update",
                        "repo_id": "cortex",
                        "phase": phase,
                    },
                },
            )

        def _progress_callback(payload: Dict[str, object]) -> None:
            phase = str(payload.get("phase", "") or "").strip()
            if phase:
                _emit(content=headline, phase=phase, final=False)

        try:
            _emit(content=headline, phase="starting", final=False)
            result = self.command_service.execute(
                session_id=session_id,
                command="/update cortex",
                progress_callback=_progress_callback,
            )
            ok = bool(result.get("ok"))
            message = str(result.get("message", "")) or (
                f"Cortex update {'complete' if ok else 'failed'}."
            )
            # One operation = one transcript message (resolved in place).
            _emit(content=message, phase="ready" if ok else "failed", final=True)
        except Exception as exc:  # pragma: no cover - defensive path
            _emit(content=f"Cortex update failed: {exc}", phase="failed", final=True)

    def _start_background_update(self, *, session_id: str) -> bool:
        with self._update_task_lock:
            if self._active_update_thread is not None and self._active_update_thread.is_alive():
                return False
            thread = threading.Thread(
                target=self._run_background_update,
                kwargs={"session_id": session_id},
                name="cortex-self-update",
                daemon=True,
            )
            self._active_update_thread = thread
            thread.start()
            return True

    def _intercept_update(self, *, session_id: str) -> Dict[str, object]:
        """/update cortex → background operation, one at a time. The quick
        no-op answers (up to date / no releases / probe failure) resolve
        synchronously — a background narration for nothing would be noise.
        """
        if self._update_in_progress():
            return {
                "ok": False,
                "message": "Cortex update already in progress — wait for it to finish.",
            }

        plan = self.update_service.plan_cortex_update()
        if not bool(plan.get("ok")) or not bool(plan.get("update_available")):
            # Nothing to do (or the release feed is unreachable): answer
            # synchronously with the plan's message.
            message = str(plan.get("message", ""))
            self._emit_event(
                session_id=session_id,
                event_type="system.notice",
                payload={"message": message},
            )
            return {"ok": bool(plan.get("ok")), "message": message}

        if not self._start_background_update(session_id=session_id):
            return {"ok": False, "message": "Another update is already running."}
        # Terse confirmation only: the transcript's live "Updating …" row
        # solely owns update state.
        message = str(plan.get("message", "Updating Cortex…"))
        return {"ok": True, "message": message, "background": True, "repo_id": "cortex"}

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
                    data={
                        "method": name,
                        "expected": params_model.__name__,
                        "received": type(params).__name__,
                    },
                )
            return handler(params)

        self.rpc_server.register(name, _wrapper)

    def _register_methods(self) -> None:
        self._register_typed_method(
            name="app.handshake", params_model=HandshakeParams, handler=self._rpc_handshake
        )
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
        self._register_typed_method(
            name="model.list", params_model=ModelListParams, handler=self._rpc_model_list
        )
        self._register_typed_method(
            name="model.select", params_model=ModelSelectParams, handler=self._rpc_model_select
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
            "supported_profiles": ["off", "read_only", "edit", "full"],
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
        # One compact startup notice instead of a stack of boxes.
        notice_parts = ["Session ready"]
        if not self._startup_notices_emitted:
            self._startup_notices_emitted = True
            notice_parts.extend(self._startup_notices)
        self._emit_event(
            session_id=session_id,
            event_type="system.notice",
            payload={"message": " · ".join(notice_parts)},
        )
        with self._update_notice_lock:
            if self._update_notice_session_id is None:
                self._update_notice_session_id = session_id
        self._emit_update_notice_if_ready()
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
        interrupted = self.session_service.request_interrupt(params.session_id)
        return {"ok": True, "interrupted": interrupted}

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
        raw_command = str(params.command or "").strip()
        command_parts = raw_command.split(maxsplit=1)
        command_keyword = command_parts[0].lower() if command_parts else ""
        command_args = command_parts[1] if len(command_parts) > 1 else ""
        if command_keyword == "/update" and command_args.strip().lower() == "cortex":
            # Background op (see _intercept_update); bare /update and
            # /update status fall through to the synchronous handler.
            return self._intercept_update(session_id=params.session_id)

        self._emit_event(
            session_id=params.session_id,
            event_type="session.status",
            payload={"status": "busy"},
        )

        try:
            result = self.command_service.execute(
                session_id=params.session_id, command=params.command
            )
        finally:
            # The background update never owns session.status, so the sync
            # command's busy is always ours to clear.
            self._emit_event(
                session_id=params.session_id,
                event_type="session.status",
                payload={"status": "idle"},
            )

        notice_message = self._notice_message_for_result(result)
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
        if not params.provider or not params.model_id:
            raise RpcMethodError(
                code=-32602,
                message="Model selection requires provider and model_id",
            )
        return self.model_service.select_cloud_model(
            provider=params.provider, model_id=params.model_id
        )

    def _rpc_cloud_auth_status(self, params: CloudAuthStatusParams):
        return self.model_service.auth_status(params.provider_enum())

    def _rpc_cloud_auth_save_key(self, params: CloudAuthSaveKeyParams):
        return self.model_service.auth_save_key(params.provider_enum(), params.api_key)

    def _rpc_cloud_auth_delete_key(self, params: CloudAuthDeleteKeyParams):
        return self.model_service.auth_delete_key(params.provider_enum())

    def run(self) -> None:
        logger.info("Starting Cortex worker RPC server protocol=%s", PROTOCOL_VERSION)

        # Death by signal (SIGTERM from the sidecar's exit path, SIGHUP when
        # the user closes the terminal window, stray SIGINT) must still reap
        # an in-flight update — default signal handling would skip both the
        # finally below and atexit.
        def _terminate(signum, _frame) -> None:
            logger.info("Worker received signal %s — shutting down", signum)
            try:
                # An in-flight installer (own process group) must not be
                # orphaned half-applied — reap it and its temp files first
                # (bounded), since os._exit skips every finally/atexit.
                self.update_service.shutdown()
            finally:
                os._exit(0)

        for signame in ("SIGTERM", "SIGHUP", "SIGINT"):
            signum = getattr(signal, signame, None)
            if signum is not None:
                try:
                    signal.signal(signum, _terminate)
                except (ValueError, OSError):
                    pass  # non-main thread / unsupported: EOF path still covers us

        try:
            self.rpc_server.run_forever()
        finally:
            # Quitting Cortex must never orphan an in-flight installer (the
            # update thread is a daemon; os._exit skips its finally cleanup).
            self.update_service.shutdown()
            # In-flight turn threads (non-daemon) would otherwise keep this
            # process alive long after stdin EOF; everything that must be
            # persisted is already flushed per-turn.
            os._exit(0)

    def __del__(self) -> None:
        return
