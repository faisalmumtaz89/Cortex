"""Worker runtime assembly for Cortex JSON-RPC mode."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import uuid
from io import TextIOBase
from typing import Any, Callable, Dict, TypeVar

from pydantic import BaseModel

from cortex.app.command_service import _CLOUD_LOGIN_PROVIDERS, CommandService
from cortex.app.model_service import ModelService
from cortex.app.permission_service import PermissionService
from cortex.app.session_service import SessionService, _WorkerToolingBridge
from cortex.cloud import CloudCredentialStore, CloudModelCatalog, CloudRouter
from cortex.cloud.types import ActiveModelTarget, CloudProvider
from cortex.lumen_runtime import (
    SWITCH_IN_FLIGHT_PREFIX,
    LumenRuntime,
    partial_download_bytes,
)
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
        gpu_validator,
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

        lumen_cfg = getattr(config, "lumen", None)
        self.lumen_runtime = LumenRuntime(
            binary=str(getattr(lumen_cfg, "lumen_binary", "lumen")),
            server_binary=str(getattr(lumen_cfg, "lumen_server_binary", "lumen-server")),
            port=int(getattr(lumen_cfg, "lumen_port", 0) or 0),
            context_len=int(getattr(lumen_cfg, "lumen_context_len", 0) or 0),
            startup_timeout_seconds=int(
                getattr(lumen_cfg, "lumen_startup_timeout_seconds", 180) or 180
            ),
            log_level=str(getattr(lumen_cfg, "lumen_log_level", "warn")),
        )
        # Local turns route through the OpenAI-compatible endpoint of the
        # managed lumen-server process.
        self.cloud_router.lumen_base_url = self.lumen_runtime.base_url

        permission_timeout = int(getattr(config.tools, "tools_idle_timeout_seconds", 45) or 45)
        self.permission_service = PermissionService(timeout_seconds=permission_timeout)

        self.tooling_bridge = _WorkerToolingBridge(
            config=config,
            conversation_manager=conversation_manager,
            cloud_router=self.cloud_router,
            lumen_runtime=self.lumen_runtime,
            permission_service=self.permission_service,
        )
        self.tooling_orchestrator = ToolingOrchestrator(cli=self.tooling_bridge)

        self.model_service = ModelService(
            config=config,
            lumen_runtime=self.lumen_runtime,
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
            lumen_runtime=self.lumen_runtime,
        )
        self._download_task_lock = threading.Lock()
        self._active_download_thread: threading.Thread | None = None
        self._download_cancel_event = threading.Event()
        self._active_download_repo_id: str | None = None
        # Background model boots (lumen-server loads take 15-60s; the command
        # returns immediately and these track the in-flight load).
        self._boot_task_lock = threading.Lock()
        self._active_boot_thread: threading.Thread | None = None
        self._active_boot_selector: str | None = None
        # Monotonic model-choice counter: a background boot/download only
        # activates its model if the user hasn't picked something newer.
        self._model_choice_seq = 0
        self._startup_notices: list[str] = self._restore_startup_target()
        if os.environ.get("CORTEX_SCRIPTED_MODEL", "").strip():
            # A scripted override answers EVERY turn with canned output — make
            # that impossible to miss in a real session.
            self._startup_notices.append(
                "! SCRIPTED MODEL ACTIVE — responses are canned (CORTEX_SCRIPTED_MODEL is set)"
            )
        self._startup_notices_emitted = False

        self._register_methods()

    def _restore_startup_target(self) -> list[str]:
        """Restore the previously active model target from persisted state."""
        notices: list[str] = []
        last_backend = (
            str(self.config.get_state_value("last_used_backend", "local") or "local")
            .strip()
            .lower()
        )
        clear_stale_local_last_used = False

        if last_backend == "cloud" and getattr(self.config.cloud, "cloud_enabled", True):
            provider = str(
                self.config.get_state_value("last_used_cloud_provider", "") or ""
            ).strip()
            model_id = str(self.config.get_state_value("last_used_cloud_model", "") or "").strip()
            if provider and model_id:
                try:
                    result = self.model_service.select_cloud_model(
                        provider=provider, model_id=model_id
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    notices.append(f"Failed to restore cloud model: {exc}")
                else:
                    if bool(result.get("ok")):
                        notices.append(f"Restored cloud · {provider}:{model_id}")
                        return notices
                    notices.append(str(result.get("message", "Failed to restore cloud model.")))

        local_model = str(
            self.config.model.last_used_model or self.config.model.default_model or ""
        ).strip()
        if local_model:
            # Lazy restore: set the target without booting the (slow) lumen
            # server — it starts on the first turn or explicit /model.
            restored = self.model_service.restore_local_target(local_model)
            if restored:
                notices.append(f"Restored local · {restored} (server starts on first use)")
                return notices

            notices.append(f"Failed to restore local · {local_model}")
            self.model_service.active_target = ActiveModelTarget.local(model_name=None)
            # Clear only runtime-state-based local restore values; keep explicit defaults.
            last_used_local = str(self.config.model.last_used_model or "").strip()
            default_local = str(self.config.model.default_model or "").strip()
            if last_used_local and local_model == last_used_local and local_model != default_local:
                clear_stale_local_last_used = True
            # Clear only runtime-state-based local restore values; keep explicit defaults.
            last_used_local = str(self.config.model.last_used_model or "").strip()
            default_local = str(self.config.model.default_model or "").strip()
            if last_used_local and local_model == last_used_local and local_model != default_local:
                clear_stale_local_last_used = True

        cloud_router = getattr(self, "cloud_router", None)
        cloud_catalog = getattr(self, "cloud_catalog", None)
        if (
            getattr(self.config.cloud, "cloud_enabled", True)
            and cloud_router is not None
            and cloud_catalog is not None
        ):
            preferred_providers: list[str] = []
            last_provider = (
                str(self.config.get_state_value("last_used_cloud_provider", "") or "")
                .strip()
                .lower()
            )
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
                    result = self.model_service.select_cloud_model(
                        provider=provider_enum.value, model_id=model_id
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    notices.append(f"Failed to restore cloud model: {exc}")
                    continue

                if bool(result.get("ok")):
                    notices.append(f"Restored cloud · {provider_enum.value}:{model_id}")
                    return notices
                notices.append(
                    str(
                        result.get(
                            "message",
                            f"Failed to restore cloud model: {provider_enum.value}:{model_id}",
                        )
                    )
                )

        self.model_service.active_target = ActiveModelTarget.local(model_name=None)
        self._normalize_startup_state(clear_stale_local_last_used=clear_stale_local_last_used)
        notices.append("No model loaded. Use /model to select local or cloud.")
        return notices

    def _set_runtime_state_if_supported(self, key: str, value: object) -> None:
        setter = getattr(self.config, "set_state_value", None)
        if callable(setter):
            try:
                setter(key, value)
            except Exception:  # pragma: no cover - defensive path
                logger.debug("Failed to persist runtime state key=%s", key, exc_info=True)

    def _normalize_startup_state(self, *, clear_stale_local_last_used: bool) -> None:
        """Normalize persisted startup state when no target can be restored."""
        self._set_runtime_state_if_supported("last_used_backend", "local")
        self._set_runtime_state_if_supported("last_used_cloud_provider", "")
        self._set_runtime_state_if_supported("last_used_cloud_model", "")
        if clear_stale_local_last_used:
            self._set_runtime_state_if_supported("last_used_model", "")
            model_cfg = getattr(self.config, "model", None)
            if model_cfg is not None:
                try:
                    model_cfg.last_used_model = ""
                except Exception:  # pragma: no cover - defensive path
                    logger.debug("Failed to clear model.last_used_model in memory", exc_info=True)

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

    def _is_download_active(self) -> bool:
        with self._download_task_lock:
            return (
                self._active_download_thread is not None and self._active_download_thread.is_alive()
            )

    def _run_background_download(
        self, *, session_id: str, command: str, repo_id: str, choice_seq: int
    ) -> None:
        """Run `lumen pull` in the background as a live system progress message.

        Two visual stages, distinguished by the progress `kind`:
          - "download": bytes are measured from the lumen cache (`*.part`
            growth — pull's own bar is TTY-only and invisible through a pipe),
            with the latest pull output line as the phase detail.
          - "model-load": emitted when the chained auto-load starts (the
            command layer marks the transition) — the UI switches to the
            GPU-load indicator; byte polling stops.
        Background model operations never touch session.status: they are not
        turns, so the turn spinner stays out of it.
        """
        progress_message_id = f"download-progress:{uuid.uuid4().hex}"
        lines: list[str] = []
        state: Dict[str, object] = {"stage": "download", "bytes": 0, "speed": 0.0}
        state_lock = threading.Lock()
        stop_polling = threading.Event()
        baseline_bytes = partial_download_bytes()

        def _emit(*, phase: str, final: bool, content: str | None = None) -> None:
            with state_lock:
                stage = str(state["stage"])
                bytes_downloaded = int(state["bytes"])  # type: ignore[call-overload]
                speed = float(state["speed"])  # type: ignore[arg-type]
            if content is None:
                if stage == "model-load":
                    content = f"Loading {repo_id}…"
                else:
                    content = f"Downloading {repo_id}…"
            progress: Dict[str, object] = {
                "kind": "model-load" if stage == "model-load" else "download",
                "repo_id": repo_id,
                "phase": phase,
            }
            if stage == "download":
                progress["bytes_downloaded"] = bytes_downloaded
                if speed > 0:
                    progress["speed_bps"] = speed
            self._emit_event(
                session_id=session_id,
                event_type="message.updated",
                payload={
                    "message_id": progress_message_id,
                    "role": "system",
                    "content": content,
                    "final": final,
                    "progress": progress,
                },
            )

        def _poll_bytes() -> None:
            # 1s cadence: real transferred bytes from the cache's .part files.
            previous = 0
            previous_at = time.time()
            while not stop_polling.wait(1.0):
                current = max(0, partial_download_bytes() - baseline_bytes)
                now = time.time()
                with state_lock:
                    if state["stage"] != "download":
                        continue
                    elapsed = max(now - previous_at, 0.001)
                    if current > previous:
                        state["speed"] = (current - previous) / elapsed
                    state["bytes"] = current
                previous, previous_at = current, now
                _emit(phase=str(lines[-1]) if lines else "transferring", final=False)

        def _progress_callback(payload: Dict[str, object]) -> None:
            kind = str(payload.get("kind", "download") or "download").strip()
            phase = str(payload.get("phase", "") or "").strip()
            if kind == "model-load":
                with state_lock:
                    state["stage"] = "model-load"
                stop_polling.set()
                _emit(phase="loading", final=False)
                return
            if not phase:
                return
            lines.append(phase)
            _emit(phase=phase, final=False)

        poller = threading.Thread(
            target=_poll_bytes, name="cortex-download-bytes", daemon=True
        )
        try:
            _emit(phase="starting", final=False)
            poller.start()
            result = self.command_service.execute(
                session_id=session_id,
                command=command,
                progress_callback=_progress_callback,
                cancel_requested=self._download_cancel_event.is_set,
                activation_guard=lambda: self._model_choice_seq == choice_seq,
            )
            stop_polling.set()
            message = str(result.get("message", "")) or (
                f"Downloaded {repo_id}." if bool(result.get("ok")) else "Download failed."
            )
            # One operation = one transcript message (resolved in place).
            _emit(
                phase="complete" if bool(result.get("ok")) else "failed",
                final=True,
                content=message,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            stop_polling.set()
            _emit(phase="failed", final=True, content=f"Download error: {exc}")
        finally:
            stop_polling.set()

    def _current_download_repo_id(self) -> str | None:
        """Repo id of the in-flight background download, if any."""
        with self._download_task_lock:
            alive = (
                self._active_download_thread is not None
                and self._active_download_thread.is_alive()
            )
            return self._active_download_repo_id if alive else None

    def _start_background_download(self, *, session_id: str, command: str, repo_id: str) -> bool:
        with self._download_task_lock:
            if self._active_download_thread is not None and self._active_download_thread.is_alive():
                return False
            self._active_download_repo_id = repo_id
            self._download_cancel_event.clear()
            thread = threading.Thread(
                target=self._run_background_download,
                kwargs={
                    "session_id": session_id,
                    "command": command,
                    "repo_id": repo_id,
                    "choice_seq": self._model_choice_seq,
                },
                name="cortex-download",
                daemon=True,
            )
            self._active_download_thread = thread
            thread.start()
            return True

    def _cancel_active_download(self) -> bool:
        with self._download_task_lock:
            if self._active_download_thread is None or not self._active_download_thread.is_alive():
                return False
            self._download_cancel_event.set()
            return True

    def _is_boot_active(self) -> bool:
        with self._boot_task_lock:
            return self._active_boot_thread is not None and self._active_boot_thread.is_alive()

    def _current_boot_selector(self) -> str | None:
        with self._boot_task_lock:
            alive = self._active_boot_thread is not None and self._active_boot_thread.is_alive()
            return self._active_boot_selector if alive else None

    def _booting_selector(self) -> str | None:
        """Selector of ANY in-flight lumen-server boot: a worker background
        boot thread, or a boot created inline by a turn (a turn is the boot
        creator when a restore/crash left the active model without a live
        server). Guards that only checked the worker's own thread were blind
        to turn-driven boots and raced them."""
        return self._current_boot_selector() or self.lumen_runtime.starting_selector()

    def _run_background_boot(self, *, session_id: str, selector: str, choice_seq: int) -> None:
        """Boot lumen-server for `selector` off the RPC thread, narrating the
        load like a download: an immediate progress message, then a ready or
        failure notice. Activation is skipped if the user picked another model
        while this one was loading."""
        progress_message_id = f"model-load:{uuid.uuid4().hex}"
        headline = f"Loading {selector}…"

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
                        "kind": "model-load",
                        "repo_id": selector,
                        "phase": phase,
                        "bytes_downloaded": 0,
                    },
                },
            )

        try:
            _emit(content=headline, phase="loading", final=False)
            ok, message = self.lumen_runtime.ensure_server(selector)
            if not ok:
                if message.startswith(SWITCH_IN_FLIGHT_PREFIX):
                    # Transient refusal — this boot raced another model's
                    # in-flight boot. Surface the runtime's retry message
                    # verbatim (same contract as the turn path in the
                    # orchestrator), never as a hard startup failure.
                    final_message = message
                else:
                    final_message = f"local · {selector} failed to start: {message}"
            elif self._model_choice_seq != choice_seq:
                final_message = (
                    f"local · {selector} finished loading (kept inactive — you "
                    "switched models)."
                )
            else:
                result = self.model_service.activate_local(selector)
                final_message = str(result.get("message", f"local · {selector} ready — now active."))
            # One operation = one transcript message: the final frame resolves
            # the live indicator in place (no duplicate system.notice row).
            _emit(content=final_message, phase="ready" if ok else "failed", final=True)
        except Exception as exc:  # pragma: no cover - defensive path
            _emit(content=f"local · {selector} failed to start: {exc}", phase="failed", final=True)

    def _start_background_boot(self, *, session_id: str, selector: str) -> bool:
        with self._boot_task_lock:
            if self._active_boot_thread is not None and self._active_boot_thread.is_alive():
                return False
            self._active_boot_selector = selector
            thread = threading.Thread(
                target=self._run_background_boot,
                kwargs={
                    "session_id": session_id,
                    "selector": selector,
                    "choice_seq": self._model_choice_seq,
                },
                name="cortex-model-boot",
                daemon=True,
            )
            self._active_boot_thread = thread
            thread.start()
            return True

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

    def _download_busy_message(self, requested_repo_id: str) -> str:
        current = self._current_download_repo_id()
        if current and current != requested_repo_id:
            return (
                f"Another download ({current}) is in progress. "
                "Wait for it to finish (/download cancel to stop it)."
            )
        return (
            "Another download is already in progress. "
            "Wait for it to finish before starting a new one."
        )

    def _maybe_intercept_model_download(
        self, *, session_id: str, selector_arg: str
    ) -> Dict[str, object] | None:
        """One-flow /model: selecting an uncached Lumen model starts its
        download (which auto-loads on completion) instead of dead-ending.

        Returns None to fall through to the synchronous handler (cloud
        selectors, cached fast path, listings, and error cases — the sync
        handler produces the precise messages for those).
        """
        if not selector_arg or selector_arg.lower() in {"list", "ls"}:
            return None
        if ":" in selector_arg:
            prefix = selector_arg.split(":", 1)[0].strip().lower()
            if prefix in _CLOUD_LOGIN_PROVIDERS:
                return None
        resolved = self.model_service.resolve_local_selector(selector_arg)
        if not bool(resolved.get("ok")):
            return None
        model = resolved.get("model")
        if model is None:
            return None
        selector = str(getattr(model, "selector", "")).strip()
        if not selector:
            return None

        if bool(getattr(model, "cached", False)):
            return self._intercept_cached_model_boot(session_id=session_id, selector=selector)

        current = self._current_download_repo_id()
        if current == selector:
            message = f"Already downloading {selector} — loads automatically when done."
            self._emit_event(
                session_id=session_id,
                event_type="system.notice",
                payload={"message": message},
            )
            return {"ok": True, "message": message, "background": True, "repo_id": selector}
        if current is not None:
            return {"ok": False, "message": self._download_busy_message(selector)}
        booting = self._booting_selector()
        if booting is not None:
            # A boot and a download must never run concurrently: the download's
            # chained auto-load would tear down the in-flight boot.
            return {
                "ok": False,
                "message": (
                    f"Still starting local · {booting} — wait for it to finish "
                    "before downloading another model."
                ),
            }

        # Background op — NOT a turn: session.status stays untouched so the
        # generic "Working…" spinner never runs; the download indicator owns
        # the signaling. Starting this download IS the user's model choice
        # (it auto-loads on completion).
        self._model_choice_seq += 1
        started = self._start_background_download(
            session_id=session_id,
            command=f"/download {selector}",
            repo_id=selector,
        )
        if not started:
            return {"ok": False, "message": self._download_busy_message(selector)}

        message = (
            f"Downloading {selector} — loads automatically when done "
            "(/download cancel to stop)."
        )
        return {"ok": True, "message": message, "background": True, "repo_id": selector}

    def _intercept_cached_model_boot(
        self, *, session_id: str, selector: str
    ) -> Dict[str, object] | None:
        """Cached local models load in the BACKGROUND with narrated progress —
        a lumen-server boot takes 15-60s and must never present as a silent
        busy spinner. Returns None only when the server is already serving the
        selector (the sync path answers instantly)."""
        if self.lumen_runtime.serving_selector() == selector:
            return None  # instant: sync handler activates and reports

        booting = self._booting_selector()
        if booting == selector:
            message = f"Already starting local · {selector} — it will be active shortly."
            self._emit_event(
                session_id=session_id,
                event_type="system.notice",
                payload={"message": message},
            )
            return {"ok": True, "message": message, "background": True, "repo_id": selector}
        if booting is not None:
            return {
                "ok": False,
                "message": (
                    f"Still starting local · {booting} — wait for it to finish "
                    "before switching."
                ),
            }
        download = self._current_download_repo_id()
        if download is not None:
            return {"ok": False, "message": self._download_busy_message(selector)}

        # Background op — NOT a turn: session.status stays untouched so the
        # generic "Working…" spinner never runs; the load indicator owns the
        # signaling. Starting this boot IS the user's model choice.
        self._model_choice_seq += 1
        if not self._start_background_boot(session_id=session_id, selector=selector):
            booting = self._current_boot_selector() or "another model"
            return {
                "ok": False,
                "message": f"Still starting local · {booting} — wait for it to finish.",
            }

        # Terse confirmation only: the transcript's live "Loading {selector}…"
        # row solely owns load state (one operation = one live message; the
        # result panel must not mirror it).
        message = f"local · {selector} selected."
        return {"ok": True, "message": message, "background": True, "repo_id": selector}

    def _maybe_intercept_setup(self, *, session_id: str) -> Dict[str, object] | None:
        """/setup boots a server too — same background flow, same messages."""
        if self.model_service.get_active_model_label() != "No model loaded":
            return None  # sync handler reports the existing model instantly
        try:
            listing = self.model_service.list_models()
        except Exception:  # pragma: no cover - defensive path
            return None
        local_models = listing.get("local", []) if isinstance(listing, dict) else []
        first_cached = ""
        for item in local_models:
            if isinstance(item, dict) and bool(item.get("cached")):
                name = str(item.get("name", "")).strip()
                if name:
                    first_cached = name
                    break
        if not first_cached:
            return None  # sync handler explains how to download
        return self._intercept_cached_model_boot(session_id=session_id, selector=first_cached)

    def _rpc_command_execute(self, params: CommandExecuteParams):
        raw_command = str(params.command or "").strip()
        command_parts = raw_command.split(maxsplit=1)
        command_keyword = command_parts[0].lower() if command_parts else ""
        command_args = command_parts[1] if len(command_parts) > 1 else ""
        if command_keyword == "/model":
            selector_arg = command_args.strip()
            # NOTE: the model-choice sequence is bumped only where an action is
            # actually ACCEPTED (a boot/download starts, or a sync selection
            # succeeds) — blocked and already-in-flight attempts must never
            # cancel a promised auto-activation.
            intercepted = self._maybe_intercept_model_download(
                session_id=params.session_id, selector_arg=selector_arg
            )
            if intercepted is not None:
                return intercepted

        if command_keyword == "/setup":
            intercepted = self._maybe_intercept_setup(session_id=params.session_id)
            if intercepted is not None:
                return intercepted

        if command_keyword == "/download":
            download_action = command_args.strip().lower()
            if download_action in {"cancel", "--cancel"}:
                cancelled = self._cancel_active_download()
                message = (
                    "Cancellation requested. The download will stop shortly."
                    if cancelled
                    else "No active download to cancel."
                )
                self._emit_event(
                    session_id=params.session_id,
                    event_type="system.notice",
                    payload={"message": message},
                )
                return {"ok": cancelled, "message": message}

            download_preflight = self.command_service.preflight_download(command_args)
            if bool(download_preflight.get("ok")) and not bool(download_preflight.get("cached")):
                repo_id = str(download_preflight.get("selector", "")).strip()
                booting = self._booting_selector()
                if booting is not None:
                    # Never run a download (whose auto-load restarts the
                    # server) while a boot is in flight.
                    return {
                        "ok": False,
                        "message": (
                            f"Still starting local · {booting} — wait for it to "
                            "finish before downloading another model."
                        ),
                    }
                # Background op — session.status untouched (see intercepts).
                # Starting this download IS the user's model choice.
                self._model_choice_seq += 1
                started = self._start_background_download(
                    session_id=params.session_id,
                    command=params.command,
                    repo_id=repo_id,
                )
                if not started:
                    message = self._download_busy_message(repo_id)
                    self._emit_event(
                        session_id=params.session_id,
                        event_type="system.notice",
                        payload={"message": message},
                    )
                    return {"ok": False, "message": message}

                start_message = (
                    f"Downloading {repo_id} — loads automatically when done "
                    "(/download cancel to stop)."
                )
                return {
                    "ok": True,
                    "message": start_message,
                    "background": True,
                    "repo_id": repo_id,
                }
            # Invalid selector or already cached: fall through to the
            # synchronous handler for its precise error/hint message.

        self._emit_event(
            session_id=params.session_id,
            event_type="session.status",
            payload={"status": "busy"},
        )

        try:
            result = self.command_service.execute(
                session_id=params.session_id, command=params.command
            )
            if (
                command_keyword == "/model"
                and command_args.strip()
                and command_args.strip().lower() not in {"list", "ls"}
                and bool(result.get("ok"))
                and not bool(result.get("background"))
            ):
                # Successful synchronous selection (cloud switch / instant
                # local activation) supersedes in-flight auto-activations.
                self._model_choice_seq += 1
        finally:
            # Background model ops never own session.status, so the sync
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
        self._model_choice_seq += 1
        if params.backend == "cloud":
            if not params.provider or not params.model_id:
                raise RpcMethodError(
                    code=-32602,
                    message="Cloud model selection requires provider and model_id",
                )
            return self.model_service.select_cloud_model(
                provider=params.provider, model_id=params.model_id
            )
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

        # Death by signal (SIGTERM from the sidecar's exit path, SIGHUP when
        # the user closes the terminal window, stray SIGINT) must never leave
        # a lumen-server running — default signal handling would skip both the
        # finally below and atexit.
        def _terminate(signum, _frame) -> None:
            logger.info("Worker received signal %s — shutting down", signum)
            try:
                self.lumen_runtime.stop()
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
            # Quitting Cortex must never leave a lumen-server behind.
            self.lumen_runtime.stop()
            # In-flight turn threads (non-daemon) would otherwise keep this
            # process alive long after stdin EOF; everything that must be
            # persisted is already flushed per-turn.
            os._exit(0)

    def __del__(self) -> None:
        return
