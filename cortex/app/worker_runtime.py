"""Worker runtime assembly for Cortex JSON-RPC mode."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from io import TextIOBase
from pathlib import Path
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
        self._download_task_lock = threading.Lock()
        self._active_download_thread: threading.Thread | None = None
        self._download_cancel_event = threading.Event()
        self._download_progress_sample_interval_seconds = 0.25
        self._download_progress_heartbeat_seconds = 2.0
        self._download_progress_stall_seconds = 6.0
        self._startup_notices: list[str] = self._restore_startup_target()
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
                        notices.append(f"Restored cloud model: {provider}:{model_id}")
                        return notices
                    notices.append(str(result.get("message", "Failed to restore cloud model.")))

        local_model = str(
            self.config.model.last_used_model or self.config.model.default_model or ""
        ).strip()
        if local_model:
            result = self.model_service.select_local_model(local_model)
            if bool(result.get("ok")):
                notices.append(
                    f"Restored local model: {self.model_service.get_active_model_label()}"
                )
                return notices

            notices.append(
                str(result.get("message", f"Failed to restore local model: {local_model}"))
            )
            self.model_service.active_target = ActiveModelTarget.local(model_name=None)
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
                    notices.append(f"Restored cloud model: {provider_enum.value}:{model_id}")
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

    @staticmethod
    def _coerce_number_to_int(value: object, *, default: int = 0) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        return default

    @staticmethod
    def _coerce_number_to_float(value: object, *, default: float = 0.0) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return default

    @staticmethod
    def _format_download_progress_message(*, repo_id: str, payload: Dict[str, object]) -> str:
        def _format_units(value: float) -> str:
            rounded = round(value)
            if abs(value - rounded) < 1e-6:
                return str(int(rounded))
            return f"{value:.1f}"

        description = str(payload.get("description", "") or "").strip()
        normalized_description = description.rstrip(":").strip()
        unit = str(payload.get("unit", "") or "").strip().lower()
        label = normalized_description if normalized_description else f"Downloading {repo_id}"
        done = bool(payload.get("done"))
        downloaded = WorkerRuntime._coerce_number_to_int(payload.get("bytes_downloaded"), default=0)
        units_downloaded_raw = payload.get("units_downloaded")
        units_downloaded = (
            float(units_downloaded_raw)
            if isinstance(units_downloaded_raw, (int, float))
            else float(downloaded)
        )
        total_raw = payload.get("bytes_total")
        total = int(total_raw) if isinstance(total_raw, (int, float)) and total_raw > 0 else 0
        units_total_raw = payload.get("units_total")
        units_total = (
            float(units_total_raw)
            if isinstance(units_total_raw, (int, float)) and units_total_raw > 0
            else float(total)
        )
        percent_raw = payload.get("percent")
        percent = float(percent_raw) if isinstance(percent_raw, (int, float)) else None

        if (
            normalized_description
            and normalized_description.lower().startswith("download complete")
            and downloaded <= 0
        ):
            return normalized_description
        if done and normalized_description and downloaded <= 0:
            return normalized_description

        if total > 0 and (
            "file" in normalized_description.lower() or unit in {"file", "files", "it"}
        ):
            if percent is None:
                percent = (units_downloaded / units_total) * 100.0 if units_total > 0 else 0.0
            completed_display = _format_units(max(units_downloaded, 0.0))
            total_display = _format_units(max(units_total, 0.0))
            return f"{label}: {percent:.1f}% ({completed_display}/{total_display} files)"

        downloaded_mb = downloaded / (1024 * 1024)
        if total > 0:
            total_mb = total / (1024 * 1024)
            if percent is None:
                percent = (downloaded / total) * 100.0 if total else 0.0
            return f"{label}: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)"
        if downloaded > 0:
            return f"{label}: {downloaded_mb:.1f} MB"
        return label

    @staticmethod
    def _format_bytes(value: int) -> str:
        size = max(int(value), 0)
        units = ("B", "KB", "MB", "GB", "TB")
        amount = float(size)
        unit = units[0]
        for candidate in units[1:]:
            if amount < 1024.0:
                break
            amount /= 1024.0
            unit = candidate
        if unit == "B":
            return f"{int(amount)} {unit}"
        return f"{amount:.1f} {unit}"

    @staticmethod
    def _format_rate(value: float | None) -> str | None:
        if value is None or value <= 1:
            return None
        return f"{WorkerRuntime._format_bytes(int(value))}/s"

    @staticmethod
    def _format_eta(seconds: float | None) -> str | None:
        if seconds is None or not isinstance(seconds, (int, float)) or seconds <= 0:
            return None
        remaining = max(1, int(round(seconds)))
        if remaining < 60:
            return f"{remaining}s"
        minutes, sec = divmod(remaining, 60)
        if minutes < 60:
            return f"{minutes}m {sec:02d}s"
        hours, mins = divmod(minutes, 60)
        return f"{hours}h {mins:02d}m"

    @staticmethod
    def _build_download_progress_payload(
        *,
        repo_id: str,
        phase: str,
        bytes_downloaded: int,
        bytes_total: int | None,
        allow_bytes_percent: bool,
        files_completed: float | None,
        files_total: float | None,
        speed_bps: float | None,
        eta_seconds: float | None,
        elapsed_seconds: float,
        stalled: bool,
    ) -> Dict[str, object]:
        normalized_phase = str(phase or "preparing").strip().lower() or "preparing"
        normalized_bytes_total = (
            int(bytes_total) if isinstance(bytes_total, int) and bytes_total > 0 else None
        )
        normalized_bytes_downloaded = int(max(bytes_downloaded, 0))
        if normalized_bytes_total is not None:
            normalized_bytes_downloaded = min(normalized_bytes_downloaded, normalized_bytes_total)
        if normalized_phase == "completed" and normalized_bytes_total is not None:
            normalized_bytes_downloaded = normalized_bytes_total

        normalized_files_total = (
            float(files_total)
            if isinstance(files_total, (int, float)) and float(files_total) > 0
            else None
        )
        normalized_files_completed = (
            float(files_completed)
            if isinstance(files_completed, (int, float)) and float(files_completed) >= 0
            else None
        )
        if normalized_files_total is not None and normalized_files_completed is not None:
            normalized_files_completed = min(
                max(normalized_files_completed, 0.0), normalized_files_total
            )
        if normalized_phase == "completed" and normalized_files_total is not None:
            normalized_files_completed = normalized_files_total

        bytes_percent: float | None = None
        if (
            allow_bytes_percent
            and normalized_bytes_total is not None
            and normalized_bytes_total > 0
        ):
            bytes_percent = max(
                0.0, min(100.0, (normalized_bytes_downloaded / normalized_bytes_total) * 100.0)
            )
        files_percent: float | None = None
        if (
            normalized_files_total is not None
            and normalized_files_total > 0
            and normalized_files_completed is not None
        ):
            files_percent = max(
                0.0, min(100.0, (normalized_files_completed / normalized_files_total) * 100.0)
            )

        is_terminal_phase = normalized_phase in {"completed", "failed", "cancelled"}
        is_finalizing_phase = normalized_phase == "finalizing"

        # Byte-level progress is more granular than file-count progress and avoids
        # coarse jumps (for example 100% files while bytes are still finalizing).
        percent: float | None = bytes_percent if bytes_percent is not None else files_percent
        if normalized_phase == "completed":
            percent = 100.0
        elif is_finalizing_phase:
            percent = 100.0
        elif percent is not None:
            # Never report a non-terminal update as fully complete.
            percent = min(percent, 99.9)

        normalized_eta_seconds = (
            float(eta_seconds)
            if (
                isinstance(eta_seconds, (int, float))
                and eta_seconds > 0
                and not is_terminal_phase
                and not is_finalizing_phase
            )
            else None
        )

        return {
            "kind": "download",
            "repo_id": repo_id,
            "phase": normalized_phase,
            "bytes_downloaded": normalized_bytes_downloaded,
            "bytes_total": normalized_bytes_total,
            "percent": round(percent, 3) if percent is not None else None,
            "files_completed": normalized_files_completed,
            "files_total": normalized_files_total,
            "speed_bps": (
                float(speed_bps) if isinstance(speed_bps, (int, float)) and speed_bps > 0 else None
            ),
            "eta_seconds": normalized_eta_seconds,
            "elapsed_seconds": float(max(elapsed_seconds, 0.0)),
            "stalled": bool(stalled) and not is_terminal_phase and not is_finalizing_phase,
        }

    @staticmethod
    def _format_download_progress_content(*, repo_id: str, progress: Dict[str, object]) -> str:
        phase = str(progress.get("phase", "preparing") or "preparing").strip().lower()
        downloaded = WorkerRuntime._coerce_number_to_int(
            progress.get("bytes_downloaded"), default=0
        )
        total_raw = progress.get("bytes_total")
        total = int(total_raw) if isinstance(total_raw, (int, float)) and total_raw > 0 else None
        percent_raw = progress.get("percent")
        percent = float(percent_raw) if isinstance(percent_raw, (int, float)) else None
        speed_raw = progress.get("speed_bps")
        speed_bps = (
            float(speed_raw) if isinstance(speed_raw, (int, float)) and speed_raw > 0 else None
        )
        eta_raw = progress.get("eta_seconds")
        eta_seconds = float(eta_raw) if isinstance(eta_raw, (int, float)) and eta_raw > 0 else None
        files_done_raw = progress.get("files_completed")
        files_done = (
            float(files_done_raw)
            if isinstance(files_done_raw, (int, float)) and files_done_raw >= 0
            else None
        )
        files_total_raw = progress.get("files_total")
        files_total = (
            float(files_total_raw)
            if isinstance(files_total_raw, (int, float)) and files_total_raw > 0
            else None
        )
        stalled = bool(progress.get("stalled"))

        if phase == "completed":
            return f"Download complete: {repo_id}"
        if phase == "failed":
            return f"Download failed: {repo_id}"
        if phase == "cancelled":
            return f"Download cancelled: {repo_id}"
        if phase == "finalizing":
            return f"Finalizing download: {repo_id}"

        status_parts: list[str] = []
        if total is not None and percent is not None:
            status_parts.append(
                f"{percent:.1f}% ({WorkerRuntime._format_bytes(downloaded)}/{WorkerRuntime._format_bytes(total)})"
            )
        elif downloaded > 0:
            status_parts.append(f"{WorkerRuntime._format_bytes(downloaded)} downloaded")

        speed = WorkerRuntime._format_rate(speed_bps)
        if speed:
            status_parts.append(speed)
        eta = WorkerRuntime._format_eta(eta_seconds)
        if eta:
            status_parts.append(f"ETA {eta}")
        if files_done is not None and files_total is not None:
            status_parts.append(f"{files_done:.1f}/{files_total:.1f} files")
        if stalled:
            status_parts.append("waiting for transfer")

        if status_parts:
            return f"Downloading {repo_id}: {' | '.join(status_parts)}"
        return f"Preparing download: {repo_id}"

    @staticmethod
    def _directory_size_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        total = 0
        for root, _dirs, files in os.walk(path):
            for filename in files:
                candidate = Path(root) / filename
                try:
                    total += candidate.stat().st_size
                except OSError:
                    continue
        return total

    def _run_background_download(
        self,
        *,
        session_id: str,
        command: str,
        repo_id: str,
        download_monitor_path: str | None = None,
    ) -> None:
        progress_message_id = f"download-progress:{uuid.uuid4().hex}"
        started_at = time.monotonic()
        monitor_target = (
            Path(download_monitor_path).expanduser().resolve() if download_monitor_path else None
        )
        sample_interval = max(
            float(getattr(self, "_download_progress_sample_interval_seconds", 0.25) or 0.25),
            0.05,
        )
        heartbeat_interval = max(
            float(getattr(self, "_download_progress_heartbeat_seconds", 2.0) or 2.0),
            sample_interval,
        )
        stall_seconds = max(
            float(getattr(self, "_download_progress_stall_seconds", 6.0) or 6.0),
            sample_interval,
        )
        speed_window_seconds = 8.0

        tracker_lock = threading.Lock()
        tracker_state: Dict[str, object] = {
            "bytes_downloaded": 0,
            "bytes_total": None,
            "files_completed": 0.0,
            "files_total": None,
            "last_activity_at": started_at,
            "last_total_change_at": started_at,
            "callback_done": False,
            "phase_override": None,
            "speed_samples": deque([(started_at, 0)], maxlen=256),
        }
        sampler_stop = threading.Event()
        sampler_thread: threading.Thread | None = None
        last_emitted_bytes_key = -1
        last_emitted_files_key = -1.0
        last_emitted_phase = ""
        last_emitted_percent_key: float | None = None
        last_emit_at = 0.0
        emitted_final = False
        emit_lock = threading.Lock()

        def _is_file_progress(description: str, unit: str) -> bool:
            normalized_description = description.strip().lower()
            normalized_unit = unit.strip().lower()
            return "file" in normalized_description or normalized_unit in {"file", "files", "it"}

        def _build_snapshot_payload(*, phase_override: str | None = None) -> Dict[str, object]:
            now = time.monotonic()
            with tracker_lock:
                bytes_downloaded = WorkerRuntime._coerce_number_to_int(
                    tracker_state.get("bytes_downloaded"), default=0
                )
                bytes_total_raw = tracker_state.get("bytes_total")
                bytes_total = (
                    int(bytes_total_raw)
                    if isinstance(bytes_total_raw, int) and bytes_total_raw > 0
                    else None
                )
                files_total_raw = tracker_state.get("files_total")
                files_total = (
                    float(files_total_raw)
                    if isinstance(files_total_raw, (int, float)) and float(files_total_raw) > 0
                    else None
                )
                files_completed_raw = tracker_state.get("files_completed")
                files_completed = (
                    float(files_completed_raw)
                    if isinstance(files_completed_raw, (int, float))
                    and float(files_completed_raw) >= 0
                    else None
                )
                callback_done = bool(tracker_state.get("callback_done"))
                state_phase_override = (
                    str(tracker_state.get("phase_override") or "").strip().lower()
                )
                last_activity_at = WorkerRuntime._coerce_number_to_float(
                    tracker_state.get("last_activity_at"), default=started_at
                )
                if last_activity_at <= 0:
                    last_activity_at = started_at
                last_total_change_at = WorkerRuntime._coerce_number_to_float(
                    tracker_state.get("last_total_change_at"), default=started_at
                )
                if last_total_change_at <= 0:
                    last_total_change_at = started_at
                speed_samples = tracker_state.get("speed_samples")
                speed_bps: float | None = None
                if isinstance(speed_samples, deque) and len(speed_samples) >= 2:
                    oldest_ts, oldest_bytes = speed_samples[0]
                    newest_ts, newest_bytes = speed_samples[-1]
                    dt = float(newest_ts) - float(oldest_ts)
                    if dt > 0:
                        db = max(int(newest_bytes) - int(oldest_bytes), 0)
                        if db > 0:
                            speed_bps = db / dt

                elapsed_seconds = max(now - started_at, 0.0)
                stalled = (now - last_activity_at) >= stall_seconds and not callback_done

            phase = phase_override or state_phase_override or ""
            if not phase:
                if callback_done:
                    phase = "completed"
                elif (
                    bytes_total is not None
                    and bytes_total > 0
                    and bytes_downloaded >= bytes_total
                    and files_total is not None
                    and files_total > 0
                    and files_completed is not None
                    and files_completed >= files_total
                ):
                    phase = "finalizing"
                elif bytes_downloaded > 0 or (files_completed is not None and files_completed > 0):
                    phase = "stalled" if stalled else "transferring"
                else:
                    phase = "preparing"

            eta_seconds: float | None = None
            if (
                speed_bps is not None
                and speed_bps > 1
                and bytes_total is not None
                and bytes_total > bytes_downloaded
            ):
                eta_seconds = (bytes_total - bytes_downloaded) / speed_bps

            bytes_percent_stable = (
                callback_done
                or (now - last_total_change_at) >= 0.75
                or files_total is None
                or (
                    files_total is not None
                    and files_completed is not None
                    and files_completed >= files_total
                )
            )

            return self._build_download_progress_payload(
                repo_id=repo_id,
                phase=phase,
                bytes_downloaded=bytes_downloaded,
                bytes_total=bytes_total,
                allow_bytes_percent=bytes_percent_stable,
                files_completed=files_completed,
                files_total=files_total,
                speed_bps=speed_bps,
                eta_seconds=eta_seconds,
                elapsed_seconds=elapsed_seconds,
                stalled=stalled,
            )

        def _update_last_emit_keys(progress_payload: Dict[str, object]) -> None:
            nonlocal last_emitted_bytes_key, last_emitted_files_key, last_emitted_phase, last_emitted_percent_key
            nonlocal last_emit_at
            last_emit_at = time.monotonic()
            bytes_downloaded = WorkerRuntime._coerce_number_to_int(
                progress_payload.get("bytes_downloaded"), default=0
            )
            last_emitted_bytes_key = bytes_downloaded // (128 * 1024)
            files_completed_raw = progress_payload.get("files_completed")
            last_emitted_files_key = (
                round(float(files_completed_raw), 3)
                if isinstance(files_completed_raw, (int, float))
                else -1.0
            )
            last_emitted_phase = str(progress_payload.get("phase", "") or "")
            percent_for_key = progress_payload.get("percent")
            last_emitted_percent_key = (
                round(float(percent_for_key), 1)
                if isinstance(percent_for_key, (int, float))
                else None
            )

        def _emit_progress_update(
            *, final: bool = False, phase_override: str | None = None
        ) -> None:
            nonlocal last_emitted_bytes_key, last_emitted_files_key, last_emitted_phase, last_emitted_percent_key
            nonlocal last_emit_at, emitted_final
            with emit_lock:
                if emitted_final and not final:
                    return
                if final:
                    if emitted_final:
                        return
                    emitted_final = True

                progress_payload = _build_snapshot_payload(phase_override=phase_override)
                content = self._format_download_progress_content(
                    repo_id=repo_id, progress=progress_payload
                )
                self._emit_event(
                    session_id=session_id,
                    event_type="message.updated",
                    payload={
                        "message_id": progress_message_id,
                        "role": "system",
                        "content": content,
                        "final": final,
                        "progress": progress_payload,
                    },
                )
                _update_last_emit_keys(progress_payload)

        def _emit_terminal_update(*, content: str, phase: str) -> None:
            nonlocal emitted_final
            with emit_lock:
                if emitted_final:
                    return
                emitted_final = True
                progress_payload = _build_snapshot_payload(phase_override=phase)
                self._emit_event(
                    session_id=session_id,
                    event_type="message.updated",
                    payload={
                        "message_id": progress_message_id,
                        "role": "system",
                        "content": content,
                        "final": True,
                        "progress": progress_payload,
                    },
                )
                _update_last_emit_keys(progress_payload)

        def _on_download_progress(payload: Dict[str, object]) -> None:
            now = time.monotonic()
            description = str(payload.get("description", "") or "").strip()
            unit = str(payload.get("unit", "") or "").strip().lower()
            done = bool(payload.get("done"))
            units_downloaded_raw = payload.get("units_downloaded")
            units_downloaded = (
                float(units_downloaded_raw)
                if isinstance(units_downloaded_raw, (int, float))
                else 0.0
            )
            units_total_raw = payload.get("units_total")
            units_total = (
                float(units_total_raw)
                if isinstance(units_total_raw, (int, float)) and float(units_total_raw) > 0
                else None
            )
            bytes_downloaded_raw = payload.get("bytes_downloaded")
            callback_bytes_downloaded = (
                int(bytes_downloaded_raw)
                if isinstance(bytes_downloaded_raw, (int, float)) and int(bytes_downloaded_raw) > 0
                else 0
            )
            bytes_total_raw = payload.get("bytes_total")
            callback_bytes_total = (
                int(bytes_total_raw)
                if isinstance(bytes_total_raw, (int, float)) and int(bytes_total_raw) > 0
                else None
            )

            with tracker_lock:
                if _is_file_progress(description, unit):
                    if units_total is not None:
                        existing_total = tracker_state.get("files_total")
                        if not isinstance(existing_total, (int, float)) or units_total > float(
                            existing_total
                        ):
                            tracker_state["files_total"] = units_total
                    if units_downloaded > 0:
                        existing_completed = WorkerRuntime._coerce_number_to_float(
                            tracker_state.get("files_completed"), default=0.0
                        )
                        if units_downloaded > existing_completed:
                            tracker_state["files_completed"] = units_downloaded
                            tracker_state["last_activity_at"] = now
                else:
                    if callback_bytes_total is not None:
                        existing_total = tracker_state.get("bytes_total")
                        if (
                            not isinstance(existing_total, int)
                            or callback_bytes_total > existing_total
                        ):
                            tracker_state["bytes_total"] = callback_bytes_total
                            tracker_state["last_total_change_at"] = now
                    if callback_bytes_downloaded > 0:
                        existing_downloaded = WorkerRuntime._coerce_number_to_int(
                            tracker_state.get("bytes_downloaded"), default=0
                        )
                        if callback_bytes_downloaded > existing_downloaded:
                            tracker_state["bytes_downloaded"] = callback_bytes_downloaded
                            tracker_state["last_activity_at"] = now

                if done or description.lower().startswith("download complete"):
                    tracker_state["callback_done"] = True

            if monitor_target is None:
                _emit_progress_update(final=False)

        def _sample_progress_loop() -> None:
            nonlocal last_emit_at
            if monitor_target is None:
                return
            while not sampler_stop.wait(sample_interval):
                sampled_bytes = self._directory_size_bytes(monitor_target)
                now = time.monotonic()
                with tracker_lock:
                    current_bytes = WorkerRuntime._coerce_number_to_int(
                        tracker_state.get("bytes_downloaded"), default=0
                    )
                    if sampled_bytes > current_bytes:
                        tracker_state["bytes_downloaded"] = sampled_bytes
                        tracker_state["last_activity_at"] = now
                    speed_samples = tracker_state.get("speed_samples")
                    if isinstance(speed_samples, deque):
                        latest_bytes = WorkerRuntime._coerce_number_to_int(
                            tracker_state.get("bytes_downloaded"), default=0
                        )
                        speed_samples.append((now, latest_bytes))
                        while (
                            speed_samples
                            and (now - float(speed_samples[0][0])) > speed_window_seconds
                        ):
                            speed_samples.popleft()

                snapshot = _build_snapshot_payload()
                percent_raw = snapshot.get("percent")
                percent_key = (
                    round(float(percent_raw), 1) if isinstance(percent_raw, (int, float)) else None
                )
                bytes_key = WorkerRuntime._coerce_number_to_int(
                    snapshot.get("bytes_downloaded"), default=0
                ) // (128 * 1024)
                files_completed_raw = snapshot.get("files_completed")
                files_key = (
                    round(float(files_completed_raw), 3)
                    if isinstance(files_completed_raw, (int, float))
                    else -1.0
                )
                phase_key = str(snapshot.get("phase", "") or "")
                should_emit = False
                if bytes_key != last_emitted_bytes_key:
                    should_emit = True
                elif files_key != last_emitted_files_key:
                    should_emit = True
                elif phase_key != last_emitted_phase:
                    should_emit = True
                elif percent_key != last_emitted_percent_key:
                    should_emit = True
                elif now - last_emit_at >= heartbeat_interval:
                    should_emit = True

                if should_emit:
                    _emit_progress_update(final=False)

        if monitor_target is not None:
            sampler_thread = threading.Thread(
                target=_sample_progress_loop,
                name="cortex-download-progress-sampler",
                daemon=True,
            )
            sampler_thread.start()

        _emit_progress_update(final=False)

        try:
            result = self.command_service.execute(
                session_id=session_id,
                command=command,
                progress_callback=_on_download_progress,
                cancel_requested=self._download_cancel_event.is_set,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Background download failed")
            failure_message = f"Download failed: {exc}"
            with tracker_lock:
                tracker_state["phase_override"] = "failed"
            _emit_terminal_update(content=failure_message, phase="failed")
        else:
            notice_message = self._notice_message_for_result(result)
            if notice_message:
                success = bool(result.get("ok"))
                phase = (
                    "completed"
                    if success
                    else "cancelled" if "cancel" in notice_message.lower() else "failed"
                )
                with tracker_lock:
                    tracker_state["phase_override"] = phase
                    if phase == "completed":
                        tracker_state["callback_done"] = True
                _emit_terminal_update(content=notice_message, phase=phase)
        finally:
            sampler_stop.set()
            if sampler_thread is not None and sampler_thread.is_alive():
                sampler_thread.join(timeout=1.0)
            if not emitted_final:
                _emit_progress_update(
                    final=True,
                    phase_override="cancelled" if self._download_cancel_event.is_set() else None,
                )
            with self._download_task_lock:
                active = self._active_download_thread
                if active is not None and active.ident == threading.get_ident():
                    self._active_download_thread = None
                self._download_cancel_event.clear()
            self._emit_event(
                session_id=session_id,
                event_type="session.status",
                payload={"status": "idle"},
            )

    def _start_background_download(
        self,
        *,
        session_id: str,
        command: str,
        repo_id: str,
        download_monitor_path: str | None = None,
    ) -> bool:
        with self._download_task_lock:
            if self._active_download_thread is not None and self._active_download_thread.is_alive():
                return False
            self._download_cancel_event.clear()
            thread = threading.Thread(
                target=self._run_background_download,
                kwargs={
                    "session_id": session_id,
                    "command": command,
                    "repo_id": repo_id,
                    "download_monitor_path": download_monitor_path,
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
        raw_command = str(params.command or "").strip()
        command_parts = raw_command.split(maxsplit=1)
        command_keyword = command_parts[0].lower() if command_parts else ""
        command_args = command_parts[1] if len(command_parts) > 1 else ""
        if command_keyword == "/download":
            download_action = command_args.strip().lower()
            if download_action in {"cancel", "--cancel"}:
                cancelled = self._cancel_active_download()
                message = (
                    "Cancellation requested. The active download will stop at the next transfer checkpoint."
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
            if bool(download_preflight.get("ok")):
                repo_id = str(download_preflight.get("repo_id", "")).strip()
                filename_raw = download_preflight.get("filename")
                filename = str(filename_raw).strip() if isinstance(filename_raw, str) else None
                download_path: str | None = None
                download_monitor_path: str | None = None

                # If the local target already exists and is not resumable, handle synchronously.
                inspect_target = getattr(
                    self.command_service.model_downloader, "inspect_download_target", None
                )
                if callable(inspect_target):
                    inspected_target: Dict[str, object] | None = None
                    try:
                        raw_inspected = inspect_target(repo_id, filename)
                        if isinstance(raw_inspected, dict):
                            inspected_target = raw_inspected
                    except Exception:
                        inspected_target = None

                    if inspected_target is not None:
                        candidate_path = inspected_target.get("path")
                        if isinstance(candidate_path, Path):
                            download_path = str(candidate_path.expanduser().resolve())
                        elif isinstance(candidate_path, str) and candidate_path.strip():
                            download_path = str(Path(candidate_path).expanduser().resolve())
                        if download_path:
                            monitor_candidate = Path(download_path)
                            if filename:
                                download_monitor_path = str(monitor_candidate.parent)
                            else:
                                download_monitor_path = str(monitor_candidate)
                        target_exists = bool(inspected_target.get("exists"))
                        target_resumable = bool(inspected_target.get("resumable"))
                        if target_exists and not target_resumable:
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
                                if not self._is_download_active():
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

                self._emit_event(
                    session_id=params.session_id,
                    event_type="session.status",
                    payload={"status": "busy"},
                )
                started = self._start_background_download(
                    session_id=params.session_id,
                    command=params.command,
                    repo_id=repo_id,
                    download_monitor_path=download_monitor_path,
                )
                if not started:
                    message = "Another download is already in progress. Wait for it to finish before starting a new one."
                    self._emit_event(
                        session_id=params.session_id,
                        event_type="system.notice",
                        payload={"message": message},
                    )
                    return {"ok": False, "message": message}

                start_message = (
                    "Download request accepted. Cortex will stream progress here and report immediately "
                    "if the target already exists. Keep Cortex running until completion."
                )
                self._emit_event(
                    session_id=params.session_id,
                    event_type="system.notice",
                    payload={"message": start_message},
                )
                return {
                    "ok": True,
                    "message": start_message,
                    "background": True,
                    "repo_id": repo_id,
                }

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
            if not self._is_download_active():
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
        self.rpc_server.run_forever()

    def __del__(self) -> None:
        return
