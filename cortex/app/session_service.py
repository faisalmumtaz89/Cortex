"""Session orchestration service for worker mode."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, cast

from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider
from cortex.conversation_manager import MessageRole
from cortex.tooling.permissions import PermissionDecision
from cortex.tooling.types import (
    ErrorEvent,
    FinishEvent,
    ModelEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from cortex.ui.cli_prompt import format_prompt_with_chat_template


class _WorkerToolingBridge:
    """Minimal CLI-compatible bridge consumed by ToolingOrchestrator."""

    def __init__(
        self,
        *,
        config,
        inference_engine,
        model_manager,
        conversation_manager,
        template_registry,
        cloud_router,
        permission_service,
    ) -> None:
        self.config = config
        self.inference_engine = inference_engine
        self.model_manager = model_manager
        self.conversation_manager = conversation_manager
        self.template_registry = template_registry
        self.cloud_router = cloud_router
        self.permission_service = permission_service

        self._active_session_id: Optional[str] = None
        self._emit_event = None
        self._lock = threading.Lock()

    def bind_turn(self, *, session_id: str, emit_event) -> None:
        with self._lock:
            self._active_session_id = session_id
            self._emit_event = emit_event

    def unbind_turn(self) -> None:
        with self._lock:
            self._active_session_id = None
            self._emit_event = None

    def prompt_tool_permission(self, request) -> PermissionDecision:
        with self._lock:
            session_id = self._active_session_id or "session"
            emitter = self._emit_event
        if emitter is None:
            return PermissionDecision.REJECT
        return self.permission_service.ask(session_id=session_id, request=request, emit_event=emitter)

    def _format_prompt_with_chat_template(self, user_input: str, include_user: bool = True) -> str:
        return format_prompt_with_chat_template(
            conversation_manager=self.conversation_manager,
            model_manager=self.model_manager,
            template_registry=self.template_registry,
            user_input=user_input,
            include_user=include_user,
        )


class SessionService:
    """Manage sessions and generation turns for JSON-RPC worker mode."""

    def __init__(
        self,
        *,
        config,
        conversation_manager,
        tooling_orchestrator,
        model_service,
        tooling_bridge: _WorkerToolingBridge,
    ) -> None:
        self.config = config
        self.conversation_manager = conversation_manager
        self.tooling_orchestrator = tooling_orchestrator
        self.model_service = model_service
        self.tooling_bridge = tooling_bridge
        self._session_to_conversation: Dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _now_ms() -> int:
        return int(datetime.now().timestamp() * 1000)

    @classmethod
    def _message_timestamp_ms(cls, message, fallback_ms: int) -> int:
        timestamp = getattr(message, "timestamp", None)
        if isinstance(timestamp, datetime):
            return int(timestamp.timestamp() * 1000)
        return fallback_ms

    def _resolve_active_target(self, payload) -> ActiveModelTarget:
        if payload is None:
            return cast(ActiveModelTarget, self.model_service.active_target)

        backend = payload.backend
        if backend == "local":
            local_model = payload.local_model or self.model_service.model_manager.current_model
            target = ActiveModelTarget.local(local_model)
            self.model_service.active_target = target
            return target

        provider = CloudProvider.from_value(payload.provider or "openai")
        model_id = (payload.model_id or "").strip()
        if not model_id:
            raise ValueError("Cloud target requires model_id")
        target = ActiveModelTarget.cloud(CloudModelRef(provider=provider, model_id=model_id))
        self.model_service.active_target = target
        return target

    def _get_or_create_conversation(self, session_id: str, conversation_id: Optional[str] = None):
        with self._lock:
            mapped_id = conversation_id or self._session_to_conversation.get(session_id)

        if mapped_id and mapped_id in self.conversation_manager.conversations:
            self.conversation_manager.switch_conversation(mapped_id)
            return self.conversation_manager.conversations[mapped_id], True

        conversation = self.conversation_manager.new_conversation()
        with self._lock:
            self._session_to_conversation[session_id] = conversation.conversation_id
        return conversation, False

    def create_or_resume(self, *, session_id: str, conversation_id: Optional[str] = None):
        conversation, restored = self._get_or_create_conversation(session_id, conversation_id)
        return {
            "session_id": session_id,
            "conversation_id": conversation.conversation_id,
            "restored": restored,
            "active_model_label": self.model_service.get_active_model_label(),
        }

    def clear_session(self, session_id: str) -> Dict[str, object]:
        conversation = self.conversation_manager.new_conversation()
        with self._lock:
            self._session_to_conversation[session_id] = conversation.conversation_id
        return {"ok": True, "message": "Conversation cleared.", "conversation_id": conversation.conversation_id}

    def save_session(self, session_id: str) -> Dict[str, object]:
        with self._lock:
            conversation_id = self._session_to_conversation.get(session_id)
        if not conversation_id:
            return {"ok": False, "message": "No active conversation to save."}

        export_data = self.conversation_manager.export_conversation(conversation_id=conversation_id, format="json")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.conversation.save_directory) / f"conversation_{timestamp}.json"
        output_path.write_text(export_data, encoding="utf-8")
        return {"ok": True, "path": str(output_path)}

    def submit_user_input(
        self,
        *,
        session_id: str,
        user_input: str,
        active_target_input,
        stop_sequences: Optional[List[str]],
        emit_event,
    ) -> Dict[str, object]:
        conversation, _ = self._get_or_create_conversation(session_id)
        active_target = self._resolve_active_target(active_target_input)
        active_model_label = self.model_service.get_active_model_label()
        turn_started_ms = self._now_ms()

        user_message = self.conversation_manager.add_message(
            MessageRole.USER,
            user_input,
            conversation_id=conversation.conversation_id,
        )
        user_created_ms = self._message_timestamp_ms(user_message, turn_started_ms)
        emit_event(
            session_id=session_id,
            event_type="message.updated",
            payload={
                "message_id": user_message.message_id,
                "role": "user",
                "content": user_input,
                "final": True,
                "created_ts_ms": user_created_ms,
            },
        )

        assistant_message_id = str(uuid.uuid4())
        assistant_chunks: List[str] = []
        assistant_started_ms = self._now_ms()
        emit_event(
            session_id=session_id,
            event_type="session.status",
            payload={"status": "busy"},
        )
        emit_event(
            session_id=session_id,
            event_type="message.updated",
            payload={
                "message_id": assistant_message_id,
                "role": "assistant",
                "content": "",
                "final": False,
                "created_ts_ms": assistant_started_ms,
                "parent_id": user_message.message_id,
                "mode": "chat",
                "model_label": active_model_label,
            },
        )

        def on_event(event: ModelEvent) -> None:
            if isinstance(event, TextDeltaEvent):
                delta = event.delta or ""
                if not delta:
                    return
                assistant_chunks.append(delta)
                emit_event(
                    session_id=session_id,
                    event_type="message.part.updated",
                    payload={
                        "message_id": assistant_message_id,
                        "part": {"type": "text", "delta": delta},
                    },
                )
                return

            if isinstance(event, ToolCallEvent):
                emit_event(
                    session_id=session_id,
                    event_type="message.part.updated",
                    payload={
                        "message_id": assistant_message_id,
                        "part": {
                            "type": "tool",
                            "call_id": event.call.id,
                            "tool": event.call.name,
                            "state": "running",
                            "input": event.call.arguments,
                        },
                    },
                )
                return

            if isinstance(event, ToolResultEvent):
                emit_event(
                    session_id=session_id,
                    event_type="message.part.updated",
                    payload={
                        "message_id": assistant_message_id,
                        "part": {
                            "type": "tool",
                            "call_id": event.result.id,
                            "tool": event.result.name,
                            "state": event.result.state.value,
                            "ok": event.result.ok,
                            "output": event.result.output,
                            "error": event.result.error,
                            "metadata": event.result.metadata,
                        },
                    },
                )
                return

            if isinstance(event, ErrorEvent):
                emit_event(
                    session_id=session_id,
                    event_type="session.error",
                    payload={"error": event.error, "metadata": event.metadata},
                )
                return

            if isinstance(event, FinishEvent):
                # Session idle transition is emitted once at turn finalization.
                # Keeping it centralized avoids duplicate status frames.
                return

        def on_wait(waited_seconds: int, attempt_num: int, total_attempts: int) -> None:
            emit_event(
                session_id=session_id,
                event_type="session.status",
                payload={
                    "status": "busy",
                    "waited_seconds": waited_seconds,
                    "attempt": attempt_num,
                    "total_attempts": total_attempts,
                },
            )

        def on_retry(attempt_num: int, total_attempts: int, reason: str) -> None:
            emit_event(
                session_id=session_id,
                event_type="session.status",
                payload={
                    "status": "retry",
                    "attempt": attempt_num,
                    "total_attempts": total_attempts,
                    "reason": reason,
                },
            )

        self.tooling_bridge.bind_turn(session_id=session_id, emit_event=emit_event)
        try:
            try:
                turn_result = self.tooling_orchestrator.run_turn(
                    user_input=user_input,
                    active_target=active_target,
                    conversation=conversation,
                    stop_sequences=stop_sequences,
                    on_event=on_event,
                    on_wait=on_wait,
                    on_retry=on_retry,
                )
            except Exception as exc:
                completed_ms = self._now_ms()
                partial_text = "".join(assistant_chunks)
                emit_event(
                    session_id=session_id,
                    event_type="session.error",
                    payload={"error": str(exc)},
                )
                emit_event(
                    session_id=session_id,
                    event_type="session.status",
                    payload={"status": "idle"},
                )
                emit_event(
                    session_id=session_id,
                    event_type="message.updated",
                    payload={
                        "message_id": assistant_message_id,
                        "role": "assistant",
                        "content": partial_text,
                        "parts": [],
                        "final": True,
                        "created_ts_ms": assistant_started_ms,
                        "completed_ts_ms": completed_ms,
                        "elapsed_ms": max(1, completed_ms - assistant_started_ms),
                        "parent_id": user_message.message_id,
                        "mode": "chat",
                        "model_label": active_model_label,
                    },
                )
                return {
                    "session_id": session_id,
                    "assistant_message_id": assistant_message_id,
                    "assistant_text": partial_text,
                    "token_count": 0,
                    "elapsed_seconds": 0.0,
                    "active_model_label": active_model_label,
                    "ok": False,
                    "error": str(exc),
                }
        finally:
            self.tooling_bridge.unbind_turn()

        final_text = turn_result.text or "".join(assistant_chunks)
        assistant_message = self.conversation_manager.add_message(
            MessageRole.ASSISTANT,
            final_text,
            conversation_id=conversation.conversation_id,
            message_id=assistant_message_id,
            parts=turn_result.parts,
        )
        assistant_completed_ms = self._message_timestamp_ms(assistant_message, self._now_ms())
        elapsed_ms = max(1, int(turn_result.elapsed_seconds * 1000))
        emit_event(
            session_id=session_id,
            event_type="message.updated",
            payload={
                "message_id": assistant_message.message_id,
                "role": "assistant",
                "content": final_text,
                "parts": turn_result.parts,
                "final": True,
                "created_ts_ms": assistant_started_ms,
                "completed_ts_ms": assistant_completed_ms,
                "elapsed_ms": elapsed_ms,
                "parent_id": user_message.message_id,
                "mode": "chat",
                "model_label": active_model_label,
            },
        )
        emit_event(session_id=session_id, event_type="session.status", payload={"status": "idle"})

        return {
            "session_id": session_id,
            "assistant_message_id": assistant_message.message_id,
            "assistant_text": final_text,
            "token_count": turn_result.token_count,
            "elapsed_seconds": turn_result.elapsed_seconds,
            "active_model_label": active_model_label,
        }
