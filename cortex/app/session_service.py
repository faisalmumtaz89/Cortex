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


class _WorkerToolingBridge:
    """Minimal CLI-compatible bridge consumed by ToolingOrchestrator."""

    def __init__(
        self,
        *,
        config,
        conversation_manager,
        cloud_router,
        lumen_runtime,
        permission_service,
    ) -> None:
        self.config = config
        self.conversation_manager = conversation_manager
        self.cloud_router = cloud_router
        self.lumen_runtime = lumen_runtime
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


class TurnInterruptedError(RuntimeError):
    """Raised inside a running turn when the user requests an interrupt."""


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
        self._interrupts: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def request_interrupt(self, session_id: str) -> bool:
        """Signal the session's active turn to stop. True if a turn was live."""
        with self._lock:
            event = self._interrupts.get(session_id)
        if event is None:
            return False
        event.set()
        return True

    def has_active_turn(self) -> bool:
        """True while ANY session's generation turn is running (each live turn
        registers its interrupt event for exactly the turn's lifetime)."""
        with self._lock:
            return bool(self._interrupts)

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
            local_model = payload.local_model or self.model_service.active_target.local_model
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
        active_backend = active_target.backend
        turn_started_ms = self._now_ms()

        # Captured by the turn closures below; REGISTERED (visible to
        # request_interrupt / has_active_turn) only at the try boundary right
        # before the turn runs, so no raise in the setup span can ever leak
        # the registry entry — see the registration site below.
        interrupt_event = threading.Event()

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
                "backend": active_backend,
            },
        )

        # call_id → tool name for every emitted tool call whose result frame
        # has not been emitted yet. Abnormal turn endings (interrupt, error)
        # resolve these to a terminal state — a row left in "running" would
        # spin forever in the transcript.
        pending_tool_calls: Dict[str, str] = {}

        def resolve_dangling_tools(error_label: str) -> None:
            for call_id, tool_name in pending_tool_calls.items():
                emit_event(
                    session_id=session_id,
                    event_type="message.part.updated",
                    payload={
                        "message_id": assistant_message_id,
                        "part": {
                            "type": "tool",
                            "call_id": call_id,
                            "tool": tool_name,
                            "state": "error",
                            "ok": False,
                            "error": error_label,
                        },
                    },
                )
            pending_tool_calls.clear()

        def on_event(event: ModelEvent) -> None:
            if isinstance(event, ToolCallEvent):
                if interrupt_event.is_set():
                    # Never surface a new tool row for a turn that is unwinding.
                    raise TurnInterruptedError()
                pending_tool_calls[event.call.id] = event.call.name
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
                # The tool genuinely ran — emit its outcome even when an
                # interrupt is already pending, THEN unwind. Raising first
                # would swallow the result and leave the row spinning.
                pending_tool_calls.pop(event.result.id, None)
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
                if interrupt_event.is_set():
                    raise TurnInterruptedError()
                return

            if interrupt_event.is_set():
                raise TurnInterruptedError()
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

        # EXCEPTION-SAFE registration: the registry entry is created
        # immediately before a try whose finally removes it — a raise
        # anywhere in the turn (bind_turn included) can never leak the entry.
        # A leaked entry pins has_active_turn() True forever, permanently
        # refusing /update lumen. The setup emissions above run unregistered
        # on purpose: they take microseconds and nothing interruptible
        # happens before run_turn.
        with self._lock:
            self._interrupts[session_id] = interrupt_event
        try:
            self.tooling_bridge.bind_turn(session_id=session_id, emit_event=emit_event)
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
            except TurnInterruptedError:
                resolve_dangling_tools("Interrupted.")
                completed_ms = self._now_ms()
                partial_text = "".join(assistant_chunks)
                assistant_message = self.conversation_manager.add_message(
                    MessageRole.ASSISTANT,
                    partial_text,
                    conversation_id=conversation.conversation_id,
                    message_id=assistant_message_id,
                )
                emit_event(
                    session_id=session_id,
                    event_type="message.updated",
                    payload={
                        "message_id": assistant_message.message_id,
                        "role": "assistant",
                        "content": partial_text,
                        "parts": [],
                        "final": True,
                        "interrupted": True,
                        "created_ts_ms": assistant_started_ms,
                        "completed_ts_ms": completed_ms,
                        "elapsed_ms": max(1, completed_ms - assistant_started_ms),
                        "parent_id": user_message.message_id,
                        "mode": "chat",
                        "model_label": active_model_label,
                        "backend": active_backend,
                    },
                )
                emit_event(
                    session_id=session_id,
                    event_type="session.status",
                    payload={"status": "idle", "interrupted": True},
                )
                return {
                    "session_id": session_id,
                    "assistant_message_id": assistant_message_id,
                    "assistant_text": partial_text,
                    "token_count": 0,
                    "elapsed_seconds": max(0.001, (completed_ms - assistant_started_ms) / 1000),
                    "active_model_label": active_model_label,
                    "ok": True,
                    "interrupted": True,
                }
            except Exception as exc:
                resolve_dangling_tools("Aborted.")
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
                        "backend": active_backend,
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
            with self._lock:
                # Identity-guarded: never remove a newer turn's registration.
                if self._interrupts.get(session_id) is interrupt_event:
                    del self._interrupts[session_id]

        # The final frame carries VERIFIED provenance labels (what actually
        # answered), not just intent — a mismatch never reaches this point
        # because the orchestrator rejects the turn.
        served_backend = turn_result.served_backend or active_backend
        served_model_label = turn_result.served_model_label or active_model_label
        self.model_service.record_turn_provenance(
            backend=served_backend,
            label=served_model_label,
            verified=turn_result.provenance_verified,
            record=turn_result.provenance,
        )

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
                "model_label": served_model_label,
                "backend": served_backend,
                "provenance_verified": turn_result.provenance_verified,
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
