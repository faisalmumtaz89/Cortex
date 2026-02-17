from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from cortex.app.session_service import SessionService
from cortex.cloud.types import ActiveModelTarget
from cortex.conversation_manager import MessageRole
from cortex.tooling.types import AssistantTurnResult, FinishEvent


class _FakeConversation:
    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.messages = []


class _FakeConversationManager:
    def __init__(self) -> None:
        self.conversations: dict[str, _FakeConversation] = {}
        self.current_conversation_id: str | None = None

    def new_conversation(self) -> _FakeConversation:
        conversation = _FakeConversation(str(uuid4()))
        self.conversations[conversation.conversation_id] = conversation
        self.current_conversation_id = conversation.conversation_id
        return conversation

    def switch_conversation(self, conversation_id: str) -> None:
        self.current_conversation_id = conversation_id

    def add_message(
        self,
        role: MessageRole,
        content: str,
        *,
        conversation_id: str,
        message_id: str | None = None,
        parts=None,
    ):
        message = SimpleNamespace(
            message_id=message_id or str(uuid4()),
            role=role,
            content=content,
            parts=parts or [],
        )
        self.conversations[conversation_id].messages.append(message)
        return message


class _FakeModelService:
    def __init__(self) -> None:
        self.active_target = ActiveModelTarget.local("test-model")
        self.model_manager = SimpleNamespace(current_model="test-model")

    def get_active_model_label(self) -> str:
        return "test-model"


class _FakeBridge:
    def bind_turn(self, *, session_id: str, emit_event) -> None:
        return None

    def unbind_turn(self) -> None:
        return None


class _FakeOrchestrator:
    def run_turn(self, *_args, **kwargs):
        on_event = kwargs.get("on_event")
        if on_event is not None:
            on_event(FinishEvent(reason="stop"))
        return AssistantTurnResult(text="done")


class _RaisingOrchestrator:
    def run_turn(self, *_args, **_kwargs):
        raise RuntimeError("No model loaded")


def test_session_service_emits_single_idle_transition_after_finish_event(tmp_path: Path) -> None:
    conversation_manager = _FakeConversationManager()
    model_service = _FakeModelService()
    service = SessionService(
        config=SimpleNamespace(conversation=SimpleNamespace(save_directory=tmp_path)),
        conversation_manager=conversation_manager,
        tooling_orchestrator=_FakeOrchestrator(),
        model_service=model_service,
        tooling_bridge=_FakeBridge(),
    )

    session_id = "session-1"
    service.create_or_resume(session_id=session_id, conversation_id=None)

    events: list[tuple[str, dict[str, object]]] = []

    def emit_event(*, session_id: str, event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    result = service.submit_user_input(
        session_id=session_id,
        user_input="hello",
        active_target_input=None,
        stop_sequences=None,
        emit_event=emit_event,
    )

    status_events = [payload.get("status") for event_type, payload in events if event_type == "session.status"]
    assert status_events.count("busy") == 1
    assert status_events.count("idle") == 1
    assert isinstance(result.get("assistant_message_id"), str) and result["assistant_message_id"]

    user_updates = [payload for event_type, payload in events if event_type == "message.updated" and payload.get("role") == "user"]
    assert len(user_updates) == 1
    assert isinstance(user_updates[0].get("created_ts_ms"), int)

    assistant_updates = [
        payload for event_type, payload in events if event_type == "message.updated" and payload.get("role") == "assistant"
    ]
    assert any(payload.get("final") is False for payload in assistant_updates)
    final_assistant = next(payload for payload in assistant_updates if payload.get("final") is True)
    assert final_assistant.get("mode") == "chat"
    assert final_assistant.get("model_label") == "test-model"
    assert isinstance(final_assistant.get("elapsed_ms"), int)
    assert isinstance(final_assistant.get("completed_ts_ms"), int)


def test_session_service_turn_failure_returns_structured_result_without_raise(tmp_path: Path) -> None:
    conversation_manager = _FakeConversationManager()
    model_service = _FakeModelService()
    service = SessionService(
        config=SimpleNamespace(conversation=SimpleNamespace(save_directory=tmp_path)),
        conversation_manager=conversation_manager,
        tooling_orchestrator=_RaisingOrchestrator(),
        model_service=model_service,
        tooling_bridge=_FakeBridge(),
    )

    session_id = "session-2"
    service.create_or_resume(session_id=session_id, conversation_id=None)

    events: list[tuple[str, dict[str, object]]] = []

    def emit_event(*, session_id: str, event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    result = service.submit_user_input(
        session_id=session_id,
        user_input="hello",
        active_target_input=None,
        stop_sequences=None,
        emit_event=emit_event,
    )

    assert result["ok"] is False
    assert "No model loaded" in str(result["error"])
    assert isinstance(result.get("assistant_message_id"), str) and result["assistant_message_id"]
    assert any(event_type == "session.error" for event_type, _payload in events)
    status_events = [payload.get("status") for event_type, payload in events if event_type == "session.status"]
    assert status_events.count("busy") == 1
    assert status_events.count("idle") == 1
