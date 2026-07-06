from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from cortex.app.session_service import SessionService
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider
from cortex.conversation_manager import MessageRole
from cortex.tooling.types import (
    AssistantTurnResult,
    FinishEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolExecutionState,
    ToolResult,
    ToolResultEvent,
)


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
        self.recorded_provenance: list[dict] = []

    def get_active_model_label(self) -> str:
        return "test-model"

    def record_turn_provenance(self, *, backend, label, verified, record) -> None:
        self.recorded_provenance.append(
            {"backend": backend, "label": label, "verified": verified, "record": record}
        )


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


def test_interrupt_mid_turn_finalizes_partial_text(tmp_path: Path) -> None:
    conversation_manager = _FakeConversationManager()
    model_service = _FakeModelService()

    class _StreamingOrchestrator:
        def __init__(self) -> None:
            self.service: SessionService | None = None

        def run_turn(self, *_args, **kwargs):
            on_event = kwargs["on_event"]
            on_event(TextDeltaEvent(delta="partial "))
            on_event(TextDeltaEvent(delta="answer"))
            assert self.service is not None
            assert self.service.request_interrupt("session-3") is True
            # Next event hits the interrupt flag and unwinds the turn.
            on_event(TextDeltaEvent(delta=" that never lands"))
            raise AssertionError("interrupt should have unwound the turn")

    orchestrator = _StreamingOrchestrator()
    service = SessionService(
        config=SimpleNamespace(conversation=SimpleNamespace(save_directory=tmp_path)),
        conversation_manager=conversation_manager,
        tooling_orchestrator=orchestrator,
        model_service=model_service,
        tooling_bridge=_FakeBridge(),
    )
    orchestrator.service = service

    session_id = "session-3"
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

    assert result["ok"] is True
    assert result["interrupted"] is True
    assert result["assistant_text"] == "partial answer"

    final = next(
        payload
        for event_type, payload in events
        if event_type == "message.updated" and payload.get("final") and payload.get("role") == "assistant"
    )
    assert final.get("interrupted") is True
    assert final.get("content") == "partial answer"

    idle = next(
        payload
        for event_type, payload in events
        if event_type == "session.status" and payload.get("status") == "idle"
    )
    assert idle.get("interrupted") is True

    # A finished turn no longer accepts interrupts.
    assert service.request_interrupt(session_id) is False


def _tool_part_frames(events: list[tuple[str, dict[str, object]]], call_id: str) -> list[dict]:
    return [
        payload["part"]  # type: ignore[index]
        for event_type, payload in events
        if event_type == "message.part.updated"
        and isinstance(payload.get("part"), dict)
        and payload["part"].get("type") == "tool"  # type: ignore[union-attr]
        and payload["part"].get("call_id") == call_id  # type: ignore[union-attr]
    ]


def _run_interrupted_tool_turn(tmp_path: Path, orchestrator_cls):
    conversation_manager = _FakeConversationManager()
    orchestrator = orchestrator_cls()
    service = SessionService(
        config=SimpleNamespace(conversation=SimpleNamespace(save_directory=tmp_path)),
        conversation_manager=conversation_manager,
        tooling_orchestrator=orchestrator,
        model_service=_FakeModelService(),
        tooling_bridge=_FakeBridge(),
    )
    orchestrator.service = service
    session_id = "session-tools"
    service.create_or_resume(session_id=session_id, conversation_id=None)

    events: list[tuple[str, dict[str, object]]] = []

    def emit_event(*, session_id: str, event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    result = service.submit_user_input(
        session_id=session_id,
        user_input="run the tool",
        active_target_input=None,
        stop_sequences=None,
        emit_event=emit_event,
    )
    return result, events


def test_interrupt_resolves_inflight_tool_row_to_terminal_state(tmp_path: Path) -> None:
    """Regression (2026-07-06 screenshot): a tool row left in 'running' when
    the turn unwound on interrupt spun forever — no terminal frame ever
    arrived for it. The interrupt finalization must resolve every pending
    tool call to state=error/'Interrupted.' BEFORE the final message frame."""

    class _ToolThenInterrupt:
        def __init__(self) -> None:
            self.service: SessionService | None = None

        def run_turn(self, *_args, **kwargs):
            on_event = kwargs["on_event"]
            on_event(ToolCallEvent(call=ToolCall(id="call_1", name="bash", arguments={"command": "sleep 99"})))
            assert self.service is not None
            assert self.service.request_interrupt("session-tools") is True
            # The turn dies before this tool ever produces a result.
            on_event(TextDeltaEvent(delta="never lands"))
            raise AssertionError("interrupt should have unwound the turn")

    result, events = _run_interrupted_tool_turn(tmp_path, _ToolThenInterrupt)

    assert result["interrupted"] is True
    frames = _tool_part_frames(events, "call_1")
    assert [frame.get("state") for frame in frames] == ["running", "error"]
    assert frames[-1].get("ok") is False
    assert frames[-1].get("error") == "Interrupted."
    # The terminal tool frame lands BEFORE the final assistant frame, so no
    # renderer ever sees a finalized turn with a still-running row.
    part_index = next(
        i for i, (event_type, p) in enumerate(events)
        if event_type == "message.part.updated"
        and isinstance(p.get("part"), dict)
        and p["part"].get("state") == "error"  # type: ignore[union-attr]
    )
    final_index = next(
        i for i, (event_type, p) in enumerate(events)
        if event_type == "message.updated" and p.get("final") and p.get("role") == "assistant"
    )
    assert part_index < final_index


def test_tool_result_arriving_after_interrupt_is_not_swallowed(tmp_path: Path) -> None:
    """The real-world repro: Esc lands while a tool executes; the tool then
    finishes and its ToolResultEvent arrives with the interrupt already
    pending. The result frame must be EMITTED (the tool genuinely ran) before
    the turn unwinds — raising first swallowed it and left the row spinning."""

    class _ResultAfterInterrupt:
        def __init__(self) -> None:
            self.service: SessionService | None = None

        def run_turn(self, *_args, **kwargs):
            on_event = kwargs["on_event"]
            on_event(ToolCallEvent(call=ToolCall(id="call_1", name="bash", arguments={"command": "ls"})))
            assert self.service is not None
            assert self.service.request_interrupt("session-tools") is True
            on_event(
                ToolResultEvent(
                    result=ToolResult(
                        id="call_1",
                        name="bash",
                        state=ToolExecutionState.COMPLETED,
                        ok=True,
                        output="done",
                    )
                )
            )
            raise AssertionError("interrupt should have unwound after the result frame")

    result, events = _run_interrupted_tool_turn(tmp_path, _ResultAfterInterrupt)

    assert result["interrupted"] is True
    frames = _tool_part_frames(events, "call_1")
    # running → completed, and NO synthetic error frame afterwards: the call
    # was no longer pending when the interrupt finalization ran.
    assert [frame.get("state") for frame in frames] == ["running", "completed"]
    assert frames[-1].get("ok") is True
    assert frames[-1].get("output") == "done"


def test_turn_error_resolves_inflight_tool_row_to_terminal_state(tmp_path: Path) -> None:
    """The generic turn-failure path has the same obligation as interrupt:
    a pending tool row must resolve (state=error/'Aborted.'), never spin."""

    class _ToolThenCrash:
        def __init__(self) -> None:
            self.service: SessionService | None = None

        def run_turn(self, *_args, **kwargs):
            on_event = kwargs["on_event"]
            on_event(ToolCallEvent(call=ToolCall(id="call_1", name="read_file", arguments={"path": "x"})))
            raise RuntimeError("provider exploded mid-tool")

    result, events = _run_interrupted_tool_turn(tmp_path, _ToolThenCrash)

    assert result["ok"] is False
    frames = _tool_part_frames(events, "call_1")
    assert [frame.get("state") for frame in frames] == ["running", "error"]
    assert frames[-1].get("error") == "Aborted."


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


def test_mid_turn_model_switch_does_not_relabel_in_flight_turn(tmp_path: Path) -> None:
    """A /model switch landing while a turn streams must not change what the
    finished turn claims it was served by."""
    conversation_manager = _FakeConversationManager()
    model_service = _FakeModelService()

    class _SwitchingOrchestrator:
        def run_turn(self, *_args, **kwargs):
            # Simulate a background boot/download completing mid-turn and
            # flipping the ACTIVE target under the running turn.
            model_service.active_target = ActiveModelTarget.cloud(
                CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1")
            )
            result = AssistantTurnResult(text="done")
            # Orchestrator verified the turn against the ORIGINAL target.
            result.provenance_verified = True
            result.served_backend = "local"
            result.served_model_label = "test-model"
            result.provenance = {"client_kind": "lumen", "reported_model": "test-model"}
            return result

    service = SessionService(
        config=SimpleNamespace(conversation=SimpleNamespace(save_directory=tmp_path)),
        conversation_manager=conversation_manager,
        tooling_orchestrator=_SwitchingOrchestrator(),
        model_service=model_service,
        tooling_bridge=_FakeBridge(),
    )
    session_id = "session-switch"
    service.create_or_resume(session_id=session_id, conversation_id=None)

    events: list[tuple[str, dict[str, object]]] = []

    def emit_event(*, session_id: str, event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    service.submit_user_input(
        session_id=session_id,
        user_input="hello",
        active_target_input=None,
        stop_sequences=None,
        emit_event=emit_event,
    )

    finals = [
        payload
        for event_type, payload in events
        if event_type == "message.updated" and payload.get("role") == "assistant" and payload.get("final")
    ]
    assert finals, "no final assistant frame emitted"
    final = finals[-1]
    # The in-flight turn keeps its own (verified) identity — not the new target.
    assert final["backend"] == "local"
    assert final["model_label"] == "test-model"
    assert final["provenance_verified"] is True
    # And the verified record was stored for /status.
    assert model_service.recorded_provenance[-1]["label"] == "test-model"
    assert model_service.recorded_provenance[-1]["verified"] is True
