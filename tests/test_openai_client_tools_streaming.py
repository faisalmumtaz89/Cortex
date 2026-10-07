from types import SimpleNamespace

from cortex.cloud.clients.openai_client import OpenAIClient
from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolExecutionState,
    ToolResult,
    ToolResultEvent,
)


class _FakeStream:
    """Events as the SDK's ResponseStream yields them: text deltas, then
    response.completed, or response.incomplete for an incomplete response."""

    def __init__(self, deltas, final_response):
        self._deltas = deltas
        self._final_response = final_response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        for delta in self._deltas:
            yield SimpleNamespace(type="response.output_text.delta", delta=delta)
        status = getattr(self._final_response, "status", "completed")
        yield SimpleNamespace(type=f"response.{status}", response=self._final_response)


class _FakeResponses:
    def __init__(self):
        self.calls = 0
        self.create_calls = 0
        self.stream_kwargs = []

    def stream(self, **kwargs):
        self.calls += 1
        self.stream_kwargs.append(kwargs)
        if self.calls == 1:
            tool_call = SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="list_dir",
                arguments='{"path":"."}',
            )
            final = SimpleNamespace(id="resp_1", output=[tool_call], output_text=None)
            return _FakeStream(["Inspecting files..."], final)

        final = SimpleNamespace(id="resp_2", output=[], output_text="Done")
        return _FakeStream(["Do", "ne"], final)

    def create(self, **kwargs):
        self.create_calls += 1
        raise AssertionError(f"unexpected non-stream fallback call: {kwargs}")


def test_tool_mode_streams_text_deltas_instead_of_buffering():
    client = OpenAIClient.__new__(OpenAIClient)
    fake_responses = _FakeResponses()
    client.client = SimpleNamespace(responses=fake_responses)

    tool_spec = SimpleNamespace(
        name="list_dir",
        description="List files",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )

    def _tool_executor(call):
        return ToolResult(
            id=call.id,
            name=call.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output='{"entries":[]}',
        )

    events = list(
        client.stream_events(
            model_id="gpt-5.1",
            messages=[{"role": "user", "content": "inspect repo"}],
            max_tokens=128,
            temperature=0.2,
            top_p=1.0,
            tools=[tool_spec],
            tool_choice="auto",
            tool_executor=_tool_executor,
        )
    )

    deltas = [event.delta for event in events if isinstance(event, TextDeltaEvent)]
    assert deltas == ["Inspecting files...", "Do", "ne"]
    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert any(isinstance(event, FinishEvent) for event in events)
    assert fake_responses.create_calls == 0
    assert fake_responses.calls == 2


class _TruncatedResponses:
    """Every response is incomplete at max_output_tokens; it carries a function call."""

    def stream(self, **kwargs):
        tool_call = SimpleNamespace(
            type="function_call", call_id="call_1", name="bash", arguments='{"comm'
        )
        final = SimpleNamespace(
            id="resp_1",
            output=[tool_call],
            output_text=None,
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        return _FakeStream(["Running"], final)

    def create(self, **kwargs):
        raise AssertionError(f"unexpected non-stream fallback call: {kwargs}")


def test_reply_cut_off_at_max_output_tokens_finishes_with_length_and_runs_no_tools():
    client = OpenAIClient.__new__(OpenAIClient)
    client.client = SimpleNamespace(responses=_TruncatedResponses())
    executed = []
    tool_spec = SimpleNamespace(
        name="bash", description="Run", parameters={"type": "object", "properties": {}}
    )

    events = list(
        client.stream_events(
            model_id="gpt-5.5",
            messages=[{"role": "user", "content": "run it"}],
            max_tokens=64,
            temperature=0.2,
            top_p=1.0,
            tools=[tool_spec],
            tool_executor=executed.append,
        )
    )

    assert [e.reason for e in events if isinstance(e, FinishEvent)] == ["length"]
    assert executed == []
    assert not any(isinstance(e, ToolCallEvent) for e in events)

    plain = list(
        client.stream_events(
            model_id="gpt-5.5",
            messages=[{"role": "user", "content": "explain"}],
            max_tokens=64,
            temperature=0.2,
            top_p=1.0,
        )
    )
    assert [e.reason for e in plain if isinstance(e, FinishEvent)] == ["length"]
