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
    def __init__(self, deltas, final_response):
        self.text_deltas = deltas
        self._final_response = final_response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_final_response(self):
        return self._final_response


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

