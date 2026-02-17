from types import SimpleNamespace

from cortex.cloud.clients.anthropic_client import AnthropicClient
from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolExecutionState,
    ToolResult,
    ToolResultEvent,
)


class _FakeMessageStream:
    def __init__(self, text_deltas, final_message):
        self.text_stream = text_deltas
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_final_message(self):
        return self._final_message


class _FakeMessages:
    def __init__(self):
        self.calls = 0
        self.create_calls = 0
        self.stream_kwargs = []

    def stream(self, **kwargs):
        self.calls += 1
        self.stream_kwargs.append(kwargs)
        if self.calls == 1:
            final_message = SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "list_dir",
                        "input": {"path": "."},
                    }
                ]
            )
            return _FakeMessageStream(["Inspecting ", "repo"], final_message)

        final_message = SimpleNamespace(content=[{"type": "text", "text": "Done"}])
        return _FakeMessageStream(["Do", "ne"], final_message)

    def create(self, **kwargs):
        self.create_calls += 1
        raise AssertionError(f"unexpected non-stream fallback call: {kwargs}")


def test_tool_mode_streams_text_deltas_instead_of_buffering():
    client = AnthropicClient.__new__(AnthropicClient)
    fake_messages = _FakeMessages()
    client.client = SimpleNamespace(messages=fake_messages)

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
            model_id="claude-sonnet-4-5",
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
    assert deltas == ["Inspecting ", "repo", "Do", "ne"]
    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert any(isinstance(event, FinishEvent) for event in events)
    assert fake_messages.create_calls == 0
    assert fake_messages.calls == 2

