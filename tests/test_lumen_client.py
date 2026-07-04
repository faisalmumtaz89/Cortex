"""LumenClient: chat-completions tool loop and wire hygiene."""

from __future__ import annotations

import json
from types import SimpleNamespace

from cortex.cloud.clients import LumenClient
from cortex.tooling.types import (
    FinishEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolExecutionState,
    ToolResult,
    ToolResultEvent,
)


def _chunk(*, content=None, tool_calls=None, finish=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice])


def _tool_delta(index, *, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeCompletions:
    def __init__(self, streams):
        self._streams = list(streams)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return iter(self._streams.pop(0))


def _client_with(streams) -> tuple[LumenClient, _FakeCompletions]:
    client = LumenClient(base_url="http://127.0.0.1:1/v1")
    completions = _FakeCompletions(streams)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_plain_text_stream_yields_deltas_and_finish() -> None:
    client, completions = _client_with(
        [[_chunk(content="Hello"), _chunk(content=" world", finish="stop")]]
    )
    events = list(
        client.stream_events(
            model_id="qwen3-5-9b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=32,
            temperature=0.7,
            top_p=1.0,
        )
    )
    assert [e.delta for e in events if isinstance(e, TextDeltaEvent)] == ["Hello", " world"]
    assert isinstance(events[-1], FinishEvent)
    # No tools requested → no tools/tool_choice fields on the wire.
    assert "tools" not in completions.requests[0]
    assert "tool_choice" not in completions.requests[0]


def test_tool_loop_executes_and_feeds_results_back() -> None:
    spec = SimpleNamespace(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    first = [
        _chunk(tool_calls=[_tool_delta(0, id="call_1", name="read_file", arguments='{"pa')]),
        _chunk(tool_calls=[_tool_delta(0, arguments='th": "calc.py"}')], finish="tool_calls"),
    ]
    second = [_chunk(content="add returns a + b", finish="stop")]
    client, completions = _client_with([first, second])

    executed = []

    def executor(call):
        executed.append(call)
        return ToolResult(
            id=call.id,
            name=call.name,
            state=ToolExecutionState.COMPLETED,
            ok=True,
            output="def add(a, b): return a + b",
            metadata={"path": "calc.py", "diff": {"huge": "ui-only"}},
        )

    events = list(
        client.stream_events(
            model_id="qwen3-5-9b",
            messages=[{"role": "user", "content": "read calc.py"}],
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            tools=[spec],
            tool_executor=executor,
        )
    )

    kinds = [type(e).__name__ for e in events]
    assert kinds.count("ToolCallEvent") == 1
    assert kinds.count("ToolResultEvent") == 1
    assert isinstance(events[-1], FinishEvent)

    call_event = next(e for e in events if isinstance(e, ToolCallEvent))
    assert call_event.call.arguments == {"path": "calc.py"}
    assert executed and executed[0].id == "call_1"

    # Wire hygiene: tools sent without tool_choice (Lumen 400s on it).
    assert "tools" in completions.requests[0]
    assert "tool_choice" not in completions.requests[0]

    # Second request carries the assistant tool_calls + tool result, with the
    # UI-only diff stripped from the model-facing metadata.
    followup = completions.requests[1]["messages"]
    assert followup[-2]["role"] == "assistant"
    assert followup[-2]["tool_calls"][0]["function"]["name"] == "read_file"
    tool_message = followup[-1]
    assert tool_message["role"] == "tool"
    payload = json.loads(tool_message["content"])
    assert payload["ok"] is True
    assert "diff" not in payload["metadata"]
    assert payload["metadata"]["path"] == "calc.py"


def test_tool_result_event_matches_executor_output() -> None:
    spec = SimpleNamespace(name="t", description="d", parameters={})
    first = [
        _chunk(tool_calls=[_tool_delta(0, id="c1", name="t", arguments="{}")], finish="tool_calls")
    ]
    second = [_chunk(content="done", finish="stop")]
    client, _ = _client_with([first, second])

    result = ToolResult(
        id="c1", name="t", state=ToolExecutionState.COMPLETED, ok=True, output="x"
    )
    events = list(
        client.stream_events(
            model_id="m",
            messages=[{"role": "user", "content": "go"}],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            tools=[spec],
            tool_executor=lambda call: result,
        )
    )
    assert next(e for e in events if isinstance(e, ToolResultEvent)).result is result
