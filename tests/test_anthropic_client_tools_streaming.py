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


class _ThinkingFakeMessages:
    """First turn returns thinking(+signature)+tool_use — the Claude 5 family
    shape (verified live 2026-07-05); second turn returns plain text."""

    def __init__(self):
        self.calls = 0
        self.stream_kwargs = []

    def stream(self, **kwargs):
        self.calls += 1
        self.stream_kwargs.append(kwargs)
        if self.calls == 1:
            final_message = SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="thinking",
                        thinking="Let me inspect the directory first.",
                        signature="sig-416-chars-opaque",
                    ),
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "list_dir",
                        "input": {"path": "."},
                    },
                ]
            )
            # Thinking text never reaches text_stream (verified live: the SDK
            # yields no text events for thinking blocks).
            return _FakeMessageStream([], final_message)
        final_message = SimpleNamespace(content=[{"type": "text", "text": "Done"}])
        return _FakeMessageStream(["Done"], final_message)

    def create(self, **kwargs):
        raise AssertionError(f"unexpected non-stream fallback call: {kwargs}")


def test_thinking_blocks_replayed_verbatim_in_tool_continuation():
    """Regression (2026-07-05): Claude 5 models emit thinking blocks by default
    on tool-use turns. The serializer rebuilt them as {"type","text"} — missing
    the REQUIRED `thinking` field — so the continuation request 400'd with
    "messages.N.content.0.thinking.thinking: Field required". Replay must be
    verbatim (thinking + signature), in original block order."""
    client = AnthropicClient.__new__(AnthropicClient)
    fake_messages = _ThinkingFakeMessages()
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
            model_id="claude-sonnet-5",
            messages=[{"role": "user", "content": "inspect repo"}],
            max_tokens=128,
            temperature=0.2,
            top_p=1.0,
            tools=[tool_spec],
            tool_choice="auto",
            tool_executor=_tool_executor,
        )
    )

    # The continuation request replays the assistant turn verbatim.
    assert fake_messages.calls == 2
    continuation = fake_messages.stream_kwargs[1]["messages"]
    assistant = next(m for m in continuation if m["role"] == "assistant")
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "Let me inspect the directory first.",
        "signature": "sig-416-chars-opaque",
    }
    assert assistant["content"][1]["type"] == "tool_use"  # original order kept

    # Internal reasoning never surfaces in user-visible text events.
    text = "".join(e.delta for e in events if isinstance(e, TextDeltaEvent))
    assert "inspect the directory first" not in text
    assert text == "Done"
    assert any(isinstance(e, FinishEvent) for e in events)


def test_empty_thinking_and_redacted_thinking_serialize_verbatim():
    """Live-observed edge (2026-07-05): a thinking block can stream ONLY a
    signature_delta — thinking is "" but the field is still required on replay.
    redacted_thinking is an opaque passthrough."""
    client = AnthropicClient.__new__(AnthropicClient)

    empty_thinking = client._serialize_content_block(
        SimpleNamespace(type="thinking", thinking="", signature="sig-only")
    )
    assert empty_thinking == {"type": "thinking", "thinking": "", "signature": "sig-only"}

    redacted = client._serialize_content_block(
        {"type": "redacted_thinking", "data": "opaque-encrypted-payload"}
    )
    assert redacted == {"type": "redacted_thinking", "data": "opaque-encrypted-payload"}


class _CreateCapturingMessages:
    """Non-stream fake for generate_once: create() captures kwargs and returns
    a minimal valid message; stream() must never be called."""

    def __init__(self):
        self.create_kwargs = []

    def create(self, **kwargs):
        self.create_kwargs.append(kwargs)
        return SimpleNamespace(
            model="claude-fable-5",
            id="msg_1",
            content=[{"type": "text", "text": "fallback"}],
        )

    def stream(self, **kwargs):
        raise AssertionError(f"unexpected stream call from generate_once: {kwargs}")


def test_requests_never_carry_sampling_params():
    """Regression (2026-07-05): Claude-5-era models 400 on `temperature`
    ("deprecated for this model"); 4-x-era models 400 when temperature and
    top_p are both sent. The client must send NEITHER on ANY of its three
    request builders (tool loop, plain stream, non-stream generate_once) —
    API defaults work on every generation (validated live)."""
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

    # Tool loop + plain streaming path both go through messages.stream.
    list(
        client.stream_events(
            model_id="claude-fable-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            tools=[tool_spec],
            tool_choice="auto",
            tool_executor=_tool_executor,
        )
    )
    list(
        client.stream_events(
            model_id="claude-fable-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
        )
    )

    assert fake_messages.stream_kwargs, "no requests captured"
    for kwargs in fake_messages.stream_kwargs:
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs

    # generate_once goes through messages.create — the reliability fallback
    # (router falls back to it when streaming yields nothing) must be equally
    # clean or it 400s exactly when the primary path already failed.
    create_client = AnthropicClient.__new__(AnthropicClient)
    create_fake = _CreateCapturingMessages()
    create_client.client = SimpleNamespace(messages=create_fake)
    text = create_client.generate_once(
        model_id="claude-fable-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
    )
    assert text == "fallback"
    assert create_fake.create_kwargs, "generate_once sent no request"
    for kwargs in create_fake.create_kwargs:
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs


def test_unknown_blocks_replay_their_own_shape_verbatim():
    """The serializer must never fabricate a text block from a server block it
    does not recognize: an SDK object replays its model_dump(), a plain dict
    replays as-is, and only a typeless block falls back to text."""
    client = AnthropicClient.__new__(AnthropicClient)

    class _SdkBlock:
        type = "server_tool_use"

        def model_dump(self):
            return {"type": "server_tool_use", "id": "st_1", "name": "web_search"}

    assert client._serialize_content_block(_SdkBlock()) == {
        "type": "server_tool_use",
        "id": "st_1",
        "name": "web_search",
    }

    dict_block = {"type": "future_block", "payload": {"x": 1}}
    replayed = client._serialize_content_block(dict_block)
    assert replayed == dict_block
    assert replayed is not dict_block  # defensive copy, caller's dict untouched

    typeless = client._serialize_content_block(SimpleNamespace(text="plain"))
    assert typeless == {"type": "text", "text": "plain"}

