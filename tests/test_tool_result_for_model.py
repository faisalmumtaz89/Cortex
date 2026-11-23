"""What of a tool's result each provider is sent: the result, without what is
for the app's row of the call only."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from cortex.tooling.types import Image, ToolCall, ToolExecutionState, ToolResult
from tests.provider_server import ProviderServer, make_client

_TOOLS = [SimpleNamespace(name="ping", description="ping",
                          parameters={"type": "object", "properties": {}})]


def _edit(call: ToolCall) -> ToolResult:
    return ToolResult(
        id=call.id, name=call.name, state=ToolExecutionState.COMPLETED, ok=True, output="pong",
        metadata={
            "exit_code": 0,
            "diff": {"path": "a.py", "rows": [{"text": "DIFF_ROW"}]},
            "permission_note": "Allowed from now on here",
            "permission_patterns": ["the matched command"],
        },
    )


@pytest.mark.parametrize("kind", ["chat", "responses", "anthropic"])
def test_the_model_is_sent_the_result_without_what_is_for_the_apps_row(kind: str) -> None:
    server = ProviderServer(kind, [{"tool": True}, {}])
    try:
        list(make_client(kind, server.base_url).stream_events(
            model_id="test-model", messages=[{"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0, top_p=1, tools=_TOOLS, tool_executor=_edit,
        ))
    finally:
        server.close()
    sent = json.dumps(server.bodies[1])
    assert "pong" in sent and "exit_code" in sent
    for kept_from_the_model in ("DIFF_ROW", "Allowed from now on here", "the matched command"):
        assert kept_from_the_model not in sent, kept_from_the_model


_PICTURE = Image("image/png", b"\x89PNG\r\n\x1a\nPIXELS")
_PICTURE_B64 = base64.b64encode(_PICTURE.data).decode()


def _read_image(call: ToolCall) -> ToolResult:
    return ToolResult(
        id=call.id, name="read_file", state=ToolExecutionState.COMPLETED, ok=True,
        output="board.png: PNG image, 15 bytes, shown to you as an image",
        metadata={"path": "board.png", "media_type": "image/png", "bytes": 15},
        images=(_PICTURE,),
    )


def _sent_after_the_image_was_read(kind: str) -> dict:
    server = ProviderServer(kind, [{"tool": True}, {}])
    try:
        list(make_client(kind, server.base_url).stream_events(
            model_id="test-model", messages=[{"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0, top_p=1, tools=_TOOLS, tool_executor=_read_image,
        ))
    finally:
        server.close()
    assert _PICTURE_B64 not in json.dumps(server.bodies[0])
    return server.bodies[1]


def test_responses_shows_the_picture_after_the_tool_results() -> None:
    body = _sent_after_the_image_was_read("responses")
    items = body["input"]
    assert items[0]["type"] == "function_call_output"
    assert _PICTURE_B64 not in items[0]["output"]
    assert items[-1] == {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "Image from read_file (board.png):"},
        {"type": "input_image", "image_url": "data:image/png;base64," + _PICTURE_B64,
         "detail": "auto"}]}


def test_anthropic_shows_the_picture_inside_the_tool_result() -> None:
    body = _sent_after_the_image_was_read("anthropic")
    result = body["messages"][-1]["content"][0]
    assert result["type"] == "tool_result"
    text, picture = result["content"]
    assert text["type"] == "text" and json.loads(text["text"])["ok"] is True
    assert picture == {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": _PICTURE_B64}}


def test_chat_completions_shows_the_picture_in_a_message_after_the_tool_results() -> None:
    messages = _sent_after_the_image_was_read("chat")["messages"]
    assert [message["role"] for message in messages[-2:]] == ["tool", "user"]
    assert messages[-1]["content"] == [
        {"type": "text", "text": "Image from read_file (board.png):"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64," + _PICTURE_B64}}]
