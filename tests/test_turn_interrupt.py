"""Interrupting a turn while each real SDK client waits on the model.

The server keeps connections alive and sends chunked bodies, as the real
providers do, so the SDKs pool connections and read the way they do live.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Dict, List

import pytest

from cortex.cloud.clients import AnthropicClient, ChatCompletionsClient, OpenAIClient
from cortex.tooling.types import (
    TextDeltaEvent,
    ToolExecutionState,
    ToolResult,
    TurnInterrupt,
    TurnInterruptedError,
)


def _sse(event: str, data: Dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _chat_events(step: Dict[str, object]) -> List[bytes]:
    def chunk(delta: Dict[str, object], finish=None) -> bytes:
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    if step.get("tool"):
        call = {"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "ping", "arguments": "{}"}}
        body = [chunk({"tool_calls": [call]}), chunk({}, "tool_calls")]
    else:
        body = [chunk({"content": "late"}), chunk({}, "stop")]
    return [chunk({"role": "assistant"}), *body, b"data: [DONE]\n\n"]


def _anthropic_events(step: Dict[str, object]) -> List[bytes]:
    message = {"id": "msg_1", "type": "message", "role": "assistant", "model": "test-model",
               "content": [], "stop_reason": None, "stop_sequence": None,
               "usage": {"input_tokens": 1, "output_tokens": 0}}
    if step.get("tool"):
        block = {"type": "tool_use", "id": "toolu_1", "name": "ping", "input": {}}
        delta = {"type": "input_json_delta", "partial_json": "{}"}
        stop_reason = "tool_use"
    else:
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": "late"}
        stop_reason = "end_turn"
    return [
        _sse("message_start", {"type": "message_start", "message": message}),
        _sse("content_block_start",
             {"type": "content_block_start", "index": 0, "content_block": block}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                               "usage": {"output_tokens": 1}}),
        _sse("message_stop", {"type": "message_stop"}),
    ]


def _responses_events(step: Dict[str, object]) -> List[bytes]:
    response = {"id": "resp_1", "object": "response", "created_at": 0, "model": "test-model",
                "status": "in_progress", "output": [], "parallel_tool_calls": True,
                "tool_choice": "auto", "tools": []}
    if step.get("tool"):
        output = [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "ping",
                   "arguments": "{}", "status": "completed"}]
        text_events = []
    else:
        output = [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": "late", "annotations": []}]}]
        text_events = [_sse("response.output_text.delta", {
            "type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
            "content_index": 0, "delta": "late", "sequence_number": 1, "logprobs": []})]
    completed = dict(response, status="completed", output=output)
    return [
        _sse("response.created",
             {"type": "response.created", "response": response, "sequence_number": 0}),
        *text_events,
        _sse("response.completed",
             {"type": "response.completed", "response": completed, "sequence_number": 2}),
    ]


_EVENTS = {"chat": _chat_events, "anthropic": _anthropic_events, "responses": _responses_events}


class _KeepAliveServer:
    """Script entries: {"tool": True} for a tool call, else a text reply;
    "header_delay" / "body_delay" seconds before the headers / the body."""

    def __init__(self, kind: str, script: List[Dict[str, object]]):
        self.script = list(script)
        self.requests: List[float] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.requests.append(time.time())
                step = server.script.pop(0) if server.script else {}
                time.sleep(float(step.get("header_delay", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                events = _EVENTS[kind](step)
                try:
                    self._chunk(events[0])
                    time.sleep(float(step.get("body_delay", 0)))
                    for event in events[1:]:
                        self._chunk(event)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    self.close_connection = True

            def _chunk(self, data: bytes) -> None:
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _client(kind: str, base_url: str, monkeypatch):
    if kind == "chat":
        return ChatCompletionsClient(base_url=f"{base_url}/v1", api_key="k", timeout_seconds=60)
    if kind == "responses":
        return OpenAIClient(api_key="k", timeout_seconds=60, base_url=f"{base_url}/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    return AnthropicClient(api_key="k", timeout_seconds=60)


_TOOLS = [SimpleNamespace(name="ping", description="ping",
                          parameters={"type": "object", "properties": {}})]


def _ping(call) -> ToolResult:
    return ToolResult(id=call.id, name=call.name, state=ToolExecutionState.COMPLETED, ok=True,
                      output="pong")


def _run_interrupted(kind, script, *, tools, interrupt_after, monkeypatch):
    """Run one client turn, interrupt it after `interrupt_after` seconds and
    return (outcome, seconds from the interrupt to the end, server)."""
    server = _KeepAliveServer(kind, script)
    client = _client(kind, server.base_url, monkeypatch)
    interrupt = TurnInterrupt()
    outcome: Dict[str, object] = {}

    def run() -> None:
        try:
            events = list(client.stream_events(
                model_id="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0,
                top_p=1,
                tools=_TOOLS if tools else None,
                tool_executor=_ping if tools else None,
                interrupt=interrupt,
            ))
            outcome["events"] = events
        except BaseException as exc:
            outcome["error"] = exc
        outcome["ended"] = time.time()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(interrupt_after)
    interrupted_at = time.time()
    interrupt.set()
    thread.join(30)
    server.close()
    return outcome, outcome["ended"] - interrupted_at, server


@pytest.mark.parametrize("tools", [True, False], ids=["tools", "no-tools"])
@pytest.mark.parametrize("kind", ["chat", "anthropic", "responses"])
def test_interrupt_stops_a_reply_the_model_is_still_working_on(kind, tools, monkeypatch):
    outcome, waited, server = _run_interrupted(
        kind, [{"body_delay": 30}], tools=tools, interrupt_after=1, monkeypatch=monkeypatch
    )

    assert isinstance(outcome.get("error"), TurnInterruptedError), outcome
    assert waited < 2
    assert len(server.requests) == 1


@pytest.mark.parametrize("kind", ["chat", "anthropic", "responses"])
def test_interrupt_before_the_next_reply_starts_sends_no_further_request(kind, monkeypatch):
    # The interrupt arrives while the second request, on the first one's
    # pooled connection, still waits for its response headers.
    outcome, waited, server = _run_interrupted(
        kind,
        [{"tool": True}, {"header_delay": 3}, {}],
        tools=True,
        interrupt_after=1.5,
        monkeypatch=monkeypatch,
    )

    assert isinstance(outcome.get("error"), TurnInterruptedError), outcome
    assert len(server.requests) == 2
    assert waited < 3


def test_uninterrupted_turn_still_reads_the_whole_reply(monkeypatch):
    server = _KeepAliveServer("anthropic", [{}])
    client = _client("anthropic", server.base_url, monkeypatch)
    try:
        events = list(client.stream_events(
            model_id="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0,
            top_p=1,
            interrupt=TurnInterrupt(),
        ))
    finally:
        server.close()

    assert [e.delta for e in events if isinstance(e, TextDeltaEvent)] == ["late"]
