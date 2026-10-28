"""In-process server speaking the streaming APIs of all three cloud clients
(Chat Completions, Anthropic Messages, OpenAI Responses) the way real
providers do: kept-alive connections and chunked SSE bodies.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List

from cortex.cloud.clients import AnthropicClient, ChatCompletionsClient, OpenAIClient


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
        item = {"id": "msg_1", "type": "message", "role": "assistant",
                "status": "in_progress", "content": []}
        part = {"type": "output_text", "text": "", "annotations": []}
        text_events = [
            _sse("response.output_item.added", {
                "type": "response.output_item.added", "output_index": 0, "item": item,
                "sequence_number": 1}),
            _sse("response.content_part.added", {
                "type": "response.content_part.added", "item_id": "msg_1", "output_index": 0,
                "content_index": 0, "part": part, "sequence_number": 1}),
            _sse("response.output_text.delta", {
                "type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
                "content_index": 0, "delta": "late", "sequence_number": 1, "logprobs": []}),
        ]
    completed = dict(response, status="completed", output=output)
    return [
        _sse("response.created",
             {"type": "response.created", "response": response, "sequence_number": 0}),
        *text_events,
        _sse("response.completed",
             {"type": "response.completed", "response": completed, "sequence_number": 2}),
    ]


_EVENTS = {"chat": _chat_events, "anthropic": _anthropic_events, "responses": _responses_events}

_ERROR_EVENTS = {
    "chat": b'data: {"error": {"code": 502, "message": "upstream failed"}}\n\n',
    "anthropic": _sse("error", {"type": "error",
                                "error": {"type": "overloaded_error", "message": "Overloaded"}}),
    "responses": _sse("error", {"type": "error", "code": "server_error", "message": "boom",
                                "param": None, "sequence_number": 1}),
}


class ProviderServer:
    """Script entries: {"tool": True} for a tool call, else a text reply;
    "header_delay" / "body_delay" seconds before the headers / the body;
    {"reset": True} resets the connection after the first event, {"error": True}
    sends the API's error event (overloaded / server error) after it."""

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
                    if step.get("reset"):
                        linger = struct.pack("ii", 1, 0)
                        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
                        self.close_connection = True
                        return
                    if step.get("error"):
                        events = [_ERROR_EVENTS[kind]]
                    time.sleep(float(step.get("body_delay", 0)))
                    for event in events[1:] if not step.get("error") else events:
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


def make_client(kind: str, base_url: str):
    if kind == "chat":
        return ChatCompletionsClient(base_url=f"{base_url}/v1", api_key="k", timeout_seconds=60)
    if kind == "responses":
        return OpenAIClient(api_key="k", timeout_seconds=60, base_url=f"{base_url}/v1")
    return AnthropicClient(api_key="k", timeout_seconds=60, base_url=base_url)
