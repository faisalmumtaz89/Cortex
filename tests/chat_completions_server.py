"""In-process OpenAI Chat Completions server for end-to-end tests.

Serves scripted SSE streams from `POST /v1/chat/completions` (one script entry
per request, in order) and records every request body and Authorization header.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List


class ChatCompletionsServer:
    """Script entries: {"text": str} and/or {"tool_calls": [{"name": str, "arguments": dict}]},
    optionally with {"finish": str} to override the finish reason,
    {"delay": seconds} to wait before the reply (a model still reasoning) and
    {"omit_ids": True} to send tool calls without ids, as some servers do."""

    def __init__(self, script: List[Dict[str, object]], *, model: str = "test-model"):
        self.script = list(script)
        self.model = model
        self.requests: List[Dict[str, object]] = []
        self.authorization: List[str] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                self._send_json({"object": "list", "data": [{"id": server.model, "object": "model"}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                server.requests.append(body)
                server.authorization.append(self.headers.get("Authorization", ""))
                step = server.script.pop(0) if server.script else {"text": "(script exhausted)"}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(float(step.get("delay") or 0))
                for chunk in server._chunks(step):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def _send_json(self, payload: Dict[str, object]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def _chunks(self, step: Dict[str, object]) -> List[Dict[str, object]]:
        def chunk(delta: Dict[str, object], finish=None) -> Dict[str, object]:
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        chunks = [chunk({"role": "assistant"})]
        if step.get("text"):
            chunks.append(chunk({"content": step["text"]}))
        calls = list(step.get("tool_calls") or [])
        for index, call in enumerate(calls):
            tool_call = {
                "index": index,
                "id": f"call_{len(self.requests)}_{index}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
            }
            if step.get("omit_ids"):
                del tool_call["id"]
            chunks.append(chunk({"tool_calls": [tool_call]}))
        chunks.append(chunk({}, finish=step.get("finish") or ("tool_calls" if calls else "stop")))
        return chunks

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
