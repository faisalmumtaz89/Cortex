import json
import os
import subprocess
import sys
import time
from pathlib import Path

LUMEN_LISTING_CACHED = """Cached models:

  qwen3-5-9b-Q4_0                          5.4 GB

Available to download:
  qwen3-5-9b           Qwen3.5 9B Q8_0
  qwen3-6-27b          Qwen3.6 27B Q4_0

Download with: lumen pull <model-name> [--quant Q8_0]
"""

LUMEN_LISTING_EMPTY_CACHE = """Cached models:


Available to download:
  qwen3-5-9b           Qwen3.5 9B Q4_0
  qwen3-5-9b           Qwen3.5 9B Q8_0

Download with: lumen pull <model-name> [--quant Q8_0]
"""


LUMEN_LISTING_ALL_CACHED = """Cached models:

  qwen3-5-9b-Q4_0                          5.4 GB
  qwen3-6-27b-Q4_0                         14.1 GB

Available to download:
  qwen3-5-9b           Qwen3.5 9B Q8_0
  qwen3-6-27b          Qwen3.6 27B Q4_0

Download with: lumen pull <model-name> [--quant Q8_0]
"""


def _lumen_stub_env(
    tmp_path: Path,
    *,
    cached: bool = True,
    with_server: bool = False,
    report_model: str | None = None,
) -> dict:
    """Worker env with a deterministic fake `lumen` CLI (no host cache/net).

    The fake `pull` is STATEFUL: it streams two progress lines and rewrites the
    models listing to the all-cached variant, so a post-pull auto-load resolves
    the selector as cached (mirroring real lumen). `with_server=True` also
    provides a working fake `lumen-server` (binds --port, serves /v1/models)
    so ensure_server/auto-load succeed end to end.
    """
    listing = LUMEN_LISTING_CACHED if cached else LUMEN_LISTING_EMPTY_CACHE
    fixture = tmp_path / "lumen-models.txt"
    fixture.write_text(listing, encoding="utf-8")
    cached_fixture = tmp_path / "lumen-models-cached.txt"
    cached_fixture.write_text(LUMEN_LISTING_ALL_CACHED, encoding="utf-8")
    stub = tmp_path / "fake-lumen"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "if sys.argv[1:2] == ['models']:\n"
        f"    print(open({str(fixture)!r}).read(), end='')\n"
        "elif sys.argv[1:2] == ['pull']:\n"
        "    print('downloading 50%', flush=True)\n"
        "    print('converted to LBC', flush=True)\n"
        f"    shutil.copyfile({str(cached_fixture)!r}, {str(fixture)!r})\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env["CORTEX_LUMEN_BINARY"] = str(stub)
    if with_server:
        server = tmp_path / "fake-lumen-server"
        server.write_text(
            "#!/usr/bin/env python3\n"
            "import http.server, json, os, sys\n"
            "args = sys.argv[1:]\n"
            "port = int(args[args.index('--port') + 1])\n"
            "model = args[args.index('--model') + 1]\n"
            "# What the server CLAIMS answered — imposter tests override this.\n"
            "report = os.environ.get('FAKE_LUMEN_REPORT_MODEL') or model\n"
            "class H(http.server.BaseHTTPRequestHandler):\n"
            "    protocol_version = 'HTTP/1.1'\n"
            "    def do_GET(self):\n"
            "        body = json.dumps({'object': 'list', 'data': [{'id': model}]}).encode()\n"
            "        self.send_response(200 if self.path == '/v1/models' else 404)\n"
            "        self.send_header('Content-Length', str(len(body)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(body)\n"
            "    def do_POST(self):\n"
            "        length = int(self.headers.get('Content-Length') or 0)\n"
            "        self.rfile.read(length)\n"
            "        if self.path != '/v1/chat/completions':\n"
            "            self.send_response(404)\n"
            "            self.send_header('Content-Length', '0')\n"
            "            self.end_headers()\n"
            "            return\n"
            "        def chunk(payload):\n"
            "            return ('data: ' + json.dumps(payload) + '\\n\\n').encode()\n"
            "        frames = [\n"
            "            chunk({'id': 'chatcmpl-fake-1', 'model': report, 'object': 'chat.completion.chunk',\n"
            "                   'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None, 'index': 0}]}),\n"
            "            chunk({'id': 'chatcmpl-fake-1', 'model': report, 'object': 'chat.completion.chunk',\n"
            "                   'choices': [{'delta': {'content': 'hello from the local engine'}, 'finish_reason': None, 'index': 0}]}),\n"
            "            chunk({'id': 'chatcmpl-fake-1', 'model': report, 'object': 'chat.completion.chunk',\n"
            "                   'choices': [{'delta': {}, 'finish_reason': 'stop', 'index': 0}]}),\n"
            "            b'data: [DONE]\\n\\n',\n"
            "        ]\n"
            "        body = b''.join(frames)\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-Type', 'text/event-stream')\n"
            "        self.send_header('Content-Length', str(len(body)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(body)\n"
            "    def log_message(self, *a):\n"
            "        pass\n"
            "server = http.server.ThreadingHTTPServer(('127.0.0.1', port), H)\n"
            "server.serve_forever()\n",
            encoding="utf-8",
        )
        server.chmod(0o755)
        env["CORTEX_LUMEN_SERVER_BINARY"] = str(server)
        if report_model:
            env["FAKE_LUMEN_REPORT_MODEL"] = report_model
    else:
        env["CORTEX_LUMEN_SERVER_BINARY"] = str(tmp_path / "missing-lumen-server")
    env.pop("CORTEX_SCRIPTED_MODEL", None)  # these paths exercise the REAL router
    return env


def test_worker_stdio_handshake_emits_jsonrpc_on_stdout_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "app.handshake",
        "params": {"protocol_version": "1.0.0", "client_name": "pytest"},
    }

    process = subprocess.run(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        input=json.dumps(request) + "\n",
        text=True,
        cwd=repo_root,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0

    stdout_lines = [line for line in process.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1

    response = json.loads(stdout_lines[0])
    assert response["id"] == 1
    assert response["result"]["protocol_version"] == "1.0.0"


def test_worker_handshake_rejects_protocol_mismatch() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "app.handshake",
        "params": {"protocol_version": "9.9.9", "client_name": "pytest"},
    }

    process = subprocess.run(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        input=json.dumps(request) + "\n",
        text=True,
        cwd=repo_root,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0
    response = json.loads(process.stdout.strip().splitlines()[0])
    assert response["error"]["message"] == "Protocol version mismatch"


def test_worker_command_execute_emits_system_notice_event() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, events = recv_until(2)
        session_id = response["result"]["session_id"]
        assert any(event["params"]["event_type"] == "system.notice" for event in events)

        send(3, "command.execute", {"session_id": session_id, "command": "/help"})
        command_response, command_events = recv_until(3)
        assert command_response["result"]["ok"] is True
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Commands:" in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_download_command_returns_validation_error_and_notice() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(3, "command.execute", {"session_id": session_id, "command": "/download badrepo"})
        command_response, command_events = recv_until(3)

        assert command_response["result"]["ok"] is False
        assert "Unknown local model 'badrepo'" in command_response["result"]["message"]
        assert any(
            event["params"]["event_type"] == "session.status"
            and event["params"]["payload"].get("status") == "busy"
            for event in command_events
        )
        assert any(
            event["params"]["event_type"] == "session.status"
            and event["params"]["payload"].get("status") == "idle"
            for event in command_events
        )
        assert not any(
            event["params"]["event_type"] == "system.notice"
            and "Download of" in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Unknown local model" in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_download_without_args_shows_usage_without_download_started_notice(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _lumen_stub_env(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(3, "command.execute", {"session_id": session_id, "command": "/download"})
        command_response, command_events = recv_until(3)

        assert command_response["result"]["ok"] is False
        assert "Usage: /download <model[:quant]>" in str(command_response["result"]["message"])
        assert not any(
            event["params"]["event_type"] == "system.notice"
            and "Download of" in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_download_existing_model_loads_it_without_background_start(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _lumen_stub_env(tmp_path, with_server=True)

    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(
            3,
            "command.execute",
            {"session_id": session_id, "command": "/download qwen3-5-9b:q4_0"},
        )
        command_response, command_events = recv_until(3)
        # One flow: an already-downloaded selector simply loads and activates.
        assert command_response["result"]["ok"] is True
        assert "Already downloaded: local · qwen3-5-9b:q4_0 — now active." in str(
            command_response["result"]["message"]
        )
        assert "background" not in command_response["result"]
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "now active" in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        subprocess.run(
            ["pkill", "-f", str(tmp_path / "fake-lumen-server")],
            check=False,
            capture_output=True,
        )


def test_worker_download_cancel_returns_actionable_message_when_idle(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _lumen_stub_env(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(3, "command.execute", {"session_id": session_id, "command": "/download cancel"})
        cancel_response, cancel_events = recv_until(3)
        assert cancel_response["result"]["ok"] is False
        assert cancel_response["result"]["message"] == "No active download to cancel."
        assert any(
            event["params"]["event_type"] == "system.notice"
            and event["params"]["payload"].get("message") == "No active download to cancel."
            for event in cancel_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_invalid_provider_commands_return_result_not_rpc_error(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _lumen_stub_env(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(3, "command.execute", {"session_id": session_id, "command": "/model unknown:gpt"})
        model_response, model_events = recv_until(3)
        assert "result" in model_response
        assert model_response["result"]["ok"] is False
        assert "Unknown local model 'unknown'" in str(model_response["result"]["message"])
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Unknown local model" in str(event["params"]["payload"].get("message", ""))
            for event in model_events
        )

        send(4, "command.execute", {"session_id": session_id, "command": "/login unknown"})
        login_response, login_events = recv_until(4)
        assert "result" in login_response
        assert login_response["result"]["ok"] is False
        assert "Unsupported provider" in str(login_response["result"]["message"])
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Unsupported provider" in str(event["params"]["payload"].get("message", ""))
            for event in login_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_setup_command_returns_result_with_notice(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _lumen_stub_env(tmp_path, cached=False)
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(3, "command.execute", {"session_id": session_id, "command": "/setup"})
        setup_response, setup_events = recv_until(3)
        assert "result" in setup_response
        assert setup_response["result"]["ok"] is False
        assert "Pick one with /model" in str(setup_response["result"]["message"])
        assert "/download" not in str(setup_response["result"]["message"])
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Pick one with /model" in str(event["params"]["payload"].get("message", ""))
            for event in setup_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_benchmark_command_reports_missing_model_with_notice(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _lumen_stub_env(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(3, "command.execute", {"session_id": session_id, "command": "/benchmark"})
        command_response, command_events = recv_until(3)

        assert command_response["result"]["ok"] is False
        assert "No local model loaded." in command_response["result"]["message"]
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "No local model loaded." in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_submit_user_input_without_model_returns_result_not_rpc_error(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _lumen_stub_env(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(3, "session.submit_user_input", {"session_id": session_id, "user_input": "hello"})
        submit_response, submit_events = recv_until(3)

        assert "result" in submit_response
        assert submit_response["result"]["ok"] is False
        assert "No model loaded" in str(submit_response["result"]["error"])
        assert isinstance(submit_response["result"].get("assistant_message_id"), str)
        assert submit_response["result"]["assistant_message_id"]
        assert any(event["params"]["event_type"] == "session.error" for event in submit_events)
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_command_matrix_covers_all_core_command_paths(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _lumen_stub_env(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        command_cases = [
            ("/help", True, "Commands:"),
            ("/status", True, "System status"),
            ("/gpu", True, "GPU status"),
            # /models was consolidated into /model; /template died with the
            # in-process template registry (Lumen renders chat templates).
            ("/models", False, "Unknown command: /models"),
            ("/template status", False, "Unknown command: /template"),
            ("/model", True, "Active model:"),
            ("/model list", True, "Active model:"),
            ("/model ls", True, "Active model:"),
            ("/model unknown:gpt", False, "Unknown local model"),
            ("/model openai:", False, "Usage: /model <provider:model>"),
            ("/login", False, "Usage: /login openai|anthropic"),
            ("/login unknown", False, "Unsupported provider"),
            ("/login lumen", False, "managed local engine"),
            ("/login openai", True, "Authentication status"),
            ("/download", False, "Usage: /download <model[:quant]>"),
            ("/download badrepo", False, "Unknown local model"),
            ("/download user/repo file.gguf extra", False, "Usage: /download <model[:quant]>"),
            ('/download "unterminated', False, "Invalid download arguments:"),
            ("/benchmark", False, "No local model loaded."),
            ("/benchmark --tokens nope", False, "positive integer"),
            ("/benchmark --prompt", False, "Usage: /benchmark [tokens] [--prompt <text>]"),
            ("/benchmark --tokens 9000", False, "between 1 and 8192"),
            ('/benchmark --prompt ""', False, "Benchmark prompt cannot be empty."),
            ("/setup", True, "Loading qwen3-5-9b:q4_0…"),
            ("/clear", True, "cleared"),
            ("/save", True, "Saved conversation:"),
            ("download badrepo", False, "Not a slash command: download badrepo"),
            ("/unknown", False, "Unknown command: /unknown"),
            ("/quit", True, None),
            ("/exit", True, None),
        ]

        next_id = 3
        for command, expected_ok, expected_message in command_cases:
            send(next_id, "command.execute", {"session_id": session_id, "command": command})
            command_response, command_events = recv_until(next_id)

            assert "result" in command_response
            assert command_response["result"]["ok"] is expected_ok
            background = bool(command_response["result"].get("background"))
            if background:
                # Background model operations are NOT turns: they must never
                # touch session.status (the turn spinner stays out of it).
                assert not any(
                    event["params"]["event_type"] == "session.status"
                    for event in command_events
                ), "background op leaked a session.status event"
                # Drain to the operation's FINAL progress frame so the stream
                # stays aligned for the next command.
                deadline = time.time() + 30
                drained = False
                while time.time() < deadline:
                    line = process.stdout.readline()
                    if not line:
                        break
                    frame = json.loads(line)
                    if frame.get("method") != "event":
                        continue
                    command_events.append(frame)
                    params = frame["params"]
                    if params["event_type"] == "session.status":
                        raise AssertionError("background op leaked a session.status event")
                    if params["event_type"] == "message.updated" and params["payload"].get(
                        "final"
                    ):
                        drained = True
                        break
                assert drained, "background operation never emitted its final frame"
            else:
                assert any(
                    event["params"]["event_type"] == "session.status"
                    and event["params"]["payload"].get("status") == "busy"
                    for event in command_events
                )
                assert any(
                    event["params"]["event_type"] == "session.status"
                    and event["params"]["payload"].get("status") == "idle"
                    for event in command_events
                )

            if expected_message is not None:
                assert expected_message in str(command_response["result"].get("message", ""))
                if command_response["result"].get("background") is True:
                    # Background model ops narrate through their own resolving
                    # progress message — no duplicate system.notice by design.
                    assert any(
                        event["params"]["event_type"] == "message.updated"
                        and expected_message
                        in str(event["params"]["payload"].get("content", ""))
                        for event in command_events
                    )
                else:
                    assert any(
                        event["params"]["event_type"] == "system.notice"
                        and expected_message in str(event["params"]["payload"].get("message", ""))
                        for event in command_events
                    )

            if command.startswith("/download"):
                if "user/repo" in command and "file.gguf extra" not in command:
                    # Potentially valid downloads emit this before work starts; invalid inputs must not.
                    pass
                else:
                    assert not any(
                        event["params"]["event_type"] == "system.notice"
                        and "Download request accepted."
                        in str(event["params"]["payload"].get("message", ""))
                        for event in command_events
                    )

            next_id += 1
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_model_select_uncached_downloads_and_autoloads(tmp_path: Path) -> None:
    """One-flow /model: selecting an uncached model starts the background
    download and, on completion, auto-loads it — no /download dead end."""
    env = _lumen_stub_env(tmp_path, with_server=True)
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    def read_events_until(predicate, timeout: float = 60.0):
        deadline = time.time() + timeout
        seen = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                seen.append(frame)
                if predicate(frame):
                    return seen
        raise AssertionError(f"timed out waiting for event; saw {len(seen)} events")

    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _events = recv_until(2)
        session_id = response["result"]["session_id"]

        send(
            3,
            "command.execute",
            {"session_id": session_id, "command": "/model qwen3-6-27b:q4_0"},
        )
        command_response, command_events = recv_until(3)
        result = command_response["result"]
        assert result["ok"] is True
        assert result["background"] is True
        assert "Downloading qwen3-6-27b:q4_0 — loads automatically when done" in str(
            result["message"]
        )
        # The immediate reply never dead-ends into a /download instruction.
        assert "Run /download" not in str(result["message"])

        # Background completion: progress frames stream, then the operation's
        # OWN message resolves in place (final frame carries the ready text —
        # no duplicate system.notice row).
        seen = read_events_until(
            lambda frame: frame["params"]["event_type"] == "message.updated"
            and frame["params"]["payload"].get("final") is True
            and "local · qwen3-6-27b:q4_0 ready — now active." in str(
                frame["params"]["payload"].get("content", "")
            )
        )
        progress_frames = [
            frame
            for frame in [*command_events, *seen]
            if frame["params"]["event_type"] == "message.updated"
            and isinstance(frame["params"]["payload"].get("progress"), dict)
        ]
        assert progress_frames, "expected streamed download progress frames"
        kinds = []
        for frame in progress_frames:
            progress = frame["params"]["payload"]["progress"]
            assert progress["kind"] in {"download", "model-load"}
            assert progress["repo_id"] == "qwen3-6-27b:q4_0"
            kinds.append(progress["kind"])
        # The pull streams as "download", then the chained GPU load flips the
        # SAME operation to "model-load" — the UI's phase transition.
        assert "download" in kinds and "model-load" in kinds
        assert kinds.index("download") < kinds.index("model-load")

        send(4, "model.list", {})
        list_response, _more = recv_until(4)
        active = list_response["result"]["active_target"]
        assert active["backend"] == "local"
        assert active["local_model"] == "qwen3-6-27b:q4_0"
    finally:
        process.terminate()
        process.wait(timeout=5)
        subprocess.run(
            ["pkill", "-f", str(tmp_path / "fake-lumen-server")],
            check=False,
            capture_output=True,
        )


def _worker_session(tmp_path: Path, env: dict):
    """Spawn a worker and return (process, send, recv_until)."""
    repo_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 90.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    return process, send, recv_until


def test_local_turn_provenance_verified_end_to_end(tmp_path: Path) -> None:
    """Real worker + real router + fake lumen server reporting the CORRECT
    model: the turn succeeds and the final frame carries VERIFIED provenance."""
    env = _lumen_stub_env(tmp_path, with_server=True)
    process, send, recv_until = _worker_session(tmp_path, env)
    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _ = recv_until(2)
        session_id = response["result"]["session_id"]

        # Activate the fake local model (cached → loads synchronously).
        send(3, "command.execute", {"session_id": session_id, "command": "/download qwen3-5-9b:q4_0"})
        command_response, _ = recv_until(3)
        assert command_response["result"]["ok"] is True

        send(4, "session.submit_user_input", {"session_id": session_id, "user_input": "hello"})
        submit_response, submit_events = recv_until(4)
        assert submit_response["result"].get("ok", True) is not False, submit_response
        assert "hello from the local engine" in submit_response["result"]["assistant_text"]

        finals = [
            event["params"]["payload"]
            for event in submit_events
            if event["params"]["event_type"] == "message.updated"
            and event["params"]["payload"].get("role") == "assistant"
            and event["params"]["payload"].get("final")
        ]
        assert finals, "no final assistant frame"
        final = finals[-1]
        assert final["backend"] == "local"
        assert final["model_label"] == "qwen3-5-9b:q4_0"
        assert final["provenance_verified"] is True

        # /status reports what actually served the last turn.
        send(5, "command.execute", {"session_id": session_id, "command": "/status"})
        status_response, _ = recv_until(5)
        assert "Last turn served by: local · qwen3-5-9b:q4_0 (verified)" in str(
            status_response["result"]["message"]
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        subprocess.run(
            ["pkill", "-f", str(tmp_path / "fake-lumen-server")],
            check=False,
            capture_output=True,
        )


def test_imposter_local_server_fails_provenance_and_rejects_turn(tmp_path: Path) -> None:
    """ADVERSARIAL: the server on the managed port answers with a DIFFERENT
    model name. The turn must fail with a provenance error — it may never
    render as a normal answer."""
    env = _lumen_stub_env(tmp_path, with_server=True, report_model="evil-model")
    process, send, recv_until = _worker_session(tmp_path, env)
    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, _ = recv_until(2)
        session_id = response["result"]["session_id"]

        send(3, "command.execute", {"session_id": session_id, "command": "/download qwen3-5-9b:q4_0"})
        command_response, _ = recv_until(3)
        assert command_response["result"]["ok"] is True

        send(4, "session.submit_user_input", {"session_id": session_id, "user_input": "hello"})
        submit_response, submit_events = recv_until(4)

        assert submit_response["result"]["ok"] is False
        error_text = str(submit_response["result"]["error"])
        assert "provenance mismatch" in error_text
        assert "evil-model" in error_text
        # The rejected turn is not presented as a verified answer.
        finals = [
            event["params"]["payload"]
            for event in submit_events
            if event["params"]["event_type"] == "message.updated"
            and event["params"]["payload"].get("role") == "assistant"
            and event["params"]["payload"].get("final")
        ]
        assert all(not payload.get("provenance_verified") for payload in finals)
        assert any(
            event["params"]["event_type"] == "session.error"
            and "provenance mismatch" in str(event["params"]["payload"].get("error", ""))
            for event in submit_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        subprocess.run(
            ["pkill", "-f", str(tmp_path / "fake-lumen-server")],
            check=False,
            capture_output=True,
        )


def test_scripted_override_banners_loudly_at_session_start(tmp_path: Path) -> None:
    """CORTEX_SCRIPTED_MODEL can never masquerade silently: the session-start
    notice banners it, and turn labels carry the (scripted) marker."""
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"responses": [[{"text": "canned"}]]}), encoding="utf-8")
    env = _lumen_stub_env(tmp_path)
    env["CORTEX_SCRIPTED_MODEL"] = str(script)
    process, send, recv_until = _worker_session(tmp_path, env)
    try:
        send(1, "app.handshake", {"protocol_version": "1.0.0"})
        recv_until(1)
        send(2, "session.create_or_resume", {})
        response, events = recv_until(2)
        session_id = response["result"]["session_id"]
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "SCRIPTED MODEL ACTIVE" in str(event["params"]["payload"].get("message", ""))
            for event in events
        )

        send(
            3,
            "session.submit_user_input",
            {
                "session_id": session_id,
                "user_input": "hello",
                "active_target": {"backend": "cloud", "provider": "azure", "model_id": "scripted"},
            },
        )
        submit_response, submit_events = recv_until(3)
        assert submit_response["result"]["assistant_text"] == "canned"
        finals = [
            event["params"]["payload"]
            for event in submit_events
            if event["params"]["event_type"] == "message.updated"
            and event["params"]["payload"].get("role") == "assistant"
            and event["params"]["payload"].get("final")
        ]
        assert finals and finals[-1]["model_label"] == "azure:scripted (scripted)"
        assert finals[-1]["provenance_verified"] is True
    finally:
        process.terminate()
        process.wait(timeout=5)
