import json
import os
import subprocess
import sys
import time
from pathlib import Path


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
        assert "Expected: username/model-name" in command_response["result"]["message"]
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Expected: username/model-name" in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_benchmark_command_reports_missing_model_with_notice(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
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
        assert "No model loaded." in command_response["result"]["message"]
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "No model loaded." in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_submit_user_input_without_model_returns_result_not_rpc_error(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
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
