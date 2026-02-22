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
            and "Download request accepted." in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Expected: username/model-name"
            in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_download_without_args_shows_usage_without_download_started_notice(
    tmp_path: Path,
) -> None:
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

        send(3, "command.execute", {"session_id": session_id, "command": "/download"})
        command_response, command_events = recv_until(3)

        assert command_response["result"]["ok"] is False
        assert "Usage: /download <repo_id>" in str(command_response["result"]["message"])
        assert not any(
            event["params"]["event_type"] == "system.notice"
            and "Download request accepted." in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_download_existing_model_returns_immediate_notice_without_background_start(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    existing = tmp_path / "models" / "hf-internal-testing--tiny-random-gpt2"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "weights.npz").write_bytes(b"ok")

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
            {"session_id": session_id, "command": "/download hf-internal-testing/tiny-random-gpt2"},
        )
        command_response, command_events = recv_until(3)
        assert command_response["result"]["ok"] is False
        assert "Model already exists: " in str(command_response["result"]["message"])
        assert "background" not in command_response["result"]
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Model already exists: " in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
        assert not any(
            event["params"]["event_type"] == "system.notice"
            and "Download request accepted." in str(event["params"]["payload"].get("message", ""))
            for event in command_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_download_cancel_returns_actionable_message_when_idle(tmp_path: Path) -> None:
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

        send(3, "command.execute", {"session_id": session_id, "command": "/model unknown:gpt"})
        model_response, model_events = recv_until(3)
        assert "result" in model_response
        assert model_response["result"]["ok"] is False
        assert "Unsupported cloud provider" in str(model_response["result"]["message"])
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Unsupported cloud provider" in str(event["params"]["payload"].get("message", ""))
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

        send(3, "command.execute", {"session_id": session_id, "command": "/setup"})
        setup_response, setup_events = recv_until(3)
        assert "result" in setup_response
        assert setup_response["result"]["ok"] is False
        assert "No local model installed." in str(setup_response["result"]["message"])
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "No local model installed." in str(event["params"]["payload"].get("message", ""))
            for event in setup_events
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


def test_worker_submit_user_input_without_model_returns_result_not_rpc_error(
    tmp_path: Path,
) -> None:
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


def test_worker_command_matrix_covers_all_core_command_paths(tmp_path: Path) -> None:
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

        command_cases = [
            ("/help", True, "Commands:"),
            ("/status", True, "System status"),
            ("/gpu", True, "GPU status"),
            ("/models", True, "Active model:"),
            ("/model", True, "Active model:"),
            ("/model list", True, "Active model:"),
            ("/model ls", True, "Active model:"),
            ("/model unknown:gpt", False, "Unsupported cloud provider"),
            ("/model :gpt", False, "Usage: /model <provider:model>"),
            ("/model openai:", False, "Usage: /model <provider:model>"),
            ("/login", False, "Usage: /login openai|anthropic"),
            ("/login unknown", False, "Unsupported provider"),
            ("/login openai", True, "Authentication status"),
            ("/login huggingface hf_secret", False, "Do not paste HuggingFace tokens"),
            ("/download", False, "Usage: /download <repo_id>"),
            ("/download badrepo", False, "Expected: username/model-name"),
            ("/download --force", False, "Unknown option: --force"),
            ("/download user/repo file.gguf extra", False, "Too many arguments"),
            ('/download "unterminated', False, "Invalid download arguments:"),
            ("/template", False, "No model loaded."),
            ("/template status", False, "No model loaded."),
            ("/template list", False, "No model loaded."),
            ("/template reset", False, "No model loaded."),
            ("/template nonsense", False, "No model loaded."),
            ("/finetune", True, "Usage: /finetune status"),
            ("/finetune train", False, "Interactive fine-tune flow"),
            ('/finetune "unterminated', False, "Invalid finetune arguments:"),
            ("/benchmark", False, "No model loaded."),
            ("/benchmark --tokens nope", False, "positive integer"),
            ("/benchmark --prompt", False, "Usage: /benchmark [tokens] [--prompt <text>]"),
            ("/benchmark --tokens 9000", False, "between 1 and 8192"),
            ('/benchmark --prompt ""', False, "Benchmark prompt cannot be empty."),
            ("/setup", False, "No local model installed."),
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

        send(next_id, "command.execute", {"session_id": session_id, "command": "/finetune status"})
        finetune_response, finetune_events = recv_until(next_id)
        assert "result" in finetune_response
        assert isinstance(finetune_response["result"].get("finetune"), dict)
        assert "mlx_available" in finetune_response["result"]["finetune"]
        assert any(
            event["params"]["event_type"] == "system.notice"
            and "Fine-tuning" in str(event["params"]["payload"].get("message", ""))
            for event in finetune_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
