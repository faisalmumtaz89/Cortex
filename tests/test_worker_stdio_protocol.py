import json
import os
import subprocess
import sys
import time
from pathlib import Path

from chat_completions_server import ChatCompletionsServer

from cortex.cloud.credentials import ENV_KEY_MAP

REPO_ROOT = Path(__file__).resolve().parents[1]


def _worker_env(tmp_path: Path) -> dict:
    """Worker env with an isolated HOME and no provider keys or scripted model,
    so startup restores nothing and every path exercises the REAL router."""
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env.pop("CORTEX_SCRIPTED_MODEL", None)
    for key in ENV_KEY_MAP.values():
        env.pop(key, None)
    env.pop("OPENAI_COMPATIBLE_BASE_URL", None)
    return env


def _worker_session(env: dict | None = None):
    """Spawn a worker and return (process, send, recv_until)."""
    process = subprocess.Popen(
        [sys.executable, "-P", "-m", "cortex", "--worker-stdio"],
        cwd=REPO_ROOT,
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


def _start_session(send, recv_until) -> tuple[str, list]:
    send(1, "app.handshake", {"protocol_version": "1.0.0"})
    recv_until(1)
    send(2, "session.create_or_resume", {})
    response, events = recv_until(2)
    return response["result"]["session_id"], events


def _final_assistant_frames(events: list) -> list[dict]:
    return [
        event["params"]["payload"]
        for event in events
        if event["params"]["event_type"] == "message.updated"
        and event["params"]["payload"].get("role") == "assistant"
        and event["params"]["payload"].get("final")
    ]


def test_worker_stdio_handshake_emits_jsonrpc_on_stdout_only() -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "app.handshake",
        "params": {"protocol_version": "1.0.0", "client_name": "pytest"},
    }

    process = subprocess.run(
        [sys.executable, "-P", "-m", "cortex", "--worker-stdio"],
        input=json.dumps(request) + "\n",
        text=True,
        cwd=REPO_ROOT,
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
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "app.handshake",
        "params": {"protocol_version": "9.9.9", "client_name": "pytest"},
    }

    process = subprocess.run(
        [sys.executable, "-P", "-m", "cortex", "--worker-stdio"],
        input=json.dumps(request) + "\n",
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0
    response = json.loads(process.stdout.strip().splitlines()[0])
    assert response["error"]["message"] == "Protocol version mismatch"


def test_worker_command_execute_emits_system_notice_event() -> None:
    process, send, recv_until = _worker_session()
    try:
        session_id, events = _start_session(send, recv_until)
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


def test_worker_session_start_reports_no_model_without_keys(tmp_path: Path) -> None:
    process, send, recv_until = _worker_session(_worker_env(tmp_path))
    try:
        _session_id, events = _start_session(send, recv_until)
        notices = [
            str(event["params"]["payload"].get("message", ""))
            for event in events
            if event["params"]["event_type"] == "system.notice"
        ]
        assert notices == [
            "Session ready · No model loaded. Pick one with /model, or add a key with /login."
        ]
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_invalid_provider_commands_return_result_not_rpc_error(tmp_path: Path) -> None:
    process, send, recv_until = _worker_session(_worker_env(tmp_path))
    try:
        session_id, _events = _start_session(send, recv_until)

        send(3, "command.execute", {"session_id": session_id, "command": "/model unknown:gpt"})
        model_response, model_events = recv_until(3)
        assert "result" in model_response
        assert model_response["result"]["ok"] is False
        expected = (
            "Unsupported provider 'unknown'. "
            "Use one of: anthropic, azure, openai, openai-compatible."
        )
        assert model_response["result"]["message"] == expected
        assert any(
            event["params"]["event_type"] == "system.notice"
            and event["params"]["payload"].get("message") == expected
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


def test_worker_submit_user_input_without_model_returns_result_not_rpc_error(
    tmp_path: Path,
) -> None:
    process, send, recv_until = _worker_session(_worker_env(tmp_path))
    try:
        session_id, _events = _start_session(send, recv_until)

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
    process, send, recv_until = _worker_session(_worker_env(tmp_path))
    try:
        session_id, _events = _start_session(send, recv_until)

        model_usage = "Usage: /model <provider>:<model> (example: /model openai:gpt-5.5)"
        command_cases = [
            ("/help", True, "Commands:"),
            ("/status", True, "System status"),
            ("/models", False, "Unknown command: /models"),
            ("/template status", False, "Unknown command: /template"),
            ("/gpu", False, "Unknown command: /gpu"),
            ("/download qwen3-5-9b", False, "Unknown command: /download"),
            ("/benchmark", False, "Unknown command: /benchmark"),
            ("/setup", False, "Unknown command: /setup"),
            ("/model", True, "Active model: No model loaded"),
            ("/model list", True, "Active model:"),
            ("/model ls", True, "Active model:"),
            ("/model unknown:gpt", False, "Unsupported provider 'unknown'"),
            ("/model openai:", False, model_usage),
            ("/model gpt-5.1", False, model_usage),
            ("/model openai:gpt-5.1", False, "openai is not authenticated"),
            ("/login", False, "Usage: /login openai|anthropic|azure|openai-compatible"),
            ("/login unknown", False, "Unsupported provider"),
            ("/login openai", True, "Authentication status"),
            ("/update everything", False, "Usage: /update [cortex]"),
            ("/clear", True, "cleared"),
            ("/save", True, "Saved conversation:"),
            ("status", False, "Not a slash command: status"),
            ("/unknown", False, "Unknown command: /unknown"),
            ("/quit", True, None),
            ("/exit", True, None),
        ]

        next_id = 3
        for command, expected_ok, expected_message in command_cases:
            send(next_id, "command.execute", {"session_id": session_id, "command": command})
            command_response, command_events = recv_until(next_id)

            assert "result" in command_response, command
            assert command_response["result"]["ok"] is expected_ok, command
            assert "background" not in command_response["result"], command
            assert any(
                event["params"]["event_type"] == "session.status"
                and event["params"]["payload"].get("status") == "busy"
                for event in command_events
            ), command
            assert any(
                event["params"]["event_type"] == "session.status"
                and event["params"]["payload"].get("status") == "idle"
                for event in command_events
            ), command

            if expected_message is not None:
                assert expected_message in str(command_response["result"].get("message", ""))
                assert any(
                    event["params"]["event_type"] == "system.notice"
                    and expected_message in str(event["params"]["payload"].get("message", ""))
                    for event in command_events
                ), command

            next_id += 1
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_worker_model_list_reports_active_target_and_providers(tmp_path: Path) -> None:
    env = _worker_env(tmp_path)
    env["OPENAI_COMPATIBLE_BASE_URL"] = "http://127.0.0.1:9/v1"
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    process, send, recv_until = _worker_session(env)
    try:
        session_id, _events = _start_session(send, recv_until)

        send(3, "model.list", {})
        empty, _ = recv_until(3)
        assert empty["result"]["active_target"] == {
            "provider": None,
            "model_id": None,
            "label": "No model loaded",
        }

        send(
            4,
            "command.execute",
            {"session_id": session_id, "command": "/model openai-compatible:test-model"},
        )
        selected, _ = recv_until(4)
        assert selected["result"]["ok"] is True
        assert selected["result"]["message"] == "openai-compatible:test-model — now active."

        send(5, "model.list", {})
        listing, _ = recv_until(5)
        result = listing["result"]
        assert set(result) == {"active_target", "cloud", "providers"}
        assert result["active_target"] == {
            "provider": "openai-compatible",
            "model_id": "test-model",
            "label": "openai-compatible:test-model",
        }
        providers = {row["provider"]: row["authenticated"] for row in result["providers"]}
        assert providers == {
            "openai": False,
            "anthropic": False,
            "azure": False,
            "openai-compatible": True,
        }
        active = [row["selector"] for row in result["cloud"] if row["active"]]
        assert active == ["openai-compatible:test-model"]
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_cloud_turn_provenance_verified_end_to_end(tmp_path: Path) -> None:
    """Real worker + real router + a local Chat Completions gateway: the turn
    succeeds and the final frame carries VERIFIED provenance."""
    server = ChatCompletionsServer([{"text": "hello from the gateway"}])
    env = _worker_env(tmp_path)
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    process, send, recv_until = _worker_session(env)
    try:
        session_id, _ = _start_session(send, recv_until)

        send(
            3,
            "command.execute",
            {"session_id": session_id, "command": f"/model openai-compatible:{server.model}"},
        )
        command_response, _ = recv_until(3)
        assert command_response["result"]["ok"] is True

        send(4, "session.submit_user_input", {"session_id": session_id, "user_input": "hello"})
        submit_response, submit_events = recv_until(4)
        assert submit_response["result"].get("ok", True) is not False, submit_response
        assert "hello from the gateway" in submit_response["result"]["assistant_text"]

        finals = _final_assistant_frames(submit_events)
        assert finals, "no final assistant frame"
        final = finals[-1]
        assert final["model_label"] == f"openai-compatible:{server.model}"
        assert final["provenance_verified"] is True

        # /status reports what actually served the last turn.
        send(5, "command.execute", {"session_id": session_id, "command": "/status"})
        status_response, _ = recv_until(5)
        assert (
            f"Last turn served by: openai-compatible:{server.model} (verified)"
            in str(status_response["result"]["message"])
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        server.close()


def test_gateway_without_model_identity_fails_provenance_and_rejects_turn(
    tmp_path: Path,
) -> None:
    """ADVERSARIAL: the gateway answers without saying which model responded.
    The turn must fail with a provenance error — it may never render as a
    normal answer."""
    server = ChatCompletionsServer([{"text": "unattributed answer"}], model="")
    env = _worker_env(tmp_path)
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    process, send, recv_until = _worker_session(env)
    try:
        session_id, _ = _start_session(send, recv_until)

        send(
            3,
            "command.execute",
            {"session_id": session_id, "command": "/model openai-compatible:test-model"},
        )
        command_response, _ = recv_until(3)
        assert command_response["result"]["ok"] is True

        send(4, "session.submit_user_input", {"session_id": session_id, "user_input": "hello"})
        submit_response, submit_events = recv_until(4)

        assert submit_response["result"]["ok"] is False
        error_text = str(submit_response["result"]["error"])
        assert "provenance mismatch" in error_text
        assert "openai-compatible:test-model" in error_text
        # The rejected turn is not presented as a verified answer.
        finals = _final_assistant_frames(submit_events)
        assert all(not payload.get("provenance_verified") for payload in finals)
        assert any(
            event["params"]["event_type"] == "session.error"
            and "provenance mismatch" in str(event["params"]["payload"].get("error", ""))
            for event in submit_events
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        server.close()


def test_scripted_override_banners_loudly_at_session_start(tmp_path: Path) -> None:
    """CORTEX_SCRIPTED_MODEL can never masquerade silently: the session-start
    notice banners it, and turn labels carry the (scripted) marker."""
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"responses": [[{"text": "canned"}]]}), encoding="utf-8")
    env = _worker_env(tmp_path)
    env["CORTEX_SCRIPTED_MODEL"] = str(script)
    process, send, recv_until = _worker_session(env)
    try:
        session_id, events = _start_session(send, recv_until)
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
                "active_target": {"provider": "azure", "model_id": "scripted"},
            },
        )
        submit_response, submit_events = recv_until(3)
        assert submit_response["result"]["assistant_text"] == "canned"
        finals = _final_assistant_frames(submit_events)
        assert finals and finals[-1]["model_label"] == "azure:scripted (scripted)"
        assert finals[-1]["provenance_verified"] is True
    finally:
        process.terminate()
        process.wait(timeout=5)
