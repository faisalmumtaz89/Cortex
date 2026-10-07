"""End-to-end validation of the agent runtime.

These tests are the empirical gate for behavioral change: they spawn the real
worker process (`python -m cortex --worker-stdio`) inside a scratch repository,
speak JSON-RPC over its stdio, and drive full agent turns. Only the model is
replaced — a deterministic script via CORTEX_SCRIPTED_MODEL — so the
orchestrator, tool registry, permission engine, event stream, and persistence
all run for real, and assertions check observable effects (files on disk,
event sequences, exit codes), not internals.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from chat_completions_server import ChatCompletionsServer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_script(path: Path, responses) -> Path:
    path.write_text(json.dumps({"responses": responses}), encoding="utf-8")
    return path


def _make_scratch_repo(root: Path) -> Path:
    repo = root / "scratch_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "README.md").write_text("# Scratch\n", encoding="utf-8")
    (repo / "src" / "app.py").write_text(
        "def main():\n    return 1\n",
        encoding="utf-8",
    )
    return repo


def _worker_env(tmp_path: Path, script: Path) -> dict:
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir(exist_ok=True)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["CORTEX_SCRIPTED_MODEL"] = str(script)
    env["OPENAI_API_KEY"] = "sk-test-scripted"
    return env


CLOUD_TARGET = {"backend": "cloud", "provider": "openai", "model_id": "scripted"}


class WorkerHarness:
    """Drives a real worker subprocess over JSON-RPC stdio."""

    def __init__(self, *, cwd: Path, env: dict):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "cortex", "--worker-stdio"],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.events: list[dict] = []
        self.responses: dict[int, dict] = {}
        self._next_id = 0

    def send(self, method: str, params: dict) -> int:
        self._next_id += 1
        request_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return request_id

    def _pump(self, deadline: float) -> bool:
        assert self.process.stdout is not None
        if time.time() >= deadline:
            return False
        line = self.process.stdout.readline()
        if not line:
            return False
        frame = json.loads(line)
        if frame.get("method") == "event":
            self.events.append(frame["params"])
        elif "id" in frame:
            self.responses[frame["id"]] = frame
        return True

    def wait_response(self, request_id: int, timeout: float = 60.0) -> dict:
        deadline = time.time() + timeout
        while request_id not in self.responses:
            if not self._pump(deadline):
                raise AssertionError(f"timed out waiting for response id={request_id}")
        return self.responses[request_id]

    def wait_event(self, event_type: str, *, after: int = 0, timeout: float = 60.0) -> dict:
        deadline = time.time() + timeout
        cursor = after
        while True:
            while cursor < len(self.events):
                event = self.events[cursor]
                cursor += 1
                if event["event_type"] == event_type:
                    return event
            if not self._pump(deadline):
                raise AssertionError(f"timed out waiting for event {event_type}")

    def events_of(self, event_type: str) -> list[dict]:
        return [event for event in self.events if event["event_type"] == event_type]

    def tool_parts(self) -> list[dict]:
        parts = []
        for event in self.events_of("message.part.updated"):
            part = event["payload"].get("part") or {}
            if part.get("type") == "tool":
                parts.append(part)
        return parts

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=5)


@pytest.fixture()
def scratch_repo(tmp_path: Path) -> Path:
    return _make_scratch_repo(tmp_path)


def _start_session(harness: WorkerHarness) -> str:
    harness.wait_response(harness.send("app.handshake", {"protocol_version": "1.0.0"}))
    response = harness.wait_response(harness.send("session.create_or_resume", {}))
    return response["result"]["session_id"]


def test_agent_reads_repository_and_answers(tmp_path: Path, scratch_repo: Path) -> None:
    script = _write_script(
        tmp_path / "script.json",
        [
            [
                {
                    "text": "Scanning the repository.",
                    "tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}],
                },
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}]},
                {"text": "app.py defines main() returning 1."},
            ]
        ],
    )
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)
        response = harness.wait_response(
            harness.send(
                "session.submit_user_input",
                {
                    "session_id": session_id,
                    "user_input": "What does app.py do?",
                    "active_target": CLOUD_TARGET,
                },
            )
        )

        assert "app.py defines main() returning 1." in response["result"]["assistant_text"]

        # Read-only tools ran silently: no permission prompt.
        assert harness.events_of("permission.asked") == []

        completed = [part for part in harness.tool_parts() if part.get("state") == "completed"]
        by_tool = {part["tool"]: part for part in completed}
        assert "list_dir" in by_tool and "read_file" in by_tool
        list_output = str(by_tool["list_dir"].get("output", ""))
        assert "README.md" in list_output and "src/" in list_output
        assert "def main" in str(by_tool["read_file"].get("output", ""))

        final = [
            event
            for event in harness.events_of("message.updated")
            if event["payload"].get("final") and event["payload"].get("role") == "assistant"
        ]
        assert final, "expected a final assistant message.updated event"
        statuses = [event["payload"].get("status") for event in harness.events_of("session.status")]
        assert statuses[-1] == "idle"
    finally:
        harness.close()


def test_agent_edit_requires_permission_and_changes_file(
    tmp_path: Path, scratch_repo: Path
) -> None:
    script = _write_script(
        tmp_path / "script.json",
        [
            [
                {
                    "text": "Updating the return value.",
                    "tool_calls": [
                        {
                            "name": "edit_file",
                            "arguments": {
                                "path": "src/app.py",
                                "old_text": "return 1",
                                "new_text": "return 2",
                            },
                        }
                    ],
                },
                {"text": "Edited app.py."},
            ],
            [
                {
                    "tool_calls": [
                        {"name": "bash", "arguments": {"command": "touch forbidden.txt"}}
                    ]
                },
                {"text": "Attempted the command."},
            ],
        ],
    )
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)

        # Turn 1: edit_file must prompt; approve it and verify the file changed.
        submit_id = harness.send(
            "session.submit_user_input",
            {
                "session_id": session_id,
                "user_input": "Change the return value to 2",
                "active_target": CLOUD_TARGET,
            },
        )
        asked = harness.wait_event("permission.asked")
        assert asked["payload"]["permission"] == "edit"
        harness.wait_response(
            harness.send(
                "permission.reply",
                {
                    "session_id": session_id,
                    "request_id": asked["payload"]["request_id"],
                    "reply": "allow_once",
                },
            )
        )
        response = harness.wait_response(submit_id)
        assert "Edited app.py." in response["result"]["assistant_text"]
        assert (scratch_repo / "src" / "app.py").read_text(encoding="utf-8") == (
            "def main():\n    return 2\n"
        )

        # Turn 2: reject bash; the command must not run.
        events_before = len(harness.events)
        submit_id = harness.send(
            "session.submit_user_input",
            {
                "session_id": session_id,
                "user_input": "Now touch forbidden.txt",
                "active_target": CLOUD_TARGET,
            },
        )
        asked = harness.wait_event("permission.asked", after=events_before)
        assert asked["payload"]["permission"] == "bash"
        harness.wait_response(
            harness.send(
                "permission.reply",
                {
                    "session_id": session_id,
                    "request_id": asked["payload"]["request_id"],
                    "reply": "reject",
                },
            )
        )
        harness.wait_response(submit_id)
        assert not (scratch_repo / "forbidden.txt").exists()
        errored = [part for part in harness.tool_parts() if part.get("tool") == "bash"]
        assert any("reject" in str(part.get("error", "")).lower() for part in errored)
    finally:
        harness.close()


def _run_headless(
    *, cwd: Path, env: dict, prompt: str, extra_args: list[str]
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cortex", "-p", prompt, "--model", "openai:scripted", *extra_args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_headless_full_auto_executes_bash(tmp_path: Path, scratch_repo: Path) -> None:
    script = _write_script(
        tmp_path / "script.json",
        [
            [
                {
                    "tool_calls": [
                        {"name": "bash", "arguments": {"command": "echo done > created.txt"}}
                    ]
                },
                {"text": "Created the file."},
            ]
        ],
    )
    process = _run_headless(
        cwd=scratch_repo,
        env=_worker_env(tmp_path, script),
        prompt="create the file",
        extra_args=["--full-auto"],
    )

    assert process.returncode == 0, process.stderr
    assert "Created the file." in process.stdout
    assert (scratch_repo / "created.txt").exists()


def test_bash_commands_inherit_no_worker_warning_filters(
    tmp_path: Path, scratch_repo: Path
) -> None:
    # Warning filters change how the user's own tooling behaves (pytest stops capturing
    # DeprecationWarning when sys.warnoptions is non-empty), so commands must see none from Cortex.
    command = f'"{sys.executable}" -c "import sys; print(sys.warnoptions)" > warnoptions.txt'
    script = _write_script(
        tmp_path / "script.json",
        [
            [
                {"tool_calls": [{"name": "bash", "arguments": {"command": command}}]},
                {"text": "Recorded the warning options."},
            ]
        ],
    )
    env = _worker_env(tmp_path, script)
    env.pop("PYTHONWARNINGS", None)
    env.pop("PYTHONDEVMODE", None)
    process = _run_headless(
        cwd=scratch_repo, env=env, prompt="record the warning options", extra_args=["--full-auto"]
    )

    assert process.returncode == 0, process.stderr
    assert (scratch_repo / "warnoptions.txt").read_text(encoding="utf-8").strip() == "[]"


def test_openai_compatible_provider_runs_tool_turn(tmp_path: Path, scratch_repo: Path) -> None:
    server = ChatCompletionsServer(
        [
            {"tool_calls": [{"name": "bash", "arguments": {"command": "echo ok > made.txt"}}]},
            {"text": "Created made.txt."},
        ]
    )
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    try:
        process = subprocess.run(
            [
                sys.executable, "-m", "cortex", "-p", "create made.txt",
                "--model", f"openai-compatible:{server.model}", "--full-auto",
            ],
            cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
        )
    finally:
        server.close()

    assert process.returncode == 0, process.stderr
    assert "Created made.txt." in process.stdout
    assert (scratch_repo / "made.txt").read_text(encoding="utf-8").strip() == "ok"
    assert server.authorization == ["Bearer sk-compatible-test"] * 2
    first, second = server.requests
    assert first["model"] == server.model and first["stream"] is True
    assert "bash" in {tool["function"]["name"] for tool in first["tools"]}
    tool_messages = [message for message in second["messages"] if message["role"] == "tool"]
    assert len(tool_messages) == 1


def _run_against_gateway(tmp_path: Path, repo: Path, script, *, full_auto: bool = False):
    server = ChatCompletionsServer(script)
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    command = [sys.executable, "-P", "-m", "cortex", "-p", "fix the bug",
               "--model", "openai-compatible:test-model"]
    if full_auto:
        command.append("--full-auto")
    try:
        process = subprocess.run(
            command, cwd=repo, env=env, capture_output=True, text=True, timeout=120
        )
    finally:
        server.close()
    return process, server


@pytest.mark.parametrize("text", ["", "The fix is to"])
def test_reply_cut_off_at_the_output_limit_fails_the_turn(
    tmp_path: Path, scratch_repo: Path, text: str
) -> None:
    process, server = _run_against_gateway(
        tmp_path, scratch_repo, [{"text": text, "finish": "length"}]
    )

    assert process.returncode == 1, process.stdout
    assert text in process.stdout
    assert "reached the output limit (max_tokens=32768)" in process.stderr
    assert server.requests[0]["max_tokens"] == 32768


def test_reply_cut_off_at_the_output_limit_stays_in_the_conversation(
    tmp_path: Path, scratch_repo: Path
) -> None:
    server = ChatCompletionsServer(
        [{"text": "First, open the parser", "finish": "length"}, {"text": "and fix it."}]
    )
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    target = {"backend": "cloud", "provider": "openai-compatible", "model_id": "test-model"}
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        session_id = _start_session(harness)
        results = [
            harness.wait_response(
                harness.send(
                    "session.submit_user_input",
                    {"session_id": session_id, "user_input": text, "active_target": target},
                )
            )["result"]
            for text in ("how do I fix the bug?", "continue")
        ]
    finally:
        harness.close()
        server.close()

    assert results[0]["ok"] is False
    assert "reached the output limit" in results[0]["error"]
    assert results[1].get("ok", True) is not False
    history = [
        (message["role"], message["content"])
        for message in server.requests[1]["messages"]
        if message["role"] != "system"
    ]
    assert history == [
        ("user", "how do I fix the bug?"),
        ("assistant", "First, open the parser"),
        ("user", "continue"),
    ]


def test_tool_call_cut_off_at_the_output_limit_is_not_executed(
    tmp_path: Path, scratch_repo: Path
) -> None:
    process, server = _run_against_gateway(
        tmp_path,
        scratch_repo,
        [{"tool_calls": [{"name": "bash", "arguments": {"command": "touch made.txt"}}],
          "finish": "length"}],
        full_auto=True,
    )

    assert process.returncode == 1
    assert "reached the output limit" in process.stderr
    assert not (scratch_repo / "made.txt").exists()
    assert len(server.requests) == 1


def test_repository_config_cannot_redirect_provider_endpoint(
    tmp_path: Path, scratch_repo: Path
) -> None:
    trusted = ChatCompletionsServer([{"text": "from the configured endpoint"}])
    planted = ChatCompletionsServer([{"text": "from the repository's endpoint"}])
    (scratch_repo / "config.yaml").write_text(
        f"cloud_openai_compatible_base_url: {planted.base_url}\n", encoding="utf-8"
    )
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env.pop("OPENAI_COMPATIBLE_BASE_URL", None)
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    user_config = Path(env["HOME"]) / ".cortex" / "config.yaml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        f"cloud_openai_compatible_base_url: {trusted.base_url}\n", encoding="utf-8"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-m", "cortex", "-p", "hi", "--model", "openai-compatible:test-model"],
            cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
        )
    finally:
        trusted.close()
        planted.close()

    assert process.returncode == 0, process.stderr
    assert "from the configured endpoint" in process.stdout
    assert planted.requests == []


def test_model_list_reports_provider_auth_and_active_uncatalogued_model(
    tmp_path: Path, scratch_repo: Path
) -> None:
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env.pop("ANTHROPIC_API_KEY", None)
    env["OPENAI_COMPATIBLE_BASE_URL"] = "http://127.0.0.1:9/v1"
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        _start_session(harness)
        selected = harness.wait_response(
            harness.send(
                "model.select",
                {"backend": "cloud", "provider": "openai-compatible", "model_id": "org/model:free"},
            )
        )
        assert selected["result"]["ok"], selected
        listing = harness.wait_response(harness.send("model.list", {}))["result"]
    finally:
        harness.close()

    providers = {row["provider"]: row for row in listing["providers"]}
    assert providers["openai-compatible"]["authenticated"] is True
    assert providers["anthropic"]["authenticated"] is False
    active = [row for row in listing["cloud"] if row["active"]]
    assert [row["selector"] for row in active] == ["openai-compatible:org/model:free"]


def test_worker_is_cloud_only_and_clears_saved_local_state(
    tmp_path: Path, scratch_repo: Path
) -> None:
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    for key in ("CORTEX_SCRIPTED_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(key, None)
    state = Path(env["HOME"]) / ".cortex" / "state.yaml"
    state.parent.mkdir(parents=True)
    state.write_text(
        "last_used_backend: local\nlast_used_model: qwen3-5-9b:q4_0\n", encoding="utf-8"
    )
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        session_id = _start_session(harness)
        listing = harness.wait_response(harness.send("model.list", {}))["result"]
        commands = {
            command: harness.wait_response(
                harness.send("command.execute", {"session_id": session_id, "command": command})
            )["result"]
            for command in ("/gpu", "/download qwen3-5-9b", "/benchmark", "/setup", "/model qwen")
        }
    finally:
        harness.close()

    assert "local" not in listing
    assert listing["active_target"]["label"] == "No model loaded"
    for command in ("/gpu", "/download qwen3-5-9b", "/benchmark", "/setup"):
        assert commands[command]["ok"] is False
        assert "Unknown command" in commands[command]["message"], command
    assert commands["/model qwen"]["ok"] is False
    assert "/model <provider>:<model>" in commands["/model qwen"]["message"]
    saved = state.read_text(encoding="utf-8")
    assert "local" not in saved and "qwen" not in saved


def test_headless_default_denies_writes(tmp_path: Path, scratch_repo: Path) -> None:
    script = _write_script(
        tmp_path / "script.json",
        [
            [
                {
                    "tool_calls": [
                        {"name": "bash", "arguments": {"command": "echo done > created.txt"}}
                    ]
                },
                {"text": "Tried to create the file."},
            ]
        ],
    )
    process = _run_headless(
        cwd=scratch_repo,
        env=_worker_env(tmp_path, script),
        prompt="create the file",
        extra_args=[],
    )

    assert process.returncode == 0, process.stderr
    assert "Tried to create the file." in process.stdout
    assert not (scratch_repo / "created.txt").exists()
