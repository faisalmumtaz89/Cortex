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
import re
import shutil
import socket
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


def _turn_answering_prompts(harness: WorkerHarness, session_id: str, text: str, reply: str):
    """Run one turn, answering every permission prompt with `reply`; return
    the prompts that were asked."""
    submit_id = harness.send(
        "session.submit_user_input",
        {"session_id": session_id, "user_input": text, "active_target": CLOUD_TARGET},
    )
    asked: list[dict] = []
    cursor = len(harness.events)
    deadline = time.time() + 60
    while submit_id not in harness.responses:
        assert harness._pump(deadline), "turn did not finish"
        for event in harness.events[cursor:]:
            if event["event_type"] == "permission.asked":
                asked.append(event["payload"])
                harness.send(
                    "permission.reply",
                    {"session_id": session_id,
                     "request_id": event["payload"]["request_id"], "reply": reply},
                )
        cursor = len(harness.events)
    return asked


def _bash_turn(command: str) -> list:
    return [{"tool_calls": [{"name": "bash", "arguments": {"command": command}}]},
            {"text": "Ran it."}]


@pytest.mark.parametrize("ripgrep", [True, False], ids=["ripgrep", "python"])
def test_search_names_files_relative_to_the_project(
    tmp_path: Path, scratch_repo: Path, ripgrep: bool
) -> None:
    """With ripgrep or without it, search names a file as the other tools do:
    relative to the project, never by its absolute path."""
    if ripgrep and shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "search", "arguments": {"query": "def main"}}]},
          {"text": "Found it."}]],
    )
    env = _worker_env(tmp_path, script)
    if not ripgrep:
        # A PATH with python and git but no rg.
        bare = tmp_path / "bin"
        bare.mkdir()
        for tool in ("git", "sh"):
            (bare / tool).symlink_to(shutil.which(tool))
        env["PATH"] = str(bare)
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        session_id = _start_session(harness)
        _turn_answering_prompts(harness, session_id, "find main", "reject")
        [part] = [part for part in harness.tool_parts() if part.get("state") == "completed"]
    finally:
        harness.close()
    assert part["output"] == "src/app.py:1: def main():", part["output"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": str(repo.parent / "home" / ".gitconfig")},
    ).stdout


def _hostile_git_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A repository whose own .git/config names a program for every setting
    git runs one for while reading, each leaving a marker file if it runs.
    The user's global config defines a filter of theirs, which must still
    apply. Returns the repository and the marker folder."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".gitconfig").write_text(
        "[user]\n\tname = Dev\n\temail = dev@example.com\n"
        "[filter \"upper\"]\n\tclean = tr a-z A-Z\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    marks = tmp_path / "marks"
    marks.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "notes.txt").write_text("one\n", encoding="utf-8")
    (repo / "loud.txt").write_text("quiet\n", encoding="utf-8")
    (repo / ".gitattributes").write_text(
        "notes.txt filter=evil diff=evil\nloud.txt filter=upper\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "first")
    _git(repo, "switch", "-qc", "feature")
    (repo / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "second")
    (repo / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    hook = repo / ".git" / "hooks" / "post-index-change"
    hook.write_text(f"#!/bin/sh\ntouch {marks}/hook\n", encoding="utf-8")
    hook.chmod(0o755)
    for key, program in {
        "core.fsmonitor": f"touch {marks}/fsmonitor; false",
        "filter.evil.clean": f"touch {marks}/clean; cat",
        "filter.evil.smudge": f"touch {marks}/smudge; cat",
        "filter.evil.process": f"touch {marks}/process",
        "diff.evil.textconv": f"touch {marks}/textconv; cat",
        "diff.evil.command": f"touch {marks}/diffcommand",
        "diff.external": f"touch {marks}/external",
        "core.pager": f"touch {marks}/pager; cat",
        "log.showSignature": "true",
        "gpg.program": f"touch {marks}/gpg",
    }.items():
        _git(repo, "config", key, program)
    return repo, marks


def test_the_git_tool_reads_a_repository_without_a_prompt_and_runs_nothing_it_names(
    tmp_path: Path,
) -> None:
    """status, diff, log, show and merge-base answer without a permission
    prompt, and nothing the repository's own config names runs."""
    repo, marks = _hostile_git_repo(tmp_path)
    calls = [
        {"command": "status"},
        {"command": "diff"},
        {"command": "diff", "revision": "main...feature", "stat": True},
        {"command": "log", "patch": True},
        {"command": "show", "revision": "main:notes.txt"},
        {"command": "merge-base", "revision": "main"},
    ]
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "git", "arguments": args}]}, {"text": "Read it."}]
         for args in calls],
    )
    env = _worker_env(tmp_path, script)
    env["GIT_EXTERNAL_DIFF"] = f"touch {marks}/env-external"  # the user's own GIT_* stay out
    harness = WorkerHarness(cwd=repo, env=env)
    try:
        session_id = _start_session(harness)
        for index in range(len(calls)):
            assert _turn_answering_prompts(harness, session_id, f"read {index}", "reject") == []
        parts = [part for part in harness.tool_parts() if part.get("state") != "running"]
    finally:
        harness.close()
    assert sorted(path.name for path in marks.iterdir()) == []
    assert [part["ok"] for part in parts] == [True] * len(calls), parts
    status, diff, branch_stat, log, show, base = (part["output"] for part in parts)
    assert "## feature" in status and " M notes.txt" in status, status
    assert "+three" in diff, diff
    # The user's own filter still applies: loud.txt is unchanged, not "modified".
    assert "loud.txt" not in status, status
    assert "notes.txt | 1 +" in branch_stat, branch_stat
    assert "Dev: second" in log and "+two" in log, log
    assert show == "one", show
    assert base.strip() == _git(repo, "rev-parse", "main").strip()


def test_the_git_tool_refuses_arguments_that_are_options_or_outside_the_project(
    tmp_path: Path,
) -> None:
    repo, marks = _hostile_git_repo(tmp_path)
    outside = tmp_path / "written.txt"
    results = _tool_results(tmp_path, repo, [
        ("git", {"command": "diff", "revision": f"--output={outside}"}),
        ("git", {"command": "log", "paths": ["../home/.gitconfig"]}),
        ("git", {"command": "status", "staged": True}),
    ])
    errors = [part["error"] for part in results]
    assert "not a revision" in errors[0], errors
    assert "outside the working directory" in errors[1], errors
    assert "staged applies to diff only" in errors[2], errors
    assert not outside.exists()
    assert sorted(path.name for path in marks.iterdir()) == []


def test_the_git_tool_takes_patch_and_stat_wherever_they_show_changes(tmp_path: Path) -> None:
    """patch asks for the changes line by line: diff and show give them anyway,
    and with stat both come; only status and merge-base have none to give."""
    repo, marks = _hostile_git_repo(tmp_path)
    results = _tool_results(tmp_path, repo, [
        ("git", {"command": "show", "patch": True}),
        ("git", {"command": "diff", "stat": True, "patch": True}),
        ("git", {"command": "status", "patch": True}),
    ])
    show, diff, status = results
    assert show["ok"] and "+two" in show["output"], show
    assert diff["ok"] and "notes.txt | 1 +" in diff["output"] and "+three" in diff["output"], diff
    assert "patch does not apply to status" in status["error"], status
    assert sorted(path.name for path in marks.iterdir()) == []


def test_headless_runs_the_git_tool_without_full_auto(tmp_path: Path) -> None:
    repo, _marks = _hostile_git_repo(tmp_path)
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "git", "arguments": {"command": "status"}}]},
          {"text": "One file changed."}]],
    )
    process = _run_headless(
        cwd=repo, env=_worker_env(tmp_path, script), prompt="what changed", extra_args=[]
    )
    assert process.returncode == 0, process.stderr
    assert "[tool] git command=status" in process.stderr, process.stderr
    assert "[tool] git ok" in process.stderr, process.stderr
    assert "refused" not in process.stderr, process.stderr


def test_a_file_tool_call_outside_the_project_is_refused_without_a_prompt(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """File tools work only inside the project: a write elsewhere is refused
    at once, not offered as a prompt that cannot say what it touches."""
    outside = tmp_path / "elsewhere" / "sweep.py"
    outside.parent.mkdir()
    calls = [
        ("write_file", {"path": str(outside), "content": "print(1)\n"}),
        ("edit_file", {"path": "../elsewhere/sweep.py", "old_text": "a", "new_text": "b"}),
    ]
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": name, "arguments": args}]}, {"text": "Done."}]
         for name, args in calls],
    )
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)
        for index in range(len(calls)):
            assert _turn_answering_prompts(harness, session_id, f"run {index}", "allow_once") == []
        errors = [part["error"] for part in harness.tool_parts() if part.get("state") == "error"]
    finally:
        harness.close()
    assert len(errors) == 2, errors
    assert all("outside the working directory" in error for error in errors), errors
    assert not outside.exists()


def test_allow_always_covers_only_that_command_in_that_repository(
    tmp_path: Path, scratch_repo: Path
) -> None:
    script = _write_script(
        tmp_path / "script.json",
        [_bash_turn("touch one.txt"), _bash_turn("touch one.txt"), _bash_turn("touch two.txt")],
    )
    env = _worker_env(tmp_path, script)
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        session_id = _start_session(harness)
        assert len(_turn_answering_prompts(harness, session_id, "one", "allow_always")) == 1
        assert _turn_answering_prompts(harness, session_id, "one again", "reject") == []
        # A different command still asks, and rejecting it keeps it from running.
        assert len(_turn_answering_prompts(harness, session_id, "two", "reject")) == 1
        assert not (scratch_repo / "two.txt").exists()
    finally:
        harness.close()

    # The same command in another repository asks again (same HOME, same store).
    other_repo = _make_scratch_repo(tmp_path / "other")
    _write_script(tmp_path / "script.json", [_bash_turn("touch one.txt")])
    harness = WorkerHarness(cwd=other_repo, env=env)
    try:
        session_id = _start_session(harness)
        assert len(_turn_answering_prompts(harness, session_id, "one", "reject")) == 1
        assert not (other_repo / "one.txt").exists()
    finally:
        harness.close()


def test_allow_always_for_a_command_in_one_directory_asks_in_another(
    tmp_path: Path, scratch_repo: Path
) -> None:
    in_src = [{"tool_calls": [{"name": "bash", "arguments": {
        "command": "touch made.txt", "workdir": "src"}}]}, {"text": "Ran it."}]
    script = _write_script(
        tmp_path / "script.json", [in_src, _bash_turn("touch made.txt")]
    )
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)
        (asked,) = _turn_answering_prompts(harness, session_id, "in src", "allow_always")
        assert asked["patterns"] == ["cd ./src || exit 1\ntouch made.txt"]
        assert len(_turn_answering_prompts(harness, session_id, "at root", "reject")) == 1
        assert (scratch_repo / "src" / "made.txt").exists()
        assert not (scratch_repo / "made.txt").exists()
    finally:
        harness.close()


@pytest.mark.parametrize(
    "workdir, pattern",
    [(5, "*"), (["src"], "*"), ("src\x00", "*"), ("/elsewhere", "*"), ("~", "*"),
     ("-P", "cd ./-P || exit 1\ntouch made.txt")],
)
def test_workdir_pattern_cannot_run_anywhere_else(
    tmp_path: Path, scratch_repo: Path, workdir, pattern
) -> None:
    """A bad or outside directory gives no rule to save; an odd name is written
    so that `cd` cannot read it as an option. Neither breaks the turn."""
    call = [{"tool_calls": [{"name": "bash", "arguments": {
        "command": "touch made.txt", "workdir": workdir}}]}, {"text": "Done."}]
    script = _write_script(tmp_path / "script.json", [call])
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)
        asked = _turn_answering_prompts(harness, session_id, "run", "allow_always")
        assert harness.events_of("session.error") == []
        assert [prompt["patterns"] for prompt in asked] == [[pattern]]
        saved = (tmp_path / "home" / ".cortex" / "tool_permissions.yaml").exists()
        assert saved == (pattern != "*")
    finally:
        harness.close()


def test_allow_once_covers_only_that_call(tmp_path: Path, scratch_repo: Path) -> None:
    script = _write_script(
        tmp_path / "script.json", [_bash_turn("touch one.txt"), _bash_turn("touch one.txt")]
    )
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)
        assert len(_turn_answering_prompts(harness, session_id, "one", "allow_once")) == 1
        assert len(_turn_answering_prompts(harness, session_id, "again", "reject")) == 1
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


def test_headless_log_shows_commands_as_typed_and_edits_as_diffs(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """A CI log shows what a run changed: a command as it was typed, and an
    edit as a unified diff."""
    script = _write_script(
        tmp_path / "script.json",
        [
            [
                {"tool_calls": [
                    {"name": "bash", "arguments": {"command": "echo one\necho 'two' > two.txt"}},
                    {"name": "edit_file", "arguments": {
                        "path": "src/app.py", "old_text": "return 1", "new_text": "return 2"}},
                ]},
                {"text": "Changed it."},
            ]
        ],
    )
    process = _run_headless(
        cwd=scratch_repo, env=_worker_env(tmp_path, script), prompt="change it",
        extra_args=["--full-auto"],
    )

    assert process.returncode == 0, process.stderr
    assert "[tool] bash $ echo one\n    echo 'two' > two.txt\n" in process.stderr, process.stderr
    assert "[tool] edit_file path=src/app.py old_text=return 1 new_text=return 2" in process.stderr
    assert (
        "--- src/app.py\n+++ src/app.py\n@@ -1,2 +1,2 @@\n def main():\n-    return 1\n"
        "+    return 2\n"
    ) in process.stderr, process.stderr
    assert process.stdout.strip() == "Changed it."


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


def _tool_results(
    tmp_path: Path, repo: Path, calls: list, *, timeout: float = 60, linger: float = 0
) -> list:
    """Run each (tool, arguments) call in its own turn, approving it; return
    each call's finished tool part. Every turn must end in time, and the worker
    must still answer afterwards."""
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": name, "arguments": args}]}, {"text": "Done."}]
         for name, args in calls],
    )
    harness = WorkerHarness(cwd=repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)
        for index in range(len(calls)):
            started = time.time()
            _turn_answering_prompts(harness, session_id, f"run {index}", "allow_once")
            assert time.time() - started < timeout, f"call {index} took too long"
        harness.wait_response(harness.send("app.handshake", {"protocol_version": "1.0.0"}))
        time.sleep(linger)  # the worker keeps running, as it does between turns
        return [part for part in harness.tool_parts()
                if part.get("state") in ("completed", "error")]
    finally:
        harness.close()


def _bash_results(tmp_path: Path, repo: Path, calls: list, **kwargs) -> list:
    return _tool_results(tmp_path, repo, [("bash", args) for args in calls], **kwargs)


def _edit(path: str, old: str, new: str) -> tuple:
    return ("edit_file", {"path": path, "old_text": old, "new_text": new})


def test_edits_keep_each_files_line_endings_and_byte_order_mark(
    tmp_path: Path, scratch_repo: Path
) -> None:
    (scratch_repo / "crlf.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    (scratch_repo / "mixed.txt").write_bytes(b"one\r\ntwo\nthree\r\n")
    (scratch_repo / "bom.txt").write_bytes(b"\xef\xbb\xbfone\ntwo\n")
    (scratch_repo / "rewrite.txt").write_bytes(b"old\r\nfile\r\n")
    results = _tool_results(tmp_path, scratch_repo, [
        _edit("crlf.txt", "one\ntwo", "ONE\nTWO"),
        _edit("mixed.txt", "three", "THREE"),
        _edit("bom.txt", "one", "ONE"),
        ("write_file", {"path": "rewrite.txt", "content": "new\ncontent\n"}),
    ])

    assert all(part.get("state") == "completed" for part in results), results
    assert (scratch_repo / "crlf.txt").read_bytes() == b"ONE\r\nTWO\r\nthree\r\n"
    assert (scratch_repo / "mixed.txt").read_bytes() == b"one\r\ntwo\nTHREE\r\n"
    assert (scratch_repo / "bom.txt").read_bytes() == b"\xef\xbb\xbfONE\ntwo\n"
    assert (scratch_repo / "rewrite.txt").read_bytes() == b"new\r\ncontent\r\n"


def test_edits_in_mixed_files_count_every_match_and_keep_other_lines(
    tmp_path: Path, scratch_repo: Path
) -> None:
    (scratch_repo / "all.txt").write_bytes(b"x = 1\r\ny\nx = 1\nz\r\n")
    (scratch_repo / "one.txt").write_bytes(b"x = 1\r\ny\nx = 1\nz\r\n")
    (scratch_repo / "span.txt").write_bytes(b"one\r\ntwo\nthree\r\n")
    (scratch_repo / "mac.txt").write_bytes(b"one\rtwo\rthree\r")
    replace_all, ambiguous, span, mac, endings_only = _tool_results(tmp_path, scratch_repo, [
        ("edit_file", {"path": "all.txt", "old_text": "x = 1\n", "new_text": "x = 2\n",
                       "replace_all": True}),
        _edit("one.txt", "x = 1\n", "x = 2\n"),
        _edit("span.txt", "one\ntwo\nthree", "ONE\nTWO"),
        _edit("mac.txt", "one\ntwo", "ONE\nTWO"),
        ("edit_file", {"path": "all.txt", "old_text": "\r\n", "new_text": "\n",
                       "replace_all": True}),
    ])

    assert (scratch_repo / "all.txt").read_bytes() == b"x = 2\r\ny\nx = 2\nz\r\n"
    assert "matches 2 locations" in str(ambiguous.get("error"))
    assert (scratch_repo / "one.txt").read_bytes() == b"x = 1\r\ny\nx = 1\nz\r\n"
    assert (scratch_repo / "span.txt").read_bytes() == b"ONE\r\nTWO\r\n"
    assert (scratch_repo / "mac.txt").read_bytes() == b"ONE\rTWO\rthree\r"
    assert "differ only in line breaks" in str(endings_only.get("error"))
    assert replace_all.get("state") == span.get("state") == mac.get("state") == "completed"


def test_edits_keep_the_owner_and_hard_links_and_refuse_read_only_files(
    tmp_path: Path, scratch_repo: Path
) -> None:
    owned = scratch_repo / "owned.txt"
    owned.write_text("one\n")
    if os.geteuid() == 0:
        os.chown(owned, 1000, 1000)
    linked = scratch_repo / "linked.txt"
    linked.write_text("one\n")
    os.link(linked, tmp_path / "other_link.txt")
    locked = scratch_repo / "locked.txt"
    locked.write_text("one\n")
    locked.chmod(0o444)
    *_, refused = _tool_results(tmp_path, scratch_repo, [
        _edit("owned.txt", "one", "two"),
        _edit("linked.txt", "one", "two"),
        _edit("locked.txt", "one", "two"),
    ])

    if os.geteuid() == 0:
        assert (owned.stat().st_uid, owned.stat().st_gid) == (1000, 1000)
    assert (tmp_path / "other_link.txt").read_text() == "two\n"
    if os.geteuid() != 0:  # root may write any file
        assert "read-only" in str(refused.get("error"))
        assert locked.read_text() == "one\n"


def test_a_recursive_listing_does_not_go_into_git_internals(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """`.git` holds hooks and objects the model never needs: the listing names
    it but does not walk into it."""
    subprocess.run(["git", "init", "-q"], cwd=scratch_repo, check=True)
    [result] = _tool_results(tmp_path, scratch_repo, [("list_dir", {"path": ".", "recursive": True})])
    output = str(result.get("output", ""))
    assert ".git/" in output, output
    assert "src/app.py" in output, output
    assert ".git/hooks" not in output and ".git/HEAD" not in output, output


def test_read_file_reads_only_up_to_the_range(tmp_path: Path, scratch_repo: Path) -> None:
    (scratch_repo / "log.txt").write_bytes(b"l1\nl2\nl3\n\xffbroken later\n")
    (result,) = _tool_results(
        tmp_path, scratch_repo, [("read_file", {"path": "log.txt", "end_line": 2})]
    )

    assert str(result.get("output")).splitlines()[:2] == ["1: l1", "2: l2"]


def test_edits_keep_the_file_mode_and_leave_no_temporary_files(
    tmp_path: Path, scratch_repo: Path
) -> None:
    script = scratch_repo / "run.sh"
    script.write_text("#!/bin/sh\necho one\n")
    script.chmod(0o755)
    _tool_results(tmp_path, scratch_repo, [_edit("run.sh", "echo one", "echo two")])

    assert script.read_text() == "#!/bin/sh\necho two\n"
    assert script.stat().st_mode & 0o777 == 0o755
    assert sorted(p.name for p in scratch_repo.iterdir()) == ["README.md", "run.sh", "src"]


def test_read_file_numbers_lines_and_says_how_to_read_on(
    tmp_path: Path, scratch_repo: Path
) -> None:
    (scratch_repo / "big.txt").write_text("".join(f"line {i}\n" for i in range(1, 3001)))
    (scratch_repo / "wide.txt").write_text("x" * 5000 + "\nend\n")
    first, wide = _tool_results(tmp_path, scratch_repo, [
        ("read_file", {"path": "big.txt"}),
        ("read_file", {"path": "wide.txt"}),
    ])

    lines = str(first.get("output")).splitlines()
    assert lines[0] == "1: line 1" and lines[1999] == "2000: line 2000"
    assert lines[-1] == (
        "(Showing lines 1-2000; the file has more lines. Use start_line=2001 to continue.)"
    )
    wide_lines = str(wide.get("output")).splitlines()
    assert wide_lines[0].endswith("... (line cut at 2000 characters)")
    assert len(wide_lines[0]) < 2100
    assert wide_lines[-1] == "2: end"


def test_bash_commands_cannot_read_the_workers_stdin(tmp_path: Path, scratch_repo: Path) -> None:
    (result,) = _bash_results(tmp_path, scratch_repo, [{"command": "cat; echo after-cat"}])

    assert "after-cat" in str(result.get("output"))


def test_bash_commands_do_not_see_provider_keys_or_cortex_internals(
    tmp_path: Path, scratch_repo: Path
) -> None:
    (result,) = _bash_results(tmp_path, scratch_repo, [{"command": "env; echo HOME_IS=$HOME"}])

    output = str(result.get("output"))
    assert "HOME_IS=" in output
    assert "OPENAI_API_KEY" not in output and "sk-test-scripted" not in output
    assert "CORTEX_" not in output


def test_bash_timeout_ends_commands_that_ignore_sigterm(tmp_path: Path, scratch_repo: Path) -> None:
    (result,) = _bash_results(
        tmp_path, scratch_repo,
        [{"command": "trap '' TERM; echo waiting; sleep 60", "timeout_ms": 1000}],
        timeout=20,
    )

    output = str(result.get("output"))
    assert "waiting" in output and "timed out" in output


def test_bash_returns_when_the_command_ends_even_if_a_background_child_runs_on(
    tmp_path: Path, scratch_repo: Path
) -> None:
    (result,) = _bash_results(
        tmp_path, scratch_repo, [{"command": "sleep 60 & echo started"}], timeout=20
    )

    assert "started" in str(result.get("output"))


def test_bash_background_child_keeps_running_after_the_call_returns(
    tmp_path: Path, scratch_repo: Path
) -> None:
    writer = "(for i in 1 2 3 4 5 6; do echo tick $i; echo alive $i >> bg.log; sleep 0.3; done) &"
    (result,) = _bash_results(
        tmp_path, scratch_repo, [{"command": f"{writer} echo started"}], linger=3
    )

    assert "started" in str(result.get("output"))
    assert (scratch_repo / "bg.log").read_text().split()[-1] == "6"


def test_bash_output_keeps_order_and_the_end_of_long_output(
    tmp_path: Path, scratch_repo: Path
) -> None:
    long_output = (
        f'"{sys.executable}" -c "import sys; print(\'x\' * 400000); '
        f'print(\'THE END\', file=sys.stderr)"'
    )
    ordered, long = _bash_results(
        tmp_path, scratch_repo,
        [{"command": "echo one; echo two >&2; echo three"}, {"command": long_output}],
    )

    assert str(ordered.get("output")).split() == ["one", "two", "three"]
    assert str(long.get("output")).rstrip().endswith("THE END")
    assert "bytes of output not shown" in str(long.get("output"))


def test_interrupt_stops_a_running_bash_command(tmp_path: Path, scratch_repo: Path) -> None:
    command = "echo started; sleep 4; touch finished.txt"
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": command}}]},
          {"text": "NOT_REACHED"}]],
    )
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)
        submit_id = harness.send(
            "session.submit_user_input",
            {"session_id": session_id, "user_input": "run", "active_target": CLOUD_TARGET},
        )
        asked = harness.wait_event("permission.asked")
        harness.send("permission.reply", {"session_id": session_id,
                     "request_id": asked["payload"]["request_id"], "reply": "allow_once"})
        time.sleep(1.5)
        started = time.time()
        harness.wait_response(harness.send("session.interrupt", {"session_id": session_id}))
        result = harness.wait_response(submit_id, timeout=15)["result"]

        assert time.time() - started < 8
        assert result.get("interrupted") is True
        assert "NOT_REACHED" not in result["assistant_text"]
        (part,) = [p for p in harness.tool_parts()
                   if p.get("tool") == "bash" and p.get("state") in ("completed", "error")]
        assert "started" in str(part.get("output")) and "interrupted" in str(part.get("output"))
        assert part.get("error") == "interrupted"
        time.sleep(4)  # past the point where the command would have finished
        assert not (scratch_repo / "finished.txt").exists()
    finally:
        harness.close()


def _bash_turn_with_prompt(tmp_path: Path, repo: Path, env_overrides: dict):
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": "touch approved.txt"}}]},
          {"text": "Done."}]],
    )
    env = _worker_env(tmp_path, script)
    env.update(env_overrides)
    harness = WorkerHarness(cwd=repo, env=env)
    session_id = _start_session(harness)
    submit_id = harness.send(
        "session.submit_user_input",
        {"session_id": session_id, "user_input": "run", "active_target": CLOUD_TARGET},
    )
    asked = harness.wait_event("permission.asked")
    return harness, session_id, submit_id, asked["payload"]["request_id"]


def test_a_permission_prompt_waits_for_its_answer(tmp_path: Path, scratch_repo: Path) -> None:
    harness, session_id, submit_id, request_id = _bash_turn_with_prompt(
        tmp_path, scratch_repo,
        {"CORTEX_CLOUD_TIMEOUT_SECONDS": "1", "CORTEX_TOOLS_IDLE_TIMEOUT_SECONDS": "1"},
    )
    try:
        time.sleep(3)  # longer than any configured timeout
        harness.send("permission.reply", {"session_id": session_id,
                     "request_id": request_id, "reply": "allow_once"})
        harness.wait_response(submit_id)
        assert (scratch_repo / "approved.txt").exists()
    finally:
        harness.close()


def test_interrupting_a_turn_answers_its_permission_prompt(
    tmp_path: Path, scratch_repo: Path
) -> None:
    harness, session_id, submit_id, _ = _bash_turn_with_prompt(tmp_path, scratch_repo, {})
    try:
        harness.wait_response(harness.send("session.interrupt", {"session_id": session_id}))
        result = harness.wait_response(submit_id, timeout=10)["result"]
        assert result.get("interrupted") is True
        assert not (scratch_repo / "approved.txt").exists()
    finally:
        harness.close()


def test_a_bad_setting_keeps_the_others_and_is_reported(tmp_path: Path, scratch_repo: Path) -> None:
    script = _write_script(tmp_path / "script.json", [[{"text": "Hi."}]])
    env = _worker_env(tmp_path, script)
    config = tmp_path / "home" / ".cortex" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("tools_max_iterations: 1001\ntemperature: 0.3\ntools_profile: read_only\n")
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        _start_session(harness)  # every stdout line must still be a JSON-RPC frame
        notice = harness.wait_event("system.notice")["payload"]["message"]
        assert "tools_max_iterations" in notice
    finally:
        harness.close()


def test_headless_rejects_a_prompt_it_cannot_answer_and_keeps_stdout_clean(
    tmp_path: Path, scratch_repo: Path
) -> None:
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": "touch asked.txt"}}]},
          {"text": "FINAL_REPLY"}]],
    )
    env = _worker_env(tmp_path, script)
    cortex_home = tmp_path / "home" / ".cortex"
    cortex_home.mkdir(parents=True, exist_ok=True)
    (cortex_home / "tool_permissions.yaml").write_text(
        "version: 2\nrules:\n- {permission: bash, pattern: '*', action: ask}\n"
    )
    (cortex_home / "config.yaml").write_text("tools_max_iterations: 1001\n")

    started = time.time()
    process = _run_headless(cwd=scratch_repo, env=env, prompt="run it", extra_args=["--full-auto"])

    assert time.time() - started < 60
    assert not (scratch_repo / "asked.txt").exists()
    assert process.stdout.strip() == "FINAL_REPLY"
    assert "Ignored tools_max_iterations: 1001" in process.stderr
    assert "Ignored tools_max_iterations: 1001" in (cortex_home / "cortex.log").read_text()


def _write_skill(root: Path, name: str, description: str, body: str, allowed=None) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    header = [f"name: {name}", f"description: {description}"]
    if allowed is not None:
        header.append(f"allowed-tools: [{', '.join(allowed)}]")
    (folder / "SKILL.md").write_text("---\n" + "\n".join(header) + "\n---\n" + body)
    return folder


def test_skills_are_listed_and_loaded_on_demand(tmp_path: Path, scratch_repo: Path) -> None:
    project = scratch_repo / ".cortex" / "skills"
    folder = _write_skill(project, "demo-report", "Use when the user asks for a demo report.",
                          "SKILL_BODY_MARKER: list the files, then summarize.\n")
    (folder / "examples").mkdir()
    (folder / "examples" / "one.md").write_text("EXAMPLE_MARKER\n")
    # A user skill of the same name is overridden by the project's.
    _write_skill(tmp_path / "home" / ".cortex" / "skills", "demo-report", "USER_COPY", "USER\n")
    _write_skill(tmp_path / "home" / ".cortex" / "skills", "user-only", "Use for user things.",
                 "USER_ONLY\n")
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "skill", "arguments": {"name": "demo-report"}}]},
            {"tool_calls": [{"name": "skill", "arguments": {
                "name": "demo-report", "file": "examples/one.md"}}]},
            {"tool_calls": [{"name": "skill", "arguments": {
                "name": "demo-report", "file": "../../../README.md"}}]},
            {"text": "Done."},
        ],
        ["give me a demo report"],
    )

    system = requests[0]["messages"][0]["content"]
    assert "demo-report: Use when the user asks for a demo report." in system
    assert "user-only: Use for user things." in system
    assert "USER_COPY" not in system and "SKILL_BODY_MARKER" not in system
    results = [m["content"] for m in requests[3]["messages"] if m["role"] == "tool"]
    assert "SKILL_BODY_MARKER" in results[0] and "examples/one.md" in results[0]
    assert "EXAMPLE_MARKER" in results[1]
    assert "outside the skill" in results[2]


def test_a_loaded_skill_limits_the_tools_of_the_rest_of_the_turn(
    tmp_path: Path, scratch_repo: Path
) -> None:
    _write_skill(scratch_repo / ".cortex" / "skills", "look-only", "Use to look around.",
                 "Only look.\n", allowed=["read_file", "list_dir", "search"])
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "skill", "arguments": {"name": "look-only"}}]},
            {"tool_calls": [{"name": "write_file", "arguments": {
                "path": "made.txt", "content": "x"}}]},
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "README.md"}}]},
            {"text": "Done."},
            {"tool_calls": [{"name": "write_file", "arguments": {
                "path": "later.txt", "content": "x"}}]},
            {"text": "Done again."},
        ],
        ["look around", "now write"],
        env_overrides={"CORTEX_TOOLS_PROFILE": "full"},
        approve=True,
    )

    results = [m["content"] for m in requests[3]["messages"] if m["role"] == "tool"]
    assert "look-only skill" in results[1] and "write_file" in results[1]
    assert "Scratch" in results[2]
    assert not (scratch_repo / "made.txt").exists()
    assert (scratch_repo / "later.txt").exists()  # the limit lasts one turn


def test_a_second_skill_narrows_the_limit_but_never_lifts_it(
    tmp_path: Path, scratch_repo: Path
) -> None:
    skills = scratch_repo / ".cortex" / "skills"
    _write_skill(skills, "look-only", "Use to look around.", "Only look.\n",
                 allowed=["read_file", "list_dir"])
    _write_skill(skills, "anything", "Use for anything.", "Do anything.\n")
    _write_skill(skills, "list-only", "Use to list.", "Only list.\n", allowed=["list_dir", "bash"])
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "skill", "arguments": {"name": "anything"}}]},
            {"tool_calls": [{"name": "write_file", "arguments": {
                "path": "free.txt", "content": "x"}}]},
            {"tool_calls": [{"name": "skill", "arguments": {"name": "look-only"}}]},
            {"tool_calls": [{"name": "skill", "arguments": {"name": "anything"}}]},
            {"tool_calls": [{"name": "write_file", "arguments": {
                "path": "made.txt", "content": "x"}}]},
            {"tool_calls": [{"name": "skill", "arguments": {"name": "list-only"}}]},
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "README.md"}}]},
            {"tool_calls": [{"name": "bash", "arguments": {"command": "touch ran.txt"}}]},
            {"text": "Done."},
        ],
        ["go"],
        env_overrides={"CORTEX_TOOLS_PROFILE": "full"},
        approve=True,
    )

    results = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"]
    assert (scratch_repo / "free.txt").exists()  # no limit before a limiting skill
    assert not (scratch_repo / "made.txt").exists()
    assert "look-only skill" in results[4]
    # Both limits apply: only list_dir is in both.
    assert "look-only" in results[6] and "list-only" in results[6]
    assert "read_file" in results[6] and "only these tools: list_dir;" in results[6]
    assert not (scratch_repo / "ran.txt").exists()


def test_shipped_commit_and_pr_writes_a_requested_description_but_never_edits(
    tmp_path: Path, scratch_repo: Path
) -> None:
    before = (scratch_repo / "README.md").read_text()
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "write_file", "arguments": {
                "path": "PR_BODY.md", "content": "## Summary\nAdds things.\n"}}]},
            {"tool_calls": [{"name": "edit_file", "arguments": {
                "path": "README.md", "old_text": "Scratch", "new_text": "Edited"}}]},
            {"text": "Done."},
        ],
        ["/commit-and-pr write the pull request description into PR_BODY.md"],
        env_overrides={"CORTEX_TOOLS_PROFILE": "full"},
        approve=True,
    )

    results = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"]
    assert (scratch_repo / "PR_BODY.md").read_text() == "## Summary\nAdds things.\n"
    assert "commit-and-pr skill does not use edit_file" in results[1]
    assert (scratch_repo / "README.md").read_text() == before


def test_shipped_map_and_plan_is_listed_and_keeps_the_turn_read_only(
    tmp_path: Path, scratch_repo: Path
) -> None:
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "write_file", "arguments": {
                "path": "made.txt", "content": "x"}}]},
            {"tool_calls": [{"name": "bash", "arguments": {"command": "touch ran.txt"}}]},
            {"text": "Plan."},
        ],
        ["/map-and-plan how would I add a --verbose flag?"],
        env_overrides={"CORTEX_TOOLS_PROFILE": "full"},
        approve=True,
    )

    system = requests[0]["messages"][0]["content"]
    for name in ("map-and-plan", "reproduce-and-fix", "safe-refactor", "review-changes",
                 "commit-and-pr"):
        assert f"- {name}: " in system
    asked = requests[0]["messages"][-1]["content"]
    assert "# Map and plan" in asked and "examples/plan.md" in asked
    results = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"]
    assert all("map-and-plan skill does not use" in result for result in results)
    assert not (scratch_repo / "made.txt").exists() and not (scratch_repo / "ran.txt").exists()


def test_skills_are_not_listed_when_tools_are_off(tmp_path: Path, scratch_repo: Path) -> None:
    _write_skill(scratch_repo / ".cortex" / "skills", "look-only", "Use to look around.", "x\n")
    env = _worker_env(tmp_path, tmp_path / "x.json")
    env["CORTEX_TOOLS_PROFILE"] = "off"
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        _start_session(harness)
        listed = harness.wait_response(harness.send("skill.list", {}))["result"]["skills"]
    finally:
        harness.close()
    assert listed == []


def test_a_slash_command_runs_a_skill_and_the_worker_lists_skills(
    tmp_path: Path, scratch_repo: Path
) -> None:
    _write_skill(scratch_repo / ".cortex" / "skills", "look-only", "Use to look around.",
                 "LOOK_ONLY_BODY: only look.\n", allowed=["read_file"])
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "write_file", "arguments": {
                "path": "made.txt", "content": "x"}}]},
            {"text": "Done."},
            {"text": "Plain answer."},
        ],
        ["/look-only check the README", "/look-onlyish is not a skill"],
        env_overrides={"CORTEX_TOOLS_PROFILE": "full"},
        approve=True,
    )

    asked = requests[0]["messages"][-1]["content"]
    assert asked.startswith("/look-only check the README")
    assert "LOOK_ONLY_BODY" in asked
    results = [m["content"] for m in requests[1]["messages"] if m["role"] == "tool"]
    assert "look-only skill" in results[0]
    assert not (scratch_repo / "made.txt").exists()
    assert requests[2]["messages"][-1]["content"] == "/look-onlyish is not a skill"

    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, tmp_path / "x.json"))
    try:
        _start_session(harness)
        listed = harness.wait_response(harness.send("skill.list", {}))["result"]["skills"]
    finally:
        harness.close()
    assert {"name": "look-only", "description": "Use to look around."} in listed


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
    assert providers["openai-compatible"]["url_saved"] is True
    assert providers["anthropic"]["authenticated"] is False
    assert providers["azure"]["url_saved"] is False
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


def test_interrupt_stops_a_model_request_in_flight(tmp_path: Path, scratch_repo: Path) -> None:
    server = ChatCompletionsServer([{"text": "late answer", "delay": 30}, {"text": "next answer"}])
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    target = {"backend": "cloud", "provider": "openai-compatible", "model_id": "test-model"}
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        session_id = _start_session(harness)
        submitted = time.time()
        turn = harness.send(
            "session.submit_user_input",
            {"session_id": session_id, "user_input": "think hard", "active_target": target},
        )
        harness.wait_event("session.status")
        time.sleep(1)
        harness.wait_response(harness.send("session.interrupt", {"session_id": session_id}))
        result = harness.wait_response(turn, timeout=20)["result"]
        elapsed = time.time() - submitted
        follow_up = harness.wait_response(
            harness.send(
                "session.submit_user_input",
                {"session_id": session_id, "user_input": "next", "active_target": target},
            )
        )["result"]
    finally:
        harness.close()
        server.close()

    assert result["interrupted"] is True
    assert result["assistant_text"] == ""
    assert elapsed < 10, elapsed
    assert len(server.requests) == 2  # no retry or fallback request after the interrupt
    assert follow_up["assistant_text"] == "next answer"


def _converse_with_gateway(
    tmp_path: Path, repo: Path, script, prompts, *, env_overrides=None, results=None,
    approve=False,
):
    """Run one worker session through `prompts` against a scripted gateway and
    return the request bodies the gateway received. Each turn must succeed,
    unless `results` is a list: then the turn results are collected in it.
    With `approve`, permission prompts are answered "allow_once"."""
    server = ChatCompletionsServer(script)
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    env.update(env_overrides or {})
    target = {"backend": "cloud", "provider": "openai-compatible", "model_id": "test-model"}
    harness = WorkerHarness(cwd=repo, env=env)
    try:
        session_id = _start_session(harness)
        for prompt in prompts:
            submit_id = harness.send(
                "session.submit_user_input",
                {"session_id": session_id, "user_input": prompt, "active_target": target},
            )
            cursor = len(harness.events)
            deadline = time.time() + 60
            while approve and submit_id not in harness.responses:
                assert harness._pump(deadline), "turn did not finish"
                for event in harness.events[cursor:]:
                    if event["event_type"] == "permission.asked":
                        harness.send("permission.reply", {
                            "session_id": session_id,
                            "request_id": event["payload"]["request_id"],
                            "reply": "allow_once"})
                cursor = len(harness.events)
            response = harness.wait_response(submit_id)
            if results is None:
                assert response["result"].get("ok") is not False, response
            else:
                results.append(response["result"])
    finally:
        harness.close()
        server.close()
    return server.requests


def test_later_turns_see_earlier_tool_calls_and_results(
    tmp_path: Path, scratch_repo: Path
) -> None:
    lines = [f"setting_{index} = {index}" for index in range(300)]
    (scratch_repo / "src" / "settings.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "src/settings.py"}}]},
            {"text": "It defines 300 settings."},
            {"text": "Second answer."},
            {"text": "Third answer."},
            {"text": "Fourth answer."},
        ],
        ["read the settings", "second", "third", "fourth"],
    )

    def replayed(request):
        calls = [
            call
            for message in request["messages"]
            if message["role"] == "assistant"
            for call in message.get("tool_calls") or []
        ]
        results = [message for message in request["messages"] if message["role"] == "tool"]
        return calls, results

    for request in requests[2:4]:
        calls, results = replayed(request)
        assert [call["function"]["name"] for call in calls] == ["read_file"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"path": "src/settings.py"}
        assert [result["tool_call_id"] for result in results] == [calls[0]["id"]]
        assert "setting_0 = 0" in results[0]["content"]
        assert "setting_299 = 299" in results[0]["content"]

    # Four turns on, the old result keeps only its beginning and end.
    calls, results = replayed(requests[4])
    assert [call["function"]["name"] for call in calls] == ["read_file"]
    assert "setting_0 = 0" in results[0]["content"]
    assert "setting_150 = 150" not in results[0]["content"]
    assert "setting_299 = 299" in results[0]["content"]
    assert "characters omitted" in results[0]["content"]
    roles = [message["role"] for message in requests[4]["messages"]]
    assert roles == [
        "system", "user", "assistant", "tool", "assistant",
        "user", "assistant", "user", "assistant", "user",
    ]


def _replayed_tool_history(request) -> list:
    """(tool name, arguments, result) for each call replayed in a request."""
    results = {
        message["tool_call_id"]: message["content"]
        for message in request["messages"]
        if message["role"] == "tool"
    }
    return [
        (call["function"]["name"], json.loads(call["function"]["arguments"]), results[call["id"]])
        for message in request["messages"]
        if message["role"] == "assistant"
        for call in message.get("tool_calls") or []
    ]


def test_tool_calls_without_ids_keep_their_own_results(
    tmp_path: Path, scratch_repo: Path
) -> None:
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "README.md"}}],
             "omit_ids": True},
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}],
             "omit_ids": True},
            {"text": "Read both."},
            {"text": "Done."},
        ],
        ["read both files", "next"],
    )

    replayed = _replayed_tool_history(requests[3])
    assert [(name, args) for name, args, _ in replayed] == [
        ("read_file", {"path": "README.md"}),
        ("read_file", {"path": "src/app.py"}),
    ]
    assert "# Scratch" in replayed[0][2]
    assert "def main" in replayed[1][2]


def test_interrupted_turn_keeps_the_tool_calls_that_ran(
    tmp_path: Path, scratch_repo: Path
) -> None:
    server = ChatCompletionsServer(
        [
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}]},
            {"text": "late", "delay": 30},
            {"text": "next answer"},
        ]
    )
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    target = {"backend": "cloud", "provider": "openai-compatible", "model_id": "test-model"}
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        session_id = _start_session(harness)
        turn = harness.send(
            "session.submit_user_input",
            {"session_id": session_id, "user_input": "read app.py", "active_target": target},
        )
        deadline = time.time() + 20
        while len(server.requests) < 2 and time.time() < deadline:
            time.sleep(0.1)
        harness.wait_response(harness.send("session.interrupt", {"session_id": session_id}))
        assert harness.wait_response(turn, timeout=20)["result"]["interrupted"] is True
        harness.wait_response(
            harness.send(
                "session.submit_user_input",
                {"session_id": session_id, "user_input": "next", "active_target": target},
            )
        )
    finally:
        harness.close()
        server.close()

    replayed = _replayed_tool_history(server.requests[2])
    assert [(name, args) for name, args, _ in replayed] == [("read_file", {"path": "src/app.py"})]
    assert "def main" in replayed[0][2]


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_tool_steps_send_back_the_reasoning_behind_their_calls(
    tmp_path: Path, scratch_repo: Path, field: str
) -> None:
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"reasoning": "The app lives in src/app.py, read it first.", "reasoning_field": field,
             "tool_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}]},
            {"reasoning": "main returns 1.", "reasoning_field": field,
             "tool_calls": [{"name": "list_dir", "arguments": {"path": "src"}}]},
            {"reasoning": "Both read; answer.", "reasoning_field": field,
             "text": "main returns 1."},
            {"text": "Next."},
        ],
        ["what does main return", "next"],
    )

    steps = [message for message in requests[2]["messages"] if message.get("tool_calls")]
    assert [step.get(field) for step in steps] == [
        "The app lives in src/app.py, read it first.",
        "main returns 1.",
    ]
    # The next turn replays every step's reasoning, the final answer's included.
    replies = [m for m in requests[3]["messages"] if m["role"] == "assistant"]
    assert [reply.get(field) for reply in replies] == [
        "The app lives in src/app.py, read it first.",
        "main returns 1.",
        "Both read; answer.",
    ]
    assert replies[-1]["content"] == "main returns 1."
    assert all(len(reply) == 3 + bool(reply.get("tool_calls")) for reply in replies)


def test_reasoning_sent_in_two_fields_goes_back_once(tmp_path: Path, scratch_repo: Path) -> None:
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"reasoning": "Read it.", "reasoning_field": ["reasoning_content", "reasoning"],
             "tool_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}]},
            {"text": "Done."},
        ],
        ["read app.py"],
    )

    (step,) = [message for message in requests[1]["messages"] if message.get("tool_calls")]
    assert step["reasoning_content"] == "Read it."
    assert "reasoning" not in step


def test_turn_stops_at_the_tool_step_limit_and_keeps_its_work(
    tmp_path: Path, scratch_repo: Path
) -> None:
    results: list = []
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"text": "Reading.",
             "tool_calls": [{"name": "read_file", "arguments": {"path": "README.md"}}]},
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}]},
            {"text": "Continuing from both files."},
        ],
        ["read everything", "continue"],
        env_overrides={"CORTEX_TOOLS_MAX_ITERATIONS": "2"},
        results=results,
    )

    first, second = results
    assert first["ok"] is False
    assert "reached its limit of model requests (tools_max_iterations=2)" in first["error"]
    assert first["assistant_text"] == "Reading."
    assert second["assistant_text"] == "Continuing from both files."
    assert len(requests) == 3
    replayed = _replayed_tool_history(requests[2])
    assert [args for _, args, _ in replayed] == [{"path": "README.md"}, {"path": "src/app.py"}]
    assert "def main" in replayed[1][2]


def test_connection_lost_mid_turn_is_retried_without_rerunning_tools(
    tmp_path: Path, scratch_repo: Path
) -> None:
    results: list = []
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [
            {"text": "Reading.",
             "tool_calls": [{"name": "read_file", "arguments": {"path": "src/app.py"}}]},
            {"reset": True},
            {"cut": True},
            {"text": "main returns 1."},
        ],
        ["what does main return"],
        env_overrides={"CORTEX_CLOUD_MAX_RETRIES": "2"},
        results=results,
    )

    assert results[0].get("ok") is not False, results
    assert results[0]["assistant_text"] == "Reading.main returns 1."
    assert len(requests) == 4
    assert requests[1]["messages"] == requests[2]["messages"] == requests[3]["messages"]


@pytest.mark.parametrize("status", [400, 401, 404])
def test_rejected_request_is_sent_once(tmp_path: Path, scratch_repo: Path, status: int) -> None:
    results: list = []
    requests = _converse_with_gateway(
        tmp_path, scratch_repo, [{"status": status}], ["hello"], results=results
    )

    assert results[0]["ok"] is False
    assert f"scripted {status}" in results[0]["error"]
    assert len(requests) == 1


def test_rate_limited_request_waits_and_is_sent_again(tmp_path: Path, scratch_repo: Path) -> None:
    results: list = []
    started = time.time()
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [{"status": 429, "retry_after": 2}, {"status": 503}, {"text": "Done."}],
        ["hello"],
        results=results,
    )

    assert results[0]["assistant_text"] == "Done."
    assert len(requests) == 3
    assert time.time() - started >= 3  # Retry-After 2s, then 2s of backoff


def test_retries_stop_at_cloud_max_retries(tmp_path: Path, scratch_repo: Path) -> None:
    results: list = []
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [{"status": 500, "retry_after": 0}] * 3,
        ["hello"],
        env_overrides={"CORTEX_CLOUD_MAX_RETRIES": "1"},
        results=results,
    )

    assert results[0]["ok"] is False
    assert len(requests) == 2


def test_reply_cut_off_after_its_text_fails_the_turn(tmp_path: Path, scratch_repo: Path) -> None:
    results: list = []
    requests = _converse_with_gateway(
        tmp_path, scratch_repo, [{"text": "The answer is", "cut": True}], ["hello"],
        results=results,
    )

    assert results[0]["ok"] is False
    assert "ended before the reply was complete" in results[0]["error"]
    assert results[0]["assistant_text"] == "The answer is"
    assert len(requests) == 1


def test_long_conversation_keeps_its_first_turn(tmp_path: Path, scratch_repo: Path) -> None:
    turns = 20
    requests = _converse_with_gateway(
        tmp_path,
        scratch_repo,
        [{"text": f"answer {index}"} for index in range(turns)],
        [f"question {index}" for index in range(turns)],
    )

    contents = [message["content"] for message in requests[-1]["messages"]]
    assert contents[1:3] == ["question 0", "answer 0"]
    assert contents[-1] == f"question {turns - 1}"


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
    # The run says how to allow what it refused, just before naming the model.
    assert "--full-auto" in process.stderr.splitlines()[-2], process.stderr


def test_headless_runs_a_command_a_saved_rule_allows(tmp_path: Path, scratch_repo: Path) -> None:
    """The documented way to allow some commands without --full-auto."""
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": "echo done > made.txt"}}]},
          {"text": "Made it."}]],
    )
    env = _worker_env(tmp_path, script)
    rules = tmp_path / "home" / ".cortex" / "tool_permissions.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        "version: 2\nrules:\n  - permission: bash\n    pattern: echo done > made.txt\n"
        f"    action: allow\n    repo: {scratch_repo}\n",
        encoding="utf-8",
    )
    process = _run_headless(cwd=scratch_repo, env=env, prompt="make it", extra_args=[])

    assert process.returncode == 0, process.stderr
    assert (scratch_repo / "made.txt").read_text(encoding="utf-8").strip() == "done"
    assert "refused" not in process.stderr, process.stderr


def test_headless_refusal_names_the_commands_a_saved_rule_allows(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """A model that tries another command first learns which ones it may run,
    instead of concluding that the shell is off."""
    rules = tmp_path / "home" / ".cortex" / "tool_permissions.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        "version: 2\nrules:\n  - permission: bash\n    pattern: python3 -m pytest -q\n"
        f"    action: allow\n    repo: {scratch_repo}\n"
        "  - permission: bash\n    pattern: ls [*].py\n    action: allow\n"
        "    repo: /somewhere/else\n",
        encoding="utf-8",
    )
    process, server = _run_against_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "bash", "arguments": {"command": "ls -la"}}]},
            {"text": "Done."},
        ],
    )

    assert process.returncode == 0, process.stderr
    tool_messages = [m for m in server.requests[1]["messages"] if m["role"] == "tool"]
    refusal = str(tool_messages[0]["content"])
    assert "`python3 -m pytest -q`" in refusal, refusal
    assert "*.py" not in refusal, refusal  # another repository's rule


def test_headless_refusal_tells_the_model_and_the_user_about_full_auto(
    tmp_path: Path, scratch_repo: Path
) -> None:
    process, server = _run_against_gateway(
        tmp_path,
        scratch_repo,
        [
            {"tool_calls": [{"name": "bash", "arguments": {"command": "git diff"}}]},
            {"text": "Could not run it."},
        ],
    )

    assert process.returncode == 0, process.stderr
    tool_messages = [m for m in server.requests[1]["messages"] if m["role"] == "tool"]
    assert "--full-auto" in str(tool_messages[0]["content"]), tool_messages
    assert "patterns=" not in process.stderr, process.stderr
    assert "[tool] bash refused" in process.stderr, process.stderr


def test_headless_shows_why_the_provider_refused_and_keeps_the_saved_model(
    tmp_path: Path, scratch_repo: Path
) -> None:
    server = ChatCompletionsServer(
        [{"status": 403, "reason": "model not allowed: 'typo-model'"}, {"text": "fine"}]
    )
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env.pop("OPENAI_API_KEY")  # else a run without --model would pick an OpenAI default
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-P", "-m", "cortex", "-p", "hi", *args],
            cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
        )

    try:
        refused = run("--model", "openai-compatible:typo-model")
        later = run()
    finally:
        server.close()

    assert refused.returncode == 1, refused.stderr
    assert "model not allowed: 'typo-model'" in refused.stderr, refused.stderr
    assert "Check the model name" in refused.stderr, refused.stderr
    assert "(key from OPENAI_COMPATIBLE_API_KEY)" in refused.stderr, refused.stderr
    # A model the provider refused is not remembered for the next run.
    assert later.returncode == 2, later.stderr
    assert "no model selected" in later.stderr, later.stderr
    assert len(server.requests) == 1


def test_headless_names_every_setting_a_provider_still_needs(
    tmp_path: Path, scratch_repo: Path
) -> None:
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    for name in ("OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"):
        env.pop(name, None)
    process = subprocess.run(
        [sys.executable, "-P", "-m", "cortex", "-p", "hi", "--model", "openai-compatible:m"],
        cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
    )

    assert process.returncode == 2, process.stderr
    assert "OPENAI_COMPATIBLE_API_KEY" in process.stderr, process.stderr
    assert "OPENAI_COMPATIBLE_BASE_URL" in process.stderr, process.stderr


def test_headless_asks_only_for_the_key_once_the_base_url_is_saved(
    tmp_path: Path, scratch_repo: Path
) -> None:
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    for name in ("OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"):
        env.pop(name, None)
    state = tmp_path / "home" / ".cortex" / "state.yaml"
    state.parent.mkdir(parents=True)
    state.write_text("openai_compatible_base_url: http://127.0.0.1:9/v1\n", encoding="utf-8")
    process = subprocess.run(
        [sys.executable, "-P", "-m", "cortex", "-p", "hi", "--model", "openai-compatible:m"],
        cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
    )

    assert process.returncode == 2, process.stderr
    assert "OPENAI_COMPATIBLE_API_KEY" in process.stderr, process.stderr
    assert "OPENAI_COMPATIBLE_BASE_URL" not in process.stderr, process.stderr
    assert "<base_url>" not in process.stderr, process.stderr


def _gateway_run(tmp_path: Path, repo: Path, args: list, **popen) -> tuple:
    server = ChatCompletionsServer([{"text": "Summary."}])
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    command = [sys.executable, "-P", "-m", "cortex", *args,
               "--model", "openai-compatible:test-model"]
    try:
        process = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True,
                                 timeout=120, **popen)
    finally:
        server.close()
    user = [m for m in server.requests[0]["messages"] if m["role"] == "user"] if server.requests else []
    return process, str(user[-1]["content"]) if user else ""


def test_headless_reports_a_command_that_exits_non_zero_as_such(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """Failing tests are the command's result, not a Cortex failure."""
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": "exit 3"}}]},
          {"text": "The tests failed."}]],
    )
    process = _run_headless(cwd=scratch_repo, env=_worker_env(tmp_path, script),
                            prompt="run the tests", extra_args=["--full-auto"])
    assert process.returncode == 0, process.stderr
    assert "[tool] bash exited with code 3" in process.stderr, process.stderr
    assert "failed: exit_code" not in process.stderr, process.stderr


def test_headless_tells_the_model_it_is_a_one_shot_run_and_what_it_may_do(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """The model knows its reply is the output and, without --full-auto, that
    edits and commands are refused, before it tries one."""
    process, server = _run_against_gateway(tmp_path, scratch_repo, [{"text": "Answer."}])
    assert process.returncode == 0, process.stderr
    system = str(server.requests[0]["messages"][0]["content"])
    assert "non-interactive" in system, system
    assert "--full-auto" in system, system

    process, server = _run_against_gateway(
        tmp_path, scratch_repo, [{"text": "Answer."}], full_auto=True
    )
    system = str(server.requests[0]["messages"][0]["content"])
    assert "non-interactive" in system, system
    assert "--full-auto" not in system, system


def test_headless_names_the_model_that_answered(tmp_path: Path, scratch_repo: Path) -> None:
    """A CI log says which model the reply came from, as the app's /status does."""
    script = _write_script(tmp_path / "script.json", [[{"text": "Hello."}]])
    process = _run_headless(cwd=scratch_repo, env=_worker_env(tmp_path, script),
                            prompt="hi", extra_args=[])
    assert process.returncode == 0, process.stderr
    assert process.stdout == "Hello.\n"
    assert "answered by openai:scripted" in process.stderr.splitlines()[-1], process.stderr


def test_headless_setup_errors_share_one_prefix(tmp_path: Path, scratch_repo: Path) -> None:
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    process = subprocess.run(
        [sys.executable, "-P", "-m", "cortex", "-p", "hi", "--model", "nosuchprovider:m"],
        cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 2, process.stderr
    assert process.stderr.startswith("error: "), process.stderr


def test_headless_reads_what_is_piped_to_it(tmp_path: Path, scratch_repo: Path) -> None:
    """`git diff | cortex -p "summarize this"` sends the diff with the prompt."""
    piped = "diff --git a/src/app.py b/src/app.py\n-    return 1\n+    return 2\n"
    process, sent = _gateway_run(tmp_path, scratch_repo, ["-p", "summarize this diff"],
                                 input=piped)
    assert process.returncode == 0, process.stderr
    assert sent.startswith("summarize this diff"), sent
    assert "+    return 2" in sent, sent


def test_headless_prompt_dash_reads_the_prompt_from_stdin(tmp_path: Path, scratch_repo: Path) -> None:
    process, sent = _gateway_run(tmp_path, scratch_repo, ["-p", "-"],
                                 input="explain src/app.py\n")
    assert process.returncode == 0, process.stderr
    assert sent.strip() == "explain src/app.py", sent


def test_headless_does_not_wait_for_an_open_stdin_that_sends_nothing(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """A script that leaves stdin open (an unused pipe) must not hang the run."""
    read_fd, write_fd = os.pipe()
    try:
        process, sent = _gateway_run(tmp_path, scratch_repo, ["-p", "say hi"], stdin=read_fd)
    finally:
        os.close(write_fd)
        os.close(read_fd)
    assert process.returncode == 0, process.stderr
    assert sent == "say hi", sent
    assert "stdin" in process.stderr, process.stderr


def test_headless_says_when_it_sends_piped_input(tmp_path: Path, scratch_repo: Path) -> None:
    """In `ls | while read f; do cortex -p ...; done` the first run takes the
    loop's remaining input: it says so, instead of the loop just stopping."""
    process, sent = _gateway_run(tmp_path, scratch_repo, ["-p", "what are these"],
                                 input="a.js\nb.js\n")
    assert process.returncode == 0, process.stderr
    assert "a.js" in sent, sent
    assert ("note: sent 10 bytes of piped input after the prompt "
            "(run with < /dev/null to send none)") in process.stderr, process.stderr
    # With -p - the input is the prompt itself.
    process, _sent = _gateway_run(tmp_path, scratch_repo, ["-p", "-"], input="explain\n")
    assert "piped input" not in process.stderr, process.stderr


def test_headless_log_ends_with_the_model_that_answered(tmp_path: Path, scratch_repo: Path) -> None:
    """The refusal note comes before the "answered by" line, which ends stderr
    as the docs say; a call with no arguments is logged by its name alone."""
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "list_dir", "arguments": {}},
                          {"name": "bash", "arguments": {"command": "ls"}}]},
          {"text": "Listed."}]],
    )
    process = _run_headless(cwd=scratch_repo, env=_worker_env(tmp_path, script),
                            prompt="list", extra_args=[])
    lines = process.stderr.splitlines()
    assert lines[-1].startswith("answered by openai:scripted"), process.stderr
    assert any(line.startswith("note: 1 edit or command call") for line in lines), lines
    assert "[tool] list_dir" in lines, lines


def test_headless_refusal_names_a_rule_saved_for_another_start_directory(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """A saved rule applies where Cortex was started when it was saved. Started
    in a subdirectory, the refusal says which rule exists and where it applies."""
    rules = tmp_path / "home" / ".cortex" / "tool_permissions.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        "version: 2\nrules:\n  - permission: bash\n    pattern: echo done\n"
        f"    action: allow\n    repo: {scratch_repo}\n",
        encoding="utf-8",
    )
    script = _write_script(
        tmp_path / "script.json",
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": "echo done"}}]},
          {"text": "Could not."}]],
    )
    process = _run_headless(cwd=scratch_repo / "src", env=_worker_env(tmp_path, script),
                            prompt="run it", extra_args=[])
    assert process.returncode == 0, process.stderr
    assert (f"note: a saved rule allows `echo done` when Cortex starts in {scratch_repo}; "
            f"this run started in {scratch_repo / 'src'}") in process.stderr, process.stderr


def test_headless_puts_a_misplaced_prompt_right(tmp_path: Path, scratch_repo: Path) -> None:
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    for args, expected in (
        (["-p", "--full-auto", "fix it"],
         'error: -p takes the prompt right after it: cortex -p "fix it" --full-auto'),
        (["fix it"],
         'error: to run one prompt, put -p before it: cortex -p "fix it"'),
        (["--full-auto", "fix it"],
         'error: to run one prompt, put -p before it: cortex -p "fix it" --full-auto'),
    ):
        process = subprocess.run([sys.executable, "-P", "-m", "cortex", *args], cwd=scratch_repo,
                                 env=env, capture_output=True, text=True, timeout=120)
        assert process.returncode == 2, process.stderr
        assert expected in process.stderr, process.stderr


def test_headless_names_cortex_login_when_a_key_is_missing(
    tmp_path: Path, scratch_repo: Path
) -> None:
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    for name in ("CORTEX_SCRIPTED_MODEL", "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"):
        env.pop(name, None)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    process = subprocess.run(
        [sys.executable, "-P", "-m", "cortex", "-p", "hi", "--model", "openai-compatible:org/m"],
        cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 2, process.stderr
    assert (
        "error: openai-compatible is not authenticated. Save a key with "
        "`cortex login openai-compatible <base_url>`, or set OPENAI_COMPATIBLE_API_KEY and "
        "OPENAI_COMPATIBLE_BASE_URL."
    ) in process.stderr, process.stderr

    # With the key set but no base URL, the URL is what it names.
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compat-e2e"
    process = subprocess.run(
        [sys.executable, "-P", "-m", "cortex", "-p", "hi", "--model", "openai-compatible:org/m"],
        cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 2, process.stderr
    assert (
        "error: openai-compatible also needs its base URL: set OPENAI_COMPATIBLE_BASE_URL, or "
        "save it with `cortex login openai-compatible <base_url>`, which saves a key with it."
    ) in process.stderr, process.stderr


def test_an_unknown_option_is_named_without_its_value(tmp_path: Path, scratch_repo: Path) -> None:
    """A key put in an option that does not exist is not repeated into a CI log."""
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    for args in (["-p", "hi", "--key=sk-in-an-option"], ["-p", "hi", "--key", "sk-in-an-option"]):
        process = subprocess.run([sys.executable, "-P", "-m", "cortex", *args], cwd=scratch_repo,
                                 env=env, capture_output=True, text=True, timeout=120)
        assert process.returncode == 2, process.stderr
        assert "unknown option --key" in process.stderr, process.stderr
        assert "sk-in-an-option" not in process.stderr, process.stderr


def _cortex_login(env: dict, cwd: Path, args: list, key_input: str):
    return subprocess.run([sys.executable, "-P", "-m", "cortex", "login", *args], cwd=cwd,
                          env=env, input=key_input, capture_output=True, text=True, timeout=120)


def test_cortex_login_saves_a_key_from_stdin_for_later_headless_runs(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """A machine without the interactive app (CI, a git hook) saves its key
    once; the key is read from stdin, never from the command line."""
    script = _write_script(tmp_path / "script.json", [[{"text": "Hello."}]])
    env = _worker_env(tmp_path, script)
    env.pop("OPENAI_API_KEY")
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    process = _cortex_login(env, scratch_repo, ["openai"], "sk-from-stdin\n")
    assert process.returncode == 0, process.stderr
    assert "API key saved to" in process.stdout, process.stdout
    assert "sk-from-stdin" not in process.stdout + process.stderr
    key_file = tmp_path / "home" / ".cortex" / "credentials.json"
    assert "sk-from-stdin" in key_file.read_text(encoding="utf-8")
    assert key_file.stat().st_mode & 0o777 == 0o600

    run = _run_headless(cwd=scratch_repo, env=env, prompt="hi", extra_args=[])
    assert run.returncode == 0, run.stderr
    assert run.stdout == "Hello.\n"


def test_cortex_login_asks_for_the_key_in_a_terminal_without_showing_it(
    tmp_path: Path, scratch_repo: Path
) -> None:
    import pty

    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    pid, fd = pty.fork()
    if pid == 0:  # the child: cortex login in a terminal
        os.chdir(scratch_repo)
        os.execve(sys.executable, [sys.executable, "-P", "-m", "cortex", "login", "anthropic"], env)
    seen = b""
    deadline = time.time() + 60
    try:
        while b"(not shown): " not in seen and time.time() < deadline:
            seen += os.read(fd, 4096)
        os.write(fd, b"sk-ant-typed\n")
        while time.time() < deadline:
            try:
                chunk = os.read(fd, 4096)
            except OSError:  # Linux: EIO once the child has exited
                break
            if not chunk:
                break
            seen += chunk
    finally:
        _, status = os.waitpid(pid, 0)
        os.close(fd)
    assert os.waitstatus_to_exitcode(status) == 0, seen
    assert b"API key for anthropic (not shown): " in seen, seen
    assert b"sk-ant-typed" not in seen, seen  # not echoed, not printed
    key_file = tmp_path / "home" / ".cortex" / "credentials.json"
    assert "sk-ant-typed" in key_file.read_text(encoding="utf-8")


def test_cortex_login_refuses_what_it_cannot_save_safely(
    tmp_path: Path, scratch_repo: Path
) -> None:
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    for name in ("OPENAI_COMPATIBLE_BASE_URL", "AZURE_OPENAI_ENDPOINT"):
        env.pop(name, None)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    cases = (
        (["openai", "sk-on-the-command-line"], "sk-x\n",
         "error: openai takes no URL. cortex login reads the key from stdin, or asks for it "
         "in a terminal: never put it on the command line"),
        (["azure"], "key\n",
         "error: azure also needs its endpoint: cortex login azure <endpoint>"),
        (["anthropic"], "",
         'error: no key on stdin. Pipe it in (printf %s "$KEY" | cortex login anthropic), '
         "or run cortex login anthropic in a terminal to type it"),
        (["nosuch"], "k\n",
         "error: unknown provider nosuch: the provider comes first (cortex login PROVIDER [URL]), "
         "one of: anthropic, azure, openai, openai-compatible"),
        # The arguments swapped: the URL is not repeated, and the order is named.
        (["http://127.0.0.1:9/v1", "openai-compatible"], "k\n",
         "error: unknown provider: the provider comes first (cortex login PROVIDER [URL])"),
        (["openai-compatible", "http://127.0.0.1:9/v1"], "",
         'no key on stdin. Pipe it in (printf %s "$KEY" | cortex login openai-compatible '
         "<base_url>)"),
        # The key where the URL goes, with or without one on stdin: not repeated back.
        (["openai-compatible", "sk-where-the-url-goes"], "sk-x\n",
         "error: openai-compatible takes its base URL after the provider, and that is not a URL "
         "(it does not start with https:// or http://). The key goes on stdin: "
         'printf %s "$KEY" | cortex login openai-compatible <base_url>'),
        (["openai-compatible", "sk-where-the-url-goes"], "", "that is not a URL"),
    )
    for args, key_input, expected in cases:
        process = _cortex_login(env, scratch_repo, args, key_input)
        assert process.returncode == 2, process.stderr
        assert expected in process.stderr, process.stderr
        assert "sk-where-the-url-goes" not in process.stderr
    assert not (tmp_path / "home" / ".cortex" / "credentials.json").exists()


def test_login_works_without_a_system_keyring(tmp_path: Path, scratch_repo: Path) -> None:
    """On Linux servers, containers and WSL there is often no keyring service:
    /login must still save the key, privately, and the next session finds it."""
    env = _worker_env(tmp_path, tmp_path / "x.json")
    env.pop("ANTHROPIC_API_KEY", None)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        session_id = _start_session(harness)
        saved = harness.wait_response(harness.send("command.execute", {
            "session_id": session_id, "command": "/login anthropic sk-ant-e2e-key"}))["result"]
    finally:
        harness.close()
    key_file = tmp_path / "home" / ".cortex" / "credentials.json"
    assert saved["ok"] is True, saved
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert "sk-ant-e2e-key" not in str(saved)

    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        _start_session(harness)
        status = harness.wait_response(harness.send(
            "cloud.auth.status", {"provider": "anthropic"}))["result"]
    finally:
        harness.close()
    assert status["active_source"] == "file", status


def test_login_brings_back_the_remembered_model_of_that_provider(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """The model chosen last could not start for want of a key; once /login
    adds the key, that model is the active one again."""
    env = _worker_env(tmp_path, tmp_path / "x.json")
    for name in ("CORTEX_SCRIPTED_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"):
        env.pop(name, None)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    state = tmp_path / "home" / ".cortex" / "state.yaml"
    state.parent.mkdir(parents=True)
    state.write_text(
        "last_used_cloud_provider: openai-compatible\n"
        "last_used_cloud_model: org/remembered\n"
        "openai_compatible_base_url: http://127.0.0.1:9/v1\n",
        encoding="utf-8",
    )
    harness = WorkerHarness(cwd=scratch_repo, env=env)
    try:
        session_id = _start_session(harness)
        saved = harness.wait_response(harness.send("command.execute", {
            "session_id": session_id, "command": "/login openai-compatible sk-e2e"}))["result"]
        listing = harness.wait_response(harness.send("model.list", {}))["result"]
    finally:
        harness.close()

    assert saved["ok"] is True, saved
    assert "openai-compatible:org/remembered" in saved["message"], saved
    assert "Next:" not in saved["message"], saved
    active = [row["selector"] for row in listing["cloud"] if row["active"]]
    assert active == ["openai-compatible:org/remembered"], listing


def test_headless_without_a_model_says_to_pass_one(tmp_path: Path, scratch_repo: Path) -> None:
    """A first headless run has no model yet; slash commands do not run
    headless, so the error must name --model, and it is a setup error."""
    env = _worker_env(tmp_path, _write_script(tmp_path / "script.json", [[{"text": "x"}]]))
    # Only an OpenAI-compatible endpoint is set up: nothing picks a default model.
    env.pop("OPENAI_API_KEY")
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-test-compatible"
    env["OPENAI_COMPATIBLE_BASE_URL"] = "http://127.0.0.1:9/v1"
    process = subprocess.run(
        [sys.executable, "-m", "cortex", "-p", "say hi"],
        cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=120,
    )

    assert process.returncode == 2, process.stderr
    assert "--model" in process.stderr and "/model" not in process.stderr
    # The example names the provider whose key is set.
    assert "--model openai-compatible:<model>" in process.stderr, process.stderr
    assert process.stdout.strip() == ""


def _narrating_script(tmp_path: Path) -> Path:
    return _write_script(
        tmp_path / "script.json",
        [[
            {"text": "Looking first.",
             "tool_calls": [{"name": "read_file", "arguments": {"path": "README.md"}}]},
            {"text": "Done."},
        ]],
    )


def test_headless_output_sent_to_a_pipe_is_the_answer_alone(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """A script reads the answer: what the model wrote before its tool calls
    ("Looking first.") goes to stderr with the tool lines."""
    process = _run_headless(
        cwd=scratch_repo, env=_worker_env(tmp_path, _narrating_script(tmp_path)),
        prompt="look", extra_args=[],
    )

    assert process.returncode == 0, process.stderr
    assert process.stdout == "Done.\n"
    assert "Looking first." in process.stderr, process.stderr


def test_headless_in_a_terminal_streams_each_message_as_a_paragraph(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """In a terminal every message shows as it streams, and text after a tool
    call starts a new paragraph instead of running on."""
    import pty
    import select

    master, slave = pty.openpty()
    shown = b""
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "cortex", "-p", "look", "--model", "openai:scripted"],
            cwd=scratch_repo, env=_worker_env(tmp_path, _narrating_script(tmp_path)),
            stdin=subprocess.DEVNULL, stdout=slave, stderr=subprocess.PIPE,
        )
        # This end stays open until all is read: macOS drops unread output
        # once the last holder of the terminal closes it.
        _, stderr = process.communicate(timeout=120)
        while select.select([master], [], [], 0.5)[0]:
            chunk = os.read(master, 65536)
            if not chunk:
                break
            shown += chunk
    finally:
        os.close(slave)
        os.close(master)

    assert process.returncode == 0, stderr
    assert shown.decode().replace("\r\n", "\n") == "Looking first.\n\nDone.\n"


def test_openai_compatible_setup_finishes_inside_the_app(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """An OpenAI-compatible endpoint needs a base URL as well as a key. /login
    takes both, and the model can be selected at once, in the same session."""
    env = _worker_env(tmp_path, tmp_path / "x.json")
    for name in ("OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"):
        env.pop(name, None)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    harness = WorkerHarness(cwd=scratch_repo, env=env)

    def run(command: str) -> dict:
        return harness.wait_response(harness.send(
            "command.execute", {"session_id": session_id, "command": command}))["result"]

    try:
        session_id = _start_session(harness)
        key_only = run("/login openai-compatible sk-compat-e2e")
        not_yet = run("/model openai-compatible:deepseek/deepseek-v4.1-flash")
        bad_url = run("/login openai-compatible sk-compat-e2e ftp://example.com")
        extra = run("/login openai sk-openai-e2e https://example.com")
        both = run("/login openai-compatible sk-compat-e2e http://127.0.0.1:9/v1/")
        selected = run("/model openai-compatible:deepseek/deepseek-v4.1-flash")
        run("/model openai:scripted")
        listed = harness.wait_response(harness.send("model.list", {}))["result"]
    finally:
        harness.close()

    assert key_only["ok"] is True, key_only
    assert "/login openai-compatible <api_key> <base_url>" in key_only["message"]
    assert not_yet["ok"] is False
    assert "/login openai-compatible <api_key> <base_url>" in not_yet["message"]
    assert bad_url["ok"] is False and "http" in bad_url["message"]
    assert extra["ok"] is False and "only an API key" in extra["message"]
    assert both["ok"] is True, both
    assert "http://127.0.0.1:9/v1" in both["message"]
    assert "/model openai-compatible:<model>" in both["message"]  # the next step
    assert selected["ok"] is True, selected
    for result in (key_only, bad_url, extra, both):
        assert "sk-compat-e2e" not in str(result) and "sk-openai-e2e" not in str(result)
    state = (tmp_path / "home" / ".cortex" / "state.yaml").read_text(encoding="utf-8")
    assert "openai_compatible_base_url: http://127.0.0.1:9/v1" in state
    # The /model picker keeps listing a model picked by name after a switch.
    selectors = [row["selector"] for row in listed["cloud"]]
    assert "openai-compatible:deepseek/deepseek-v4.1-flash" in selectors, selectors


def test_edit_prompts_show_the_change_before_it_is_made(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """The permission prompt for an edit carries the diff the edit would make,
    computed without writing; an edit that cannot apply still asks, without
    a diff, and then fails as before."""
    script = _write_script(
        tmp_path / "script.json",
        [[
            {"tool_calls": [{"name": "edit_file", "arguments": {
                "path": "src/app.py", "old_text": "    return 1", "new_text": "    return 2"}}]},
            {"tool_calls": [{"name": "write_file", "arguments": {
                "path": "src/new.py", "content": "VALUE = 1\n"}}]},
            {"tool_calls": [{"name": "edit_file", "arguments": {
                "path": "src/app.py", "old_text": "not in the file", "new_text": "x"}}]},
            {"text": "done"},
        ]],
    )
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    try:
        session_id = _start_session(harness)
        asked = _turn_answering_prompts(harness, session_id, "edit", "reject")
    finally:
        harness.close()

    edit, write, cannot_apply = (prompt["metadata"] for prompt in asked)
    rows = [(row["kind"], row["text"]) for hunk in edit["diff"]["hunks"] for row in hunk["rows"]]
    assert ("del", "    return 1") in rows and ("add", "    return 2") in rows
    assert edit["diff"]["path"] == "src/app.py"
    assert [row["text"] for row in write["diff"]["hunks"][0]["rows"]] == ["VALUE = 1"]
    assert write["diff"]["op"] == "create"
    assert "diff" not in cannot_apply
    # Every prompt was rejected: nothing was written.
    assert (scratch_repo / "src" / "app.py").read_text(encoding="utf-8") == "def main():\n    return 1\n"
    assert not (scratch_repo / "src" / "new.py").exists()


@pytest.mark.parametrize("command", ["/update", "/model openai:gpt-5.5", "/login openai sk-x"])
def test_headless_says_where_an_app_command_runs(
    tmp_path: Path, scratch_repo: Path, command: str
) -> None:
    """A slash command of the interactive app does nothing headless: it is
    not sent to the model (that wastes a turn); the run says where it works."""
    script = _write_script(tmp_path / "script.json", [[{"text": "SENT_TO_MODEL"}]])
    process = _run_headless(
        cwd=scratch_repo, env=_worker_env(tmp_path, script), prompt=command, extra_args=[]
    )

    assert process.returncode == 2, process.stderr
    assert "SENT_TO_MODEL" not in process.stdout
    assert "interactive app" in process.stderr
    if command.startswith("/model"):
        assert "--model" in process.stderr
    if command.startswith("/login"):
        assert "cortex login PROVIDER [URL]" in process.stderr
    assert "sk-x" not in process.stderr


def test_a_run_without_the_key_keeps_the_remembered_model(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """Forgetting the key in a new terminal must not erase the model chosen
    earlier: that run says the key is missing, and the next run with the key
    uses the remembered model again."""
    script = _write_script(tmp_path / "script.json", [[{"text": "PONG"}]] * 3)
    env = _worker_env(tmp_path, script)
    env.pop("OPENAI_API_KEY")
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compat-e2e"
    env["OPENAI_COMPATIBLE_BASE_URL"] = "http://127.0.0.1:9/v1"
    model = "openai-compatible:deepseek/deepseek-v4.1-flash"

    def run(environ: dict, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "cortex", "-p", "ping", *extra],
            cwd=scratch_repo, env=environ, capture_output=True, text=True, timeout=120,
        )

    first = run(env, "--model", model)
    no_key = run({k: v for k, v in env.items() if k != "OPENAI_COMPATIBLE_API_KEY"})
    again = run(env)

    assert first.returncode == 0 and "PONG" in first.stdout, first.stderr
    assert no_key.returncode == 2
    assert model in no_key.stderr and "OPENAI_COMPATIBLE_API_KEY" in no_key.stderr, no_key.stderr
    assert again.returncode == 0 and "PONG" in again.stdout, again.stderr


@pytest.mark.parametrize(
    "args", [["-p", ""], ["-p", "   "], ["--full-auto"], ["--model", "openai:gpt-5.5"]]
)
def test_headless_flags_without_a_prompt_are_refused(
    tmp_path: Path, scratch_repo: Path, args: list[str]
) -> None:
    """An empty prompt, or a headless-only flag with no -p, must not start
    the interactive app (a script would hang on it): it is a usage error."""
    env = _worker_env(tmp_path, _write_script(tmp_path / "script.json", [[{"text": "x"}]]))
    process = subprocess.run(
        [sys.executable, "-m", "cortex", *args],
        cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=30,
        stdin=subprocess.DEVNULL,
    )

    assert process.returncode == 2, (process.stdout[-300:], process.stderr[-300:])
    assert "-p" in process.stderr


def test_help_shows_model_ids_whole_and_hides_the_worker_flag() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "cortex", "--help"], capture_output=True, text=True, timeout=30,
        env={**os.environ, "COLUMNS": "80"},
    )
    assert process.returncode == 0
    assert "openai-compatible:deepseek/deepseek-v4.1-flash" in process.stdout
    assert "--worker-stdio" not in process.stdout


def test_headless_model_without_its_provider_names_the_flag(
    tmp_path: Path, scratch_repo: Path
) -> None:
    env = _worker_env(tmp_path, _write_script(tmp_path / "script.json", [[{"text": "x"}]]))
    process = subprocess.run(
        [sys.executable, "-m", "cortex", "-p", "hi", "--model", "deepseek-v4.1-flash"],
        cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == 2
    assert "--model provider:model" in process.stderr and "/model" not in process.stderr.replace("--model", "")


def test_the_documented_pre_commit_hook_blocks_a_bug_and_fails_closed(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """The hook in docs/cli.md, as written: a review that ends in VERDICT: BUG
    stops the commit, one that ends in VERDICT: OK lets it through, and a
    review that cannot run (no key) stops it too."""
    docs = (REPO_ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    hook = re.search(r"```bash\n(#!/bin/sh\n# \.git/hooks/pre-commit[^`]*)```", docs)
    assert hook, "the pre-commit hook was not found in docs/cli.md"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cortex").write_text(f'#!/bin/sh\nexec "{sys.executable}" -m cortex "$@"\n')
    (bin_dir / "cortex").chmod(0o755)
    script = tmp_path / "script.json"
    env = _worker_env(tmp_path, script)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=scratch_repo, env=env, capture_output=True,
                              text=True, timeout=180)

    git("init", "-q")
    git("config", "user.email", "dev@example.com")
    git("config", "user.name", "Dev")
    git("add", "-A")
    assert git("commit", "-qm", "start").returncode == 0
    hook_path = scratch_repo / ".git" / "hooks" / "pre-commit"
    hook_path.write_text(hook.group(1))
    hook_path.chmod(0o755)

    (scratch_repo / "src" / "app.py").write_text("def main():\n    return 2\n", encoding="utf-8")
    git("add", "src/app.py")
    _write_script(script, [[{"text": "main() now returns the wrong value.\nVERDICT: BUG"}]])
    blocked = git("commit", "-qm", "change")
    assert blocked.returncode == 1, blocked.stderr
    assert "the review found a problem" in blocked.stderr
    assert git("rev-list", "--count", "HEAD").stdout.strip() == "1"

    env.pop("OPENAI_API_KEY")
    unreviewed = git("commit", "-qm", "change")
    assert unreviewed.returncode == 1
    assert "the review did not run (exit 2)" in unreviewed.stderr, unreviewed.stderr
    assert "openai is not authenticated" in (scratch_repo / ".git" / "cortex-review.log").read_text()

    env["OPENAI_API_KEY"] = "sk-test-scripted"
    _write_script(script, [[{"text": "The change is correct.\nVERDICT: OK"}]])
    passed = git("commit", "-qm", "change")
    assert passed.returncode == 0, passed.stderr
    assert git("rev-list", "--count", "HEAD").stdout.strip() == "2"


def test_headless_says_when_a_slash_name_is_no_skill(tmp_path: Path, scratch_repo: Path) -> None:
    """A mistyped skill (/review for /review-changes) still reaches the model,
    as any prompt does, and the run says it ran no skill."""
    script = _write_script(tmp_path / "script.json", [[{"text": "ANSWERED"}]] * 2)
    env = _worker_env(tmp_path, script)
    typo = _run_headless(cwd=scratch_repo, env=env, prompt="/review my changes", extra_args=[])
    assert typo.returncode == 0, typo.stderr
    assert "ANSWERED" in typo.stdout
    assert (
        "note: no skill is named review (skills here: commit-and-pr, map-and-plan, "
        "reproduce-and-fix, review-changes, safe-refactor); the prompt is sent as written"
    ) in typo.stderr, typo.stderr
    skill = _run_headless(cwd=scratch_repo, env=env, prompt="/review-changes my changes",
                          extra_args=[])
    assert skill.returncode == 0, skill.stderr
    assert "no skill is named" not in skill.stderr


def test_a_run_that_fails_remembers_no_model_it_did_not_choose(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """A run whose --model cannot be used must not leave behind the default
    model Cortex picked for the key it did find: the next run without
    --model would send that key to a provider nobody chose."""
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    for name in ("OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"):
        env.pop(name, None)
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-P", "-m", "cortex", "-p", "hi", *args],
                              cwd=scratch_repo, env=env, capture_output=True, text=True,
                              timeout=120)

    failed = run("--model", "openai-compatible:org/m")
    assert failed.returncode == 2, failed.stderr
    state = tmp_path / "home" / ".cortex" / "state.yaml"
    assert "last_used_cloud" not in (state.read_text() if state.exists() else ""), state.read_text()
    again = run()
    assert again.returncode == 2, again.stderr
    assert "error: no model selected" in again.stderr, again.stderr


def test_a_headless_run_says_what_it_refuses_and_which_variables_it_ignores(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """Without --full-auto a run says up front that it refuses edits and
    commands: a model that never tries one leaves no refusal to report. A set
    OPENAI_BASE_URL is named as ignored."""
    script = _write_script(tmp_path / "script.json", [[{"text": "No changes needed."}]] * 2)
    env = _worker_env(tmp_path, script)
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
    plain = _run_headless(cwd=scratch_repo, env=env, prompt="fix the bug", extra_args=[])
    assert plain.returncode == 0, plain.stderr
    assert ("note: this run refuses edits and commands; pass --full-auto to allow them"
            in plain.stderr), plain.stderr
    assert ("warning: Ignored OPENAI_BASE_URL: the openai provider always uses its official "
            "host") in plain.stderr, plain.stderr
    full = _run_headless(cwd=scratch_repo, env=env, prompt="fix the bug",
                         extra_args=["--full-auto"])
    assert full.returncode == 0, full.stderr
    assert "refuses edits" not in full.stderr, full.stderr


def test_a_run_ended_by_a_signal_stops_the_command_it_runs(
    tmp_path: Path, scratch_repo: Path
) -> None:
    """`timeout 5 cortex -p ...` sends SIGTERM: the command Cortex is running
    in its own process group must end with it, not run on as an orphan."""
    import signal

    script = _write_script(tmp_path / "script.json", [[
        {"tool_calls": [{"name": "bash", "arguments": {
            "command": "sleep 31.25; touch late.txt"}}]},
        {"text": "done"},
    ]])
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "-p", "wait", "--model", "openai:scripted",
         "--full-auto"],
        cwd=scratch_repo, env=_worker_env(tmp_path, script), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    _wait_for_sleep(process)
    process.send_signal(signal.SIGTERM)
    _out, err = process.communicate(timeout=30)
    assert process.returncode == 128 + signal.SIGTERM, err
    _assert_sleep_ends()
    assert not (scratch_repo / "late.txt").exists()

    # The worker of the interactive app, ended the same way (the terminal closed).
    harness = WorkerHarness(cwd=scratch_repo, env=_worker_env(tmp_path, script))
    session_id = _start_session(harness)
    harness.send("session.submit_user_input", {
        "session_id": session_id, "user_input": "wait", "active_target": CLOUD_TARGET})
    harness.send("permission.reply", {"session_id": session_id, "request_id": harness.wait_event(
        "permission.asked")["payload"]["request_id"], "reply": "allow_once"})
    _wait_for_sleep(harness.process)
    harness.process.send_signal(signal.SIGHUP)
    harness.process.wait(timeout=30)
    _assert_sleep_ends()
    assert not (scratch_repo / "late.txt").exists()


def _sleeping() -> bool:
    table = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any(line.startswith("sleep 31.25") for line in table.splitlines())


def _wait_for_sleep(process: subprocess.Popen) -> None:
    deadline = time.time() + 60
    while not _sleeping():
        assert time.time() < deadline and process.poll() is None, "the command did not start"
        time.sleep(0.2)


def _assert_sleep_ends() -> None:
    deadline = time.time() + 10
    while _sleeping() and time.time() < deadline:
        time.sleep(0.2)
    assert not _sleeping(), "the command outlived Cortex"


def test_a_failed_connection_names_the_host_and_why(tmp_path: Path, scratch_repo: Path) -> None:
    """`Connection error.` gives a CI author nothing to act on: the message
    names the URL (credentials in it masked) and the cause. A 404 from an
    endpoint of your own also points at its base URL."""
    env = _worker_env(tmp_path, tmp_path / "unused.json")
    env.pop("CORTEX_SCRIPTED_MODEL")
    env["OPENAI_COMPATIBLE_API_KEY"] = "sk-compatible-test"
    env["CORTEX_CLOUD_MAX_RETRIES"] = "0"
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()  # nothing listens there now
    env["OPENAI_COMPATIBLE_BASE_URL"] = f"http://ci:sk-in-the-url@127.0.0.1:{closed_port}/v1"
    command = [sys.executable, "-P", "-m", "cortex", "-p", "hi",
               "--model", "openai-compatible:test-model"]
    refused = subprocess.run(command, cwd=scratch_repo, env=env, capture_output=True, text=True,
                             timeout=120)
    assert refused.returncode == 1, refused.stderr
    assert f"could not reach http://***@127.0.0.1:{closed_port}/v1 (" in refused.stderr, \
        refused.stderr
    assert "sk-in-the-url" not in refused.stderr

    env["OPENAI_COMPATIBLE_BASE_URL"] = f"127.0.0.1:{closed_port}/v1"  # no scheme
    unusable = subprocess.run(command, cwd=scratch_repo, env=env, capture_output=True, text=True,
                              timeout=120)
    assert unusable.returncode == 2, unusable.stderr
    assert (f"openai-compatible's URL 127.0.0.1:{closed_port}/v1 must start with https:// or "
            "http://: fix OPENAI_COMPATIBLE_BASE_URL") in unusable.stderr, unusable.stderr

    server = ChatCompletionsServer([{"status": 404}])
    try:
        env["OPENAI_COMPATIBLE_BASE_URL"] = server.base_url.removesuffix("/v1")
        missing = subprocess.run(command, cwd=scratch_repo, env=env, capture_output=True,
                                 text=True, timeout=120)
    finally:
        server.close()
    assert missing.returncode == 1, missing.stderr
    assert (f"and that {server.base_url.removesuffix('/v1')} is where the API's paths start "
            "(often /v1).") in missing.stderr, missing.stderr
