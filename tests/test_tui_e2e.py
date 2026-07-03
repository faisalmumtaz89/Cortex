"""Deterministic end-to-end tests for the real TUI.

Runs the actual sidecar binary + worker inside tmux with the scripted model
(CORTEX_SCRIPTED_MODEL — no network), sends keystrokes, and asserts on
captured frames. This is the empirical gate for TUI behavior: rendering
(folded tool calls, chronological answer placement), the permission flow's
real side effects, the working animation, and input queueing.

Skipped when tmux or the built sidecar binary is unavailable
(`cd frontend/cortex-tui && bun run build` produces it).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SIDECAR = REPO_ROOT / "cortex" / "ui_runtime" / "bin" / "cortex-tui"

pytestmark = [
    pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed"),
    pytest.mark.skipif(not SIDECAR.exists(), reason="sidecar binary not built"),
]


class TuiSession:
    """Drives the real TUI inside a detached tmux session."""

    def __init__(self, *, project_dir: Path, home_dir: Path, script_path: Path):
        self.name = f"cortex-tui-test-{uuid.uuid4().hex[:8]}"
        env = {
            "HOME": str(home_dir),
            "PYTHONPATH": str(REPO_ROOT),
            "PATH": os.environ.get("PATH", ""),
            "CORTEX_SCRIPTED_MODEL": str(script_path),
            "AZURE_OPENAI_API_KEY": "dummy-key",
            "AZURE_OPENAI_ENDPOINT": "https://dummy.invalid",
            "CORTEX_TUI_FORCE_BUNDLED": "1",
            "TERM": os.environ.get("TERM", "xterm-256color"),
        }
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        launch = (
            f"cd {shlex.quote(str(project_dir))} && "
            f"env {env_prefix} {shlex.quote(sys.executable)} -m cortex"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.name, "-x", "130", "-y", "40", launch],
            check=True,
        )

    def capture(self, *, ansi: bool = False) -> str:
        # -e preserves ANSI escapes so tests can assert on color/attributes.
        args = ["tmux", "capture-pane", "-t", self.name, "-p"]
        if ansi:
            args.append("-e")
        result = subprocess.run(args, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else ""

    def send_line(self, text: str) -> None:
        subprocess.run(["tmux", "send-keys", "-t", self.name, text, "Enter"], check=True)

    def send_key(self, key: str) -> None:
        subprocess.run(["tmux", "send-keys", "-t", self.name, key], check=True)

    def wait_for(self, needle: str, *, timeout: float = 40.0) -> str:
        deadline = time.time() + timeout
        frame = ""
        while time.time() < deadline:
            frame = self.capture()
            if needle in frame:
                return frame
            time.sleep(0.4)
        raise AssertionError(
            f"timed out waiting for {needle!r} in TUI frame.\n--- last frame ---\n{frame}"
        )

    def wait_until(self, predicate, *, description: str, timeout: float = 20.0) -> str:
        """Wait until a predicate over the current frame holds (e.g. for a
        filtered/absent state that wait_for's substring match can't express)."""
        deadline = time.time() + timeout
        frame = ""
        while time.time() < deadline:
            frame = self.capture()
            if predicate(frame):
                return frame
            time.sleep(0.3)
        raise AssertionError(f"timed out waiting for {description}.\n--- last frame ---\n{frame}")

    def close(self) -> None:
        subprocess.run(["tmux", "kill-session", "-t", self.name], capture_output=True)


@pytest.fixture()
def tui_project(tmp_path: Path):
    """Scratch project + isolated HOME; yields a factory for TUI sessions."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()

    sessions: list[TuiSession] = []

    def start(responses) -> TuiSession:
        script = tmp_path / "script.json"
        script.write_text(json.dumps({"responses": responses}), encoding="utf-8")
        session = TuiSession(project_dir=project, home_dir=home, script_path=script)
        sessions.append(session)
        return session

    yield project, start

    for session in sessions:
        session.close()


def _select_scripted_model(session: TuiSession) -> None:
    # "Session ready" is the bootstrap-complete signal (worker handshake done).
    session.wait_for("Session ready")
    session.send_line("/model azure:scripted")
    session.wait_for("Active cloud model set to azure:scripted")


def test_tools_fold_and_answer_renders_after_them(tui_project) -> None:
    project, start = tui_project
    session = start(
        [
            [
                {
                    "text": "Scanning.",
                    "tool_calls": [
                        {"name": "list_dir", "arguments": {"path": "."}},
                        {"name": "read_file", "arguments": {"path": "calc.py"}},
                    ],
                },
                {"text": "FINAL_MARKER: add() returns the sum of a and b."},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("explain calc.py")
    frame = session.wait_for("FINAL_MARKER")

    # Compact one-line tool call: TitleCase verb, no paren wrapping, contents folded.
    assert "Read calc.py" in frame
    assert "return a + b" not in frame
    assert "def add(a, b):" not in frame
    # Chronological: the answer text renders after (below) the tool calls.
    lines = frame.splitlines()
    tool_row = next(i for i, line in enumerate(lines) if "Read calc.py" in line)
    answer_row = next(i for i, line in enumerate(lines) if "FINAL_MARKER" in line)
    assert answer_row > tool_row, f"answer at {answer_row} should be below tools at {tool_row}"
    # Turn metadata footer present and rendered last, after the answer.
    footer_row = next(i for i, line in enumerate(lines) if "▣ Chat" in line)
    assert footer_row > answer_row, f"footer at {footer_row} should be below answer at {answer_row}"

    # Ctrl+O expands the folds in place.
    session.send_key("C-o")
    expanded = session.wait_for("return a + b")
    assert "def add(a, b):" in expanded


def test_fast_tool_running_state_is_visibly_dwelled(tui_project) -> None:
    """A tool that completes in milliseconds must still visibly pass through
    its running presentation (min-dwell): without it the amber state was
    faster than one frame and users only ever saw green."""
    project, start = tui_project
    session = start(
        [
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "calc.py"}}]},
                {"text": "DWELL_DONE"},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("read it")

    # The read completes near-instantly, so only the display dwell makes the
    # running phase observable: the ROW's glyph is an animated spinner while
    # running (the footer working line also spins, so the predicate requires a
    # spinner char on the same line as the tool subject).
    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    session.wait_until(
        lambda frame: any(
            "Read calc.py" in line and any(ch in line for ch in spinner_chars)
            for line in frame.splitlines()
        ),
        description="dwelled running spinner visible on an instant tool row",
        timeout=10.0,
    )

    # And it settles: the answer can arrive while the row is still dwelling
    # amber, so poll until the diamond lands on its exact final green.
    session.wait_for("DWELL_DONE")
    deadline = time.time() + 3
    ansi = ""
    while time.time() < deadline:
        ansi = session.capture(ansi=True)
        if "38;2;16;185;129m◆" in ansi:
            break
        time.sleep(0.2)
    assert "38;2;16;185;129m◆" in ansi, "settled diamond must be exact green (no mid-fade)"
    assert "◆ Read calc.py" in session.capture()


def test_many_tool_calls_collapse_with_compact_lines(tui_project) -> None:
    project, start = tui_project
    for i in range(12):
        (project / f"module_{i}.py").write_text(f"VALUE = {i}\n", encoding="utf-8")
    calls = [{"name": "read_file", "arguments": {"path": f"module_{i}.py"}} for i in range(12)]
    session = start([[{"text": "Scanning.", "tool_calls": calls}, {"text": "COLLAPSE_DONE"}]])
    _select_scripted_model(session)
    session.send_line("review modules")
    frame = session.wait_for("COLLAPSE_DONE")

    # Older tools fold to a summary; only the last few show as one-liners.
    assert "earlier steps (ctrl+o to expand)" in frame
    visible_tools = frame.count("Read module_")
    assert visible_tools <= 6, f"expected <=6 visible tool lines, saw {visible_tools}"
    # The last tool is a single compact line (no separate "⎿ +N lines" fold block).
    assert "module_11.py" in frame

    # Ctrl+O reveals the folded steps (the collapse marker goes away).
    import time

    session.send_key("C-o")
    time.sleep(2)
    expanded = session.capture()
    assert "earlier steps" not in expanded
    assert expanded.count("Read module_") > visible_tools


def test_slash_palette_opens_filters_executes_without_polluting_transcript(tui_project) -> None:
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]])
    session.wait_for("Session ready")

    # Typing '/' opens the palette in the input area (not the chat).
    session.send_key("/")
    palette = session.wait_for("Tab complete", timeout=10)
    assert "/help" in palette and "/status" in palette
    assert "↑↓ select" in palette

    # Filtering narrows the list (check a palette-only description string, since
    # command names like "/download" also appear in the welcome banner).
    session.send_key("he")  # "/he" → filters to /help; no retype/clear races
    filtered = session.wait_until(
        lambda f: "List available commands" in f and "Download a model from HuggingFace" not in f,
        description="palette filtered to /help only",
    )
    assert "Download a model from HuggingFace" not in filtered  # /download filtered out

    # Enter executes the highlighted no-arg /help, rendered from the registry
    # into the ephemeral panel (one "name — description" row per command).
    session.send_key("Enter")
    result = session.wait_for("Exit Cortex", timeout=15)  # the /quit row
    assert "✓ /help" in result
    assert "Esc to dismiss" in result
    assert "/quit (/exit)" in result  # alias surfaced
    assert "[1]" not in result and "[2]" not in result  # no numbered rows

    # Esc dismisses the result panel; nothing command-related remains — this is
    # the no-pollution guarantee (the output never entered the scroll history).
    session.send_key("Escape")
    time.sleep(1.5)
    cleared = session.capture()
    assert "Esc to dismiss" not in cleared
    assert "Ask anything about this repository" in cleared
    assert "Exit Cortex" not in cleared


def test_model_command_opens_interactive_picker_no_numbers(tui_project) -> None:
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]])
    session.wait_for("Session ready")

    # Bare /model opens the interactive picker (not a numbered list).
    session.send_key("/model")
    time.sleep(1)
    session.send_key("Enter")
    picker = session.wait_for("Select a model", timeout=10)
    assert "azure:gpt-5.5" in picker
    assert "↑↓ select · Enter load · Esc cancel" in picker
    # No numbered rows anywhere.
    assert "[1]" not in picker and "- [1]" not in picker
    # A status tag is shown, not an index.
    assert "ready" in picker or "login required" in picker

    # Up from the top wraps to the last entry (azure:gpt-5.5, the only "ready"
    # one); load it via Enter — no typing a number.
    session.send_key("Up")
    time.sleep(0.5)
    session.send_key("Enter")
    result = session.wait_for("Active cloud model set to azure:gpt-5.5", timeout=10)
    assert "✓" in result

    # Esc dismisses the result; picker leaves nothing behind.
    session.send_key("Escape")
    time.sleep(1)
    assert "Select a model" not in session.capture()


def test_slash_palette_esc_clears_input_entirely(tui_project) -> None:
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]])
    session.wait_for("Session ready")

    session.send_key("/stat")
    session.wait_for("Tab complete", timeout=10)
    session.send_key("Escape")
    time.sleep(1.5)
    frame = session.capture()
    # Palette gone AND the slash text is gone from the input.
    assert "Tab complete" not in frame
    assert "/stat" not in frame
    assert "Ask Cortex" in frame  # placeholder returns → input is empty


def test_slash_palette_login_masks_api_key(tui_project) -> None:
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]])
    session.wait_for("Session ready")

    # /login with a space enters arg-hint mode; type a fake secret and run it.
    session.send_key("/login openai sk-supersecret123")
    time.sleep(1)
    session.send_key("Enter")
    result = session.wait_for("Esc to dismiss", timeout=10)
    # The key must be masked and never rendered.
    assert "sk-supersecret123" not in result
    assert "****" in result


def test_clear_resets_the_transcript(tui_project) -> None:
    project, start = tui_project
    session = start([[{"text": "FIRST_ANSWER_MARKER"}]])
    _select_scripted_model(session)
    session.send_line("first question")
    session.wait_for("FIRST_ANSWER_MARKER")

    session.send_line("/clear")
    # The prior answer and question must disappear; welcome screen returns.
    import time

    time.sleep(2)
    frame = session.capture()
    assert "FIRST_ANSWER_MARKER" not in frame
    assert "first question" not in frame
    assert "Ask anything about this repository" in frame


def test_edit_file_renders_green_red_diff(tui_project) -> None:
    project, start = tui_project
    (project / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    session = start(
        [
            [
                {
                    "tool_calls": [
                        {
                            "name": "edit_file",
                            "arguments": {
                                "path": "calc.py",
                                "old_text": "    return a + b",
                                "new_text": "    # subtract instead\n    return a - b",
                            },
                        }
                    ]
                },
                {"text": "EDIT_DONE_MARKER"},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("edit the file")

    modal = session.wait_for("Permission required")
    assert "Edit file: calc.py" in modal
    session.send_key("Enter")  # Allow once

    # The diff summary line only appears once the edit completes (the "Update"
    # verb now shows during running too, so wait on the summary instead).
    diff = session.wait_for("removed line", timeout=20)
    assert "Update calc.py" in diff
    assert "added line" in diff
    # The removed and added source lines both appear with +/- gutters.
    assert "- " in diff and "+ " in diff

    ansi = session.capture(ansi=True)
    assert "48;2;18;50;31" in ansi, "added rows need the green diff background"
    assert "48;2;58;26;29" in ansi, "removed rows need the red diff background"
    assert "38;2;63;185;80" in ansi, "additions need the green gutter/marker"
    assert "38;2;248;81;73" in ansi, "removals need the red gutter/marker"


def test_permission_modal_shows_command_and_executes_it(tui_project) -> None:
    project, start = tui_project
    session = start(
        [
            [
                {
                    "tool_calls": [
                        {"name": "bash", "arguments": {"command": "echo proof > proof.txt"}}
                    ]
                },
                {"text": "WROTE_FILE_MARKER"},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("create the proof file")

    modal = session.wait_for("Permission required")
    assert "Run command: echo proof > proof.txt" in modal
    # Arrow menu (no numbers): "Allow once" is the default highlight.
    assert "❯ Allow once" in modal
    assert "Allow always" in modal and "Reject" in modal
    assert "[1]" not in modal and "[2]" not in modal
    assert "↑↓ select · Enter confirm · Esc reject" in modal

    # Enter confirms the highlighted "Allow once".
    session.send_key("Enter")
    session.wait_for("WROTE_FILE_MARKER")
    assert (project / "proof.txt").read_text(encoding="utf-8").strip() == "proof"


def test_markdown_and_syntax_highlighting_render_with_color(tui_project) -> None:
    project, start = tui_project
    answer = (
        "## Summary\n\n"
        "The `add` function does **integer addition**:\n\n"
        "- takes `a` and `b`\n"
        "- returns their *sum*\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    # simple addition\n"
        "    return a + b\n"
        "```\n\n"
        "See run_all_tests in test_helpers.py. MARKDOWN_DONE"
    )
    session = start([[{"text": answer}]])
    _select_scripted_model(session)
    session.send_line("explain add")
    session.wait_for("MARKDOWN_DONE")
    ansi = session.wait_for("MARKDOWN_DONE", timeout=5)
    ansi = session.capture(ansi=True)

    # Heading: bold (ESC[1m) and heading blue (#61afef -> 97;175;239).
    assert "\x1b[1m" in ansi, "heading should be bold"
    assert "38;2;97;175;239m" in ansi, "heading should use heading color"
    # Bold inline text uses bold attribute; italic uses ESC[3m.
    assert "\x1b[3m" in ansi, "italic text should render with italic attribute"
    # Syntax highlighting inside the code block:
    assert "38;2;198;120;221m" in ansi, "python keywords should be purple"
    assert "38;2;97;175;239m" in ansi, "function names should be blue"
    assert "38;2;92;99;112m" in ansi, "comments should be gray"
    # Code block has its own background fill.
    assert "48;2;27;31;39m" in ansi, "code block should have a background"

    # Structure is preserved as plain text too.
    plain = session.capture()
    assert "• takes" in plain
    assert "def add(a, b):" in plain
    # snake_case identifiers must survive — underscores are not italic markers.
    assert "run_all_tests" in plain
    assert "test_helpers.py" in plain
    # Raw markdown markers must NOT leak into the rendered output.
    assert "## Summary" not in plain
    assert "**integer" not in plain
    assert "```" not in plain


def test_working_spinner_and_queued_input_dispatch(tui_project) -> None:
    project, start = tui_project
    session = start(
        [
            [{"delay_ms": 3500, "text": "SLOW_TURN_DONE"}],
            [{"text": "SECOND_TURN_MARKER"}],
        ]
    )
    _select_scripted_model(session)
    session.send_line("first request")
    session.wait_for("Working…", timeout=10)
    session.send_line("second request")
    queued_frame = session.wait_for("↳ queued: second request", timeout=10)
    assert "Working…" in queued_frame

    final = session.wait_for("SECOND_TURN_MARKER")
    assert "SLOW_TURN_DONE" in final
    assert "↳ queued" not in final
