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
import re
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

    def __init__(
        self,
        *,
        project_dir: Path,
        home_dir: Path,
        script_path: Path,
        extra_env: dict | None = None,
        python_flags: str = "",
    ):
        self.name = f"cortex-tui-test-{uuid.uuid4().hex[:8]}"
        self.log_file = home_dir / ".cortex" / "cortex.log"
        env = {
            "HOME": str(home_dir),
            "PYTHONPATH": str(REPO_ROOT),
            "PATH": os.environ.get("PATH", ""),
            "CORTEX_SCRIPTED_MODEL": str(script_path),
            "AZURE_OPENAI_API_KEY": "dummy-key",
            "AZURE_OPENAI_ENDPOINT": "https://dummy.invalid",
            "CORTEX_TUI_FORCE_BUNDLED": "1",
            # Hermeticity: never probe GitHub for releases from the suite.
            # The update e2e test opts back in with a local stub probe server.
            "CORTEX_AUTO_UPDATE_CHECK": "false",
            "TERM": os.environ.get("TERM", "xterm-256color"),
            # A truecolor terminal, as the color assertions expect: without it
            # OpenTUI falls back to 256 colors under TERM=xterm-256color.
            "COLORTERM": "truecolor",
        }
        env.update(extra_env or {})
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        launch = (
            f"cd {shlex.quote(str(project_dir))} && "
            f"env {env_prefix} {shlex.quote(sys.executable)} {python_flags} -m cortex"
        )
        # remain-on-exit keeps a pane whose TUI exited, with its last output
        # and exit status, so a failing wait can say why instead of showing "".
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.name, "-x", "130", "-y", "40", launch,
             ";", "set-option", "-t", self.name, "remain-on-exit", "on"],
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

    def exited(self) -> bool:
        """True once the launched `python -m cortex` has exited (or the session is gone)."""
        result = subprocess.run(
            ["tmux", "display-message", "-p", "-t", self.name, "#{pane_dead}"],
            capture_output=True,
            text=True,
        )
        return result.returncode != 0 or result.stdout.strip() == "1"

    def _failure(self, waiting_for: str, frame: str) -> AssertionError:
        status = subprocess.run(
            ["tmux", "display-message", "-p", "-t", self.name, "#{pane_dead} #{pane_dead_status}"],
            capture_output=True,
            text=True,
        )
        pane = status.stdout.strip() if status.returncode == 0 else f"gone: {status.stderr.strip()}"
        log = ""
        if self.log_file.exists():
            log = self.log_file.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(log.splitlines()[-40:])
        tui_log = self.log_file.with_name("cortex-tui.log")
        errors = tui_log.read_text(encoding="utf-8", errors="replace") if tui_log.exists() else ""
        return AssertionError(
            f"{waiting_for}.\n--- pane (dead, exit status) ---\n{pane}"
            f"\n--- last frame ---\n{frame}\n--- cortex.log ---\n{tail}"
            f"\n--- cortex-tui.log ---\n{errors}"
        )

    def wait_for(self, needle: str, *, timeout: float = 40.0) -> str:
        return self.wait_until(
            lambda frame: needle in frame, description=f"{needle!r} in TUI frame", timeout=timeout
        )

    def wait_until(
        self, predicate, *, description: str, timeout: float = 20.0, interval: float = 0.3
    ) -> str:
        """Wait until a predicate over the current frame holds (e.g. for a
        filtered/absent state that wait_for's substring match can't express).
        `interval` is the pause between captures; 0 catches a state that shows
        for under half a second, even on a slow machine."""
        deadline = time.time() + timeout
        frame = ""
        while time.time() < deadline:
            frame = self.capture()
            if predicate(frame):
                # A capture can land mid-repaint: look again and keep the
                # later frame when it still matches.
                time.sleep(0.15)
                settled = self.capture()
                return settled if predicate(settled) else frame
            if self.exited():
                raise self._failure(f"the TUI exited while waiting for {description}", frame)
            time.sleep(interval)
        raise self._failure(f"timed out waiting for {description}", frame)

    def process_tree(self) -> set[int]:
        """PIDs of every process started in the pane (launcher, sidecar, worker)."""
        result = subprocess.run(
            ["tmux", "display-message", "-p", "-t", self.name, "#{pane_pid}"],
            capture_output=True,
            text=True,
            check=True,
        )
        table = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True).stdout
        children: dict[int, list[int]] = {}
        for line in table.splitlines():
            pid, ppid = (int(field) for field in line.split())
            children.setdefault(ppid, []).append(pid)
        tree, pending = set(), [int(result.stdout.strip())]
        while pending:
            pid = pending.pop()
            tree.add(pid)
            pending.extend(children.get(pid, []))
        return tree

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

    def start(
        responses, *, extra_env: dict | None = None, python_flags: str = ""
    ) -> TuiSession:
        script = tmp_path / "script.json"
        script.write_text(json.dumps({"responses": responses}), encoding="utf-8")
        session = TuiSession(
            project_dir=project,
            home_dir=home,
            script_path=script,
            extra_env=extra_env,
            python_flags=python_flags,
        )
        sessions.append(session)
        return session

    yield project, start

    for session in sessions:
        session.close()
    # Any error the TUI hit, even one it drew past, fails the test, except
    # one known in OpenTUI 0.1.107: now and then a frame's call into its
    # WebAssembly layout engine (Yoga) traps with "access to a null
    # reference". The frame is dropped and the next one draws; OpenTUI 0.5
    # and later do not use Yoga.
    log = home / ".cortex" / "cortex-tui.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r"\n(?=\d{4}-\d\d-\d\dT)", text.strip())
        unknown = [
            entry for entry in entries
            if not ("access to a null reference" in entry and "activateFrame" in entry)
        ]
        assert not unknown, "\n".join(unknown)


def _picker_rows() -> list[str]:
    """The /model picker's rows exactly as the TUI builds them: the row that
    starts naming an Azure deployment (azure is logged in through the
    harness's env), then every catalog entry in catalog order (the harness's
    isolated HOME has no override file). Deriving rows from the real catalog
    keeps the picker test decoupled from catalog cardinality — a routine
    model-list edit cannot silently break it."""
    from cortex.cloud.catalog import CloudModelCatalog

    catalog = CloudModelCatalog(override_path=Path("/nonexistent-cloud-override.json"))
    return ["azure:… type a deployment name", *(ref.selector for ref in catalog.list_models())]


def _selection_window_size() -> int:
    """The shared picker window (rows shown before '+N more'), read from the
    component's own default so the test tracks the source of truth."""
    source = (REPO_ROOT / "frontend/cortex-tui/src/components/selection_list.tsx").read_text(
        encoding="utf-8"
    )
    match = re.search(r"maxVisibleRows \?\? (\d+)", source)
    assert match, "selection_list.tsx window default not found"
    return int(match.group(1))


def _select_scripted_model(session: TuiSession) -> None:
    # "Session ready" is the bootstrap-complete signal (worker handshake done).
    session.wait_for("Session ready")
    session.send_line("/model azure:scripted")
    session.wait_for("azure:scripted — now active.")


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


def test_a_git_tool_row_reads_as_the_git_command_and_asks_nothing(tui_project) -> None:
    project, start = tui_project
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    session = start(
        [[{"tool_calls": [{"name": "git", "arguments": {"command": "diff", "staged": True,
                                                         "stat": True}}]},
          {"text": "GIT_DONE_MARKER"}]]
    )
    _select_scripted_model(session)
    session.send_line("what is staged")
    frame = session.wait_for("GIT_DONE_MARKER")
    assert "Git diff --staged --stat" in frame, frame
    assert "Permission required" not in frame, frame


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
        interval=0,  # the dwell lasts 450 ms: two tmux calls and a pause can miss it
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
    assert "earlier steps (Ctrl+O to expand)" in frame
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

    # Filtering narrows the list (check palette-only description strings, since
    # command names like "/model" also appear in the welcome banner).
    session.send_key("he")  # "/he" → filters to /help; no retype/clear races
    filtered = session.wait_until(
        lambda f: "List available commands" in f and "Show the active model" not in f,
        description="palette filtered to /help only",
    )
    assert "Show the active model" not in filtered  # /status filtered out

    # Enter executes the highlighted no-arg /help, rendered from the registry
    # into the ephemeral panel (one "name — description" row per command).
    session.send_key("Enter")
    result = session.wait_for("Exit Cortex", timeout=15)  # the /quit row
    assert "✓ /help" in result
    assert "Esc to dismiss" in result
    assert "/quit (/exit)" in result  # alias surfaced
    assert "[1]" not in result and "[2]" not in result  # no numbered rows
    # Every name, the longest skill's included, keeps a gap before its text.
    for name in ("/reproduce-and-fix", "/review-changes", "/quit (/exit)"):
        assert re.search(re.escape(name) + r" {2,}\S", result), (name, result)

    # Esc dismisses the result panel; nothing command-related remains — this is
    # the no-pollution guarantee (the output never entered the scroll history).
    session.send_key("Escape")
    time.sleep(1.5)
    cleared = session.capture()
    assert "Esc to dismiss" not in cleared
    assert "Ask anything about this repository" in cleared
    assert "Exit Cortex" not in cleared


def test_a_skill_is_in_the_palette_and_its_slash_command_runs_a_turn(tui_project) -> None:
    project, start = tui_project
    skill = project / ".cortex" / "skills" / "look-only"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: look-only\ndescription: Use to look around without changing files.\n"
        "allowed-tools: [read_file]\n---\nOnly look.\n",
        encoding="utf-8",
    )
    session = start(
        [
            [
                {"tool_calls": [{"name": "write_file", "arguments": {
                    "path": "made.txt", "content": "x"}}]},
                {"text": "SKILL_TURN_DONE"},
            ]
        ]
    )
    _select_scripted_model(session)

    session.send_key("/look")
    palette = session.wait_for("Use to look around without changing files.", timeout=10)
    assert "/look-only" in palette
    session.send_key("Tab")  # completes to "/look-only " for the request
    session.send_key("check calc.py")
    session.send_key("Enter")

    frame = session.wait_for("SKILL_TURN_DONE")
    assert "/look-only check calc.py" in frame  # sent as a message, not a command
    assert "Unknown command" not in frame
    assert "Permission required" not in frame
    assert not (project / "made.txt").exists()  # the skill's tool limit held


def test_model_picker_lists_models_and_selects_with_wraparound(tui_project) -> None:
    """Bare /model opens one list of provider:model rows with auth tags,
    windowed to the selection; Up from the top wraps to the last row; Enter
    selects it; reopening lands on the active row; no numbered selection."""
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]])
    session.wait_for("Session ready")

    # Palette Tab completes commands: "/mod" + Tab → arg-hint for /model.
    session.send_key("/mod")
    time.sleep(0.8)
    session.send_key("Tab")
    session.wait_for("<provider:model>", timeout=10)
    session.send_key("Escape")
    time.sleep(0.8)

    # Rows and windowing are DERIVED (real catalog + the picker's own window
    # default) so a routine catalog edit cannot silently break this test.
    rows = _picker_rows()
    window = _selection_window_size()
    hidden = max(0, len(rows) - window)
    first_row, last_row = rows[0], rows[-1]
    # The wrap-select leg needs the env-authenticated azure deployment as the
    # catalog's final row; if the order ever changes, update that leg too.
    assert last_row.startswith("azure:"), f"catalog order changed: {rows}"

    session.send_key("/model")
    time.sleep(1)
    session.send_key("Enter")
    # A capture can land mid-repaint (title drawn, rows not yet): wait for the
    # whole picker, its rows and its own footer, before asserting on it.
    picker = session.wait_until(
        lambda frame: "Select a model" in frame
        and "Enter use" in frame
        and all(row in frame for row in rows[:window]),
        description="the drawn /model picker",
        timeout=10,
    )
    for visible_row in rows[:window]:
        assert visible_row in picker, f"windowed row missing: {visible_row}"
    for windowed_out in rows[window:]:
        assert windowed_out not in picker, f"row should be windowed out: {windowed_out}"
    if hidden:
        assert f"+{hidden} more" in picker
    assert first_row in picker
    assert "ready" in picker or "login required" in picker
    assert "Local" not in picker and "Tab" not in picker
    assert "[1]" not in picker and "- [1]" not in picker

    # Up from the top wraps to the LAST entry — the windowing follows the
    # selection, revealing it; select it via Enter (azure is the one
    # env-authenticated provider in this harness).
    session.send_key("Up")
    session.wait_for(last_row, timeout=10)
    session.send_key("Enter")
    result = session.wait_for(f"{last_row} — now active.", timeout=10)
    assert "✓" in result
    session.send_key("Escape")
    time.sleep(1)
    assert "Select a model" not in session.capture()

    # Reopen: the window is scrolled to the active (last) row.
    session.send_key("/model")
    time.sleep(1)
    session.send_key("Enter")
    reopened = session.wait_until(
        lambda frame: "Select a model" in frame and "Enter use" in frame and last_row in frame,
        description="the reopened /model picker on the active row",
        timeout=10,
    )
    assert last_row in reopened
    session.send_key("Escape")
    time.sleep(1)
    assert "Select a model" not in session.capture()


def test_single_ctrl_c_exits_cleanly(tui_project) -> None:
    """ONE Ctrl+C must exit the whole TUI. Regression: OpenTUI's exitOnCtrlC
    only destroyed the renderer — the spinner interval and worker pipes kept
    the sidecar alive, so the pane looked dead until a second Ctrl+C."""
    project, start = tui_project
    session = start([[{"text": "hi"}]])
    session.wait_for("Session ready")

    session.send_key("C-c")

    # The launch command IS the pane: when python -m cortex exits, the pane
    # is dead. Old behavior: the pane lingered indefinitely. Give the teardown
    # a generous-but-bounded window.
    deadline = time.time() + 8
    alive = True
    while time.time() < deadline:
        if session.exited():
            alive = False
            break
        time.sleep(0.25)
    if alive:
        tree = session.process_tree()
        states = subprocess.run(
            ["ps", "-o", "pid=,stat=,args=", "-p", ",".join(map(str, tree))],
            capture_output=True,
            text=True,
        ).stdout
        raise session._failure(
            f"TUI still running 8s after a single Ctrl+C; processes:\n{states}", session.capture()
        )


def test_closing_the_terminal_ends_the_tui_and_its_worker(tui_project) -> None:
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]])
    session.wait_for("Session ready")
    tree = session.process_tree()
    assert len(tree) >= 3, tree  # launcher, sidecar, worker

    session.close()

    def alive() -> set[int]:
        table = subprocess.run(["ps", "-eo", "pid=,stat="], capture_output=True, text=True).stdout
        running = {int(pid) for pid, stat in (line.split() for line in table.splitlines())
                   if not stat.startswith("Z")}
        return tree & running

    deadline = time.time() + 8
    while alive() and time.time() < deadline:
        time.sleep(0.25)
    assert not alive(), f"processes outlived their terminal: {sorted(alive())}"


def test_worker_never_imports_modules_from_the_project(tui_project) -> None:
    project, start = tui_project
    marker = project / "imported-from-project"
    (project / "yaml.py").write_text(
        f"open({str(marker)!r}, 'w').close()\nraise SystemExit(3)\n", encoding="utf-8"
    )
    # -P keeps the project off the launcher's own sys.path, as the installed
    # `cortex` script does, so only the worker spawn is under test.
    session = start([[{"text": "IGNORED"}]], python_flags="-P")
    session.wait_until(
        lambda frame: "Session ready" in frame or marker.exists(),
        description="worker ready or project module imported",
        timeout=40.0,
    )
    assert not marker.exists()


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


def test_login_picker_shows_each_provider_auth_status(tui_project) -> None:
    project, start = tui_project
    session = start(
        [[{"text": "IGNORED"}]], extra_env={"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9/v1"}
    )
    session.wait_for("Session ready")

    session.send_key("/login")
    time.sleep(1)
    session.send_key("Enter")
    picker = session.wait_for("Log in to a provider", timeout=10)

    def row(provider: str) -> str:
        return next(line for line in picker.splitlines() if f" {provider} " in line)

    assert "logged in · env" in row("azure"), picker  # the harness sets its key
    assert "not logged in" in row("anthropic"), picker
    # A base URL alone is half the setup: the row says the key is what is missing.
    assert "URL saved, no key" in row("openai-compatible"), picker


def test_the_tui_leaves_no_files_in_the_temp_folder(tui_project, tmp_path) -> None:
    """The compiled sidecar unpacks a native library into the temp folder at
    every start; Cortex must remove it when the TUI exits, while commands the
    agent runs still see the user's own TMPDIR."""
    project, start = tui_project
    user_tmp = tmp_path / "user-tmp"
    user_tmp.mkdir()
    session = start(
        [[
            {"tool_calls": [{"name": "bash", "arguments": {
                "command": 'printf %s "$TMPDIR" > tmpdir.txt'}}]},
            {"text": "TMP_DONE"},
        ]],
        extra_env={"TMPDIR": str(user_tmp)},
    )
    _select_scripted_model(session)
    session.send_line("record the temp folder")
    session.wait_for("Permission required")
    session.send_key("Enter")
    session.wait_for("TMP_DONE")
    assert (project / "tmpdir.txt").read_text(encoding="utf-8") == str(user_tmp)

    session.send_key("C-c")
    session.wait_until(lambda frame: session.exited(), description="the TUI to exit", timeout=10)
    leftovers = [p.name for p in user_tmp.rglob("*") if p.is_file()]
    assert leftovers == [], leftovers


def test_replies_wrap_between_words_after_non_ascii_text(tui_project) -> None:
    """A line of styled text with a non-ASCII character (an em dash) after its
    first span used to wrap one character off, and every later line with it:
    "cannot be see" / "n from here". Rows must break only between words."""
    project, start = tui_project
    risks = (
        "nothing in the repo calls greet, so no call sites break; the only compatibility "
        "risk is callers outside the repo that pass a second positional argument today "
        "(unknown — they cannot be seen from here) or that would need greeting to be "
        "keyword-only. Behaviour for an empty or None greeting is unvalidated, so the "
        "current code has no validation either. The stale cache needs no migration — "
        "Python recompiles it because of the version tag."
    )
    session = start([[{"text": f"**Risks**: {risks}\n\nWRAP_DONE"}]])
    _select_scripted_model(session)
    session.send_line("risks")
    frame = session.wait_for("WRAP_DONE")

    lines = frame.splitlines()
    first = next(i for i, line in enumerate(lines) if "Risks:" in line)
    paragraph = []
    for line in lines[first:]:
        if not line.strip() or "WRAP_DONE" in line:
            break
        paragraph.append(line.strip())
    assert " ".join(paragraph) == f"Risks: {risks}", paragraph


def test_login_and_model_typed_fast_open_their_pickers(tui_project) -> None:
    """"/login" and Enter arriving together (fast typing, a paste) must open
    the provider picker, as a slower Enter does, not fail with usage text."""
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]])
    session.wait_for("Session ready")

    session.send_line("/login")
    picker = session.wait_for("Log in to a provider", timeout=10)
    assert "Usage:" not in picker
    session.send_key("Escape")
    time.sleep(0.8)

    session.send_line("/model")
    session.wait_until(
        lambda frame: "Select a model" in frame and "Enter use" in frame,
        description="the /model picker from a fast '/model' + Enter",
        timeout=10,
    )


def test_openai_compatible_login_takes_the_base_url_and_needs_no_restart(tui_project) -> None:
    """Picking openai-compatible in the /login picker asks for the key AND the
    base URL; after that the model can be selected at once, in the same
    session. "/login " and "/model " with nothing after them open their
    pickers instead of failing with a usage error."""
    project, start = tui_project
    session = start(
        [[{"text": "IGNORED"}]],
        extra_env={"PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring"},
    )
    session.wait_for("Session ready")

    session.send_key("/login ")
    time.sleep(1)
    session.send_key("Enter")
    session.wait_for("Log in to a provider", timeout=10)
    for _ in range(3):  # openai, anthropic, azure, openai-compatible
        session.send_key("Down")
        time.sleep(0.3)
    session.send_key("Enter")
    hint = session.wait_until(
        lambda frame: "/login openai-compatible" in frame and "<base_url>" in frame,
        description="the key and base URL prompt for openai-compatible",
        timeout=10,
    )
    assert "<api_key>" in hint

    session.send_line("sk-tui-e2e http://127.0.0.1:9/v1")
    saved = session.wait_for("Using http://127.0.0.1:9/v1.", timeout=15)
    assert "sk-tui-e2e" not in saved
    session.send_key("Escape")
    time.sleep(0.8)

    # The picker has a row for the provider just logged in, though its
    # models are named by the user: Enter starts the /model command.
    session.send_line("/model")
    picker = session.wait_for("Select a model", timeout=10)
    assert "❯ openai-compatible:… type a model name" in picker, picker
    session.send_key("Enter")
    session.wait_until(
        lambda frame: "Select a model" not in frame
        and "/model openai-compatible:" in frame
        and "Type the model name as your endpoint calls it" in frame,
        description="the prompt prefilled with /model openai-compatible: and what to type",
        timeout=10,
    )
    session.send_line("deepseek/deepseek-v4.1-flash")
    active = session.wait_for("openai-compatible:deepseek/deepseek-v4.1-flash — now active.",
                              timeout=15)
    # A 130-column header has room for the whole name.
    assert "openai-compatible:deepseek/deepseek-v4.1-flash · idle" in active.splitlines()[1], active
    session.send_key("Escape")
    time.sleep(0.8)

    session.send_key("/model ")
    time.sleep(1)
    session.send_key("Enter")
    picker = session.wait_until(
        lambda frame: "Select a model" in frame and "Enter use" in frame,
        description="the /model picker from '/model ' + Enter",
        timeout=10,
    )
    assert "openai-compatible:deepseek/deepseek-v4.1-flash" in picker
    assert "/model provider:model" in picker  # how to pick a model the list lacks
    session.send_key("Escape")
    time.sleep(0.8)

    # Logging in again: the saved URL is used, so only the key is asked for.
    session.send_key("/login openai-compatible ")
    hint = session.wait_until(
        lambda frame: "[base_url]" in frame and "the saved URL is used" in frame,
        description="the /login hint with the base URL optional",
        timeout=10,
    )
    assert "<base_url>" not in hint


def test_bare_login_opens_provider_picker_and_prefills_key_prompt(tui_project) -> None:
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]])
    session.wait_for("Session ready")

    # Bare /login + Enter opens the provider picker (not arg-hint mode).
    session.send_key("/login")
    time.sleep(1)
    session.send_key("Enter")
    picker = session.wait_for("Log in to a provider", timeout=10)
    assert "openai" in picker
    assert "anthropic" in picker
    assert "azure" in picker
    # Auth-status tags render (azure has env creds in this harness; the others
    # may or may not — at least one status tag of each kind's vocabulary shows).
    assert "logged in" in picker
    # No numbered selection anywhere.
    assert "[1]" not in picker

    # Enter on the highlighted provider pre-fills the key prompt (arg-hint).
    session.send_key("Down")  # anthropic
    time.sleep(0.4)
    session.send_key("Enter")
    session.wait_until(
        lambda frame: "Log in to a provider" not in frame and "/login" in frame,
        description="picker closed into pre-filled /login prompt",
        timeout=10.0,
    )
    frame = session.capture()
    assert "/login anthropic" in frame  # pre-filled input, awaiting the key

    # Esc cancels cleanly: input empties, no picker, no residue.
    session.send_key("Escape")
    time.sleep(1)
    cleared = session.capture()
    assert "Log in to a provider" not in cleared
    assert "/login anthropic" not in cleared

    # Re-open and Esc directly from the picker also leaves a clean input.
    session.send_key("/login")
    time.sleep(0.8)
    session.send_key("Enter")
    session.wait_for("Log in to a provider", timeout=10)
    session.send_key("Escape")
    time.sleep(1)
    assert "Log in to a provider" not in session.capture()


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


def test_a_message_that_starts_with_a_command_name_goes_to_the_model(tui_project) -> None:
    """Only a command name typed alone runs without its slash: "Clear up the
    docstring…" once ran /clear and wiped the model's context."""
    project, start = tui_project
    session = start([[{"text": "FIRST_ANSWER_MARKER"}], [{"text": "SECOND_ANSWER_MARKER"}]])
    session.wait_for("No model loaded. Pick one with /model")
    _select_scripted_model(session)
    # How to get a model leaves the transcript once one is active.
    session.wait_until(
        lambda frame: "No model loaded" not in frame, description="the setup hint gone"
    )
    session.send_line("first question")
    session.wait_for("FIRST_ANSWER_MARKER")

    session.send_line("Clear up the docstring wording in calc.py")
    frame = session.wait_for("SECOND_ANSWER_MARKER")
    assert "Conversation cleared" not in frame, frame
    assert "FIRST_ANSWER_MARKER" in frame, frame
    assert "Clear up the docstring wording in calc.py" in frame, frame

    session.send_line("help")  # alone, a command name still runs the command
    session.wait_for("Esc to dismiss", timeout=15)


def test_a_small_screen_edit_prompt_shows_the_change_itself(tui_project) -> None:
    """At 80x24 there is room for a few diff rows: they must be the change,
    not the unchanged lines above it."""
    project, start = tui_project
    lines = [f"line_{n} = {n}" for n in range(1, 9)]
    (project / "values.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    session = start(
        [
            [
                {"tool_calls": [{"name": "edit_file", "arguments": {
                    "path": "values.py", "old_text": "line_6 = 6", "new_text": "line_6 = 60"}}]},
                {"text": "EDIT_DONE_MARKER"},
            ]
        ]
    )
    _select_scripted_model(session)
    subprocess.run(["tmux", "resize-window", "-t", session.name, "-x", "80", "-y", "24"],
                   check=True)
    session.send_line("edit the file")
    prompt = session.wait_for("Permission required")
    assert "6 + line_6 = 60" in prompt, prompt
    assert "unchanged lines above" in prompt, prompt


def test_a_wide_table_fits_the_screen_and_shows_no_markup(tui_project) -> None:
    """A table wider than the screen is narrowed to fit, so each row stays on
    its own lines, and a cell shows its text without backticks or **."""
    project, start = tui_project
    table = (
        "| Command | Before | After |\n"
        "|---|---|---|\n"
        "| `python3 -m unittest discover -s tests -v` (the README's command, run from the "
        "project root with the default settings) | 4 run, 1 FAIL | 4 run, **OK** |\n"
    )
    session = start([[{"text": f"Results:\n\n{table}\nTABLE_DONE_MARKER"}]])
    _select_scripted_model(session)
    session.send_line("show the results")
    frame = session.wait_for("TABLE_DONE_MARKER")
    row = next((line for line in frame.splitlines() if "4 run, 1 FAIL" in line), "")
    assert "4 run, OK" in row, frame  # the last column did not wrap to the next line
    assert "`" not in frame and "**" not in frame, frame
    separator = next((line for line in frame.splitlines() if "┼" in line), "")
    assert separator, frame
    assert len(separator.rstrip()) <= 130, frame


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
    # The prompt shows the change before it is approved: the line it removes,
    # the lines it adds, and their line numbers, while the file is untouched.
    prompt = session.wait_until(
        lambda frame: "- " in frame and "# subtract instead" in frame,
        description="the edit's diff in the permission prompt",
        timeout=10,
    )
    assert "2 -     return a + b" in prompt
    assert "2 +     # subtract instead" in prompt and "3 +     return a - b" in prompt
    assert "return a + b" in (project / "calc.py").read_text(encoding="utf-8")
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


def test_enter_after_typing_a_message_queues_it_and_leaves_the_prompt_open(tui_project) -> None:
    """A prompt that appears while you type is not answered by the Enter that
    ends your sentence: the message is queued, and the prompt still waits."""
    project, start = tui_project
    session = start(
        [
            [
                {"tool_calls": [{"name": "bash", "arguments": {"command": "echo proof > proof.txt"}}]},
                {"text": "WROTE_FILE_MARKER"},
            ],
            [{"text": "FOLLOW_UP_MARKER"}],
        ]
    )
    _select_scripted_model(session)
    session.send_line("create the proof file")
    modal = session.wait_for("Waiting for your answer to the prompt above")
    assert "Permission required" in modal and "Working…" not in modal, modal
    assert "Enter answer prompt" in modal, modal

    session.send_line("also update the README")
    frame = session.wait_for("↳ queued: also update the README", timeout=10)
    assert "Permission required" in frame, frame
    assert not (project / "proof.txt").exists()

    session.send_key("Enter")  # an empty input: Enter answers the prompt
    session.wait_for("WROTE_FILE_MARKER")
    session.wait_for("FOLLOW_UP_MARKER")  # the queued message ran next
    assert (project / "proof.txt").read_text(encoding="utf-8").strip() == "proof"


def test_permission_modal_shows_the_directory_a_command_runs_in(tui_project) -> None:
    project, start = tui_project
    (project / "sub").mkdir()
    session = start(
        [
            [
                {"tool_calls": [{"name": "bash", "arguments": {
                    "command": "echo here > here.txt", "workdir": "sub"}}]},
                {"text": "RAN_IN_SUB_MARKER"},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("write the file in sub")

    modal = session.wait_for("Permission required")
    rows = [line.strip().lstrip("┃").strip() for line in modal.splitlines()]
    assert "Run command: cd ./sub || exit 1\\n" in rows, modal
    assert "echo here > here.txt" in rows, modal

    session.send_key("Enter")
    session.wait_for("RAN_IN_SUB_MARKER")
    assert (project / "sub" / "here.txt").exists()


def test_permission_modal_shows_a_command_of_several_lines_line_by_line(tui_project) -> None:
    """Each line of a command starts a row of the prompt, its line break still
    shown as \\n, instead of one run of escapes."""
    project, start = tui_project
    command = "cd src\npython3 -m unittest discover -s tests\necho done"
    session = start(
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": command}}]},
          {"text": "DONE_MARKER"}]]
    )
    _select_scripted_model(session)
    session.send_line("run it")
    modal = session.wait_for("Permission required")
    rows = [line.strip().lstrip("┃").strip() for line in modal.splitlines()]
    assert "Run command: cd src\\n" in rows, modal
    assert "python3 -m unittest discover -s tests\\n" in rows, modal
    assert "echo done" in rows, modal
    assert "❯ Allow once" in modal, modal  # all of it is shown, so Enter may allow it
    session.send_key("Escape")
    session.wait_for("DONE_MARKER")


def test_permission_modal_shows_every_line_of_a_command(tui_project) -> None:
    project, start = tui_project
    command = "echo hello" + " " * 20000 + "\n" * 60 + "; touch HIDDEN.txt"
    session = start(
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": command}}]},
          {"text": "DONE_MARKER"}]]
    )
    _select_scripted_model(session)
    session.send_line("run it")

    modal = session.wait_for("Permission required")
    flat = "".join(line.strip().lstrip("┃").strip() for line in modal.splitlines())
    # The blank run is counted, and each line break stays visible.
    assert "⟨20000 blank or invisible characters⟩\\n\\n" in flat, modal
    assert "\\n; touch HIDDEN.txt" in flat, modal
    assert "Allow once" in modal and "Reject" in modal
    session.send_key("Escape")
    session.wait_for("DONE_MARKER")
    assert not (project / "HIDDEN.txt").exists()


def test_permission_modal_shows_a_long_command_whole_when_it_fits(tui_project) -> None:
    """A command of a few hundred characters fills the free rows instead of
    hiding its middle; seen in full, the prompt starts on Allow once."""
    project, start = tui_project
    command = "python3 -c \"" + "".join(f"x{i} = {i}; " for i in range(80)) + "print('END')\""
    session = start(
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": command}}]},
          {"text": "DONE_MARKER"}]]
    )
    _select_scripted_model(session)
    session.send_line("run it")

    modal = session.wait_for("Permission required")
    assert "characters not shown" not in modal, modal
    flat = "".join(line.strip().lstrip("┃").strip() for line in modal.splitlines())
    assert "x40 = 40;" in flat and "print('END')" in flat, modal
    assert "❯ Allow once" in modal
    session.send_key("Escape")
    session.wait_for("DONE_MARKER")


def test_permission_modal_for_a_command_too_long_to_show_starts_on_reject(tui_project) -> None:
    project, start = tui_project
    command = "echo " + "A" * 9000 + "; touch LONG.txt"
    session = start(
        [[{"tool_calls": [{"name": "bash", "arguments": {"command": command}}]},
          {"text": "DONE_MARKER"}]]
    )
    subprocess.run(["tmux", "resize-window", "-t", session.name, "-x", "80", "-y", "24"],
                   check=True)
    _select_scripted_model(session)
    session.send_line("run it")

    modal = session.wait_for("Permission required")
    # The kept text fills the free rows, so the marker can wrap.
    flat = "".join(line.strip().lstrip("┃").strip() for line in modal.splitlines())
    assert "characters not shown⟩" in flat, modal
    assert "touch LONG.txt" in modal
    assert "❯ Reject" in modal
    session.send_key("Enter")
    session.wait_for("DONE_MARKER")
    assert not (project / "LONG.txt").exists()


def test_interrupt_during_tool_stops_it_and_resolves_the_tool_row(tui_project) -> None:
    """Esc while a bash command runs stops the command at once, and its tool
    row resolves: no spinner survives the interrupted turn (regression,
    2026-07-06: the row kept spinning when its result frame arrived after the
    interrupt)."""
    project, start = tui_project
    session = start(
        [
            [
                {"tool_calls": [{"name": "bash", "arguments": {
                    "command": "sleep 6 && touch finished.txt"}}]},
                {"text": "NEVER_SHOWN_AFTER_INTERRUPT"},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("run a slow command")

    modal = session.wait_for("Permission required")
    assert "sleep 6" in modal
    session.send_key("Enter")  # Allow once — bash starts

    # The tool row is live (spinner) while sleep runs.
    session.wait_until(
        lambda frame: "Bash sleep 6" in frame
        and any(ch in frame for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        description="running bash row with spinner",
        timeout=10,
    )
    session.send_key("Escape")  # interrupt mid-execution

    # The command is stopped, so the turn ends at once.
    interrupted = session.wait_for("Interrupted.", timeout=4)
    resolved = session.wait_until(
        lambda frame: "Interrupted." in frame
        and not any(ch in frame for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        description="no spinner survives the interrupted turn",
        timeout=4,
    )
    assert "Bash sleep 6" in resolved
    assert "NEVER_SHOWN_AFTER_INTERRUPT" not in resolved
    assert "interrupted" in interrupted
    time.sleep(6)  # past the point where the command would have finished
    assert not (project / "finished.txt").exists()


def test_a_bad_setting_is_reported_when_the_session_starts(tui_project) -> None:
    project, start = tui_project
    config = project.parent / "home" / ".cortex" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("tools_max_iterations: 1001\n", encoding="utf-8")

    session = start([[{"text": "HI"}]])
    frame = session.wait_for("Session ready")

    assert "Ignored tools_max_iterations: 1001" in frame.replace("\n", " ").replace("┃", "")


def test_absolute_tool_paths_display_repo_relative(tui_project) -> None:
    """Models sometimes pass ABSOLUTE paths in tool arguments. Tool rows and
    the permission modal must DISPLAY them repo-relative when they are inside
    the repo (display only — the tool input itself is untouched: the edit
    still lands on the real file)."""
    project, start = tui_project
    abs_path = str((project / "calc.py").resolve())
    session = start(
        [
            [
                {
                    "tool_calls": [
                        {"name": "read_file", "arguments": {"path": abs_path}},
                        {
                            "name": "edit_file",
                            "arguments": {
                                "path": abs_path,
                                "old_text": "    return a + b",
                                "new_text": "    return a - b",
                            },
                        },
                    ]
                },
                {"text": "ABS_PATH_DONE"},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("edit it with absolute paths")

    # Permission modal: repo-relative summary, no absolute path anywhere, and
    # no redundant Target row (the normalized pattern repeats the summary).
    modal = session.wait_for("Permission required")
    assert "Edit file: calc.py" in modal
    assert "project/calc.py" not in modal
    assert "Target:" not in modal
    session.send_key("Enter")  # Allow once

    frame = session.wait_for("ABS_PATH_DONE")
    # Tool rows show the repo-relative form, not the truncated absolute tail.
    assert "Read calc.py" in frame
    assert "Update calc.py" in frame
    assert "project/calc.py" not in frame
    # The edit was applied to the real absolute-path file (inputs untouched).
    assert "return a - b" in (project / "calc.py").read_text(encoding="utf-8")


def test_reply_cut_off_at_the_output_limit_shows_the_limit_error(tui_project) -> None:
    from tests.chat_completions_server import ChatCompletionsServer

    server = ChatCompletionsServer([{"text": "The fix is to", "finish": "length"}])
    project, start = tui_project
    try:
        session = start(
            [[{"text": "IGNORED"}]],
            extra_env={
                "CORTEX_SCRIPTED_MODEL": "",
                "OPENAI_COMPATIBLE_BASE_URL": server.base_url,
                "OPENAI_COMPATIBLE_API_KEY": "sk-compatible-test",
            },
        )
        session.wait_for("Session ready")
        session.send_key("/model openai-compatible:test-model")
        time.sleep(0.5)
        session.send_key("Enter")
        session.wait_for("openai-compatible:test-model — now active.")
        session.send_key("Escape")
        time.sleep(0.5)

        session.send_line("fix the bug")
        frame = session.wait_for("reached the output limit (max_tokens=32768)", timeout=30)
        assert "The fix is to" in frame
        assert "Raise max_tokens" in frame
    finally:
        server.close()


def test_turn_at_its_request_limit_shows_the_limit_and_can_continue(tui_project) -> None:
    project, start = tui_project
    session = start(
        [
            [
                {"text": "Reading.",
                 "tool_calls": [{"name": "read_file", "arguments": {"path": "calc.py"}}]},
                {"tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}]},
                {"text": "NEVER_SHOWN_PAST_THE_LIMIT"},
            ],
            [{"text": "CONTINUED_MARKER"}],
        ],
        extra_env={"CORTEX_TOOLS_MAX_ITERATIONS": "2"},
    )
    _select_scripted_model(session)
    session.send_line("read everything")

    frame = session.wait_until(
        lambda frame: "tools_max_iterations=2" in frame
        and not any(ch in frame for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        description="limit notice with no spinner",
        timeout=30,
    )
    assert "reached its limit of model requests" in frame
    assert "Reading." in frame
    assert "Read calc.py" in frame
    assert "NEVER_SHOWN_PAST_THE_LIMIT" not in frame

    session.send_line("continue")
    session.wait_for("CONTINUED_MARKER", timeout=30)


def test_failed_request_shows_retry_and_recovers(tui_project) -> None:
    from tests.chat_completions_server import ChatCompletionsServer

    server = ChatCompletionsServer([{"status": 503, "retry_after": 3}, {"text": "RECOVERED"}])
    project, start = tui_project
    try:
        session = start(
            [[{"text": "IGNORED"}]],
            extra_env={
                "CORTEX_SCRIPTED_MODEL": "",
                "OPENAI_COMPATIBLE_BASE_URL": server.base_url,
                "OPENAI_COMPATIBLE_API_KEY": "sk-compatible-test",
            },
        )
        session.wait_for("Session ready")
        session.send_key("/model openai-compatible:test-model")
        time.sleep(0.5)
        session.send_key("Enter")
        session.wait_for("openai-compatible:test-model — now active.")
        session.send_key("Escape")
        time.sleep(0.5)

        session.send_line("hello")
        session.wait_for("retry", timeout=10)
        frame = session.wait_for("RECOVERED", timeout=20)
        assert "scripted 503" not in frame
        assert len(server.requests) == 2
    finally:
        server.close()


def test_escape_stops_a_turn_waiting_to_retry(tui_project) -> None:
    from tests.chat_completions_server import ChatCompletionsServer

    server = ChatCompletionsServer([{"status": 503, "retry_after": 30}, {"text": "LATE_ANSWER"}])
    project, start = tui_project
    try:
        session = start(
            [[{"text": "IGNORED"}]],
            extra_env={
                "CORTEX_SCRIPTED_MODEL": "",
                "OPENAI_COMPATIBLE_BASE_URL": server.base_url,
                "OPENAI_COMPATIBLE_API_KEY": "sk-compatible-test",
            },
        )
        session.wait_for("Session ready")
        session.send_key("/model openai-compatible:test-model")
        time.sleep(0.5)
        session.send_key("Enter")
        session.wait_for("openai-compatible:test-model — now active.")
        session.send_key("Escape")
        time.sleep(0.5)

        session.send_line("hello")
        session.wait_for("retry", timeout=10)
        session.send_key("Escape")
        frame = session.wait_for("Interrupted.", timeout=5)
        assert "LATE_ANSWER" not in frame
        assert len(server.requests) == 1
    finally:
        server.close()


def test_escape_stops_a_model_request_in_flight(tui_project) -> None:
    from tests.chat_completions_server import ChatCompletionsServer

    server = ChatCompletionsServer([{"text": "LATE_ANSWER", "delay": 60}, {"text": "NEXT_ANSWER"}])
    project, start = tui_project
    try:
        session = start(
            [[{"text": "IGNORED"}]],
            extra_env={
                "CORTEX_SCRIPTED_MODEL": "",
                "OPENAI_COMPATIBLE_BASE_URL": server.base_url,
                "OPENAI_COMPATIBLE_API_KEY": "sk-compatible-test",
            },
        )
        session.wait_for("Session ready")
        session.send_key("/model openai-compatible:test-model")
        time.sleep(0.5)
        session.send_key("Enter")
        session.wait_for("openai-compatible:test-model — now active.")
        session.send_key("Escape")
        time.sleep(0.5)

        session.send_line("think hard")
        session.wait_until(
            lambda frame: any(ch in frame for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
            description="spinner while the model works",
            timeout=10,
        )
        time.sleep(1)
        session.send_key("Escape")
        frame = session.wait_until(
            lambda frame: "Interrupted." in frame
            and not any(ch in frame for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
            description="interrupted turn with no spinner",
            timeout=5,
        )
        assert "LATE_ANSWER" not in frame

        session.send_line("next")
        session.wait_for("NEXT_ANSWER", timeout=20)
        assert len(server.requests) == 2
    finally:
        server.close()


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


def test_update_notice_status_panel_and_live_update_narration(tui_project, tmp_path) -> None:
    """With the release probe stubbed to a local server: the daily update
    check posts its notice exactly once at session start, /update renders the
    installed-vs-latest status panel, and /update cortex narrates LIVE — the
    indicator must carry the latest installer output line, not sit static —
    then resolves in place with the exact final message."""
    from cortex.update_check import installed_cortex_version
    from tests.test_worker_update_flow import _cortex_release_assets, _probe_server

    installed = installed_cortex_version()
    stub_pip = tmp_path / "slow-stub-pip"
    stub_pip.write_text(
        "#!/usr/bin/env bash\n"
        'echo "Installing cortex-llm 9.9.9..."\n'
        "sleep 2.5\n"
        'echo "Successfully installed cortex-llm"\n',
        encoding="utf-8",
    )
    stub_pip.chmod(0o755)

    project, start = tui_project
    _wheel_name, assets = _cortex_release_assets("v9.9.9", b"stub cortex release wheel")
    with _probe_server(cortex_tag="v9.9.9", assets=assets) as probe_base:
        session = start(
            [[{"text": "IGNORED"}]],
            extra_env={
                "CORTEX_UPDATE_PROBE_BASE": probe_base,
                "CORTEX_SELF_PIP": str(stub_pip),
                "CORTEX_SELF_INSTALL_KIND": "installed",
                "CORTEX_AUTO_UPDATE_CHECK": "true",  # opt back in (stubbed probe)
            },
        )
        session.wait_for("Session ready")

        # The startup notice renders in the transcript — exactly once.
        frame = session.wait_for("Cortex 9.9.9 available — update with /update cortex", timeout=30)
        assert frame.count("Cortex 9.9.9 available") == 1

        # /update from the palette: its argument is optional, so one Enter
        # runs the bare status report.
        session.send_key("/update")
        time.sleep(1)
        session.send_key("Enter")
        panel = session.wait_for(
            f"Cortex: {installed} installed · 9.9.9 available — /update cortex", timeout=20
        )
        assert "✓ /update" in panel
        assert panel.count("Cortex 9.9.9 available") == 1  # still exactly one notice

        # Esc dismisses the panel cleanly.
        session.send_key("Escape")
        time.sleep(1)
        assert "✓ /update" not in session.capture()

        # /update cortex: the live indicator must show the LATEST installer
        # output line as its phase detail (the 2.5s stub sleep keeps
        # "Installing cortex-llm 9.9.9..." on screen), with a spinner.
        session.send_key("/update cortex")
        time.sleep(1)
        session.send_key("Enter")
        live = session.wait_until(
            lambda f: any(
                "Updating Cortex…" in line
                and "Installing cortex-llm 9.9.9" in line
                and any(ch in line for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                for line in f.splitlines()
            ),
            description="live update indicator with installer phase detail",
            timeout=20,
        )
        assert "Working…" not in live  # background op: never the turn spinner

        # Resolves in place: final message, no lingering live indicator.
        final = "Cortex 9.9.9 installed — restart Cortex to apply."
        session.wait_for(final, timeout=30)
        session.wait_until(
            lambda f: "Updating Cortex…" not in f and final in f,
            description="update indicator resolved into the final message",
            timeout=10,
        )


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


def test_help_with_a_long_transcript_keeps_the_prompt_and_every_row_readable(
    tui_project,
) -> None:
    """A rejected call's line reads as a sentence, and /help's panel never
    pushes the prompt off screen or squeezes transcript rows into each other."""
    project, start = tui_project
    session = start(
        [
            [
                {"tool_calls": [{"name": "bash", "arguments": {"command": "python3 -c 'print(1)'"}}]},
                {"text": "The command was rejected, so I did not run it."},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("run a check")
    session.wait_for("Permission required")
    session.send_key("Escape")  # Esc rejects
    frame = session.wait_for("so I did not run it.")
    assert "Rejected at the permission prompt: the command was not carried out." in frame
    assert "patterns=" not in frame

    subprocess.run(["tmux", "resize-window", "-t", session.name, "-x", "120", "-y", "40"],
                   check=True)
    session.send_line("/help")
    session.wait_for("Esc to dismiss", timeout=15)
    time.sleep(1.5)  # let the panel settle; the bug was a steady state
    frame = session.capture()
    assert "Ctrl+C quit" in frame, frame  # the footer under the prompt
    assert "Ask Cortex" in frame, frame
    # No transcript row drawn over another (a tool line and its result).
    assert not [line for line in frame.splitlines() if "◆" in line and "⎿" in line], frame
    # A skill shows one sentence, not the paragraph written for the model.
    assert "Not for only explaining an error" not in frame, frame

    # On a small window the rows that do not fit are counted, with the way to
    # see them, and the prompt stays.
    subprocess.run(["tmux", "resize-window", "-t", session.name, "-x", "80", "-y", "24"],
                   check=True)
    small = session.wait_for("make the window taller to see them", timeout=10)
    assert "Ask Cortex" in small, small


def test_a_new_message_brings_the_transcript_back_to_the_bottom(tui_project) -> None:
    """After scrolling up to reread, sending a message shows that turn: its
    tool calls and the permission prompt it raises, not the old text."""
    project, start = tui_project
    long_answer = "\n\n".join(f"OLD_LINE_{i}" for i in range(80))
    session = start(
        [
            [{"text": long_answer}],
            [{"tool_calls": [{"name": "bash", "arguments": {"command": "echo NEW_TURN_RAN"}}]},
             {"text": "NEW_TURN_DONE"}],
        ]
    )
    _select_scripted_model(session)
    session.send_line("first question")
    session.wait_for("OLD_LINE_79")
    for _ in range(4):
        session.send_key("PPage")
        time.sleep(0.3)
    session.wait_until(lambda frame: "OLD_LINE_79" not in frame,
                       description="the transcript scrolled up", timeout=10)

    session.send_line("SECOND_QUESTION_MARKER")
    frame = session.wait_for("Permission required", timeout=20)
    assert "SECOND_QUESTION_MARKER" in frame, frame
    assert "OLD_LINE_40" not in frame, frame
    session.send_key("Enter")
    session.wait_for("NEW_TURN_DONE", timeout=20)
