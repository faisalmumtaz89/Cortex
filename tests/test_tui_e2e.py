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
    ):
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
        env.update(extra_env or {})
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

    def start(responses, *, extra_env: dict | None = None) -> TuiSession:
        script = tmp_path / "script.json"
        script.write_text(json.dumps({"responses": responses}), encoding="utf-8")
        session = TuiSession(
            project_dir=project, home_dir=home, script_path=script, extra_env=extra_env
        )
        sessions.append(session)
        return session

    yield project, start

    for session in sessions:
        session.close()


def _cloud_picker_rows() -> list[str]:
    """The Cloud tab's rows exactly as the worker builds them: every catalog
    entry in catalog order (the harness's isolated HOME has no override file).
    Deriving rows from the real catalog keeps the picker test decoupled from
    catalog cardinality — a routine model-list edit cannot silently break it."""
    from cortex.cloud.catalog import CloudModelCatalog

    catalog = CloudModelCatalog(override_path=Path("/nonexistent-cloud-override.json"))
    return [ref.selector for ref in catalog.list_models()]


def _selection_window_size() -> int:
    """The shared picker window (rows shown before '+N more'), read from the
    component's own default so the test tracks the source of truth."""
    source = (REPO_ROOT / "frontend/cortex-tui/src/components/selection_list.tsx").read_text(
        encoding="utf-8"
    )
    match = re.search(r"maxVisibleRows \?\? (\d+)", source)
    assert match, "selection_list.tsx window default not found"
    return int(match.group(1))


def _fake_lumen_env(tmp_path: Path) -> dict:
    """Deterministic local catalog for picker tests: one cached model (with a
    size), two available-to-download — no host lumen cache or network."""
    listing = (
        "Cached models:\n\n"
        "  qwen3-5-9b-Q4_0                          5.4 GB\n\n"
        "Available to download:\n"
        "  qwen3-5-9b           Qwen3.5 9B Q8_0\n"
        "  qwen3-6-27b          Qwen3.6 27B Q4_0\n\n"
        "Download with: lumen pull <model-name> [--quant Q8_0]\n"
    )
    fixture = tmp_path / "tui-lumen-models.txt"
    fixture.write_text(listing, encoding="utf-8")
    stub = tmp_path / "tui-fake-lumen"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1:2] == ['models']:\n"
        f"    print(open({str(fixture)!r}).read(), end='')\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return {"CORTEX_LUMEN_BINARY": str(stub)}


def _full_fake_lumen_env(tmp_path: Path, *, boot_delay: float = 4.0) -> dict:
    """A pull-capable fake `lumen` CLI + a fake `lumen-server` that becomes
    ready after `boot_delay` seconds — enough to observe the load indicator.
    `pull` streams output lines while growing a real `.part` file inside a
    scratch LUMEN_CACHE_DIR (exercising Cortex's byte polling), then rewrites
    the listing so the selector resolves as cached for the chained auto-load."""
    cache_dir = tmp_path / "lumen-cache"
    cache_dir.mkdir(exist_ok=True)
    listing_start = (
        "Cached models:\n\n"
        "  qwen3-5-9b-Q4_0                          5.4 GB\n\n"
        "Available to download:\n"
        "  qwen3-5-9b           Qwen3.5 9B Q8_0\n\n"
        "Download with: lumen pull <model-name> [--quant Q8_0]\n"
    )
    listing_after = (
        "Cached models:\n\n"
        "  qwen3-5-9b-Q4_0                          5.4 GB\n"
        "  qwen3-5-9b-Q8_0                          8.9 GB\n\n"
        "Available to download:\n\n"
        "Download with: lumen pull <model-name> [--quant Q8_0]\n"
    )
    fixture = tmp_path / "flow-lumen-models.txt"
    fixture.write_text(listing_start, encoding="utf-8")
    after_fixture = tmp_path / "flow-lumen-models-after.txt"
    after_fixture.write_text(listing_after, encoding="utf-8")

    cli = tmp_path / "flow-fake-lumen"
    cli.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys, time\n"
        "if sys.argv[1:2] == ['models']:\n"
        f"    print(open({str(fixture)!r}).read(), end='')\n"
        "elif sys.argv[1:2] == ['pull']:\n"
        "    print('Downloading: https://huggingface.co/example.gguf', flush=True)\n"
        f"    part = {str(cache_dir)!r} + '/example.gguf.part'\n"
        "    with open(part, 'wb') as fh:\n"
        "        for _ in range(3):\n"
        "            fh.write(b'x' * 1024 * 1024)\n"
        "            fh.flush()\n"
        "            time.sleep(1.1)\n"
        "    import os\n"
        "    os.remove(part)\n"
        "    print('Saved: example.gguf (SHA-256: abc)', flush=True)\n"
        f"    shutil.copyfile({str(after_fixture)!r}, {str(fixture)!r})\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)

    server = tmp_path / "flow-fake-lumen-server"
    server.write_text(
        "#!/usr/bin/env python3\n"
        "import http.server, json, sys, time\n"
        "args = sys.argv[1:]\n"
        "port = int(args[args.index('--port') + 1])\n"
        "model = args[args.index('--model') + 1]\n"
        f"time.sleep({boot_delay})\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        body = json.dumps({'object': 'list', 'data': [{'id': model}]}).encode()\n"
        "        self.send_response(200 if self.path == '/v1/models' else 404)\n"
        "        self.send_header('Content-Length', str(len(body)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(body)\n"
        "    def log_message(self, *a):\n"
        "        pass\n"
        "http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()\n",
        encoding="utf-8",
    )
    server.chmod(0o755)

    return {
        "CORTEX_LUMEN_BINARY": str(cli),
        "CORTEX_LUMEN_SERVER_BINARY": str(server),
        "LUMEN_CACHE_DIR": str(cache_dir),
    }


def _select_scripted_model(session: TuiSession) -> None:
    # "Session ready" is the bootstrap-complete signal (worker handshake done).
    session.wait_for("Session ready")
    session.send_line("/model azure:scripted")
    session.wait_for("cloud · azure:scripted — now active.")


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


def test_cached_model_load_shows_load_indicator_not_download_bar(tui_project, tmp_path) -> None:
    """Scenario A + C + G: selecting a CACHED model shows the GPU-load
    indicator (spinner + 'Loading … into GPU memory'), never download
    artifacts and never the generic turn spinner; re-selecting the serving
    model answers instantly with a non-empty 'already active' panel."""
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]], extra_env=_full_fake_lumen_env(tmp_path))
    session.wait_for("Session ready")

    session.send_line("/model qwen3-5-9b:q4_0")
    loading = session.wait_until(
        lambda frame: "Loading qwen3-5-9b:q4_0…" in frame,
        description="minimal load indicator visible during the boot",
        timeout=15,
    )
    # Minimal-line spec: spinner + "Loading <selector>…" and NOTHING else.
    assert "into GPU memory" not in loading
    assert "large models" not in loading
    assert not re.search(r"Loading qwen3-5-9b:q4_0…[^\n]*\d+s", loading), "no elapsed timer"
    # Wrong-artifact guards: no download language, no bytes, no turn spinner.
    assert "Downloading" not in loading
    assert "downloaded" not in loading
    assert "0 B" not in loading
    assert "Working…" not in loading
    assert "Esc to interrupt" not in loading

    # Exactly ONE live indicator row (spinner-prefixed) — and exactly ONE
    # "Loading …" line in the WHOLE frame: the /model result panel must stay a
    # terse selection confirmation, never a second live-state mirror (the
    # cold-boot double-"Loading" regression).
    spinner_rows = [
        line
        for line in loading.splitlines()
        if "Loading qwen3-5-9b:q4_0…" in line and any(ch in line for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    ]
    assert len(spinner_rows) == 1, spinner_rows
    loading_rows = [line for line in loading.splitlines() if "Loading qwen3-5-9b:q4_0…" in line]
    assert loading_rows == spinner_rows, (
        f"transcript row must solely own live load state, saw: {loading_rows}"
    )
    # The result panel confirms the selection tersely (no "Loading" mirror).
    panel = session.wait_until(
        lambda frame: "local · qwen3-5-9b:q4_0 selected." in frame,
        description="terse selection confirmation in the result panel",
        timeout=10,
    )
    assert len([line for line in panel.splitlines() if "Loading qwen3-5-9b:q4_0…" in line]) <= 1

    session.wait_for("local · qwen3-5-9b:q4_0 ready — now active.", timeout=30)
    session.send_key("Escape")  # dismiss the /model result panel
    # RESOLVE IN PLACE: the live indicator row must be GONE — one operation,
    # one transcript message (this is the stale-"Loading… 58s"-row regression).
    resolved = session.wait_until(
        lambda frame: "Loading qwen3-5-9b:q4_0…" not in frame
        and "local · qwen3-5-9b:q4_0 ready — now active." in frame,
        description="loading indicator resolved into the ready line",
        timeout=10,
    )
    assert resolved.count("ready — now active.") == 1, "exactly one resolved row (no duplicate notice)"

    # Scenario C: re-selecting the serving model is instant and explicit.
    session.send_line("/model qwen3-5-9b:q4_0")
    result = session.wait_for("local · qwen3-5-9b:q4_0 is already active.", timeout=15)
    assert "✓ /model qwen3-5-9b:q4_0" in result
    session.send_key("Escape")


def test_uncached_select_shows_download_then_load_transition(tui_project, tmp_path) -> None:
    """Scenario B/D: an uncached select streams REAL transferred bytes during
    the pull, then flips the SAME message to the GPU-load indicator — never
    '0 B downloaded', never a byte bar during the load phase."""
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]], extra_env=_full_fake_lumen_env(tmp_path, boot_delay=2.5))
    session.wait_for("Session ready")

    session.send_line("/model qwen3-5-9b:q8_0")
    session.wait_for("Downloading qwen3-5-9b:q8_0 — loads automatically when done", timeout=15)

    # Download stage: the indicator carries real cache-side bytes.
    downloading = session.wait_until(
        lambda frame: "Downloading qwen3-5-9b:q8_0" in frame and "MB" in frame,
        description="download indicator with live byte count",
        timeout=20,
    )
    assert "0 B downloaded" not in downloading
    assert "Working…" not in downloading

    # Phase transition: same operation, now the minimal load indicator.
    load_stage = session.wait_until(
        lambda frame: "Loading qwen3-5-9b:q8_0…" in frame,
        description="transition to the load indicator after the pull",
        timeout=25,
    )
    assert "0 B" not in load_stage
    assert "into GPU memory" not in load_stage
    assert "large models" not in load_stage

    session.wait_for("local · qwen3-5-9b:q8_0 ready — now active.", timeout=30)
    # Resolve in place: no lingering live indicator after completion.
    session.wait_until(
        lambda frame: "Loading qwen3-5-9b:q8_0…" not in frame
        and "Downloading qwen3-5-9b:q8_0…" not in frame,
        description="indicators resolved after download+load completes",
        timeout=10,
    )


def test_model_picker_tabs_split_local_and_cloud(tui_project, tmp_path) -> None:
    """The picker shows ONE origin at a time: a Local tab (downloaded models
    with sizes first, then download candidates) and a Cloud tab; Tab/←→
    switch; it opens on the active backend's tab; no numbered selection."""
    project, start = tui_project
    session = start([[{"text": "IGNORED"}]], extra_env=_fake_lumen_env(tmp_path))
    session.wait_for("Session ready")

    # Palette Tab still completes commands (picker tab-switching must not
    # steal it): "/mod" + Tab → arg-hint for /model.
    session.send_key("/mod")
    time.sleep(0.8)
    session.send_key("Tab")
    session.wait_for("<name | provider:model>", timeout=10)
    session.send_key("Escape")
    time.sleep(0.8)

    # Rows and windowing are DERIVED (real catalog + the picker's own window
    # default) so a routine catalog edit cannot silently break this test.
    cloud_rows = _cloud_picker_rows()
    window = _selection_window_size()
    hidden = max(0, len(cloud_rows) - window)
    first_row, last_row = cloud_rows[0], cloud_rows[-1]
    # The wrap-select leg needs the env-authenticated azure deployment as the
    # catalog's final row; if the order ever changes, update that leg too.
    assert last_row.startswith("azure:"), f"catalog order changed: {cloud_rows}"

    # Bare /model opens the picker. No model is active → Local tab first.
    session.send_key("/model")
    time.sleep(1)
    session.send_key("Enter")
    picker = session.wait_for("Select a model", timeout=10)
    assert "Local" in picker and "Cloud" in picker  # tab bar
    assert "qwen3-5-9b:q4_0" in picker  # downloaded row…
    assert "5.4 GB" in picker  # …with its on-disk size
    assert "select to download" in picker  # download candidates below
    assert first_row not in picker  # cloud rows live on the OTHER tab
    assert "Tab cloud" in picker  # footer names the other tab
    # No numbered rows anywhere.
    assert "[1]" not in picker and "- [1]" not in picker

    # Tab → Cloud tab: cloud rows appear, local rows disappear; rows past the
    # window hide behind the "+N more" hint until selection reaches them.
    session.send_key("Tab")
    cloud_tab = session.wait_for(first_row, timeout=10)
    assert "qwen3-5-9b:q4_0" not in cloud_tab
    for visible_row in cloud_rows[:window]:
        assert visible_row in cloud_tab, f"windowed row missing: {visible_row}"
    for windowed_out in cloud_rows[window:]:
        assert windowed_out not in cloud_tab, f"row should be windowed out: {windowed_out}"
    if hidden:
        assert f"+{hidden} more" in cloud_tab
    assert "Tab local" in cloud_tab
    assert "ready" in cloud_tab or "login required" in cloud_tab

    # Left arrow flips back to Local; Right returns to Cloud.
    session.send_key("Left")
    session.wait_until(
        lambda frame: "qwen3-5-9b:q4_0" in frame and first_row not in frame,
        description="left arrow returns to the Local tab",
        timeout=10,
    )
    session.send_key("Right")
    session.wait_for(first_row, timeout=10)

    # Up from the top wraps to the LAST cloud entry — the windowing follows
    # the selection, revealing it; select it via Enter (azure is the one
    # env-authenticated provider in this harness).
    session.send_key("Up")
    session.wait_for(last_row, timeout=10)
    session.send_key("Enter")
    result = session.wait_for(f"cloud · {last_row} — now active.", timeout=10)
    assert "✓" in result
    session.send_key("Escape")
    time.sleep(1)
    assert "Select a model" not in session.capture()

    # Reopen: the active backend is cloud, so the picker opens on Cloud with
    # the window scrolled to the active (last) row.
    session.send_key("/model")
    time.sleep(1)
    session.send_key("Enter")
    reopened = session.wait_for("Select a model", timeout=10)
    assert last_row in reopened
    assert "qwen3-5-9b:q4_0" not in reopened
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

    # The launch command IS the pane: when python -m cortex exits, the tmux
    # session dies and capture() returns "". Old behavior: the pane lingered
    # indefinitely. Give the teardown a generous-but-bounded window.
    deadline = time.time() + 8
    alive = True
    while time.time() < deadline:
        if session.capture() == "":
            alive = False
            break
        time.sleep(0.25)
    assert not alive, "TUI still running 8s after a single Ctrl+C"


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
    assert "logged in" in picker or "not configured" in picker
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


def test_interrupt_during_tool_resolves_the_tool_row(tui_project) -> None:
    """Regression (2026-07-06 screenshot): Esc while a bash tool was executing
    finalized the turn as interrupted but left the tool row spinning forever —
    the tool's result frame arrived after the interrupt flag and was
    swallowed. The row must resolve to a terminal state (here: the tool
    actually completes, so ◆ completed), and no spinner may survive an
    interrupted turn."""
    project, start = tui_project
    session = start(
        [
            [
                {"tool_calls": [{"name": "bash", "arguments": {"command": "sleep 4 && echo done"}}]},
                {"text": "NEVER_SHOWN_AFTER_INTERRUPT"},
            ]
        ]
    )
    _select_scripted_model(session)
    session.send_line("run a slow command")

    modal = session.wait_for("Permission required")
    assert "sleep 4" in modal
    session.send_key("Enter")  # Allow once — bash starts, 4s window

    # The tool row is live (spinner) while sleep runs.
    session.wait_until(
        lambda frame: "Bash sleep 4" in frame
        and any(ch in frame for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        description="running bash row with spinner",
        timeout=10,
    )
    session.send_key("Escape")  # interrupt mid-execution

    # The turn unwinds once bash finishes: interrupted notice + resolved row.
    interrupted = session.wait_for("Interrupted.", timeout=20)
    resolved = session.wait_until(
        lambda frame: "Interrupted." in frame
        and not any(ch in frame for ch in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        description="no spinner survives the interrupted turn",
        timeout=15,
    )
    # The bash row resolved (the tool genuinely ran to completion — its result
    # frame must not be swallowed by the interrupt unwind).
    assert "Bash sleep 4" in resolved
    assert "NEVER_SHOWN_AFTER_INTERRUPT" not in resolved
    assert "interrupted" in interrupted


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
