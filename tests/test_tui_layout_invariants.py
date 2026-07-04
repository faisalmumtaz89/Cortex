from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cortex.ui_runtime import launcher


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def _try_decode_utf8(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def test_session_layout_uses_single_flex_scroll_transcript() -> None:
    source = _read("frontend/cortex-tui/src/routes/session.tsx")

    assert "maxTranscriptHeight" not in source
    assert "useScrollableTranscript" not in source
    assert "estimatedTranscriptLines" not in source
    assert 'stickyStart="bottom"' in source
    assert "stickyScroll" in source
    assert "ref={(r: ScrollBoxRenderable)" in source
    assert "flexGrow={1}" in source
    assert "<PromptPanel" in source
    assert "scrollAcceleration={scrollAcceleration()}" in source
    assert "class CustomSpeedScroll implements ScrollAcceleration" in source
    assert "new MacOSScrollAccel()" in source
    assert "paddingTop={1}" in source
    assert "store.state.notices" not in source
    assert "colorForSessionStatus" in source


def test_prompt_panel_contract_is_bottom_anchor_stable() -> None:
    source = _read("frontend/cortex-tui/src/components/prompt_panel.tsx")

    assert (
        'placeholder={props.hasPendingPermission ? "Permission modal active..." : "Ask Cortex..."}'
        in source
    )
    assert "minHeight={1}" in source
    assert "maxHeight={6}" in source
    # Hint bar is adaptive (sheds segments to never overlap the status at narrow
    # widths) rather than a single fixed string.
    assert '{ text: "Enter send"' in source
    assert '{ text: "Ctrl+C quit"' in source
    assert "buildHint(" in source
    assert 'vertical: "┃"' in source
    # Input band: symmetric padding (pad / text / pad) centers the cursor; the
    # dark-gray background provides the edges — no ▀ floor row.
    assert "PROMPT_TOP_SPACER_ROWS = 1" in source
    assert "PROMPT_BOTTOM_SPACER_ROWS = 1" in source
    assert "▀" not in source


def test_index_startup_uses_full_screen_alt_buffer() -> None:
    source = _read("frontend/cortex-tui/src/index.tsx")

    assert "useAlternateScreen: true" in source
    assert "experimental_splitHeight" not in source
    assert "queryCursorRow" not in source
    assert "computeSplitHeight" not in source


def test_model_hint_copy_uses_ascii_safe_separators() -> None:
    source = _read("frontend/cortex-tui/src/routes/session.tsx")

    # Single-line header: branch + path left, model · status right.
    assert "readGitBranch(" in source
    assert "headerModel()" in source
    assert "Type: setup | model | download" not in source
    assert "Type: setup · model · download" not in source


def test_app_disables_stdout_interception_without_startup_clear_hacks() -> None:
    source = _read("frontend/cortex-tui/src/app.tsx")

    assert "renderer.disableStdoutInterception()" in source
    assert "CORTEX_TUI_CLAIM_SCREEN" not in source


def test_launcher_no_longer_disables_alt_screen() -> None:
    launcher_source = _read("cortex/ui_runtime/launcher.py")
    wrapper_path = _repo_root() / "cortex/ui_runtime/bin/cortex-tui"

    assert "OTUI_USE_ALTERNATE_SCREEN" not in launcher_source
    assert "CORTEX_TUI_ALT_SCREEN" not in launcher_source
    if wrapper_path.exists():
        wrapper_text = _try_decode_utf8(wrapper_path.read_bytes())
    else:
        wrapper_text = None

    if wrapper_text is not None:
        assert "OTUI_USE_ALTERNATE_SCREEN" not in wrapper_text
        assert "CORTEX_TUI_ALT_SCREEN" not in wrapper_text


def test_role_specific_message_components_exist_and_generic_renderer_removed() -> None:
    repo = _repo_root()

    assert (repo / "frontend/cortex-tui/src/components/messages/user_message.tsx").exists()
    assert (repo / "frontend/cortex-tui/src/components/messages/assistant_message.tsx").exists()
    assert (repo / "frontend/cortex-tui/src/components/messages/system_message.tsx").exists()
    assert not (repo / "frontend/cortex-tui/src/components/message_text.tsx").exists()


def test_message_components_render_panel_metadata_rows() -> None:
    user_source = _read("frontend/cortex-tui/src/components/messages/user_message.tsx")
    assistant_source = _read("frontend/cortex-tui/src/components/messages/assistant_message.tsx")
    system_source = _read("frontend/cortex-tui/src/components/messages/system_message.tsx")

    # User turn: a single accent "spine" (┃) + subtle panel fill, no role label
    # or timestamp (opencode-style cleanliness). The metadata footer on the
    # assistant identifies turns instead.
    assert 'vertical: "┃"' in user_source
    assert "backgroundColor={UI_PALETTE.panel}" in user_source
    assert "▣ " in assistant_source
    assert "modeLabel()" in assistant_source
    assert "modelLabel()" in assistant_source
    assert "formatDuration(props.message.elapsedMs)" in assistant_source
    # Live/resolved gating MUST be reactive accessors: plain component-body
    # consts run once in Solid and froze the indicator forever (the stale
    # "Loading… 58s" row that outlived its own completion).
    assert "const showLiveProgress = () =>" in system_source
    assert "const progress = () =>" in system_source
    # Minimal one-line indicators: spinner + "Loading X…" — no GPU verbiage,
    # no duration coaching, no elapsed timers; downloads show bytes only.
    assert "Loading ${progress.repoID}…" in system_source
    assert "into GPU memory" not in system_source
    assert "large models" not in system_source
    assert "elapsedSeconds" not in system_source
    assert "Downloading ${progress.repoID}" in system_source
    assert 'progress()?.kind === "model-load"' in system_source
    assert 'progress()?.kind === "download"' in system_source
    # The fake-fullness artifacts are gone: no empty progress bar, no
    # hardcoded "N downloaded" byte line.
    assert "progressBar(" not in system_source
    assert "downloaded`" not in system_source


def test_store_merge_logic_protects_streamed_content_and_dedupes_per_message() -> None:
    source = _read("frontend/cortex-tui/src/context/store.tsx")

    assert "const dedupeKey = `${messageID}:${callKey}:${status}:${statusHash}`" in source
    assert "if (!incomingFinal && nextContent.length < current.length)" in source
    assert "current.length >= text.length ? current : text" in source
    assert "created_ts_ms" in source
    assert "completed_ts_ms" in source
    assert "elapsed_ms" in source
    assert 'setState("notices"' not in source


def test_frontend_colors_are_centralized_in_shared_palette_module() -> None:
    repo = _repo_root()
    src_root = repo / "frontend/cortex-tui/src"
    palette_file = src_root / "components/ui_palette.ts"

    assert palette_file.exists()

    pattern = re.compile(
        r'(?:fg|borderColor|backgroundColor|textColor|focusedTextColor|cursorColor|focusedBackgroundColor)\s*=\s*"[^"]+"'
        r"|RGBA\.fromHex\(\"#[0-9A-Fa-f]{3,8}\"\)"
        r'|return\s+"(?:black|red|green|yellow|blue|magenta|cyan|white|gray|grey)"'
    )

    offenders: list[str] = []
    for path in src_root.rglob("*"):
        if path.suffix not in {".ts", ".tsx"}:
            continue
        if path.resolve() == palette_file.resolve():
            continue
        source = path.read_text(encoding="utf-8")
        if pattern.search(source):
            offenders.append(str(path.relative_to(repo)))

    assert offenders == [], f"raw color literals found outside shared palette: {offenders}"


@pytest.mark.skipif(shutil.which("expect") is None, reason="expect not available")
def test_tui_core_chat_surface_smoke_runs_under_pty() -> None:
    repo = _repo_root()
    python_candidates = [
        repo / "venv" / "bin" / "python",
        repo / ".venv" / "bin" / "python",
    ]
    python_bin = next((candidate for candidate in python_candidates if candidate.exists()), None)
    python_cmd = str(python_bin) if python_bin is not None else sys.executable

    if launcher._candidate_command() is None:  # type: ignore[attr-defined]
        completed = subprocess.run(
            [python_cmd, "-m", "cortex"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=45,
            env=dict(os.environ),
        )
        transcript = f"{completed.stdout}\n{completed.stderr}"
        assert completed.returncode == 127
        assert "OpenTUI sidecar not available in this environment." in transcript
        return

    expect_script = f"""
set timeout 45
spawn {python_cmd} -m cortex
after 3500
send "/help\r"
after 1500
send "\003"
expect eof
"""
    completed = subprocess.run(
        ["expect", "-c", expect_script],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=90,
        env=dict(os.environ),
    )

    transcript = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0
    assert "Traceback" not in transcript


def test_model_picker_tabs_and_origin_labels() -> None:
    """Picker separates origins into Local/Cloud TABS (one origin visible at a
    time); Tab and arrow keys switch; the session header and turn footer keep
    the origin wording."""
    session = _read("frontend/cortex-tui/src/routes/session.tsx")
    assert 'labels: ["Local", "Cloud"]' in session
    assert "modelPickerTab" in session
    assert "switchPickerTab" in session
    # Downloaded local rows carry their on-disk size next to the name.
    assert "entry.size" in session
    # Opening tab follows the active backend ("local · x" / "cloud · y").
    assert "store.state.activeBackend" in session
    assert "stepPickerIndex" in session  # Up/Down skip the divider row
    # The old mixed-list section headers are gone.
    assert "Local — downloaded" not in session
    assert "Local — available to download" not in session

    selection = _read("frontend/cortex-tui/src/components/selection_list.tsx")
    assert "SelectionTabs" in selection  # tab bar rendered by the shared list
    assert "isHeader" in selection  # divider rows stay non-selectable

    footer = _read("frontend/cortex-tui/src/components/messages/assistant_message.tsx")
    assert "props.message.backend" in footer
