from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


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

    assert "placeholder={props.hasPendingPermission ? \"Permission modal active...\" : \"Ask Cortex...\"}" in source
    assert "minHeight={1}" in source
    assert "maxHeight={6}" in source
    assert "Enter submit · Esc reject permission" in source
    assert "vertical: \"┃\"" in source
    assert "bottomLeft: \"╹\"" in source


def test_index_startup_uses_full_screen_alt_buffer() -> None:
    source = _read("frontend/cortex-tui/src/index.tsx")

    assert "useAlternateScreen: true" in source
    assert "experimental_splitHeight" not in source
    assert "queryCursorRow" not in source
    assert "computeSplitHeight" not in source


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

    assert "formatTimestamp(props.message.createdTsMs)" in user_source
    assert "backgroundColor={UI_PALETTE.panel}" in user_source
    assert "▣ " in assistant_source
    assert "modeLabel()" in assistant_source
    assert "modelLabel()" in assistant_source
    assert "formatDuration(props.message.elapsedMs)" in assistant_source


def test_store_merge_logic_protects_streamed_content_and_dedupes_per_message() -> None:
    source = _read("frontend/cortex-tui/src/context/store.tsx")

    assert "const dedupeKey = `${messageID}:${callKey}:${status}:${statusHash}`" in source
    assert "if (!incomingFinal && nextContent.length < current.length)" in source
    assert "current.length >= text.length ? current : text" in source
    assert "created_ts_ms" in source
    assert "completed_ts_ms" in source
    assert "elapsed_ms" in source
    assert "setState(\"notices\"" not in source


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
    expect_script = r"""
set timeout 45
spawn ./venv/bin/python -m cortex
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
    assert "status:" in transcript
    assert "Ask Cortex..." in transcript
