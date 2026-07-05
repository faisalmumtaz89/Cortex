"""Frontend unit tests (bun test) as part of the pytest gate.

The TUI's pure logic modules (frontend/cortex-tui/tests/*.test.ts) are unit-
tested with bun's built-in runner — no extra dependencies. This wrapper makes
them a first-class gate alongside the Python suite; it skips only where bun
itself is unavailable (mirroring the tmux/sidecar skips of the TUI e2e).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "cortex-tui"

pytestmark = pytest.mark.skipif(shutil.which("bun") is None, reason="bun not installed")


def test_frontend_unit_suite_passes() -> None:
    completed = subprocess.run(
        ["bun", "test"],
        cwd=FRONTEND_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    transcript = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, f"bun test failed:\n{transcript}"
    assert " 0 fail" in transcript
