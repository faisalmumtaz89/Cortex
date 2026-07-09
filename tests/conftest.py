"""Global test hermeticity.

Every in-process test gets a scratch permission store. The production default
is the developer's real ~/.cortex/tool_permissions.yaml, whose persisted rules
(e.g. a user-level `bash * allow`) would otherwise leak into any test that
builds a PermissionManager/PermissionStore without an explicit path and
silently flip prompt/deny assertions. Subprocess-driven tests (worker stdio,
TUI e2e) are already hermetic through their own isolated HOME.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import cortex.tooling.permissions as permissions

# The worker's daily update check (auto_update_check, default ON) probes
# GitHub for releases. The suite must make ZERO real network calls, so the
# check is force-disabled here for this process AND every worker subprocess
# (they inherit os.environ). Update-flow tests opt back in per-subprocess by
# setting CORTEX_AUTO_UPDATE_CHECK=true in their own env, always alongside a
# CORTEX_UPDATE_PROBE_BASE that points at a local stub server.
os.environ["CORTEX_AUTO_UPDATE_CHECK"] = "false"


@pytest.fixture(autouse=True)
def _hermetic_permission_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        permissions,
        "default_store_path",
        lambda: tmp_path / "tool_permissions.yaml",
    )
