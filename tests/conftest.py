"""Global test hermeticity.

Every in-process test gets a scratch permission store. The production default
is the developer's real ~/.cortex/tool_permissions.yaml, whose persisted rules
(e.g. a user-level `bash * allow`) would otherwise leak into any test that
builds a PermissionManager/PermissionStore without an explicit path and
silently flip prompt/deny assertions. Subprocess-driven tests (worker stdio,
TUI e2e) are already hermetic through their own isolated HOME.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import cortex.tooling.permissions as permissions


@pytest.fixture(autouse=True)
def _hermetic_permission_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        permissions,
        "default_store_path",
        lambda: tmp_path / "tool_permissions.yaml",
    )
