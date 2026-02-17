from pathlib import Path

import pytest

from cortex.tooling.permissions import (
    PermissionDecision,
    PermissionDeniedError,
    PermissionManager,
    PermissionStore,
)
from cortex.tooling.types import PermissionAction, PermissionRule


def test_permission_allow_once_is_session_scoped():
    manager = PermissionManager()
    calls = {"count": 0}

    def prompt(_request):
        calls["count"] += 1
        return PermissionDecision.ALLOW_ONCE

    manager.request(
        permission="read",
        patterns=["src/*"],
        session_id="s1",
        prompt_callback=prompt,
    )
    manager.request(
        permission="read",
        patterns=["src/*"],
        session_id="s1",
        prompt_callback=prompt,
    )
    assert calls["count"] == 1

    manager.request(
        permission="read",
        patterns=["src/*"],
        session_id="s2",
        prompt_callback=prompt,
    )
    assert calls["count"] == 2


def test_permission_allow_always_persists(tmp_path: Path):
    store = PermissionStore(path=tmp_path / "tool_permissions.yaml")
    manager = PermissionManager(store=store)

    manager.request(
        permission="grep",
        patterns=["*"],
        session_id="s1",
        prompt_callback=lambda _request: PermissionDecision.ALLOW_ALWAYS,
    )

    reloaded = PermissionManager(store=store)
    assert reloaded.evaluate(permission="grep", patterns=["*"], session_id="new") == PermissionAction.ALLOW


def test_permission_denied_by_rule_short_circuits_prompt():
    manager = PermissionManager(
        rules=[PermissionRule(permission="bash", pattern="*", action=PermissionAction.DENY)]
    )

    with pytest.raises(PermissionDeniedError):
        manager.request(
            permission="bash",
            patterns=["*"],
            session_id="s1",
            prompt_callback=lambda _request: PermissionDecision.ALLOW_ONCE,
        )
