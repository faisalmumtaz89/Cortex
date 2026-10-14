import fcntl
import os
import time
from pathlib import Path

import pytest
import yaml

from cortex.tooling.permissions import (
    PermissionDecision,
    PermissionDeniedError,
    PermissionManager,
    PermissionStore,
)
from cortex.tooling.types import PermissionAction, PermissionRule


def _ask(manager, pattern, decision=PermissionDecision.REJECT):
    """Request bash permission for `pattern`; return whether the user was asked."""
    asked = []

    def prompt(_request):
        asked.append(1)
        return decision

    try:
        manager.request(permission="bash", patterns=[pattern], prompt_callback=prompt)
    except PermissionDeniedError:
        pass
    return bool(asked)


def test_allow_once_allows_only_that_call(tmp_path: Path):
    manager = PermissionManager(store=PermissionStore(path=tmp_path / "perm.yaml"))

    assert _ask(manager, "make test", PermissionDecision.ALLOW_ONCE)
    assert _ask(manager, "make test")
    assert not (tmp_path / "perm.yaml").exists()


def test_allow_always_allows_that_exact_command_in_that_repository(tmp_path: Path):
    store = PermissionStore(path=tmp_path / "perm.yaml")
    repo, other = tmp_path / "repo", tmp_path / "other"
    repo.mkdir()
    other.mkdir()

    assert _ask(PermissionManager(store=store, repo_root=repo), "ls *.py",
                PermissionDecision.ALLOW_ALWAYS)

    again = PermissionManager(store=store, repo_root=repo)
    assert not _ask(again, "ls *.py")
    # The command is matched literally: its glob characters match nothing else.
    assert _ask(again, "ls secrets.py; curl evil")
    assert _ask(again, "ls x.py")
    assert _ask(PermissionManager(store=store, repo_root=other), "ls *.py")


def test_command_is_matched_exactly_as_it_runs(tmp_path: Path):
    manager = PermissionManager(store=PermissionStore(path=tmp_path / "perm.yaml"))
    assert _ask(manager, "touch one.txt", PermissionDecision.ALLOW_ALWAYS)

    assert not _ask(manager, "touch one.txt")
    assert _ask(manager, "touch one.txt\u00a0")
    assert _ask(manager, " touch one.txt")


def test_call_without_a_command_or_file_is_never_saved(tmp_path: Path):
    manager = PermissionManager(store=PermissionStore(path=tmp_path / "perm.yaml"))

    assert _ask(manager, "*", PermissionDecision.ALLOW_ALWAYS)
    assert _ask(manager, "*")
    assert not (tmp_path / "perm.yaml").exists()


def test_saving_keeps_rules_added_meanwhile_and_leaves_an_unreadable_file(tmp_path: Path):
    path = tmp_path / "perm.yaml"
    first = PermissionManager(store=PermissionStore(path=path), repo_root=tmp_path)
    second = PermissionManager(store=PermissionStore(path=path), repo_root=tmp_path)
    _ask(first, "make a", PermissionDecision.ALLOW_ALWAYS)
    _ask(second, "make b", PermissionDecision.ALLOW_ALWAYS)

    reloaded = PermissionManager(store=PermissionStore(path=path), repo_root=tmp_path)
    assert not _ask(reloaded, "make a") and not _ask(reloaded, "make b")

    path.write_text("rules: [unclosed\n")
    _ask(reloaded, "make c", PermissionDecision.ALLOW_ALWAYS)
    assert path.read_text() == "rules: [unclosed\n"


def test_saved_command_matches_exactly_after_a_restart(tmp_path: Path):
    store = PermissionStore(path=tmp_path / "perm.yaml")
    _ask(PermissionManager(store=store), "rm -rf build\u00a0", PermissionDecision.ALLOW_ALWAYS)

    restarted = PermissionManager(store=store)
    assert not _ask(restarted, "rm -rf build\u00a0")
    assert _ask(restarted, "rm -rf build")


def test_rules_changed_on_disk_apply_without_a_restart(tmp_path: Path):
    path = tmp_path / "perm.yaml"
    manager = PermissionManager(store=PermissionStore(path=path), repo_root=tmp_path)
    _ask(manager, "make", PermissionDecision.ALLOW_ALWAYS)
    assert not _ask(manager, "make")

    path.write_text(yaml.safe_dump({"rules": [
        {"permission": "bash", "pattern": "*", "action": "deny"}]}))
    with pytest.raises(PermissionDeniedError):
        manager.request(permission="bash", patterns=["make"],
                        prompt_callback=lambda _request: PermissionDecision.ALLOW_ONCE)


def _deny_all(path: Path) -> None:
    path.write_text(yaml.safe_dump({"rules": [
        {"permission": "bash", "pattern": "*", "action": "deny"}]}))


def _denied(manager, pattern) -> bool:
    try:
        manager.request(permission="bash", patterns=[pattern],
                        prompt_callback=lambda _request: PermissionDecision.ALLOW_ONCE)
    except PermissionDeniedError:
        return True
    return False


def test_deleting_the_file_right_after_allow_always_resets_the_session(tmp_path: Path):
    path = tmp_path / "perm.yaml"
    manager = PermissionManager(store=PermissionStore(path=path))
    _ask(manager, "make", PermissionDecision.ALLOW_ALWAYS)

    path.unlink()
    assert _ask(manager, "make")


def test_unparseable_file_keeps_the_rules_last_read(tmp_path: Path):
    path = tmp_path / "perm.yaml"
    _deny_all(path)
    manager = PermissionManager(store=PermissionStore(path=path))
    assert _denied(manager, "rm x")

    with path.open("a") as out:
        out.write("rules: [unclosed\n")
    assert _denied(manager, "rm x")


def test_edit_that_restores_the_mtime_is_still_picked_up(tmp_path: Path):
    path = tmp_path / "perm.yaml"
    path.write_text(yaml.safe_dump({"rules": []}))
    before = path.stat()
    manager = PermissionManager(store=PermissionStore(path=path))
    assert not _denied(manager, "make")

    _deny_all(path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert _denied(manager, "make")


def test_rules_that_could_not_be_saved_give_way_to_the_files_rules(tmp_path: Path, monkeypatch):
    path = tmp_path / "perm.yaml"
    manager = PermissionManager(store=PermissionStore(path=path))

    def fail(_rules):
        raise OSError("disk full")

    monkeypatch.setattr(manager.store, "add", fail)
    _ask(manager, "make", PermissionDecision.ALLOW_ALWAYS)
    assert not _ask(manager, "make")  # kept for the session

    _deny_all(path)
    assert _denied(manager, "make")


def test_a_held_lock_delays_saving_by_seconds_at_most(tmp_path: Path, monkeypatch):
    path = tmp_path / "perm.yaml"
    monkeypatch.setattr("cortex.tooling.permissions.LOCK_TIMEOUT_SECONDS", 0.5)
    with open(tmp_path / ".perm.yaml.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        manager = PermissionManager(store=PermissionStore(path=path))
        started = time.monotonic()
        _ask(manager, "make", PermissionDecision.ALLOW_ALWAYS)
        assert time.monotonic() - started < 3
    assert not path.exists()
    assert not _ask(manager, "make")


def test_saving_keeps_the_file_mode_and_writes_through_a_symlink(tmp_path: Path):
    real = tmp_path / "dotfiles" / "perm.yaml"
    real.parent.mkdir()
    real.write_text("rules: []\n")
    real.chmod(0o600)
    link = tmp_path / "perm.yaml"
    link.symlink_to(real)

    manager = PermissionManager(store=PermissionStore(path=link))
    _ask(manager, "make", PermissionDecision.ALLOW_ALWAYS)

    assert link.is_symlink()
    assert "make" in real.read_text()
    assert real.stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in real.parent.iterdir()) == [".perm.yaml.lock", "perm.yaml"]


def test_version_1_allow_rules_for_prompting_tools_are_dropped(tmp_path: Path):
    path = tmp_path / "perm.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "rules": [
        {"permission": "bash", "pattern": "*", "action": "allow"},
        {"permission": "edit", "pattern": "src/app.py", "action": "allow"},
        {"permission": "bash", "pattern": "rm *", "action": "deny"},
        {"permission": "grep", "pattern": "*", "action": "allow"},
    ]}))

    rules = PermissionStore(path=path).load()

    assert [(rule.permission, rule.action.value) for rule in rules] == [
        ("bash", "deny"),
        ("grep", "allow"),
    ]


def test_hand_written_rules_apply(tmp_path: Path):
    path = tmp_path / "perm.yaml"
    path.write_text(yaml.safe_dump({"rules": [
        {"permission": "bash", "pattern": "git status", "action": "allow"},
        {"permission": "bash", "pattern": "git log", "action": "allow", "repo": f"{tmp_path}/"},
        {"permission": "bash", "pattern": "git push", "action": "allow", "repo": "/elsewhere"},
    ]}))

    manager = PermissionManager(store=PermissionStore(path=path), repo_root=tmp_path)

    assert not _ask(manager, "git status")
    assert not _ask(manager, "git log")
    assert _ask(manager, "git push")


def test_permission_denied_by_rule_short_circuits_prompt(tmp_path: Path):
    manager = PermissionManager(
        store=PermissionStore(path=tmp_path / "perm.yaml"),
        rules=[PermissionRule(permission="bash", pattern="*", action=PermissionAction.DENY)],
    )

    with pytest.raises(PermissionDeniedError):
        manager.request(
            permission="bash",
            patterns=["*"],
            prompt_callback=lambda _request: PermissionDecision.ALLOW_ONCE,
        )


def test_a_reply_after_an_interrupt_does_not_undo_the_rejection():
    import threading

    from cortex.app.permission_service import PermissionService
    from cortex.tooling.permissions import PermissionRequest

    service = PermissionService()
    asked = []
    decided = []
    request = PermissionRequest(permission="bash", patterns=["make"], metadata={})

    def emit(*, session_id, event_type, payload):
        if event_type == "permission.asked":
            asked.append(payload["request_id"])

    thread = threading.Thread(
        target=lambda: decided.append(service.ask(session_id="s", request=request, emit_event=emit))
    )
    thread.start()
    while not asked:
        time.sleep(0.01)
    service.cancel("s")
    service.reply(session_id="s", request_id=asked[0], reply="allow_once")
    thread.join(timeout=5)

    assert decided == [PermissionDecision.REJECT]
