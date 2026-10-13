"""Rule-based permission engine for tooling."""

from __future__ import annotations

import fcntl
import glob
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from cortex.tooling.types import PermissionAction, PermissionRule

logger = logging.getLogger(__name__)


class PermissionDecision(str, Enum):
    """Interactive user decision for a permission request."""

    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    REJECT = "reject"


@dataclass(frozen=True)
class PermissionRequest:
    """Permission request payload for UI prompt callbacks."""

    permission: str
    patterns: List[str]
    metadata: Dict[str, object]


# Version 2 stores the repository each "Allow always" rule belongs to.
STORE_VERSION = 2
LOCK_TIMEOUT_SECONDS = 2.0
# Never equal to a file's stamp, nor to the None of a missing file.
_STALE = object()


class PermissionDeniedError(RuntimeError):
    """Raised when permission is denied by rule or user decision."""


def default_store_path() -> Path:
    """Default on-disk permission store. The single seam for the default —
    tests point it at a scratch file (see tests/conftest.py) so the user's
    real persisted rules can never leak into assertions."""
    return Path.home() / ".cortex" / "tool_permissions.yaml"


class PermissionStore:
    """Persistent rule store for tooling permissions."""

    def __init__(self, path: Optional[Path] = None):
        self.path = (path or default_store_path()).expanduser()

    def _read(self) -> Optional[Dict[str, object]]:
        """The file's contents; {} when it does not exist, None when it cannot
        be read or parsed."""
        if not self.path.exists():
            return {}
        try:
            payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Ignoring unreadable permission rules in %s: %s", self.path, exc)
            return None
        if not isinstance(payload, dict):
            logger.warning("Ignoring permission rules in %s: not a mapping", self.path)
            return None
        return payload

    def load(self) -> Optional[List[PermissionRule]]:
        """The saved rules; None when the file cannot be read or parsed."""
        payload = self._read()
        return None if payload is None else _parse_rules(payload)

    def stamp(self) -> Optional[Tuple[int, ...]]:
        """Identifies the file's current version, to tell whether rules need
        reloading; the change time catches edits that restore the mtime."""
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def add(self, new_rules: Sequence[PermissionRule]) -> None:
        """Add rules to the file as it is now on disk, so rules other sessions
        or the user saved meanwhile are kept; a lock keeps two sessions from
        saving at once. Raises when the file cannot be read or the lock not
        taken in time: the file is then left as it is. The file keeps its
        mode, and a symlinked file is updated where it points."""
        target = self.path.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.with_name(f".{target.name}.lock")
        lock = os.open(
            lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        try:
            deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{lock_path} is locked by another session")
                    time.sleep(0.05)
            payload = self._read()
            if payload is None:
                raise ValueError(f"{self.path} cannot be read; it was left unchanged")
            rules = _parse_rules(payload)
            rules.extend(rule for rule in new_rules if rule not in rules)
            text = yaml.safe_dump(
                {
                    "version": STORE_VERSION,
                    "rules": [
                        {
                            "permission": rule.permission,
                            "pattern": rule.pattern,
                            "action": rule.action.value,
                            **({"repo": rule.repo} if rule.repo else {}),
                        }
                        for rule in rules
                    ],
                },
                sort_keys=False,
            )
            mode = target.stat().st_mode & 0o777 if target.exists() else 0o600
            # Created exclusively with mode 0600, so it cannot be a planted symlink.
            fd, partial = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as out:
                    out.write(text)
                    os.fchmod(out.fileno(), mode)
                os.replace(partial, target)
            finally:
                Path(partial).unlink(missing_ok=True)
        finally:
            os.close(lock)


def _parse_rules(payload: Dict[str, object]) -> List[PermissionRule]:
    raw_rules = payload.get("rules", [])
    # Version 1 files were written by Cortex, whose "Allow always" then covered
    # every command or file in every repository: their allow rules for the
    # prompting tools are dropped, so those tools ask again.
    legacy = payload.get("version") == 1
    rules: List[PermissionRule] = []
    for raw in raw_rules if isinstance(raw_rules, list) else []:
        if not isinstance(raw, dict):
            continue
        permission = str(raw.get("permission", "")).strip()
        # Not stripped: a saved command must match exactly as it was approved.
        pattern = str(raw.get("pattern", "")) or "*"
        action_raw = str(raw.get("action", "")).strip().lower()
        if not permission:
            continue
        try:
            action = PermissionAction(action_raw)
        except Exception:
            continue
        if legacy and action == PermissionAction.ALLOW and permission in ("bash", "edit"):
            continue
        repo = str(raw.get("repo", "") or "").strip()
        if repo:
            repo = str(Path(repo).expanduser().resolve())
        rules.append(
            PermissionRule(permission=permission, pattern=pattern, action=action, repo=repo)
        )
    return rules


def default_rules() -> List[PermissionRule]:
    """Default policy: read-only tools run silently; edits and bash prompt."""
    return [
        PermissionRule(permission="read", pattern="*", action=PermissionAction.ALLOW),
        PermissionRule(permission="list", pattern="*", action=PermissionAction.ALLOW),
        PermissionRule(permission="grep", pattern="*", action=PermissionAction.ALLOW),
    ]


class PermissionManager:
    """Evaluate and resolve tool permission requests.

    Patterns name exactly what a call does: the command a bash call runs, or
    the file an edit changes. "Allow once" allows that one call; "Allow
    always" allows the same pattern again in this repository only."""

    def __init__(
        self,
        *,
        store: Optional[PermissionStore] = None,
        rules: Optional[Iterable[PermissionRule]] = None,
        repo_root: Optional[Path] = None,
    ):
        self.store = store or PermissionStore()
        self.base_rules: List[PermissionRule] = list(rules or [])
        self.persisted_rules: List[PermissionRule] = self.store.load() or []
        self._loaded_stamp: object = self.store.stamp()
        self._unsaved: List[PermissionRule] = []
        self.repo = str((repo_root or Path.cwd()).resolve())

    def _reload_if_changed(self) -> None:
        """Pick up rules the user or another session changed on disk. While the
        file cannot be parsed, the rules last read stay in force."""
        stamp = self.store.stamp()
        if stamp == self._loaded_stamp:
            return
        on_disk = self.store.load()
        if on_disk is None:
            return
        # Rules this session could not save come first: the file's rules,
        # the user's own edits included, override them.
        self.persisted_rules = [*(r for r in self._unsaved if r not in on_disk), *on_disk]
        self._loaded_stamp = stamp

    def _matches(self, rule: PermissionRule, permission: str, pattern: str) -> bool:
        return (
            rule.repo in ("", self.repo)
            and fnmatchcase(permission, rule.permission)
            and fnmatchcase(pattern, rule.pattern)
        )

    def _evaluate_one(self, *, permission: str, pattern: str) -> PermissionAction:
        decision: PermissionAction = PermissionAction.ASK
        self._reload_if_changed()
        for rule in [*self.base_rules, *self.persisted_rules]:
            if self._matches(rule, permission, pattern):
                decision = rule.action
        return decision

    def evaluate(self, *, permission: str, patterns: Sequence[str]) -> PermissionAction:
        """Evaluate aggregate decision across requested patterns."""
        decisions = [
            self._evaluate_one(permission=permission, pattern=pattern) for pattern in patterns
        ]
        if PermissionAction.DENY in decisions:
            return PermissionAction.DENY
        if PermissionAction.ASK in decisions:
            return PermissionAction.ASK
        return PermissionAction.ALLOW

    def _record_allow_always(self, *, permission: str, patterns: Sequence[str]) -> None:
        if "*" in patterns:
            return  # a call with no command or file: allowed once, never saved
        added: List[PermissionRule] = []
        for pattern in patterns:
            # Escaped, so the rule matches this exact command or path only.
            candidate = PermissionRule(
                permission=permission,
                pattern=glob.escape(pattern),
                action=PermissionAction.ALLOW,
                repo=self.repo,
            )
            if candidate not in self.persisted_rules:
                self.persisted_rules.append(candidate)
                added.append(candidate)
        try:
            self.store.add(added)
        except Exception as exc:  # the rules still apply in this session
            logger.warning("Could not save permission rules to %s: %s", self.store.path, exc)
            self._unsaved.extend(added)
        self._loaded_stamp = _STALE  # reload: the file now holds the merged rules

    def request(
        self,
        *,
        permission: str,
        patterns: Sequence[str],
        metadata: Optional[Dict[str, object]] = None,
        prompt_callback: Optional[Callable[[PermissionRequest], PermissionDecision]] = None,
    ) -> None:
        """Resolve permission request or raise PermissionDeniedError."""
        # Patterns are matched exactly as given: stripping could make a command
        # that runs differently match an approved one.
        normalized_patterns = [str(pattern) or "*" for pattern in patterns] or ["*"]
        decision = self.evaluate(permission=permission, patterns=normalized_patterns)

        if decision == PermissionAction.ALLOW:
            return

        if decision == PermissionAction.DENY:
            raise PermissionDeniedError(
                f"Permission denied by rule: permission={permission} patterns={normalized_patterns}"
            )

        if prompt_callback is None:
            raise PermissionDeniedError(
                f"Permission requires approval but no prompt callback is configured: {permission}"
            )

        prompt = PermissionRequest(
            permission=permission,
            patterns=normalized_patterns,
            metadata=metadata or {},
        )
        user_decision = prompt_callback(prompt)

        if user_decision == PermissionDecision.ALLOW_ONCE:
            return

        if user_decision == PermissionDecision.ALLOW_ALWAYS:
            self._record_allow_always(permission=permission, patterns=normalized_patterns)
            return

        raise PermissionDeniedError(
            f"Permission rejected by user: permission={permission} patterns={normalized_patterns}"
        )
