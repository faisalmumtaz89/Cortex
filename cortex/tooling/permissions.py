"""Rule-based permission engine for tooling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import yaml

from cortex.tooling.types import PermissionAction, PermissionRule


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


class PermissionDeniedError(RuntimeError):
    """Raised when permission is denied by rule or user decision."""


class PermissionStore:
    """Persistent rule store for tooling permissions."""

    def __init__(self, path: Optional[Path] = None):
        self.path = (path or (Path.home() / ".cortex" / "tool_permissions.yaml")).expanduser()

    def load(self) -> List[PermissionRule]:
        if not self.path.exists():
            return []

        try:
            payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            return []

        raw_rules = payload.get("rules", []) if isinstance(payload, dict) else []
        rules: List[PermissionRule] = []
        for raw in raw_rules:
            if not isinstance(raw, dict):
                continue
            permission = str(raw.get("permission", "")).strip()
            pattern = str(raw.get("pattern", "")).strip() or "*"
            action_raw = str(raw.get("action", "")).strip().lower()
            if not permission:
                continue
            try:
                action = PermissionAction(action_raw)
            except Exception:
                continue
            rules.append(PermissionRule(permission=permission, pattern=pattern, action=action))
        return rules

    def save(self, rules: Sequence[PermissionRule]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "version": 1,
            "rules": [
                {
                    "permission": rule.permission,
                    "pattern": rule.pattern,
                    "action": rule.action.value,
                }
                for rule in rules
            ],
        }
        self.path.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")


class PermissionManager:
    """Evaluate and resolve tool permission requests."""

    def __init__(self, *, store: Optional[PermissionStore] = None, rules: Optional[Iterable[PermissionRule]] = None):
        self.store = store or PermissionStore()
        self.base_rules: List[PermissionRule] = list(rules or [])
        self.persisted_rules: List[PermissionRule] = self.store.load()
        self.session_allow_once: Dict[str, List[PermissionRule]] = {}

    def _matches(self, rule: PermissionRule, permission: str, pattern: str) -> bool:
        return fnmatchcase(permission, rule.permission) and fnmatchcase(pattern, rule.pattern)

    def _evaluate_one(self, *, permission: str, pattern: str, session_id: str) -> PermissionAction:
        rules: List[PermissionRule] = []
        rules.extend(self.base_rules)
        rules.extend(self.persisted_rules)
        rules.extend(self.session_allow_once.get(session_id, []))

        decision: PermissionAction = PermissionAction.ASK
        for rule in rules:
            if self._matches(rule, permission, pattern):
                decision = rule.action
        return decision

    def evaluate(self, *, permission: str, patterns: Sequence[str], session_id: str) -> PermissionAction:
        """Evaluate aggregate decision across requested patterns."""
        decisions = [self._evaluate_one(permission=permission, pattern=pattern, session_id=session_id) for pattern in patterns]
        if PermissionAction.DENY in decisions:
            return PermissionAction.DENY
        if PermissionAction.ASK in decisions:
            return PermissionAction.ASK
        return PermissionAction.ALLOW

    def _record_allow_once(self, *, permission: str, patterns: Sequence[str], session_id: str) -> None:
        rules = self.session_allow_once.setdefault(session_id, [])
        rules.extend(
            PermissionRule(permission=permission, pattern=pattern, action=PermissionAction.ALLOW)
            for pattern in patterns
        )

    def _record_allow_always(self, *, permission: str, patterns: Sequence[str]) -> None:
        for pattern in patterns:
            candidate = PermissionRule(permission=permission, pattern=pattern, action=PermissionAction.ALLOW)
            if candidate not in self.persisted_rules:
                self.persisted_rules.append(candidate)
        self.store.save(self.persisted_rules)

    def request(
        self,
        *,
        permission: str,
        patterns: Sequence[str],
        metadata: Optional[Dict[str, object]] = None,
        session_id: str,
        prompt_callback: Optional[Callable[[PermissionRequest], PermissionDecision]] = None,
    ) -> None:
        """Resolve permission request or raise PermissionDeniedError."""
        normalized_patterns = [str(pattern).strip() or "*" for pattern in patterns] or ["*"]
        decision = self.evaluate(permission=permission, patterns=normalized_patterns, session_id=session_id)

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
            self._record_allow_once(
                permission=permission,
                patterns=normalized_patterns,
                session_id=session_id,
            )
            return

        if user_decision == PermissionDecision.ALLOW_ALWAYS:
            self._record_allow_always(
                permission=permission,
                patterns=normalized_patterns,
            )
            self._record_allow_once(
                permission=permission,
                patterns=normalized_patterns,
                session_id=session_id,
            )
            return

        raise PermissionDeniedError(
            f"Permission rejected by user: permission={permission} patterns={normalized_patterns}"
        )
