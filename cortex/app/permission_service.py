"""Permission request/reply bridge for worker protocol."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from cortex.tooling.permissions import PermissionDecision, PermissionRequest


@dataclass
class _PendingPermission:
    session_id: str
    request: PermissionRequest
    decision: Optional[PermissionDecision]
    condition: threading.Condition


class PermissionService:
    """Coordinate async frontend permission replies with blocking tool prompts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Dict[str, _PendingPermission] = {}

    def ask(
        self,
        *,
        session_id: str,
        request: PermissionRequest,
        emit_event,
    ) -> PermissionDecision:
        """Emit permission request event and block until the frontend replies,
        or the turn is interrupted (which rejects the request)."""
        request_id = uuid.uuid4().hex
        cond = threading.Condition()
        pending = _PendingPermission(
            session_id=session_id,
            request=request,
            decision=None,
            condition=cond,
        )

        with self._lock:
            self._pending[request_id] = pending

        emit_event(
            session_id=session_id,
            event_type="permission.asked",
            payload={
                "request_id": request_id,
                "permission": request.permission,
                "patterns": list(request.patterns),
                "metadata": dict(request.metadata or {}),
                "options": ["allow_once", "allow_always", "reject"],
            },
        )

        with cond:
            while pending.decision is None:
                cond.wait()

        with self._lock:
            self._pending.pop(request_id, None)

        decision = pending.decision or PermissionDecision.REJECT
        emit_event(
            session_id=session_id,
            event_type="permission.replied",
            payload={"request_id": request_id, "decision": decision.value},
        )
        return decision

    def cancel(self, session_id: str) -> None:
        """Reject the session's pending requests: its turn was interrupted."""
        with self._lock:
            pending = [p for p in self._pending.values() if p.session_id == session_id]
        for item in pending:
            with item.condition:
                if item.decision is None:
                    item.decision = PermissionDecision.REJECT
                item.condition.notify_all()

    def reply(self, *, session_id: str, request_id: str, reply: str) -> bool:
        """Apply frontend permission reply for a pending request."""
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None or pending.session_id != session_id:
                return False

        mapped = {
            "allow_once": PermissionDecision.ALLOW_ONCE,
            "allow_always": PermissionDecision.ALLOW_ALWAYS,
            "reject": PermissionDecision.REJECT,
        }.get(reply)
        if mapped is None:
            return False

        with pending.condition:
            if pending.decision is not None:
                return False  # already rejected: the turn was interrupted
            pending.decision = mapped
            pending.condition.notify_all()
        return True
