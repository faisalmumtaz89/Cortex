"""Permission request/reply bridge for worker protocol."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from cortex.tooling.permissions import PermissionDecision, PermissionRequest


@dataclass
class _PendingPermission:
    session_id: str
    request: PermissionRequest
    decision: Optional[PermissionDecision]
    created_at: float
    condition: threading.Condition


class PermissionService:
    """Coordinate async frontend permission replies with blocking tool prompts."""

    def __init__(self, *, timeout_seconds: int = 300) -> None:
        self.timeout_seconds = max(1, int(timeout_seconds))
        self._lock = threading.Lock()
        self._pending: Dict[str, _PendingPermission] = {}

    def ask(
        self,
        *,
        session_id: str,
        request: PermissionRequest,
        emit_event,
    ) -> PermissionDecision:
        """Emit permission request event and block until frontend reply or timeout."""
        request_id = uuid.uuid4().hex
        cond = threading.Condition()
        pending = _PendingPermission(
            session_id=session_id,
            request=request,
            decision=None,
            created_at=time.time(),
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

        deadline = time.time() + self.timeout_seconds
        with cond:
            while pending.decision is None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    pending.decision = PermissionDecision.REJECT
                    break
                cond.wait(timeout=remaining)

        with self._lock:
            self._pending.pop(request_id, None)

        decision = pending.decision or PermissionDecision.REJECT
        emit_event(
            session_id=session_id,
            event_type="permission.replied",
            payload={"request_id": request_id, "decision": decision.value},
        )
        return decision

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
            pending.decision = mapped
            pending.condition.notify_all()
        return True
