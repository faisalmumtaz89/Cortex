"""Event sequencing and emission helpers for worker protocol."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict

from cortex.protocol.types import EventEnvelope, EventFrame, EventType


class SessionSequencer:
    """Generate monotonic sequence IDs per session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq_by_session: Dict[str, int] = {}

    def next(self, session_id: str) -> int:
        with self._lock:
            current = self._seq_by_session.get(session_id, 0) + 1
            self._seq_by_session[session_id] = current
            return current

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._seq_by_session.pop(session_id, None)


class EventEmitter:
    """Emit validated protocol events through a transport callback."""

    def __init__(self, send: Callable[[dict], None]) -> None:
        self._send = send
        self._sequencer = SessionSequencer()

    def emit(self, *, session_id: str, event_type: EventType, payload: dict) -> EventEnvelope:
        envelope = EventEnvelope(
            session_id=session_id,
            seq=self._sequencer.next(session_id),
            ts_ms=int(time.time() * 1000),
            event_type=event_type,
            payload=payload,
        )
        frame = EventFrame(params=envelope)
        self._send(frame.model_dump())
        return envelope
