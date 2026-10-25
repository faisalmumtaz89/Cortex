"""Interrupting a turn while each real SDK client waits on the model.

The server keeps connections alive and sends chunked bodies, as the real
providers do, so the SDKs pool connections and read the way they do live.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Dict

import pytest

from cortex.tooling.types import (
    TextDeltaEvent,
    ToolExecutionState,
    ToolResult,
    TurnInterrupt,
    TurnInterruptedError,
)
from tests.provider_server import ProviderServer, make_client

_TOOLS = [SimpleNamespace(name="ping", description="ping",
                          parameters={"type": "object", "properties": {}})]


def _ping(call) -> ToolResult:
    return ToolResult(id=call.id, name=call.name, state=ToolExecutionState.COMPLETED, ok=True,
                      output="pong")


def _run_interrupted(kind, script, *, tools, interrupt_after, monkeypatch):
    """Run one client turn, interrupt it after `interrupt_after` seconds and
    return (outcome, seconds from the interrupt to the end, server)."""
    server = ProviderServer(kind, script)
    client = make_client(kind, server.base_url, monkeypatch)
    interrupt = TurnInterrupt()
    outcome: Dict[str, object] = {}

    def run() -> None:
        try:
            events = list(client.stream_events(
                model_id="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0,
                top_p=1,
                tools=_TOOLS if tools else None,
                tool_executor=_ping if tools else None,
                interrupt=interrupt,
            ))
            outcome["events"] = events
        except BaseException as exc:
            outcome["error"] = exc
        outcome["ended"] = time.time()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(interrupt_after)
    interrupted_at = time.time()
    interrupt.set()
    thread.join(30)
    server.close()
    return outcome, outcome["ended"] - interrupted_at, server


@pytest.mark.parametrize("tools", [True, False], ids=["tools", "no-tools"])
@pytest.mark.parametrize("kind", ["chat", "anthropic", "responses"])
def test_interrupt_stops_a_reply_the_model_is_still_working_on(kind, tools, monkeypatch):
    outcome, waited, server = _run_interrupted(
        kind, [{"body_delay": 30}], tools=tools, interrupt_after=1, monkeypatch=monkeypatch
    )

    assert isinstance(outcome.get("error"), TurnInterruptedError), outcome
    assert waited < 2
    assert len(server.requests) == 1


@pytest.mark.parametrize("kind", ["chat", "anthropic", "responses"])
def test_interrupt_before_the_next_reply_starts_sends_no_further_request(kind, monkeypatch):
    # The interrupt arrives while the second request, on the first one's
    # pooled connection, still waits for its response headers.
    outcome, waited, server = _run_interrupted(
        kind,
        [{"tool": True}, {"header_delay": 3}, {}],
        tools=True,
        interrupt_after=1.5,
        monkeypatch=monkeypatch,
    )

    assert isinstance(outcome.get("error"), TurnInterruptedError), outcome
    assert len(server.requests) == 2
    assert waited < 3


def test_uninterrupted_turn_still_reads_the_whole_reply(monkeypatch):
    server = ProviderServer("anthropic", [{}])
    client = make_client("anthropic", server.base_url, monkeypatch)
    try:
        events = list(client.stream_events(
            model_id="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0,
            top_p=1,
            interrupt=TurnInterrupt(),
        ))
    finally:
        server.close()

    assert [e.delta for e in events if isinstance(e, TextDeltaEvent)] == ["late"]
