"""What of a tool's result each provider is sent: the result, without what is
for the app's row of the call only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cortex.tooling.types import ToolCall, ToolExecutionState, ToolResult
from tests.provider_server import ProviderServer, make_client

_TOOLS = [SimpleNamespace(name="ping", description="ping",
                          parameters={"type": "object", "properties": {}})]


def _edit(call: ToolCall) -> ToolResult:
    return ToolResult(
        id=call.id, name=call.name, state=ToolExecutionState.COMPLETED, ok=True, output="pong",
        metadata={
            "exit_code": 0,
            "diff": {"path": "a.py", "rows": [{"text": "DIFF_ROW"}]},
            "permission_note": "Allowed from now on here",
            "permission_patterns": ["the matched command"],
        },
    )


@pytest.mark.parametrize("kind", ["chat", "responses", "anthropic"])
def test_the_model_is_sent_the_result_without_what_is_for_the_apps_row(kind: str) -> None:
    server = ProviderServer(kind, [{"tool": True}, {}])
    try:
        list(make_client(kind, server.base_url).stream_events(
            model_id="test-model", messages=[{"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0, top_p=1, tools=_TOOLS, tool_executor=_edit,
        ))
    finally:
        server.close()
    sent = json.dumps(server.bodies[1])
    assert "pong" in sent and "exit_code" in sent
    for kept_from_the_model in ("DIFF_ROW", "Allowed from now on here", "the matched command"):
        assert kept_from_the_model not in sent, kept_from_the_model
