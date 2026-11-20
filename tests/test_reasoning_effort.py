"""reasoning_effort: asked of the provider when it is set, in the field its
API has for it, and not at all when it is not."""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from tests.provider_server import ProviderServer, make_client

_KINDS = ["chat", "responses", "anthropic"]
_ASKED = {
    "chat": lambda body: body.get("reasoning_effort"),
    "responses": lambda body: (body.get("reasoning") or {}).get("effort"),
    "anthropic": lambda body: (body.get("output_config") or {}).get("effort"),
}


def _bodies(kind: str, effort: Optional[str]) -> List[Dict[str, object]]:
    server = ProviderServer(kind, [{}])
    client = make_client(kind, server.base_url)
    try:
        list(client.stream_events(
            model_id="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0,
            top_p=1,
            reasoning_effort=effort,
        ))
    finally:
        server.close()
    return server.bodies


@pytest.mark.parametrize("kind", _KINDS)
def test_the_effort_set_is_asked_for_in_the_field_the_api_has(kind: str) -> None:
    assert [_ASKED[kind](body) for body in _bodies(kind, "high")] == ["high"]


@pytest.mark.parametrize("kind", _KINDS)
def test_no_effort_is_asked_for_when_none_is_set(kind: str) -> None:
    (body,) = _bodies(kind, None)
    assert not {"reasoning_effort", "reasoning", "output_config", "thinking"} & set(body), body
