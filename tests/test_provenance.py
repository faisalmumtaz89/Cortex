"""Per-turn model provenance: the shown model must be the answering model.

Unit-verifies the rule table in cortex/tooling/provenance.py and, at the
orchestrator level, that a turn whose response reports the WRONG model (or no
provenance at all) is rejected loudly — never rendered as a normal answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider
from cortex.conversation_manager import MessageRole
from cortex.tooling.orchestrator import ToolingOrchestrator
from cortex.tooling.provenance import normalize_model_name, verify_turn_provenance
from cortex.tooling.types import ErrorEvent, FinishEvent, TextDeltaEvent

LUMEN_ENDPOINT = "http://127.0.0.1:8399/v1"


def _lumen_provenance(model: str = "qwen3-5-9b", endpoint: str = LUMEN_ENDPOINT) -> dict:
    return {
        "client_kind": "lumen",
        "reported_model": model,
        "response_id": "chatcmpl-lumen-1",
        "endpoint": endpoint,
    }


# ---- rule table -------------------------------------------------------------


def test_normalize_treats_dot_and_dash_as_equal() -> None:
    assert normalize_model_name("qwen3.5-9b") == normalize_model_name("qwen3-5-9b")
    assert normalize_model_name("Qwen3_5 9B") == normalize_model_name("qwen3.5-9b")


def test_missing_provenance_fails() -> None:
    verdict = verify_turn_provenance(
        provider=CloudProvider.OPENAI, requested_model="gpt-5.1", provenance=None
    )
    assert not verdict.ok
    assert "no provenance" in verdict.reason


def test_lumen_reported_model_must_match_selector_model() -> None:
    verdict = verify_turn_provenance(
        provider=CloudProvider.LUMEN,
        requested_model="qwen3-5-9b",
        provenance=_lumen_provenance(model="evil-model"),
        expected_endpoint=LUMEN_ENDPOINT,
        lumen_ready=True,
    )
    assert not verdict.ok
    assert "evil-model" in verdict.reason


def test_lumen_accepts_dotted_spelling_of_same_model() -> None:
    verdict = verify_turn_provenance(
        provider=CloudProvider.LUMEN,
        requested_model="qwen3-5-9b",
        provenance=_lumen_provenance(model="qwen3.5-9b"),
        expected_endpoint=LUMEN_ENDPOINT,
        lumen_ready=True,
    )
    assert verdict.ok


def test_lumen_endpoint_must_be_the_managed_server() -> None:
    verdict = verify_turn_provenance(
        provider=CloudProvider.LUMEN,
        requested_model="qwen3-5-9b",
        provenance=_lumen_provenance(endpoint="http://127.0.0.1:9999/v1"),
        expected_endpoint=LUMEN_ENDPOINT,
        lumen_ready=True,
    )
    assert not verdict.ok
    assert "endpoint" in verdict.reason


def test_lumen_dead_server_fails_even_with_matching_record() -> None:
    verdict = verify_turn_provenance(
        provider=CloudProvider.LUMEN,
        requested_model="qwen3-5-9b",
        provenance=_lumen_provenance(),
        expected_endpoint=LUMEN_ENDPOINT,
        lumen_ready=False,
    )
    assert not verdict.ok
    assert "alive" in verdict.reason or "ready" in verdict.reason


def test_wrong_client_kind_fails() -> None:
    provenance = _lumen_provenance()
    provenance["client_kind"] = "openai"
    verdict = verify_turn_provenance(
        provider=CloudProvider.LUMEN,
        requested_model="qwen3-5-9b",
        provenance=provenance,
        expected_endpoint=LUMEN_ENDPOINT,
        lumen_ready=True,
    )
    assert not verdict.ok
    assert "openai client" in verdict.reason


def test_openai_allows_date_suffixed_release_names_only() -> None:
    ok = verify_turn_provenance(
        provider=CloudProvider.OPENAI,
        requested_model="gpt-5.1",
        provenance={
            "client_kind": "openai",
            "reported_model": "gpt-5.1-2026-01-15",
            "response_id": "resp_1",
            "endpoint": "https://api.openai.com/v1",
        },
    )
    assert ok.ok

    wrong = verify_turn_provenance(
        provider=CloudProvider.OPENAI,
        requested_model="gpt-5.1",
        provenance={
            "client_kind": "openai",
            "reported_model": "totally-different-model",
            "response_id": "resp_1",
            "endpoint": "https://api.openai.com/v1",
        },
    )
    assert not wrong.ok


def test_date_suffix_aliasing_accepts_both_provider_forms() -> None:
    """The two real-world pairs (validated live 2026-07-05): OpenAI dashed
    dates and Anthropic compact dates both alias to the bare id."""
    openai_pair = verify_turn_provenance(
        provider=CloudProvider.OPENAI,
        requested_model="gpt-5.5",
        provenance={
            "client_kind": "openai",
            "reported_model": "gpt-5.5-2026-04-23",
            "response_id": "resp_1",
            "endpoint": "https://api.openai.com/v1",
        },
    )
    assert openai_pair.ok

    anthropic_pair = verify_turn_provenance(
        provider=CloudProvider.ANTHROPIC,
        requested_model="claude-haiku-4-5",
        provenance={
            "client_kind": "anthropic",
            "reported_model": "claude-haiku-4-5-20251001",
            "response_id": "msg_1",
            "endpoint": "https://api.anthropic.com",
        },
    )
    assert anthropic_pair.ok


def test_variant_suffixes_are_different_models_not_aliases() -> None:
    """-mini/-codex variants share the prefix but are DIFFERENT models: a
    request for gpt-5.4 answered by gpt-5.4-mini must be rejected."""
    mini = verify_turn_provenance(
        provider=CloudProvider.OPENAI,
        requested_model="gpt-5.4",
        provenance={
            "client_kind": "openai",
            "reported_model": "gpt-5.4-mini-2026-03-17",
            "response_id": "resp_1",
            "endpoint": "https://api.openai.com/v1",
        },
    )
    assert not mini.ok

    codex = verify_turn_provenance(
        provider=CloudProvider.OPENAI,
        requested_model="gpt-5.5",
        provenance={
            "client_kind": "openai",
            "reported_model": "gpt-5.5-codex",
            "response_id": "resp_1",
            "endpoint": "https://api.openai.com/v1",
        },
    )
    assert not codex.ok


def _openai_provenance(reported_model: str) -> dict:
    return {
        "client_kind": "openai",
        "reported_model": reported_model,
        "response_id": "resp_1",
        "endpoint": "https://api.openai.com/v1",
    }


def test_date_pinned_request_is_satisfied_only_by_that_exact_snapshot() -> None:
    """The date-release strip applies to the REPORTED side only. A user who
    pins a snapshot (via ~/.cortex/cloud_models.json or a typed
    provider:model selector) must get exactly that snapshot: a different
    date — or a bare report that proves nothing about the snapshot — fails."""
    exact = verify_turn_provenance(
        provider=CloudProvider.OPENAI,
        requested_model="gpt-5.5-2026-04-23",
        provenance=_openai_provenance("gpt-5.5-2026-04-23"),
    )
    assert exact.ok

    different_snapshot = verify_turn_provenance(
        provider=CloudProvider.OPENAI,
        requested_model="gpt-5.5-2026-04-23",
        provenance=_openai_provenance("gpt-5.5-2026-05-30"),
    )
    assert not different_snapshot.ok

    bare_report = verify_turn_provenance(
        provider=CloudProvider.OPENAI,
        requested_model="gpt-5.5-2026-04-23",
        provenance=_openai_provenance("gpt-5.5"),
    )
    assert not bare_report.ok


def test_azure_binds_identity_via_client_not_deployment_name() -> None:
    # Deployment names are user aliases; a present-but-different reported
    # model passes, an EMPTY reported model still fails.
    ok = verify_turn_provenance(
        provider=CloudProvider.AZURE,
        requested_model="my-deployment",
        provenance={
            "client_kind": "openai",
            "reported_model": "gpt-5.5",
            "response_id": "resp_1",
            "endpoint": "https://res.openai.azure.com/openai/v1/",
        },
    )
    assert ok.ok

    empty = verify_turn_provenance(
        provider=CloudProvider.AZURE,
        requested_model="my-deployment",
        provenance={
            "client_kind": "openai",
            "reported_model": "",
            "response_id": "resp_1",
            "endpoint": "https://res.openai.azure.com/openai/v1/",
        },
    )
    assert not empty.ok


def test_scripted_passes_but_is_flagged() -> None:
    verdict = verify_turn_provenance(
        provider=CloudProvider.AZURE,
        requested_model="scripted",
        provenance={"client_kind": "scripted", "reported_model": "scripted"},
    )
    assert verdict.ok
    assert verdict.scripted


# ---- orchestrator enforcement ------------------------------------------------


class _Router:
    def __init__(self, events):
        self._events = events

    def stream_events(self, **kwargs):
        yield from self._events


class _Runtime:
    def __init__(self, *, ready: bool = True):
        self._ready = ready

    def ensure_server(self, selector):
        return True, "ok"

    def base_url(self):
        return LUMEN_ENDPOINT

    def status(self):
        return {"running": True, "ready": self._ready, "selector": "qwen3-5-9b:q4_0"}


def _cli(router, runtime) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            inference=SimpleNamespace(max_tokens=64, temperature=0.0, top_p=1.0),
            tools=SimpleNamespace(
                tools_enabled=False, tools_profile="off", tools_max_iterations=4
            ),
        ),
        cloud_router=router,
        lumen_runtime=runtime,
    )


def _conversation():
    return SimpleNamespace(
        conversation_id="conv-prov",
        messages=[SimpleNamespace(role=MessageRole.USER, content="hi")],
    )


def _run(router_events, *, runtime=None, target=None, collect=None):
    orchestrator = ToolingOrchestrator(cli=_cli(_Router(router_events), runtime or _Runtime()))
    return orchestrator.run_turn(
        user_input="hi",
        active_target=target or ActiveModelTarget.local("qwen3-5-9b:q4_0"),
        conversation=_conversation(),
        on_event=collect,
    )


def test_turn_with_matching_local_provenance_is_verified() -> None:
    result = _run(
        [TextDeltaEvent(delta="hello"), FinishEvent(reason="stop", provenance=_lumen_provenance())]
    )
    assert result.provenance_verified is True
    assert result.served_backend == "local"
    assert result.served_model_label == "qwen3-5-9b:q4_0"


def test_turn_reporting_wrong_model_is_rejected_loudly() -> None:
    seen: list = []
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        _run(
            [
                TextDeltaEvent(delta="hello"),
                FinishEvent(reason="stop", provenance=_lumen_provenance(model="evil-model")),
            ],
            collect=seen.append,
        )
    errors = [event for event in seen if isinstance(event, ErrorEvent)]
    assert errors and "evil-model" in errors[0].error


def test_turn_without_provenance_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="provenance"):
        _run([TextDeltaEvent(delta="hello"), FinishEvent(reason="stop")])


def test_cloud_turn_verifies_served_label() -> None:
    target = ActiveModelTarget.cloud(
        CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1")
    )
    result = _run(
        [
            TextDeltaEvent(delta="hello"),
            FinishEvent(
                reason="stop",
                provenance={
                    "client_kind": "openai",
                    "reported_model": "gpt-5.1-2026-01-15",
                    "response_id": "resp_9",
                    "endpoint": "https://api.openai.com/v1",
                },
            ),
        ],
        target=target,
    )
    assert result.provenance_verified is True
    assert result.served_backend == "cloud"
    assert result.served_model_label == "openai:gpt-5.1"


def test_scripted_turn_is_labeled_scripted() -> None:
    target = ActiveModelTarget.cloud(
        CloudModelRef(provider=CloudProvider.AZURE, model_id="scripted")
    )
    result = _run(
        [
            TextDeltaEvent(delta="canned"),
            FinishEvent(
                reason="stop",
                provenance={"client_kind": "scripted", "reported_model": "scripted"},
            ),
        ],
        target=target,
    )
    assert result.provenance_verified is True
    assert result.served_model_label == "azure:scripted (scripted)"
