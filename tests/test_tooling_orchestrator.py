from types import SimpleNamespace

from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider
from cortex.conversation_manager import MessageRole
from cortex.tooling.orchestrator import NO_TOOLS_SYSTEM_INSTRUCTION, ToolingOrchestrator
from cortex.tooling.types import FinishEvent, TextDeltaEvent


class _DummyCloudRouter:
    def __init__(self, events):
        self._events = events
        self.last_kwargs = None

    def stream_events(self, **kwargs):
        self.last_kwargs = kwargs
        for event in self._events:
            yield event


def _make_cli(*, cloud_router, tools_enabled=False, tools_profile="off"):
    return SimpleNamespace(
        config=SimpleNamespace(
            inference=SimpleNamespace(
                max_tokens=256,
                temperature=0.2,
                top_p=1.0,
                top_k=40,
                repetition_penalty=1.1,
                stream_output=True,
                seed=-1,
            ),
            tools=SimpleNamespace(
                tools_enabled=tools_enabled,
                tools_profile=tools_profile,
                tools_local_mode="disabled",
                tools_max_iterations=4,
            ),
        ),
        cloud_router=cloud_router,
        inference_engine=None,
        _format_prompt_with_chat_template=lambda user_input, include_user=False: user_input,
    )


def _make_conversation():
    return SimpleNamespace(
        conversation_id="conv-test",
        messages=[SimpleNamespace(role=MessageRole.USER, content="hello")],
    )


def test_orchestrator_deduplicates_cumulative_cloud_chunks():
    router = _DummyCloudRouter(
        events=[
            TextDeltaEvent(delta="Based on the current codebase"),
            TextDeltaEvent(delta="Based on the current codebase, there are two"),
            TextDeltaEvent(delta="Based on the current codebase, there are two main input field implementations."),
            FinishEvent(reason="stop"),
        ]
    )
    cli = _make_cli(cloud_router=router, tools_enabled=False, tools_profile="off")
    orchestrator = ToolingOrchestrator(cli=cli)
    target = ActiveModelTarget.cloud(CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1"))

    seen_deltas = []
    result = orchestrator.run_turn(
        user_input="analyze input field",
        active_target=target,
        conversation=_make_conversation(),
        on_event=lambda event: seen_deltas.append(getattr(event, "delta", None)),
    )

    assert result.text == "Based on the current codebase, there are two main input field implementations."
    assert [delta for delta in seen_deltas if isinstance(delta, str)] == [
        "Based on the current codebase",
        ", there are two",
        " main input field implementations.",
    ]


def test_orchestrator_prepends_no_tools_instruction_when_tools_off():
    router = _DummyCloudRouter(events=[TextDeltaEvent(delta="ok"), FinishEvent(reason="stop")])
    cli = _make_cli(cloud_router=router, tools_enabled=False, tools_profile="off")
    orchestrator = ToolingOrchestrator(cli=cli)
    target = ActiveModelTarget.cloud(CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1"))

    orchestrator.run_turn(
        user_input="hello",
        active_target=target,
        conversation=_make_conversation(),
    )

    assert router.last_kwargs is not None
    outbound_messages = router.last_kwargs["messages"]
    assert outbound_messages[0]["role"] == "system"
    assert outbound_messages[0]["content"] == NO_TOOLS_SYSTEM_INSTRUCTION


def test_orchestrator_prepends_tools_instruction_when_tools_on():
    router = _DummyCloudRouter(events=[TextDeltaEvent(delta="ok"), FinishEvent(reason="stop")])
    cli = _make_cli(cloud_router=router, tools_enabled=True, tools_profile="read_only")
    orchestrator = ToolingOrchestrator(cli=cli)
    target = ActiveModelTarget.cloud(CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1"))

    orchestrator.run_turn(
        user_input="inspect the repository and find input handler",
        active_target=target,
        conversation=_make_conversation(),
    )

    assert router.last_kwargs is not None
    outbound_messages = router.last_kwargs["messages"]
    assert outbound_messages[0]["role"] == "system"
    assert "local repository" in outbound_messages[0]["content"]
