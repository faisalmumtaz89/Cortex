from __future__ import annotations
from types import SimpleNamespace

from cortex.conversation_manager import MessageRole
from cortex.tooling.types import AssistantTurnResult, TextDeltaEvent
from cortex.ui import generation


class _ConversationManager:
    def __init__(self):
        self._messages = []

    def add_message(self, role, content, parts=None):
        self._messages.append(SimpleNamespace(role=role, content=content, parts=parts))

    def get_current_conversation(self):
        return SimpleNamespace(conversation_id="conv-test", messages=self._messages)


class _TemplateRegistry:
    def setup_model(self, *args, **kwargs):
        return None


class _ToolingOrchestrator:
    def run_turn(self, *, on_event=None, **kwargs):
        if on_event is not None:
            on_event(TextDeltaEvent(delta="Hello"))
            on_event(TextDeltaEvent(delta=" "))
            on_event(TextDeltaEvent(delta="world"))
        return AssistantTurnResult(text="Hello world", token_count=3, parts=[])


class _Console:
    is_terminal = False

    @staticmethod
    def print(text: str = "", end: str = "\n") -> None:
        print(text, end=end)


def _build_cli():
    return SimpleNamespace(
        config=SimpleNamespace(
            cloud=SimpleNamespace(cloud_enabled=False),
            inference=SimpleNamespace(
                max_tokens=256,
                temperature=0.2,
                top_p=1.0,
                top_k=40,
                repetition_penalty=1.1,
                stream_output=True,
                seed=-1,
            ),
            ui=SimpleNamespace(markdown_rendering=False, syntax_highlighting=True),
            tools=SimpleNamespace(
                tools_enabled=False,
                tools_profile="off",
                tools_local_mode="disabled",
                tools_max_iterations=4,
            ),
        ),
        model_manager=SimpleNamespace(current_model="dummy", tokenizers={}),
        template_registry=_TemplateRegistry(),
        conversation_manager=_ConversationManager(),
        tooling_orchestrator=_ToolingOrchestrator(),
        active_model_target=SimpleNamespace(backend="local", cloud_model=None),
        console=_Console(),
        generating=False,
        on_modal_prompt_start=None,
        on_modal_prompt_end=None,
        get_terminal_width=lambda: 120,
    )


def test_non_live_stream_prefix_printed_once(capsys):
    cli = _build_cli()
    generation.generate_response(cli=cli, user_input="hello")

    out = capsys.readouterr().out
    assert out.count("⏺") == 1
    assert "Hello world" in out
    assert cli.conversation_manager._messages[-1].role == MessageRole.ASSISTANT
