from types import SimpleNamespace

from cortex.conversation_manager import MessageRole
from cortex.ui.cli_prompt import format_prompt_with_chat_template


class _DummyProfile:
    def __init__(self):
        self.last_messages = None

    def format_messages(self, messages, add_generation_prompt=True):
        self.last_messages = list(messages)
        return "formatted"


class _DummyTemplateRegistry:
    def __init__(self, profile):
        self.profile = profile

    def setup_model(self, *_args, **_kwargs):
        return self.profile


def test_local_prompt_context_uses_sliding_window_only():
    profile = _DummyProfile()
    template_registry = _DummyTemplateRegistry(profile)

    conversation_messages = [SimpleNamespace(role=MessageRole.SYSTEM, content="legacy system message")]
    for idx in range(30):
        role = MessageRole.USER if idx % 2 == 0 else MessageRole.ASSISTANT
        conversation_messages.append(SimpleNamespace(role=role, content=f"m{idx}"))

    conversation_manager = SimpleNamespace(
        get_current_conversation=lambda: SimpleNamespace(messages=conversation_messages)
    )
    model_manager = SimpleNamespace(current_model="dummy", tokenizers={})

    formatted = format_prompt_with_chat_template(
        conversation_manager=conversation_manager,
        model_manager=model_manager,
        template_registry=template_registry,
        user_input="ping",
        include_user=False,
    )

    assert formatted == "formatted"
    assert profile.last_messages is not None
    assert len(profile.last_messages) == 20
    assert profile.last_messages[0]["content"] == "m10"
    assert profile.last_messages[-1]["content"] == "m29"
