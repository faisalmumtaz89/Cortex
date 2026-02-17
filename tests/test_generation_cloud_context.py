from types import SimpleNamespace

from cortex.conversation_manager import MessageRole
from cortex.ui.generation import _build_cloud_messages


def test_cloud_message_builder_uses_sliding_window_only():
    messages = [SimpleNamespace(role=MessageRole.SYSTEM, content="system message")]
    for idx in range(40):
        role = MessageRole.USER if idx % 2 == 0 else MessageRole.ASSISTANT
        messages.append(SimpleNamespace(role=role, content=f"m{idx}"))

    cli = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_current_conversation=lambda: SimpleNamespace(messages=messages)
        )
    )

    cloud_messages = _build_cloud_messages(cli=cli)
    assert len(cloud_messages) == 30
    assert cloud_messages[0]["content"] == "m10"
    assert cloud_messages[-1]["content"] == "m39"
