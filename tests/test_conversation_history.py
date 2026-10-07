from types import SimpleNamespace

from cortex.conversation_manager import MessageRole
from cortex.tooling.orchestrator import conversation_history


def _conversation(*messages):
    return SimpleNamespace(
        messages=[
            SimpleNamespace(role=role, content=content, parts=parts)
            for role, content, parts in messages
        ]
    )


def _call(call_id, tool, arguments):
    return {"type": "tool", "state": "running", "call_id": call_id, "tool": tool,
            "input": arguments}


def _result(call_id, tool, output, ok=True, error=None):
    return {"type": "tool", "state": "completed" if ok else "error", "call_id": call_id,
            "tool": tool, "ok": ok, "output": output, "error": error}


def test_reply_steps_become_tool_calls_followed_by_their_results():
    history = conversation_history(_conversation(
        (MessageRole.USER, "fix it", None),
        (MessageRole.ASSISTANT, "Looking.Fixed.", [
            {"type": "text", "delta": "Looking."},
            _call("call_1", "read_file", {"path": "a.py"}),
            _result("call_1", "read_file", "x = 1"),
            _call("call_2", "edit_file", {"path": "a.py"}),
            _result("call_2", "edit_file", "", ok=False, error="denied"),
            {"type": "text", "delta": "Fixed."},
            {"type": "finish", "reason": "stop"},
        ]),
    ))

    assert history == [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "Looking.", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
            {"id": "call_2", "type": "function",
             "function": {"name": "edit_file", "arguments": '{"path": "a.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "x = 1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "error: denied"},
        {"role": "assistant", "content": "Fixed."},
    ]


def test_ids_and_names_take_a_form_every_provider_accepts():
    history = conversation_history(_conversation(
        (MessageRole.USER, "go", None),
        (MessageRole.ASSISTANT, "", [
            _call("functions.read:0", "repo_browser.open_file", {}),
            _result("functions.read:0", "repo_browser.open_file", "one"),
            _call("functions.read:0", "", {}),
            _result("functions.read:0", "", "two"),
            _call("call_9", "read_file", {}),
            _call("x" * 70, "read_file", {}),
            _result("x" * 70, "read_file", "three"),
            _call("x" * 64, "read_file", {}),
            _result("x" * 64, "read_file", "four"),
        ]),
    ))

    calls = history[1]["tool_calls"]
    assert [(c["id"], c["function"]["name"]) for c in calls] == [
        ("functions_read_0", "repo_browser_open_file"),
        ("functions_read_0_2", "unknown_tool"),
        ("x" * 64, "read_file"),
        ("x" * 62 + "_2", "read_file"),
    ]
    assert [(m["tool_call_id"], m["content"]) for m in history[2:]] == [
        ("functions_read_0", "one"),
        ("functions_read_0_2", "two"),
        ("x" * 64, "three"),
        ("x" * 62 + "_2", "four"),
    ]


def test_older_turns_keep_the_ends_of_long_results_and_arguments():
    body = "".join(f"line {index}\n" for index in range(1000))
    turns = [
        (MessageRole.USER, "write it", None),
        (MessageRole.ASSISTANT, "", [
            _call("call_1", "write_file", {"path": "big.txt", "content": body}),
            _result("call_1", "write_file", body),
        ]),
    ]
    for index in range(3):
        turns += [(MessageRole.USER, f"q{index}", None), (MessageRole.ASSISTANT, f"a{index}", [])]

    history = conversation_history(_conversation(*turns))

    arguments = history[1]["tool_calls"][0]["function"]["arguments"]
    for text in (arguments, history[2]["content"]):
        assert "line 0" in text and "line 999" in text
        assert "line 500" not in text
        assert "characters omitted" in text
        assert len(text) < 2000
    assert '"path": "big.txt"' in arguments
