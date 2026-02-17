from cortex.cloud.clients.openai_client import OpenAIClient


def _client_without_init() -> OpenAIClient:
    return OpenAIClient.__new__(OpenAIClient)


def test_normalize_messages_splits_system_and_keeps_dialogue():
    client = _client_without_init()
    system_text, messages = client._normalize_messages(
        [
            {"role": "system", "content": "s1"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "s2"},
            {"role": "tool", "content": "ignored"},
        ]
    )

    assert system_text == "s1\n\ns2"
    assert messages == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_extract_output_text_prefers_output_text_string():
    client = _client_without_init()

    class _Resp:
        output_text = "hello"

    assert client._extract_output_text(_Resp()) == "hello"


def test_extract_output_text_falls_back_to_output_items():
    client = _client_without_init()

    class _Part:
        text = "hello "

    class _Part2:
        text = "world"

    class _Item:
        content = [_Part(), _Part2()]

    class _Resp:
        output_text = None
        output = [_Item()]

    assert client._extract_output_text(_Resp()) == "hello world"


def test_build_response_kwargs_omits_sampling_controls_for_gpt5_models():
    client = _client_without_init()
    kwargs = client._build_response_kwargs(
        model_id="gpt-5-nano",
        request_input=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
        system_text=None,
    )

    assert kwargs["model"] == "gpt-5-nano"
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_build_response_kwargs_keeps_sampling_controls_for_non_gpt5_models():
    client = _client_without_init()
    kwargs = client._build_response_kwargs(
        model_id="gpt-4.1-mini",
        request_input=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
        system_text=None,
    )

    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.95
