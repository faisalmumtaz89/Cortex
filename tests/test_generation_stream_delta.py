from cortex.ui.generation import _coerce_stream_delta, _looks_like_repo_inspection_request


def test_coerce_stream_delta_accepts_plain_deltas():
    assembled = ""
    delta, assembled = _coerce_stream_delta("Hello", assembled)
    assert delta == "Hello"
    assert assembled == "Hello"

    delta, assembled = _coerce_stream_delta(" world", assembled)
    assert delta == " world"
    assert assembled == "Hello world"


def test_coerce_stream_delta_handles_cumulative_chunks():
    assembled = ""
    delta, assembled = _coerce_stream_delta("Hello", assembled)
    assert delta == "Hello"
    assert assembled == "Hello"

    # Provider emits cumulative text instead of pure delta.
    delta, assembled = _coerce_stream_delta("Hello world", assembled)
    assert delta == " world"
    assert assembled == "Hello world"


def test_coerce_stream_delta_ignores_repeated_snapshot():
    assembled = "Based on the current codebase, there are two main input field handlers."
    delta, updated = _coerce_stream_delta("Based on the current codebase, there are two main", assembled)
    assert delta == ""
    assert updated == assembled


def test_coerce_stream_delta_preserves_non_snapshot_chunks():
    assembled = "Hello world"
    delta, updated = _coerce_stream_delta("world!\n", assembled)
    assert delta == "world!\n"
    assert updated == "Hello worldworld!\n"


def test_repo_inspection_detector_matches_codebase_queries():
    assert _looks_like_repo_inspection_request(
        "evaluate the current codebase and find the function that handles input field"
    )
    assert _looks_like_repo_inspection_request("inspect repo files and find script")
    assert not _looks_like_repo_inspection_request("tell me a joke")
