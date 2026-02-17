from cortex.tooling.stream_normalizer import merge_stream_text


def test_merge_stream_text_handles_cumulative_snapshots():
    assembled = ""
    deltas = []
    for chunk in [
        "Based on the current codebase",
        "Based on the current codebase, there are two",
        "Based on the current codebase, there are two main functions.",
    ]:
        delta, assembled = merge_stream_text(chunk, assembled)
        if delta:
            deltas.append(delta)

    assert deltas == [
        "Based on the current codebase",
        ", there are two",
        " main functions.",
    ]
    assert assembled == "Based on the current codebase, there are two main functions."


def test_merge_stream_text_drops_long_retransmitted_snapshots():
    assembled = ""
    deltas = []
    for chunk in [
        "Based on the current codebase, there are two main input field handlers.",
        "Based on the current codebase, there are two main",
        "Based on the current codebase, there are two main input field handlers.",
        "Based on the current codebase, there are two main input field handlers. One is prompt_input_box.",
    ]:
        delta, assembled = merge_stream_text(chunk, assembled)
        if delta:
            deltas.append(delta)

    assert deltas == [
        "Based on the current codebase, there are two main input field handlers.",
        " One is prompt_input_box.",
    ]
    assert assembled == "Based on the current codebase, there are two main input field handlers. One is prompt_input_box."


def test_merge_stream_text_does_not_force_overlap_stitching():
    assembled = ""
    deltas = []
    for chunk in [
        "abcdef",
        "defghi",
    ]:
        delta, assembled = merge_stream_text(chunk, assembled)
        if delta:
            deltas.append(delta)

    assert deltas == ["abcdef", "defghi"]
    assert assembled == "abcdefdefghi"


def test_merge_stream_text_stitches_large_overlap_snapshots():
    assembled = ""
    deltas = []
    for chunk in [
        "Based on the current codebase, there are two main input field implementations",
        "there are two main input field implementations, one for CLI and one for Textual.",
    ]:
        delta, assembled = merge_stream_text(chunk, assembled)
        if delta:
            deltas.append(delta)

    assert deltas == [
        "Based on the current codebase, there are two main input field implementations",
        ", one for CLI and one for Textual.",
    ]
    assert assembled == (
        "Based on the current codebase, there are two main input field implementations"
        ", one for CLI and one for Textual."
    )


def test_merge_stream_text_preserves_similar_but_new_content():
    assembled = ""
    deltas = []
    for chunk in [
        "The function responsible is prompt_input_box in cortex/ui/input_box.py.",
        " The function is called from cortex/ui/cli.py and supports multiline rendering.",
        " It also uses _get_protected_input for raw key handling.",
    ]:
        delta, assembled = merge_stream_text(chunk, assembled)
        if delta:
            deltas.append(delta)

    assert deltas == [
        "The function responsible is prompt_input_box in cortex/ui/input_box.py.",
        " The function is called from cortex/ui/cli.py and supports multiline rendering.",
        " It also uses _get_protected_input for raw key handling.",
    ]
    assert assembled == (
        "The function responsible is prompt_input_box in cortex/ui/input_box.py."
        " The function is called from cortex/ui/cli.py and supports multiline rendering."
        " It also uses _get_protected_input for raw key handling."
    )


def test_merge_stream_text_ignores_whitespace_prefixed_retransmit():
    assembled = "Based on the current codebase, there are two main functions."
    delta, updated = merge_stream_text(
        "\n\nBased on the current codebase, there are two main functions.",
        assembled,
    )
    assert delta == ""
    assert updated == assembled
