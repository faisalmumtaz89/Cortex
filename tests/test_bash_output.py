import sys
import threading
import time

import pytest

from cortex.tooling.builtin import bash
from cortex.tooling.builtin.bash import _Output
from cortex.tooling.types import TurnInterrupt


def test_output_within_limits_is_kept_whole_and_decoded_in_one_piece():
    output = _Output()
    data = ("a" + "é" * 12500).encode()  # a character straddles the head's end
    for start in range(0, len(data), 4096):
        output.add(data[start : start + 4096])

    assert output.omitted == 0
    assert output.text() == "a" + "é" * 12500


@pytest.mark.parametrize("character", ["é", "😀"])
def test_long_output_keeps_exactly_its_start_and_end_cut_at_whole_characters(character):
    output = _Output()
    # The odd framing puts both cuts inside a character.
    data = b"a" + (character * 400_000).encode() + b"b"
    for start in range(0, len(data), 65536):
        output.add(data[start : start + 65536])

    assert len(output.head) == bash._HEAD_BYTES
    assert output.tail_size == bash._TAIL_BYTES
    assert output.omitted == len(data) - bash._HEAD_BYTES - bash._TAIL_BYTES
    head, _, tail = output.text().partition("bytes of output not shown ...]")
    assert "\ufffd" not in head and "\ufffd" not in tail
    assert f"[... {output.omitted} " in head
    assert tail.endswith(character + "b")


def test_carriage_returns_become_line_breaks():
    output = _Output()
    output.add(b"progress 10%\rprogress 100%\r\ndone\n")

    assert output.text() == "progress 10%\nprogress 100%\ndone\n"


def test_stopping_a_command_also_ends_children_that_ignore_sigterm(tmp_path):
    child = (
        f'"{sys.executable}" -c "import signal, time; '
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5); "
        "open('escaped.txt', 'w')\"; true"
    )
    interrupt = TurnInterrupt()
    threading.Timer(1.0, interrupt.set).start()

    _, _, ended_by = bash.run_command(child, cwd=tmp_path, timeout=60, interrupt=interrupt)

    assert ended_by == "interrupt"
    time.sleep(5)
    assert not (tmp_path / "escaped.txt").exists()


def test_interrupt_keeps_the_output_when_the_system_answers_eperm_for_a_dead_group(
    tmp_path, monkeypatch
):
    """macOS answers a signal to a process group that only has zombies left
    with EPERM, not ESRCH. That must read as "the group is gone", or the
    interrupted command's output is lost to an error."""
    real_killpg = bash.os.killpg

    def macos_killpg(pgid, signum):
        try:
            real_killpg(pgid, signum)
        except ProcessLookupError as exc:
            raise PermissionError(1, "Operation not permitted") from exc

    monkeypatch.setattr(bash.os, "killpg", macos_killpg)
    interrupt = TurnInterrupt()
    threading.Timer(1.0, interrupt.set).start()

    output, _, ended_by = bash.run_command(
        "echo started; sleep 4; touch finished.txt", cwd=tmp_path, timeout=60, interrupt=interrupt
    )

    assert ended_by == "interrupt"
    assert "started" in output.text()
    assert not (tmp_path / "finished.txt").exists()
