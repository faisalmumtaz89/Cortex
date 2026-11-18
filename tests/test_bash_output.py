import io
import os
import subprocess
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


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="reads /proc")
def test_a_group_left_with_only_zombies_counts_as_ended():
    """An exited child that nobody has reaped yet stays in its group as a
    zombie (one whose parent is init is reaped only when init gets to it):
    it runs nothing, so waiting for it would only delay Cortex."""
    ended = subprocess.Popen(["true"], start_new_session=True)
    running = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while _state(ended.pid) != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _state(ended.pid) == "Z"
        assert bash._signal_group(ended.pid, 0)  # still in its group...
        assert not bash._group_has_live_member(ended.pid)  # ...but not running
        assert bash._group_has_live_member(running.pid)
    finally:
        running.kill()
        running.wait()
        ended.wait()


@pytest.mark.parametrize(
    ("members", "live"),
    [
        ([("Z", 1)], False),  # only a zombie left
        ([("Z", 2)], True),  # its main thread ended, another thread runs on
        ([("Z", 1), ("S", 1)], True),
        ([], True),  # /proc shows none of a group that answers a signal
    ],
)
def test_a_group_is_live_while_a_member_runs_a_thread(monkeypatch, members, live):
    pgid = 4242
    stats = {
        str(5000 + i): f"{5000 + i} (a) b) {state} 1 {pgid} {' '.join(['0'] * 14)} {threads} 0 0"
        for i, (state, threads) in enumerate(members)
    }
    stats["7"] = f"7 (other) S 1 7 {' '.join(['0'] * 14)} 1 0 0"
    monkeypatch.setattr(bash.os, "listdir", lambda _path: [*stats, "self"])
    monkeypatch.setattr(
        bash, "open", lambda path, _mode: io.BytesIO(stats[path.split("/")[2]].encode()),
        raising=False,
    )
    assert bash._group_has_live_member(pgid) is live


def _state(pid: int) -> str:
    with open(f"/proc/{pid}/stat", "rb") as stat:
        return stat.read().rsplit(b")", 1)[1].split()[0].decode()


def test_a_signal_stops_commands_while_the_same_thread_registers_one():
    """A headless turn runs commands on the main thread, where the signal
    handler runs too: a signal between taking the lock and releasing it must
    not deadlock the handler. In a process of its own: a deadlock hangs it."""
    script = (
        "import os, signal, time\n"
        "from cortex.tooling.builtin import bash\n"
        "signal.signal(signal.SIGTERM, lambda signum, _: (bash.stop_running_commands(), os._exit(7)))\n"
        "with bash._running_lock:\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    time.sleep(10)\n"
    )
    process = subprocess.run([sys.executable, "-c", script], timeout=30, capture_output=True)
    assert process.returncode == 7, process.stderr


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
