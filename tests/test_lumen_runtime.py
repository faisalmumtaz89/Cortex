"""Lumen runtime: catalog parsing, server lifecycle, and pull invocation.

Parsing is validated against byte-exact `lumen models` output captured from
lumen v0.2.0. Lifecycle tests spawn REAL subprocesses via stub binaries that
mimic lumen-server's observable behavior (bind port, serve /v1/models after a
delay), so start/readiness/switch/stop are exercised end to end without a
multi-gigabyte model load. An opt-in test (CORTEX_LUMEN_REAL=1) boots the real
engine.
"""

from __future__ import annotations

import json
import os
import stat
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest

from cortex.lumen_runtime import (
    SWITCH_IN_FLIGHT_PREFIX,
    LumenRuntime,
    LumenServerState,
    parse_models_output,
    parse_selector,
)

# Captured verbatim from `lumen models` (lumen v0.2.0, 2026-07-04).
REAL_MODELS_OUTPUT = """Cached models:

  qwen3-5-0.8b-Q4_0-draft-metal            1.2 GB
  qwen3-5-9b-Q4_0                          5.4 GB
  qwen3-5-9b-Q8_0                          8.9 GB

Available to download:
  qwen3-5-9b           Qwen3.5 9B BF16
  qwen3-5-moe-35b-a3b  Qwen3.5 MoE 35B-A3B Q8_0
  qwen3-5-moe-35b-a3b  Qwen3.5 MoE 35B-A3B BF16
  qwen3-5-moe-35b-a3b  Qwen3.5 MoE 35B-A3B Q4_0
  qwen3-6-27b          Qwen3.6 27B Q4_0
  qwen3-6-27b          Qwen3.6 27B Q8_0
  qwen3-6-27b          Qwen3.6 27B BF16

Download with: lumen pull <model-name> [--quant Q8_0]
"""


def test_parse_models_output_real_capture() -> None:
    models = parse_models_output(REAL_MODELS_OUTPUT)

    cached = [m for m in models if m.cached]
    available = [m for m in models if not m.cached]

    # Draft models are filtered; both real cached quants survive.
    assert [m.selector for m in cached] == ["qwen3-5-9b:q4_0", "qwen3-5-9b:q8_0"]
    assert cached[0].size == "5.4 GB"

    assert len(available) == 7
    assert ("qwen3-5-moe-35b-a3b", "Q4_0") in [(m.name, m.quant) for m in available]
    assert available[0].display_name.startswith("Qwen3.5 9B")


def test_parse_selector_forms() -> None:
    assert parse_selector("qwen3-5-9b:q4_0") == ("qwen3-5-9b", "Q4_0")
    assert parse_selector("qwen3-5-9b") == ("qwen3-5-9b", "Q8_0")
    assert parse_selector(" qwen3-6-27b:BF16 ") == ("qwen3-6-27b", "BF16")


def _write_script(path: Path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _stub_server(
    tmp_path: Path, *, ready_delay: float = 0.2, die: bool = False, exit_delay: float = 0.0
) -> str:
    """A fake lumen-server: parses --port, optionally exits immediately,
    otherwise serves GET /v1/models after `ready_delay` seconds. Every
    instance drops a `<pid>.pid` file into `<tmp_path>/pids/` so tests can
    assert no process outlives the runtime. `exit_delay` simulates a slow GPU
    teardown by delaying the SIGTERM exit."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir(exist_ok=True)
    body = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import http.server, json, os, signal, sys, time

        args = sys.argv[1:]
        port = int(args[args.index("--port") + 1])
        model = args[args.index("--model") + 1]
        open({str(pid_dir)!r} + f"/{{os.getpid()}}.pid", "w").write(str(port))
        if {die!r}:
            print("boom: model load failed", flush=True)
            sys.exit(3)
        if {exit_delay!r}:
            def _slow_exit(signum, frame):
                time.sleep({exit_delay})
                os._exit(0)
            signal.signal(signal.SIGTERM, _slow_exit)
        time.sleep({ready_delay})

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/v1/models":
                    payload = json.dumps(
                        {{"object": "list", "data": [{{"id": model, "object": "model"}}]}}
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *a):
                pass

        http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
        """
    )
    return _write_script(tmp_path / "fake-lumen-server", body)


def _stub_cli(tmp_path: Path) -> str:
    """A fake `lumen` CLI: `models` prints the captured output; `pull` records
    its args and prints progress lines."""
    fixture = tmp_path / "models.txt"
    fixture.write_text(REAL_MODELS_OUTPUT, encoding="utf-8")
    record = tmp_path / "pull-args.json"
    body = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import json, sys
        if sys.argv[1] == "models":
            print(open({str(fixture)!r}).read(), end="")
        elif sys.argv[1] == "pull":
            json.dump(sys.argv[1:], open({str(record)!r}, "w"))
            print("downloading 10%")
            print("downloading 100%")
            print("converted to LBC")
        """
    )
    return _write_script(tmp_path / "fake-lumen", body)


def test_list_models_via_stub_cli(tmp_path: Path) -> None:
    runtime = LumenRuntime(binary=_stub_cli(tmp_path), server_binary="missing-server")
    models = runtime.list_models()
    assert [m.selector for m in models if m.cached] == [
        "qwen3-5-9b:q4_0",
        "qwen3-5-9b:q8_0",
    ]


def test_missing_binary_reports_install_hint(tmp_path: Path) -> None:
    runtime = LumenRuntime(
        binary="definitely-not-lumen", server_binary="definitely-not-lumen-server"
    )
    ok, message = runtime.available()
    assert ok is False
    assert "servelumen.com" in message
    started, why = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert started is False
    assert "servelumen.com" in why


def test_server_lifecycle_start_switch_stop(tmp_path: Path) -> None:
    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path),
        startup_timeout_seconds=15,
        log_path=tmp_path / "server.log",
    )

    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok, message
    first = runtime.server
    assert first is not None
    base = runtime.base_url()
    assert base is not None

    # The managed endpoint answers with the model we started.
    with urllib.request.urlopen(f"{base}/models", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["data"][0]["id"] == "qwen3-5-9b"
    assert runtime.active_selector() == "qwen3-5-9b:q4_0"

    # Same selector → no restart.
    ok, _ = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok
    assert runtime.server is first

    # Different selector → old process replaced by a new one.
    ok, message = runtime.ensure_server("qwen3-5-9b:q8_0")
    assert ok, message
    second = runtime.server
    assert second is not None and second.process.pid != first.process.pid
    assert first.process.poll() is not None  # old server terminated
    assert runtime.active_selector() == "qwen3-5-9b:q8_0"

    runtime.stop()
    assert runtime.server is None
    assert second.process.poll() is not None
    assert runtime.base_url() is None
    assert runtime.status() == {"running": False}


def test_server_crash_surfaces_log_tail(tmp_path: Path) -> None:
    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path, die=True),
        startup_timeout_seconds=10,
        log_path=tmp_path / "server.log",
    )
    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok is False
    assert "exited with code 3" in message
    assert "boom" in message  # the log tail is surfaced to the user
    assert runtime.server is None


def test_pull_invokes_lumen_with_yes_and_streams_lines(tmp_path: Path) -> None:
    runtime = LumenRuntime(binary=_stub_cli(tmp_path), server_binary="missing")
    lines: list[str] = []
    ok, message = runtime.pull("qwen3-6-27b:q4_0", on_line=lines.append)
    assert ok, message
    assert lines == ["downloading 10%", "downloading 100%", "converted to LBC"]
    assert message == "converted to LBC"
    recorded = json.loads((tmp_path / "pull-args.json").read_text(encoding="utf-8"))
    assert recorded == ["pull", "qwen3-6-27b", "--quant", "Q4_0", "--yes"]


@pytest.mark.skipif(
    os.environ.get("CORTEX_LUMEN_REAL") != "1",
    reason="opt-in: real lumen engine (set CORTEX_LUMEN_REAL=1; needs a cached model)",
)
def test_real_lumen_server_end_to_end() -> None:
    runtime = LumenRuntime(startup_timeout_seconds=180)
    models = runtime.list_models()
    cached = [m for m in models if m.cached]
    assert cached, "no cached lumen models on this machine"
    try:
        ok, message = runtime.ensure_server(cached[0].selector)
        assert ok, message
        base = runtime.base_url()
        assert base is not None
        with urllib.request.urlopen(f"{base}/models", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["data"], payload
    finally:
        runtime.stop()


def test_concurrent_ensure_server_waits_for_inflight_boot(tmp_path: Path) -> None:
    """A second caller during a boot must WAIT for it (one process, both ok) —
    this is the turn-during-boot path."""
    import threading

    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path, ready_delay=1.2),
        startup_timeout_seconds=20,
        log_path=tmp_path / "server.log",
    )
    results: dict[str, tuple] = {}

    def _first() -> None:
        results["first"] = runtime.ensure_server("qwen3-5-9b:q4_0")

    thread = threading.Thread(target=_first)
    thread.start()
    # Give the first caller time to spawn the (slow-ready) server process.
    deadline = time.time() + 5
    while time.time() < deadline and runtime.starting_selector() is None:
        time.sleep(0.05)
    assert runtime.starting_selector() == "qwen3-5-9b:q4_0"
    assert runtime.serving_selector() is None
    assert runtime.status().get("ready") is False

    second = runtime.ensure_server("qwen3-5-9b:q4_0")  # waits, no double-start
    thread.join(timeout=20)

    assert results["first"][0] is True, results["first"][1]
    assert second[0] is True, second[1]
    assert runtime.serving_selector() == "qwen3-5-9b:q4_0"
    assert runtime.status().get("ready") is True
    first_state = runtime.server
    assert first_state is not None
    runtime.stop()


def test_turn_during_switch_is_refused_and_incoming_boot_survives(tmp_path: Path) -> None:
    """THE mid-switch race: a turn still bound to the OUTGOING model must be
    refused with a clear retry message while a switch's boot is in flight —
    never tear the incoming server down. (Previously the racing turn stopped
    the booting server; that boot's failure cleanup then stopped the turn's
    replacement — mutual SIGTERM, zero servers left.)"""
    import threading

    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path, ready_delay=1.5),
        startup_timeout_seconds=20,
        log_path=tmp_path / "server.log",
    )
    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok, message
    outgoing = runtime.server
    assert outgoing is not None

    results: dict[str, tuple] = {}

    def _switch() -> None:
        results["switch"] = runtime.ensure_server("qwen3-5-9b:q8_0")

    thread = threading.Thread(target=_switch)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and runtime.starting_selector() != "qwen3-5-9b:q8_0":
        time.sleep(0.05)
    assert runtime.starting_selector() == "qwen3-5-9b:q8_0"
    incoming = runtime.server
    assert incoming is not None

    # The racing "turn" targets the outgoing model mid-boot: clean refusal.
    turn_ok, turn_message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert turn_ok is False
    assert turn_message.startswith(SWITCH_IN_FLIGHT_PREFIX)
    assert "qwen3-5-9b:q8_0" in turn_message
    assert "retry in a moment" in turn_message

    thread.join(timeout=20)
    assert results["switch"][0] is True, results["switch"][1]
    # Clean end state: the INCOMING server is up; exactly one process alive.
    assert runtime.serving_selector() == "qwen3-5-9b:q8_0"
    assert runtime.server is incoming
    assert incoming.process.poll() is None
    assert outgoing.process.poll() is not None  # replaced by the switch itself
    runtime.stop()
    assert incoming.process.poll() is not None


def test_failed_boot_cleanup_discards_only_its_own_state(tmp_path: Path) -> None:
    """A failed boot's cleanup must terminate ITS process only — never the
    server a concurrent caller installed since (targeted discard, not a global
    stop). This is the second half of the mutual-termination fix."""
    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path),
        startup_timeout_seconds=15,
        log_path=tmp_path / "server.log",
    )
    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok, message
    current = runtime.server
    assert current is not None

    # A stale failed-boot state (its process already dead, as after a crash).
    import subprocess as sp
    import sys

    dead = sp.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=10)
    stale = LumenServerState(process=dead, selector="qwen3-5-9b:q8_0", port=1)

    runtime._discard(stale)

    # The live server installed by the other caller is untouched.
    assert runtime.server is current
    assert current.process is not None and current.process.poll() is None
    assert runtime.serving_selector() == "qwen3-5-9b:q4_0"
    runtime.stop()


def _live_stub_pids(tmp_path: Path) -> list[int]:
    """PIDs of stub servers (recorded via their pid files) still alive."""
    pids: list[int] = []
    pid_dir = tmp_path / "pids"
    if not pid_dir.exists():
        return pids
    for pid_file in pid_dir.glob("*.pid"):
        pid = int(pid_file.stem)
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        pids.append(pid)
    return pids


def test_failed_spawn_prep_discards_claim_and_next_boot_succeeds(tmp_path: Path) -> None:
    """An exception BETWEEN the boot claim and the spawn (here: an unwritable
    log path) must discard the placeholder — a wedged not-ready claim would
    refuse every future boot against a startup nothing will ever finish."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file where the log DIRECTORY should be", encoding="utf-8")
    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path),
        startup_timeout_seconds=15,
        log_path=blocker / "server.log",  # parent is a FILE → mkdir raises
    )
    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok is False
    assert "Failed to launch lumen-server" in message
    assert runtime.server is None  # claim discarded, runtime not wedged

    # The runtime recovers fully once the obstacle is gone.
    runtime.log_path = tmp_path / "server.log"
    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok, message
    assert runtime.serving_selector() == "qwen3-5-9b:q4_0"
    runtime.stop()
    assert _live_stub_pids(tmp_path) == []


def test_switch_teardown_does_not_block_concurrent_accessors(tmp_path: Path) -> None:
    """Replacing a READY server must not stall pointer reads: the switch
    claims its placeholder under the lock and reaps the outgoing server
    OUTSIDE it, so status()/selector reads (the UI's data sources) and a
    racing turn's refusal answer instantly even while the old process drags
    out its SIGTERM exit."""
    import threading

    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path, ready_delay=0.2, exit_delay=3.0),
        startup_timeout_seconds=20,
        log_path=tmp_path / "server.log",
    )
    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok, message
    outgoing = runtime.server
    assert outgoing is not None

    results: dict[str, tuple] = {}
    thread = threading.Thread(
        target=lambda: results.update(switch=runtime.ensure_server("qwen3-5-9b:q8_0"))
    )
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and runtime.starting_selector() != "qwen3-5-9b:q8_0":
        time.sleep(0.02)
    assert runtime.starting_selector() == "qwen3-5-9b:q8_0"
    # The claim is already visible while the outgoing process is mid-teardown.
    assert outgoing.process is not None and outgoing.process.poll() is None

    # Pointer reads and the racing turn's refusal must answer instantly (never
    # queued behind the outgoing terminate()+wait()).
    for accessor in (runtime.status, runtime.serving_selector, runtime.base_url):
        started = time.time()
        accessor()
        assert time.time() - started < 0.5, f"{accessor.__name__} blocked on teardown"
    started = time.time()
    turn_ok, turn_message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert time.time() - started < 0.5, "racing turn blocked on teardown"
    assert turn_ok is False
    assert turn_message.startswith(SWITCH_IN_FLIGHT_PREFIX)

    thread.join(timeout=30)
    assert results["switch"][0] is True, results["switch"][1]
    assert runtime.serving_selector() == "qwen3-5-9b:q8_0"
    runtime.stop()
    assert _live_stub_pids(tmp_path) == []


def test_stop_during_switch_claim_window_leaves_no_processes(tmp_path: Path) -> None:
    """A stop() (user quit / signal path) landing inside the switch's claim
    window must win: the switch aborts, the freshly spawned process never
    outlives the runtime, and no lumen-server survives."""
    import threading

    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path, ready_delay=0.2, exit_delay=3.0),
        startup_timeout_seconds=20,
        log_path=tmp_path / "server.log",
    )
    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok, message

    results: dict[str, tuple] = {}
    thread = threading.Thread(
        target=lambda: results.update(switch=runtime.ensure_server("qwen3-5-9b:q8_0"))
    )
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and runtime.starting_selector() != "qwen3-5-9b:q8_0":
        time.sleep(0.02)
    assert runtime.starting_selector() == "qwen3-5-9b:q8_0"

    runtime.stop()  # lands while the switch is still reaping the outgoing server

    thread.join(timeout=30)
    assert results["switch"][0] is False
    assert "stopped during startup" in results["switch"][1]
    assert runtime.server is None
    assert _live_stub_pids(tmp_path) == []


def test_same_selector_caller_coalesces_during_switch_claim(tmp_path: Path) -> None:
    """A second caller asking for the INCOMING selector while the switch is
    still reaping the outgoing server must wait for that boot instead of
    double-starting: exactly one process per selector ever spawns."""
    import threading

    runtime = LumenRuntime(
        binary=_stub_cli(tmp_path),
        server_binary=_stub_server(tmp_path, ready_delay=0.5, exit_delay=2.0),
        startup_timeout_seconds=20,
        log_path=tmp_path / "server.log",
    )
    ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
    assert ok, message

    results: dict[str, tuple] = {}
    switch = threading.Thread(
        target=lambda: results.update(switch=runtime.ensure_server("qwen3-5-9b:q8_0"))
    )
    switch.start()
    deadline = time.time() + 5
    while time.time() < deadline and runtime.starting_selector() != "qwen3-5-9b:q8_0":
        time.sleep(0.02)
    assert runtime.starting_selector() == "qwen3-5-9b:q8_0"

    coalesced = runtime.ensure_server("qwen3-5-9b:q8_0")  # waits for that boot
    switch.join(timeout=30)
    assert results["switch"][0] is True, results["switch"][1]
    assert coalesced == (True, "Lumen already serving qwen3-5-9b:q8_0.")
    assert runtime.serving_selector() == "qwen3-5-9b:q8_0"
    # Two boots total (q4_0, then q8_0): the coalesced caller spawned nothing.
    assert len(list((tmp_path / "pids").glob("*.pid"))) == 2
    runtime.stop()
    assert _live_stub_pids(tmp_path) == []


@pytest.mark.skipif(
    os.environ.get("CORTEX_LUMEN_REAL") != "1",
    reason="opt-in: real lumen engine (set CORTEX_LUMEN_REAL=1; needs cached 9B q4_0+q8_0)",
)
def test_real_lumen_switch_race_regression() -> None:
    """Real-engine regression for the mid-switch turn race: boot 9B q4_0,
    start a switch to q8_0, immediately fire a racing turn-style
    ensure_server(q4_0) — the turn must be refused cleanly, the switch must
    complete, and exactly one server survives. SHORT session; stops on exit."""
    import threading

    runtime = LumenRuntime(startup_timeout_seconds=180)
    try:
        ok, message = runtime.ensure_server("qwen3-5-9b:q4_0")
        assert ok, message
        assert runtime.serving_selector() == "qwen3-5-9b:q4_0"

        results: dict[str, tuple] = {}
        thread = threading.Thread(
            target=lambda: results.update(switch=runtime.ensure_server("qwen3-5-9b:q8_0"))
        )
        thread.start()
        deadline = time.time() + 30
        while time.time() < deadline and runtime.starting_selector() != "qwen3-5-9b:q8_0":
            time.sleep(0.05)
        assert runtime.starting_selector() == "qwen3-5-9b:q8_0"

        # Immediate racing turn bound to the outgoing model.
        turn_ok, turn_message = runtime.ensure_server("qwen3-5-9b:q4_0")
        assert turn_ok is False, turn_message
        assert turn_message.startswith(SWITCH_IN_FLIGHT_PREFIX), turn_message

        thread.join(timeout=240)
        assert results["switch"][0] is True, results["switch"][1]
        assert runtime.serving_selector() == "qwen3-5-9b:q8_0"
        state = runtime.server
        assert state is not None and state.process.poll() is None
    finally:
        runtime.stop()
        final = runtime.server
        assert final is None


def test_log_tail_dedupes_repeated_failure_line(tmp_path: Path) -> None:
    """Retried failed boots append the same log line; the surfaced tail must
    not read like a double failure ('boom | boom')."""
    log = tmp_path / "server.log"
    log.write_text("boom: model load failed\nboom: model load failed\n", encoding="utf-8")
    runtime = LumenRuntime(log_path=log)
    assert runtime._log_tail() == "boom: model load failed"
    log.write_text("first line\nsecond line\n", encoding="utf-8")
    assert runtime._log_tail() == "first line | second line"
