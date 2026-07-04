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
    LumenRuntime,
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


def _stub_server(tmp_path: Path, *, ready_delay: float = 0.2, die: bool = False) -> str:
    """A fake lumen-server: parses --port, optionally exits immediately,
    otherwise serves GET /v1/models after `ready_delay` seconds."""
    body = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import http.server, json, sys, time, threading

        args = sys.argv[1:]
        port = int(args[args.index("--port") + 1])
        model = args[args.index("--model") + 1]
        if {die!r}:
            print("boom: model load failed", flush=True)
            sys.exit(3)
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


def test_log_tail_dedupes_repeated_failure_line(tmp_path: Path) -> None:
    """Retried failed boots append the same log line; the surfaced tail must
    not read like a double failure ('boom | boom')."""
    log = tmp_path / "server.log"
    log.write_text("boom: model load failed\nboom: model load failed\n", encoding="utf-8")
    runtime = LumenRuntime(log_path=log)
    assert runtime._log_tail() == "boom: model load failed"
    log.write_text("first line\nsecond line\n", encoding="utf-8")
    assert runtime._log_tail() == "first line | second line"
