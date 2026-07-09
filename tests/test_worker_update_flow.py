"""Worker-stdio e2e tests for the /update command family and the startup
update notice.

Fully hermetic: the GitHub "releases/latest" redirect probe AND the release
wheel assets are served by a local stub HTTP server (CORTEX_UPDATE_PROBE_BASE
is the shared origin for discovery and downloads, like github.com), the Lumen
installer is a stub bash script injected via CORTEX_LUMEN_INSTALLER_URL (a
local file path — the downloader's test seam), pip for the cortex self-update
is a stub recorder (CORTEX_SELF_PIP — a real pip would write the suite's own
venv), and `lumen` itself is a stub whose --version is backed by a rewritable
version file, so the installer can observably "replace the binary". No real
network, no real Lumen, no GPU."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from cortex.update_check import installed_cortex_version

REPO_ROOT = Path(__file__).resolve().parents[1]

LUMEN_LISTING = """Cached models:

  qwen3-5-9b-Q4_0                          5.4 GB

Available to download:
  qwen3-5-9b           Qwen3.5 9B Q8_0
  qwen3-6-27b          Qwen3.6 27B Q4_0

Download with: lumen pull <model-name> [--quant Q8_0]
"""


# ---- local stub of GitHub's releases/latest redirect ----------------------


@contextmanager
def _probe_server(
    *,
    lumen_tag: str | None = "v0.4.0",
    cortex_tag: str | None = None,
    assets: dict[str, bytes] | None = None,
):
    """Serves /{repo}/releases/latest with a 302 → …/releases/tag/<tag>
    (GitHub's real behavior) or 404 when the repo has no releases. ``assets``
    optionally maps releases/download/... paths to bytes, so the SAME origin
    also serves release assets — exactly like github.com."""
    tags = {
        "/faisalmumtaz89/Lumen/releases/latest": lumen_tag,
        "/faisalmumtaz89/Cortex/releases/latest": cortex_tag,
    }
    asset_map = assets or {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            tag = tags.get(self.path)
            body = asset_map.get(self.path)
            if tag:
                self.send_response(302)
                self.send_header(
                    "Location", f"https://github.com/example/repo/releases/tag/{tag}"
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif body is not None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# ---- worker environment ----------------------------------------------------


def _update_stub_env(
    tmp_path: Path,
    *,
    probe_base: str,
    with_server: bool = False,
    boot_delay: float = 0.0,
    pull_delay: float = 0.0,
    installer_sleep: float = 0.0,
    installer_new_version: str = "0.4.0",
) -> dict:
    """Worker env with a version-file-backed fake `lumen`, a stub installer
    script, and the probe pointed at the local stub server."""
    version_file = tmp_path / "lumen-version.txt"
    version_file.write_text("0.3.0", encoding="utf-8")
    listing_file = tmp_path / "lumen-models.txt"
    listing_file.write_text(LUMEN_LISTING, encoding="utf-8")

    cli = tmp_path / "fake-lumen"
    cli.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "if sys.argv[1:2] == ['--version']:\n"
        f"    print('lumen v' + open({str(version_file)!r}).read().strip())\n"
        "elif sys.argv[1:2] == ['models']:\n"
        f"    print(open({str(listing_file)!r}).read(), end='')\n"
        "elif sys.argv[1:2] == ['pull']:\n"
        "    print('downloading 50%', flush=True)\n"
        f"    time.sleep({pull_delay!r})\n"
        "    print('converted to LBC', flush=True)\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)

    installer_record = tmp_path / "installer-env.txt"
    installer = tmp_path / "stub-lumen-install.sh"
    installer.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        f'echo "LUMEN_TAG=${{LUMEN_TAG:-}}" > {installer_record}\n'
        f'echo "LUMEN_MODEL=${{LUMEN_MODEL:-}}" >> {installer_record}\n'
        f'echo "LUMEN_QUANT=${{LUMEN_QUANT:-}}" >> {installer_record}\n'
        f'echo "STDIN=$(stat -f %HT /dev/fd/0 2>/dev/null || echo unknown)" >> {installer_record}\n'
        'echo "Installing Lumen ${LUMEN_TAG:-unknown}..."\n'
        f"sleep {installer_sleep}\n"
        f'printf "%s" "{installer_new_version}" > {version_file}\n'
        'echo "Binaries installed."\n',
        encoding="utf-8",
    )
    installer.chmod(0o755)

    # The cortex self-update's pip (CORTEX_SELF_PIP): a recorder, ALWAYS
    # stubbed — a real pip would install the downloaded wheel into the
    # suite's own venv. Records argv, the wheel's sha256, and stdin type.
    pip_record = tmp_path / "pip-args.txt"
    stub_pip = tmp_path / "stub-pip"
    stub_pip.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "ARGS=$*" > {pip_record}\n'
        f"shasum -a 256 \"$2\" | awk '{{print \"SHA256=\"$1}}' >> {pip_record}\n"
        f'echo "STDIN=$(stat -f %HT /dev/fd/0 2>/dev/null || echo unknown)" >> {pip_record}\n'
        'echo "Successfully installed cortex-llm"\n',
        encoding="utf-8",
    )
    stub_pip.chmod(0o755)

    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env["CORTEX_LUMEN_BINARY"] = str(cli)
    env["CORTEX_UPDATE_PROBE_BASE"] = probe_base
    env["CORTEX_LUMEN_INSTALLER_URL"] = str(installer)
    env["CORTEX_SELF_PIP"] = str(stub_pip)
    env["CORTEX_AUTO_UPDATE_CHECK"] = "false"  # notice test opts back in
    env.pop("CORTEX_SCRIPTED_MODEL", None)
    # The worker runs FROM this repo — a source checkout, which /update
    # cortex refuses. Tests of the normal wheel path opt in to "installed".
    env.pop("CORTEX_SELF_INSTALL_KIND", None)

    if with_server:
        server = tmp_path / "fake-lumen-server"
        server.write_text(
            "#!/usr/bin/env python3\n"
            "import http.server, json, sys, time\n"
            "args = sys.argv[1:]\n"
            "port = int(args[args.index('--port') + 1])\n"
            "model = args[args.index('--model') + 1]\n"
            f"time.sleep({boot_delay!r})\n"
            "class H(http.server.BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        body = json.dumps({'object': 'list', 'data': [{'id': model}]}).encode()\n"
            "        self.send_response(200 if self.path == '/v1/models' else 404)\n"
            "        self.send_header('Content-Length', str(len(body)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(body)\n"
            "    def log_message(self, *a):\n"
            "        pass\n"
            "http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()\n",
            encoding="utf-8",
        )
        server.chmod(0o755)
        env["CORTEX_LUMEN_SERVER_BINARY"] = str(server)
    else:
        env["CORTEX_LUMEN_SERVER_BINARY"] = str(tmp_path / "missing-lumen-server")
    return env


def _installer_record(tmp_path: Path) -> dict[str, str]:
    text = (tmp_path / "installer-env.txt").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.strip().splitlines())


def _pip_record(tmp_path: Path) -> dict[str, str]:
    text = (tmp_path / "pip-args.txt").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.strip().splitlines())


def _cortex_release_assets(tag: str, wheel_bytes: bytes) -> tuple[str, dict[str, bytes]]:
    """(wheel asset name, asset paths → bytes) for a stub Cortex release."""
    wheel_name = f"cortex_llm-{tag.lstrip('v')}-py3-none-macosx_13_0_arm64.whl"
    prefix = f"/faisalmumtaz89/Cortex/releases/download/{tag}"
    digest = hashlib.sha256(wheel_bytes).hexdigest()
    return wheel_name, {
        f"{prefix}/{wheel_name}": wheel_bytes,
        f"{prefix}/{wheel_name}.sha256": f"{digest}  {wheel_name}\n".encode("utf-8"),
    }


def _worker_session(env: dict):
    """Spawn a worker; returns (process, send, recv_until, read_events_until)."""
    process = subprocess.Popen(
        [sys.executable, "-m", "cortex", "--worker-stdio"],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None

    def send(request_id: int, method: str, params: dict) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def recv_until(request_id: int, timeout: float = 60.0):
        deadline = time.time() + timeout
        events = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                events.append(frame)
                continue
            if frame.get("id") == request_id:
                return frame, events
        raise AssertionError(f"timed out waiting for response id={request_id}")

    def read_events_until(predicate, timeout: float = 60.0):
        deadline = time.time() + timeout
        seen = []
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            frame = json.loads(line)
            if frame.get("method") == "event":
                seen.append(frame)
                if predicate(frame):
                    return seen
        raise AssertionError(f"timed out waiting for event; saw {len(seen)} events")

    return process, send, recv_until, read_events_until


def _bootstrap(send, recv_until) -> str:
    send(1, "app.handshake", {"protocol_version": "1.0.0"})
    recv_until(1)
    send(2, "session.create_or_resume", {})
    response, _events = recv_until(2)
    return response["result"]["session_id"]


def _final_update_frame(frame: dict) -> bool:
    params = frame["params"]
    if params["event_type"] != "message.updated":
        return False
    payload = params["payload"]
    progress = payload.get("progress")
    return (
        isinstance(progress, dict)
        and progress.get("kind") == "engine-update"
        and payload.get("final") is True
    )


# ---- tests ---------------------------------------------------------------------


def test_update_status_reports_installed_vs_latest(tmp_path: Path) -> None:
    with _probe_server(lumen_tag="v0.4.0", cortex_tag=None) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update"})
            response, events = recv_until(3)
            result = response["result"]
            assert result["ok"] is True
            message = str(result["message"])
            assert "Lumen: 0.3.0 installed · 0.4.0 available — /update lumen" in message
            assert (
                f"Cortex: {installed_cortex_version()} installed · no published releases yet"
                in message
            )
            # Synchronous command: busy/idle status plus the notice.
            statuses = [
                event["params"]["payload"].get("status")
                for event in events
                if event["params"]["event_type"] == "session.status"
            ]
            assert "busy" in statuses and "idle" in statuses
            assert any(
                event["params"]["event_type"] == "system.notice"
                and "0.4.0 available" in str(event["params"]["payload"].get("message", ""))
                for event in events
            )
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_update_lumen_background_flow_end_to_end(tmp_path: Path) -> None:
    """/update lumen: stops the running managed server, runs the (stub)
    installer pinned via env, narrates engine-update frames, verifies the new
    version, and resolves in place with the exact final message."""
    with _probe_server(lumen_tag="v0.4.0") as base:
        env = _update_stub_env(tmp_path, probe_base=base, with_server=True)
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)

            # Boot the cached model so a managed server is genuinely running.
            send(3, "command.execute", {"session_id": session_id, "command": "/model qwen3-5-9b:q4_0"})
            boot_response, _ = recv_until(3)
            assert boot_response["result"]["ok"] is True
            read_events_until(
                lambda frame: frame["params"]["event_type"] == "message.updated"
                and frame["params"]["payload"].get("final") is True
                and "ready — now active." in str(frame["params"]["payload"].get("content", ""))
            )

            send(4, "command.execute", {"session_id": session_id, "command": "/update lumen"})
            update_response, update_events = recv_until(4)
            result = update_response["result"]
            assert result["ok"] is True, result
            assert result["background"] is True
            assert result["repo_id"] == "lumen"
            assert "Updating Lumen to 0.4.0" in str(result["message"])

            seen = read_events_until(_final_update_frame)
            all_events = [*update_events, *seen]
            # Background op: never touches session.status.
            assert not any(
                event["params"]["event_type"] == "session.status" for event in all_events
            ), "background update leaked a session.status event"

            progress_frames = [
                event["params"]["payload"]
                for event in all_events
                if event["params"]["event_type"] == "message.updated"
                and isinstance(event["params"]["payload"].get("progress"), dict)
            ]
            assert progress_frames, "expected engine-update narration frames"
            message_ids = {payload["message_id"] for payload in progress_frames}
            assert len(message_ids) == 1, "one operation = one transcript message"
            for payload in progress_frames:
                assert payload["progress"]["kind"] == "engine-update"
                assert payload["progress"]["repo_id"] == "lumen"
            phases = [str(payload["progress"]["phase"]) for payload in progress_frames]
            assert any("stopped lumen-server (qwen3-5-9b:q4_0)" in phase for phase in phases)
            assert any("Installing Lumen v0.4.0" in phase for phase in phases)
            assert "verifying installed version" in phases
            final_payload = progress_frames[-1]
            assert final_payload["final"] is True
            assert final_payload["progress"]["phase"] == "ready"
            assert (
                final_payload["content"]
                == "local · Lumen 0.4.0 installed — server restarts on next use."
            )

            # The installer ran pinned (tag + already-cached model) with its
            # stdin detached from the JSON-RPC pipe.
            record = _installer_record(tmp_path)
            assert record["LUMEN_TAG"] == "v0.4.0"
            assert record["LUMEN_MODEL"] == "qwen3-5-9b"
            assert record["LUMEN_QUANT"] == "q4_0"
            assert record["STDIN"] == "Character Device"
            # The "binary" was really replaced (version verification passed).
            assert (tmp_path / "lumen-version.txt").read_text(encoding="utf-8") == "0.4.0"

            # The managed server stayed DOWN (restarts lazily on next use).
            send(5, "command.execute", {"session_id": session_id, "command": "/gpu"})
            gpu_response, _ = recv_until(5)
            assert gpu_response["result"]["status"]["lumen_server"] == "stopped"
        finally:
            process.terminate()
            process.wait(timeout=5)
            subprocess.run(
                ["pkill", "-f", str(tmp_path / "fake-lumen-server")],
                check=False,
                capture_output=True,
            )


def test_update_refused_while_boot_in_flight(tmp_path: Path) -> None:
    with _probe_server(lumen_tag="v0.4.0") as base:
        env = _update_stub_env(tmp_path, probe_base=base, with_server=True, boot_delay=3.0)
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/model qwen3-5-9b:q4_0"})
            boot_response, _ = recv_until(3)
            assert boot_response["result"]["background"] is True

            send(4, "command.execute", {"session_id": session_id, "command": "/update lumen"})
            update_response, _ = recv_until(4)
            assert update_response["result"]["ok"] is False
            message = str(update_response["result"]["message"])
            assert "Still starting local · qwen3-5-9b:q4_0" in message
            assert "before updating" in message
            # The installer must never have run.
            assert not (tmp_path / "installer-env.txt").exists()

            # Drain the boot so teardown is clean.
            read_events_until(
                lambda frame: frame["params"]["event_type"] == "message.updated"
                and frame["params"]["payload"].get("final") is True
            )
        finally:
            process.terminate()
            process.wait(timeout=5)
            subprocess.run(
                ["pkill", "-f", str(tmp_path / "fake-lumen-server")],
                check=False,
                capture_output=True,
            )


def test_boot_download_and_second_update_refused_while_update_in_flight(
    tmp_path: Path,
) -> None:
    """The other direction of the mutual-exclusion matrix: while /update lumen
    runs, model boots, downloads, and further updates are all refused."""
    with _probe_server(lumen_tag="v0.4.0") as base:
        env = _update_stub_env(
            tmp_path, probe_base=base, with_server=True, installer_sleep=4.0
        )
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update lumen"})
            update_response, _ = recv_until(3)
            assert update_response["result"]["ok"] is True
            assert update_response["result"]["background"] is True

            send(4, "command.execute", {"session_id": session_id, "command": "/model qwen3-5-9b:q4_0"})
            model_response, _ = recv_until(4)
            assert model_response["result"]["ok"] is False
            assert (
                "Lumen update in progress — wait for it to finish before switching models."
                in str(model_response["result"]["message"])
            )

            send(5, "command.execute", {"session_id": session_id, "command": "/download qwen3-5-9b:q8_0"})
            download_response, _ = recv_until(5)
            assert download_response["result"]["ok"] is False
            assert (
                "Lumen update in progress — wait for it to finish before downloading a model."
                in str(download_response["result"]["message"])
            )

            send(6, "command.execute", {"session_id": session_id, "command": "/update lumen"})
            second_update, _ = recv_until(6)
            assert second_update["result"]["ok"] is False
            assert "Lumen update already in progress" in str(second_update["result"]["message"])

            # Drain the update's final frame; it still completes successfully.
            seen = read_events_until(_final_update_frame)
            final_payload = seen[-1]["params"]["payload"]
            assert (
                final_payload["content"]
                == "local · Lumen 0.4.0 installed — server restarts on next use."
            )
        finally:
            process.terminate()
            process.wait(timeout=5)
            subprocess.run(
                ["pkill", "-f", str(tmp_path / "fake-lumen-server")],
                check=False,
                capture_output=True,
            )


def test_update_refused_while_download_in_flight(tmp_path: Path) -> None:
    with _probe_server(lumen_tag="v0.4.0") as base:
        env = _update_stub_env(tmp_path, probe_base=base, pull_delay=4.0)
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            # Uncached selector → background download starts.
            send(3, "command.execute", {"session_id": session_id, "command": "/download qwen3-5-9b:q8_0"})
            download_response, _ = recv_until(3)
            assert download_response["result"]["background"] is True

            send(4, "command.execute", {"session_id": session_id, "command": "/update lumen"})
            update_response, _ = recv_until(4)
            assert update_response["result"]["ok"] is False
            message = str(update_response["result"]["message"])
            assert "A download (qwen3-5-9b:q8_0) is in progress" in message
            assert "before updating" in message
            assert not (tmp_path / "installer-env.txt").exists()

            read_events_until(
                lambda frame: frame["params"]["event_type"] == "message.updated"
                and frame["params"]["payload"].get("final") is True
            )
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_update_cortex_reports_no_published_releases(tmp_path: Path) -> None:
    with _probe_server(lumen_tag="v0.4.0", cortex_tag=None) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update cortex"})
            response, events = recv_until(3)
            result = response["result"]
            assert result["ok"] is True
            assert "background" not in result
            expected = (
                f"Cortex has no published releases yet — you're on "
                f"{installed_cortex_version()} (source install)."
            )
            assert result["message"] == expected
            assert any(
                event["params"]["event_type"] == "system.notice"
                and event["params"]["payload"].get("message") == expected
                for event in events
            )
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_update_cortex_background_flow_end_to_end(tmp_path: Path) -> None:
    """/update cortex: resolves the strictly-newer release via the stub
    redirect probe, downloads the wheel + .sha256 assets from the SAME stub
    origin, verifies the checksum, hands the verified wheel to the (stub)
    pip, narrates engine-update frames, and resolves in place with the exact
    restart-to-apply message."""
    wheel_bytes = b"stub cortex release wheel"
    wheel_name, assets = _cortex_release_assets("v9.9.9", wheel_bytes)
    with _probe_server(lumen_tag="v0.4.0", cortex_tag="v9.9.9", assets=assets) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        env["CORTEX_SELF_INSTALL_KIND"] = "installed"  # repo checkout would refuse
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update cortex"})
            response, update_events = recv_until(3)
            result = response["result"]
            assert result["ok"] is True, result
            assert result["background"] is True
            assert result["repo_id"] == "cortex"
            assert "Updating Cortex to 9.9.9" in str(result["message"])

            seen = read_events_until(_final_update_frame)
            all_events = [*update_events, *seen]
            progress_frames = [
                event["params"]["payload"]
                for event in all_events
                if event["params"]["event_type"] == "message.updated"
                and isinstance(event["params"]["payload"].get("progress"), dict)
            ]
            assert progress_frames, "expected engine-update narration frames"
            message_ids = {payload["message_id"] for payload in progress_frames}
            assert len(message_ids) == 1, "one operation = one transcript message"
            for payload in progress_frames:
                assert payload["progress"]["kind"] == "engine-update"
                assert payload["progress"]["repo_id"] == "cortex"
            phases = [str(payload["progress"]["phase"]) for payload in progress_frames]
            assert f"downloading {wheel_name}" in phases
            assert "verifying checksum" in phases
            final_payload = progress_frames[-1]
            assert final_payload["final"] is True
            assert final_payload["progress"]["phase"] == "ready"
            assert final_payload["content"] == (
                "Cortex 9.9.9 installed — restart Cortex to apply."
            )

            # The stub pip received `install <verified local wheel>`: the
            # canonical asset filename, byte-identical to the published
            # wheel, with stdin detached from the JSON-RPC pipe.
            record = _pip_record(tmp_path)
            args = record["ARGS"].split()
            assert args[0] == "install"
            assert Path(args[1]).name == wheel_name
            assert record["SHA256"] == hashlib.sha256(wheel_bytes).hexdigest()
            assert record["STDIN"] == "Character Device"
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_update_cortex_refused_from_source_checkout(tmp_path: Path) -> None:
    """Finding-2 regression at the worker boundary: the worker runs from this
    repo (a source checkout), so an actionable release must be refused
    synchronously — pip replacing the editable install could overwrite
    working-tree files through install.sh's site-packages symlink."""
    wheel_bytes = b"stub cortex release wheel"
    _wheel_name, assets = _cortex_release_assets("v9.9.9", wheel_bytes)
    with _probe_server(lumen_tag="v0.4.0", cortex_tag="v9.9.9", assets=assets) as base:
        env = _update_stub_env(tmp_path, probe_base=base)  # no kind override
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update cortex"})
            response, _events = recv_until(3)
            result = response["result"]
            assert result["ok"] is False
            assert "background" not in result
            message = str(result["message"])
            assert "Cortex 9.9.9 is available" in message
            assert "source checkout" in message
            assert "git pull" in message
            # Nothing was downloaded or installed.
            assert not (tmp_path / "pip-args.txt").exists()
        finally:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize("shutdown_mode", ["sigterm", "stdin-eof"])
def test_worker_shutdown_waits_for_self_install_pip(
    tmp_path: Path, shutdown_mode: str
) -> None:
    """Finding-1 regression, both worker shutdown paths: quitting Cortex
    while the self-update's pip is rewriting the venv must NOT kill pip (a
    signal death skips pip's rollback and would strand a half-removed
    cortex-llm that cannot relaunch) — the worker waits for pip to finish,
    then exits. The slow stub pip proves it ran to completion."""
    wheel_bytes = b"stub cortex release wheel"
    wheel_name, assets = _cortex_release_assets("v9.9.9", wheel_bytes)
    pid_file = tmp_path / "pip.pid"
    done_file = tmp_path / "pip-done.txt"
    with _probe_server(lumen_tag="v0.4.0", cortex_tag="v9.9.9", assets=assets) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        env["CORTEX_SELF_INSTALL_KIND"] = "installed"  # repo checkout would refuse
        slow_pip = tmp_path / "slow-pip"
        slow_pip.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$$" > {pid_file}\n'
            "sleep 2\n"
            'echo "Successfully installed cortex-llm"\n'
            f'echo "COMPLETED" > {done_file}\n',
            encoding="utf-8",
        )
        slow_pip.chmod(0o755)
        env["CORTEX_SELF_PIP"] = str(slow_pip)
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update cortex"})
            response, _events = recv_until(3)
            assert response["result"]["background"] is True

            deadline = time.time() + 15
            while time.time() < deadline and not pid_file.exists():
                time.sleep(0.05)
            assert pid_file.exists(), "self-install pip never started"

            if shutdown_mode == "sigterm":
                process.terminate()  # the sidecar's exit path sends SIGTERM
            else:
                assert process.stdin is not None
                process.stdin.close()  # stdin EOF: run_forever's finally path
            process.wait(timeout=30)

            # pip ran to COMPLETION under the exiting worker — never signaled.
            assert done_file.exists(), "worker exit killed the self-install pip"
            assert done_file.read_text(encoding="utf-8").strip() == "COMPLETED"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_update_usage_error_for_unknown_component(tmp_path: Path) -> None:
    with _probe_server() as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update everything"})
            response, _ = recv_until(3)
            assert response["result"]["ok"] is False
            assert "Usage: /update [lumen|cortex]" in str(response["result"]["message"])
        finally:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize("shutdown_mode", ["sigterm", "stdin-eof"])
def test_worker_shutdown_reaps_in_flight_installer(
    tmp_path: Path, shutdown_mode: str
) -> None:
    """Finding-1 regression, both worker shutdown paths (signal handler and
    stdin-EOF finally): a worker exit mid-update must terminate the whole
    installer process GROUP (leader + children) and remove the downloaded
    temp installer — never orphan a half-applied install."""
    temp_dir = tmp_path / "worker-tmp"
    temp_dir.mkdir()
    pid_file = tmp_path / "installer.pid"
    installer_body = (
        "#!/usr/bin/env bash\n"
        f'echo "$$" > {pid_file}\n'
        "sleep 30 &\n"
        "sleep 30\n"
    ).encode("utf-8")

    class InstallerHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(installer_body)))
            self.end_headers()
            self.wfile.write(installer_body)

        def log_message(self, *args):
            pass

    installer_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), InstallerHandler)
    threading.Thread(target=installer_server.serve_forever, daemon=True).start()
    try:
        with _probe_server(lumen_tag="v0.4.0") as base:
            env = _update_stub_env(tmp_path, probe_base=base)
            # Downloaded-installer path (curl → temp file) in a scoped TMPDIR
            # so the temp-cleanup contract is observable.
            env["CORTEX_LUMEN_INSTALLER_URL"] = (
                f"http://127.0.0.1:{installer_server.server_address[1]}/install.sh"
            )
            env["TMPDIR"] = str(temp_dir)
            process, send, recv_until, _ = _worker_session(env)
            try:
                session_id = _bootstrap(send, recv_until)
                send(3, "command.execute", {"session_id": session_id, "command": "/update lumen"})
                response, _events = recv_until(3)
                assert response["result"]["background"] is True

                deadline = time.time() + 15
                while time.time() < deadline and not pid_file.exists():
                    time.sleep(0.05)
                assert pid_file.exists(), "installer never started"
                pgid = int(pid_file.read_text(encoding="utf-8").strip())
                assert list(temp_dir.glob("cortex-installer-*.sh")), (
                    "downloaded temp installer should exist mid-install"
                )

                if shutdown_mode == "sigterm":
                    process.terminate()  # the sidecar's exit path sends SIGTERM
                else:
                    assert process.stdin is not None
                    process.stdin.close()  # stdin EOF: run_forever's finally path
                process.wait(timeout=15)

                deadline = time.time() + 6
                group_gone = False
                while time.time() < deadline:
                    try:
                        os.killpg(pgid, 0)
                    except ProcessLookupError:
                        group_gone = True
                        break
                    time.sleep(0.1)
                assert group_gone, f"installer group {pgid} outlived the worker"
                assert list(temp_dir.glob("cortex-installer-*.sh")) == [], (
                    "temp installer file must be removed on worker shutdown"
                )
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
    finally:
        installer_server.shutdown()
        installer_server.server_close()


def test_update_lumen_refused_while_turn_is_running(tmp_path: Path) -> None:
    """Finding-4 regression: /update lumen stops the managed server, so it
    must refuse while a generation turn is live (killing the server would end
    the turn with a stream error) — and the guard lifts once the turn ends."""
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps({"responses": [[{"delay_ms": 4000, "text": "SLOW_TURN_DONE"}]]}),
        encoding="utf-8",
    )
    with _probe_server(lumen_tag="v0.4.0") as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        env["CORTEX_SCRIPTED_MODEL"] = str(script)
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)

            # Start a slow (4s) scripted turn; wait until it is genuinely
            # RUNNING (session.status busy) before racing the update.
            send(
                3,
                "session.submit_user_input",
                {
                    "session_id": session_id,
                    "user_input": "hello",
                    "active_target": {
                        "backend": "cloud",
                        "provider": "azure",
                        "model_id": "scripted",
                    },
                },
            )
            read_events_until(
                lambda frame: frame["params"]["event_type"] == "session.status"
                and frame["params"]["payload"].get("status") == "busy"
            )

            send(4, "command.execute", {"session_id": session_id, "command": "/update lumen"})
            update_response, _ = recv_until(4)
            assert update_response["result"]["ok"] is False
            message = str(update_response["result"]["message"])
            assert "A turn is still running" in message
            assert "before updating Lumen" in message
            assert not (tmp_path / "installer-env.txt").exists()

            # The turn itself completes untouched.
            turn_response, _ = recv_until(3)
            assert turn_response["result"]["assistant_text"] == "SLOW_TURN_DONE"

            # Guard lifts once the turn is over: the update now starts.
            send(5, "command.execute", {"session_id": session_id, "command": "/update lumen"})
            retry_response, _ = recv_until(5)
            assert retry_response["result"]["ok"] is True
            assert retry_response["result"]["background"] is True
            seen = read_events_until(_final_update_frame)
            assert (
                seen[-1]["params"]["payload"]["content"]
                == "local · Lumen 0.4.0 installed — server restarts on next use."
            )
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_startup_update_notice_emitted_once(tmp_path: Path) -> None:
    """auto_update_check=true (the default in production) emits exactly one
    system.notice for the newer Lumen release — and never again for later
    commands in the same session."""
    with _probe_server(lumen_tag="v0.4.0", cortex_tag=None) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        env["CORTEX_AUTO_UPDATE_CHECK"] = "true"
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            send(1, "app.handshake", {"protocol_version": "1.0.0"})
            recv_until(1)
            send(2, "session.create_or_resume", {})
            response, create_events = recv_until(2)
            session_id = response["result"]["session_id"]

            expected = "Lumen 0.4.0 available — update with /update lumen"

            def _is_update_notice(frame: dict) -> bool:
                params = frame["params"]
                return params["event_type"] == "system.notice" and expected in str(
                    params["payload"].get("message", "")
                )

            already = [event for event in create_events if _is_update_notice(event)]
            if already:
                notice_event = already[0]
            else:
                # The check runs on a daemon thread — the notice may land
                # shortly after session creation.
                notice_event = read_events_until(_is_update_notice, timeout=30.0)[-1]

            # The wire contract the TUI store relies on: the async notice is
            # marked out-of-band so a command in flight can never drop it.
            assert notice_event["params"]["payload"].get("origin") == "update-check"

            # No duplicate notice on subsequent activity.
            send(3, "command.execute", {"session_id": session_id, "command": "/help"})
            _response, help_events = recv_until(3)
            assert not any(_is_update_notice(event) for event in help_events)
        finally:
            process.terminate()
            process.wait(timeout=5)
