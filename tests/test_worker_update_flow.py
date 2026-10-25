"""Worker-stdio e2e tests for the /update command family and the startup
update notice.

Fully hermetic: the GitHub "releases/latest" redirect probe AND the release
wheel assets are served by a local stub HTTP server (CORTEX_UPDATE_PROBE_BASE
is the shared origin for discovery and downloads, like github.com), and pip
for the cortex self-update is a stub recorder (CORTEX_SELF_PIP — a real pip
would write the suite's own venv). No real network."""

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


# ---- local stub of GitHub's releases/latest redirect ----------------------


@contextmanager
def _probe_server(
    *,
    cortex_tag: str | None = None,
    assets: dict[str, bytes] | None = None,
    stall_paths: frozenset[str] = frozenset(),
):
    """Serves /{repo}/releases/latest with a 302 → …/releases/tag/<tag>
    (GitHub's real behavior) or 404 when the repo has no releases. ``assets``
    optionally maps releases/download/... paths to bytes, so the SAME origin
    also serves release assets — exactly like github.com. A path in
    ``stall_paths`` sends its first bytes and then stalls until the server
    closes, leaving that download in flight."""
    tags = {"/faisalmumtaz89/Cortex/releases/latest": cortex_tag}
    asset_map = assets or {}
    release = threading.Event()

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
            elif self.path in stall_paths:
                self.send_response(200)
                self.send_header("Content-Length", str(1024 * 1024))
                self.end_headers()
                self.wfile.write(b"x" * 1024)
                self.wfile.flush()
                release.wait(30)
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
        release.set()
        server.shutdown()
        server.server_close()


# ---- worker environment ----------------------------------------------------


def _update_stub_env(tmp_path: Path, *, probe_base: str) -> dict:
    """Worker env with a recording stub pip and the probe pointed at the
    local stub server."""
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
    env["CORTEX_UPDATE_PROBE_BASE"] = probe_base
    env["CORTEX_SELF_PIP"] = str(stub_pip)
    env["CORTEX_AUTO_UPDATE_CHECK"] = "false"  # notice test opts back in
    env.pop("CORTEX_SCRIPTED_MODEL", None)
    # The worker runs FROM this repo — a source checkout, which /update
    # cortex refuses. Tests of the normal wheel path opt in to "installed".
    env.pop("CORTEX_SELF_INSTALL_KIND", None)
    return env


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


def _worker_session(env: dict, cwd: Path = REPO_ROOT):
    """Spawn a worker; returns (process, send, recv_until, read_events_until)."""
    process = subprocess.Popen(
        [sys.executable, "-P", "-m", "cortex", "--worker-stdio"],
        cwd=cwd,
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
    with _probe_server(cortex_tag="v9999.0.0") as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update"})
            response, events = recv_until(3)
            result = response["result"]
            assert result["ok"] is True
            expected = (
                f"Cortex: {installed_cortex_version()} installed · "
                "9999.0.0 available — /update cortex"
            )
            assert result["message"] == expected
            assert result["update_status"] == {
                "cortex_installed": installed_cortex_version(),
                "cortex_latest": "9999.0.0",
            }
            # Synchronous command: busy/idle status plus the notice.
            statuses = [
                event["params"]["payload"].get("status")
                for event in events
                if event["params"]["event_type"] == "session.status"
            ]
            assert "busy" in statuses and "idle" in statuses
            assert any(
                event["params"]["event_type"] == "system.notice"
                and event["params"]["payload"].get("message") == expected
                for event in events
            )
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_update_status_reports_no_published_releases(tmp_path: Path) -> None:
    with _probe_server(cortex_tag=None) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update"})
            response, _events = recv_until(3)
            assert response["result"]["message"] == (
                f"Cortex: {installed_cortex_version()} installed · no published releases yet"
            )
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_update_cortex_reports_no_published_releases(tmp_path: Path) -> None:
    with _probe_server(cortex_tag=None) as base:
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
    with _probe_server(cortex_tag="v9.9.9", assets=assets) as base:
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


def test_self_update_pip_never_imports_modules_from_the_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "pip").mkdir(parents=True)
    marker = tmp_path / "imported-from-project"
    (project / "pip" / "__init__.py").write_text("", encoding="utf-8")
    (project / "pip" / "__main__.py").write_text(
        f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8"
    )
    _wheel_name, assets = _cortex_release_assets("v9.9.9", b"not a real wheel")
    with _probe_server(cortex_tag="v9.9.9", assets=assets) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        env["CORTEX_SELF_INSTALL_KIND"] = "installed"
        env.pop("CORTEX_SELF_PIP")
        process, send, recv_until, read_events_until = _worker_session(env, cwd=project)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update cortex"})
            recv_until(3)
            read_events_until(_final_update_frame)
        finally:
            process.terminate()
            process.wait(timeout=5)

    assert not marker.exists()


def test_update_cortex_refused_from_source_checkout(tmp_path: Path) -> None:
    """The worker runs from this repo (a source checkout), so an actionable
    release must be refused synchronously — pip replacing the editable install
    could overwrite working-tree files through install.sh's site-packages
    symlink."""
    wheel_bytes = b"stub cortex release wheel"
    _wheel_name, assets = _cortex_release_assets("v9.9.9", wheel_bytes)
    with _probe_server(cortex_tag="v9.9.9", assets=assets) as base:
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
    """Both worker shutdown paths: quitting Cortex while the self-update's pip
    is rewriting the venv must NOT kill pip (a signal death skips pip's
    rollback and would strand a half-removed cortex-llm that cannot relaunch)
    — the worker waits for pip to finish, then exits. The slow stub pip proves
    it ran to completion."""
    wheel_bytes = b"stub cortex release wheel"
    wheel_name, assets = _cortex_release_assets("v9.9.9", wheel_bytes)
    pid_file = tmp_path / "pip.pid"
    done_file = tmp_path / "pip-done.txt"
    with _probe_server(cortex_tag="v9.9.9", assets=assets) as base:
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


def test_second_update_refused_while_update_in_flight(tmp_path: Path) -> None:
    """One self-update at a time: a second /update cortex while the first is
    still installing is refused, and the first still completes."""
    _wheel_name, assets = _cortex_release_assets("v9.9.9", b"stub cortex release wheel")
    with _probe_server(cortex_tag="v9.9.9", assets=assets) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        env["CORTEX_SELF_INSTALL_KIND"] = "installed"  # repo checkout would refuse
        pid_file = tmp_path / "pip.pid"
        slow_pip = tmp_path / "slow-pip"
        slow_pip.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$$" > {pid_file}\n'
            "sleep 3\n"
            'echo "Successfully installed cortex-llm"\n',
            encoding="utf-8",
        )
        slow_pip.chmod(0o755)
        env["CORTEX_SELF_PIP"] = str(slow_pip)
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update cortex"})
            first, first_events = recv_until(3)
            assert first["result"]["ok"] is True
            assert first["result"]["background"] is True

            send(4, "command.execute", {"session_id": session_id, "command": "/update cortex"})
            second, second_events = recv_until(4)
            assert second["result"] == {
                "ok": False,
                "message": "Cortex update already in progress — wait for it to finish.",
            }

            # The first update still completes successfully.
            seen = [*first_events, *second_events]
            if not any(_final_update_frame(event) for event in seen):
                seen.extend(read_events_until(_final_update_frame))
            final_frame = [event for event in seen if _final_update_frame(event)][-1]
            assert final_frame["params"]["payload"]["content"] == (
                "Cortex 9.9.9 installed — restart Cortex to apply."
            )
            assert pid_file.exists(), "the first update's pip must have run"
        finally:
            process.terminate()
            process.wait(timeout=5)


def _download_running(url: str) -> bool:
    """True while a process whose command line carries ``url`` is alive."""
    probe = subprocess.run(["pgrep", "-f", url], capture_output=True, check=False)
    return probe.returncode == 0


@pytest.mark.parametrize("shutdown_mode", ["sigterm", "stdin-eof"])
def test_worker_shutdown_reaps_in_flight_download(tmp_path: Path, shutdown_mode: str) -> None:
    """Both worker shutdown paths (signal handler and stdin-EOF finally): a
    worker exit mid-download must terminate the download's process and remove
    the staged wheel directory — never orphan a half-finished update."""
    temp_dir = tmp_path / "worker-tmp"
    temp_dir.mkdir()
    wheel_name, assets = _cortex_release_assets("v9.9.9", b"stub cortex release wheel")
    wheel_path = f"/faisalmumtaz89/Cortex/releases/download/v9.9.9/{wheel_name}"
    with _probe_server(
        cortex_tag="v9.9.9", assets=assets, stall_paths=frozenset({wheel_path})
    ) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        env["CORTEX_SELF_INSTALL_KIND"] = "installed"  # repo checkout would refuse
        # Scoped TMPDIR so the staged-directory cleanup contract is observable.
        env["TMPDIR"] = str(temp_dir)
        wheel_url = f"{base}{wheel_path}"
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update cortex"})
            response, _events = recv_until(3)
            assert response["result"]["background"] is True

            deadline = time.time() + 15
            while time.time() < deadline and not list(temp_dir.glob("cortex-wheel-*/*.whl")):
                time.sleep(0.05)
            assert list(temp_dir.glob("cortex-wheel-*/*.whl")), "wheel download never started"
            assert _download_running(wheel_url), "download child should be running"

            if shutdown_mode == "sigterm":
                process.terminate()  # the sidecar's exit path sends SIGTERM
            else:
                assert process.stdin is not None
                process.stdin.close()  # stdin EOF: run_forever's finally path
            process.wait(timeout=15)

            deadline = time.time() + 6
            while time.time() < deadline and _download_running(wheel_url):
                time.sleep(0.1)
            assert not _download_running(wheel_url), "download outlived the worker"
            assert list(temp_dir.glob("cortex-wheel-*")) == [], (
                "staged wheel directory must be removed on worker shutdown"
            )
            assert not (tmp_path / "pip-args.txt").exists()  # nothing installed
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_update_usage_error_for_unknown_component(tmp_path: Path) -> None:
    with _probe_server(cortex_tag=None) as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        process, send, recv_until, _ = _worker_session(env)
        try:
            session_id = _bootstrap(send, recv_until)
            send(3, "command.execute", {"session_id": session_id, "command": "/update everything"})
            response, _ = recv_until(3)
            assert response["result"]["ok"] is False
            assert "Usage: /update [cortex]" in str(response["result"]["message"])
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_startup_update_notice_emitted_once(tmp_path: Path) -> None:
    """auto_update_check=true (the default in production) emits exactly one
    system.notice for the newer Cortex release — and never again for later
    commands in the same session."""
    with _probe_server(cortex_tag="v9999.0.0") as base:
        env = _update_stub_env(tmp_path, probe_base=base)
        env["CORTEX_AUTO_UPDATE_CHECK"] = "true"
        env["CORTEX_SELF_INSTALL_KIND"] = "installed"  # repo checkout gets the git hint
        process, send, recv_until, read_events_until = _worker_session(env)
        try:
            send(1, "app.handshake", {"protocol_version": "1.0.0"})
            recv_until(1)
            send(2, "session.create_or_resume", {})
            response, create_events = recv_until(2)
            session_id = response["result"]["session_id"]

            expected = "Cortex 9999.0.0 available — update with /update cortex"

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
