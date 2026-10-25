"""UpdateService tests: status report, planning, and REAL installer execution
via stub scripts (a subprocess boundary, but zero network — probes use
injected openers, and the cortex self-update downloads its wheel asset from a
local stub HTTP server exactly how the worker e2e does it)."""

from __future__ import annotations

import email.message
import hashlib
import http.server
import io
import os
import tempfile
import threading
import time
import urllib.error
from contextlib import contextmanager
from pathlib import Path

import pytest

import cortex.app.update_service as update_service_module
from cortex.app.update_service import UpdateService
from cortex.update_check import UpdateCheckCache, installed_cortex_version

# ---- probe fakes -----------------------------------------------------------


def _headers(location: str | None) -> email.message.Message:
    headers = email.message.Message()
    if location is not None:
        headers["Location"] = location
    return headers


class ReleaseOpener:
    """Redirects to the configured release tag (None → 404 / no releases)."""

    def __init__(self, *, cortex_tag: str | None = None) -> None:
        self.cortex_tag = cortex_tag
        self.calls: list[str] = []

    def open(self, url: str, *, timeout: float | None = None):
        self.calls.append(url)
        if self.cortex_tag is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", _headers(None), io.BytesIO(b""))
        location = f"https://github.com/repo/x/releases/tag/{self.cortex_tag}"
        raise urllib.error.HTTPError(url, 302, "Found", _headers(location), io.BytesIO(b""))


class FailingOpener:
    def open(self, url: str, *, timeout: float | None = None):
        raise urllib.error.URLError("offline")


def _service(tmp_path: Path, *, cortex_tag: str | None = None) -> UpdateService:
    return UpdateService(
        cache=UpdateCheckCache(tmp_path / "update-check.json"),
        opener=ReleaseOpener(cortex_tag=cortex_tag),
    )


# ---- status / notices ----------------------------------------------------------


def test_status_report_no_releases(tmp_path: Path) -> None:
    report = _service(tmp_path, cortex_tag=None).status_report()
    assert report["ok"] is True
    assert report["message"] == (
        f"Cortex: {installed_cortex_version()} installed · no published releases yet"
    )
    assert report["update_status"] == {
        "cortex_installed": installed_cortex_version(),
        "cortex_latest": None,
    }


def test_status_report_newer_release_available(tmp_path: Path) -> None:
    report = _service(tmp_path, cortex_tag="v9999.0.0").status_report()
    assert report["message"] == (
        f"Cortex: {installed_cortex_version()} installed · 9999.0.0 available — /update cortex"
    )
    assert report["update_status"] == {
        "cortex_installed": installed_cortex_version(),
        "cortex_latest": "9999.0.0",
    }


def test_status_report_up_to_date_and_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = f"v{installed_cortex_version()}"
    service = _service(tmp_path, cortex_tag=current)
    assert service.status_report()["message"] == (
        f"Cortex: {installed_cortex_version()} installed · up to date"
    )

    monkeypatch.setattr(update_service_module, "installed_cortex_version", lambda: None)
    service = _service(tmp_path / "b", cortex_tag=current)
    assert service.status_report()["message"] == "Cortex: not installed"


def test_status_report_falls_back_to_cache_when_probe_fails(tmp_path: Path) -> None:
    cache = UpdateCheckCache(tmp_path / "update-check.json")
    cache.store(cortex_latest="v9999.0.0")
    service = UpdateService(cache=cache, opener=FailingOpener())
    assert service.status_report()["message"] == (
        f"Cortex: {installed_cortex_version()} installed · 9999.0.0 available — /update cortex"
    )


def test_status_report_transient_failure_is_not_no_releases(tmp_path: Path) -> None:
    """A network failure with a COLD cache must never be reported as the
    definitive 'no published releases yet' — that claim is reserved for
    GitHub's authoritative 404."""
    service = UpdateService(
        cache=UpdateCheckCache(tmp_path / "update-check.json"),
        opener=FailingOpener(),
    )
    message = str(service.status_report()["message"])
    assert "no published releases yet" not in message
    assert message == (
        f"Cortex: {installed_cortex_version()} installed · "
        "could not determine the latest release"
    )


def test_startup_notice_only_when_strictly_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    newer = _service(tmp_path / "a", cortex_tag="v9999.0.0")
    assert newer.startup_notice() == "Cortex 9999.0.0 available — update with /update cortex"

    current = _service(tmp_path / "b", cortex_tag=f"v{installed_cortex_version()}")
    assert current.startup_notice() is None

    older = _service(tmp_path / "c", cortex_tag="v0.0.1")
    assert older.startup_notice() is None

    no_releases = _service(tmp_path / "d", cortex_tag=None)
    assert no_releases.startup_notice() is None

    prerelease = _service(tmp_path / "e", cortex_tag="v9999.0.0-rc1")
    assert prerelease.startup_notice() is None

    monkeypatch.setattr(update_service_module, "installed_cortex_version", lambda: None)
    unknown_installed = _service(tmp_path / "f", cortex_tag="v9999.0.0")
    assert unknown_installed.startup_notice() is None


def test_startup_notice_cortex_source_checkout_points_to_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The startup nudge must never steer a source-checkout developer into
    /update cortex — the wheel install would replace their editable install."""
    monkeypatch.delenv("CORTEX_SELF_INSTALL_KIND", raising=False)
    service = _service(tmp_path, cortex_tag="v9999.0.0")
    # This suite runs from the repo — a source checkout.
    notice = str(service.startup_notice())
    assert "Cortex 9999.0.0 released — source checkout: update with git pull" == notice
    assert "/update cortex" not in notice

    # Normal installs keep the /update cortex nudge.
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    installed = _service(tmp_path / "b", cortex_tag="v9999.0.0")
    assert installed.startup_notice() == (
        "Cortex 9999.0.0 available — update with /update cortex"
    )


# ---- cortex update execution ---------------------------------------------------
#
# The self-update downloads the release's wheel asset + .sha256 sibling from
# a LOCAL stub HTTP server (CORTEX_UPDATE_PROBE_BASE doubles as the asset
# base — one origin serves discovery and download, exactly like github.com),
# verifies the checksum, and hands the verified local wheel to a stub pip
# (CORTEX_SELF_PIP) so the suite never writes a real venv. The suite runs
# FROM this repo — a source checkout, which /update cortex refuses — so tests
# of the normal wheel path force CORTEX_SELF_INSTALL_KIND=installed.

CORTEX_WHEEL_ASSET = "cortex_llm-9.9.9-py3-none-macosx_13_0_arm64.whl"
CORTEX_ASSET_DIR = "/faisalmumtaz89/Cortex/releases/download/v9.9.9"


def _release_assets(
    wheel_bytes: bytes = b"stub wheel bytes", *, checksum_of: bytes | None = None
) -> dict[str, bytes]:
    """Asset paths → bytes, laid out like a GitHub release. ``checksum_of``
    lets a test publish a .sha256 that does NOT match the wheel bytes."""
    digest = hashlib.sha256(
        wheel_bytes if checksum_of is None else checksum_of
    ).hexdigest()
    return {
        f"{CORTEX_ASSET_DIR}/{CORTEX_WHEEL_ASSET}": wheel_bytes,
        f"{CORTEX_ASSET_DIR}/{CORTEX_WHEEL_ASSET}.sha256": (
            f"{digest}  {CORTEX_WHEEL_ASSET}\n".encode("utf-8")
        ),
    }


@contextmanager
def _asset_server(assets: dict[str, bytes]):
    """Local stand-in for github.com's releases/download asset paths."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = assets.get(self.path)
            if body is None:
                self.send_response(404)
                self.send_header("Content-Length", "9")
                self.end_headers()
                self.wfile.write(b"Not Found")
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _stub_pip(tmp_path: Path, *, exit_code: int = 0) -> Path:
    """Stand-in for `python -m pip` (the CORTEX_SELF_PIP seam): records its
    argv AND the sha256 of the wheel it was told to install — proving the
    installed file is byte-identical to the verified download (no TOCTOU
    window between verify and install)."""
    record = tmp_path / "pip-args.txt"
    script = tmp_path / "stub-pip"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "ARGS=$*" > {record}\n'
        f"shasum -a 256 \"$2\" | awk '{{print \"SHA256=\"$1}}' >> {record}\n"
        'echo "Successfully installed cortex-llm"\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _pip_record(tmp_path: Path) -> dict[str, str]:
    text = (tmp_path / "pip-args.txt").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.strip().splitlines())


def test_update_cortex_no_releases_message(tmp_path: Path) -> None:
    service = _service(tmp_path, cortex_tag=None)
    result = service.update_cortex()
    assert result["ok"] is True
    assert str(result["message"]) == (
        f"Cortex has no published releases yet — you're on {installed_cortex_version()} "
        "(source install)."
    )


def test_update_cortex_transient_probe_failure_is_not_no_releases(tmp_path: Path) -> None:
    """A transient probe failure with a cold cache must give the honest
    'check your network' answer, never the false factual claim 'no published
    releases yet'."""
    service = UpdateService(
        cache=UpdateCheckCache(tmp_path / "update-check.json"),
        opener=FailingOpener(),
    )
    result = service.update_cortex()
    assert result["ok"] is False
    message = str(result["message"])
    assert message == (
        "Could not determine the latest Cortex release — check your network and try again."
    )
    assert "no published releases" not in message


def test_update_cortex_refuses_from_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This suite runs FROM the repo — a source
    checkout — so an actionable release must be refused: pip would replace
    the editable install (and can write through install.sh's site-packages
    symlink into the working tree). Nothing is downloaded or installed."""
    monkeypatch.delenv("CORTEX_SELF_INSTALL_KIND", raising=False)
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", "http://198.51.100.7")  # never contacted
    service = _service(tmp_path, cortex_tag="v9.9.9")
    result = service.update_cortex()
    assert result["ok"] is False
    message = str(result["message"])
    assert "Cortex 9.9.9 is available" in message
    assert "source checkout" in message
    assert "git pull" in message
    assert not (tmp_path / "pip-args.txt").exists()  # pip never ran


def test_update_cortex_downloads_verifies_and_installs_release_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_bytes = b"cortex release wheel payload"
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    with _asset_server(_release_assets(wheel_bytes)) as base:
        monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", base)
        service = _service(tmp_path, cortex_tag="v9.9.9")
        phases: list[str] = []
        result = service.update_cortex(
            progress_callback=lambda payload: phases.append(str(payload["phase"]))
        )
    assert result["ok"] is True, result
    assert result["message"] == "Cortex 9.9.9 installed — restart Cortex to apply."
    record = _pip_record(tmp_path)
    args = record["ARGS"].split()
    assert args[0] == "install"
    wheel_path = Path(args[1])
    # Canonical PEP 427 filename preserved — pip rejects renamed wheels.
    assert wheel_path.name == CORTEX_WHEEL_ASSET
    # pip installed EXACTLY the verified bytes.
    assert record["SHA256"] == hashlib.sha256(wheel_bytes).hexdigest()
    # The staged temp dir (wheel + .sha256) is removed after the install.
    assert not wheel_path.parent.exists()
    assert f"downloading {CORTEX_WHEEL_ASSET}" in phases
    assert "verifying checksum" in phases


def test_update_cortex_rejects_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel that does not match its published .sha256 installs NOTHING and
    leaves no unverified artifact behind."""
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    assets = _release_assets(b"tampered wheel bytes", checksum_of=b"published wheel bytes")
    with _asset_server(assets) as base:
        monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", base)
        service = _service(tmp_path, cortex_tag="v9.9.9")
        result = service.update_cortex()
    assert result["ok"] is False
    message = str(result["message"])
    assert "Cortex update failed" in message
    assert "checksum mismatch" in message
    assert "installing nothing" in message
    assert not (tmp_path / "pip-args.txt").exists()  # pip never ran
    assert list(staging.glob("cortex-wheel-*")) == []  # tampered wheel not left behind


def test_update_cortex_refuses_missing_sha256_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    assets = _release_assets()
    del assets[f"{CORTEX_ASSET_DIR}/{CORTEX_WHEEL_ASSET}.sha256"]
    with _asset_server(assets) as base:
        monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", base)
        service = _service(tmp_path, cortex_tag="v9.9.9")
        result = service.update_cortex()
    assert result["ok"] is False
    assert "refusing to install an unverified wheel" in str(result["message"])
    assert not (tmp_path / "pip-args.txt").exists()


def test_update_cortex_missing_wheel_asset_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    with _asset_server({}) as base:
        monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", base)
        service = _service(tmp_path, cortex_tag="v9.9.9")
        result = service.update_cortex()
    assert result["ok"] is False
    message = str(result["message"])
    assert "Cortex update failed" in message
    assert "download failed" in message
    assert "v9.9.9" in message  # names the release it looked for
    assert not (tmp_path / "pip-args.txt").exists()


def test_update_cortex_refuses_untrusted_asset_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-loopback plain-http asset bases are refused BEFORE any download:
    the .sha256 shares the wheel's origin, so verification cannot protect
    against a hostile base."""
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", "http://198.51.100.7")  # never contacted
    service = _service(tmp_path, cortex_tag="v9.9.9")
    result = service.update_cortex()
    assert result["ok"] is False
    assert "refusing to download update artifacts" in str(result["message"])
    assert not (tmp_path / "pip-args.txt").exists()


def test_update_cortex_older_release_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    service = _service(tmp_path, cortex_tag="v0.0.1")
    result = service.update_cortex()
    assert result["ok"] is True
    assert "up to date" in str(result["message"])
    assert not (tmp_path / "pip-args.txt").exists()  # nothing downloaded or installed


def test_update_cortex_pip_failure_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path, exit_code=7)))
    with _asset_server(_release_assets()) as base:
        monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", base)
        service = _service(tmp_path, cortex_tag="v9.9.9")
        result = service.update_cortex()
    assert result["ok"] is False
    assert "Cortex update failed" in str(result["message"])


# ---- shutdown safety: no orphaned child process group ------------------------


@contextmanager
def _stalling_asset_server():
    """Serves the release wheel's first bytes, then stalls until released —
    an in-flight download that only a shutdown can end."""
    started = threading.Event()
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != f"{CORTEX_ASSET_DIR}/{CORTEX_WHEEL_ASSET}":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(1024 * 1024))
            self.end_headers()
            self.wfile.write(b"x" * 1024)
            self.wfile.flush()
            started.set()
            release.wait(30)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", started
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def _wait_for_file(path: Path, *, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _assert_group_gone(pgid: int, *, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    raise AssertionError(f"process group {pgid} still alive after shutdown")


def test_shutdown_terminates_in_flight_download_and_removes_staged_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker-exit contract for an ordinary child: shutdown() must reap the
    in-flight wheel download's WHOLE process group and remove the staged temp
    directory (the update thread's finally never runs when the worker exits
    through os._exit). The interrupted update fails LOUD and installs nothing."""
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))

    with _stalling_asset_server() as (base, download_started):
        monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", base)
        service = _service(tmp_path, cortex_tag="v9.9.9")
        outcome: dict[str, dict] = {}
        thread = threading.Thread(
            target=lambda: outcome.update(result=service.update_cortex()), daemon=True
        )
        thread.start()
        assert download_started.wait(10), "wheel download never started"
        child = service._active_child
        assert child is not None, "download child should be tracked mid-flight"
        pgid = child.pid  # session leader: pid == pgid
        assert list(staging.glob("cortex-wheel-*")), "staged wheel dir should exist mid-download"

        service.shutdown()

        thread.join(timeout=10)
        assert not thread.is_alive(), "update thread wedged after shutdown"
        _assert_group_gone(pgid)
        assert list(staging.glob("cortex-wheel-*")) == []
        result = outcome["result"]
        assert result["ok"] is False  # killed-mid-download fails LOUD
        message = str(result["message"])
        assert "Cortex update failed" in message
        assert "download failed" in message
        assert not (tmp_path / "pip-args.txt").exists()  # nothing installed

        service.shutdown()  # idempotent: nothing left to reap, never raises


def test_shutdown_waits_for_in_flight_self_install_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker-exit contract for the SELF-update: the pip child is rewriting
    cortex-llm inside the running venv, and a signal death skips pip's
    rollback — shutdown() must WAIT for it to finish (never signal it), then
    remove the staged temp dir. The completed update stays a success."""
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    pid_file = tmp_path / "pip.pid"
    done_file = tmp_path / "pip-done.txt"
    slow_pip = tmp_path / "slow-pip"
    slow_pip.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$$" > {pid_file}\n'
        "sleep 1\n"
        'echo "Successfully installed cortex-llm"\n'
        f'echo "COMPLETED" > {done_file}\n',
        encoding="utf-8",
    )
    slow_pip.chmod(0o755)
    monkeypatch.setenv("CORTEX_SELF_PIP", str(slow_pip))

    with _asset_server(_release_assets()) as base:
        monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", base)
        service = _service(tmp_path, cortex_tag="v9.9.9")
        outcome: dict[str, dict] = {}
        thread = threading.Thread(
            target=lambda: outcome.update(result=service.update_cortex()), daemon=True
        )
        thread.start()
        _wait_for_file(pid_file)

        service.shutdown()  # default grace ≫ the stub's runtime

        # pip ran to COMPLETION — it was waited on, not signaled.
        assert done_file.read_text(encoding="utf-8").strip() == "COMPLETED"
        thread.join(timeout=10)
        assert not thread.is_alive(), "update thread wedged after shutdown"
        assert list(staging.glob("cortex-wheel-*")) == []  # staged dir removed
        result = outcome["result"]
        assert result["ok"] is True, result
        assert result["message"] == "Cortex 9.9.9 installed — restart Cortex to apply."

        service.shutdown()  # idempotent: nothing left to reap, never raises


def test_shutdown_kills_wedged_self_install_pip_as_last_resort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-install pip wedged past the (bounded) grace is SIGKILLed —
    worker exit must not hang forever — its whole process group included,
    and the staged wheel temp DIRECTORY (wheel + .sha256 — a plain file
    unlink cannot cover it) is removed; the interrupted update fails LOUD."""
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    pid_file = tmp_path / "pip.pid"
    sleeping_pip = tmp_path / "sleeping-pip"
    sleeping_pip.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$$" > {pid_file}\n'
        "sleep 30 &\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    sleeping_pip.chmod(0o755)
    monkeypatch.setenv("CORTEX_SELF_PIP", str(sleeping_pip))

    with _asset_server(_release_assets()) as base:
        monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", base)
        service = _service(tmp_path, cortex_tag="v9.9.9")
        outcome: dict[str, dict] = {}
        thread = threading.Thread(
            target=lambda: outcome.update(result=service.update_cortex()), daemon=True
        )
        thread.start()
        _wait_for_file(pid_file)
        pgid = int(pid_file.read_text(encoding="utf-8").strip())
        staged = list(staging.glob("cortex-wheel-*"))
        assert staged, "staged wheel temp dir should exist mid-install"
        assert (staged[0] / CORTEX_WHEEL_ASSET).exists()

        service.shutdown(self_install_grace=0.5)

        thread.join(timeout=10)
        assert not thread.is_alive(), "update thread wedged after shutdown"
        _assert_group_gone(pgid)
        assert list(staging.glob("cortex-wheel-*")) == []
        result = outcome["result"]
        assert result["ok"] is False  # killed-mid-install fails LOUD
        assert "Cortex update failed" in str(result["message"])

        service.shutdown()  # idempotent: nothing left to reap, never raises
