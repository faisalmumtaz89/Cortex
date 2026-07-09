"""UpdateService tests: status report, planning, and REAL installer execution
via stub bash scripts (a subprocess boundary, but zero network — probes use
injected openers, the Lumen installer is a local file injected through the
env-overridable installer-URL seam, and the cortex self-update downloads its
wheel asset from a local stub HTTP server exactly how the worker e2e does
it)."""

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

from cortex.app.update_service import UpdateService
from cortex.update_check import UpdateCheckCache, installed_cortex_version
from tests.lumen_fakes import FakeLumenRuntime, catalog

# ---- probe fakes -----------------------------------------------------------


def _headers(location: str | None) -> email.message.Message:
    headers = email.message.Message()
    if location is not None:
        headers["Location"] = location
    return headers


class PerRepoOpener:
    """Redirects each repo to its configured tag (None → 404 / no releases)."""

    def __init__(self, *, lumen_tag: str | None = None, cortex_tag: str | None = None) -> None:
        self.lumen_tag = lumen_tag
        self.cortex_tag = cortex_tag
        self.calls: list[str] = []

    def open(self, url: str, *, timeout: float | None = None):
        self.calls.append(url)
        tag = self.lumen_tag if "/Lumen/" in url else self.cortex_tag
        if tag is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", _headers(None), io.BytesIO(b""))
        location = f"https://github.com/repo/x/releases/tag/{tag}"
        raise urllib.error.HTTPError(url, 302, "Found", _headers(location), io.BytesIO(b""))


class FailingOpener:
    def open(self, url: str, *, timeout: float | None = None):
        raise urllib.error.URLError("offline")


def _service(
    tmp_path: Path,
    *,
    runtime: FakeLumenRuntime,
    lumen_tag: str | None = None,
    cortex_tag: str | None = None,
) -> UpdateService:
    return UpdateService(
        lumen_runtime=runtime,  # type: ignore[arg-type]
        cache=UpdateCheckCache(tmp_path / "update-check.json"),
        opener=PerRepoOpener(lumen_tag=lumen_tag, cortex_tag=cortex_tag),
    )


# ---- stub installers --------------------------------------------------------


def _lumen_installer_stub(
    tmp_path: Path, *, version_file: Path, new_version: str = "0.4.0", exit_code: int = 0
) -> Path:
    """A stand-in for servelumen.com/install.sh: records its env + stdin type,
    'replaces the binary' by rewriting the version file, prints output lines."""
    record = tmp_path / "installer-env.txt"
    script = tmp_path / "stub-lumen-install.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        f'echo "LUMEN_TAG=${{LUMEN_TAG:-}}" > {record}\n'
        f'echo "LUMEN_MODEL=${{LUMEN_MODEL:-}}" >> {record}\n'
        f'echo "LUMEN_QUANT=${{LUMEN_QUANT:-}}" >> {record}\n'
        # The worker's stdin is the JSON-RPC pipe; the child's MUST be
        # /dev/null (a character device, never a fifo).
        f'echo "STDIN=$(stat -f %HT /dev/fd/0 2>/dev/null || echo unknown)" >> {record}\n'
        'echo "Installing Lumen ${LUMEN_TAG:-unknown}..."\n'
        'echo "Binaries installed."\n'
        f"if [ {exit_code} -eq 0 ]; then\n"
        f'  printf "%s" "{new_version}" > {version_file}\n'
        "fi\n"
        'echo "Done."\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _installer_record(tmp_path: Path) -> dict[str, str]:
    text = (tmp_path / "installer-env.txt").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.strip().splitlines())


# ---- status / notices ----------------------------------------------------------


def test_status_report_lines_available_and_no_releases(tmp_path: Path) -> None:
    runtime = FakeLumenRuntime(version="0.3.0")
    service = _service(tmp_path, runtime=runtime, lumen_tag="v0.4.0", cortex_tag=None)
    report = service.status_report()
    assert report["ok"] is True
    lines = str(report["message"]).splitlines()
    assert lines[0] == "Lumen: 0.3.0 installed · 0.4.0 available — /update lumen"
    assert lines[1] == (
        f"Cortex: {installed_cortex_version()} installed · no published releases yet"
    )


def test_status_report_up_to_date_and_not_installed(tmp_path: Path) -> None:
    runtime = FakeLumenRuntime(version="0.4.0")
    service = _service(tmp_path, runtime=runtime, lumen_tag="v0.4.0", cortex_tag=None)
    assert "Lumen: 0.4.0 installed · up to date" in str(service.status_report()["message"])

    missing = FakeLumenRuntime(version=None)
    service = _service(tmp_path, runtime=missing, lumen_tag="v0.4.0", cortex_tag=None)
    assert "Lumen: not installed" in str(service.status_report()["message"])


def test_status_report_falls_back_to_cache_when_probe_fails(tmp_path: Path) -> None:
    cache = UpdateCheckCache(tmp_path / "update-check.json")
    cache.store(lumen_latest="v0.4.0", cortex_latest=None)
    service = UpdateService(
        lumen_runtime=FakeLumenRuntime(version="0.3.0"),  # type: ignore[arg-type]
        cache=cache,
        opener=FailingOpener(),
    )
    message = str(service.status_report()["message"])
    assert "Lumen: 0.3.0 installed · 0.4.0 available — /update lumen" in message


def test_status_report_transient_failure_is_not_no_releases(tmp_path: Path) -> None:
    """Finding-3 regression: a network failure with a COLD cache must never be
    reported as the definitive 'no published releases yet' — that claim is
    reserved for GitHub's authoritative 404."""
    service = UpdateService(
        lumen_runtime=FakeLumenRuntime(version="0.3.0"),  # type: ignore[arg-type]
        cache=UpdateCheckCache(tmp_path / "update-check.json"),
        opener=FailingOpener(),
    )
    message = str(service.status_report()["message"])
    assert "no published releases yet" not in message
    assert "Lumen: 0.3.0 installed · could not determine the latest release" in message
    assert (
        f"Cortex: {installed_cortex_version()} installed · "
        "could not determine the latest release"
    ) in message


def test_startup_notice_only_when_strictly_newer(tmp_path: Path) -> None:
    newer = _service(
        tmp_path, runtime=FakeLumenRuntime(version="0.3.0"), lumen_tag="v0.4.0"
    )
    assert newer.startup_notice() == "Lumen 0.4.0 available — update with /update lumen"

    current = _service(
        tmp_path / "b", runtime=FakeLumenRuntime(version="0.4.0"), lumen_tag="v0.4.0"
    )
    (tmp_path / "b").mkdir(exist_ok=True)
    assert current.startup_notice() is None

    unknown_installed = _service(
        tmp_path / "c", runtime=FakeLumenRuntime(version=None), lumen_tag="v0.4.0"
    )
    assert unknown_installed.startup_notice() is None

    no_releases = _service(tmp_path / "d", runtime=FakeLumenRuntime(version="0.3.0"))
    assert no_releases.startup_notice() is None


def test_startup_notice_cortex_source_checkout_points_to_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding-2 regression: the startup nudge must never steer a
    source-checkout developer into /update cortex — the wheel install would
    replace their editable install."""
    service = _service(
        tmp_path, runtime=FakeLumenRuntime(version="0.4.0"), cortex_tag="v9999.0.0"
    )
    # This suite runs from the repo — a source checkout.
    notice = str(service.startup_notice())
    assert "Cortex 9999.0.0 released — source checkout: update with git pull" == notice
    assert "/update cortex" not in notice

    # Normal installs keep the /update cortex nudge.
    monkeypatch.setenv("CORTEX_SELF_INSTALL_KIND", "installed")
    installed = _service(
        tmp_path / "b", runtime=FakeLumenRuntime(version="0.4.0"), cortex_tag="v9999.0.0"
    )
    assert installed.startup_notice() == (
        "Cortex 9999.0.0 available — update with /update cortex"
    )


# ---- lumen update execution ---------------------------------------------------


def test_update_lumen_end_to_end_with_stub_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_file = tmp_path / "version.txt"
    version_file.write_text("0.3.0", encoding="utf-8")
    installer = _lumen_installer_stub(tmp_path, version_file=version_file)
    monkeypatch.setenv("CORTEX_LUMEN_INSTALLER_URL", str(installer))

    runtime = FakeLumenRuntime(version_file=version_file)
    runtime.ensure_server("qwen3-5-9b:q4_0")  # a managed server is running
    service = _service(tmp_path, runtime=runtime, lumen_tag="v0.4.0")

    phases: list[str] = []
    result = service.update_lumen(
        progress_callback=lambda payload: phases.append(str(payload["phase"]))
    )

    assert result["ok"] is True, result
    assert result["message"] == "local · Lumen 0.4.0 installed — server restarts on next use."
    # The running server was stopped BEFORE the installer replaced binaries.
    assert runtime.stopped == 1
    assert any("stopped lumen-server (qwen3-5-9b:q4_0)" in phase for phase in phases)
    # Installer output streamed as phases.
    assert any("Installing Lumen v0.4.0" in phase for phase in phases)
    assert "verifying installed version" in phases
    # Model pull pinned to the already-cached model; stdin was /dev/null.
    record = _installer_record(tmp_path)
    assert record["LUMEN_TAG"] == "v0.4.0"
    assert record["LUMEN_MODEL"] == "qwen3-5-9b"
    assert record["LUMEN_QUANT"] == "q4_0"
    assert record["STDIN"] == "Character Device"
    # The catalog was refreshed after the install.
    assert runtime.list_models_calls >= 2


def test_update_lumen_supports_file_url_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_file = tmp_path / "version.txt"
    version_file.write_text("0.3.0", encoding="utf-8")
    installer = _lumen_installer_stub(tmp_path, version_file=version_file)
    monkeypatch.setenv("CORTEX_LUMEN_INSTALLER_URL", f"file://{installer}")
    service = _service(
        tmp_path, runtime=FakeLumenRuntime(version_file=version_file), lumen_tag="v0.4.0"
    )
    assert service.update_lumen()["ok"] is True


def test_update_lumen_without_cached_model_warns_about_default_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_file = tmp_path / "version.txt"
    version_file.write_text("0.3.0", encoding="utf-8")
    installer = _lumen_installer_stub(tmp_path, version_file=version_file)
    monkeypatch.setenv("CORTEX_LUMEN_INSTALLER_URL", str(installer))

    runtime = FakeLumenRuntime(
        models=catalog(("qwen3-6-27b", "Q4_0", False)), version_file=version_file
    )
    service = _service(tmp_path, runtime=runtime, lumen_tag="v0.4.0")
    phases: list[str] = []
    result = service.update_lumen(
        progress_callback=lambda payload: phases.append(str(payload["phase"]))
    )
    assert result["ok"] is True
    assert any("installer will download Lumen's default model" in phase for phase in phases)
    record = _installer_record(tmp_path)
    assert record["LUMEN_MODEL"] == ""  # nothing pinned — installer defaults apply
    assert record["LUMEN_QUANT"] == ""


def test_update_lumen_up_to_date_and_probe_failure_short_circuit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No installer URL override needed: neither path may reach the installer.
    monkeypatch.setenv("CORTEX_LUMEN_INSTALLER_URL", str(tmp_path / "must-not-run.sh"))

    up_to_date = _service(
        tmp_path, runtime=FakeLumenRuntime(version="0.4.0"), lumen_tag="v0.4.0"
    )
    result = up_to_date.update_lumen()
    assert result == {"ok": True, "message": "Lumen 0.4.0 is up to date."}

    offline = UpdateService(
        lumen_runtime=FakeLumenRuntime(version="0.3.0"),  # type: ignore[arg-type]
        cache=UpdateCheckCache(tmp_path / "empty-cache.json"),
        opener=FailingOpener(),
    )
    result = offline.update_lumen()
    assert result["ok"] is False
    assert "Could not determine the latest Lumen release" in str(result["message"])


def test_update_lumen_fails_on_installer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_file = tmp_path / "version.txt"
    version_file.write_text("0.3.0", encoding="utf-8")
    installer = _lumen_installer_stub(tmp_path, version_file=version_file, exit_code=3)
    monkeypatch.setenv("CORTEX_LUMEN_INSTALLER_URL", str(installer))
    service = _service(
        tmp_path, runtime=FakeLumenRuntime(version_file=version_file), lumen_tag="v0.4.0"
    )
    result = service.update_lumen()
    assert result["ok"] is False
    assert "Lumen update failed" in str(result["message"])
    assert version_file.read_text(encoding="utf-8") == "0.3.0"  # nothing replaced


def test_update_lumen_fails_on_version_mismatch_after_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer 'succeeds' but the binary still reports the old version —
    the update must be reported as FAILED, never silently trusted."""
    version_file = tmp_path / "version.txt"
    version_file.write_text("0.3.0", encoding="utf-8")
    installer = _lumen_installer_stub(
        tmp_path, version_file=version_file, new_version="0.3.0"
    )
    monkeypatch.setenv("CORTEX_LUMEN_INSTALLER_URL", str(installer))
    service = _service(
        tmp_path, runtime=FakeLumenRuntime(version_file=version_file), lumen_tag="v0.4.0"
    )
    result = service.update_lumen()
    assert result["ok"] is False
    message = str(result["message"])
    assert "reports 0.3.0" in message
    assert "expected 0.4.0" in message


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
    service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag=None)
    result = service.update_cortex()
    assert result["ok"] is True
    assert str(result["message"]) == (
        f"Cortex has no published releases yet — you're on {installed_cortex_version()} "
        "(source install)."
    )


def test_update_cortex_transient_probe_failure_is_not_no_releases(tmp_path: Path) -> None:
    """Finding-3 regression: a transient probe failure with a cold cache must
    mirror the Lumen plan's honest 'check your network' answer, never the
    false factual claim 'no published releases yet'."""
    service = UpdateService(
        lumen_runtime=FakeLumenRuntime(),  # type: ignore[arg-type]
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
    """Finding-2 regression: this suite runs FROM the repo — a source
    checkout — so an actionable release must be refused: pip would replace
    the editable install (and can write through install.sh's site-packages
    symlink into the working tree). Nothing is downloaded or installed."""
    monkeypatch.delenv("CORTEX_SELF_INSTALL_KIND", raising=False)
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    monkeypatch.setenv("CORTEX_UPDATE_PROBE_BASE", "http://198.51.100.7")  # never contacted
    service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
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
        service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
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
        service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
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
        service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
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
        service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
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
    service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
    result = service.update_cortex()
    assert result["ok"] is False
    assert "refusing to download update artifacts" in str(result["message"])
    assert not (tmp_path / "pip-args.txt").exists()


def test_update_cortex_older_release_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORTEX_SELF_PIP", str(_stub_pip(tmp_path)))
    service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v0.0.1")
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
        service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
        result = service.update_cortex()
    assert result["ok"] is False
    assert "Cortex update failed" in str(result["message"])


# ---- shutdown safety: no orphaned installer process group -----------------------


def _sleeping_installer(tmp_path: Path, *, pid_file: Path) -> Path:
    """An installer that records its pid (== its process group, since it is
    spawned as a session leader) and then hangs with a background child —
    two group members, so only a GROUP kill reaps everything."""
    script = tmp_path / "sleeping-install.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$$" > {pid_file}\n'
        "sleep 30 &\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


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


def test_shutdown_terminates_in_flight_installer_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker-exit contract: shutdown() must reap the WHOLE installer process
    group (leader + children), and the interrupted update must come back as a
    loud failure — never a silent orphan continuing to replace binaries."""
    version_file = tmp_path / "version.txt"
    version_file.write_text("0.3.0", encoding="utf-8")
    pid_file = tmp_path / "installer.pid"
    installer = _sleeping_installer(tmp_path, pid_file=pid_file)
    monkeypatch.setenv("CORTEX_LUMEN_INSTALLER_URL", str(installer))

    service = _service(
        tmp_path, runtime=FakeLumenRuntime(version_file=version_file), lumen_tag="v0.4.0"
    )
    outcome: dict[str, dict] = {}
    thread = threading.Thread(
        target=lambda: outcome.update(result=service.update_lumen()), daemon=True
    )
    thread.start()
    _wait_for_file(pid_file)
    pgid = int(pid_file.read_text(encoding="utf-8").strip())

    service.shutdown()

    thread.join(timeout=10)
    assert not thread.is_alive(), "update thread wedged after shutdown"
    _assert_group_gone(pgid)
    result = outcome["result"]
    assert result["ok"] is False  # killed-mid-install fails LOUD
    assert "Lumen update failed" in str(result["message"])
    assert version_file.read_text(encoding="utf-8") == "0.3.0"  # nothing half-applied

    service.shutdown()  # idempotent: nothing left to reap, never raises


def test_shutdown_removes_downloaded_temp_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the installer was DOWNLOADED (curl → temp file), shutdown() must
    remove the temp file too — the update thread's finally never runs when
    the worker exits through os._exit."""
    temp_dir = tmp_path / "tmpdir"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))

    version_file = tmp_path / "version.txt"
    version_file.write_text("0.3.0", encoding="utf-8")
    pid_file = tmp_path / "installer.pid"
    installer_body = _sleeping_installer(tmp_path, pid_file=pid_file).read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(installer_body)))
            self.end_headers()
            self.wfile.write(installer_body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv(
            "CORTEX_LUMEN_INSTALLER_URL",
            f"http://127.0.0.1:{server.server_address[1]}/install.sh",
        )
        service = _service(
            tmp_path, runtime=FakeLumenRuntime(version_file=version_file), lumen_tag="v0.4.0"
        )
        thread = threading.Thread(target=service.update_lumen, daemon=True)
        thread.start()
        _wait_for_file(pid_file)
        pgid = int(pid_file.read_text(encoding="utf-8").strip())
        temp_installers = list(temp_dir.glob("cortex-installer-*.sh"))
        assert temp_installers, "downloaded installer temp file should exist mid-install"

        service.shutdown()

        thread.join(timeout=10)
        assert not thread.is_alive()
        _assert_group_gone(pgid)
        assert list(temp_dir.glob("cortex-installer-*.sh")) == []
    finally:
        server.shutdown()
        server.server_close()


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
        service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
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
        service = _service(tmp_path, runtime=FakeLumenRuntime(), cortex_tag="v9.9.9")
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
