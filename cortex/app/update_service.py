"""Update orchestration for the /update command family.

Three responsibilities, all built on cortex.update_check's probe/cache:

  - status: installed vs latest for both components (fresh probe with a short
    timeout, falling back to the last cached answer),
  - Lumen engine upgrade: stop the managed server, re-run the official
    installer pinned to the target tag (and to an already-cached model so its
    unconditional `lumen pull` is a no-op), verify `lumen --version`,
  - Cortex self-upgrade: only ever acts on a GitHub release whose tag parses
    GREATER than the running version, and installs that release's wheel asset
    (download → sha256 verification against the .sha256 sibling asset →
    `pip install <verified local wheel>` into the RUNNING venv). GitHub
    Releases is the single distribution channel — discovery and install
    source are the same release, so a discoverable version is installable by
    construction. The new code applies on the next start ("restart Cortex to
    apply").

The execution paths stream installer output through a progress callback using
the same payload shape as /download (kind "engine-update"), so the worker can
narrate them with the established background-operation pattern.

Self-update safety rules (asymmetric with the Lumen path on purpose):
  - Source checkouts refuse the wheel install: pip replacing an editable
    `pip install -e` dist (which install.sh wires through a site-packages
    symlink into the working tree) can overwrite a developer's checkout.
  - Worker shutdown never SIGNALS an in-flight self-install pip — pip
    rewrites cortex's own site-packages and a signal death skips its
    Python-level rollback, stranding a half-removed install the next launch
    cannot recover from. shutdown() WAITS (bounded) for that one child
    instead; every other child (curl, the Lumen installer) is still reaped
    immediately, where a kill only leaves a loud, recoverable state.

Test seams (production uses the HTTPS defaults):
  - CORTEX_LUMEN_INSTALLER_URL: the Lumen installer script (URL, file:// URL,
    or plain local file path).
  - CORTEX_UPDATE_PROBE_BASE (update_check's seam): base origin for BOTH the
    releases/latest discovery probe and the release asset downloads, so one
    local stub server serves the whole flow.
  - CORTEX_SELF_PIP: replaces `<running python> -m pip` for the wheel
    install, so suites never write the real venv.
  - CORTEX_SELF_INSTALL_KIND: overrides source-checkout detection
    ("installed" forces the normal wheel-update path) — the suite runs FROM
    this repo, which IS a checkout, and monkeypatching cannot cross the
    worker process boundary.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path
from typing import Callable, Dict, Optional

from cortex.lumen_runtime import LumenModel, LumenRuntime
from cortex.update_check import (
    CORTEX_REPO,
    UpdateCheckCache,
    UpdateCheckResult,
    UrlOpener,
    check_for_updates,
    installed_cortex_version,
    is_newer,
    normalize_version,
    probe_base_url,
)

logger = logging.getLogger(__name__)

DEFAULT_LUMEN_INSTALLER_URL = "https://servelumen.com/install.sh"
LUMEN_INSTALLER_URL_ENV = "CORTEX_LUMEN_INSTALLER_URL"

# The release wheel's asset name is deterministic: the PEP 427-normalized
# project name (cortex-llm → cortex_llm) plus the platform tag declared in
# setup.cfg [bdist_wheel] plat_name. Cross-pinned by tests/test_packaging.py:
# drift against setup.cfg would 404 every /update cortex.
CORTEX_WHEEL_ASSET_TEMPLATE = "cortex_llm-{version}-py3-none-macosx_13_0_arm64.whl"

# Test-only: replaces `<running python> -m pip` for the self-update wheel
# install. Monkeypatching cannot cross the worker process boundary, and a
# real pip would write the suite's own venv.
CORTEX_PIP_ENV = "CORTEX_SELF_PIP"

# Test-only: overrides _source_checkout_root's detection ("installed" → treat
# as a normal wheel install). Only weakens a developer-protection guard —
# hostile env control already owns CORTEX_SELF_PIP outright.
CORTEX_INSTALL_KIND_ENV = "CORTEX_SELF_INSTALL_KIND"

# /update status wants live answers but must stay snappy on a bad network.
STATUS_PROBE_TIMEOUT_SECONDS = 2.5
INSTALLER_DOWNLOAD_TIMEOUT_SECONDS = 120
# The wheel bundles the 65MB TUI sidecar (~23MB compressed) — give slow links
# more room than a small installer script gets.
WHEEL_DOWNLOAD_TIMEOUT_SECONDS = 600
# How long shutdown() waits for an in-flight SELF-INSTALL pip to finish
# before the last-resort SIGKILL. A local-wheel install takes seconds; the
# bound only bites when pip wedges (e.g. hung dependency download), where
# corruption risk is accepted over an unbounded worker exit — recovery is
# re-running install.sh.
SELF_INSTALL_REAP_GRACE_SECONDS = 60.0

ProgressCallback = Callable[[Dict[str, object]], None]


def _source_checkout_root() -> Optional[Path]:
    """The repository root when Cortex runs from a source checkout, else None.

    CORTEX_SELF_INSTALL_KIND=installed (test-only) forces the not-a-checkout
    answer so suites — which always run from this repo, a checkout — can
    exercise the normal wheel-update path.
    """
    if os.environ.get(CORTEX_INSTALL_KIND_ENV, "").strip().lower() == "installed":
        return None
    try:
        import cortex

        root = Path(cortex.__file__).resolve().parent.parent
    except Exception:
        return None
    if (root / "pyproject.toml").exists() and (root / "install.sh").exists():
        return root
    return None


def _untrusted_asset_base(base: str) -> Optional[str]:
    """Why release assets must NOT be downloaded from ``base``, or None.

    The .sha256 sibling shares the wheel's origin, so checksum verification is
    worthless against a hostile base — only the real https GitHub origin may
    serve update artifacts. Loopback http is allowed for the documented
    TEST-ONLY probe-base seam (local stub servers).
    """
    try:
        parts = urllib.parse.urlsplit(base)
    except ValueError:
        return f"unparseable release base {base!r}"
    if parts.scheme == "https" and parts.hostname == "github.com":
        return None
    if parts.scheme in ("http", "https") and parts.hostname in ("127.0.0.1", "localhost", "::1"):
        return None
    return (
        f"release asset base {base!r} is not https://github.com — "
        "refusing to download update artifacts from it"
    )


def _sha256_verification_error(artifact: Path, sha_path: Path) -> Optional[str]:
    """None when ``artifact`` matches its .sha256 sibling, else the reason.

    The sibling is shasum's standard ``<64-hex>  <filename>`` line; only the
    FIRST whitespace token is consumed (filename-independent, like Lumen's
    installer). Anything unparseable refuses — never installs.
    """
    try:
        tokens = sha_path.read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError):
        return f"unreadable checksum file {sha_path.name} — refusing to install"
    want = tokens[0].lower() if tokens else ""
    if len(want) != 64 or any(char not in "0123456789abcdef" for char in want):
        return f"malformed checksum in {sha_path.name} — refusing to install"
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return f"could not hash {artifact.name}: {exc}"
    got = digest.hexdigest()
    if got != want:
        return f"checksum mismatch for {artifact.name} (want {want}, got {got}) — installing nothing"
    return None


class UpdateService:
    """Backend for /update: release discovery plus installer execution."""

    def __init__(
        self,
        *,
        lumen_runtime: LumenRuntime,
        cache: UpdateCheckCache | None = None,
        opener: UrlOpener | None = None,
    ) -> None:
        self.lumen_runtime = lumen_runtime
        self.cache = cache or UpdateCheckCache()
        self.opener = opener
        # Live child (curl / installer bash) and downloaded temp installer.
        # Children run in their OWN process group (start_new_session), and the
        # worker's shutdown paths call shutdown() — the update thread is a
        # daemon and both worker exits go through os._exit, so finally-based
        # cleanup inside update_lumen/update_cortex never runs there.
        self._child_lock = threading.Lock()
        self._active_child: subprocess.Popen | None = None
        # True while the active child is the SELF-INSTALL pip: killing that
        # one can corrupt the running venv, so shutdown() waits it out.
        self._active_child_critical = False
        self._active_temp_installer: Path | None = None
        self._closing = False

    # ---- shutdown safety ---------------------------------------------------

    def _register_child(self, process: subprocess.Popen, *, critical: bool = False) -> None:
        with self._child_lock:
            self._active_child = process
            self._active_child_critical = critical
            closing = self._closing
        if closing:
            # shutdown() already ran (or is mid-flight): a child spawned past
            # its snapshot must not escape the reap — kill it immediately.
            # Safe even for the critical pip: registration happens right
            # after Popen, long before pip's interpreter has booted, so it
            # cannot have started mutating site-packages yet ( _run_installer
            # additionally refuses to spawn at all once closing is set).
            self._signal_group(process, signal.SIGTERM)

    def _clear_child(self, process: subprocess.Popen) -> None:
        with self._child_lock:
            if self._active_child is process:
                self._active_child = None
                self._active_child_critical = False

    def _register_temp_installer(self, path: Path) -> None:
        with self._child_lock:
            self._active_temp_installer = path

    def _clear_temp_installer(self, path: Path) -> None:
        with self._child_lock:
            if self._active_temp_installer == path:
                self._active_temp_installer = None

    @staticmethod
    def _signal_group(process: subprocess.Popen, signum: int) -> None:
        """Signal the child's whole process group (it is a session leader —
        start_new_session — so pid == pgid), falling back to the child alone."""
        try:
            os.killpg(process.pid, signum)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.send_signal(signum)
            except (ProcessLookupError, OSError):
                pass

    def shutdown(
        self,
        *,
        timeout: float = 2.0,
        self_install_grace: float = SELF_INSTALL_REAP_GRACE_SECONDS,
    ) -> None:
        """Resolve any in-flight installer/downloader child and remove the
        staged temp artifacts. Called from the worker's shutdown paths
        (signal handler and stdin-EOF finally) right before os._exit: without
        it a mid-update child would be orphaned. Idempotent, never raises.

        Two-tier policy, because the children differ in what a kill costs:

          - Ordinary children (curl downloads, the Lumen installer bash):
            SIGTERM the group, wait `timeout`, then SIGKILL. A kill here
            leaves a LOUD but recoverable state — Cortex itself stays
            runnable (Lumen is an external binary, verified by the next
            /update lumen; an interrupted download installs nothing).
          - The SELF-INSTALL pip (critical child): NEVER signaled first. It
            is rewriting cortex-llm inside the very venv Cortex launches
            from, and a signal death skips pip's Python-level uninstall
            rollback — the venv would be left half-removed and the next
            launch (and thus any in-app recovery) impossible. shutdown()
            WAITS up to `self_install_grace` for it to finish; only a pip
            wedged past that bound is SIGKILLed as a last resort (recovery:
            re-run install.sh).
        """
        try:
            with self._child_lock:
                self._closing = True  # children registered from now on self-reap
                process = self._active_child
                critical = self._active_child_critical
                temp_path = self._active_temp_installer
            if process is not None and process.poll() is None:
                if critical:
                    # Self-install pip: wait for completion — a completed
                    # install is exactly the "restart Cortex to apply" state.
                    try:
                        process.wait(timeout=self_install_grace)
                    except subprocess.TimeoutExpired:
                        self._signal_group(process, signal.SIGKILL)
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                else:
                    self._signal_group(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        self._signal_group(process, signal.SIGKILL)
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
            if temp_path is not None:
                try:
                    if temp_path.is_dir():
                        # The cortex self-update stages wheel + .sha256 in a
                        # temp DIRECTORY; unlink() cannot remove those.
                        shutil.rmtree(temp_path, ignore_errors=True)
                    else:
                        temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception:  # pragma: no cover - shutdown must never raise
            logger.debug("update-service shutdown cleanup failed", exc_info=True)

    # ---- discovery -------------------------------------------------------

    def startup_notice(self) -> Optional[str]:
        """One-line update notice for session start, or None when everything
        is current, unknown, or unreleased. Cache-honoring: at most one real
        probe per TTL window."""
        result = check_for_updates(cache=self.cache, opener=self.opener)
        parts: list[str] = []
        lumen_installed = self.lumen_runtime.installed_version()
        if is_newer(result.lumen_latest, lumen_installed):
            parts.append(
                f"Lumen {normalize_version(result.lumen_latest)} available — "
                "update with /update lumen"
            )
        cortex_installed = installed_cortex_version()
        if is_newer(result.cortex_latest, cortex_installed):
            if _source_checkout_root() is not None:
                # Never steer a source-checkout developer into /update cortex:
                # the wheel install would replace their editable install.
                parts.append(
                    f"Cortex {normalize_version(result.cortex_latest)} released — "
                    "source checkout: update with git pull"
                )
            else:
                parts.append(
                    f"Cortex {normalize_version(result.cortex_latest)} available — "
                    "update with /update cortex"
                )
        return " · ".join(parts) if parts else None

    def _fresh_or_cached_result(self) -> UpdateCheckResult:
        """A live probe (short timeout). check_for_updates' per-component
        merge keeps the previously cached tag for any component whose probe
        fails transiently; the result's ``*_resolved`` flags say whether each
        None tag is authoritative ("no releases") or just unknown."""
        return check_for_updates(
            cache=self.cache,
            opener=self.opener,
            timeout=STATUS_PROBE_TIMEOUT_SECONDS,
            force=True,
        )

    @staticmethod
    def _status_line(
        component: str,
        installed: Optional[str],
        latest_tag: Optional[str],
        *,
        resolved: bool,
    ) -> str:
        name = component.capitalize()
        if installed is None:
            return f"{name}: not installed"
        latest = normalize_version(latest_tag) if latest_tag else None
        if latest is None:
            if resolved and latest_tag is None:
                # Authoritative answer (GitHub 404): the repo has no releases.
                return f"{name}: {installed} installed · no published releases yet"
            # Transient probe failure (or an unparseable tag): the latest is
            # UNKNOWN — never claim "no releases" for a network failure.
            return f"{name}: {installed} installed · could not determine the latest release"
        if is_newer(latest, installed):
            return f"{name}: {installed} installed · {latest} available — /update {component}"
        return f"{name}: {installed} installed · up to date"

    def status_report(self) -> Dict[str, object]:
        result = self._fresh_or_cached_result()
        lumen_latest, cortex_latest = result.lumen_latest, result.cortex_latest
        lumen_installed = self.lumen_runtime.installed_version()
        cortex_installed = installed_cortex_version()
        lines = [
            self._status_line(
                "lumen", lumen_installed, lumen_latest, resolved=result.lumen_resolved
            ),
            self._status_line(
                "cortex", cortex_installed, cortex_latest, resolved=result.cortex_resolved
            ),
        ]
        return {
            "ok": True,
            "message": "\n".join(lines),
            "update_status": {
                "lumen_installed": lumen_installed,
                "lumen_latest": normalize_version(lumen_latest) if lumen_latest else None,
                "cortex_installed": cortex_installed,
                "cortex_latest": normalize_version(cortex_latest) if cortex_latest else None,
            },
        }

    # ---- planning ---------------------------------------------------------

    def plan_lumen_update(self) -> Dict[str, object]:
        """Decide whether a Lumen update is actionable (no side effects)."""
        lumen_latest = self._fresh_or_cached_result().lumen_latest
        target_version = normalize_version(lumen_latest) if lumen_latest else None
        if lumen_latest is None or target_version is None:
            return {
                "ok": False,
                "update_available": False,
                "message": (
                    "Could not determine the latest Lumen release — "
                    "check your network and try again."
                ),
            }
        installed = self.lumen_runtime.installed_version()
        if installed is not None and not is_newer(lumen_latest, installed):
            return {
                "ok": True,
                "update_available": False,
                "message": f"Lumen {installed} is up to date.",
            }
        return {
            "ok": True,
            "update_available": True,
            "target_tag": lumen_latest,
            "target_version": target_version,
            "message": f"Updating Lumen to {target_version}…",
        }

    def plan_cortex_update(self) -> Dict[str, object]:
        """Decide whether a Cortex self-update is actionable (no side effects).

        Only a GitHub release tag parsing GREATER than the running version is
        actionable — anything else reports and stops (never a fallback to an
        unpinned install). Source checkouts refuse even an actionable release:
        pip replacing the editable install (wired through a site-packages
        symlink by install.sh's source mode) can overwrite working-tree files.
        """
        installed = installed_cortex_version()
        result = self._fresh_or_cached_result()
        cortex_latest = result.cortex_latest
        target_version = normalize_version(cortex_latest) if cortex_latest else None
        checkout = _source_checkout_root()
        if target_version is None:
            if result.cortex_resolved and cortex_latest is None:
                # GitHub answered authoritatively: the repo has no releases.
                install_kind = "source install" if checkout else "installed"
                return {
                    "ok": True,
                    "update_available": False,
                    "message": (
                        "Cortex has no published releases yet — "
                        f"you're on {installed or 'an unknown version'} ({install_kind})."
                    ),
                }
            # Transient probe failure (or an unparseable tag): mirror the
            # Lumen plan's honest wording — a network failure must never be
            # presented as the factual claim "no releases exist".
            return {
                "ok": False,
                "update_available": False,
                "message": (
                    "Could not determine the latest Cortex release — "
                    "check your network and try again."
                ),
            }
        if not is_newer(cortex_latest, installed):
            return {
                "ok": True,
                "update_available": False,
                "message": f"Cortex {installed or target_version} is up to date.",
            }
        if checkout is not None:
            return {
                "ok": False,
                "update_available": False,
                "message": (
                    f"Cortex {target_version} is available, but this is a source "
                    f"checkout ({checkout}) — /update cortex would replace your "
                    "editable install with the release wheel. Update with `git pull` "
                    "(re-run ./install.sh if dependencies changed)."
                ),
            }
        return {
            "ok": True,
            "update_available": True,
            # RECONSTRUCTED tag, never the raw redirect tag: the probe's
            # Location header is unquoted after splitting, so a hostile
            # redirect could smuggle path characters through the raw tag.
            # Only normalize_version() output is ever spliced into URLs.
            "target_tag": f"v{target_version}",
            "target_version": target_version,
            "message": f"Updating Cortex to {target_version}…",
        }

    # ---- execution ---------------------------------------------------------

    def update_lumen(
        self, *, progress_callback: ProgressCallback | None = None
    ) -> Dict[str, object]:
        """Run the full Lumen upgrade; blocks until done (the worker wraps
        this in its background-operation narration)."""

        def emit(phase: str) -> None:
            if progress_callback is not None:
                progress_callback(
                    {"kind": "engine-update", "repo_id": "lumen", "phase": phase}
                )

        plan = self.plan_lumen_update()
        if not bool(plan.get("ok")) or not bool(plan.get("update_available")):
            return {"ok": bool(plan.get("ok")), "message": str(plan.get("message", ""))}
        target_tag = str(plan["target_tag"])
        target_version = str(plan["target_version"])

        # The installer replaces binaries via install(1) — a new inode — so a
        # running server would keep serving the OLD version forever. Stop it
        # first; it lazily reboots on next use.
        active_selector = self.lumen_runtime.active_selector()
        self.lumen_runtime.stop()
        if active_selector:
            emit(f"stopped lumen-server ({active_selector}) — it restarts on next use")

        env = dict(os.environ)
        env["LUMEN_TAG"] = target_tag
        pinned = self._first_cached_model()
        if pinned is not None:
            # The installer ALWAYS ends with `lumen pull $LUMEN_MODEL:$LUMEN_QUANT`
            # (multi-GB default). Pinning to an already-cached model makes that
            # pull a no-op.
            env["LUMEN_MODEL"] = pinned.name
            env["LUMEN_QUANT"] = pinned.quant.lower()
            emit(f"pinning installer model to cached {pinned.selector} (no new download)")
        else:
            emit("no cached model found — the installer will download Lumen's default model")

        installer_url = os.environ.get(LUMEN_INSTALLER_URL_ENV, "").strip() or (
            DEFAULT_LUMEN_INSTALLER_URL
        )
        emit(f"fetching installer ({installer_url})")
        installer_path, cleanup, fetch_error = self._fetch_installer(installer_url)
        if installer_path is None:
            return {"ok": False, "message": f"Lumen update failed: {fetch_error}"}
        try:
            ok, detail = self._run_installer(
                ["bash", str(installer_path), "--yes"], env=env, on_line=emit
            )
        finally:
            cleanup()
        if not ok:
            return {"ok": False, "message": f"Lumen update failed: {detail}"}

        emit("verifying installed version")
        reported = self.lumen_runtime.installed_version()
        if reported != target_version:
            return {
                "ok": False,
                "message": (
                    "Lumen update failed: the installer finished but `lumen --version` "
                    f"reports {reported or 'nothing parseable'} (expected {target_version})."
                ),
            }
        # Refresh the catalog so pickers reflect whatever the new engine offers.
        try:
            self.lumen_runtime.list_models()
        except Exception:
            logger.debug("post-update model list refresh failed", exc_info=True)
        return {
            "ok": True,
            "message": f"local · Lumen {target_version} installed — server restarts on next use.",
            "update": {"component": "lumen", "version": target_version},
        }

    def update_cortex(
        self, *, progress_callback: ProgressCallback | None = None
    ) -> Dict[str, object]:
        """Run the Cortex self-upgrade, pinned to the released version.

        Single-channel flow: download the release's wheel asset and its
        .sha256 sibling into a private temp directory, verify the checksum
        (any failure installs NOTHING), then `pip install <verified local
        wheel>` into the RUNNING venv. The running process keeps executing
        the old code — hence the "restart Cortex to apply" contract.
        """

        def emit(phase: str) -> None:
            if progress_callback is not None:
                progress_callback(
                    {"kind": "engine-update", "repo_id": "cortex", "phase": phase}
                )

        plan = self.plan_cortex_update()
        if not bool(plan.get("ok")) or not bool(plan.get("update_available")):
            return {"ok": bool(plan.get("ok")), "message": str(plan.get("message", ""))}
        target_version = str(plan["target_version"])
        target_tag = str(plan["target_tag"])

        base = probe_base_url()
        refusal = _untrusted_asset_base(base)
        if refusal is not None:
            return {"ok": False, "message": f"Cortex update failed: {refusal}."}
        asset_name = CORTEX_WHEEL_ASSET_TEMPLATE.format(version=target_version)
        wheel_url = f"{base}/{CORTEX_REPO}/releases/download/{target_tag}/{asset_name}"

        # Wheel + .sha256 live in one tracked temp dir: shutdown() rmtree's it
        # if the worker exits mid-update (the update thread is a daemon, so
        # this finally never runs through os._exit).
        temp_dir = Path(tempfile.mkdtemp(prefix="cortex-wheel-"))
        self._register_temp_installer(temp_dir)
        try:
            wheel_path = temp_dir / asset_name
            sha_path = temp_dir / f"{asset_name}.sha256"

            emit(f"downloading {asset_name}")
            error = self._download_file(
                wheel_url, wheel_path, timeout=WHEEL_DOWNLOAD_TIMEOUT_SECONDS
            )
            if error:
                return {
                    "ok": False,
                    "message": (
                        f"Cortex update failed: {error} — does release {target_tag} "
                        "have a wheel asset?"
                    ),
                }
            error = self._download_file(
                f"{wheel_url}.sha256",
                sha_path,
                timeout=INSTALLER_DOWNLOAD_TIMEOUT_SECONDS,
            )
            if error:
                return {
                    "ok": False,
                    "message": (
                        f"Cortex update failed: no checksum asset for {asset_name} "
                        f"({error}) — refusing to install an unverified wheel."
                    ),
                }

            emit("verifying checksum")
            verification_error = _sha256_verification_error(wheel_path, sha_path)
            if verification_error is not None:
                return {"ok": False, "message": f"Cortex update failed: {verification_error}."}

            emit(f"installing {asset_name} into the running environment")
            # critical=True: this pip rewrites cortex-llm inside the running
            # venv — shutdown() must wait it out, never kill it mid-mutation.
            ok, detail = self._run_installer(
                self._pip_install_command(wheel_path),
                env=dict(os.environ),
                on_line=emit,
                critical=True,
            )
            if not ok:
                return {"ok": False, "message": f"Cortex update failed: {detail}"}
        finally:
            self._clear_temp_installer(temp_dir)
            shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            "ok": True,
            "message": f"Cortex {target_version} installed — restart Cortex to apply.",
            "update": {"component": "cortex", "version": target_version},
        }

    # ---- helpers -------------------------------------------------------------

    def _first_cached_model(self) -> Optional[LumenModel]:
        try:
            for model in self.lumen_runtime.list_models():
                if model.cached:
                    return model
        except Exception:
            logger.debug("cached-model lookup for installer pin failed", exc_info=True)
        return None

    def _fetch_installer(
        self,
        source: str,
    ) -> tuple[Optional[Path], Callable[[], None], str]:
        """Materialize the installer script as a local file.

        Returns (path, cleanup, error). A plain local path or file:// URL is
        used in place (test seam — never registered for deletion); anything
        else is downloaded to a temp file with `curl -fsSL` — stdin detached
        so a misbehaving child can never touch the worker's JSON-RPC pipe.
        The temp path and the curl child are tracked so shutdown() can reap
        them if the worker exits mid-download.
        """
        def no_cleanup() -> None:
            return None

        if source.startswith("file://"):
            local = Path(urllib.parse.urlsplit(source).path)
            if local.is_file():
                return local, no_cleanup, ""
            return None, no_cleanup, f"installer not found at {local}"
        candidate = Path(source).expanduser()
        if candidate.is_file():
            return candidate, no_cleanup, ""

        handle = tempfile.NamedTemporaryFile(
            mode="wb", prefix="cortex-installer-", suffix=".sh", delete=False
        )
        temp_path = Path(handle.name)
        handle.close()
        self._register_temp_installer(temp_path)

        def cleanup() -> None:
            self._clear_temp_installer(temp_path)
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

        error = self._download_file(
            source, temp_path, timeout=INSTALLER_DOWNLOAD_TIMEOUT_SECONDS
        )
        if error:
            cleanup()
            return None, no_cleanup, f"installer {error}"
        return temp_path, cleanup, ""

    def _download_file(self, url: str, destination: Path, *, timeout: float) -> str:
        """Download ``url`` to ``destination`` with curl; '' on success, else
        the failure reason (prefixed "download failed: ").

        `-L` is load-bearing: GitHub release asset URLs answer 302 to a
        signed, time-limited CDN URL. Stdin is detached (the worker's stdin
        is the JSON-RPC pipe) and the curl child runs in its own process
        group, tracked so shutdown() can reap a mid-flight download.
        """
        try:
            process = subprocess.Popen(
                ["curl", "-fsSL", url, "-o", str(destination)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # own group: shutdown() can reap it
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"download failed: {exc}"
        self._register_child(process)
        try:
            try:
                _stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(process, signal.SIGKILL)
                process.communicate()
                return "download failed: timed out"
        finally:
            self._clear_child(process)
        if process.returncode != 0:
            detail = (stderr or "").strip().splitlines()
            reason = detail[-1] if detail else f"curl exited with code {process.returncode}"
            return f"download failed: {reason}"
        return ""

    @staticmethod
    def _pip_install_command(wheel_path: Path) -> list[str]:
        """`<running python> -m pip install <verified wheel>` — the worker's
        sys.executable IS the running venv's python, so the wheel lands in
        the environment Cortex restarts from. pip upgrades AND downgrades
        from a local wheel without extra flags; never add --force-reinstall
        (it would reinstall every dependency and uninstall the running
        version even when equal). CORTEX_SELF_PIP (test-only) swaps in a
        stub pip so suites never write a real venv."""
        override = os.environ.get(CORTEX_PIP_ENV, "").strip()
        if override:
            return [override, "install", str(wheel_path)]
        return [sys.executable, "-m", "pip", "install", str(wheel_path)]

    def _run_installer(
        self,
        command: list[str],
        *,
        env: Dict[str, str],
        on_line: Callable[[str], None],
        critical: bool = False,
    ) -> tuple[bool, str]:
        """Run an installer, streaming stdout lines to `on_line`.

        CHILD STDIN IS /dev/null BY CONTRACT: the worker's stdin is the
        JSON-RPC transport — an installer that prompts would otherwise eat
        protocol bytes. The child runs in its OWN process group
        (start_new_session, mirroring the bash tool's setsid pattern) and is
        tracked on the service, so the worker's shutdown paths can resolve it
        instead of orphaning it mid-install. ``critical=True`` marks a child
        whose kill can corrupt the running venv (the self-update's pip):
        shutdown() waits for it instead of signaling it.
        """
        with self._child_lock:
            if self._closing:
                # shutdown() already ran: starting a new installer now would
                # escape its snapshot (and, for the critical pip, begin a venv
                # mutation nobody will wait for).
                return False, "worker is shutting down"
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"failed to launch installer: {exc}"
        self._register_child(process, critical=critical)
        try:
            last_line = ""
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if line:
                    last_line = line
                    on_line(line)
            code = process.wait()
        finally:
            self._clear_child(process)
        if code != 0:
            return False, last_line or f"installer exited with code {code}"
        return True, last_line
