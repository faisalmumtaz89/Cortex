"""Lumen inference-engine runtime management.

Cortex delegates ALL local inference to Lumen (https://github.com/faisalmumtaz89/Lumen):
a single Rust binary that runs Qwen3.5/3.6-family models GPU-resident on Apple
Silicon and serves an OpenAI-compatible HTTP API with native tool calls.

This module owns the full lifecycle so the user never touches Lumen directly:
  - binary discovery (`lumen`, `lumen-server`) with a friendly install hint
  - the supported-model catalog (parsed from `lumen models` — cached vs available)
  - `lumen-server` process management: start, readiness poll, stop, model switch
    (one model per server process — switching restarts the server)
  - model downloads via `lumen pull` with line-level progress forwarding

Local turns then reuse Cortex's OpenAI-compatible streaming client pointed at
the managed server's base URL, so local and cloud models share one agent loop.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

INSTALL_HINT = "Install Lumen: curl -fsSL https://servelumen.com/install.sh | bash"

# ensure_server's refusal while another model is mid-boot starts with this
# prefix. The orchestrator keys on it to surface the message verbatim (it is
# already user-actionable) instead of wrapping it as a startup failure.
SWITCH_IN_FLIGHT_PREFIX = "Model is switching to "

_QUANTS = ("BF16", "F16", "Q8_0", "Q4_0")
# Cached entries print as "<model>-<QUANT>[ -suffix]", e.g. "qwen3-5-9b-Q4_0".
_CACHED_LINE = re.compile(r"^\s{2,}(\S+)\s+([\d.]+\s*[KMGT]?B)\s*$")
# Available entries print as "<model>  <Display Name ...> <QUANT>".
_AVAILABLE_LINE = re.compile(r"^\s{2,}(\S+)\s{2,}(.+?)\s+(BF16|F16|Q8_0|Q4_0)\s*$")


@dataclass(frozen=True)
class LumenModel:
    """One (model, quant) pair from Lumen's registry."""

    name: str  # registry name as printed by `lumen models`, e.g. "qwen3-5-9b"
    quant: str  # upper-case quant tag, e.g. "Q4_0"
    cached: bool
    display_name: str = ""
    size: str = ""  # human size for cached entries, e.g. "5.4 GB"

    @property
    def selector(self) -> str:
        """Cortex-facing selector, e.g. "qwen3-5-9b:q4_0"."""
        return f"{self.name}:{self.quant.lower()}"


def parse_selector(selector: str) -> Tuple[str, str]:
    """Split "name:quant" (quant optional → Lumen's default Q8_0)."""
    raw = selector.strip()
    if ":" in raw:
        name, quant = raw.split(":", 1)
        return name.strip(), quant.strip().upper() or "Q8_0"
    return raw, "Q8_0"


def parse_models_output(output: str) -> List[LumenModel]:
    """Parse `lumen models` text output into a model list.

    Draft models (speculative-decode helpers, "-draft" in the stem) are not
    chat models and are filtered out.
    """
    models: List[LumenModel] = []
    section = ""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Cached models"):
            section = "cached"
            continue
        if stripped.startswith("Available to download"):
            section = "available"
            continue
        if stripped.startswith("Download with"):
            section = ""
            continue
        if not stripped:
            continue

        if section == "cached":
            match = _CACHED_LINE.match(line)
            if not match:
                continue
            stem, size = match.group(1), match.group(2).strip()
            if "-draft" in stem:
                continue
            name, quant = _split_cached_stem(stem)
            if name is None or quant is None:
                continue
            models.append(LumenModel(name=name, quant=quant, cached=True, size=size))
        elif section == "available":
            match = _AVAILABLE_LINE.match(line)
            if not match:
                continue
            name, display, quant = match.group(1), match.group(2).strip(), match.group(3)
            models.append(
                LumenModel(name=name, quant=quant, cached=False, display_name=display)
            )
    return models


def _split_cached_stem(stem: str) -> Tuple[Optional[str], Optional[str]]:
    """"qwen3-5-9b-Q4_0" → ("qwen3-5-9b", "Q4_0")."""
    for quant in _QUANTS:
        marker = f"-{quant}"
        if stem.endswith(marker):
            return stem[: -len(marker)], quant
        # Tolerate a platform suffix after the quant (e.g. "...-Q4_0-metal").
        index = stem.find(marker + "-")
        if index > 0:
            return stem[:index], quant
    return None, None


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def lumen_cache_dir() -> Path:
    """Lumen's model cache, mirroring its own resolution order:
    $LUMEN_CACHE_DIR → $XDG_CACHE_HOME/lumen → ~/.cache/lumen."""
    override = os.environ.get("LUMEN_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "lumen"
    return Path.home() / ".cache" / "lumen"


def partial_download_bytes() -> int:
    """Bytes written so far by an in-flight `lumen pull` (its downloader
    streams into `*.part` files inside the cache, renamed on completion).
    `lumen pull` renders its progress bar only on a TTY — through a pipe we
    get no byte lines — so Cortex measures real progress from the cache."""
    total = 0
    root = lumen_cache_dir()
    try:
        for part in root.rglob("*.part"):
            try:
                total += part.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


@dataclass
class LumenServerState:
    selector: str
    port: int
    # The OS process backing this state. None only inside ensure_server's
    # claim window: the placeholder is installed first (making the boot
    # exclusive and its selector visible) and the outgoing server's teardown
    # plus the replacement's spawn then run OUTSIDE the runtime lock.
    process: Optional[subprocess.Popen] = None
    started_at: float = field(default_factory=time.time)
    # Boot synchronization: `ready` is set once /v1/models answers (or the boot
    # fails — then `error` says why). Concurrent ensure_server callers wait on
    # it instead of double-starting or racing a half-booted server.
    ready: threading.Event = field(default_factory=threading.Event)
    error: Optional[str] = None


class LumenRuntime:
    """Owns the lumen binaries and the managed lumen-server process."""

    def __init__(
        self,
        *,
        binary: str = "lumen",
        server_binary: str = "lumen-server",
        port: int = 0,
        context_len: int = 0,
        startup_timeout_seconds: int = 180,
        log_level: str = "warn",
        log_path: Optional[Path] = None,
    ) -> None:
        self.binary = binary
        self.server_binary = server_binary
        self.port_config = port
        self.context_len = context_len
        self.startup_timeout_seconds = startup_timeout_seconds
        self.log_level = log_level
        self.log_path = log_path or (Path.home() / ".cortex" / "lumen-server.log")
        self._server: Optional[LumenServerState] = None
        self._lock = threading.RLock()
        atexit.register(self.stop)

    # ---- availability -------------------------------------------------

    def resolve_binary(self, name: str) -> Optional[str]:
        path = Path(name).expanduser()
        if path.is_absolute() and path.exists():
            return str(path)
        return shutil.which(name)

    def available(self) -> Tuple[bool, str]:
        """Are both Lumen binaries present?"""
        cli = self.resolve_binary(self.binary)
        server = self.resolve_binary(self.server_binary)
        if cli and server:
            return True, ""
        missing = self.binary if not cli else self.server_binary
        return False, f"Lumen binary not found: {missing}. {INSTALL_HINT}"

    # ---- catalog -------------------------------------------------------

    def list_models(self) -> List[LumenModel]:
        """Supported local models straight from `lumen models`."""
        cli = self.resolve_binary(self.binary)
        if cli is None:
            return []
        result = subprocess.run(
            [cli, "models"], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return []
        return parse_models_output(result.stdout)

    # ---- server lifecycle ----------------------------------------------

    @property
    def server(self) -> Optional[LumenServerState]:
        with self._lock:
            state = self._server
            if (
                state is not None
                and state.process is not None
                and state.process.poll() is not None
            ):
                self._server = None  # crashed / exited underneath us
                return None
            return state

    def base_url(self) -> Optional[str]:
        state = self.server
        if state is None:
            return None
        return f"http://127.0.0.1:{state.port}/v1"

    def active_selector(self) -> Optional[str]:
        state = self.server
        return state.selector if state else None

    def serving_selector(self) -> Optional[str]:
        """Selector of a READY server (None while still booting)."""
        state = self.server
        return state.selector if state is not None and state.ready.is_set() else None

    def starting_selector(self) -> Optional[str]:
        """Selector of a server that is still booting (loading into GPU memory)."""
        state = self.server
        if state is None or state.ready.is_set():
            return None
        return state.selector

    def ensure_server(self, selector: str) -> Tuple[bool, str]:
        """Start (or switch to) `selector`; blocks until the API is ready.

        Thread-safe: if another caller is already booting the SAME selector,
        this waits for that boot instead of double-starting. An IN-FLIGHT boot
        of a different selector is never torn down — a racing caller (e.g. a
        turn still bound to the outgoing model during a switch) is refused
        with a clear retry message. Killing the incoming boot here is exactly
        how a mid-switch turn used to leave ZERO servers running: the turn's
        switch killed the booting server, and that boot's failure cleanup then
        killed the turn's replacement. Only a READY server is replaced.

        The creator claims the boot by installing a placeholder state under
        the lock FIRST; the outgoing server's teardown and the replacement's
        spawn then run OUTSIDE the lock. Concurrent callers therefore coalesce
        (same selector) or are refused (different selector) immediately, and
        pointer reads (status / base_url / selectors) never block on a process
        teardown — the lock only ever guards pointer state, not process waits.
        """
        ok, message = self.available()
        if not ok:
            return False, message

        name, quant = parse_selector(selector)
        canonical = f"{name}:{quant.lower()}"

        outgoing: Optional[LumenServerState] = None
        with self._lock:
            state = self.server
            if state is not None and state.selector == canonical:
                creator = False
            elif state is not None and not state.ready.is_set():
                # A different model is mid-boot: refuse instead of replacing.
                return False, (
                    f"{SWITCH_IN_FLIGHT_PREFIX}{state.selector} — retry in a moment."
                )
            else:
                outgoing = state  # READY server being replaced (or None)
                state = LumenServerState(
                    selector=canonical, port=self.port_config or find_free_port()
                )
                self._server = state
                creator = True

        if not creator:
            # Ready already, or another thread is booting it — wait either way.
            if not state.ready.wait(timeout=self.startup_timeout_seconds + 10):
                return False, "Timed out waiting for the in-flight lumen-server startup."
            if state.error:
                return False, state.error
            if state.process is None or state.process.poll() is not None:
                return False, "lumen-server exited unexpectedly."
            return True, f"Lumen already serving {canonical}."

        # Everything from here until the process is installed runs while this
        # thread HOLDS THE CLAIM (the not-ready placeholder). Any failure —
        # reaping the old server, resolving the binary, opening the log,
        # spawning — must discard that claim, or the runtime would refuse
        # every future boot against a placeholder nothing will ever finish.
        try:
            # Fully reap the outgoing server BEFORE spawning the replacement —
            # two GPU-resident models must never co-exist. The claim keeps
            # this exclusive without holding the lock through the wait.
            if outgoing is not None:
                self._shutdown_state(outgoing)

            server_bin = self.resolve_binary(self.server_binary)
            assert server_bin is not None  # guaranteed by available()
            command = [
                server_bin,
                "--model",
                name,
                "--quant",
                quant.lower(),
                "--port",
                str(state.port),
                "--log-level",
                self.log_level,
            ]
            if self.context_len:
                command += ["--context-len", str(self.context_len)]

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(self.log_path, "ab")
            try:
                process = subprocess.Popen(
                    command, stdout=log_handle, stderr=subprocess.STDOUT
                )
            finally:
                # Popen inherited the descriptor; the parent copy can close.
                log_handle.close()
        except OSError as exc:
            state.error = f"Failed to launch lumen-server: {exc}"
            state.ready.set()  # release any coalesced waiters with the error
            self._discard(state)
            return False, state.error
        except BaseException:
            # Unexpected failure while holding the claim: release waiters and
            # discard the placeholder first, then let the error surface.
            state.error = "lumen-server startup failed unexpectedly."
            state.ready.set()
            self._discard(state)
            raise

        with self._lock:
            state.process = process
            installed = self._server is state
        if not installed:
            # stop() detached this boot while the process was spawning (user
            # shutdown / signal): the fresh process must not outlive it.
            self._shutdown_state(state)
            return False, state.error or "lumen-server was stopped during startup."

        ready, why = self._wait_ready(state)
        if not ready:
            tail = self._log_tail()
            detail = f" Server log: {tail}" if tail else ""
            state.error = f"lumen-server failed to become ready: {why}.{detail}"
            state.ready.set()  # release any waiters with the error recorded
            # Targeted cleanup: discard only OUR failed state. A global stop()
            # here would kill whatever server a concurrent caller may have
            # started since (the other half of the mutual-termination race).
            self._discard(state)
            return False, state.error
        state.ready.set()
        return True, f"Lumen serving {canonical} on port {state.port}."

    def _log_tail(self, max_chars: int = 300) -> str:
        """Last log line(s) — surfaced when startup fails so the user sees why."""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        if not text:
            return ""
        lines = [line for line in text.splitlines() if line.strip()]
        tail_lines = lines[-2:]
        if len(tail_lines) == 2 and tail_lines[0] == tail_lines[1]:
            tail_lines = tail_lines[-1:]  # repeated attempts log the same line
        tail = " | ".join(tail_lines)
        return tail[-max_chars:]

    def _wait_ready(self, state: LumenServerState) -> Tuple[bool, str]:
        deadline = time.time() + self.startup_timeout_seconds
        url = f"http://127.0.0.1:{state.port}/v1/models"
        while time.time() < deadline:
            process = state.process
            if process is None:  # defensive: creator installs before waiting
                return False, "lumen-server was stopped during startup"
            if process.poll() is not None:
                return False, f"process exited with code {process.returncode}"
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        json.loads(response.read().decode("utf-8"))
                        return True, ""
            except Exception:
                pass
            time.sleep(0.5)
        return False, f"timed out after {self.startup_timeout_seconds}s"

    def stop(self) -> None:
        """Stop whatever server is CURRENT (user-facing shutdown/switch path)."""
        with self._lock:
            state = self._server
            self._server = None
        if state is None:
            return
        self._shutdown_state(state)

    def _discard(self, state: LumenServerState) -> None:
        """Terminate `state`'s process, clearing it from the runtime ONLY if it
        is still the current server — a concurrent caller may have replaced it,
        and that replacement must survive this caller's cleanup."""
        with self._lock:
            if self._server is state:
                self._server = None
        self._shutdown_state(state)

    @staticmethod
    def _shutdown_state(state: LumenServerState) -> None:
        if not state.ready.is_set():
            # Release any boot waiters before killing the process under them.
            state.error = state.error or "lumen-server was stopped during startup."
            state.ready.set()
        process = state.process
        if process is None or process.poll() is not None:
            return  # placeholder (no process yet) or already exited
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                pass

    def status(self) -> Dict[str, object]:
        state = self.server
        if state is None:
            return {"running": False}
        return {
            "running": True,
            "ready": state.ready.is_set(),
            "selector": state.selector,
            "port": state.port,
            "uptime_seconds": round(time.time() - state.started_at, 1),
        }

    # ---- downloads -----------------------------------------------------

    def pull(
        self,
        selector: str,
        *,
        on_line: Optional[Callable[[str], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str]:
        """Run `lumen pull` for `selector`, forwarding output lines."""
        cli = self.resolve_binary(self.binary)
        if cli is None:
            return False, f"Lumen binary not found: {self.binary}. {INSTALL_HINT}"
        name, quant = parse_selector(selector)
        process = subprocess.Popen(
            [cli, "pull", name, "--quant", quant, "--yes"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_line = ""
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                last_line = line
                if on_line is not None:
                    on_line(line)
            if cancel_requested is not None and cancel_requested():
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False, "Download cancelled."
        code = process.wait()
        if code != 0:
            return False, last_line or f"lumen pull exited with code {code}"
        return True, last_line or f"Pulled {name}:{quant.lower()}."
