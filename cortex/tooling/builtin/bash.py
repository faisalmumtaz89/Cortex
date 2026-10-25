"""Shell command execution tool."""

from __future__ import annotations

import codecs
import os
import selectors
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import IO

from pydantic import BaseModel, Field

from cortex.cloud.credentials import ENV_KEY_MAP
from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.types import ToolExecutionState, ToolResult, TurnInterrupt

# Output kept from a command: its first and last bytes.
_HEAD_BYTES = 20_000
_TAIL_BYTES = 230_000
# How long output is still read after the command exits, and how long a
# timed-out command gets between SIGTERM and SIGKILL.
_DRAIN_SECONDS = 1.0
_KILL_GRACE_SECONDS = 3.0


class BashArgs(BaseModel):
    command: str = Field(min_length=1)
    timeout_ms: int = Field(default=120_000, ge=1, le=600_000)
    workdir: str | None = Field(default=None)


class BashTool(BaseTool[BashArgs]):
    name = "bash"
    description = "Execute a shell command with timeout and output capture."
    permission = "bash"
    args_model = BashArgs

    def run(self, parsed_args: BashArgs, context: ToolContext) -> ToolResult:
        cwd = context.repo_root
        if parsed_args.workdir:
            requested = Path(parsed_args.workdir).expanduser()
            if not requested.is_absolute():
                requested = (context.repo_root / requested).resolve()
            else:
                requested = requested.resolve()
            cwd = requested

        if not cwd.exists() or not cwd.is_dir():
            raise ToolExecutionError(f"invalid workdir: {cwd}")

        if not cwd.is_relative_to(context.repo_root):
            raise ToolExecutionError(
                f"workdir escapes repository root: {cwd}"
            )

        if context.interrupt.is_set():  # the turn ended before the command started
            return ToolResult(
                id=context.call_id,
                name=self.name,
                state=ToolExecutionState.ERROR,
                ok=False,
                output="[command not run: the turn was interrupted]",
                error="interrupted",
                metadata={"exit_code": None, "timed_out": False, "workdir": str(cwd),
                          "truncated": False, "interrupted": True},
            )
        output, exit_code, ended_by = _run(
            parsed_args.command,
            cwd=cwd,
            timeout=parsed_args.timeout_ms / 1000,
            interrupt=context.interrupt,
        )
        truncated = output.omitted > 0
        timed_out = ended_by == "timeout"
        ending = {"timeout": "timed out", "interrupt": "interrupted"}.get(ended_by, "")
        combined = output.text().rstrip()
        if ending:
            combined = f"{combined}\n\n[command {ending}]"
        ok = exit_code == 0 and not ending

        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED if ok else ToolExecutionState.ERROR,
            ok=ok,
            output=combined.strip(),
            error=None if ok else (ending or f"exit_code={exit_code}"),
            metadata={
                "exit_code": exit_code,
                "timed_out": timed_out,
                "interrupted": ended_by == "interrupt",
                "workdir": str(cwd),
                "truncated": truncated,
            },
        )


class _Output:
    """A command's output, kept within limits: its first _HEAD_BYTES and its
    last _TAIL_BYTES, which hold what usually matters (errors, test results,
    the last lines)."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail: deque[bytes] = deque()
        self.tail_size = 0
        self.omitted = 0

    def add(self, chunk: bytes) -> None:
        room = _HEAD_BYTES - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if not chunk:
            return
        self.tail.append(chunk)
        self.tail_size += len(chunk)
        while self.tail_size > _TAIL_BYTES:
            excess = self.tail_size - _TAIL_BYTES
            first = self.tail[0]
            if len(first) <= excess:
                self.tail.popleft()
                dropped = len(first)
            else:
                self.tail[0] = first[excess:]
                dropped = excess
            self.tail_size -= dropped
            self.omitted += dropped

    def text(self) -> str:
        tail = b"".join(self.tail)
        if not self.omitted:
            return _decode(bytes(self.head) + tail)
        # Cut at whole characters: drop a character split by either cut.
        head = codecs.getincrementaldecoder("utf-8")("replace").decode(bytes(self.head))
        start = 0
        while start < min(3, len(tail)) and 0x80 <= tail[start] < 0xC0:
            start += 1
        return (
            f"{_newlines(head)}\n\n[... {self.omitted} bytes of output not shown ...]\n\n"
            f"{_decode(tail[start:])}"
        )


def _decode(data: bytes) -> str:
    return _newlines(data.decode("utf-8", errors="replace"))


def _newlines(text: str) -> str:
    """Line breaks as \\n only: a lone carriage return (progress bars) would
    otherwise move the cursor where the output is shown."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _command_env() -> dict[str, str]:
    """The environment commands run in: the user's, without the provider API
    keys Cortex uses and without Cortex's own CORTEX_* variables."""
    hidden = set(ENV_KEY_MAP.values())
    return {
        name: value
        for name, value in os.environ.items()
        if name not in hidden and not name.startswith("CORTEX_")
    }


def _run(
    command: str, *, cwd: Path, timeout: float, interrupt: TurnInterrupt
) -> tuple[_Output, int, str]:
    """Run `command` in its own process group, with no stdin (the worker's
    stdin is its JSON-RPC channel) and stderr merged into stdout. Returns when
    the command exits; a background child it started keeps running, and its
    later output is read and discarded so writing it does not kill the child.
    At the timeout, or when the turn is interrupted, the whole group gets
    SIGTERM, then SIGKILL if it is still running after a grace period. The
    third value says which ended it: "timeout", "interrupt" or ""."""
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_command_env(),
        start_new_session=True,
    )
    assert proc.stdout is not None
    pipe = proc.stdout
    output = _Output()
    deadline = time.monotonic() + timeout
    exited_at: float | None = None
    ended_by = ""
    closed = False
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(pipe, selectors.EVENT_READ)
            while True:
                now = time.monotonic()
                if exited_at is None and proc.poll() is not None:
                    exited_at = now
                if exited_at is not None and (closed or now - exited_at > _DRAIN_SECONDS):
                    break
                if exited_at is None and (now >= deadline or interrupt.is_set()):
                    ended_by = "interrupt" if interrupt.is_set() else "timeout"
                    _stop(proc)
                    exited_at = time.monotonic()
                    continue
                if closed:  # the command closed its output but still runs
                    time.sleep(0.05)
                    continue
                if selector.select(timeout=0.05):
                    chunk = os.read(pipe.fileno(), 65536)
                    if chunk:
                        output.add(chunk)
                    else:
                        closed = True
    except BaseException:
        _stop(proc)
        pipe.close()
        raise
    if closed:
        pipe.close()
    else:
        # A background child still holds the output open.
        threading.Thread(target=_discard, args=(pipe,), daemon=True).start()
    return output, proc.returncode, ended_by


def _discard(pipe: IO[bytes]) -> None:
    with pipe:
        while os.read(pipe.fileno(), 65536):
            pass


def _stop(proc: subprocess.Popen) -> None:
    """SIGTERM the command's process group; SIGKILL the group if anything in
    it, the shell or a child it started, is still running after the grace
    period."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + _KILL_GRACE_SECONDS
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return  # the whole group is gone
        time.sleep(0.05)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
