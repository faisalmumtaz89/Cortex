"""Shell command execution tool."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from cortex.tooling.base import BaseTool, ToolContext, ToolExecutionError
from cortex.tooling.types import ToolExecutionState, ToolResult

_MAX_OUTPUT_BYTES = 250_000


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

        proc = subprocess.Popen(
            parsed_args.command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=parsed_args.timeout_ms / 1000)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.kill()
            stdout, stderr = proc.communicate()

        combined = (stdout or "") + (stderr or "")
        truncated = False
        if len(combined.encode("utf-8", errors="ignore")) > _MAX_OUTPUT_BYTES:
            truncated = True
            combined = combined[:_MAX_OUTPUT_BYTES] + "\n\n[output truncated]"

        if timed_out:
            combined = combined + "\n\n[command timed out]"

        return ToolResult(
            id=context.call_id,
            name=self.name,
            state=ToolExecutionState.COMPLETED if proc.returncode == 0 else ToolExecutionState.ERROR,
            ok=(proc.returncode == 0 and not timed_out),
            output=combined.strip(),
            error=None if proc.returncode == 0 and not timed_out else f"exit_code={proc.returncode}",
            metadata={
                "exit_code": proc.returncode,
                "timed_out": timed_out,
                "workdir": str(cwd),
                "truncated": truncated,
            },
        )
