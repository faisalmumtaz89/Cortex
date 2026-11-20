"""What the Harbor adapter runs in a task's container, as plain text: kept
apart from the adapter so that it is tested without Harbor installed."""

from __future__ import annotations

import re
import shlex

REPOSITORY = "faisalmumtaz89/Cortex"
OUTPUT = "/logs/agent/cortex.txt"

# Cortex's documented maximums (docs/configuration.md): a benchmark task is
# one long turn, and the defaults are sized for a person at a terminal.
LIMITS = {
    "CORTEX_TOOLS_MAX_ITERATIONS": "1000",
    "CORTEX_CLOUD_TIMEOUT_SECONDS": "600",
    "CORTEX_CLOUD_MAX_RETRIES": "10",
}


def install_command(version: object) -> str:
    """Install that release with its own installer, which checks the wheel
    against the release's checksum. The installer is the one at the release's
    tag, so a later installer cannot change what a run installs. It leaves
    the task user's shell files alone: the adapter calls Cortex by its path."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", str(version or "")):
        raise ValueError(
            f"pass the Cortex release to install, as in --ak version=2.0.15 (got {version!r})"
        )
    installer = f"https://raw.githubusercontent.com/{REPOSITORY}/v{version}/install.sh"
    return f"curl -fsSL {installer} | CORTEX_NO_MODIFY_PATH=1 bash -s -- {version}"


def model_selector(model: str | None) -> str:
    """The model as Cortex names it, or an error that says how to pass it."""
    if ":" not in (model or ""):
        raise ValueError(
            "pass the model as Cortex names it, provider:model, as in "
            f"-m openai-compatible:deepseek/deepseek-v4.1-flash (got {model!r})"
        )
    return str(model)


def run_command(instruction: str, model: str | None) -> str:
    """One headless turn that may edit files and run commands, its output
    kept in the trial's logs. Harbor runs commands with pipefail, so the exit
    code is Cortex's: a turn that failed is recorded as the trial's error,
    and the task's tests still decide its score."""
    if instruction.lstrip().startswith("/"):
        # Cortex reads a prompt that starts with / as a command or a skill.
        raise ValueError("a task whose text starts with / cannot be passed to cortex -p as a task")
    return (
        # -p=: a task that starts with a dash is the prompt, not an option.
        f'"$HOME/.local/bin/cortex" -p={shlex.quote(instruction)} '
        f"--model {shlex.quote(model_selector(model))} --full-auto 2>&1 </dev/null | tee {OUTPUT}"
    )
