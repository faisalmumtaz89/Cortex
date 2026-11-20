"""The commands the Harbor adapter runs in a task's container."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from cortex_commands import LIMITS, OUTPUT, install_command, run_command  # noqa: E402

from cortex.config import Config  # noqa: E402


def test_the_install_is_the_named_release_by_its_own_installer() -> None:
    assert install_command("2.0.15") == (
        "curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Cortex/v2.0.15/install.sh "
        "| CORTEX_NO_MODIFY_PATH=1 bash -s -- 2.0.15"
    )


@pytest.mark.parametrize("version", [
    "", None, 2, "latest", "2.0", "2.0.15; rm -rf /", "v2.0.15", "٢.٠.١٥"])
def test_a_run_must_name_the_release_it_installs(version: object) -> None:
    with pytest.raises(ValueError, match="--ak version=2.0.15"):
        install_command(version)


def _run(tmp_path: Path, task: str, exit_code: int) -> subprocess.CompletedProcess:
    """The command as Harbor runs it (bash, pipefail), with a cortex that
    writes down its arguments and exits as told."""
    launcher = tmp_path / ".local" / "bin" / "cortex"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(
        f"#!/bin/bash\nprintf '%s\\0' \"$@\" > \"$HOME/arguments\"\necho said\nexit {exit_code}\n")
    launcher.chmod(0o755)
    command = run_command(task, "openai-compatible:deepseek/deepseek-v4.1-flash")
    command = command.replace(OUTPUT, str(tmp_path / "cortex.txt"))
    return subprocess.run(
        ["bash", "-c", f"set -o pipefail; {command}"],
        env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("task", [
    "Fix it's `bug`; then $(reboot)\nand \"report\".",
    "- You are given a model.",
    "--help",
    "-",
])
def test_the_task_reaches_cortex_whole_and_is_logged(tmp_path: Path, task: str) -> None:
    done = _run(tmp_path, task, 0)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "arguments").read_text().split("\0")[:-1] == [
        f"-p={task}", "--model", "openai-compatible:deepseek/deepseek-v4.1-flash", "--full-auto"]
    assert (tmp_path / "cortex.txt").read_text() == "said\n"


def test_a_dash_first_prompt_is_a_prompt_to_cortex(tmp_path: Path) -> None:
    """What -p= is for: the real command line takes --help as the prompt (and
    stops for want of a model, as it would for any prompt here)."""
    done = subprocess.run(
        [sys.executable, "-P", "-m", "cortex", "-p=--help"],
        env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]}, cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 2 and "no model selected" in done.stderr, done.stderr
    assert "usage:" not in done.stdout + done.stderr


def test_the_exit_code_is_cortexs(tmp_path: Path) -> None:
    assert _run(tmp_path, "task", 1).returncode == 1


def test_a_task_that_cortex_would_read_as_a_command_is_refused() -> None:
    with pytest.raises(ValueError, match="starts with /"):
        run_command("  /help me fix the build", "openai:gpt-5.5")


@pytest.mark.parametrize("model", ["", None, "deepseek/deepseek-v4.1-flash"])
def test_a_model_not_named_as_cortex_names_it_is_refused(model: str | None) -> None:
    with pytest.raises(ValueError, match="provider:model"):
        run_command("task", model)


def test_the_limits_are_settings_cortex_takes_at_their_maximums(tmp_path, monkeypatch) -> None:
    for name, value in LIMITS.items():
        monkeypatch.setenv(name, value)
    config = Config(config_path=tmp_path / "config.yaml")
    assert not config.warnings
    assert (config.tools.tools_max_iterations, config.cloud.cloud_timeout_seconds,
            config.cloud.cloud_max_retries) == (1000, 600, 10)
    for name, value in LIMITS.items():  # one more is refused: these are the maximums
        monkeypatch.setenv(name, str(int(value) + 1))
    assert len(Config(config_path=tmp_path / "config.yaml").warnings) == len(LIMITS)
