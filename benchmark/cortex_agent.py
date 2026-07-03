"""Harbor (Terminal-Bench 2.0) adapter for the Cortex coding agent.

Usage:
    PYTHONPATH=benchmark harbor run -d terminal-bench@2.0 \
        -a "cortex_agent:CortexAgent" -m azure/gpt-5.5 -y ...

The adapter uploads a locally built Cortex wheel (see benchmark/build_wheel.sh)
into the task container, installs it cloud-only (no MLX), and runs one headless
agent turn: `cortex -p "<instruction>" --model <provider:model> --full-auto`.

Model credentials come from the host environment (AZURE_OPENAI_API_KEY /
AZURE_OPENAI_ENDPOINT, or OPENAI_API_KEY / ANTHROPIC_API_KEY) — the harness
itself never calls a model.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_HOST_REPO = Path(__file__).resolve().parents[1]
_WHEEL_DIR = _HOST_REPO / "benchmark" / "dist"
_VENDOR_DIR = _HOST_REPO / "benchmark" / "vendor"
_REMOTE_VENV = "/installed-agent/venv"
_UV_RELEASE = "https://github.com/astral-sh/uv/releases/latest/download"

# Cloud-only runtime dependencies (the wheel is installed with --no-deps so the
# macOS-only MLX/GGUF stack never reaches the Linux container).
_CLOUD_DEPS = "pydantic pyyaml psutil openai anthropic"

_BOOTSTRAP_PYTHON = """
set -e
python_ready() {
    command -v python3 >/dev/null 2>&1 \
        && python3 -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null \
        && python3 -m ensurepip --version >/dev/null 2>&1
}
if ! python_ready; then
    if [ -f /etc/alpine-release ]; then
        apk add --no-cache python3 py3-pip || true
    elif command -v apt-get >/dev/null 2>&1; then
        (apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip) || true
    elif command -v yum >/dev/null 2>&1; then
        yum install -y -q python3 python3-pip || true
    fi
fi
if python_ready; then echo CORTEX_PY_OK; else echo CORTEX_PY_MISSING; fi
uname -m
"""

_INSTALL_WITH_SYSTEM_PYTHON = f"""
set -e
python3 -m venv {_REMOTE_VENV}
{_REMOTE_VENV}/bin/pip install --quiet --no-cache-dir --upgrade pip
{_REMOTE_VENV}/bin/pip install --quiet --no-cache-dir {_CLOUD_DEPS}
{_REMOTE_VENV}/bin/pip install --quiet --no-cache-dir --no-deps {{remote_wheel}}
ln -sf {_REMOTE_VENV}/bin/cortex /usr/local/bin/cortex
chmod -R a+rX /installed-agent
cortex --version
"""

# Fallback for images without python >= 3.11: a static uv binary provisions a
# managed CPython and the venv.
_INSTALL_WITH_UV = f"""
set -e
chmod +x /installed-agent/uv
export UV_PYTHON_INSTALL_DIR=/installed-agent/python UV_CACHE_DIR=/installed-agent/uv-cache
/installed-agent/uv venv --quiet --python 3.12 {_REMOTE_VENV}
/installed-agent/uv pip install --quiet --python {_REMOTE_VENV}/bin/python {_CLOUD_DEPS}
/installed-agent/uv pip install --quiet --python {_REMOTE_VENV}/bin/python --no-deps {{remote_wheel}}
ln -sf {_REMOTE_VENV}/bin/cortex /usr/local/bin/cortex
chmod -R a+rX /installed-agent
cortex --version
"""


class CortexAgent(BaseInstalledAgent):
    """Installed-agent adapter that runs Cortex headless inside the container."""

    _OUTPUT_FILENAME = "cortex.txt"

    @staticmethod
    def name() -> str:
        return "cortex"

    def get_version_command(self) -> str | None:
        return "cortex --version"

    def populate_context_post_run(self, context: AgentContext) -> None:
        """No trajectory/cost parsing yet; transcripts live in /logs/agent."""

    def _wheel_path(self) -> Path:
        wheels = sorted(_WHEEL_DIR.glob("cortex_llm-*.whl"))
        if not wheels:
            raise FileNotFoundError(
                f"No cortex wheel in {_WHEEL_DIR}. Run benchmark/build_wheel.sh first."
            )
        return wheels[-1]

    def _model_selector(self) -> str:
        # Harbor convention is provider/model (e.g. azure/gpt-5.5); Cortex uses
        # provider:model.
        model = self.model_name or "azure/gpt-5.5"
        if "/" in model:
            provider, model_id = model.split("/", 1)
            return f"{provider}:{model_id}"
        return model

    def _host_uv_binary(self, machine: str) -> Path:
        """Fetch (once) and cache a static uv binary for the container arch."""
        arch = "aarch64" if machine in {"aarch64", "arm64"} else "x86_64"
        binary = _VENDOR_DIR / f"uv-{arch}"
        if binary.exists():
            return binary
        _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        archive = f"uv-{arch}-unknown-linux-musl"
        subprocess.run(
            f"curl -fsSL {_UV_RELEASE}/{archive}.tar.gz | tar -xz -C {_VENDOR_DIR} "
            f"--strip-components=1 {archive}/uv && mv {_VENDOR_DIR}/uv {binary}",
            shell=True,
            check=True,
        )
        return binary

    async def install(self, environment: BaseEnvironment) -> None:
        wheel = self._wheel_path()
        remote_wheel = f"/installed-agent/{wheel.name}"
        probe = await self.exec_as_root(environment, command=_BOOTSTRAP_PYTHON)
        probe_out = str(getattr(probe, "stdout", "") or "")
        await environment.upload_file(str(wheel), remote_wheel)

        if "CORTEX_PY_OK" in probe_out:
            script = _INSTALL_WITH_SYSTEM_PYTHON.format(remote_wheel=remote_wheel)
        else:
            machine = probe_out.strip().splitlines()[-1].strip() if probe_out else "x86_64"
            self.logger.info("No python>=3.11 in image; using uv fallback (%s)", machine)
            await environment.upload_file(
                str(self._host_uv_binary(machine)), "/installed-agent/uv"
            )
            script = _INSTALL_WITH_UV.format(remote_wheel=remote_wheel)

        await self.exec_as_root(environment, command=script)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        env = {
            "AZURE_OPENAI_API_KEY": self._get_env("AZURE_OPENAI_API_KEY") or "",
            "AZURE_OPENAI_ENDPOINT": self._get_env("AZURE_OPENAI_ENDPOINT") or "",
            "OPENAI_API_KEY": self._get_env("OPENAI_API_KEY") or "",
            "ANTHROPIC_API_KEY": self._get_env("ANTHROPIC_API_KEY") or "",
            # Benchmark runtime tuning via Cortex env-config overrides.
            "CORTEX_TOOLS_MAX_ITERATIONS": "100",
            "CORTEX_TOOLS_IDLE_TIMEOUT_SECONDS": "600",
            "CORTEX_CLOUD_TIMEOUT_SECONDS": "600",
            "CORTEX_CLOUD_MAX_RETRIES": "3",
            "CORTEX_MAX_TOKENS": "32000",
        }
        env = {key: value for key, value in env.items() if value}

        command = (
            f"cortex -p {shlex.quote(instruction)} "
            f"--model {shlex.quote(self._model_selector())} --full-auto "
            f"2>&1 </dev/null | tee /logs/agent/{self._OUTPUT_FILENAME}"
        )
        await self.exec_as_agent(environment, command=command, env=env)
