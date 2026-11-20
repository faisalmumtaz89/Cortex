"""Harbor adapter for Cortex: installs a released Cortex in the task's
container and runs the task as one headless turn. See README.md."""

from __future__ import annotations

from cortex_commands import LIMITS, install_command, model_selector, run_command
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class CortexAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "cortex"

    def get_version_command(self) -> str | None:
        return '"$HOME/.local/bin/cortex" --version'

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().removeprefix("cortex ")

    async def install(self, environment: BaseEnvironment) -> None:
        model_selector(self.model_name)  # refused before anything is installed
        await self.ensure_system_dependencies(environment, ("curl",))
        await self.exec_as_agent(environment, command=install_command(self.version()))

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await self.exec_as_agent(
            environment, command=run_command(instruction, self.model_name), env=dict(LIMITS)
        )
