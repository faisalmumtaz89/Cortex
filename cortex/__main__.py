"""Main entry point for Cortex."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from io import TextIOBase
from typing import Any

from cortex.ui_runtime.launcher import launch_tui


def _build_components() -> tuple[Any, Any]:
    from cortex.config import Config
    from cortex.conversation_manager import ConversationManager
    from cortex.logging_config import configure_logging

    config = Config()
    configure_logging(config)
    for warning in config.warnings:
        logging.getLogger("cortex.config").warning(warning)
    conversation_manager = ConversationManager(config)
    return config, conversation_manager


def _run_worker_stdio() -> None:
    from cortex.app.worker_runtime import WorkerRuntime
    from cortex.runtime_io import bound_redirected_stdio_files

    os.environ["CORTEX_WORKER_MODE"] = "1"
    bound_redirected_stdio_files()

    # Reserve true stdio for JSON-RPC transport.
    rpc_stdin = sys.stdin
    rpc_stdout = sys.stdout
    if not isinstance(rpc_stdin, TextIOBase) or not isinstance(rpc_stdout, TextIOBase):
        raise RuntimeError("Worker stdio transport requires text-based stdio streams")

    # Route any accidental stdout writes away from JSON-RPC channel.
    sys.stdout = sys.stderr

    config, conversation_manager = _build_components()
    runtime = WorkerRuntime(
        config=config,
        conversation_manager=conversation_manager,
        rpc_stdin=rpc_stdin,
        rpc_stdout=rpc_stdout,
    )
    runtime.run()


def _run_headless(args) -> None:
    from cortex.app.headless import run_headless

    components = _build_components()
    exit_code = run_headless(
        prompt=args.print,
        components=components,
        model=args.model,
        full_auto=args.full_auto,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("cortex-llm")
    except Exception:
        return "dev"


def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            prog="cortex",
            description=(
                "Terminal AI coding agent. Run `cortex` in a repository for the\n"
                "interactive app, or `cortex -p PROMPT --model PROVIDER:MODEL` for one turn."
            ),
            # Written out line by line so a model id is never split across lines.
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                "Provider keys come from /login in the interactive app, or from\n"
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_OPENAI_API_KEY (with\n"
                "AZURE_OPENAI_ENDPOINT) or OPENAI_COMPATIBLE_API_KEY (with\n"
                "OPENAI_COMPATIBLE_BASE_URL).\n"
                "\n"
                "Models are written provider:model, for example:\n"
                "  openai:gpt-5.5\n"
                "  openai-compatible:deepseek/deepseek-v4.1-flash\n"
                "\n"
                "Each -p run is a new conversation."
            ),
        )
        parser.add_argument(
            "--version",
            action="version",
            version=f"cortex {_package_version()}",
        )
        parser.add_argument(
            "--worker-stdio", action="store_true", help=argparse.SUPPRESS  # the TUI's backend
        )
        parser.add_argument(
            "-p",
            "--print",
            metavar="PROMPT",
            help="Run one agent turn headlessly, print the reply, and exit.",
        )
        parser.add_argument(
            "--model",
            help="Model for headless mode, as provider:model (same syntax as /model). "
            "Remembered for later runs.",
        )
        parser.add_argument(
            "--full-auto",
            action="store_true",
            help="Headless mode: auto-approve edits and shell commands.",
        )
        args = parser.parse_args()

        if args.worker_stdio:
            _run_worker_stdio()
            return

        if args.print is not None:
            if not args.print.strip():
                parser.error("-p needs a prompt, for example: cortex -p \"fix the failing test\"")
            _run_headless(args)
            return
        if args.model or args.full_auto:
            # They apply to one headless turn; the interactive app has /model.
            parser.error("--model and --full-auto go with -p PROMPT (headless mode)")

        exit_code = launch_tui()
        if exit_code == 0:
            return

        if exit_code == 127:
            print(
                "OpenTUI sidecar not available in this environment. "
                "If running from source, execute `./install.sh` at the repository root to build and install the sidecar.",
                file=sys.stderr,
            )
            raise SystemExit(exit_code)

        print(
            "Failed to start OpenTUI frontend runtime.",
            file=sys.stderr,
        )
        raise SystemExit(exit_code)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
