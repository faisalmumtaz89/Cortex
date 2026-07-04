"""Main entry point for Cortex."""

from __future__ import annotations

import argparse
import os
import sys
from io import TextIOBase
from typing import Any

# Disable multiprocessing resource tracking before any imports that might use it.
# This prevents semaphore leak warnings from HuggingFace tokenizer internals.
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"

# Monkey-patch resource tracker before it's used by subprocess-heavy libs.
try:
    from multiprocessing import resource_tracker

    def _dummy_register(*args, **kwargs):
        return None

    def _dummy_unregister(*args, **kwargs):
        return None

    resource_tracker.register = _dummy_register
    resource_tracker.unregister = _dummy_unregister
except ImportError:
    pass

from cortex.ui_runtime.launcher import launch_tui


def _build_components() -> tuple[Any, Any, Any]:
    from cortex.config import Config
    from cortex.conversation_manager import ConversationManager
    from cortex.gpu_validator import GPUValidator
    from cortex.logging_config import configure_logging

    config = Config()
    configure_logging(config)
    gpu_validator = GPUValidator()
    conversation_manager = ConversationManager(config)
    return config, gpu_validator, conversation_manager


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

    config, gpu_validator, conversation_manager = _build_components()
    runtime = WorkerRuntime(
        config=config,
        gpu_validator=gpu_validator,
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
        parser = argparse.ArgumentParser(prog="cortex")
        parser.add_argument(
            "--version",
            action="version",
            version=f"cortex {_package_version()}",
        )
        parser.add_argument(
            "--worker-stdio", action="store_true", help="Run backend worker over JSON-RPC stdio."
        )
        parser.add_argument(
            "-p",
            "--print",
            metavar="PROMPT",
            help="Run one agent turn headlessly, print the reply, and exit.",
        )
        parser.add_argument(
            "--model",
            help="Model selector for headless mode (same syntax as /model).",
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

        if args.print:
            _run_headless(args)
            return

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
