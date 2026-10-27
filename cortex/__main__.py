"""Main entry point for Cortex."""

from __future__ import annotations

import argparse
import logging
import os
import select
import stat
import sys
import tempfile
from io import TextIOBase
from pathlib import Path
from typing import Any

from cortex.ui_runtime.launcher import USER_TMPDIR_VARIABLE, launch_tui


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


def _restore_user_tmpdir() -> None:
    """The launcher gives the TUI its own temp folder (see launcher.py); the
    worker, and every command it runs, use the user's TMPDIR."""
    if USER_TMPDIR_VARIABLE not in os.environ:
        return
    user_tmpdir = os.environ.pop(USER_TMPDIR_VARIABLE)
    if user_tmpdir:
        os.environ["TMPDIR"] = user_tmpdir
    else:
        os.environ.pop("TMPDIR", None)
    tempfile.tempdir = None  # computed again from the restored environment


def _run_worker_stdio() -> None:
    from cortex.app.worker_runtime import WorkerRuntime
    from cortex.runtime_io import bound_redirected_stdio_files

    os.environ["CORTEX_WORKER_MODE"] = "1"
    _restore_user_tmpdir()
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


# How long a run waits for piped input to start, and how much it takes.
_STDIN_WAIT_SECONDS = 3.0
_STDIN_LIMIT = 512 * 1024


def _piped_input() -> str:
    """What was piped to this run ("" if nothing). A pipe left open with
    nothing in it is not waited for past _STDIN_WAIT_SECONDS."""
    stream = sys.stdin
    if stream is None or stream.isatty():
        return ""
    try:
        ready = select.select([stream], [], [], _STDIN_WAIT_SECONDS)[0]
    except (OSError, ValueError):
        return ""
    if not ready:
        sys.stderr.write(
            f"note: stdin sent nothing for {_STDIN_WAIT_SECONDS:g}s; not reading it "
            "(run with < /dev/null to skip the wait)\n"
        )
        return ""
    data = stream.buffer.read(_STDIN_LIMIT + 1)
    if len(data) > _STDIN_LIMIT:
        print(f"note: only the first {_STDIN_LIMIT // 1024} KB of stdin are sent", file=sys.stderr)
        data = data[:_STDIN_LIMIT]
    return data.decode("utf-8", errors="replace")


def _run_headless(args) -> None:
    from cortex.app.headless import run_headless

    piped = _piped_input()
    if args.print == "-":
        # `cortex -p -`: the prompt itself comes from stdin.
        prompt = piped
        if not prompt.strip():
            sys.stderr.write('error: -p - reads the prompt from stdin, and stdin was empty\n')
            raise SystemExit(2)
    elif piped.strip():
        prompt = f"{args.print}\n\nInput piped to this run:\n{piped}"
    else:
        prompt = args.print

    components = _build_components()
    exit_code = run_headless(
        prompt=prompt,
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


def _has_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _private_home_folder() -> None:
    """~/.cortex holds conversations, logs, settings and maybe keys: only its
    owner may open it. A folder that is a link, or not the user's, is left
    as it is."""
    folder = Path.home() / ".cortex"
    try:
        folder.mkdir(mode=0o700, exist_ok=True)
        info = folder.lstat()
        if stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and info.st_mode & 0o077:
            folder.chmod(0o700)
    except OSError:
        pass  # Cortex still runs; the files say why they cannot be written.


def main() -> None:
    _private_home_folder()
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
                "Each -p run is a new conversation. Input piped to it is sent after\n"
                "the prompt: git diff | cortex -p \"summarize this diff\"."
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
        if not _has_terminal():
            sys.stderr.write(
                "cortex: the interactive app needs a terminal. Without one, run a single "
                'prompt: cortex -p "..." --model provider:model\n'
            )
            sys.exit(2)

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
