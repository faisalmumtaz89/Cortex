"""Main entry point for Cortex."""

from __future__ import annotations

import argparse
import os
import sys
from io import TextIOBase
from typing import Any

# Disable multiprocessing resource tracking before any imports that might use it
# This prevents semaphore leak warnings from transformers internals.
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


def _build_components(
    *,
    strict_gpu_validation: bool = True,
    lazy_inference_engine: bool = False,
    eager_gpu_validation: bool = True,
) -> tuple[Any, Any, Any, Any, Any]:
    from cortex.app.lazy_inference_engine import LazyProxy
    from cortex.config import Config
    from cortex.conversation_manager import ConversationManager
    from cortex.gpu_validator import GPUValidator
    from cortex.logging_config import configure_logging
    from cortex.model_manager import ModelManager

    config = Config()
    configure_logging(config)
    gpu_validator = GPUValidator()

    if eager_gpu_validation:
        is_valid, _gpu_info, errors = gpu_validator.validate()
        if not is_valid:
            if strict_gpu_validation:
                print(
                    "Error: GPU validation failed. Cortex requires Apple Silicon with Metal support.",
                    file=sys.stderr,
                )
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                raise SystemExit(1)

            print(
                "Warning: GPU validation failed. Continuing in worker mode with limited local inference capability.",
                file=sys.stderr,
            )
            for error in errors:
                print(f"  - {error}", file=sys.stderr)

    model_manager = ModelManager(config, gpu_validator)

    def _create_inference_engine():
        # Import lazily to avoid paying MLX/Torch startup costs in cloud-only worker turns.
        from cortex.inference_engine import InferenceEngine

        return InferenceEngine(config, model_manager)

    if lazy_inference_engine:
        inference_engine = LazyProxy(_create_inference_engine)
    else:
        inference_engine = _create_inference_engine()
    conversation_manager = ConversationManager(config)
    return config, gpu_validator, model_manager, inference_engine, conversation_manager


def _cleanup_inference_engine(inference_engine) -> None:
    if inference_engine is not None and hasattr(inference_engine, "get_if_initialized"):
        inference_engine = inference_engine.get_if_initialized()

    if inference_engine is None:
        return

    if (
        inference_engine is not None
        and hasattr(inference_engine, "memory_pool")
        and inference_engine.memory_pool
    ):
        inference_engine.memory_pool.cleanup()

    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except Exception:
        pass


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

    inference_engine = None
    try:
        (
            config,
            _gpu_validator,
            model_manager,
            inference_engine,
            conversation_manager,
        ) = _build_components(
            strict_gpu_validation=False,
            lazy_inference_engine=True,
            eager_gpu_validation=False,
        )

        runtime = WorkerRuntime(
            config=config,
            model_manager=model_manager,
            gpu_validator=_gpu_validator,
            inference_engine=inference_engine,
            conversation_manager=conversation_manager,
            rpc_stdin=rpc_stdin,
            rpc_stdout=rpc_stdout,
        )
        runtime.run()
    finally:
        _cleanup_inference_engine(inference_engine)


def main() -> None:
    try:
        parser = argparse.ArgumentParser(prog="cortex")
        parser.add_argument(
            "--worker-stdio", action="store_true", help="Run backend worker over JSON-RPC stdio."
        )
        args = parser.parse_args()

        if args.worker_stdio:
            _run_worker_stdio()
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
