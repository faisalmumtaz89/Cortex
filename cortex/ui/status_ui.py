"""Status and benchmark UI helpers for CLI."""

from __future__ import annotations

from typing import Any


def _emit(cli: Any, text: str = "") -> None:
    """Render ANSI-formatted text to the active console/stdout."""
    console = getattr(cli, "console", None)
    if console is not None and hasattr(console, "print"):
        try:
            if text:
                console.print(text)
            else:
                console.print()
            return
        except Exception:
            pass
    if not text:
        print()
        return
    print(text)


def show_status(*, cli: Any) -> None:
    """Show current setup status."""
    is_valid, gpu_info, errors = cli.gpu_validator.validate()

    width = min(cli.get_terminal_width() - 2, 70)

    _emit(cli)
    cli.print_box_header("Current Setup", width)
    cli.print_empty_line(width)

    if gpu_info:
        cli.print_box_line(f"  \033[2mGPU:\033[0m \033[93m{gpu_info.chip_name}\033[0m", width)
        cli.print_box_line(f"  \033[2mCores:\033[0m \033[93m{gpu_info.gpu_cores}\033[0m", width)

        mem_gb = gpu_info.total_memory / (1024**3)
        mem_str = f"{mem_gb:.1f} GB"
        cli.print_box_line(f"  \033[2mMemory:\033[0m \033[93m{mem_str}\033[0m", width)

    active_target = cli.active_model_target
    if active_target.backend == "cloud" and active_target.cloud_model:
        cloud_ref = active_target.cloud_model
        is_auth, source = cli.cloud_router.get_auth_status(cloud_ref.provider)
        auth_label = f"yes ({source})" if is_auth and source else "yes" if is_auth else "no"
        cli.print_box_line("  \033[2mBackend:\033[0m \033[93mCloud API\033[0m", width)
        cli.print_box_line(f"  \033[2mModel:\033[0m \033[93m{cloud_ref.selector[:43]}\033[0m", width)
        cli.print_box_line(f"  \033[2mAuthenticated:\033[0m \033[93m{auth_label}\033[0m", width)
    elif cli.model_manager.current_model:
        model_info = cli.model_manager.get_current_model()
        if model_info:
            cli.print_box_line("  \033[2mBackend:\033[0m \033[93mLocal\033[0m", width)
            cli.print_box_line(f"  \033[2mModel:\033[0m \033[93m{model_info.name[:43]}\033[0m", width)

            profile = cli.template_registry.get_template(model_info.name)
            if profile:
                template_name = profile.config.name
                cli.print_box_line(f"  \033[2mTemplate:\033[0m \033[93m{template_name}\033[0m", width)
    else:
        cli.print_box_line("  \033[2mBackend:\033[0m \033[93mLocal\033[0m", width)
        cli.print_box_line("  \033[2mModel:\033[0m \033[31mNone loaded\033[0m", width)

    cli.print_empty_line(width)
    cli.print_box_footer(width)


def show_gpu_status(*, cli: Any) -> None:
    """Show GPU status."""
    is_valid, gpu_info, errors = cli.gpu_validator.validate()
    if gpu_info:
        _emit(cli, "\n\033[96mGPU Information:\033[0m")
        _emit(cli, f"  Chip: \033[93m{gpu_info.chip_name}\033[0m")
        _emit(cli, f"  GPU Cores: \033[93m{gpu_info.gpu_cores}\033[0m")
        _emit(cli, f"  Total Memory: \033[93m{gpu_info.total_memory / (1024**3):.1f} GB\033[0m")
        _emit(cli, f"  Available Memory: \033[93m{gpu_info.available_memory / (1024**3):.1f} GB\033[0m")
        metal_status = "\033[32mYes\033[0m" if gpu_info.has_metal else "\033[31mNo\033[0m"
        mps_status = "\033[32mYes\033[0m" if gpu_info.has_mps else "\033[31mNo\033[0m"
        _emit(cli, f"  Metal Support: {metal_status}")
        _emit(cli, f"  MPS Support: {mps_status}")

    memory_status = cli.model_manager.get_memory_status()
    _emit(cli, "\n\033[96mMemory Status:\033[0m")
    _emit(cli, f"  Available: \033[93m{memory_status['available_gb']:.1f} GB\033[0m")
    _emit(cli, f"  Models Loaded: \033[93m{memory_status['models_loaded']}\033[0m")
    _emit(cli, f"  Model Memory: \033[93m{memory_status['model_memory_gb']:.1f} GB\033[0m")


def run_benchmark(*, cli: Any) -> None:
    """Run performance benchmark."""
    if cli.active_model_target.backend == "cloud":
        _emit(cli, "\033[31m✗\033[0m Benchmark is currently available for local models only.")
        _emit(cli, "\033[2mSwitch to a local model with /model to benchmark.\033[0m")
        return

    if not cli.model_manager.current_model:
        _emit(cli, "\033[31m✗\033[0m No model loaded.")
        return

    _emit(cli, "\033[96m⚡\033[0m Running benchmark (100 tokens)...")
    metrics = cli.inference_engine.benchmark()

    if metrics:
        _emit(cli, "\n\033[96mBenchmark Results:\033[0m")
        _emit(cli, f"  Tokens Generated: \033[93m{metrics.tokens_generated}\033[0m")
        _emit(cli, f"  Time: \033[93m{metrics.time_elapsed:.2f}s\033[0m")
        _emit(cli, f"  Tokens/Second: \033[93m{metrics.tokens_per_second:.1f}\033[0m")
        _emit(cli, f"  First Token: \033[93m{metrics.first_token_latency:.3f}s\033[0m")
        _emit(cli, f"  GPU Usage: \033[93m{metrics.gpu_utilization:.1f}%\033[0m")
        _emit(cli, f"  Memory: \033[93m{metrics.memory_used_gb:.1f}GB\033[0m")
