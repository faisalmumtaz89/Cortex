"""Status and benchmark UI helpers for CLI."""

from __future__ import annotations

from typing import Any


def show_status(*, cli: Any) -> None:
    """Show current setup status."""
    is_valid, gpu_info, errors = cli.gpu_validator.validate()

    width = min(cli.get_terminal_width() - 2, 70)

    print()
    cli.print_box_header("Current Setup", width)
    cli.print_empty_line(width)

    if gpu_info:
        cli.print_box_line(f"  \033[2mGPU:\033[0m \033[93m{gpu_info.chip_name}\033[0m", width)
        cli.print_box_line(f"  \033[2mCores:\033[0m \033[93m{gpu_info.gpu_cores}\033[0m", width)

        mem_gb = gpu_info.total_memory / (1024**3)
        mem_str = f"{mem_gb:.1f} GB"
        cli.print_box_line(f"  \033[2mMemory:\033[0m \033[93m{mem_str}\033[0m", width)

    if cli.model_manager.current_model:
        model_info = cli.model_manager.get_current_model()
        if model_info:
            cli.print_box_line(f"  \033[2mModel:\033[0m \033[93m{model_info.name[:43]}\033[0m", width)

            tokenizer = cli.model_manager.tokenizers.get(model_info.name)
            profile = cli.template_registry.get_template(model_info.name)
            if profile:
                template_name = profile.config.name
                cli.print_box_line(f"  \033[2mTemplate:\033[0m \033[93m{template_name}\033[0m", width)
    else:
        cli.print_box_line("  \033[2mModel:\033[0m \033[31mNone loaded\033[0m", width)

    cli.print_empty_line(width)
    cli.print_box_footer(width)


def show_gpu_status(*, cli: Any) -> None:
    """Show GPU status."""
    is_valid, gpu_info, errors = cli.gpu_validator.validate()
    if gpu_info:
        print("\n\033[96mGPU Information:\033[0m")
        print(f"  Chip: \033[93m{gpu_info.chip_name}\033[0m")
        print(f"  GPU Cores: \033[93m{gpu_info.gpu_cores}\033[0m")
        print(f"  Total Memory: \033[93m{gpu_info.total_memory / (1024**3):.1f} GB\033[0m")
        print(f"  Available Memory: \033[93m{gpu_info.available_memory / (1024**3):.1f} GB\033[0m")
        print(f"  Metal Support: {'\033[32mYes\033[0m' if gpu_info.has_metal else '\033[31mNo\033[0m'}")
        print(f"  MPS Support: {'\033[32mYes\033[0m' if gpu_info.has_mps else '\033[31mNo\033[0m'}")

    memory_status = cli.model_manager.get_memory_status()
    print("\n\033[96mMemory Status:\033[0m")
    print(f"  Available: \033[93m{memory_status['available_gb']:.1f} GB\033[0m")
    print(f"  Models Loaded: \033[93m{memory_status['models_loaded']}\033[0m")
    print(f"  Model Memory: \033[93m{memory_status['model_memory_gb']:.1f} GB\033[0m")


def run_benchmark(*, cli: Any) -> None:
    """Run performance benchmark."""
    if not cli.model_manager.current_model:
        print("\033[31m✗\033[0m No model loaded.")
        return

    print("\033[96m⚡\033[0m Running benchmark (100 tokens)...")
    metrics = cli.inference_engine.benchmark()

    if metrics:
        print("\n\033[96mBenchmark Results:\033[0m")
        print(f"  Tokens Generated: \033[93m{metrics.tokens_generated}\033[0m")
        print(f"  Time: \033[93m{metrics.time_elapsed:.2f}s\033[0m")
        print(f"  Tokens/Second: \033[93m{metrics.tokens_per_second:.1f}\033[0m")
        print(f"  First Token: \033[93m{metrics.first_token_latency:.3f}s\033[0m")
        print(f"  GPU Usage: \033[93m{metrics.gpu_utilization:.1f}%\033[0m")
        print(f"  Memory: \033[93m{metrics.memory_used_gb:.1f}GB\033[0m")
