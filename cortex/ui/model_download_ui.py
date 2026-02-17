"""Model download UI helpers for CLI."""

from __future__ import annotations

from typing import Any

from rich.text import Text


def _emit(cli: Any, text: str = "") -> None:
    """Render ANSI-formatted legacy text through Rich console."""
    if not text:
        cli.console.print()
        return
    cli.console.print(Text.from_ansi(text))


def download_model(*, cli: Any, args: str = "") -> None:
    """Download a model from HuggingFace."""
    if args:
        parts = args.split()
        repo_id = parts[0]
        filename = parts[1] if len(parts) > 1 else None
    else:
        width = min(cli.get_terminal_width() - 2, 70)

        _emit(cli)
        cli.print_box_header("Model Manager", width)
        cli.print_empty_line(width)

        option_num = 1
        available = cli.model_manager.discover_available_models()

        if available:
            cli.print_box_line("  \033[96mLoad Existing Model:\033[0m", width)
            cli.print_empty_line(width)

            for model in available[:5]:
                name = model["name"][: width - 15]
                size = f"{model['size_gb']:.1f}GB"
                line = f"    \033[93m[{option_num}]\033[0m {name} \033[2m({size})\033[0m"
                cli.print_box_line(line, width)
                option_num += 1

            if len(available) > 5:
                line = f"    \033[93m[{option_num}]\033[0m \033[2mShow all {len(available)} models...\033[0m"
                cli.print_box_line(line, width)
                option_num += 1

            cli.print_empty_line(width)
            cli.print_box_separator(width)
            cli.print_empty_line(width)

        cli.print_box_line("  \033[96mDownload New Model:\033[0m", width)
        cli.print_empty_line(width)

        line = "    \033[2mEnter repository ID (e.g., meta-llama/Llama-3.2-3B)\033[0m"
        cli.print_box_line(line, width)

        cli.print_empty_line(width)
        cli.print_box_footer(width)

        choice = cli.get_input_with_escape()

        if choice is None:
            return

        try:
            choice_num = int(choice)

            if available and choice_num <= len(available[:5]):
                model = available[choice_num - 1]
                _emit(cli, f"\n\033[96m⚡\033[0m Loading {model['name']}...")
                success, msg = cli.model_manager.load_model(model["path"])
                if success:
                    model_info = cli.model_manager.get_current_model()
                    if model_info:
                        cli.set_active_local_model(model_info.name)
                    _emit(cli, "\033[32m✓\033[0m Model loaded successfully!")

                    if model_info:
                        tokenizer = cli.model_manager.tokenizers.get(model_info.name)
                        profile = cli.template_registry.setup_model(
                            model_info.name,
                            tokenizer=tokenizer,
                            interactive=False,
                        )
                        if profile:
                            template_name = profile.config.name
                            _emit(cli, f"   \033[2m• Template: {template_name}\033[0m")
                else:
                    _emit(cli, f"\033[31m✗\033[0m Failed to load: {msg}")
                return

            if available and choice_num == len(available[:5]) + 1 and len(available) > 5:
                _emit(cli)
                cli.manage_models()
                return

            _emit(cli, "\033[31m✗ Invalid choice\033[0m")
            return

        except ValueError:
            repo_id = choice
            parts = repo_id.split()
            repo_id = parts[0]
            filename = parts[1] if len(parts) > 1 else None

    if "/" not in repo_id:
        _emit(cli, "\n\033[31m✗ Invalid format. Expected: username/model-name\033[0m")
        return

    _emit(cli, f"\n\033[96m⬇\033[0m Downloading: \033[93m{repo_id}\033[0m")
    if filename:
        _emit(cli, f"   File: \033[93m{filename}\033[0m")
    _emit(cli)

    success, message, path = cli.model_downloader.download_model(repo_id, filename)

    if success:
        width = min(cli.get_terminal_width() - 2, 70)
        _emit(cli)
        title_with_color = " \033[32mDownload Complete\033[0m "
        visible_len = cli.get_visible_length(title_with_color)
        padding = width - visible_len - 3
        _emit(cli, f"╭─{title_with_color}" + "─" * padding + "╮")
        cli.print_box_line("  \033[32m✓\033[0m Model downloaded successfully!", width)

        location_str = str(path)[: width - 13]
        cli.print_box_line(f"  \033[2mLocation: {location_str}\033[0m", width)
        cli.print_empty_line(width)
        cli.print_box_line("  \033[96mLoad this model now?\033[0m", width)
        cli.print_box_line("  \033[93m[Y]es\033[0m  \033[2m[N]o\033[0m", width)
        cli.print_box_footer(width)

        choice = cli.get_input_with_escape()
        if choice and choice.lower() in ["y", "yes"]:
            _emit(cli, "\n\033[96m⚡\033[0m Loading model...")
            load_success, load_msg = cli.model_manager.load_model(str(path))
            if load_success:
                model_info = cli.model_manager.get_current_model()
                if model_info:
                    cli.set_active_local_model(model_info.name)
                _emit(cli, "\033[32m✓\033[0m Model loaded successfully!")
            else:
                _emit(cli, f"\033[31m✗\033[0m Failed to load: {load_msg}")
    else:
        _emit(cli, f"\n\033[31m✗\033[0m {message}")
