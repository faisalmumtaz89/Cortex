"""Startup helpers for the CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cortex.ui import version_check


def print_welcome(*, cli) -> None:
    """Print welcome message."""
    width = min(cli.get_terminal_width() - 2, 70)

    cwd = os.getcwd()

    welcome_lines = [
        "\033[96m✻ Welcome to Cortex!\033[0m",
        "",
        "  \033[93m/help\033[0m for help, \033[93m/status\033[0m for your current setup",
        "",
        f"  \033[2mcwd:\033[0m {cwd}",
    ]

    print(cli.create_box(welcome_lines, width))
    print()

    if cli.config.model.last_used_model:
        display_name = cli.config.model.last_used_model
        if display_name.startswith("_Users_") and (
            "_4bit" in display_name or "_5bit" in display_name or "_8bit" in display_name
        ):
            parts = display_name.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "").split("_")
            if len(parts) > 3:
                display_name = parts[-1]
        print(f" \033[2m※ Last model:\033[0m \033[93m{display_name}\033[0m")

    print(" \033[2m※ Tip: Use\033[0m \033[93m/download\033[0m \033[2mto get models from HuggingFace\033[0m")
    update_status = version_check.get_update_status(config=cli.config)
    if update_status:
        print(
            " \033[93m↑ Update available:\033[0m "
            f"\033[2m{update_status.current_version} → {update_status.latest_version}\033[0m"
        )
        print(" \033[2mRun:\033[0m \033[93mpipx upgrade cortex-llm --force\033[0m")
    print()


def load_default_model(*, cli) -> None:
    """Load the last used model or default model if configured."""
    model_to_load = cli.config.model.last_used_model or cli.config.model.default_model

    if not model_to_load:
        print("\n \033[96m⚡\033[0m No model loaded. Use \033[93m/model\033[0m to select a model.")
        return

    if "_4bit" in model_to_load or "_5bit" in model_to_load or "_8bit" in model_to_load:
        clean_name = model_to_load
        if clean_name.startswith("_Users_"):
            parts = clean_name.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "").split("_")
            if len(parts) > 3:
                clean_name = parts[-1]

        print(f"\n \033[96m⚡\033[0m Loading: \033[93m{clean_name}\033[0m \033[2m(MLX optimized)\033[0m...")
        success, message = cli.model_manager.load_model(model_to_load)

        if success:
            model_info = cli.model_manager.get_current_model()
            if model_info:
                if "_4bit" in model_to_load:
                    quant_type = "4-bit"
                elif "_8bit" in model_to_load:
                    quant_type = "8-bit"
                elif "_5bit" in model_to_load:
                    quant_type = "5-bit"
                else:
                    quant_type = ""

                print(f" \033[32m✓\033[0m Model ready: \033[93m{clean_name}\033[0m")
                if quant_type:
                    print(f"   \033[2m• Size: {model_info.size_gb:.1f}GB ({quant_type} quantized)\033[0m")
                else:
                    print(f"   \033[2m• Size: {model_info.size_gb:.1f}GB (quantized)\033[0m")
                print("   \033[2m• Optimizations: AMX acceleration, operation fusion\033[0m")
                print("   \033[2m• Format: MLX (Apple Silicon optimized)\033[0m")

                tokenizer = cli.model_manager.tokenizers.get(model_info.name)
                profile = cli.template_registry.setup_model(
                    model_info.name,
                    tokenizer=tokenizer,
                    interactive=False,
                )
                if profile:
                    template_name = profile.config.name
                    print(f"   \033[2m• Template: {template_name}\033[0m")
        else:
            base_name = model_to_load.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "")
            if base_name.startswith("_Users_"):
                original_path = "/" + base_name[1:].replace("_", "/")
                if Path(original_path).exists():
                    print(" \033[2m※ Cached model not found, reconverting from original...\033[0m")
                    success, message = cli.model_manager.load_model(original_path)
                    if success:
                        model_info = cli.model_manager.get_current_model()
                        if model_info:
                            print(
                                f" \033[32m✓\033[0m Model loaded: \033[93m{model_info.name}\033[0m "
                                f"\033[2m({model_info.size_gb:.1f}GB, {model_info.format.value})\033[0m"
                            )
                        return

            print(f"\n \033[31m⚠\033[0m Previously used model not found: \033[93m{model_to_load}\033[0m")
            print(" Use \033[93m/model\033[0m to select a different model or \033[93m/download\033[0m to get new models.")
        return

    model_path = None

    potential_path = Path(model_to_load).expanduser()
    if potential_path.exists():
        model_path = potential_path
    else:
        potential_path = cli.config.model.model_path / model_to_load
        if potential_path.exists():
            model_path = potential_path
        else:
            available = cli.model_manager.discover_available_models()
            for model in available:
                if model["name"] == model_to_load:
                    model_path = Path(model["path"])
                    break

    if not model_path:
        print(f"\n \033[31m⚠\033[0m Previously used model not found: \033[93m{model_to_load}\033[0m")
        print(" Use \033[93m/model\033[0m to select a different model or \033[93m/download\033[0m to get new models.")
        return

    print(f"\n \033[96m⚡\033[0m Loading: \033[93m{model_to_load}\033[0m...")
    success, message = cli.model_manager.load_model(str(model_path))

    if success:
        model_info = cli.model_manager.get_current_model()
        if model_info:
            print(
                f" \033[32m✓\033[0m Model loaded: \033[93m{model_info.name}\033[0m "
                f"\033[2m({model_info.size_gb:.1f}GB, {model_info.format.value})\033[0m"
            )

            tokenizer = cli.model_manager.tokenizers.get(model_info.name)
            profile = cli.template_registry.setup_model(
                model_info.name,
                tokenizer=tokenizer,
                interactive=False,
            )
            if profile:
                template_name = profile.config.name
                print(f"   \033[2m• Template: {template_name}\033[0m")
    else:
        print(f" \033[31m✗\033[0m Failed to load model: {message}", file=sys.stderr)
        print(" Use \033[93m/model\033[0m to select a different model.")
