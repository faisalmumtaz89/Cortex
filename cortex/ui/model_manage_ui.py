"""Model management UI helpers for CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def manage_models(*, cli: Any, args: str = "") -> None:
    """Interactive model manager - simplified for better UX."""
    if args:
        print(f"\033[96m⚡\033[0m Loading model: \033[93m{args}\033[0m...")
        success, message = cli.model_manager.load_model(args)
        if success:
            print("\033[32m✓\033[0m Model loaded successfully")
        else:
            print(f"\033[31m✗\033[0m Failed: {message}", file=sys.stderr)
        return

    available = cli.model_manager.discover_available_models()

    if not available:
        print(f"\n\033[31m✗\033[0m No models found in \033[2m{cli.config.model.model_path}\033[0m")
        print("Use \033[93m/download\033[0m to download models from HuggingFace")
        return

    width = min(cli.get_terminal_width() - 2, 70)

    print()
    cli.print_box_header("Select Model", width)
    cli.print_empty_line(width)

    for i, model in enumerate(available, 1):
        name = model["name"][: width - 30]
        size = f"{model['size_gb']:.1f}GB"

        current_model = cli.model_manager.current_model or ""
        is_current = (
            model["name"] == current_model
            or model.get("mlx_name") == current_model
            or current_model.endswith(model["name"])
        )

        status_parts = []
        if model.get("mlx_optimized"):
            status_parts.append("\033[36m⚡ MLX\033[0m")
        elif model.get("mlx_available"):
            status_parts.append("\033[2m○ MLX ready\033[0m")

        if is_current:
            status_parts.append("\033[32m● loaded\033[0m")

        status = " ".join(status_parts) if status_parts else ""

        line = f"  \033[93m[{i}]\033[0m {name} \033[2m({size})\033[0m {status}"
        cli.print_box_line(line, width)

    cli.print_empty_line(width)
    cli.print_box_separator(width)
    cli.print_empty_line(width)

    cli.print_box_line("  \033[93m[D]\033[0m Delete a model", width)
    cli.print_box_line("  \033[93m[N]\033[0m Download new model", width)

    cli.print_empty_line(width)
    cli.print_box_footer(width)

    choice = cli.get_input_with_escape(f"Select model to load (1-{len(available)}) or option")

    if choice is None:
        return

    choice = choice.lower()

    if choice == "n":
        cli.download_model()
        return
    if choice == "d":
        del_choice = cli.get_input_with_escape(f"Select model to delete (1-{len(available)})")
        if del_choice is None:
            return
        try:
            model_idx = int(del_choice) - 1
            if 0 <= model_idx < len(available):
                selected_model = available[model_idx]
                print(f"\n\033[31m⚠\033[0m Delete \033[93m{selected_model['name']}\033[0m?")
                print(f"   This will free \033[93m{selected_model['size_gb']:.1f}GB\033[0m of disk space.")
                confirm = cli.get_input_with_escape("Confirm deletion (\033[93my\033[0m/\033[2mN\033[0m)")
                if confirm is None:
                    return
                confirm = confirm.lower()

                if confirm == "y":
                    model_path = Path(selected_model["path"])
                    try:
                        if model_path.is_file():
                            model_path.unlink()
                        elif model_path.is_dir():
                            import shutil

                            shutil.rmtree(model_path)

                        print(
                            "\033[32m✓\033[0m Model deleted successfully. "
                            f"Freed \033[93m{selected_model['size_gb']:.1f}GB\033[0m."
                        )

                        if selected_model["name"] == cli.model_manager.current_model:
                            cli.model_manager.current_model = None
                            print(
                                "\033[2mNote: Deleted model was currently loaded. "
                                "Load another model to continue.\033[0m"
                            )
                    except Exception as e:
                        print(f"\033[31m✗\033[0m Failed to delete: {str(e)}")
                else:
                    print("\033[2mDeletion cancelled.\033[0m")
        except (ValueError, IndexError):
            print("\033[31m✗\033[0m Invalid selection")
        return

    try:
        model_idx = int(choice) - 1
        if 0 <= model_idx < len(available):
            selected_model = available[model_idx]

            if selected_model["name"] == cli.model_manager.current_model:
                print(f"\033[2mModel already loaded: {selected_model['name']}\033[0m")
                return

            print(f"\n\033[96m⚡\033[0m Loading \033[93m{selected_model['name']}\033[0m...")
            success, message = cli.model_manager.load_model(selected_model["path"])
            if success:
                model_info = cli.model_manager.get_current_model()
                if model_info:
                    model_name = model_info.name
                    if "_4bit" in model_name or "4bit" in str(model_info.quantization):
                        quant_type = "4-bit"
                    elif "_5bit" in model_name or "5bit" in str(model_info.quantization):
                        quant_type = "5-bit"
                    elif "_8bit" in model_name or "8bit" in str(model_info.quantization):
                        quant_type = "8-bit"
                    else:
                        quant_type = ""

                    clean_name = selected_model["name"]
                    if clean_name.startswith("_Users_"):
                        parts = clean_name.split("_")
                        for i, part in enumerate(parts):
                            if "models" in part:
                                clean_name = "_".join(parts[i + 1 :])
                                break
                    clean_name = clean_name.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "")

                    format_display = model_info.format.value
                    if format_display.lower() == "mlx":
                        format_display = "MLX (Apple Silicon optimized)"
                    elif format_display.lower() == "gguf":
                        format_display = "GGUF"
                    elif format_display.lower() == "safetensors":
                        format_display = "SafeTensors"
                    elif format_display.lower() == "pytorch":
                        format_display = "PyTorch"

                    print(f" \033[32m✓\033[0m Model ready: \033[93m{clean_name}\033[0m")
                    if quant_type:
                        print(f"   \033[2m• Size: {model_info.size_gb:.1f}GB ({quant_type} quantized)\033[0m")
                    else:
                        print(f"   \033[2m• Size: {model_info.size_gb:.1f}GB (quantized)\033[0m")
                    print("   \033[2m• Optimizations: AMX acceleration, operation fusion\033[0m")
                    print(f"   \033[2m• Format: {format_display}\033[0m")

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
                    print("\033[32m✓\033[0m Model loaded successfully!")
            else:
                print(f"\033[31m✗\033[0m Failed to load: {message}")
        else:
            print("\033[31m✗\033[0m Invalid selection")
    except ValueError:
        print("\033[31m✗\033[0m Invalid choice")
