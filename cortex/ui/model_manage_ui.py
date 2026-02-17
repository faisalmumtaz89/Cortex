"""Model management UI helpers for CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.text import Text

from cortex.cloud.types import CloudModelRef


def _emit(cli: Any, text: str = "") -> None:
    """Render ANSI-formatted legacy text through Rich console."""
    if not text:
        cli.console.print()
        return
    cli.console.print(Text.from_ansi(text))


def manage_models(*, cli: Any, args: str = "") -> None:
    """Interactive model manager for local and cloud models."""
    selector = args.strip()
    if selector:
        _handle_direct_selector(cli=cli, selector=selector)
        return

    local_models = cli.model_manager.discover_available_models()
    cloud_enabled = getattr(cli.config.cloud, "cloud_enabled", True)
    cloud_models = cli.cloud_catalog.list_models() if cloud_enabled else []
    width = min(cli.get_terminal_width() - 2, 70)

    if not local_models and not cloud_models:
        _emit(cli, f"\n\033[31m✗\033[0m No models found in \033[2m{cli.config.model.model_path}\033[0m")
        _emit(cli, "Use \033[93m/download\033[0m to download local models.")
        return

    entries: List[Tuple[str, object]] = []

    _emit(cli)
    cli.print_box_header("Select Model", width)
    cli.print_empty_line(width)

    if local_models:
        cli.print_box_line("  \033[96mLocal Models:\033[0m", width)
        cli.print_empty_line(width)
        for model in local_models:
            entry_number = len(entries) + 1
            entries.append(("local", model))
            _print_local_model_line(cli=cli, width=width, index=entry_number, model=model)
    else:
        cli.print_box_line("  \033[2mNo local models found.\033[0m", width)

    cli.print_empty_line(width)
    cli.print_box_separator(width)
    cli.print_empty_line(width)

    if cloud_enabled:
        cli.print_box_line("  \033[96mCloud Models:\033[0m", width)
        cli.print_empty_line(width)
        for cloud_ref in cloud_models:
            entry_number = len(entries) + 1
            entries.append(("cloud", cloud_ref))
            _print_cloud_model_line(cli=cli, width=width, index=entry_number, model_ref=cloud_ref)
    else:
        cli.print_box_line("  \033[2mCloud models are disabled by config (cloud_enabled=false).\033[0m", width)

    cli.print_empty_line(width)
    cli.print_box_separator(width)
    cli.print_empty_line(width)
    cli.print_box_line("  \033[93m[D]\033[0m Delete a local model", width)
    cli.print_box_line("  \033[93m[N]\033[0m Download new local model", width)
    if cloud_enabled:
        cli.print_box_line("  \033[93m[C]\033[0m Enter cloud model ID (provider:model)", width)
    cli.print_empty_line(width)
    cli.print_box_footer(width)

    choice = cli.get_input_with_escape()
    if choice is None:
        return

    normalized = choice.strip().lower()
    if normalized == "n":
        cli.download_model()
        return
    if normalized == "d":
        _delete_local_model(cli=cli, local_models=local_models)
        return
    if normalized == "c":
        if not cloud_enabled:
            _emit(cli, "\033[31m✗\033[0m Cloud models are disabled in config.yaml (cloud_enabled=false).")
            return
        manual = cli.get_input_with_escape("Enter selector (openai:model or anthropic:model)")
        if manual:
            _handle_direct_selector(cli=cli, selector=manual)
        return

    try:
        selected_idx = int(normalized) - 1
    except ValueError:
        _emit(cli, "\033[31m✗\033[0m Invalid choice")
        return

    if selected_idx < 0 or selected_idx >= len(entries):
        _emit(cli, "\033[31m✗\033[0m Invalid selection")
        return

    entry_type, payload = entries[selected_idx]
    if entry_type == "local":
        _load_local_model(cli=cli, model=payload)  # type: ignore[arg-type]
        return

    _select_cloud_model(cli=cli, model_ref=payload)  # type: ignore[arg-type]


def _handle_direct_selector(*, cli: Any, selector: str) -> None:
    """Handle /model argument form."""
    cloud_ref = cli.cloud_catalog.parse_selector(selector)
    if cloud_ref is not None:
        if not getattr(cli.config.cloud, "cloud_enabled", True):
            _emit(cli, "\033[31m✗\033[0m Cloud models are disabled in config.yaml (cloud_enabled=false).")
            return
        _select_cloud_model(cli=cli, model_ref=cloud_ref)
        return

    _emit(cli, f"\033[96m⚡\033[0m Loading local model: \033[93m{selector}\033[0m...")
    success, message = cli.model_manager.load_model(selector)
    if success:
        model_info = cli.model_manager.get_current_model()
        if model_info:
            cli.set_active_local_model(model_info.name)
        _emit(cli, "\033[32m✓\033[0m Local model loaded successfully")
    else:
        _emit(cli, f"\033[31m✗\033[0m Failed: {message}")


def _print_local_model_line(*, cli: Any, width: int, index: int, model: Dict[str, Any]) -> None:
    name = model["name"][: width - 35]
    size = f"{model['size_gb']:.1f}GB"

    active_is_local = cli.active_model_target.backend == "local"
    current_model = cli.model_manager.current_model or ""
    is_current = active_is_local and (
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
        status_parts.append("\033[32m● active\033[0m")

    status = " ".join(status_parts) if status_parts else ""
    line = f"  \033[93m[{index}]\033[0m {name} \033[2m({size})\033[0m {status}"
    cli.print_box_line(line, width)


def _print_cloud_model_line(*, cli: Any, width: int, index: int, model_ref: CloudModelRef) -> None:
    is_auth, source = cli.cloud_router.get_auth_status(model_ref.provider)
    source_label = source or "none"
    active = (
        cli.active_model_target.backend == "cloud"
        and cli.active_model_target.cloud_model is not None
        and cli.active_model_target.cloud_model.selector == model_ref.selector
    )
    if active and is_auth:
        status_label = "\033[32m● active\033[0m"
    elif is_auth:
        status_label = "\033[32m● ready\033[0m"
    elif active:
        status_label = "\033[33m● active\033[0m \033[31m○ login required\033[0m"
    else:
        status_label = "\033[31m○ login required\033[0m"

    line = (
        f"  \033[93m[{index}]\033[0m "
        f"{model_ref.selector[: width - 35]} \033[2m({source_label})\033[0m {status_label}"
    )
    cli.print_box_line(line, width)


def _select_cloud_model(*, cli: Any, model_ref: CloudModelRef) -> None:
    is_auth, _ = cli.cloud_router.get_auth_status(model_ref.provider)
    if not is_auth:
        _emit(
            cli,
            "\n\033[31m✗\033[0m Missing API key for "
            f"\033[93m{model_ref.provider.value}\033[0m. "
            f"Use \033[93m/login {model_ref.provider.value}\033[0m."
        )
        return

    ok, message = cli.set_active_cloud_model(model_ref.provider, model_ref.model_id)
    if not ok:
        _emit(cli, f"\033[31m✗\033[0m {message}")
        return

    _emit(cli, f"\n\033[32m✓\033[0m {message}")
    _emit(cli, "\033[2mYou can now chat with the selected cloud model.\033[0m")


def _delete_local_model(*, cli: Any, local_models: List[Dict[str, Any]]) -> None:
    if not local_models:
        _emit(cli, "\033[31m✗\033[0m No local models to delete.")
        return

    delete_choice = cli.get_input_with_escape()
    if delete_choice is None:
        return

    try:
        model_idx = int(delete_choice) - 1
    except ValueError:
        _emit(cli, "\033[31m✗\033[0m Invalid selection")
        return

    if model_idx < 0 or model_idx >= len(local_models):
        _emit(cli, "\033[31m✗\033[0m Invalid selection")
        return

    selected_model = local_models[model_idx]
    _emit(cli, f"\n\033[31m⚠\033[0m Delete \033[93m{selected_model['name']}\033[0m?")
    _emit(cli, f"   This will free \033[93m{selected_model['size_gb']:.1f}GB\033[0m of disk space.")
    confirm = cli.get_input_with_escape()
    if confirm is None or confirm.lower() != "y":
        _emit(cli, "\033[2mDeletion cancelled.\033[0m")
        return

    model_path = Path(selected_model["path"])
    try:
        if model_path.is_file():
            model_path.unlink()
        elif model_path.is_dir():
            import shutil

            shutil.rmtree(model_path)
    except Exception as exc:
        _emit(cli, f"\033[31m✗\033[0m Failed to delete: {exc}")
        return

    _emit(
        cli,
        "\033[32m✓\033[0m Model deleted successfully. "
        f"Freed \033[93m{selected_model['size_gb']:.1f}GB\033[0m."
    )

    if selected_model["name"] == cli.model_manager.current_model:
        cli.model_manager.current_model = None
        if cli.active_model_target.backend == "local":
            cli.set_active_local_model(None)
        _emit(
            cli,
            "\033[2mNote: Deleted model was currently loaded. "
            "Load another model or switch to cloud to continue.\033[0m"
        )


def _load_local_model(*, cli: Any, model: Dict[str, Any]) -> None:
    if model["name"] == cli.model_manager.current_model and cli.active_model_target.backend == "local":
        _emit(cli, f"\033[2mModel already active: {model['name']}\033[0m")
        return

    if model["name"] == cli.model_manager.current_model:
        cli.set_active_local_model(model["name"])
        _emit(cli, f"\033[32m✓\033[0m Switched active backend to local model: \033[93m{model['name']}\033[0m")
        return

    _emit(cli, f"\n\033[96m⚡\033[0m Loading \033[93m{model['name']}\033[0m...")
    success, message = cli.model_manager.load_model(model["path"])
    if not success:
        _emit(cli, f"\033[31m✗\033[0m Failed to load: {message}")
        return

    model_info = cli.model_manager.get_current_model()
    if not model_info:
        _emit(cli, "\033[32m✓\033[0m Model loaded successfully.")
        return

    cli.set_active_local_model(model_info.name)
    model_name = model_info.name
    if "_4bit" in model_name or "4bit" in str(model_info.quantization):
        quant_type = "4-bit"
    elif "_5bit" in model_name or "5bit" in str(model_info.quantization):
        quant_type = "5-bit"
    elif "_8bit" in model_name or "8bit" in str(model_info.quantization):
        quant_type = "8-bit"
    else:
        quant_type = ""

    clean_name = model["name"]
    if clean_name.startswith("_Users_"):
        parts = clean_name.split("_")
        for idx, part in enumerate(parts):
            if "models" in part:
                clean_name = "_".join(parts[idx + 1 :])
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

    _emit(cli, f" \033[32m✓\033[0m Model ready: \033[93m{clean_name}\033[0m")
    if quant_type:
        _emit(cli, f"   \033[2m• Size: {model_info.size_gb:.1f}GB ({quant_type} quantized)\033[0m")
    else:
        _emit(cli, f"   \033[2m• Size: {model_info.size_gb:.1f}GB (quantized)\033[0m")
    _emit(cli, "   \033[2m• Optimizations: AMX acceleration, operation fusion\033[0m")
    _emit(cli, f"   \033[2m• Format: {format_display}\033[0m")

    tokenizer = cli.model_manager.tokenizers.get(model_info.name)
    profile = cli.template_registry.setup_model(
        model_info.name,
        tokenizer=tokenizer,
        interactive=False,
    )
    if profile:
        template_name = profile.config.name
        _emit(cli, f"   \033[2m• Template: {template_name}\033[0m")
