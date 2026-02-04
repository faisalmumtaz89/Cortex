"""Fine-tuning UI helpers for CLI."""

from __future__ import annotations


def run_finetune(*, cli) -> None:
    """Run the interactive fine-tuning wizard."""
    available = cli.model_manager.discover_available_models()
    if not available:
        print("\n\033[31m✗\033[0m No models found. Use \033[93m/download\033[0m to download a model first.")
        return

    cli.fine_tune_wizard.cli = cli

    success, message = cli.fine_tune_wizard.start()

    if success:
        print(f"\n\033[32m✓\033[0m {message}")
    else:
        if "cancelled" not in message.lower():
            print(f"\n\033[31m✗\033[0m {message}")
