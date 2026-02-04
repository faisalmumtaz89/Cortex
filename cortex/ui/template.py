"""Template management UI helpers for CLI."""

from __future__ import annotations


def manage_template(*, cli, args: str = "") -> None:
    """Manage template configuration for the current model."""
    if not cli.model_manager.current_model:
        print("\033[31m✗\033[0m No model loaded.")
        return

    model_name = cli.model_manager.current_model
    tokenizer = cli.model_manager.tokenizers.get(model_name)

    if args:
        args_parts = args.split()
        subcommand = args_parts[0].lower()

        if subcommand == "reset":
            if cli.template_registry.reset_model_config(model_name):
                print(f"\033[32m✓\033[0m Template configuration reset for {model_name}")
            else:
                print(f"\033[31m✗\033[0m No configuration found for {model_name}")
            return
        if subcommand == "status":
            config = cli.template_registry.config_manager.get_model_config(model_name)
            if config:
                cli.template_registry.interactive.show_current_config(model_name, config)
            else:
                print(f"\033[33m⚠\033[0m No template configuration for {model_name}")
            return

    print(f"\n\033[96m⚙\033[0m Configuring template for: \033[93m{model_name}\033[0m")

    cli.template_registry.setup_model(
        model_name,
        tokenizer=tokenizer,
        interactive=True,
        force_setup=True,
    )

    print("\n\033[32m✓\033[0m Template configured successfully!")
