"""Interactive template setup for user-friendly configuration."""

import re
import sys
import termios
from typing import Any, List, Optional, Protocol

from cortex.template_registry.auto_detector import TemplateDetector
from cortex.template_registry.config_manager import ModelTemplateConfig
from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateType


class _Printer(Protocol):
    def print(self, *args: object, **kwargs: object) -> object: ...


_RICH_TAG_PATTERN = re.compile(r"\[/?[^\]]+\]")


def _strip_rich_markup(text: str) -> str:
    """Remove lightweight markup tags for plain terminal output."""
    return _RICH_TAG_PATTERN.sub("", text)


class _PlainConsole:
    """Small console adapter with a console-style print API."""

    def print(self, *args: object, **kwargs: object) -> None:
        sep = str(kwargs.get("sep", " "))
        end = str(kwargs.get("end", "\n"))
        text = sep.join(str(arg) for arg in args)
        text = _strip_rich_markup(text)
        print(text, end=end)


class InteractiveTemplateSetup:
    """Interactive template configuration wizard."""

    def __init__(self, console: Optional[_Printer] = None):
        """Initialize the interactive setup."""
        self.console: _Printer = console or _PlainConsole()
        self.detector = TemplateDetector()

    def _read_line_with_escape_cancel(self) -> tuple[str, bool]:
        """Read a line from stdin and cancel immediately on ESC."""
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~termios.ICANON
            new_settings[3] = new_settings[3] & ~termios.ECHO
            new_settings[3] = new_settings[3] & ~termios.ISIG
            new_settings[6][termios.VMIN] = 1
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)

            buffer: list[str] = []
            while True:
                char = sys.stdin.read(1)

                if char in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "".join(buffer), False

                if char == "\x1b":
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "", True

                if char in ("\x7f", "\x08"):
                    if buffer:
                        buffer.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue

                if char == "\x04":
                    raise EOFError
                if char in ("\x03", "\x1a", "\x1c"):
                    raise KeyboardInterrupt

                if ord(char) >= 32:
                    buffer.append(char)
                    sys.stdout.write(char)
                    sys.stdout.flush()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _ask_selection(
        self,
        *,
        valid_choices: List[str],
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Ask for a selection with ESC-to-cancel behavior."""
        while True:
            sys.stdout.write("\n\033[96m▶\033[0m Select an option or Esc to cancel: ")
            sys.stdout.flush()
            if sys.stdin.isatty():
                value, cancelled = self._read_line_with_escape_cancel()
                if cancelled:
                    return None
            else:
                value = input().strip()
                if value == "0":
                    return None

            value = value.strip()
            if value == "" and default is not None:
                value = default

            if value in valid_choices:
                return value

            self.console.print("[red]Invalid selection.[/red]")

    def _ask_yes_no(self, *, question: str, default: bool = False) -> Optional[bool]:
        """Ask a yes/no question with ESC-to-cancel behavior."""
        self.console.print(question)
        default_choice = "y" if default else "n"
        choice = self._ask_selection(valid_choices=["y", "yes", "n", "no"], default=default_choice)
        if choice is None:
            return None
        return choice.lower() in {"y", "yes"}

    def setup_model_template(
        self,
        model_name: str,
        tokenizer: Any = None,
        current_config: Optional[ModelTemplateConfig] = None
    ) -> ModelTemplateConfig:
        """Interactive setup for model template.

        Args:
            model_name: Name of the model
            tokenizer: Optional tokenizer object
            current_config: Existing configuration if any

        Returns:
            Updated model template configuration
        """
        self.console.print(f"\n✓ Model loaded: [bold cyan]{model_name}[/bold cyan]")

        # Detect template
        profile, confidence = self.detector.detect_template(model_name, tokenizer=tokenizer)

        if confidence < 0.5:
            self.console.print("\n⚠️  [yellow]Template configuration needed for optimal performance[/yellow]")

        self.console.print("\nDetecting template format...")
        self.console.print(f"✓ Found: [green]{profile.config.description}[/green] (confidence: {confidence:.0%})")

        # Check if this is a reasoning model
        is_reasoning = profile.config.template_type == TemplateType.REASONING

        if is_reasoning:
            self.console.print("\n[yellow]Note: This model includes internal reasoning/analysis in its output.[/yellow]")

        # Show options
        self.console.print("\nHow would you like to handle this model's output?\n")

        options = []
        if is_reasoning:
            options = [
                ("simple", "Simple mode - Hide internal reasoning (recommended)", True),
                ("full", "Full mode - Show all model outputs", False),
                ("custom", "Custom - Configure manually", False),
                ("test", "Test - See examples of each mode", False)
            ]
        else:
            options = [
                ("auto", "Automatic - Use detected template", True),
                ("custom", "Custom - Configure manually", False),
                ("test", "Test - See examples with different templates", False)
            ]

        # Display options
        for i, (key, desc, recommended) in enumerate(options, 1):
            marker = " [green](recommended)[/green]" if recommended else ""
            self.console.print(f"[{i}] {desc}{marker}")

        # Get user choice
        choice = self._ask_selection(
            valid_choices=[str(i) for i in range(1, len(options) + 1)],
            default="1",
        )
        if choice is None:
            if current_config:
                return current_config
            return ModelTemplateConfig(
                detected_type=profile.config.template_type.value,
                user_preference="auto",
                custom_filters=profile.config.custom_filters,
                show_reasoning=False,
                confidence=confidence,
            )

        selected_key = options[int(choice) - 1][0]

        # Handle selection
        if selected_key == "test":
            self._show_template_tests(model_name, profile)
            # Recurse to get actual selection
            return self.setup_model_template(model_name, tokenizer, current_config)

        elif selected_key == "custom":
            return self._custom_setup(model_name, profile)

        else:
            # Create configuration
            config = ModelTemplateConfig(
                detected_type=profile.config.template_type.value,
                user_preference=selected_key,
                custom_filters=profile.config.custom_filters,
                show_reasoning=(selected_key == "full") if is_reasoning else False,
                confidence=confidence
            )

            self.console.print(f"\n✓ Template configured: [green]{selected_key} mode[/green]")
            self.console.print("✓ Configuration saved for future use")
            self.console.print("\n[dim]Tip: Use /template to adjust settings anytime[/dim]")

            return config

    def _show_template_tests(self, model_name: str, detected_profile: BaseTemplateProfile) -> None:
        """Show examples of different template modes."""
        self.console.print("\n[bold]Testing different template modes:[/bold]\n")

        test_prompt = "What is 2+2?"

        # Test different profiles
        profiles_to_test = []

        if detected_profile.config.template_type == TemplateType.REASONING:
            # Test with and without reasoning
            simple_profile = detected_profile.__class__()
            simple_profile.config.show_reasoning = False

            full_profile = detected_profile.__class__()
            full_profile.config.show_reasoning = True

            profiles_to_test = [
                ("Simple Mode", simple_profile),
                ("Full Mode", full_profile)
            ]
        else:
            # Test different template types
            from cortex.template_registry.template_profiles.standard import (
                ChatMLProfile,
                LlamaProfile,
                SimpleProfile,
            )

            profiles_to_test = [
                ("Detected", detected_profile),
                ("ChatML", ChatMLProfile()),
                ("Llama", LlamaProfile()),
                ("Simple", SimpleProfile())
            ]

        for name, profile in profiles_to_test:
            result = self.detector.test_template(profile, test_prompt)

            self.console.print(f"[bold cyan]{name}:[/bold cyan]")
            self.console.print("─" * 40)

            # Show formatted prompt
            self.console.print("[dim]Formatted prompt:[/dim]")
            self.console.print(f"  {result['formatted_prompt'][:100]}..." if len(result['formatted_prompt']) > 100 else f"  {result['formatted_prompt']}")

            # Show processed response
            self.console.print("[dim]Output:[/dim]")
            self.console.print(f"  {result['processed_response']}")
            self.console.print()

    def _custom_setup(
        self,
        model_name: str,
        detected_profile: Optional[BaseTemplateProfile],
    ) -> ModelTemplateConfig:
        """Custom template configuration."""
        self.console.print("\n[bold]Custom Template Configuration[/bold]\n")

        if detected_profile is None:
            detected_profile, confidence = self.detector.detect_template(model_name)
        else:
            confidence = 1.0

        # Select template type
        template_types = [
            ("chatml", "ChatML format"),
            ("llama", "Llama format"),
            ("alpaca", "Alpaca format"),
            ("reasoning", "Reasoning/CoT format"),
            ("simple", "Simple format")
        ]

        self.console.print("Available template types:")
        for i, (key, desc) in enumerate(template_types, 1):
            self.console.print(f"[{i}] {desc}")

        choice = self._ask_selection(
            valid_choices=[str(i) for i in range(1, len(template_types) + 1)],
            default="1",
        )
        if choice is None:
            selected_type = detected_profile.config.template_type.value
            return ModelTemplateConfig(
                detected_type=selected_type,
                user_preference="auto",
                custom_filters=detected_profile.config.custom_filters,
                show_reasoning=False,
                confidence=confidence,
            )

        selected_type = template_types[int(choice) - 1][0]

        # Configure filters
        custom_filters = []
        configure_filters = self._ask_yes_no(
            question="Configure custom output filters? (y/n)",
            default=False,
        )
        if configure_filters is None:
            return ModelTemplateConfig(
                detected_type=detected_profile.config.template_type.value,
                user_preference="auto",
                custom_filters=detected_profile.config.custom_filters,
                show_reasoning=False,
                confidence=confidence,
            )
        if configure_filters:
            filters_input = input("Enter tokens to filter (comma-separated): ")
            custom_filters = [f.strip() for f in filters_input.split(",") if f.strip()]

        # Show reasoning option
        show_reasoning = False
        if selected_type == "reasoning":
            show_reasoning_choice = self._ask_yes_no(
                question="Show internal reasoning/analysis? (y/n)",
                default=False,
            )
            if show_reasoning_choice is None:
                return ModelTemplateConfig(
                    detected_type=detected_profile.config.template_type.value,
                    user_preference="auto",
                    custom_filters=detected_profile.config.custom_filters,
                    show_reasoning=False,
                    confidence=confidence,
                )
            show_reasoning = bool(show_reasoning_choice)

        config = ModelTemplateConfig(
            detected_type=selected_type,
            user_preference="custom",
            custom_filters=custom_filters,
            show_reasoning=show_reasoning,
            confidence=1.0  # User manually configured
        )

        self.console.print("\n✓ Custom template configured")
        return config

    def show_current_config(self, model_name: str, config: ModelTemplateConfig) -> None:
        """Display current configuration for a model."""
        self.console.print(f"\nTemplate Configuration for {model_name}")
        self.console.print("Setting           Value")
        self.console.print("---------------  --------------------------------")
        self.console.print(f"Template Type    {config.detected_type}")
        self.console.print(f"User Preference  {config.user_preference}")
        self.console.print(f"Show Reasoning   {config.show_reasoning}")
        filters = ", ".join(config.custom_filters) if config.custom_filters else "None"
        self.console.print(f"Custom Filters   {filters}")
        self.console.print(f"Confidence       {config.confidence:.0%}")
        self.console.print(f"Last Updated     {config.last_updated}")

    def quick_adjust_template(self, model_name: str, config: ModelTemplateConfig) -> ModelTemplateConfig:
        """Quick adjustment interface for template settings."""
        self.console.print(f"\n[bold]Adjust template for {model_name}[/bold]\n")

        self.show_current_config(model_name, config)

        self.console.print("\n[1] Toggle reasoning display")
        self.console.print("[2] Change template type")
        self.console.print("[3] Edit filters")
        self.console.print("[4] Reset to defaults")
        self.console.print("[0] Cancel")

        choice = self._ask_selection(valid_choices=["0", "1", "2", "3", "4"])
        if choice is None:
            return config

        if choice == "1":
            config.show_reasoning = not config.show_reasoning
            self.console.print(f"✓ Reasoning display: [green]{'enabled' if config.show_reasoning else 'disabled'}[/green]")

        elif choice == "2":
            return self._custom_setup(model_name, None)

        elif choice == "3":
            current_filters = ",".join(config.custom_filters)
            prompt = "Enter tokens to filter (comma-separated)"
            if current_filters:
                prompt = f"{prompt} [{current_filters}]"
            filters_input = input(f"{prompt}: ").strip() or current_filters
            config.custom_filters = [f.strip() for f in filters_input.split(",") if f.strip()]
            self.console.print("✓ Filters updated")

        elif choice == "4":
            # Reset to detected defaults
            profile, confidence = self.detector.detect_template(model_name)
            config = ModelTemplateConfig(
                detected_type=profile.config.template_type.value,
                user_preference="auto",
                custom_filters=profile.config.custom_filters,
                show_reasoning=False,
                confidence=confidence
            )
            self.console.print("✓ Reset to defaults")

        return config
