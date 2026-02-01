"""Interactive template setup for user-friendly configuration."""

from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateType
from cortex.template_registry.auto_detector import TemplateDetector
from cortex.template_registry.config_manager import ModelTemplateConfig


class InteractiveTemplateSetup:
    """Interactive template configuration wizard."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the interactive setup."""
        self.console = console or Console()
        self.detector = TemplateDetector()
    
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
        
        self.console.print(f"\nDetecting template format...")
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
        choice = Prompt.ask(
            f"\nSelect option (1-{len(options)})",
            default="1",
            choices=[str(i) for i in range(1, len(options) + 1)]
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
                ChatMLProfile, LlamaProfile, SimpleProfile
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
    
    def _custom_setup(self, model_name: str, detected_profile: BaseTemplateProfile) -> ModelTemplateConfig:
        """Custom template configuration."""
        self.console.print("\n[bold]Custom Template Configuration[/bold]\n")
        
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
        
        choice = Prompt.ask(
            "Select template type",
            default="1",
            choices=[str(i) for i in range(1, len(template_types) + 1)]
        )
        
        selected_type = template_types[int(choice) - 1][0]
        
        # Configure filters
        custom_filters = []
        if Confirm.ask("Configure custom output filters?", default=False):
            filters_input = Prompt.ask("Enter tokens to filter (comma-separated)")
            custom_filters = [f.strip() for f in filters_input.split(",") if f.strip()]
        
        # Show reasoning option
        show_reasoning = False
        if selected_type == "reasoning":
            show_reasoning = Confirm.ask("Show internal reasoning/analysis?", default=False)
        
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
        table = Table(title=f"Template Configuration for {model_name}")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Template Type", config.detected_type)
        table.add_row("User Preference", config.user_preference)
        table.add_row("Show Reasoning", str(config.show_reasoning))
        table.add_row("Custom Filters", ", ".join(config.custom_filters) if config.custom_filters else "None")
        table.add_row("Confidence", f"{config.confidence:.0%}")
        table.add_row("Last Updated", config.last_updated)
        
        self.console.print(table)
    
    def quick_adjust_template(self, model_name: str, config: ModelTemplateConfig) -> ModelTemplateConfig:
        """Quick adjustment interface for template settings."""
        self.console.print(f"\n[bold]Adjust template for {model_name}[/bold]\n")
        
        self.show_current_config(model_name, config)
        
        self.console.print("\n[1] Toggle reasoning display")
        self.console.print("[2] Change template type")
        self.console.print("[3] Edit filters")
        self.console.print("[4] Reset to defaults")
        self.console.print("[0] Cancel")
        
        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])
        
        if choice == "1":
            config.show_reasoning = not config.show_reasoning
            self.console.print(f"✓ Reasoning display: [green]{'enabled' if config.show_reasoning else 'disabled'}[/green]")
        
        elif choice == "2":
            return self._custom_setup(model_name, None)
        
        elif choice == "3":
            filters_input = Prompt.ask("Enter tokens to filter (comma-separated)", default=",".join(config.custom_filters))
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