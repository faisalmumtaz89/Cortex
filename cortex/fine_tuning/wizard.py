"""Interactive fine-tuning wizard for Cortex."""

import json
import logging
import sys
import termios
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from cortex.config import Config
from cortex.model_manager import ModelManager

from .dataset import DatasetPreparer
from .mlx_lora_trainer import MLXLoRATrainer
from .trainer import SmartConfigFactory, TrainingConfig

logger = logging.getLogger(__name__)


class WizardCLI(Protocol):
    """Protocol for CLI capabilities used by the fine-tuning wizard."""

    def get_terminal_width(self) -> int:
        ...

    def get_input_with_escape(self) -> Optional[str]:
        ...

    def print_box_header(self, title: str, width: int) -> None:
        ...

    def print_box_footer(self, width: int) -> None:
        ...

    def print_box_line(self, content: str, width: int) -> None:
        ...

    def print_empty_line(self, width: int) -> None:
        ...

    def print_box_separator(self, width: int) -> None:
        ...


class FineTuneWizard:
    """Interactive wizard for fine-tuning models - Cortex style."""

    def __init__(self, model_manager: ModelManager, config: Config):
        """Initialize the fine-tuning wizard."""
        self.model_manager = model_manager
        self.config = config
        self.trainer: MLXLoRATrainer | None = None
        self.dataset_preparer = DatasetPreparer()
        self.cli: WizardCLI | None = None  # Set by CLI runner.

    def _require_cli(self) -> WizardCLI:
        """Return an attached CLI implementation or fail fast."""
        if self.cli is None:
            raise RuntimeError("FineTuneWizard requires an attached CLI renderer")
        return self.cli

    def get_terminal_width(self) -> int:
        """Get terminal width."""
        if self.cli:
            return self.cli.get_terminal_width()
        return 80

    def _read_line_with_escape_cancel(self) -> tuple[str, bool]:
        """Read a line and cancel immediately when ESC is pressed."""
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~termios.ICANON
            new_settings[3] = new_settings[3] & ~termios.ECHO
            new_settings[3] = new_settings[3] & ~termios.ISIG
            new_settings[6][termios.VMIN] = 1
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)

            buffer: List[str] = []
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

    def _get_selection_input(self, default: Optional[str] = None) -> Optional[str]:
        """Read a selection-style input with ESC cancellation support."""
        if self.cli:
            value = self.cli.get_input_with_escape()
        else:
            print("\n\033[96m▶\033[0m Select an option or Esc to cancel: ", end="", flush=True)
            if not sys.stdin.isatty():
                raw = input().strip()
                if raw == "0":
                    return None
                value = raw
            else:
                raw, cancelled = self._read_line_with_escape_cancel()
                if cancelled:
                    return None
                value = raw.strip()
            if value == "0":
                return None

        if value is None:
            return None

        if value == "" and default is not None:
            return default

        return value.strip()

    def start(self) -> Tuple[bool, str]:
        """Start the interactive fine-tuning experience."""

        try:
            # Hard block if MLX is not installed/available. Fine-tuning depends on it.
            if not MLXLoRATrainer.is_available():
                message = "Fine-tuning requires MLX/Metal, but the MLX stack is not available in this environment."
                print(f"\n\033[31m✗\033[0m {message}")
                return False, message
            # Step 1: Select base model
            base_model = self._select_base_model()
            if not base_model:
                return False, "Fine-tuning cancelled"

            # Step 2: Select or prepare dataset
            dataset_path = self._prepare_dataset()
            if not dataset_path:
                return False, "Fine-tuning cancelled"

            # Step 3: Configure training settings
            training_config = self._configure_training(base_model, dataset_path)
            if not training_config:
                return False, "Fine-tuning cancelled"

            # Step 4: Choose output name
            output_name = self._get_output_name(base_model)
            if not output_name:
                return False, "Fine-tuning cancelled"

            # Step 5: Confirm and start training
            if not self._confirm_settings(base_model, dataset_path, training_config, output_name):
                return False, "Fine-tuning cancelled"

            # Step 6: Run training
            success = self._run_training(base_model, dataset_path, training_config, output_name)

            if success:
                return True, f"Fine-tuned model saved as: {output_name}"
            else:
                return False, "Training failed"

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            print(f"\n\033[31m✗\033[0m File not found: {e}")
            return False, f"File not found: {str(e)}"
        except PermissionError as e:
            logger.error(f"Permission denied: {e}")
            print(f"\n\033[31m✗\033[0m Permission denied: {e}")
            return False, f"Permission denied: {str(e)}"
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            print(f"\n\033[31m✗\033[0m Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Fine-tuning failed: {str(e)}"

    def _select_base_model(self) -> Optional[str]:
        """Select the base model to fine-tune."""
        cli = self._require_cli()
        width = min(self.get_terminal_width() - 2, 70)

        # Get available models
        models = self._get_available_models()

        if not models:
            print("\033[31m✗\033[0m No models available. Use \033[93m/download\033[0m to get models.")
            return None

        # Check if a model is already loaded
        if self.model_manager.current_model:
            current_model_name = str(self.model_manager.current_model)

            # Create dialog box for current model
            print()
            cli.print_box_header("Fine-Tuning Setup", width)
            cli.print_empty_line(width)

            cli.print_box_line(f"  \033[96mCurrent Model:\033[0m \033[93m{current_model_name}\033[0m", width)

            cli.print_empty_line(width)
            cli.print_box_separator(width)
            cli.print_empty_line(width)

            cli.print_box_line("  Use this model for fine-tuning?", width)
            cli.print_empty_line(width)
            cli.print_box_line("  \033[93m[Y]\033[0m Yes, use this model", width)
            cli.print_box_line("  \033[93m[N]\033[0m No, select another", width)

            cli.print_empty_line(width)
            cli.print_box_footer(width)

            choice = self._get_selection_input(default="y")
            if choice is None:
                return None
            choice = choice.lower()

            if choice in ['y', 'yes', '']:
                print(f"\033[32m✓\033[0m Using: {current_model_name}")
                return current_model_name

        # Show model selection dialog
        print()
        cli.print_box_header("Select Base Model", width)
        cli.print_empty_line(width)

        # List models with numbers
        for i, (name, info) in enumerate(models[:10], 1):
            size_str = f"{info['size_gb']:.1f}GB"
            format_str = info['format']
            line = f"  \033[93m[{i}]\033[0m {name} \033[2m({size_str}, {format_str})\033[0m"
            cli.print_box_line(line, width)

        if len(models) > 10:
            cli.print_empty_line(width)
            cli.print_box_line(f"  \033[2m... and {len(models) - 10} more models available\033[0m", width)

        cli.print_empty_line(width)
        cli.print_box_footer(width)

        # Get user selection
        choice = self._get_selection_input()

        if choice is None:
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model = models[idx][0]
                print(f"\033[32m✓\033[0m Selected: {selected_model}")
                return selected_model
            else:
                print("\033[31m✗\033[0m Invalid selection")
                return None
        except ValueError:
            print("\033[31m✗\033[0m Please enter a valid number")
            return None

    def _prepare_dataset(self) -> Optional[Path]:
        """Prepare the training dataset."""
        cli = self._require_cli()
        width = min(self.get_terminal_width() - 2, 70)

        # Show dataset options dialog
        print()
        cli.print_box_header("Training Data", width)
        cli.print_empty_line(width)

        cli.print_box_line("  \033[96mSelect data source:\033[0m", width)
        cli.print_empty_line(width)

        cli.print_box_line("  \033[93m[1]\033[0m Load from file \033[2m(JSONL/CSV/TXT)\033[0m", width)
        cli.print_box_line("  \033[93m[2]\033[0m Create interactively", width)
        cli.print_box_line("  \033[93m[3]\033[0m Use sample dataset \033[2m(for testing)\033[0m", width)

        cli.print_empty_line(width)
        cli.print_box_footer(width)

        choice = self._get_selection_input()

        if choice is None:
            return None

        if choice == "1":
            return self._load_existing_dataset()
        elif choice == "2":
            return self._create_interactive_dataset()
        elif choice == "3":
            return self._create_sample_dataset()
        else:
            print("\033[31m✗\033[0m Invalid selection")
            return None

    def _load_existing_dataset(self) -> Optional[Path]:
        """Load an existing dataset file."""
        while True:
            raw_path = input("\n\033[96m▶\033[0m Path to dataset file: ").strip()
            if not raw_path:
                return None

            # Expand user path
            file_path = Path(raw_path).expanduser()

            if file_path.exists():
                # Validate dataset format
                print("\033[96m⚡\033[0m Validating dataset...")
                valid, message, processed_path = self.dataset_preparer.validate_dataset(file_path)
                if valid:
                    print(f"\033[32m✓\033[0m {message}")
                    if isinstance(processed_path, Path):
                        return processed_path
                    return Path(str(processed_path))
                else:
                    print(f"\033[31m✗\033[0m {message}")
                    print("   Try a different dataset file? \033[2m[Y/N]\033[0m")
                    retry = self._get_selection_input(default="n")
                    if retry is None or retry.lower() not in ['y', 'yes']:
                        return None
            else:
                print(f"\033[31m✗\033[0m File not found: {file_path}")
                print("   Try a different dataset file? \033[2m[Y/N]\033[0m")
                retry = self._get_selection_input(default="n")
                if retry is None or retry.lower() not in ['y', 'yes']:
                    return None

    def _create_interactive_dataset(self) -> Optional[Path]:
        """Create a dataset interactively."""
        cli = self._require_cli()
        width = min(self.get_terminal_width() - 2, 70)

        print()
        cli.print_box_header("Interactive Dataset Creation", width)
        cli.print_empty_line(width)
        cli.print_box_line("  Enter prompt-response pairs.", width)
        cli.print_box_line("  Type '\033[93mdone\033[0m' when finished.", width)
        cli.print_box_line("  \033[2mMinimum 5 examples recommended.\033[0m", width)
        cli.print_empty_line(width)
        cli.print_box_footer(width)

        examples: List[Dict[str, str]] = []
        example_num = 1

        while True:
            print(f"\n\033[96mExample {example_num}:\033[0m")
            prompt = input("  \033[96m▶\033[0m Prompt: ").strip()

            if prompt.lower() == "done":
                if len(examples) < 5:
                    print(f"\033[93m⚠\033[0m You have {len(examples)} examples. Minimum recommended: 5")
                    print("   Continue with this smaller dataset? \033[2m[Y/N]\033[0m")
                    cont = self._get_selection_input(default="n")
                    if cont is None or cont.lower() != 'y':
                        continue
                break

            if not prompt:
                break

            response = input("  \033[96m▶\033[0m Response: ").strip()
            if not response:
                print("\033[31m✗\033[0m Response required")
                continue

            examples.append({
                "prompt": prompt,
                "response": response
            })

            example_num += 1
            print("\033[32m✓\033[0m Added")

        if not examples:
            print("\033[31m✗\033[0m No examples provided")
            return None

        # Save to temporary file
        dataset_path = Path.home() / ".cortex" / "temp_datasets" / "interactive_dataset.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dataset_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

        print(f"\033[32m✓\033[0m Created dataset with {len(examples)} examples")
        return dataset_path

    def _create_sample_dataset(self) -> Optional[Path]:
        """Create a sample dataset for testing."""
        print("\n\033[96m⚡\033[0m Creating sample dataset...")

        dataset_path = self.dataset_preparer.create_sample_dataset("general")
        print("\033[32m✓\033[0m Sample dataset created (5 examples)")

        if isinstance(dataset_path, Path):
            return dataset_path
        return Path(str(dataset_path))

    def _configure_training(self, base_model: str, dataset_path: Path) -> Optional[TrainingConfig]:
        """Configure training settings using intelligent presets."""
        cli = self._require_cli()
        width = min(self.get_terminal_width() - 2, 70)

        # Get model information for smart configuration
        model_info = self._get_model_info(base_model)
        model_size_gb = model_info.get('size_gb', 1.0) if model_info else 1.0
        model_path = str(model_info.get('path', '')) if model_info else None

        # Analyze model and dataset for smart defaults with accurate parameter detection
        model_category, estimated_params = SmartConfigFactory.categorize_model_size(
            model_size_gb, self.model_manager, model_path
        )
        dataset_info = SmartConfigFactory.analyze_dataset(dataset_path)

        # Show intelligent configuration dialog
        print()
        cli.print_box_header("Smart Training Configuration", width)
        cli.print_empty_line(width)

        # Show detected characteristics
        cli.print_box_line("  \033[96mDetected:\033[0m", width)
        cli.print_box_line(f"  Model: \033[93m{model_category.title()}\033[0m ({estimated_params:.1f}B params, {model_size_gb:.1f}GB)", width)
        cli.print_box_line(f"  Dataset: \033[93m{dataset_info['size_category'].title()}\033[0m ({dataset_info['size']} examples)", width)
        cli.print_box_line(f"  Task type: \033[93m{dataset_info['task_type'].title()}\033[0m", width)

        cli.print_empty_line(width)
        cli.print_box_separator(width)
        cli.print_empty_line(width)

        cli.print_box_line("  \033[96mSelect training preset:\033[0m", width)
        cli.print_empty_line(width)

        cli.print_box_line("  \033[93m[1]\033[0m Quick    \033[2m(fast experimentation)\033[0m", width)
        cli.print_box_line("  \033[93m[2]\033[0m Balanced \033[2m(recommended for most cases)\033[0m", width)
        cli.print_box_line("  \033[93m[3]\033[0m Quality  \033[2m(best results, longer training)\033[0m", width)
        cli.print_box_line("  \033[93m[4]\033[0m Expert   \033[2m(full customization)\033[0m", width)

        cli.print_empty_line(width)
        cli.print_box_footer(width)

        choice = self._get_selection_input()

        if choice is None:
            return None

        preset_map = {
            "1": "quick",
            "2": "balanced",
            "3": "quality"
        }

        config: TrainingConfig | None = None

        if choice in preset_map:
            # Use smart configuration
            preset = preset_map[choice]
            config = SmartConfigFactory.create_smart_config(
                model_size_gb=model_size_gb,
                dataset_path=dataset_path,
                preset=preset,
                model_manager=self.model_manager,
                model_path=model_path
            )

            # Show what the smart config decided
            print("\n\033[96m⚡\033[0m Smart configuration applied:")
            guidance = SmartConfigFactory.generate_guidance_message(config, base_model)
            print(f"   {guidance}")

        elif choice == "4":
            # Expert mode - full customization
            config = self._expert_configuration(model_size_gb, dataset_path, model_category, model_path)
        else:
            print("\033[31m✗\033[0m Invalid selection")
            return None

        if config is None:
            return None

        # Auto-adjust quantization based on model size
        if model_size_gb > 30 and not config.quantization_bits:
            config.quantization_bits = 4
            print("\033[93m※\033[0m Auto-enabled 4-bit quantization for large model")
        elif model_size_gb > 13 and not config.quantization_bits:
            config.quantization_bits = 8
            print("\033[93m※\033[0m Auto-enabled 8-bit quantization for medium model")

        return config

    def _expert_configuration(self, model_size_gb: float, dataset_path: Path, model_category: str, model_path: Optional[str] = None) -> Optional[TrainingConfig]:
        """Expert mode configuration with full customization."""
        cli = self._require_cli()
        width = min(self.get_terminal_width() - 2, 70)

        print()
        cli.print_box_header("Expert Configuration", width)
        cli.print_empty_line(width)
        cli.print_box_line("  \033[96mConfigure advanced settings:\033[0m", width)
        cli.print_box_line("  \033[2mPress Enter to use smart defaults\033[0m", width)
        cli.print_empty_line(width)
        cli.print_box_footer(width)

        # Get smart defaults as starting point
        smart_config = SmartConfigFactory.create_smart_config(
            model_size_gb=model_size_gb,
            dataset_path=dataset_path,
            preset="balanced",
            model_manager=self.model_manager,
            model_path=model_path
        )

        try:
            # Core training parameters
            print("\n\033[96m━━━ Core Training Parameters ━━━\033[0m")
            epochs_str = input(f"\033[96m▶\033[0m Epochs \033[2m[{smart_config.epochs}]\033[0m: ").strip()
            epochs = int(epochs_str) if epochs_str else smart_config.epochs

            lr_str = input(f"\033[96m▶\033[0m Learning rate \033[2m[{smart_config.learning_rate:.1e}]\033[0m: ").strip()
            learning_rate = float(lr_str) if lr_str else smart_config.learning_rate

            batch_str = input(f"\033[96m▶\033[0m Batch size \033[2m[{smart_config.batch_size}]\033[0m: ").strip()
            batch_size = int(batch_str) if batch_str else smart_config.batch_size

            grad_acc_str = input(f"\033[96m▶\033[0m Gradient accumulation steps \033[2m[{smart_config.gradient_accumulation_steps}]\033[0m: ").strip()
            grad_acc_steps = int(grad_acc_str) if grad_acc_str else smart_config.gradient_accumulation_steps

            # LoRA parameters
            print("\n\033[96m━━━ LoRA Parameters ━━━\033[0m")
            lora_r_str = input(f"\033[96m▶\033[0m LoRA rank \033[2m[{smart_config.lora_r}]\033[0m: ").strip()
            lora_r = int(lora_r_str) if lora_r_str else smart_config.lora_r

            lora_alpha_str = input(f"\033[96m▶\033[0m LoRA alpha \033[2m[{smart_config.lora_alpha}]\033[0m: ").strip()
            lora_alpha = int(lora_alpha_str) if lora_alpha_str else smart_config.lora_alpha

            lora_dropout_str = input(f"\033[96m▶\033[0m LoRA dropout \033[2m[{smart_config.lora_dropout}]\033[0m: ").strip()
            lora_dropout = float(lora_dropout_str) if lora_dropout_str else smart_config.lora_dropout

            # Advanced options (optional)
            print("\n\033[96m━━━ Advanced Options (Optional) ━━━\033[0m")
            weight_decay_str = input(f"\033[96m▶\033[0m Weight decay \033[2m[{smart_config.weight_decay}]\033[0m: ").strip()
            weight_decay = float(weight_decay_str) if weight_decay_str else smart_config.weight_decay

            warmup_ratio_str = input(f"\033[96m▶\033[0m Warmup ratio \033[2m[{smart_config.warmup_ratio}]\033[0m: ").strip()
            warmup_ratio = float(warmup_ratio_str) if warmup_ratio_str else smart_config.warmup_ratio

            max_seq_len_str = input(f"\033[96m▶\033[0m Max sequence length \033[2m[{smart_config.max_sequence_length}]\033[0m: ").strip()
            max_seq_len = int(max_seq_len_str) if max_seq_len_str else smart_config.max_sequence_length

            # Create custom configuration
            config = TrainingConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_acc_steps,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                max_sequence_length=max_seq_len,
                task_type=smart_config.task_type,
                model_size_category=smart_config.model_size_category,
                estimated_parameters_b=smart_config.estimated_parameters_b,
                auto_configured=False,
                configuration_source="expert"
            )

            # Validate configuration
            valid, message = config.validate()
            if not valid:
                print(f"\033[31m✗\033[0m Configuration error: {message}")
                return None

            print("\033[32m✓\033[0m Expert configuration created")
            return config

        except ValueError as e:
            print(f"\033[31m✗\033[0m Invalid value entered: {e}")
            return None

    def _get_output_name(self, base_model: str) -> Optional[str]:
        """Get the output model name from user."""
        cli = self._require_cli()
        width = min(self.get_terminal_width() - 2, 70)

        default_name = f"{base_model}-finetuned"

        # Show output name dialog
        print()
        cli.print_box_header("Output Model", width)
        cli.print_empty_line(width)
        cli.print_box_line("  Enter name for fine-tuned model:", width)
        cli.print_box_line(f"  \033[2mDefault: {default_name}\033[0m", width)
        cli.print_empty_line(width)
        cli.print_box_footer(width)

        name = input(f"\n\033[96m▶\033[0m Model name \033[2m[{default_name}]\033[0m: ").strip()
        name = name if name else default_name

        # Check if name already exists
        existing_models = self._get_available_models()
        if any(model_name == name for model_name, _ in existing_models):
            print(f"\n\033[93m⚠\033[0m Model \033[93m{name}\033[0m already exists.")
            print("   Overwrite existing model? \033[2m[Y/N]\033[0m")
            choice = self._get_selection_input(default="n")
            if choice is None or choice.lower() != 'y':
                return None

        return name

    def _confirm_settings(self, base_model: str, dataset_path: Path,
                         config: TrainingConfig, output_name: str) -> bool:
        """Show summary and confirm settings."""
        cli = self._require_cli()
        width = min(self.get_terminal_width() - 2, 70)

        # Count dataset examples
        example_count = sum(1 for _ in open(dataset_path))

        # Estimate training time
        estimated_time = self._estimate_training_time(example_count, config)

        # Show summary dialog
        print()
        cli.print_box_header("Training Summary", width)
        cli.print_empty_line(width)

        cli.print_box_line("  \033[96mConfiguration:\033[0m", width)
        cli.print_empty_line(width)

        cli.print_box_line(f"  Base model:    \033[93m{base_model}\033[0m", width)
        cli.print_box_line(f"  Output model:  \033[93m{output_name}\033[0m", width)
        cli.print_box_line(f"  Dataset:       {dataset_path.name} \033[2m({example_count} examples)\033[0m", width)

        cli.print_empty_line(width)

        cli.print_box_line(f"  Model size:    \033[93m{config.model_size_category.title()}\033[0m ({config.estimated_parameters_b:.1f}B params)", width)
        cli.print_box_line(f"  Task type:     {config.task_type.title()}", width)
        cli.print_box_line(f"  Config source: {config.configuration_source.replace('_', ' ').title()}", width)

        cli.print_empty_line(width)

        cli.print_box_line(f"  Epochs:        {config.epochs}", width)
        cli.print_box_line(f"  Learning rate: {config.learning_rate:.1e}", width)
        cli.print_box_line(f"  LoRA rank:     {config.lora_r}", width)
        cli.print_box_line(f"  Batch size:    {config.batch_size} (x{config.gradient_accumulation_steps} acc.)", width)
        if config.quantization_bits:
            cli.print_box_line(f"  Quantization:  {config.quantization_bits}-bit", width)

        cli.print_empty_line(width)
        cli.print_box_line(f"  \033[2mEstimated time: {estimated_time}\033[0m", width)

        cli.print_empty_line(width)
        cli.print_box_separator(width)
        cli.print_empty_line(width)

        cli.print_box_line("  Start fine-tuning?", width)
        cli.print_empty_line(width)
        cli.print_box_line("  \033[93m[Y]\033[0m Yes, start training", width)
        cli.print_box_line("  \033[93m[N]\033[0m No, cancel", width)

        cli.print_empty_line(width)
        cli.print_box_footer(width)

        choice = self._get_selection_input(default="y")
        if choice is None:
            return False
        return choice.lower() in ['y', 'yes']

    def _run_training(self, base_model: str, dataset_path: Path,
                     config: TrainingConfig, output_name: str) -> bool:
        """Run the actual training."""
        print("\n\033[96m⚡\033[0m Starting fine-tuning...")

        try:
            # Hard requirement: MLX must be available for fine-tuning.
            if not MLXLoRATrainer.is_available():
                print("\n\033[31m✗\033[0m Fine-tuning requires MLX/Metal, but MLX is not available in this environment.")
                return False
            # Use MLXLoRATrainer for proper LoRA implementation
            trainer = MLXLoRATrainer(self.model_manager, self.config)
            self.trainer = trainer

            # Progress tracking
            start_time = time.time()
            last_update = start_time

            def update_progress(epoch, step, loss):
                nonlocal last_update
                current_time = time.time()

                # Update every 0.5 seconds
                if current_time - last_update > 0.5:
                    progress = ((epoch * 100) + min(step, 99)) / (config.epochs * 100)

                    # Create progress bar
                    bar_width = 30
                    filled = int(bar_width * progress)
                    bar = "█" * filled + "░" * (bar_width - filled)

                    # Print progress
                    sys.stdout.write(f"\r   {bar} {progress*100:.0f}% | Epoch {epoch+1}/{config.epochs} | Loss: {loss:.4f}")
                    sys.stdout.flush()
                    last_update = current_time

            # Run training
            success = trainer.train(
                base_model_name=base_model,
                dataset_path=dataset_path,
                output_name=output_name,
                training_config=config,
                progress_callback=update_progress
            )

            print()  # New line after progress

            if success:
                print("\n\033[32m✓\033[0m Fine-tuning completed!")

                # Show where the model was saved
                mlx_path = Path.home() / ".cortex" / "mlx_models" / output_name
                if mlx_path.exists():
                    print(f"\n\033[96m📍\033[0m Model saved to: \033[93m{mlx_path}\033[0m")
                    print("\n\033[96m💡\033[0m To load your fine-tuned model:")
                    print(f"   \033[93m/model {mlx_path}\033[0m")

                    # Check if adapter weights exist
                    adapter_file = mlx_path / "adapter.safetensors"
                    if adapter_file.exists():
                        size_mb = adapter_file.stat().st_size / (1024 * 1024)
                        print(f"\n\033[2m   LoRA adapter size: {size_mb:.1f} MB\033[0m")
                        print(f"\033[2m   Base model: {base_model}\033[0m")

                return True
            else:
                print("\n\033[31m✗\033[0m Fine-tuning failed")
                return False

        except Exception as e:
            logger.error(f"Training failed: {e}")
            print(f"\n\n\033[31m✗\033[0m Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_available_models(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get list of available models."""
        models = []

        # Get models from model manager
        discovered = self.model_manager.discover_available_models()

        for model_info in discovered:
            try:
                name = model_info.get('name', 'Unknown')
                info = {
                    'path': Path(model_info.get('path', '')),
                    'format': model_info.get('format', 'Unknown'),
                    'size_gb': model_info.get('size_gb', 0.0)
                }
                models.append((name, info))
            except Exception as e:
                logger.debug(f"Error processing model info: {e}")
                continue

        return sorted(models, key=lambda x: x[0])

    def _get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model."""
        models = self._get_available_models()
        for name, info in models:
            if name == model_name:
                return info
        return None

    def _estimate_training_time(self, example_count: int, config: TrainingConfig) -> str:
        """Estimate training time based on dataset size, epochs, and model characteristics."""
        # Base time estimation adjusted for model size and batch settings
        base_seconds_per_example = {
            "tiny": 0.1,     # Very fast for small models
            "small": 0.3,    # Fast
            "medium": 0.7,   # Standard
            "large": 1.5,    # Slower for large models
            "xlarge": 3.0    # Much slower
        }.get(config.model_size_category, 0.7)

        # Adjust for gradient accumulation (more accumulation = fewer actual updates)
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        batch_factor = max(0.5, 1.0 / (effective_batch_size ** 0.5))  # Larger batches are more efficient

        # Adjust for quantization (if enabled, training is faster)
        quant_factor = 0.7 if config.quantization_bits else 1.0

        # Calculate total time
        adjusted_time_per_example = base_seconds_per_example * batch_factor * quant_factor
        total_seconds = example_count * config.epochs * adjusted_time_per_example

        if total_seconds < 60:
            return f"~{int(total_seconds)} seconds"
        elif total_seconds < 3600:
            return f"~{int(total_seconds / 60)} minutes"
        else:
            return f"~{total_seconds / 3600:.1f} hours"
