"""CLI interface for Cortex"""

import logging
import readline
import shutil
import signal
import sys
import termios
from datetime import datetime
from logging import Handler
from pathlib import Path
from typing import Callable, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cortex.cloud import CloudCredentialStore, CloudModelCatalog, CloudRouter
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider
from cortex.config import Config
from cortex.conversation_manager import ConversationManager
from cortex.fine_tuning import FineTuneWizard
from cortex.gpu_validator import GPUValidator
from cortex.inference_engine import InferenceEngine
from cortex.model_downloader import ModelDownloader
from cortex.model_manager import ModelManager
from cortex.template_registry import TemplateRegistry
from cortex.tooling.orchestrator import ToolingOrchestrator
from cortex.tooling.permissions import PermissionDecision, PermissionRequest
from cortex.ui import (
    box_rendering,
    generation,
    help_ui,
    model_download_ui,
    model_login_ui,
    model_manage_ui,
    startup,
    status_ui,
)
from cortex.ui import finetune as finetune_ui
from cortex.ui import template as template_ui
from cortex.ui.cli_commands import CommandHandlers
from cortex.ui.cli_commands import handle_command as dispatch_command
from cortex.ui.cli_prompt import format_prompt_with_chat_template
from cortex.ui.input_box import prompt_input_box

logger = logging.getLogger(__name__)


class CortexCLI:
    """Command-line interface for Cortex"""

    def __init__(
        self,
        config: Config,
        gpu_validator: GPUValidator,
        model_manager: ModelManager,
        inference_engine: InferenceEngine,
        conversation_manager: ConversationManager
    ):
        self.config = config
        self.gpu_validator = gpu_validator
        self.model_manager = model_manager
        self.inference_engine = inference_engine
        self.conversation_manager = conversation_manager
        self.model_downloader = ModelDownloader(config.model.model_path)

        # Initialize template registry with console for interactive setup
        self.console = Console()
        self.template_registry = TemplateRegistry(console=self.console)

        # Initialize fine-tuning wizard
        self.fine_tune_wizard = FineTuneWizard(model_manager, config)

        # Cloud model support
        self.cloud_credentials = CloudCredentialStore()
        self.cloud_catalog = CloudModelCatalog()
        self.cloud_router = CloudRouter(config, self.cloud_credentials)
        self.tooling_orchestrator = ToolingOrchestrator(cli=self)
        self.active_model_target = ActiveModelTarget.local()
        self.bottom_gutter_lines = 3
        self.ui_modal_prompt_active = False
        self.on_modal_prompt_start: Optional[Callable[[], None]] = None
        self.on_modal_prompt_end: Optional[Callable[[], None]] = None

        self.running = True
        self.generating = False

        # Set up readline for better input handling (fallback)
        self._setup_readline()

        # Set up signal handlers for clean exit on hard-stop signals
        for sig in (
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGQUIT", None),
            getattr(signal, "SIGTSTP", None),
        ):
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_interrupt)
                if hasattr(signal, "siginterrupt"):
                    signal.siginterrupt(sig, True)
            except Exception:
                continue
        # Ctrl+Z is treated as a clean exit for consistent hard-stop behavior

    def _setup_readline(self):
        """Set up readline for better command-line editing."""
        # Enable tab completion
        readline.parse_and_bind("tab: complete")

        # Set up command history
        histfile = Path.home() / ".cortex_history"
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass

        # Save history on exit
        import atexit
        atexit.register(readline.write_history_file, histfile)

        # Set up auto-completion
        readline.set_completer(self._completer)

    def _read_line_with_escape_cancel(self) -> tuple[str, bool]:
        """Read one line and cancel immediately when ESC is pressed."""
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
                    # ESC pressed: cancel prompt immediately.
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "", True

                if char in ("\x7f", "\x08"):  # backspace
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

    def get_input_with_escape(self, prompt: str = "Select an option or Esc to cancel") -> Optional[str]:
        """Get user input with ESC key support for cancellation.

        Returns:
            User input string, or None if cancelled (ESC, Ctrl+C, or '0')
        """
        # Get input with ESC to cancel
        self.console.print()
        self.reserve_bottom_gutter()
        self.console.print(f"[cyan]▶[/cyan] {prompt}: ", end="")
        self.console.file.flush()
        if sys.stdin.isatty():
            try:
                user_input, cancelled = self._read_line_with_escape_cancel()
            except (EOFError, KeyboardInterrupt):
                return None
            user_input = user_input.strip()
            if cancelled:
                return None
        else:
            try:
                user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                return None

        if user_input == "":
            return ""

        if user_input == "\x1b":
            return None

        # Check for cancel input
        if user_input == '0':
            return None

        return user_input

    def _completer(self, text, state):
        """Auto-complete commands."""
        commands = ['/help', '/status', '/download', '/model',
                   '/clear', '/save', '/gpu', '/benchmark', '/template', '/finetune', '/login', '/quit']


        # Filter matching commands
        matches = [cmd for cmd in commands if cmd.startswith(text)]

        if state < len(matches):
            return matches[state]
        return None

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C interruption."""
        self._request_shutdown()
        raise KeyboardInterrupt

    def _request_shutdown(self) -> None:
        """Request a clean shutdown across the CLI."""
        if self.generating:
            self.inference_engine.cancel_generation()
        self.generating = False
        self.running = False

    def reserve_bottom_gutter(self, lines: Optional[int] = None) -> None:
        """Reserve a blank gutter at the bottom of the terminal."""
        gutter = self.bottom_gutter_lines if lines is None else max(0, int(lines))
        if gutter <= 0:
            return

        sys.stdout.write("\n" * gutter)
        sys.stdout.write(f"\033[{gutter}A")
        sys.stdout.flush()

    def get_terminal_width(self) -> int:
        """Get terminal width."""
        return shutil.get_terminal_size(fallback=(80, 24)).columns

    def get_terminal_height(self) -> int:
        """Get terminal height."""
        return shutil.get_terminal_size(fallback=(80, 24)).lines

    def get_visible_length(self, text: str) -> int:
        """Get visible length of text, ignoring ANSI escape codes and accounting for wide characters."""
        return box_rendering.get_visible_length(text)

    def print_box_line(self, content: str, width: int, align: str = 'left'):
        """Print a single line in a box with proper padding."""
        box_rendering.print_box_line(content, width, align=align)

    def print_box_header(self, title: str, width: int):
        """Print a box header with title."""
        box_rendering.print_box_header(title, width)

    def print_box_footer(self, width: int):
        """Print a box footer."""
        box_rendering.print_box_footer(width)

    def print_box_separator(self, width: int):
        """Print a separator line inside a box."""
        box_rendering.print_box_separator(width)

    def print_empty_line(self, width: int):
        """Print an empty line inside a box."""
        box_rendering.print_empty_line(width)

    def create_box(self, lines: List[str], width: Optional[int] = None) -> str:
        """Create a box with Unicode borders."""
        return box_rendering.create_box(
            lines,
            width=width,
            terminal_width=self.get_terminal_width(),
        )

    def print_welcome(self):
        """Print welcome message"""
        startup.print_welcome(cli=self)

    def load_default_model(self):
        """Load the last used model or default model if configured."""
        startup.load_default_model(cli=self)

    def get_active_model_label(self) -> str:
        """Return active target label for UI and status displays."""
        if self.active_model_target.backend == "cloud" and self.active_model_target.cloud_model:
            return self.active_model_target.cloud_model.selector

        if self.model_manager.current_model:
            return self.model_manager.current_model

        return self.active_model_target.label

    def set_active_local_model(self, model_name: Optional[str] = None, persist: bool = True) -> None:
        """Set local backend as active target."""
        selected_model = model_name or self.model_manager.current_model
        self.active_model_target = ActiveModelTarget.local(model_name=selected_model)
        if persist:
            self.config.set_state_value("last_used_backend", "local")

    def set_active_cloud_model(
        self,
        provider: Union[str, CloudProvider],
        model_id: str,
        persist: bool = True,
    ) -> tuple[bool, str]:
        """Set cloud backend as active target."""
        try:
            provider_enum = CloudProvider.from_value(provider)
        except ValueError as exc:
            return False, str(exc)

        clean_model_id = model_id.strip()
        if not clean_model_id:
            return False, "Cloud model ID cannot be empty."

        model_ref = CloudModelRef(provider=provider_enum, model_id=clean_model_id)
        self.active_model_target = ActiveModelTarget.cloud(model_ref)

        if persist:
            self.config.set_state_value("last_used_backend", "cloud")
            self.config.set_state_value("last_used_cloud_provider", provider_enum.value)
            self.config.set_state_value("last_used_cloud_model", clean_model_id)

        return True, f"Active cloud model set to {model_ref.selector}"

    def handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns False to exit."""
        handlers = CommandHandlers(
            show_help=self.show_help,
            manage_models=self.manage_models,
            download_model=self.download_model,
            clear_conversation=self.clear_conversation,
            save_conversation=self.save_conversation,
            show_status=self.show_status,
            show_gpu_status=self.show_gpu_status,
            run_benchmark=self.run_benchmark,
            manage_template=self.manage_template,
            run_finetune=self.run_finetune,
            login=self.login,
            show_shortcuts=self.show_shortcuts,
            unknown_command=self.show_unknown_command,
        )
        return dispatch_command(command, handlers)

    def show_unknown_command(self, command: str) -> None:
        """Show unknown-command help."""
        self.console.print(f"[red]Unknown command:[/red] {command}")
        self.console.print("[dim]Type /help for available commands[/dim]")

    def show_shortcuts(self):
        """Show keyboard shortcuts."""
        help_ui.show_shortcuts(
            terminal_width=self.get_terminal_width(),
            box=self,
        )

    def show_help(self):
        """Show available commands."""
        help_ui.show_help(
            terminal_width=self.get_terminal_width(),
            box=self,
        )

    def download_model(self, args: str = ""):
        """Download a model from HuggingFace."""
        model_download_ui.download_model(cli=self, args=args)

    def login(self, args: str = ""):
        """Run login flow for cloud providers or HuggingFace."""
        model_login_ui.login(cli=self, args=args)

    def manage_models(self, args: str = ""):
        """Interactive model manager - simplified for better UX.
        If args provided, tries to load that model directly."""
        model_manage_ui.manage_models(cli=self, args=args)

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_manager.new_conversation()
        self.console.print("[green]✓[/green] Conversation cleared.")

    def save_conversation(self):
        """Save current conversation."""
        try:
            export_data = self.conversation_manager.export_conversation(format="json")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.config.conversation.save_directory / f"conversation_{timestamp}.json"

            with open(filename, 'w') as f:
                f.write(export_data)

            self.console.print(f"[green]✓[/green] Conversation saved to {filename}")
        except Exception as e:
            self.console.print(f"[red]✗[/red] Failed to save: {str(e)}")

    def show_status(self):
        """Show current setup status."""
        status_ui.show_status(cli=self)

    def show_gpu_status(self):
        """Show GPU status."""
        status_ui.show_gpu_status(cli=self)

    def run_benchmark(self):
        """Run performance benchmark."""
        status_ui.run_benchmark(cli=self)

    def manage_template(self, args: str = ""):
        """Manage template configuration for the current model."""
        template_ui.manage_template(cli=self, args=args)

    def run_finetune(self):
        """Run the interactive fine-tuning wizard."""
        finetune_ui.run_finetune(cli=self)

    def prompt_tool_permission(self, request: PermissionRequest) -> PermissionDecision:
        """Prompt user for tool permission decision."""
        width = min(self.get_terminal_width() - 2, 84)
        patterns = ", ".join(request.patterns[:3])
        if len(request.patterns) > 3:
            patterns += f", +{len(request.patterns) - 3} more"

        self.ui_modal_prompt_active = True
        if callable(self.on_modal_prompt_start):
            self.on_modal_prompt_start()
        try:
            # Clear any transient status line before drawing the permission modal.
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()

            body = Text()
            body.append("Tool permission: ")
            body.append(request.permission, style="yellow")
            body.append("\n")
            body.append("Target: ")
            body.append(patterns, style="dim")
            body.append("\n\n")
            body.append("[1] ", style="yellow")
            body.append("Allow once\n")
            body.append("[2] ", style="yellow")
            body.append("Allow always\n")
            body.append("[3] ", style="yellow")
            body.append("Reject")

            self.console.print()
            self.console.print(
                Panel(
                    body,
                    title="Tool Permission",
                    border_style="cyan",
                    width=width,
                    padding=(1, 2),
                )
            )
            self.console.print("[cyan]▶[/cyan] Select an option or Esc to cancel: ", end="")
            self.console.file.flush()

            if sys.stdin.isatty():
                try:
                    choice, cancelled = self._read_line_with_escape_cancel()
                except (EOFError, KeyboardInterrupt):
                    return PermissionDecision.REJECT
                if cancelled:
                    return PermissionDecision.REJECT
                choice = choice.strip()
            else:
                try:
                    choice = input().strip()
                except (EOFError, KeyboardInterrupt):
                    return PermissionDecision.REJECT

            if choice == "1":
                return PermissionDecision.ALLOW_ONCE
            if choice == "2":
                return PermissionDecision.ALLOW_ALWAYS
            return PermissionDecision.REJECT
        finally:
            self.ui_modal_prompt_active = False
            if callable(self.on_modal_prompt_end):
                self.on_modal_prompt_end()

    def generate_response(self, user_input: str):
        """Generate and stream response from the model."""
        generation.generate_response(cli=self, user_input=user_input)

    def get_user_input(self) -> str:
        """Get user input with standard prompt."""
        try:
            if sys.stdin.isatty() and sys.stdout.isatty():
                print()
                self.reserve_bottom_gutter()
            user_input = input("> ")
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            raise

    def _format_prompt_with_chat_template(self, user_input: str, include_user: bool = True) -> str:
        """Format the prompt with appropriate chat template for the model."""
        return format_prompt_with_chat_template(
            conversation_manager=self.conversation_manager,
            model_manager=self.model_manager,
            template_registry=self.template_registry,
            user_input=user_input,
            include_user=include_user,
            logger=logger,
        )

    def get_input_from_box(self) -> str:
        """Get user input from a styled input box."""
        return prompt_input_box(
            terminal_width=self.get_terminal_width(),
            current_model_path=self.get_active_model_label(),
            bottom_gutter_lines=self.bottom_gutter_lines,
        )

    def run(self):
        """Main REPL loop."""
        self.print_welcome()
        self.load_default_model()

        # Start new conversation
        self.conversation_manager.new_conversation()
        use_styled_input_box = bool(sys.stdin.isatty() and sys.stdout.isatty())

        while self.running:
            try:
                # Use styled input only for interactive TTY sessions.
                user_input = self.get_input_from_box() if use_styled_input_box else self.get_user_input()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit']:
                    break

                # Handle shortcuts
                if user_input == '?':
                    self.show_shortcuts()
                    # Don't increment message count for shortcuts
                    continue

                # Handle slash commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    # Don't increment message count for commands
                    continue

                # Generate response
                self.generate_response(user_input)

            except EOFError:
                self._request_shutdown()
                break
            except KeyboardInterrupt:
                # Clean exit on Ctrl+C, same as /quit
                self._request_shutdown()
                break
            except Exception as e:
                self.console.print(f"[red]✗ Error:[/red] {str(e)}")

        self.console.print("\n[dim]Goodbye![/dim]")


def main():
    """Main entry point for CLI."""
    # Initialize components
    config = Config()
    configure_logging(config)
    gpu_validator = GPUValidator()

    # Validate GPU
    is_valid, gpu_info, errors = gpu_validator.validate()
    if not is_valid:
        print("GPU validation failed. Cortex requires Apple Silicon with Metal support.")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Initialize managers
    model_manager = ModelManager(config, gpu_validator)
    inference_engine = InferenceEngine(config, model_manager)
    conversation_manager = ConversationManager(config)

    # Create and run CLI
    cli = CortexCLI(
        config=config,
        gpu_validator=gpu_validator,
        model_manager=model_manager,
        inference_engine=inference_engine,
        conversation_manager=conversation_manager
    )

    cli.run()

def configure_logging(config: Config) -> None:
    """Configure application logging outputs from config."""
    log_level_name = str(getattr(config.logging, "log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_file = Path(getattr(config.logging, "log_file", Path.home() / ".cortex" / "cortex.log")).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler: Handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    handlers: List[Handler] = [file_handler]
    if getattr(config.developer, "debug_mode", False):
        stderr_handler: Handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(log_level)
        stderr_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handlers.append(stderr_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    logging.getLogger(__name__).info("Logging initialized level=%s file=%s", log_level_name, log_file)


if __name__ == "__main__":
    main()
