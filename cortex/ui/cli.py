"""CLI interface for Cortex"""

import sys
import signal
import shutil
import readline
import logging
from typing import Optional, List
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)

from cortex.config import Config
from cortex.gpu_validator import GPUValidator
from cortex.model_manager import ModelManager
from cortex.inference_engine import InferenceEngine
from cortex.conversation_manager import ConversationManager, MessageRole
from cortex.model_downloader import ModelDownloader
from cortex.template_registry import TemplateRegistry
from cortex.fine_tuning import FineTuneWizard
from cortex.tools import ToolRunner
from cortex.ui import box_rendering
from cortex.ui.cli_commands import CommandHandlers, handle_command as dispatch_command
from cortex.ui.cli_prompt import format_prompt_with_chat_template
from cortex.ui import help_ui
from cortex.ui import model_download_ui
from cortex.ui import model_login_ui
from cortex.ui import model_manage_ui
from cortex.ui import status_ui
from cortex.ui.input_box import prompt_input_box
from cortex.ui import startup
from cortex.ui import generation
from cortex.ui import template as template_ui
from cortex.ui import finetune as finetune_ui


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
        from rich.console import Console
        self.console = Console()
        self.template_registry = TemplateRegistry(console=self.console)
        
        # Initialize fine-tuning wizard
        self.fine_tune_wizard = FineTuneWizard(model_manager, config)

        # Tooling support (always enabled)
        self.tool_runner = ToolRunner(Path.cwd())
        self.tool_runner.set_confirm_callback(self._confirm_tool_change)
        self.max_tool_iterations = 4
        
        
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
    
    def get_input_with_escape(self, prompt: str = "Select option") -> Optional[str]:
        """Get user input with ESC key support for cancellation.
        
        Returns:
            User input string, or None if cancelled (ESC, Ctrl+C, or '0')
        """
        # Get input with ESC to cancel
        print()
        print(f"\033[96m▶\033[0m {prompt}: ", end='')
        user_input = input().strip()
        
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

    def _confirm_tool_change(self, prompt: str) -> bool:
        """Prompt user to approve a tool-driven change."""
        print("\n" + prompt)
        response = input("Apply change? [y/N]: ").strip().lower()
        return response in {"y", "yes"}

    def _ensure_tool_instructions(self) -> None:
        """Inject tool instructions into the conversation once."""
        conversation = self.conversation_manager.get_current_conversation()
        if conversation is None:
            conversation = self.conversation_manager.new_conversation()
        marker = "[CORTEX_TOOL_INSTRUCTIONS v5]"
        for message in conversation.messages:
            if message.role == MessageRole.SYSTEM and marker in message.content:
                return
        self.conversation_manager.add_message(MessageRole.SYSTEM, self.tool_runner.tool_instructions())

    
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
            hf_login=self.hf_login,
            show_shortcuts=self.show_shortcuts,
        )
        return dispatch_command(command, handlers)
    
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
    
    def hf_login(self):
        """Login to HuggingFace for accessing gated models."""
        model_login_ui.hf_login(cli=self)
    
    def manage_models(self, args: str = ""):
        """Interactive model manager - simplified for better UX.
        If args provided, tries to load that model directly."""
        model_manage_ui.manage_models(cli=self, args=args)
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_manager.new_conversation()
        print("\033[32m✓\033[0m Conversation cleared.")
    
    def save_conversation(self):
        """Save current conversation."""
        try:
            export_data = self.conversation_manager.export_conversation(format="json")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.config.conversation.save_directory / f"conversation_{timestamp}.json"
            
            with open(filename, 'w') as f:
                f.write(export_data)
            
            print(f"\033[32m✓\033[0m Conversation saved to {filename}")
        except Exception as e:
            print(f"\033[31m✗\033[0m Failed to save: {str(e)}", file=sys.stderr)
    
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
    
    def generate_response(self, user_input: str):
        """Generate and stream response from the model."""
        generation.generate_response(cli=self, user_input=user_input)

    def get_user_input(self) -> str:
        """Get user input with standard prompt."""
        try:
            print()
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
            console=self.console,
            terminal_width=self.get_terminal_width(),
            current_model_path=self.model_manager.current_model,
        )
    
    def run(self):
        """Main REPL loop."""
        self.print_welcome()
        self.load_default_model()
        
        # Start new conversation
        self.conversation_manager.new_conversation()
        
        while self.running:
            try:
                # Get input from styled box
                user_input = self.get_input_from_box()
                
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
                print(f"\033[31m✗ Error:\033[0m {str(e)}", file=sys.stderr)
        
        print("\n\033[2mGoodbye!\033[0m")


def main():
    """Main entry point for CLI."""
    # Initialize components
    config = Config()
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


if __name__ == "__main__":
    main()
