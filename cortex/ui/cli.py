"""CLI interface for Cortex with Claude Code-style UI."""

import os
import sys
import signal
import shutil
import readline
import time
import threading
import logging
import termios
import tty
import getpass
from typing import Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from textwrap import wrap

from rich.live import Live
from rich.style import Style


logger = logging.getLogger(__name__)

from cortex.config import Config
from cortex.gpu_validator import GPUValidator
from cortex.model_manager import ModelManager
from cortex.inference_engine import InferenceEngine, GenerationRequest
from cortex.conversation_manager import ConversationManager, MessageRole
from cortex.model_downloader import ModelDownloader
from cortex.template_registry import TemplateRegistry
from cortex.fine_tuning import FineTuneWizard
from cortex.ui.markdown_render import ThinkMarkdown, PrefixedRenderable, render_plain_with_think


class CortexCLI:
    """Command-line interface for Cortex with Claude Code-style UI."""
    
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
        
        
        self.running = True
        self.generating = False
        
        # Set up readline for better input handling (fallback)
        self._setup_readline()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        # SIGTSTP (Ctrl+Z) - let it suspend normally, no special handling needed
        # The default behavior is fine for suspension
    
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
        if self.generating:
            print("\n\nGeneration cancelled.", file=sys.stderr)
            self.inference_engine.cancel_generation()
            self.generating = False
        else:
            # Set running to False to exit the main loop gracefully
            self.running = False
            # Don't call sys.exit() here - let the main loop exit naturally
            # This prevents traceback from the parent process
            print("\n", file=sys.stderr)  # Just add a newline for cleaner output
    
    
    def get_terminal_width(self) -> int:
        """Get terminal width."""
        return shutil.get_terminal_size(fallback=(80, 24)).columns
    
    def get_terminal_height(self) -> int:
        """Get terminal height."""
        return shutil.get_terminal_size(fallback=(80, 24)).lines
    
    def get_visible_length(self, text: str) -> int:
        """Get visible length of text, ignoring ANSI escape codes and accounting for wide characters."""
        import re
        import unicodedata
        
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        visible_text = ansi_escape.sub('', text)
        
        # Calculate display width accounting for wide/ambiguous characters
        display_width = 0
        for char in visible_text:
            width = unicodedata.east_asian_width(char)
            if width in ('W', 'F'):  # Wide or Fullwidth - always 2 columns
                display_width += 2
            elif width == 'A' and char in '●○':  # Ambiguous - might be 2 in some terminals
                # For now, treat these as single-width since most Western terminals do
                # But if alignment issues appear with these characters, change to += 2
                display_width += 1
            else:
                display_width += 1
        
        return display_width
    
    def print_box_line(self, content: str, width: int, align: str = 'left'):
        """Print a single line in a box with proper padding."""
        visible_len = self.get_visible_length(content)
        padding = width - visible_len - 2  # -2 for the borders
        
        if align == 'center':
            left_pad = padding // 2
            right_pad = padding - left_pad
            print(f"│{' ' * left_pad}{content}{' ' * right_pad}│")
        else:  # left align
            print(f"│{content}{' ' * padding}│")
    
    def print_box_header(self, title: str, width: int):
        """Print a box header with title."""
        if title:
            title_with_color = f" \033[96m{title}\033[0m "
            visible_len = self.get_visible_length(title_with_color)
            padding = width - visible_len - 3  # -3 for "╭─" and "╮"
            print(f"╭─{title_with_color}" + "─" * padding + "╮")
        else:
            print("╭" + "─" * (width - 2) + "╮")
    
    def print_box_footer(self, width: int):
        """Print a box footer."""
        print("╰" + "─" * (width - 2) + "╯")
    
    def print_box_separator(self, width: int):
        """Print a separator line inside a box."""
        # Width already includes space for borders, so we need exact width-2 for the line
        print("├" + "─" * (width - 2) + "┤")
    
    def print_empty_line(self, width: int):
        """Print an empty line inside a box."""
        print("│" + " " * (width - 2) + "│")
    
    def create_box(self, lines: List[str], width: Optional[int] = None) -> str:
        """Create a box with Unicode borders."""
        if width is None:
            width = min(self.get_terminal_width() - 2, 80)
        
        # Box drawing characters
        top_left = "╭"
        top_right = "╮"
        bottom_left = "╰"
        bottom_right = "╯"
        horizontal = "─"
        vertical = "│"
        
        # Calculate inner width
        inner_width = width - 4  # Account for borders and padding
        
        # Build box
        result = []
        result.append(top_left + horizontal * (width - 2) + top_right)
        
        for line in lines:
            # Calculate visible length to handle ANSI codes
            visible_len = self.get_visible_length(line)
            # Calculate padding needed
            padding_needed = inner_width - visible_len
            # Create padded line with correct spacing
            padded = f" {line}{' ' * padding_needed} "
            result.append(vertical + padded + vertical)
        
        result.append(bottom_left + horizontal * (width - 2) + bottom_right)
        
        return "\n".join(result)
    
    def print_welcome(self):
        """Print welcome message in Claude Code style."""
        width = min(self.get_terminal_width() - 2, 70)
        
        # Get current working directory
        cwd = os.getcwd()
        
        # Welcome box content
        welcome_lines = [
            "\033[96m✻ Welcome to Cortex!\033[0m",
            "",
            "  \033[93m/help\033[0m for help, \033[93m/status\033[0m for your current setup",
            "",
            f"  \033[2mcwd:\033[0m {cwd}"
        ]
        
        print(self.create_box(welcome_lines, width))
        print()
        
        # Show last used model if configured
        if self.config.model.last_used_model:
            # Clean up the model name for display
            display_name = self.config.model.last_used_model
            if display_name.startswith("_Users_") and ("_4bit" in display_name or "_5bit" in display_name or "_8bit" in display_name):
                # Extract clean model name from cached path
                parts = display_name.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "").split("_")
                if len(parts) > 3:
                    display_name = parts[-1]  # Get just the model name
            print(f" \033[2m※ Last model:\033[0m \033[93m{display_name}\033[0m")
        
        print(" \033[2m※ Tip: Use\033[0m \033[93m/download\033[0m \033[2mto get models from HuggingFace\033[0m")
        
        # Show input mode info
        print(" \033[2m※ Basic input mode (install prompt-toolkit for enhanced features)\033[0m")
        print()
    
    def load_default_model(self):
        """Load the last used model or default model if configured."""
        # Try to load last used model first
        model_to_load = self.config.model.last_used_model or self.config.model.default_model
        
        if not model_to_load:
            print("\n \033[96m⚡\033[0m No model loaded. Use \033[93m/model\033[0m to select a model.")
            return
        
        # Check if this is a cached MLX model (contains _4bit, _5bit, etc.)
        if "_4bit" in model_to_load or "_5bit" in model_to_load or "_8bit" in model_to_load:
            # Extract clean model name from cached path
            clean_name = model_to_load
            if clean_name.startswith("_Users_"):
                # Extract the actual model name from the path
                parts = clean_name.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "").split("_")
                if len(parts) > 3:
                    clean_name = parts[-1]  # Get just the model name
            
            # This is a cached MLX model, try to load it directly
            print(f"\n \033[96m⚡\033[0m Loading: \033[93m{clean_name}\033[0m \033[2m(MLX optimized)\033[0m...")
            success, message = self.model_manager.load_model(model_to_load)
            
            if success:
                model_info = self.model_manager.get_current_model()
                if model_info:
                    # Show clean model info
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
                    print(f"   \033[2m• Optimizations: AMX acceleration, operation fusion\033[0m")
                    print(f"   \033[2m• Format: MLX (Apple Silicon optimized)\033[0m")
                    
                    # Show template information
                    tokenizer = self.model_manager.tokenizers.get(model_info.name)
                    profile = self.template_registry.setup_model(
                        model_info.name, 
                        tokenizer=tokenizer,
                        interactive=False
                    )
                    if profile:
                        template_name = profile.config.name
                        print(f"   \033[2m• Template: {template_name}\033[0m")
            else:
                # Try to extract original model name and reload
                base_name = model_to_load.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "")
                if base_name.startswith("_Users_"):
                    # Extract original path
                    original_path = "/" + base_name[1:].replace("_", "/")
                    if Path(original_path).exists():
                        print(f" \033[2m※ Cached model not found, reconverting from original...\033[0m")
                        success, message = self.model_manager.load_model(original_path)
                        if success:
                            model_info = self.model_manager.get_current_model()
                            if model_info:
                                print(f" \033[32m✓\033[0m Model loaded: \033[93m{model_info.name}\033[0m \033[2m({model_info.size_gb:.1f}GB, {model_info.format.value})\033[0m")
                            return
                
                print(f"\n \033[31m⚠\033[0m Previously used model not found: \033[93m{model_to_load}\033[0m")
                print(" Use \033[93m/model\033[0m to select a different model or \033[93m/download\033[0m to get new models.")
            return
        
        # Try to find the model
        model_path = None
        
        # First, check if it's a direct path that exists
        potential_path = Path(model_to_load).expanduser()
        if potential_path.exists():
            model_path = potential_path
        else:
            # Try in the models directory
            potential_path = self.config.model.model_path / model_to_load
            if potential_path.exists():
                model_path = potential_path
            else:
                # Search for the model in available models
                available = self.model_manager.discover_available_models()
                for model in available:
                    if model['name'] == model_to_load:
                        model_path = Path(model['path'])
                        break
        
        if not model_path:
            print(f"\n \033[31m⚠\033[0m Previously used model not found: \033[93m{model_to_load}\033[0m")
            print(" Use \033[93m/model\033[0m to select a different model or \033[93m/download\033[0m to get new models.")
            return
        
        print(f"\n \033[96m⚡\033[0m Loading: \033[93m{model_to_load}\033[0m...")
        success, message = self.model_manager.load_model(str(model_path))
        
        if success:
            model_info = self.model_manager.get_current_model()
            if model_info:
                print(f" \033[32m✓\033[0m Model loaded: \033[93m{model_info.name}\033[0m \033[2m({model_info.size_gb:.1f}GB, {model_info.format.value})\033[0m")
                
                # Show template information
                tokenizer = self.model_manager.tokenizers.get(model_info.name)
                profile = self.template_registry.setup_model(
                    model_info.name, 
                    tokenizer=tokenizer,
                    interactive=False
                )
                if profile:
                    template_name = profile.config.name
                    print(f"   \033[2m• Template: {template_name}\033[0m")
        else:
            print(f" \033[31m✗\033[0m Failed to load model: {message}", file=sys.stderr)
            print(" Use \033[93m/model\033[0m to select a different model.")
    
    def handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns False to exit."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/help":
            self.show_help()
        elif cmd == "/model":
            self.manage_models(args)
        elif cmd == "/download":
            self.download_model(args)
        elif cmd == "/clear":
            self.clear_conversation()
        elif cmd == "/save":
            self.save_conversation()
        elif cmd == "/status":
            self.show_status()
        elif cmd == "/gpu":
            self.show_gpu_status()
        elif cmd == "/benchmark":
            self.run_benchmark()
        elif cmd == "/template":
            self.manage_template(args)
        elif cmd == "/finetune":
            self.run_finetune()
        elif cmd == "/login":
            self.hf_login()
        elif cmd in ["/quit", "/exit"]:
            return False
        elif cmd == "?":
            self.show_shortcuts()
        else:
            print(f"\033[31mUnknown command: {cmd}\033[0m")
            print("\033[2mType /help for available commands\033[0m")
        
        return True
    
    def show_shortcuts(self):
        """Show keyboard shortcuts."""
        width = min(self.get_terminal_width() - 2, 70)
        
        print()
        self.print_box_header("Keyboard Shortcuts", width)
        self.print_empty_line(width)
        
        shortcuts = [
            ("Ctrl+C", "Cancel current generation"),
            ("Ctrl+D", "Exit Cortex"),
            ("Tab", "Auto-complete commands"),
            ("/help", "Show all commands"),
            ("?", "Show this help")
        ]
        
        for key, desc in shortcuts:
            # Color the key/command in yellow
            colored_key = f"\033[93m{key}\033[0m"
            # Calculate padding
            key_width = len(key)
            padding = " " * (12 - key_width)  # Align descriptions at column 14
            line = f"  {colored_key}{padding}{desc}"
            self.print_box_line(line, width)
        
        self.print_empty_line(width)
        self.print_box_footer(width)
    
    def show_help(self):
        """Show available commands."""
        width = min(self.get_terminal_width() - 2, 70)
        
        print()
        self.print_box_header("Available Commands", width)
        self.print_empty_line(width)
        
        commands = [
            ("/help", "Show this help message"),
            ("/status", "Show current setup and GPU info"),
            ("/download", "Download a model from HuggingFace"),
            ("/model", "Manage models (load/delete/info)"),
            ("/finetune", "Fine-tune a model interactively"),
            ("/clear", "Clear conversation history"),
            ("/save", "Save current conversation"),
            ("/template", "Manage chat templates"),
            ("/gpu", "Show GPU status"),
            ("/benchmark", "Run performance benchmark"),
            ("/login", "Login to HuggingFace for gated models"),
            ("/quit", "Exit Cortex")
        ]
        
        for cmd, desc in commands:
            # Format: "  /command    description"
            # Color the command in yellow
            colored_cmd = f"\033[93m{cmd}\033[0m"
            # Calculate padding between command and description
            cmd_width = len(cmd)
            padding = " " * (12 - cmd_width)  # Align descriptions at column 14
            line = f"  {colored_cmd}{padding}{desc}"
            self.print_box_line(line, width)
        
        self.print_empty_line(width)
        self.print_box_footer(width)
    
    def download_model(self, args: str = ""):
        """Download a model from HuggingFace."""
        if args:
            # Direct download with provided args
            parts = args.split()
            repo_id = parts[0]
            filename = parts[1] if len(parts) > 1 else None
        else:
            # Interactive mode with numbered options
            width = min(self.get_terminal_width() - 2, 70)
            
            # Create download UI box using helper methods
            print()
            self.print_box_header("Model Manager", width)
            self.print_empty_line(width)
            
            option_num = 1
            available = self.model_manager.discover_available_models()
            
            # Show already downloaded models with numbers to load
            if available:
                self.print_box_line("  \033[96mLoad Existing Model:\033[0m", width)
                self.print_empty_line(width)
                
                for model in available[:5]:  # Show up to 5 downloaded models
                    name = model['name'][:width-15]
                    size = f"{model['size_gb']:.1f}GB"
                    line = f"    \033[93m[{option_num}]\033[0m {name} \033[2m({size})\033[0m"
                    self.print_box_line(line, width)
                    option_num += 1
                
                if len(available) > 5:
                    line = f"    \033[93m[{option_num}]\033[0m \033[2mShow all {len(available)} models...\033[0m"
                    self.print_box_line(line, width)
                    option_num += 1
                
                self.print_empty_line(width)
                self.print_box_separator(width)
                self.print_empty_line(width)
            
            # Download new model options
            self.print_box_line("  \033[96mDownload New Model:\033[0m", width)
            self.print_empty_line(width)
            
            # Show format in dimmed color
            line = f"    \033[2mEnter repository ID (e.g., meta-llama/Llama-3.2-3B)\033[0m"
            self.print_box_line(line, width)
            
            self.print_empty_line(width)
            self.print_box_footer(width)
            
            # Get user choice
            choice = self.get_input_with_escape("Choice or repo ID")
            
            if choice is None:
                return
            
            try:
                choice_num = int(choice)
                
                # Load existing model
                if available and choice_num <= len(available[:5]):
                    model = available[choice_num - 1]
                    print(f"\n\033[96m⚡\033[0m Loading {model['name']}...")
                    success, msg = self.model_manager.load_model(model['path'])
                    if success:
                        print(f"\033[32m✓\033[0m Model loaded successfully!")
                        
                        # Show template information
                        model_info = self.model_manager.get_current_model()
                        if model_info:
                            tokenizer = self.model_manager.tokenizers.get(model_info.name)
                            profile = self.template_registry.setup_model(
                                model_info.name, 
                                tokenizer=tokenizer,
                                interactive=False
                            )
                            if profile:
                                template_name = profile.config.name
                                print(f"   \033[2m• Template: {template_name}\033[0m")
                    else:
                        print(f"\033[31m✗\033[0m Failed to load: {msg}")
                    return
                
                # Show all models
                elif available and choice_num == len(available[:5]) + 1 and len(available) > 5:
                    print()
                    self.manage_models()  # Use the unified model manager
                    return
                else:
                    print(f"\033[31m✗ Invalid choice\033[0m")
                    return
                    
            except ValueError:
                # Not a number, treat as repository ID
                repo_id = choice
                # Check if filename is provided
                parts = repo_id.split()
                repo_id = parts[0]
                filename = parts[1] if len(parts) > 1 else None
        
        # Validate format
        if '/' not in repo_id:
            print(f"\n\033[31m✗ Invalid format. Expected: username/model-name\033[0m")
            return
        
        # Show download starting
        print(f"\n\033[96m⬇\033[0m Downloading: \033[93m{repo_id}\033[0m")
        if filename:
            print(f"   File: \033[93m{filename}\033[0m")
        print()
        
        success, message, path = self.model_downloader.download_model(repo_id, filename)
        
        if success:
            # Success message in a nice box
            width = min(self.get_terminal_width() - 2, 70)
            print()
            # Create a custom header with green color for success
            title_with_color = " \033[32mDownload Complete\033[0m "
            visible_len = self.get_visible_length(title_with_color)
            padding = width - visible_len - 3  # -3 for "╭─" and "╮"
            print(f"╭─{title_with_color}" + "─" * padding + "╮")
            self.print_box_line("  \033[32m✓\033[0m Model downloaded successfully!", width)
            
            location_str = str(path)[:width-13]
            self.print_box_line(f"  \033[2mLocation: {location_str}\033[0m", width)
            self.print_empty_line(width)
            self.print_box_line("  \033[96mLoad this model now?\033[0m", width)
            self.print_box_line("  \033[93m[Y]es\033[0m  \033[2m[N]o\033[0m", width)
            self.print_box_footer(width)
            
            try:
                choice = input("\n\033[96m▶\033[0m Choice (\033[93my\033[0m/\033[2mn\033[0m): ").strip().lower()
                if choice in ['y', 'yes']:
                    print(f"\n\033[96m⚡\033[0m Loading model...")
                    load_success, load_msg = self.model_manager.load_model(str(path))
                    if load_success:
                        print(f"\033[32m✓\033[0m Model loaded successfully!")
                    else:
                        print(f"\033[31m✗\033[0m Failed to load: {load_msg}")
            except KeyboardInterrupt:
                print("\n\033[2mCancelled\033[0m")
        else:
            print(f"\n\033[31m✗\033[0m {message}")
    
    def hf_login(self):
        """Login to HuggingFace for accessing gated models."""
        try:
            from huggingface_hub import login, HfApi
            from huggingface_hub.utils import HfHubHTTPError
        except ImportError:
            print("\n\033[31m✗\033[0m huggingface-hub not installed. Install with: pip install huggingface-hub")
            return
        
        width = min(self.get_terminal_width() - 2, 70)
        
        # Create login UI box
        print()
        self.print_box_header("HuggingFace Login", width)
        self.print_empty_line(width)
        
        # Check if already logged in
        try:
            api = HfApi()
            user_info = api.whoami()
            if user_info:
                username = user_info.get('name', 'Unknown')
                self.print_box_line(f"  \033[32m✓\033[0m Already logged in as: \033[93m{username}\033[0m", width)
                self.print_empty_line(width)
                self.print_box_line("  \033[96mOptions:\033[0m", width)
                self.print_box_line("  \033[93m[1]\033[0m Login with new token", width)
                self.print_box_line("  \033[93m[2]\033[0m Logout", width)
                self.print_box_line("  \033[93m[3]\033[0m Cancel", width)
                self.print_box_footer(width)
                
                choice = self.get_input_with_escape("Select option (1-3)")
                if choice == '1':
                    # Continue to login flow
                    pass
                elif choice == '2':
                    # Logout
                    from huggingface_hub import logout
                    logout()
                    print("\n\033[32m✓\033[0m Successfully logged out from HuggingFace")
                    return
                else:
                    return
        except:
            # Not logged in, continue to login flow
            pass
        
        # Show login instructions
        print()
        self.print_box_header("HuggingFace Login", width)
        self.print_empty_line(width)
        self.print_box_line("  To access gated models, you need a HuggingFace token.", width)
        self.print_empty_line(width)
        self.print_box_line("  \033[96m1.\033[0m Get your token from:", width)
        self.print_box_line("     \033[93mhttps://huggingface.co/settings/tokens\033[0m", width)
        self.print_empty_line(width)
        self.print_box_line("  \033[96m2.\033[0m Create a token with \033[93mread\033[0m permissions", width)
        self.print_empty_line(width)
        self.print_box_line("  \033[96m3.\033[0m Paste the token below (input hidden)", width)
        self.print_box_footer(width)
        
        # Get token with hidden input
        print()
        token = getpass.getpass("\033[96m▶\033[0m Enter token \033[2m(or press Enter to cancel)\033[0m: ")
        
        if not token:
            print("\033[2mCancelled\033[0m")
            return
        
        # Try to login
        print("\n\033[96m⚡\033[0m Authenticating with HuggingFace...")
        try:
            login(token=token, add_to_git_credential=True)
            
            # Verify login
            api = HfApi()
            user_info = api.whoami()
            username = user_info.get('name', 'Unknown')
            
            print(f"\033[32m✓\033[0m Successfully logged in as: \033[93m{username}\033[0m")
            print("\033[2m  Token saved for future use\033[0m")
            print("\033[2m  You can now download gated models\033[0m")
            
        except HfHubHTTPError as e:
            if "Invalid token" in str(e):
                print("\033[31m✗\033[0m Invalid token. Please check your token and try again.")
            else:
                print(f"\033[31m✗\033[0m Login failed: {str(e)}")
        except Exception as e:
            print(f"\033[31m✗\033[0m Login failed: {str(e)}")
    
    def manage_models(self, args: str = ""):
        """Interactive model manager - simplified for better UX.
        If args provided, tries to load that model directly."""
        
        # If args provided, try direct load
        if args:
            print(f"\033[96m⚡\033[0m Loading model: \033[93m{args}\033[0m...")
            success, message = self.model_manager.load_model(args)
            if success:
                print(f"\033[32m✓\033[0m Model loaded successfully")
            else:
                print(f"\033[31m✗\033[0m Failed: {message}", file=sys.stderr)
            return
        
        # Interactive mode
        available = self.model_manager.discover_available_models()
        
        if not available:
            print(f"\n\033[31m✗\033[0m No models found in \033[2m{self.config.model.model_path}\033[0m")
            print("Use \033[93m/download\033[0m to download models from HuggingFace")
            return
        
        width = min(self.get_terminal_width() - 2, 70)
        
        # Build the model manager dialog using helper methods
        print()
        self.print_box_header("Select Model", width)
        self.print_empty_line(width)
        
        # List models with numbers - simplified view
        for i, model in enumerate(available, 1):
            # Model name and size
            name = model['name'][:width-30]
            size = f"{model['size_gb']:.1f}GB"
            
            # Check if currently loaded (handle both original name and MLX cached name)
            current_model = self.model_manager.current_model or ""
            is_current = (model['name'] == current_model or 
                         model.get('mlx_name') == current_model or
                         current_model.endswith(model['name']))
            
            # Build status indicators
            status_parts = []
            if model.get('mlx_optimized'):
                status_parts.append("\033[36m⚡ MLX\033[0m")  # Cyan lightning for MLX
            elif model.get('mlx_available'):
                status_parts.append("\033[2m○ MLX ready\033[0m")  # Dim circle for can be optimized
            
            if is_current:
                status_parts.append("\033[32m● loaded\033[0m")
            
            status = " ".join(status_parts) if status_parts else ""
            
            # Format the line
            if model.get('mlx_optimized'):
                # Show optimized model with special formatting
                line = f"  \033[93m[{i}]\033[0m {name} \033[2m({size})\033[0m {status}"
            else:
                line = f"  \033[93m[{i}]\033[0m {name} \033[2m({size})\033[0m {status}"
            
            self.print_box_line(line, width)
        
        self.print_empty_line(width)
        self.print_box_separator(width)
        self.print_empty_line(width)
        
        # Additional options
        self.print_box_line(f"  \033[93m[D]\033[0m Delete a model", width)
        self.print_box_line(f"  \033[93m[N]\033[0m Download new model", width)
        
        self.print_empty_line(width)
        self.print_box_footer(width)
        
        # Get user choice
        choice = self.get_input_with_escape(f"Select model to load (1-{len(available)}) or option")
        
        if choice is None:
            return
        
        choice = choice.lower()
        
        if choice == 'n':
            self.download_model()
            return
        elif choice == 'd':
            # Delete mode - show models again for deletion
            del_choice = self.get_input_with_escape(f"Select model to delete (1-{len(available)})")
            if del_choice is None:
                return
            try:
                model_idx = int(del_choice) - 1
                if 0 <= model_idx < len(available):
                    selected_model = available[model_idx]
                    print(f"\n\033[31m⚠\033[0m Delete \033[93m{selected_model['name']}\033[0m?")
                    print(f"   This will free \033[93m{selected_model['size_gb']:.1f}GB\033[0m of disk space.")
                    confirm = self.get_input_with_escape("Confirm deletion (\033[93my\033[0m/\033[2mN\033[0m)")
                    if confirm is None:
                        return
                    confirm = confirm.lower()
                    
                    if confirm == 'y':
                        # Delete the model
                        model_path = Path(selected_model['path'])
                        try:
                            if model_path.is_file():
                                model_path.unlink()
                            elif model_path.is_dir():
                                import shutil
                                shutil.rmtree(model_path)
                            
                            print(f"\033[32m✓\033[0m Model deleted successfully. Freed \033[93m{selected_model['size_gb']:.1f}GB\033[0m.")
                            
                            # If this was the current model, clear it
                            if selected_model['name'] == self.model_manager.current_model:
                                self.model_manager.current_model = None
                                print("\033[2mNote: Deleted model was currently loaded. Load another model to continue.\033[0m")
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
                
                # If already loaded, inform user
                if selected_model['name'] == self.model_manager.current_model:
                    print(f"\033[2mModel already loaded: {selected_model['name']}\033[0m")
                    return
                
                # Load model directly - no second prompt
                print(f"\n\033[96m⚡\033[0m Loading \033[93m{selected_model['name']}\033[0m...")
                success, message = self.model_manager.load_model(selected_model['path'])
                if success:
                    # Show the same detailed info as startup
                    model_info = self.model_manager.get_current_model()
                    if model_info:
                        # Determine quantization type from name or model info
                        model_name = model_info.name
                        if "_4bit" in model_name or "4bit" in str(model_info.quantization):
                            quant_type = "4-bit"
                        elif "_5bit" in model_name or "5bit" in str(model_info.quantization):
                            quant_type = "5-bit"
                        elif "_8bit" in model_name or "8bit" in str(model_info.quantization):
                            quant_type = "8-bit"
                        else:
                            quant_type = ""  # Don't duplicate "quantized"
                        
                        # Clean model name for display
                        clean_name = selected_model['name']
                        if clean_name.startswith("_Users_"):
                            # Extract just the model name from the path
                            parts = clean_name.split("_")
                            for i, part in enumerate(parts):
                                if "models" in part:
                                    clean_name = "_".join(parts[i+1:])
                                    break
                        clean_name = clean_name.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "")
                        
                        # Format the model format nicely
                        format_display = model_info.format.value
                        if format_display.lower() == "mlx":
                            format_display = "MLX (Apple Silicon optimized)"
                        elif format_display.lower() == "gguf":
                            format_display = "GGUF"  # Remove redundant "(quantized)"
                        elif format_display.lower() == "safetensors":
                            format_display = "SafeTensors"
                        elif format_display.lower() == "pytorch":
                            format_display = "PyTorch"
                        
                        print(f" \033[32m✓\033[0m Model ready: \033[93m{clean_name}\033[0m")
                        # Show quantization info only if we have specific type
                        if quant_type:
                            print(f"   \033[2m• Size: {model_info.size_gb:.1f}GB ({quant_type} quantized)\033[0m")
                        else:
                            print(f"   \033[2m• Size: {model_info.size_gb:.1f}GB (quantized)\033[0m")
                        print(f"   \033[2m• Optimizations: AMX acceleration, operation fusion\033[0m")
                        print(f"   \033[2m• Format: {format_display}\033[0m")
                        
                        # Show template information
                        tokenizer = self.model_manager.tokenizers.get(model_info.name)
                        profile = self.template_registry.setup_model(
                            model_info.name,
                            tokenizer=tokenizer,
                            interactive=False
                        )
                        if profile:
                            template_name = profile.config.name
                            print(f"   \033[2m• Template: {template_name}\033[0m")
                    else:
                        print(f"\033[32m✓\033[0m Model loaded successfully!")
                else:
                    print(f"\033[31m✗\033[0m Failed to load: {message}")
            else:
                print("\033[31m✗\033[0m Invalid selection")
        except ValueError:
            print("\033[31m✗\033[0m Invalid choice")
    
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
        is_valid, gpu_info, errors = self.gpu_validator.validate()
        
        width = min(self.get_terminal_width() - 2, 70)  # Consistent width with other dialogs
        
        print()
        self.print_box_header("Current Setup", width)
        self.print_empty_line(width)
        
        # GPU Info
        if gpu_info:
            self.print_box_line(f"  \033[2mGPU:\033[0m \033[93m{gpu_info.chip_name}\033[0m", width)
            self.print_box_line(f"  \033[2mCores:\033[0m \033[93m{gpu_info.gpu_cores}\033[0m", width)
            
            mem_gb = gpu_info.total_memory / (1024**3)
            mem_str = f"{mem_gb:.1f} GB"
            self.print_box_line(f"  \033[2mMemory:\033[0m \033[93m{mem_str}\033[0m", width)
        
        # Model Info
        if self.model_manager.current_model:
            model_info = self.model_manager.get_current_model()
            if model_info:
                self.print_box_line(f"  \033[2mModel:\033[0m \033[93m{model_info.name[:43]}\033[0m", width)
                
                # Template info
                tokenizer = self.model_manager.tokenizers.get(model_info.name)
                profile = self.template_registry.get_template(model_info.name)
                if profile:
                    template_name = profile.config.name
                    self.print_box_line(f"  \033[2mTemplate:\033[0m \033[93m{template_name}\033[0m", width)
        else:
            self.print_box_line(f"  \033[2mModel:\033[0m \033[31mNone loaded\033[0m", width)
        
        self.print_empty_line(width)
        self.print_box_footer(width)
    
    def show_gpu_status(self):
        """Show GPU status."""
        is_valid, gpu_info, errors = self.gpu_validator.validate()
        if gpu_info:
            print(f"\n\033[96mGPU Information:\033[0m")
            print(f"  Chip: \033[93m{gpu_info.chip_name}\033[0m")
            print(f"  GPU Cores: \033[93m{gpu_info.gpu_cores}\033[0m")
            print(f"  Total Memory: \033[93m{gpu_info.total_memory / (1024**3):.1f} GB\033[0m")
            print(f"  Available Memory: \033[93m{gpu_info.available_memory / (1024**3):.1f} GB\033[0m")
            print(f"  Metal Support: {'\033[32mYes\033[0m' if gpu_info.has_metal else '\033[31mNo\033[0m'}")
            print(f"  MPS Support: {'\033[32mYes\033[0m' if gpu_info.has_mps else '\033[31mNo\033[0m'}")
        
        memory_status = self.model_manager.get_memory_status()
        print(f"\n\033[96mMemory Status:\033[0m")
        print(f"  Available: \033[93m{memory_status['available_gb']:.1f} GB\033[0m")
        print(f"  Models Loaded: \033[93m{memory_status['models_loaded']}\033[0m")
        print(f"  Model Memory: \033[93m{memory_status['model_memory_gb']:.1f} GB\033[0m")
    
    def run_benchmark(self):
        """Run performance benchmark."""
        if not self.model_manager.current_model:
            print("\033[31m✗\033[0m No model loaded.")
            return
        
        print("\033[96m⚡\033[0m Running benchmark (100 tokens)...")
        metrics = self.inference_engine.benchmark()
        
        if metrics:
            print(f"\n\033[96mBenchmark Results:\033[0m")
            print(f"  Tokens Generated: \033[93m{metrics.tokens_generated}\033[0m")
            print(f"  Time: \033[93m{metrics.time_elapsed:.2f}s\033[0m")
            print(f"  Tokens/Second: \033[93m{metrics.tokens_per_second:.1f}\033[0m")
            print(f"  First Token: \033[93m{metrics.first_token_latency:.3f}s\033[0m")
            print(f"  GPU Usage: \033[93m{metrics.gpu_utilization:.1f}%\033[0m")
            print(f"  Memory: \033[93m{metrics.memory_used_gb:.1f}GB\033[0m")
    
    def manage_template(self, args: str = ""):
        """Manage template configuration for the current model."""
        if not self.model_manager.current_model:
            print("\033[31m✗\033[0m No model loaded.")
            return
        
        model_name = self.model_manager.current_model
        tokenizer = self.model_manager.tokenizers.get(model_name)
        
        # If args provided, handle specific subcommands
        if args:
            args_parts = args.split()
            subcommand = args_parts[0].lower()
            
            if subcommand == "reset":
                if self.template_registry.reset_model_config(model_name):
                    print(f"\033[32m✓\033[0m Template configuration reset for {model_name}")
                else:
                    print(f"\033[31m✗\033[0m No configuration found for {model_name}")
                return
            elif subcommand == "status":
                config = self.template_registry.config_manager.get_model_config(model_name)
                if config:
                    self.template_registry.interactive.show_current_config(model_name, config)
                else:
                    print(f"\033[33m⚠\033[0m No template configuration for {model_name}")
                return
        
        # Interactive template configuration
        print(f"\n\033[96m⚙\033[0m Configuring template for: \033[93m{model_name}\033[0m")
        
        # Force interactive setup
        profile = self.template_registry.setup_model(
            model_name,
            tokenizer=tokenizer,
            interactive=True,
            force_setup=True
        )
        
        print(f"\n\033[32m✓\033[0m Template configured successfully!")
    
    def run_finetune(self):
        """Run the interactive fine-tuning wizard."""
        # Check if any models are available
        available = self.model_manager.discover_available_models()
        if not available:
            print(f"\n\033[31m✗\033[0m No models found. Use \033[93m/download\033[0m to download a model first.")
            return
        
        # Pass CLI instance to wizard so it can use the box methods
        self.fine_tune_wizard.cli = self
        
        # Run the wizard
        success, message = self.fine_tune_wizard.start()
        
        if success:
            print(f"\n\033[32m✓\033[0m {message}")
        else:
            if "cancelled" not in message.lower():
                print(f"\n\033[31m✗\033[0m {message}")
            # If cancelled, wizard already handles the message
    
    def generate_response(self, user_input: str):
        """Generate and stream response from the model."""
        if not self.model_manager.current_model:
            print("\n\033[31m✗\033[0m No model loaded. Use \033[93m/model\033[0m to load a model or \033[93m/download\033[0m to download one.")
            return
        
        # Get current model name and tokenizer
        model_name = self.model_manager.current_model
        tokenizer = self.model_manager.tokenizers.get(model_name)
        
        # Setup model template to get the profile
        template_profile = None
        uses_reasoning_template = False
        try:
            template_profile = self.template_registry.setup_model(
                model_name,
                tokenizer=tokenizer,
                interactive=False
            )
            # Check if this is a reasoning template
            if template_profile and hasattr(template_profile.config, 'template_type'):
                from cortex.template_registry.template_profiles.base import TemplateType
                uses_reasoning_template = (template_profile.config.template_type == TemplateType.REASONING)
        except Exception as e:
            logger.debug(f"Failed to get template profile: {e}")
        
        # Build conversation context with proper formatting BEFORE adding to conversation
        formatted_prompt = self._format_prompt_with_chat_template(user_input)
        
        # DEBUG: Uncomment these lines to see the exact prompt being sent to the model
        # This is crucial for debugging when models give unexpected responses
        # It shows the formatted prompt with all special tokens and formatting
        # print(f"\033[33m[DEBUG] Formatted prompt being sent to model:\033[0m", file=sys.stderr)
        # print(f"\033[33m{repr(formatted_prompt[:200])}...\033[0m", file=sys.stderr)
        
        # Now add user message to conversation history  
        self.conversation_manager.add_message(MessageRole.USER, user_input)
        
        # Start response on a new line; prefix is rendered with the markdown output.
        print()
        
        # Get stop sequences from template profile
        stop_sequences = []
        if template_profile and hasattr(template_profile, 'get_stop_sequences'):
            try:
                stop_sequences = template_profile.get_stop_sequences()
                logger.debug(f"Using stop sequences from template: {stop_sequences}")
            except Exception as e:
                logger.debug(f"Could not get stop sequences: {e}")
        
        # Create generation request with formatted prompt
        request = GenerationRequest(
            prompt=formatted_prompt,
            max_tokens=self.config.inference.max_tokens,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            top_k=self.config.inference.top_k,
            repetition_penalty=self.config.inference.repetition_penalty,
            stream=self.config.inference.stream_output,
            seed=self.config.inference.seed if self.config.inference.seed >= 0 else None,
            stop_sequences=stop_sequences
        )
        
        # Generate response
        self.generating = True
        generated_text = ""
        start_time = time.time()
        token_count = 0
        first_token_time = None

        try:
            # Reset streaming state for reasoning templates if supported
            if uses_reasoning_template and template_profile and template_profile.supports_streaming():
                if hasattr(template_profile, 'reset_streaming_state'):
                    template_profile.reset_streaming_state()

            display_text = ""
            accumulated_response = ""
            last_render_time = 0.0
            render_interval = 0.05  # seconds
            prefix_style = Style(color="cyan")

            def build_renderable(text: str):
                if getattr(self.config.ui, "markdown_rendering", True):
                    markdown = ThinkMarkdown(
                        text,
                        code_theme="monokai",
                        use_line_numbers=False,
                        syntax_highlighting=getattr(self.config.ui, "syntax_highlighting", True),
                    )
                    renderable = markdown
                else:
                    renderable = render_plain_with_think(text)

                return PrefixedRenderable(renderable, prefix="⏺ ", prefix_style=prefix_style, indent="  ")

            with Live(
                build_renderable(""),
                console=self.console,
                refresh_per_second=20,
                transient=False,
            ) as live:
                for token in self.inference_engine.generate(request):
                    if first_token_time is None:
                        first_token_time = time.time()

                    generated_text += token
                    token_count += 1

                    display_token = token
                    if uses_reasoning_template and template_profile and template_profile.supports_streaming():
                        display_token, should_display = template_profile.process_streaming_response(
                            token, accumulated_response
                        )
                        accumulated_response += token
                        if not should_display:
                            display_token = ""

                    if display_token:
                        display_text += display_token

                    now = time.time()
                    if display_token and ("\n" in display_token or now - last_render_time >= render_interval):
                        live.update(build_renderable(display_text))
                        last_render_time = now

                if uses_reasoning_template and template_profile:
                    final_text = template_profile.process_response(generated_text)
                    generated_text = final_text
                    if not template_profile.config.show_reasoning:
                        display_text = final_text

                live.update(build_renderable(display_text))

            # Add blank line for spacing between response and metrics
            print()
            
            # Display final metrics in a clean, professional way
            elapsed = time.time() - start_time
            if token_count > 0 and elapsed > 0:
                tokens_per_sec = token_count / elapsed
                first_token_latency = first_token_time - start_time if first_token_time else 0
                
                # Build metrics parts - all will be wrapped in dim for subtlety
                metrics_parts = []
                
                if first_token_latency > 0.1:
                    # First token latency
                    metrics_parts.append(f"first {first_token_latency:.2f}s")
                
                # Total time
                metrics_parts.append(f"total {elapsed:.1f}s")
                
                # Token count
                metrics_parts.append(f"tokens {token_count}")
                
                # Throughput
                metrics_parts.append(f"speed {tokens_per_sec:.1f} tok/s")
                
                # Print entire metrics line as dim/secondary to make it less prominent
                # Indent metrics to align with response text
                metrics_line = " · ".join(metrics_parts)
                print(f"  \033[2m{metrics_line}\033[0m")
            
            # Add assistant message to conversation history
            self.conversation_manager.add_message(MessageRole.ASSISTANT, generated_text)
            
        except Exception as e:
            print(f"\n\033[31m✗ Error:\033[0m {str(e)}", file=sys.stderr)
        
        finally:
            self.generating = False

    def get_user_input(self) -> str:
        """Get user input with standard prompt."""
        try:
            print()
            user_input = input("> ")
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            raise
    
    def _format_prompt_with_chat_template(self, user_input: str) -> str:
        """Format the prompt with appropriate chat template for the model."""
        # Get current conversation context
        conversation = self.conversation_manager.get_current_conversation()
        
        # Get the tokenizer for the current model
        model_name = self.model_manager.current_model
        tokenizer = self.model_manager.tokenizers.get(model_name)
        
        # Build messages list from conversation history
        messages = []
        
        # Add conversation history if exists
        if conversation and conversation.messages:
            # Include recent context (last few messages)
            context_messages = conversation.messages[-10:]  # Last 10 messages for context
            for msg in context_messages:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Use template registry to format messages
        try:
            # Setup model template if not already configured
            profile = self.template_registry.setup_model(
                model_name, 
                tokenizer=tokenizer,
                interactive=False  # Non-interactive for smoother experience
            )
            
            # Format messages using the template
            formatted = profile.format_messages(messages, add_generation_prompt=True)
            
            # DEBUG: Uncomment to see formatted prompt
            # print(f"\033[36m[DEBUG] Using template: {profile.config.name}\033[0m", file=sys.stderr)
            # print(f"\033[36m[DEBUG] Formatted prompt preview: {formatted[:200]}...\033[0m", file=sys.stderr)
            
            return formatted
            
        except (AttributeError, TypeError, ValueError) as e:
            # Fallback to old method if template registry fails
            logger.debug(f"Template registry failed: {e}, using fallback")
            
            if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
                # Try direct tokenizer method
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    return formatted
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug(f"Tokenizer apply_chat_template failed: {e}")
        
        # Fallback: For TinyLlama and other chat models, use the proper format
        # Check if it's a chat model
        if model_name and "chat" in model_name.lower():
            # DEBUG: Uncomment to see when fallback chat format is used
            # This occurs when tokenizer doesn't have apply_chat_template method
            # print(f"\033[35m[DEBUG] Using chat model fallback for: {model_name}\033[0m", file=sys.stderr)
            
            # Use the proper chat format for TinyLlama and similar models
            # Build conversation history
            history = ""
            if conversation and conversation.messages:
                recent_messages = conversation.messages[-6:]  # Get last few messages
                for msg in recent_messages:
                    if msg.role == MessageRole.USER:
                        history += f"<|user|>\n{msg.content}</s>\n"
                    elif msg.role == MessageRole.ASSISTANT:
                        history += f"<|assistant|>\n{msg.content}</s>\n"
            
            # Add current user message with proper format
            prompt = f"{history}<|user|>\n{user_input}</s>\n<|assistant|>\n"
            
            # DEBUG: Uncomment to confirm fallback format was applied
            # print(f"\033[35m[DEBUG] Chat fallback format used\033[0m", file=sys.stderr)
            return prompt
        
        # Generic fallback for non-chat models
        if conversation and len(conversation.messages) > 0:
            # Include some conversation history
            context = ""
            recent_messages = conversation.messages[-6:]  # Get last few messages
            for msg in recent_messages:
                if msg.role == MessageRole.USER:
                    context += f"User: {msg.content}\n"
                elif msg.role == MessageRole.ASSISTANT:
                    context += f"Assistant: {msg.content}\n"
            
            # Add current exchange
            prompt = f"{context}User: {user_input}\nAssistant:"
        else:
            # First message in conversation - use simple format
            prompt = f"User: {user_input}\nAssistant:"
        
        return prompt
    
    def get_input_from_box(self) -> str:
        """Get user input from a styled input box.
        
        Displays a green-bordered input box, collects user input, then converts
        the box to a simple prompt in the conversation history.
        
        Guarantees that the input box is fully cleared after submission so
        no borders/pipes remain on screen.
        """
        width = self.get_terminal_width()
        
        # ANSI codes
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        DIM = "\033[2m"
        RESET = "\033[0m"
        CLEAR_LINE = "\033[2K"
        CLEAR_TO_EOL = "\033[K"
        CURSOR_UP = "\033[A"
        CURSOR_DOWN = "\033[B"
        MOVE_COL = lambda n: f"\033[{n}G"
        
        # Get current model name for display
        current_model = ""
        if self.model_manager.current_model:
            model_name = os.path.basename(self.model_manager.current_model)
            # Display full model name without truncation
            current_model = f"{DIM}Model:{RESET} {YELLOW}{model_name}{RESET}"
        
        # Draw the input box with dim borders
        print()
        print(f"{DIM}╭{'─' * (width - 2)}╮{RESET}")
        print(f"{DIM}│{RESET}{' ' * (width - 2)}{DIM}│{RESET}")
        print(f"{DIM}│{RESET}{' ' * (width - 2)}{DIM}│{RESET}")
        print(f"{DIM}│{RESET}{' ' * (width - 2)}{DIM}│{RESET}")
        print(f"{DIM}╰{'─' * (width - 2)}╯{RESET}")
        
        # Bottom hint: show current model aligned with box
        if current_model:
            print(f"{current_model}")
        else:
            print()  # Empty line if no model loaded
        
        # Move cursor to input position inside the box (center of 3 interior lines)
        sys.stdout.write("\033[4A")  # Move up 4 lines to the input line
        sys.stdout.write(f"\r{DIM}│{RESET} > ")  # Position at prompt
        sys.stdout.flush()
        
        try:
            # Get user input with custom character handling
            user_input = self._get_protected_input(width)
            
            # After _get_protected_input returns, the cursor is at the start of the
            # bottom border line (due to CRLFs when Enter was pressed).
            # Explicitly clear the entire input box region using relative moves.
            # 1) Clear hint line (one line below bottom border)
            sys.stdout.write(f"{CURSOR_DOWN}\r{CLEAR_LINE}")
            # 2) Clear bottom border
            sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")
            # 3) Clear padding line
            sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")
            # 4) Clear input line
            sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")
            # 5) Clear padding line
            sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")
            # 6) Clear top border
            sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")
            
            # Position cursor at the start of where the top border was and print
            # the clean prompt that represents the submitted user message.
            sys.stdout.write("\r> " + user_input.strip() + "\n")
            sys.stdout.flush()
            
            return user_input.strip()
            
        except KeyboardInterrupt:
            # Cleanup already done in _get_protected_input before raising
            raise
        except EOFError:
            # Clean up the box on Ctrl+D by clearing the lines if possible.
            # We are on the input line.
            try:
                sys.stdout.write(f"\r{CLEAR_LINE}")  # input line
                sys.stdout.write(f"{CURSOR_DOWN}\r{CLEAR_LINE}")  # padding line
                sys.stdout.write(f"{CURSOR_DOWN}\r{CLEAR_LINE}")  # bottom border
                sys.stdout.write(f"{CURSOR_DOWN}\r{CLEAR_LINE}")  # hint line
                sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")  # bottom border
                sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")  # padding line
                sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")  # input line
                sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")  # padding line
                sys.stdout.write(f"{CURSOR_UP}\r{CLEAR_LINE}")  # top border
                sys.stdout.flush()
            finally:
                pass
            raise
    
    def _get_protected_input(self, box_width: int) -> str:
        """Get input with protection against deleting the prompt.
        
        This method reads input character by character and prevents
        the user from backspacing past the beginning of their input.
        """
        DIM = "\033[2m"
        RESET = "\033[0m"
        CLEAR_TO_END = "\033[K"
        SAVE_CURSOR = "\033[s"
        RESTORE_CURSOR = "\033[u"
        
        # Calculate usable width for text (box_width - borders - prompt)
        # box_width - 2 (borders) - 4 (prompt " > ")
        max_display_width = box_width - 6
        
        # Store terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode for character-by-character input
            # Disable ISIG so we can handle Ctrl+C manually for clean exit
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~termios.ICANON  # Disable canonical mode
            new_settings[3] = new_settings[3] & ~termios.ECHO    # Disable echo
            new_settings[3] = new_settings[3] & ~termios.ISIG     # Disable signals - we'll handle Ctrl+C manually
            new_settings[6][termios.VMIN] = 1   # Read at least 1 character
            new_settings[6][termios.VTIME] = 0  # No timeout
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
            
            input_buffer = []
            cursor_pos = 0
            view_offset = 0  # For horizontal scrolling when text exceeds width
            
            def redraw_line():
                """Redraw the entire input line with proper boundaries."""
                nonlocal view_offset
                
                # Calculate what portion of text to display
                if len(input_buffer) <= max_display_width:
                    # Text fits within box
                    display_text = ''.join(input_buffer)
                    display_cursor_pos = cursor_pos
                else:
                    # Text needs scrolling
                    # Ensure cursor is visible in the viewport
                    if cursor_pos < view_offset:
                        # Cursor moved left out of view
                        view_offset = cursor_pos
                    elif cursor_pos >= view_offset + max_display_width:
                        # Cursor moved right out of view
                        view_offset = cursor_pos - max_display_width + 1
                    
                    # Extract visible portion
                    display_text = ''.join(input_buffer[view_offset:view_offset + max_display_width])
                    display_cursor_pos = cursor_pos - view_offset
                
                # Clear line and redraw
                sys.stdout.write(f"\r{DIM}│{RESET} > {display_text}{CLEAR_TO_END}")
                
                # Draw right border at the correct position
                # box_width is the full width including borders, so border is at box_width position
                sys.stdout.write(f"\033[{box_width}G")  # Move to border column
                sys.stdout.write(f"{DIM}│{RESET}")
                
                # Position cursor at the correct location
                cursor_column = 5 + display_cursor_pos  # 5 = "│ > " 
                sys.stdout.write(f"\033[{cursor_column}G")
                sys.stdout.flush()
            
            # Initial display
            redraw_line()
            
            while True:
                char = sys.stdin.read(1)
                
                # Handle special characters
                if char == '\r' or char == '\n':  # Enter key
                    sys.stdout.write('\r\n')
                    sys.stdout.write('\r\n')
                    sys.stdout.flush()
                    break
                    
                elif char == '\x7f' or char == '\x08':  # Backspace (DEL or BS)
                    # Only allow backspace if there are characters to delete
                    if cursor_pos > 0:
                        cursor_pos -= 1
                        input_buffer.pop(cursor_pos)
                        redraw_line()
                    # If cursor_pos is 0, do nothing (can't delete the prompt)
                    
                elif char == '\x03':  # Ctrl+C
                    # Clean up the display before raising KeyboardInterrupt
                    # We're in the input line, need to clear the entire box
                    sys.stdout.write("\r\033[2K")  # Clear current line
                    sys.stdout.write("\033[1B\r\033[2K")  # Down 1, clear padding line
                    sys.stdout.write("\033[1B\r\033[2K")  # Down 1, clear bottom border
                    sys.stdout.write("\033[1B\r\033[2K")  # Down 1, clear model line  
                    sys.stdout.write("\033[4A\r\033[2K")  # Up 4 to padding line, clear
                    sys.stdout.write("\033[1A\r\033[2K")  # Up 1 to top border, clear
                    sys.stdout.write("\033[1A\r\033[2K")  # Up 1 to empty line, clear
                    sys.stdout.write("\r")  # Position at start
                    sys.stdout.flush()
                    # Now raise the interrupt for clean exit
                    raise KeyboardInterrupt
                    
                elif char == '\x04':  # Ctrl+D
                    raise EOFError
                    
                elif char == '\x1b':  # ESC sequence (arrow keys, etc.)
                    # Read the rest of the escape sequence
                    next1 = sys.stdin.read(1)
                    if next1 == '[':
                        next2 = sys.stdin.read(1)
                        if next2 == 'D':  # Left arrow
                            if cursor_pos > 0:
                                cursor_pos -= 1
                                redraw_line()
                        elif next2 == 'C':  # Right arrow
                            if cursor_pos < len(input_buffer):
                                cursor_pos += 1
                                redraw_line()
                        elif next2 == 'H':  # Home
                            cursor_pos = 0
                            view_offset = 0
                            redraw_line()
                        elif next2 == 'F':  # End
                            cursor_pos = len(input_buffer)
                            redraw_line()
                        # For other sequences, continue without action
                        continue
                    
                elif ord(char) >= 32:  # Printable character
                    # Insert character at cursor position
                    input_buffer.insert(cursor_pos, char)
                    cursor_pos += 1
                    redraw_line()
            
            return ''.join(input_buffer)
            
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    
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
                break
            except KeyboardInterrupt:
                # Clean exit on Ctrl+C, same as /quit
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
