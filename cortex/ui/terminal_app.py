"""Terminal application UI using Textual."""

from typing import Optional, List
from datetime import datetime
import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Input, Static
from textual.reactive import reactive
from rich.text import Text
from rich.console import Group

from cortex.config import Config
from cortex.gpu_validator import GPUValidator
from cortex.model_manager import ModelManager
from cortex.inference_engine import InferenceEngine, GenerationRequest
from cortex.conversation_manager import ConversationManager, MessageRole
from cortex.ui import UIComponents
from cortex.ui.markdown_render import ThinkMarkdown

class MessageDisplay(Static):
    """Widget to display a single message."""

    def __init__(self, role: str, content: str, timestamp: datetime, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.timestamp = timestamp

    def render(self):
        """Render the message."""
        role_colors = {
            "system": "yellow",
            "user": "cyan",
            "assistant": "green"
        }

        role_text = Text(f"{self.role.title()}", style=f"bold {role_colors.get(self.role, 'white')}")
        timestamp_text = Text(f" ({self.timestamp.strftime('%H:%M:%S')})", style="dim")

        header = Text()
        header.append(role_text)
        header.append(timestamp_text)
        header.append("\n")

        content_renderable = ThinkMarkdown(
            self.content,
            code_theme="monokai",
            use_line_numbers=True,
        )

        return Group(header, content_renderable)

class ConversationView(ScrollableContainer):
    """Widget to display conversation messages."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[MessageDisplay] = []
    
    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """Add a message to the conversation view."""
        timestamp = timestamp or datetime.now()
        message_widget = MessageDisplay(role, content, timestamp)
        self.messages.append(message_widget)
        self.mount(message_widget)
        self.scroll_end()
    
    def clear_messages(self):
        """Clear all messages."""
        for message in self.messages:
            message.remove()
        self.messages.clear()
    
    def update_last_message(self, content: str):
        """Update the content of the last message."""
        if self.messages:
            last_message = self.messages[-1]
            last_message.content = content
            last_message.refresh()

class StatusBar(Static):
    """Status bar showing model and performance info."""
    
    model_name = reactive("No model loaded")
    status = reactive("idle")
    tokens_per_second = reactive(0.0)
    gpu_utilization = reactive(0.0)
    memory_gb = reactive(0.0)
    
    def render(self) -> Text:
        """Render the status bar."""
        status_icons = UIComponents.STATUS_ICONS
        icon = status_icons.get(self.status, "⚪")
        
        if self.model_name != "No model loaded":
            perf_text = UIComponents.format_performance_metrics(
                self.tokens_per_second,
                self.gpu_utilization,
                self.memory_gb
            )
            return Text(f"{icon} {self.model_name} | {perf_text}")
        else:
            return Text(f"{icon} {self.model_name}")

class CommandInput(Input):
    """Input widget with command support."""
    
    def __init__(self, **kwargs):
        super().__init__(placeholder="Type your message or / for commands...", **kwargs)
        self.command_mode = False
    
    def on_key(self, event):
        """Handle key events."""
        if event.key == "escape":
            self.command_mode = False
            self.placeholder = "Type your message or / for commands..."
        elif event.key == "/" and len(self.value) == 0:
            self.command_mode = True
            self.placeholder = "Enter command..."
        
        return super().on_key(event)

class TerminalApp(App):
    """Main terminal application."""
    
    CSS = """
    Screen {
        background: $background;
    }
    
    Header {
        background: $primary;
        color: $text;
        height: 3;
    }
    
    Footer {
        background: $primary;
        color: $text;
        height: 2;
    }
    
    ConversationView {
        border: solid $border;
        padding: 1;
        margin: 1;
        height: 100%;
    }
    
    MessageDisplay {
        margin-bottom: 1;
        padding: 1;
    }
    
    StatusBar {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text;
        padding: 0 1;
    }
    
    CommandInput {
        dock: bottom;
        height: 3;
        margin: 0;
        width: 100%;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+n", "new_conversation", "New Conversation"),
        Binding("ctrl+c", "cancel_generation", "Cancel"),
        Binding("ctrl+s", "save_conversation", "Save"),
        Binding("ctrl+l", "load_model", "Load Model"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+h", "help", "Help"),
    ]
    
    def __init__(
        self,
        config: Config,
        gpu_validator: GPUValidator,
        model_manager: ModelManager,
        inference_engine: InferenceEngine,
        conversation_manager: ConversationManager,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.gpu_validator = gpu_validator
        self.model_manager = model_manager
        self.inference_engine = inference_engine
        self.conversation_manager = conversation_manager
        
        self.title = "Cortex - GPU-Accelerated LLM Terminal"
        self.sub_title = "Apple Silicon | Metal Performance Shaders"
        
        self.conversation_view: Optional[ConversationView] = None
        self.status_bar: Optional[StatusBar] = None
        self.input_widget: Optional[CommandInput] = None
        
        self.generating = False
        self.generation_thread: Optional[threading.Thread] = None
    
    def compose(self) -> ComposeResult:
        """Compose the UI."""
        yield Header()
        
        self.conversation_view = ConversationView()
        yield self.conversation_view
        
        self.status_bar = StatusBar()
        yield self.status_bar
        
        self.input_widget = CommandInput()
        yield self.input_widget
        
        yield Footer()
    
    async def on_mount(self):
        """Called when app is mounted."""
        self.conversation_manager.new_conversation()
        
        if self.config.ui.show_gpu_utilization:
            self.set_interval(1.0, self.update_status)
        
        if self.config.model.default_model:
            await self.load_default_model()
        
        self.conversation_view.add_message(
            "system",
            "Welcome to Cortex! Type your message or use / for commands.",
            datetime.now()
        )
    
    async def load_default_model(self):
        """Load the default model."""
        model_path = str(self.config.model.model_path / self.config.model.default_model)
        success, message = self.model_manager.load_model(model_path)
        
        if success:
            model_info = self.model_manager.get_current_model()
            if model_info:
                self.status_bar.model_name = model_info.name
                self.conversation_view.add_message(
                    "system",
                    f"Loaded model: {model_info.name} ({model_info.size_gb:.1f}GB)",
                    datetime.now()
                )
        else:
            self.conversation_view.add_message(
                "system",
                f"Failed to load default model: {message}",
                datetime.now()
            )
    
    async def on_input_submitted(self, event):
        """Handle input submission."""
        input_text = self.input_widget.value.strip()
        
        if not input_text:
            return
        
        self.input_widget.value = ""
        
        if input_text.startswith("/"):
            await self.handle_command(input_text)
        else:
            await self.handle_message(input_text)
    
    async def handle_command(self, command: str):
        """Handle slash commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/model":
            await self.command_model(args)
        elif cmd == "/clear":
            self.command_clear()
        elif cmd == "/save":
            self.command_save()
        elif cmd == "/help":
            self.command_help()
        elif cmd == "/gpu":
            self.command_gpu_status()
        elif cmd == "/benchmark":
            await self.command_benchmark()
        elif cmd == "/quit":
            self.exit()
        else:
            self.conversation_view.add_message(
                "system",
                f"Unknown command: {cmd}",
                datetime.now()
            )
    
    async def handle_message(self, message: str):
        """Handle user message and generate response."""
        if self.generating:
            self.conversation_view.add_message(
                "system",
                "Generation already in progress. Press Ctrl+C to cancel.",
                datetime.now()
            )
            return
        
        if not self.model_manager.current_model:
            self.conversation_view.add_message(
                "system",
                "No model loaded. Use /model <path> to load a model.",
                datetime.now()
            )
            return
        
        self.conversation_view.add_message("user", message, datetime.now())
        self.conversation_manager.add_message(MessageRole.USER, message)
        
        self.conversation_view.add_message("assistant", "", datetime.now())
        
        self.generating = True
        self.status_bar.status = "generating"
        
        request = GenerationRequest(
            prompt=message,
            max_tokens=self.config.inference.max_tokens,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            top_k=self.config.inference.top_k,
            repetition_penalty=self.config.inference.repetition_penalty,
            stream=True
        )
        
        generated_text = ""
        
        def generate_worker():
            nonlocal generated_text
            try:
                for token in self.inference_engine.generate(request):
                    generated_text += token
                    self.call_from_thread(
                        self.conversation_view.update_last_message,
                        generated_text
                    )
                
                self.conversation_manager.add_message(MessageRole.ASSISTANT, generated_text)
                
                if self.inference_engine.current_metrics:
                    metrics = self.inference_engine.current_metrics
                    self.status_bar.tokens_per_second = metrics.tokens_per_second
                    self.status_bar.gpu_utilization = metrics.gpu_utilization
                    self.status_bar.memory_gb = metrics.memory_used_gb
                
            except Exception as e:
                self.call_from_thread(
                    self.conversation_view.add_message,
                    "system",
                    f"Error during generation: {str(e)}",
                    datetime.now()
                )
            finally:
                self.generating = False
                self.status_bar.status = "idle"
        
        self.generation_thread = threading.Thread(target=generate_worker)
        self.generation_thread.start()
    
    async def command_model(self, model_path: str):
        """Load a model."""
        if not model_path:
            models = self.model_manager.list_models()
            if models:
                model_list = "\n".join([f"- {m['name']} ({m['size_gb']:.1f}GB)" for m in models])
                self.conversation_view.add_message(
                    "system",
                    f"Loaded models:\n{model_list}",
                    datetime.now()
                )
            else:
                self.conversation_view.add_message(
                    "system",
                    "No models loaded. Use /model <path> to load a model.",
                    datetime.now()
                )
            return
        
        self.status_bar.status = "loading"
        success, message = self.model_manager.load_model(model_path)
        
        if success:
            model_info = self.model_manager.get_current_model()
            if model_info:
                self.status_bar.model_name = model_info.name
                self.conversation_view.add_message(
                    "system",
                    f"Loaded model: {model_info.name} ({model_info.size_gb:.1f}GB)",
                    datetime.now()
                )
        else:
            self.conversation_view.add_message(
                "system",
                f"Failed to load model: {message}",
                datetime.now()
            )
        
        self.status_bar.status = "idle"
    
    def command_clear(self):
        """Clear conversation."""
        self.conversation_view.clear_messages()
        self.conversation_manager.new_conversation()
        self.conversation_view.add_message(
            "system",
            "Conversation cleared.",
            datetime.now()
        )
    
    def command_save(self):
        """Save conversation."""
        try:
            export_data = self.conversation_manager.export_conversation(format="json")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.config.conversation.save_directory / f"conversation_{timestamp}.json"
            
            with open(filename, 'w') as f:
                f.write(export_data)
            
            self.conversation_view.add_message(
                "system",
                f"Conversation saved to {filename}",
                datetime.now()
            )
        except Exception as e:
            self.conversation_view.add_message(
                "system",
                f"Failed to save conversation: {str(e)}",
                datetime.now()
            )
    
    def command_help(self):
        """Show help information."""
        help_text = "Available commands:\n"
        for cmd, desc in UIComponents.COMMANDS.items():
            help_text += f"  {cmd:<15} - {desc}\n"
        
        help_text += "\nKeyboard shortcuts:\n"
        for action, key in UIComponents.SHORTCUTS.items():
            help_text += f"  {key:<15} - {action.replace('_', ' ').title()}\n"
        
        self.conversation_view.add_message("system", help_text, datetime.now())
    
    def command_gpu_status(self):
        """Show GPU status."""
        self.gpu_validator.print_gpu_info()
        memory_status = self.model_manager.get_memory_status()
        
        status_text = f"GPU Status:\n"
        status_text += f"  Total Memory: {memory_status['total_gb']:.1f} GB\n"
        status_text += f"  Available: {memory_status['available_gb']:.1f} GB\n"
        status_text += f"  Models Loaded: {memory_status['models_loaded']}\n"
        status_text += f"  Model Memory: {memory_status['model_memory_gb']:.1f} GB\n"
        
        self.conversation_view.add_message("system", status_text, datetime.now())
    
    async def command_benchmark(self):
        """Run benchmark."""
        if not self.model_manager.current_model:
            self.conversation_view.add_message(
                "system",
                "No model loaded for benchmark.",
                datetime.now()
            )
            return
        
        self.conversation_view.add_message(
            "system",
            "Running benchmark (100 tokens)...",
            datetime.now()
        )
        
        metrics = self.inference_engine.benchmark()
        
        if metrics:
            benchmark_text = f"Benchmark Results:\n"
            benchmark_text += f"  Tokens Generated: {metrics.tokens_generated}\n"
            benchmark_text += f"  Time Elapsed: {metrics.time_elapsed:.2f}s\n"
            benchmark_text += f"  Tokens/Second: {metrics.tokens_per_second:.1f}\n"
            benchmark_text += f"  First Token Latency: {metrics.first_token_latency:.3f}s\n"
            benchmark_text += f"  GPU Utilization: {metrics.gpu_utilization:.1f}%\n"
            benchmark_text += f"  Memory Used: {metrics.memory_used_gb:.1f}GB\n"
            
            self.conversation_view.add_message("system", benchmark_text, datetime.now())
    
    def update_status(self):
        """Update status bar with current metrics."""
        if self.inference_engine.current_metrics:
            metrics = self.inference_engine.current_metrics
            self.status_bar.tokens_per_second = metrics.tokens_per_second
            self.status_bar.gpu_utilization = metrics.gpu_utilization
            self.status_bar.memory_gb = metrics.memory_used_gb
    
    def action_new_conversation(self):
        """Action for new conversation."""
        self.command_clear()
    
    def action_cancel_generation(self):
        """Action to cancel generation."""
        if self.generating:
            self.inference_engine.cancel_generation()
            self.conversation_view.add_message(
                "system",
                "Generation cancelled.",
                datetime.now()
            )
    
    def action_save_conversation(self):
        """Action to save conversation."""
        self.command_save()
    
    def action_load_model(self):
        """Action to load model."""
        self.conversation_view.add_message(
            "system",
            "Use /model <path> to load a model.",
            datetime.now()
        )
    
    def action_help(self):
        """Action to show help."""
        self.command_help()
