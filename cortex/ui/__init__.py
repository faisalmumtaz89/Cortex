"""Terminal UI components for Cortex."""

from typing import Dict, Any, Optional

__all__ = [
    "Theme",
    "UIComponents"
]

class Theme:
    """UI theme configuration."""
    
    DEFAULT_THEME = {
        "primary": "#00ff00",
        "secondary": "#0080ff",
        "background": "#000000",
        "text": "#ffffff",
        "error": "#ff0000",
        "warning": "#ffff00",
        "success": "#00ff00",
        "info": "#0080ff",
        "border": "#444444",
        "highlight": "#333333"
    }
    
    DARK_THEME = DEFAULT_THEME
    
    LIGHT_THEME = {
        "primary": "#0080ff",
        "secondary": "#00c853",
        "background": "#ffffff",
        "text": "#000000",
        "error": "#d32f2f",
        "warning": "#f57c00",
        "success": "#388e3c",
        "info": "#1976d2",
        "border": "#cccccc",
        "highlight": "#eeeeee"
    }
    
    @classmethod
    def get_theme(cls, theme_name: str = "default") -> Dict[str, str]:
        """Get theme by name."""
        themes = {
            "default": cls.DEFAULT_THEME,
            "dark": cls.DARK_THEME,
            "light": cls.LIGHT_THEME
        }
        return themes.get(theme_name, cls.DEFAULT_THEME)

class UIComponents:
    """Common UI component definitions."""
    
    HEADER_HEIGHT = 3
    FOOTER_HEIGHT = 2
    SIDEBAR_WIDTH = 30
    MIN_TERMINAL_WIDTH = 80
    MIN_TERMINAL_HEIGHT = 24
    
    SHORTCUTS = {
        "new_conversation": "ctrl+n",
        "switch_conversation": "ctrl+tab",
        "cancel_generation": "ctrl+c",
        "duplicate_message": "ctrl+d",
        "command_palette": "ctrl+/",
        "quit": "ctrl+q",
        "save_conversation": "ctrl+s",
        "load_model": "ctrl+l",
        "settings": "ctrl+,",
        "help": "ctrl+h"
    }
    
    COMMANDS = {
        "/help": "Show available commands",
        "/status": "Show current setup and GPU info",
        "/download": "Download models from HuggingFace",
        "/model": "Switch or load a model",
        "/models": "List available models",
        "/clear": "Clear current conversation",
        "/save": "Save conversation",
        "/load": "Load conversation",
        "/export": "Export conversation",
        "/config": "Open settings",
        "/benchmark": "Run benchmark",
        "/gpu": "Show GPU status",
        "/quit": "Exit application"
    }
    
    STATUS_ICONS = {
        "idle": "⚪",
        "loading": "🔄",
        "generating": "⚡",
        "completed": "✅",
        "error": "❌",
        "cancelled": "⚠️"
    }
    
    @classmethod
    def format_performance_metrics(
        cls,
        tokens_per_second: float,
        gpu_utilization: float,
        memory_gb: float
    ) -> str:
        """Format performance metrics for display."""
        return f"⚡{tokens_per_second:.1f}t/s | GPU: {gpu_utilization:.0f}% | Mem: {memory_gb:.1f}GB"
    
    @classmethod
    def format_model_info(
        cls,
        model_name: str,
        quantization: str,
        context_used: int,
        context_total: int
    ) -> str:
        """Format model information for display."""
        context_percent = (context_used / context_total * 100) if context_total > 0 else 0
        return f"{model_name} [{quantization}] | Context: {context_used}/{context_total} ({context_percent:.0f}%)"

# from cortex.ui.terminal_app import TerminalApp  # Commented out - only needed for TUI mode