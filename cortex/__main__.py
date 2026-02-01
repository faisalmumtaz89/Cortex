"""Main entry point for Cortex."""

import sys
from pathlib import Path
import os
import warnings

# Disable multiprocessing resource tracking before any imports that might use it
# This prevents the semaphore leak warning from transformers library
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:multiprocessing.resource_tracker'

# Apply MLX compatibility shims before any MLX/MLX-LM imports.
try:
    from cortex.metal.mlx_compat import patch_mlx_lm_device_info
    patch_mlx_lm_device_info()
except Exception:
    pass

# Alternative: Monkey-patch the resource tracker before it's used
try:
    from multiprocessing import resource_tracker
    def dummy_register(*args, **kwargs):
        pass
    def dummy_unregister(*args, **kwargs):
        pass
    resource_tracker.register = dummy_register
    resource_tracker.unregister = dummy_unregister
except ImportError:
    pass

from cortex.config import Config
from cortex.gpu_validator import GPUValidator
from cortex.model_manager import ModelManager
from cortex.inference_engine import InferenceEngine
from cortex.conversation_manager import ConversationManager
from cortex.ui.cli import CortexCLI


def main():
    """Main entry point."""
    
    inference_engine = None
    try:
        # Load configuration
        config = Config()
        
        # Initialize GPU validator
        gpu_validator = GPUValidator()
        
        # Validate GPU
        is_valid, gpu_info, errors = gpu_validator.validate()
        if not is_valid:
            print("Error: GPU validation failed. Cortex requires Apple Silicon with Metal support.", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
        
        # Initialize components
        model_manager = ModelManager(config, gpu_validator)
        inference_engine = InferenceEngine(config, model_manager)
        conversation_manager = ConversationManager(config)
        
        # Create and run the CLI
        cli = CortexCLI(
            config=config,
            gpu_validator=gpu_validator,
            model_manager=model_manager,
            inference_engine=inference_engine,
            conversation_manager=conversation_manager
        )
        
        cli.run()
    finally:
        # Clean up resources
        if inference_engine is not None and hasattr(inference_engine, 'memory_pool') and inference_engine.memory_pool:
            inference_engine.memory_pool.cleanup()
        
        # Force PyTorch cleanup
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
