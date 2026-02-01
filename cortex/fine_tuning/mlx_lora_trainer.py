"""MLX LoRA trainer using mlx_lm implementation."""

import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import shutil
import math

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset, CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train, evaluate, TrainingCallback
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    # Keep the module importable when MLX/metal is missing so we can show a clear message.
    MLX_AVAILABLE = False
    mx = nn = mlx_load = linear_to_lora_layers = mlx_load_dataset = CacheDataset = TrainingArgs = train = evaluate = TrainingCallback = optim = None  # type: ignore
    _MLX_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class MLXLoRATrainer:
    """LoRA trainer using mlx_lm's implementation."""
    
    def __init__(self, model_manager, config):
        """Initialize the trainer."""
        self.model_manager = model_manager
        self.config = config
    
    @staticmethod
    def is_available() -> bool:
        """Return True when MLX/Metal stack is importable."""
        return MLX_AVAILABLE
    
    def train(
        self,
        base_model_name: str,
        dataset_path: Path,
        output_name: str,
        training_config: Any,
            progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Train a model using LoRA with mlx_lm.
        
        Args:
            base_model_name: Name of the base model to fine-tune
            dataset_path: Path to the training dataset
            output_name: Name for the fine-tuned model
            training_config: Training configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if training succeeded, False otherwise
        """
        try:
            if not MLX_AVAILABLE:
                logger.error("MLX is not available; fine-tuning requires MLX.")
                if "_MLX_IMPORT_ERROR" in globals():
                    logger.debug(f"MLX import error: {_MLX_IMPORT_ERROR}")  # type: ignore[name-defined]
                return False

            logger.info(f"Starting MLX LoRA training: {base_model_name} -> {output_name}")
            
            # Get the model path
            model_path = self._get_model_path(base_model_name)
            if not model_path:
                logger.error(f"Could not find model path for {base_model_name}")
                return False
            
            # Try to reuse an already loaded model to avoid a second full load in unified memory.
            model = None
            tokenizer = None
            if self.model_manager:
                cache_keys = [
                    str(model_path),
                    base_model_name,
                    getattr(self.model_manager, "current_model", None),
                ]
                for key in cache_keys:
                    if not key:
                        continue
                    if model is None:
                        model = self.model_manager.model_cache.get(key)
                    if tokenizer is None:
                        tokenizer = self.model_manager.tokenizers.get(key)
                    if model is not None and tokenizer is not None:
                        logger.info(f"Reusing loaded model from cache (key: {key})")
                        break
            
            # Load model and tokenizer using mlx_lm if not already cached
            if model is None or tokenizer is None:
                logger.info(f"Loading model from {model_path}")
                model, tokenizer = mlx_load(str(model_path))
            
            # Apply LoRA layers
            logger.info(f"Applying LoRA with rank={training_config.lora_r}")
            lora_config = LoRAConfig(
                rank=training_config.lora_r,
                alpha=training_config.lora_alpha,
                dropout=training_config.lora_dropout,
                target_modules=training_config.target_modules
            )
            
            # Convert linear layers to LoRA layers
            # Note: linear_to_lora_layers modifies the model in-place and returns None
            linear_to_lora_layers(
                model,
                num_layers=training_config.num_lora_layers if hasattr(training_config, 'num_lora_layers') else 16,
                config={
                    "rank": lora_config.rank,
                    "dropout": lora_config.dropout,
                    "scale": lora_config.alpha / lora_config.rank
                }
            )
            
            # Model freezing is handled automatically by linear_to_lora_layers
            # Only LoRA parameters will be trainable
            
            # Load dataset
            logger.info(f"Loading dataset from {dataset_path}")
            train_data = self._load_dataset(dataset_path, tokenizer, training_config)
            
            if not train_data:
                logger.error("Failed to load dataset")
                return False
            
            # Get dataset length properly
            if hasattr(train_data, '__len__'):
                dataset_len = len(train_data)
            elif hasattr(train_data, 'data') and hasattr(train_data.data, '__len__'):
                dataset_len = len(train_data.data)
            else:
                dataset_len = 1  # Fallback
            
            logger.info(f"Dataset contains {dataset_len} examples")
            
            # Setup training arguments
            adapter_file = str(Path.home() / ".cortex" / "adapters" / output_name / "adapter.safetensors")
            Path(adapter_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate iterations: total examples / (effective batch) * epochs
            effective_batch = max(1, training_config.batch_size) * max(
                1, getattr(training_config, "gradient_accumulation_steps", 1)
            )
            num_iters = max(1, math.ceil((dataset_len * training_config.epochs) / effective_batch))
            
            training_args = TrainingArgs(
                batch_size=training_config.batch_size,
                iters=num_iters,
                steps_per_report=10,
                # Avoid extra evaluation passes for small datasets by setting eval steps beyond total iters
                steps_per_eval=num_iters + 1,
                val_batches=1,  # Just 1 validation batch
                steps_per_save=100,
                adapter_file=adapter_file,
                grad_checkpoint=training_config.gradient_checkpointing if hasattr(training_config, 'gradient_checkpointing') else False,
            )
            
            # Setup optimizer with learning rate
            optimizer = optim.AdamW(
                learning_rate=training_config.learning_rate,
                weight_decay=training_config.weight_decay if hasattr(training_config, 'weight_decay') else 0.01
            )
            
            # Create a simple progress tracker
            class ProgressTracker(TrainingCallback):
                def __init__(self, callback, dataset_len, epochs):
                    self.callback = callback
                    self.total_iters = training_args.iters
                    self.steps_per_epoch = max(1, dataset_len // training_args.batch_size)
                    self.epochs = epochs
                
                def on_train_loss_report(self, train_info: dict):
                    """Called when training loss is reported."""
                    if self.callback:
                        iteration = train_info.get('iteration', 0)
                        loss = train_info.get('train_loss', 0.0)
                        # MLX iterations start at 1, not 0, so adjust
                        actual_iter = iteration - 1
                        # Calculate epoch based on actual iteration
                        epoch = actual_iter // self.steps_per_epoch
                        step = actual_iter % self.steps_per_epoch
                        # Ensure epoch doesn't exceed total epochs
                        epoch = min(epoch, self.epochs - 1)
                        self.callback(epoch, step, loss)
            
            tracker = ProgressTracker(progress_callback, dataset_len, training_config.epochs) if progress_callback else None
            
            # Prepare validation dataset
            # For MLX training, we always need a validation dataset (can't be None)
            # For small datasets, we'll use the same data for validation
            val_data = train_data  # Default to using training data for validation
            logger.info("Using training data for validation (small dataset)")
            
            # Training loop
            logger.info("Starting training...")
            # Note: train() doesn't return anything, it modifies model in-place and saves weights
            train(
                model,
                optimizer,
                train_dataset=train_data,
                val_dataset=val_data,  # Use proper validation dataset or None
                args=training_args,
                training_callback=tracker
            )
            
            # Save the fine-tuned model
            logger.info(f"Saving fine-tuned model to {output_name}")
            adapter_dir = Path(training_args.adapter_file).parent
            success = self._save_model(
                model=model,
                tokenizer=tokenizer,
                output_name=output_name,
                base_model_name=base_model_name,
                adapter_path=str(adapter_dir)
            )
            
            if success:
                logger.info(f"Successfully saved fine-tuned model as {output_name}")
                # Clean up training checkpoints after successful save
                self._cleanup_checkpoints(adapter_dir)
                return True
            else:
                logger.error("Failed to save fine-tuned model")
                return False
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            print("\n\033[93m⚠\033[0m Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            print(f"\n\033[31m✗\033[0m Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the path to the model, prioritizing MLX models."""
        # First check if it's already an MLX model (converted or fine-tuned)
        mlx_path = Path.home() / ".cortex" / "mlx_models" / model_name
        if mlx_path.exists():
            logger.info(f"Found MLX model at: {mlx_path}")
            return mlx_path
        
        # Check in models directory
        models_path = Path.home() / ".cortex" / "models" / model_name
        if models_path.exists():
            logger.info(f"Found model at: {models_path}")
            return models_path
        
        # Check in configured models directory (most common location)
        if self.model_manager and self.model_manager.config:
            try:
                config_model_path = Path(self.model_manager.config.model.model_path).expanduser().resolve()
                config_path = config_model_path / model_name
                if config_path.exists():
                    logger.info(f"Found model in configured path: {config_path}")
                    return config_path
            except Exception as e:
                logger.debug(f"Could not check configured model path: {e}")
        
        # Check if it's a full path
        if Path(model_name).exists():
            full_path = Path(model_name).resolve()
            logger.info(f"Found model at full path: {full_path}")
            return full_path
        
        # Last resort: check if it's a relative path in current directory
        current_path = Path.cwd() / model_name
        if current_path.exists():
            logger.info(f"Found model at current directory: {current_path}")
            return current_path
        
        logger.error(f"Model not found: {model_name}")
        return None
    
    def _load_dataset(self, dataset_path: Path, tokenizer: Any, training_config: Any) -> Optional[Any]:
        """Load and prepare the dataset."""
        try:
            from mlx_lm.tuner.datasets import CacheDataset, TextDataset
            
            # Load JSONL dataset
            examples = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    examples.append(data)
            
            # Check data format and create appropriate dataset
            if not examples:
                logger.error("No examples found in dataset")
                return None
            
            sample = examples[0]
            
            # Convert all formats to text format for simplicity
            # This avoids issues with tokenizers that don't have chat templates
            text_examples = []
            max_seq_len = getattr(training_config, "max_sequence_length", None)
            # crude char-level guard to avoid very long sequences; token-level truncation happens in tokenizer
            max_chars = max_seq_len * 4 if max_seq_len else None
            for example in examples:
                if 'prompt' in example and 'response' in example:
                    # Format as a simple conversation
                    text = f"User: {example['prompt']}\n\nAssistant: {example['response']}"
                elif 'prompt' in example and 'completion' in example:
                    text = f"User: {example['prompt']}\n\nAssistant: {example['completion']}"
                elif 'text' in example:
                    text = example['text']
                else:
                    logger.warning(f"Skipping example with unsupported format: {example}")
                    continue
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars]
                text_examples.append({'text': text})
            
            if not text_examples:
                logger.error("No valid examples found after conversion")
                return None
            
            # Create TextDataset which just uses tokenizer.encode()
            dataset = TextDataset(
                data=text_examples,
                tokenizer=tokenizer,
                text_key='text'
            )
            
            # Wrap with CacheDataset for efficiency
            cached_dataset = CacheDataset(dataset)
            
            logger.info(f"Loaded {len(text_examples)} training examples")
            return cached_dataset
            
        except ImportError as e:
            logger.error(f"Required dataset classes not available: {e}")
            return None
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_model(
        self,
        model: Any,
        tokenizer: Any,
        output_name: str,
        base_model_name: str,
        adapter_path: str
    ) -> bool:
        """Save the fine-tuned model with integrated LoRA weights."""
        try:
            # Always save to MLX models directory for consistent loading
            output_dir = Path.home() / ".cortex" / "mlx_models" / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get base model path
            base_model_path = self._get_model_path(base_model_name)
            if not base_model_path or not base_model_path.exists():
                logger.error(f"Base model path not found: {base_model_name}")
                return False
            
            logger.info(f"Saving fine-tuned model to {output_dir}")
            
            # Copy base model files and add adapter
            # Note: mlx_lm doesn't have a save function, the adapter is saved separately by train()
            logger.info(f"Copying base model files from {base_model_path} to {output_dir}")
            for file in base_model_path.glob("*"):
                if file.is_file():
                    shutil.copy2(file, output_dir / file.name)
                elif file.is_dir():
                    shutil.copytree(file, output_dir / file.name, dirs_exist_ok=True)
            
            # Copy adapter files (only the final adapter, not checkpoints)
            adapter_path = Path(adapter_path)
            if adapter_path.exists():
                for adapter_file in adapter_path.glob("*.safetensors"):
                    # Skip checkpoint files (e.g., 0000100_adapters.safetensors)
                    if adapter_file.name.endswith('_adapters.safetensors'):
                        logger.debug(f"Skipping checkpoint: {adapter_file.name}")
                        continue
                    logger.info(f"Copying adapter: {adapter_file.name}")
                    shutil.copy2(adapter_file, output_dir / adapter_file.name)
                
                if (adapter_path / "adapter_config.json").exists():
                    shutil.copy2(adapter_path / "adapter_config.json", output_dir / "adapter_config.json")
            
            # Update config to mark as fine-tuned
            config_path = output_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Add fine-tuning metadata
                config["fine_tuned"] = True
                config["base_model"] = base_model_name
                config["fine_tuning_method"] = "LoRA"
                config["lora_adapter"] = True
                config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Create a marker file for proper detection
            with open(output_dir / "fine_tuned.marker", 'w') as f:
                f.write(f"LoRA fine-tuned version of {base_model_name}\n")
                f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Adapter path: {adapter_path}\n")
                f.write(f"Output directory: {output_dir}\n")
            
            logger.info(f"Fine-tuned model successfully saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cleanup_checkpoints(self, adapter_dir: Path) -> None:
        """
        Clean up training checkpoint files after successful training.
        
        Checkpoints are intermediate saves during training (e.g., 0000100_adapters.safetensors).
        We keep them during training for crash recovery but delete after successful completion.
        
        Args:
            adapter_dir: Directory containing adapter files and checkpoints
        """
        try:
            if not adapter_dir.exists():
                return
            
            checkpoint_files = []
            total_size = 0
            
            # Find all checkpoint files (pattern: NNNNNNN_adapters.safetensors)
            for file in adapter_dir.glob("*_adapters.safetensors"):
                # Check if filename matches checkpoint pattern (digits followed by _adapters.safetensors)
                filename = file.name
                if filename.endswith("_adapters.safetensors"):
                    # Extract the prefix before _adapters
                    prefix = filename[:-len("_adapters.safetensors")]
                    # Check if prefix is all digits (checkpoint pattern)
                    if prefix.isdigit():
                        checkpoint_files.append(file)
                        total_size += file.stat().st_size
            
            if checkpoint_files:
                # Convert size to human-readable format
                size_gb = total_size / (1024 ** 3)
                size_str = f"{size_gb:.2f}GB" if size_gb >= 1 else f"{total_size / (1024 ** 2):.1f}MB"
                
                logger.info(f"Cleaning up {len(checkpoint_files)} training checkpoints ({size_str})")
                
                # Delete checkpoint files
                for checkpoint in checkpoint_files:
                    try:
                        checkpoint.unlink()
                        logger.debug(f"Deleted checkpoint: {checkpoint.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete checkpoint {checkpoint.name}: {e}")
                
                logger.info(f"✓ Freed {size_str} by removing training checkpoints")
                print(f"\033[92m✓\033[0m Cleaned up {len(checkpoint_files)} training checkpoints ({size_str})")
            else:
                logger.debug("No checkpoint files to clean up")
                
        except Exception as e:
            # Don't fail the training if cleanup fails, just log the error
            logger.warning(f"Checkpoint cleanup failed (non-critical): {e}")
            # Still inform user that training succeeded but cleanup had issues
            print(f"\033[93m⚠\033[0m Training succeeded but checkpoint cleanup encountered issues: {e}")
