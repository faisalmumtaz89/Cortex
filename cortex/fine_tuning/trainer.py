"""LoRA training implementation using MLX."""

import logging
import time
import os
import math
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import json
import shutil

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    MLX_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    MLX_AVAILABLE = False
    mx = nn = optim = tree_map = None  # type: ignore
    _MLX_IMPORT_ERROR = exc

# Import MLX LM functions
try:
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train
    from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset
except ImportError:
    # Fallback implementations
    mlx_load = None
    LoRALinear = None
    TrainingArgs = None
    mlx_train = None
    mlx_load_dataset = None

from cortex.model_manager import ModelManager
from cortex.config import Config
from cortex.metal.mlx_accelerator import MLXAccelerator, MLXConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Enhanced configuration for fine-tuning with intelligent defaults."""
    # Core training parameters
    epochs: int = 2
    learning_rate: float = 3e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    # LoRA parameters
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.1
    target_modules: list = None  # Auto-detect if None
    num_lora_layers: int = 16  # Number of layers to apply LoRA to
    
    # Optimization parameters
    optimizer_type: str = "adamw"  # adamw, sgd, adafactor
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: Optional[int] = None  # If None, calculated from warmup_ratio
    warmup_ratio: float = 0.1
    lr_scheduler: str = "linear"  # linear, cosine, constant, polynomial
    
    # Memory and performance
    gradient_checkpointing: bool = False
    quantization_bits: Optional[int] = None  # 4 or 8 bit quantization
    dataloader_num_workers: int = 0
    fp16: bool = True
    bf16: bool = False
    
    # Task-specific settings
    task_type: str = "chat"  # chat, completion, structured
    max_sequence_length: int = 2048
    response_template: Optional[str] = None
    
    # Dataset settings
    train_test_split: float = 0.0  # If > 0, split dataset for validation
    shuffle_dataset: bool = True
    
    # Advanced settings
    seed: int = 42
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    save_steps: int = 500
    early_stopping_patience: Optional[int] = None
    
    # Model-aware settings (populated automatically)
    model_size_category: str = "medium"  # tiny, small, medium, large, xlarge
    estimated_parameters_b: float = 2.0  # Estimated parameters in billions
    auto_configured: bool = False  # Whether config was auto-generated
    configuration_source: str = "manual"  # manual, smart_quick, smart_balanced, smart_quality
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for LoRA
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration settings."""
        if self.learning_rate <= 0 or self.learning_rate > 1:
            return False, f"Invalid learning rate: {self.learning_rate}"
        
        if self.epochs < 1 or self.epochs > 100:
            return False, f"Invalid number of epochs: {self.epochs}"
        
        if self.batch_size < 1 or self.batch_size > 128:
            return False, f"Invalid batch size: {self.batch_size}"
        
        if self.lora_r < 1 or self.lora_r > 256:
            return False, f"Invalid LoRA rank: {self.lora_r}"
        
        if self.quantization_bits and self.quantization_bits not in [4, 8]:
            return False, f"Invalid quantization bits: {self.quantization_bits}. Must be 4 or 8."
        
        return True, "Configuration is valid"


class SmartConfigFactory:
    """Factory for creating intelligent training configurations based on model and data characteristics."""
    
    # Model size categories (parameters in billions)
    MODEL_CATEGORIES = {
        "tiny": (0, 0.5),      # < 500M parameters (e.g., DistilBERT, small GPT-2)
        "small": (0.5, 2),     # 500M-2B (e.g., GPT-2, small Llama)
        "medium": (2, 8),      # 2B-8B (e.g., Gemma-7B, Llama-2-7B)
        "large": (8, 20),      # 8B-20B (e.g., Llama-2-13B, Mistral-7B variants)
        "xlarge": (20, float('inf'))  # 20B+ (e.g., Llama-2-70B, GPT-3.5+)
    }
    
    # Optimal settings by model size category
    CATEGORY_DEFAULTS = {
        "tiny": {
            "learning_rate": 5e-4,  # Higher LR for small models
            "epochs": 5,            # More epochs needed
            "lora_r": 8,            # Lower rank sufficient
            "lora_alpha": 16,
            "batch_size": 4,        # Can handle larger batches
            "gradient_accumulation_steps": 2,
            "warmup_ratio": 0.05,   # Less warmup needed
            "weight_decay": 0.001,  # Less regularization
        },
        "small": {
            "learning_rate": 3e-4,
            "epochs": 4,
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        },
        "medium": {
            "learning_rate": 1e-4,  # Standard settings for most models
            "epochs": 3,
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        },
        "large": {
            "learning_rate": 5e-5,  # Lower LR for stability
            "epochs": 2,
            "lora_r": 32,           # Higher rank for complex models
            "lora_alpha": 64,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "warmup_ratio": 0.15,   # More warmup
            "weight_decay": 0.01,
        },
        "xlarge": {
            "learning_rate": 2e-5,  # Very conservative
            "epochs": 2,
            "lora_r": 64,           # High rank for very large models
            "lora_alpha": 128,
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "warmup_ratio": 0.2,
            "weight_decay": 0.01,
        }
    }
    
    @classmethod
    def categorize_model_size(cls, size_gb: float, model_manager=None, model_path=None) -> Tuple[str, float]:
        """Categorize model based on actual parameters if possible, fallback to size estimation."""
        estimated_params_b = size_gb / 2.0  # Fallback estimation
        
        # Try to get accurate parameter count if model_manager and path are provided
        if model_manager and model_path:
            try:
                from pathlib import Path
                actual_params_b = model_manager.get_model_parameters_smart(Path(model_path))
                if actual_params_b is not None:
                    estimated_params_b = actual_params_b  # Already in billions
                    logger.info(f"Using accurate parameter count: {estimated_params_b:.2f}B parameters")
                else:
                    logger.warning(f"Could not detect parameters, using size estimation: {estimated_params_b:.2f}B")
            except Exception as e:
                logger.warning(f"Parameter detection failed: {e}, using size estimation")
        
        for category, (min_params, max_params) in cls.MODEL_CATEGORIES.items():
            if min_params <= estimated_params_b < max_params:
                return category, estimated_params_b
        
        # Fallback to medium if can't categorize
        return "medium", estimated_params_b
    
    @classmethod
    def analyze_dataset(cls, dataset_path: Path) -> Dict[str, Any]:
        """Analyze dataset to inform training configuration."""
        try:
            examples = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    examples.append(json.loads(line.strip()))
            
            dataset_size = len(examples)
            
            # Analyze content to detect task type
            task_type = "chat"  # Default
            avg_length = 0
            
            if examples:
                sample = examples[0]
                
                # Detect task type from structure
                if 'prompt' in sample and 'response' in sample:
                    task_type = "chat"
                elif 'prompt' in sample and 'completion' in sample:
                    task_type = "completion"
                elif 'text' in sample:
                    task_type = "completion"
                
                # Calculate average text length
                total_chars = 0
                for example in examples[:100]:  # Sample first 100
                    text = ""
                    if 'text' in example:
                        text = example['text']
                    elif 'prompt' in example and 'response' in example:
                        text = example['prompt'] + example['response']
                    elif 'prompt' in example and 'completion' in example:
                        text = example['prompt'] + example['completion']
                    total_chars += len(text)
                
                avg_length = total_chars // min(len(examples), 100)
            
            return {
                "size": dataset_size,
                "task_type": task_type,
                "avg_length": avg_length,
                "size_category": cls._get_dataset_size_category(dataset_size)
            }
        except Exception as e:
            logger.warning(f"Failed to analyze dataset: {e}")
            return {
                "size": 0,
                "task_type": "chat",
                "avg_length": 1000,
                "size_category": "small"
            }
    
    @classmethod
    def _get_dataset_size_category(cls, size: int) -> str:
        """Categorize dataset by size."""
        if size < 50:
            return "tiny"
        elif size < 500:
            return "small"
        elif size < 2000:
            return "medium"
        elif size < 10000:
            return "large"
        else:
            return "xlarge"
    
    @classmethod
    def create_smart_config(
        cls, 
        model_size_gb: float, 
        dataset_path: Path,
        preset: str = "balanced",
        custom_settings: Optional[Dict[str, Any]] = None,
        model_manager = None,
        model_path: Optional[str] = None
    ) -> TrainingConfig:
        """Create an intelligent training configuration."""
        
        # Analyze model with accurate parameter detection
        model_category, estimated_params = cls.categorize_model_size(
            model_size_gb, model_manager, model_path
        )
        
        # Analyze dataset
        dataset_info = cls.analyze_dataset(dataset_path)
        
        # Get base settings for model category
        base_config = cls.CATEGORY_DEFAULTS[model_category].copy()
        
        # Apply preset modifications
        if preset == "quick":
            base_config["epochs"] = max(1, base_config["epochs"] - 1)
            base_config["learning_rate"] *= 1.5  # Faster learning
        elif preset == "quality":
            base_config["epochs"] += 1
            base_config["learning_rate"] *= 0.8  # More conservative
            base_config["lora_r"] = min(64, base_config["lora_r"] * 2)  # Higher rank
        
        # Adjust for dataset size
        dataset_size = dataset_info["size"]
        if dataset_size < 100:  # Small dataset
            base_config["epochs"] = min(base_config["epochs"] + 2, 8)  # More epochs
            base_config["weight_decay"] *= 0.5  # Less regularization
        elif dataset_size > 5000:  # Large dataset
            base_config["epochs"] = max(1, base_config["epochs"] - 1)  # Fewer epochs
        
        # Adjust for sequence length
        if dataset_info["avg_length"] > 2000:
            base_config["gradient_accumulation_steps"] *= 2  # Handle memory
            base_config["max_sequence_length"] = 4096
        
        total_mem_gb = cls._get_total_memory_gb()
        memory_guard_applied = cls._apply_memory_guards(base_config, total_mem_gb)
        
        # Apply custom settings if provided
        if custom_settings:
            base_config.update(custom_settings)
        
        # Create configuration
        config = TrainingConfig(
            # Core parameters
            epochs=base_config["epochs"],
            learning_rate=base_config["learning_rate"],
            batch_size=base_config["batch_size"],
            gradient_accumulation_steps=base_config["gradient_accumulation_steps"],
            
            # LoRA parameters
            lora_r=base_config["lora_r"],
            lora_alpha=base_config["lora_alpha"],
            
            # Optimization
            weight_decay=base_config["weight_decay"],
            warmup_ratio=base_config["warmup_ratio"],
            
            # Task-specific
            task_type=dataset_info["task_type"],
            max_sequence_length=base_config.get("max_sequence_length", 2048),
            
            # Metadata
            model_size_category=model_category,
            estimated_parameters_b=estimated_params,
            auto_configured=True,
            configuration_source=f"smart_{preset}{'_memory_guarded' if memory_guard_applied else ''}"
        )
        
        return config
    
    @classmethod
    def get_preset_configs(cls) -> Dict[str, Dict[str, Any]]:
        """Get preset configuration descriptions."""
        return {
            "quick": {
                "name": "Quick",
                "description": "Fast training with fewer epochs",
                "use_case": "Quick experimentation and testing",
                "time_factor": 0.7
            },
            "balanced": {
                "name": "Balanced", 
                "description": "Optimal balance of speed and quality",
                "use_case": "Most general use cases (recommended)",
                "time_factor": 1.0
            },
            "quality": {
                "name": "Quality",
                "description": "Best results with more training",
                "use_case": "Production models, important tasks",
                "time_factor": 1.5
            }
        }
    
    @classmethod
    def generate_guidance_message(cls, config: TrainingConfig, model_name: str) -> str:
        """Generate helpful guidance message for the user."""
        messages = []
        if config.configuration_source.endswith("memory_guarded"):
            messages.append("Applied memory guard for this machine: capped batch/seq/accum to avoid GPU/UM pressure")
        
        # Model-specific guidance
        if config.model_size_category == "tiny":
            messages.append(f"Detected tiny model ({config.estimated_parameters_b:.1f}B params) - using higher learning rate for better convergence")
        elif config.model_size_category == "small":
            messages.append(f"Detected small model ({config.estimated_parameters_b:.1f}B params) - using optimized settings")
        elif config.model_size_category == "large":
            messages.append(f"Detected large model ({config.estimated_parameters_b:.1f}B params) - using careful settings for stability")
        elif config.model_size_category == "xlarge":
            messages.append(f"Detected very large model ({config.estimated_parameters_b:.1f}B params) - using conservative settings for stability")
        
        # Learning rate guidance
        if config.learning_rate > 1e-4:
            messages.append(f"Using accelerated learning rate ({config.learning_rate:.1e}) - suitable for smaller models")
        elif config.learning_rate < 5e-5:
            messages.append(f"Using conservative learning rate ({config.learning_rate:.1e}) - prevents overfitting in large models")
        
        # LoRA guidance
        if config.lora_r >= 32:
            messages.append(f"Using high LoRA rank ({config.lora_r}) - captures more model complexity")
        elif config.lora_r <= 8:
            messages.append(f"Using low LoRA rank ({config.lora_r}) - efficient for simpler adaptations")
        
        # Epoch guidance
        if config.epochs >= 5:
            messages.append(f"Training for {config.epochs} epochs - extra iterations for small datasets")
        elif config.epochs == 1:
            messages.append(f"Single epoch training - suitable for large datasets")
        
        if not messages:
            messages.append(f"Using optimized settings for {config.model_size_category} model")
        
        return "\n   ".join(messages)

    @staticmethod
    def _get_total_memory_gb() -> Optional[float]:
        """Approximate total unified memory on macOS (used as GPU-visible memory)."""
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            total_bytes = page_size * phys_pages
            return round(total_bytes / (1024**3), 1)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Total memory detection failed: {exc}")
            return None

    @classmethod
    def _apply_memory_guards(cls, cfg: Dict[str, Any], total_mem_gb: Optional[float]) -> bool:
        """
        Downscale aggressive settings on lower-memory Apple Silicon to reduce GPU/UM hangs.
        
        Heuristics:
            - <=16GB: cap seq length to 1024, batch=1, grad_acc<=2
            - <=32GB: cap seq length to 2048, batch<=2, grad_acc<=4
            - Additionally cap effective tokens (batch*grad_acc*max_seq) to avoid runaway memory.
        """
        if not total_mem_gb:
            return False
        
        guard_applied = False
        effective_tokens = lambda c: c["batch_size"] * c["gradient_accumulation_steps"] * c.get("max_sequence_length", 2048)
        
        if total_mem_gb <= 16:
            if cfg["batch_size"] > 1:
                cfg["batch_size"] = 1
                guard_applied = True
            if cfg["gradient_accumulation_steps"] > 2:
                cfg["gradient_accumulation_steps"] = 2
                guard_applied = True
            max_seq = cfg.get("max_sequence_length", 2048)
            if max_seq > 1024:
                cfg["max_sequence_length"] = 1024
                guard_applied = True
            target_tokens = 4096
        elif total_mem_gb <= 32:
            if cfg["batch_size"] > 2:
                cfg["batch_size"] = 2
                guard_applied = True
            if cfg["gradient_accumulation_steps"] > 4:
                cfg["gradient_accumulation_steps"] = 4
                guard_applied = True
            max_seq = cfg.get("max_sequence_length", 2048)
            if max_seq > 2048:
                cfg["max_sequence_length"] = 2048
                guard_applied = True
            target_tokens = 8192
        else:
            target_tokens = 12288  # Leave roomy settings for higher-memory hosts

        # Gradient checkpointing trades compute for memory; enable when guarding.
        if guard_applied and not cfg.get("gradient_checkpointing", False):
            cfg["gradient_checkpointing"] = True
        
        # If the overall token budget is still too high, scale down grad_acc first, then seq length.
        curr_tokens = effective_tokens(cfg)
        if curr_tokens > target_tokens:
            scale = max(1, math.ceil(curr_tokens / target_tokens))
            new_grad_acc = max(1, cfg["gradient_accumulation_steps"] // scale)
            if new_grad_acc < cfg["gradient_accumulation_steps"]:
                cfg["gradient_accumulation_steps"] = new_grad_acc
                guard_applied = True
            curr_tokens = effective_tokens(cfg)
            if curr_tokens > target_tokens:
                new_seq = max(256, cfg.get("max_sequence_length", 2048) // scale)
                if new_seq < cfg.get("max_sequence_length", 2048):
                    cfg["max_sequence_length"] = new_seq
                    guard_applied = True
        
        if guard_applied:
            logger.info(
                f"Memory guard applied (total_mem={total_mem_gb}GB): "
                f"batch={cfg['batch_size']}, grad_acc={cfg['gradient_accumulation_steps']}, "
                f"max_seq={cfg.get('max_sequence_length', 2048)}"
            )
        return guard_applied



class LoRATrainer:
    """Trainer for LoRA fine-tuning using MLX."""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        """Initialize the trainer."""
        self.model_manager = model_manager
        self.config = config
        self.mlx_accelerator = MLXAccelerator(MLXConfig())
        
    def train(
        self,
        base_model_name: str,
        dataset_path: Path,
        output_name: str,
        config: TrainingConfig,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Train a model using LoRA.
        
        Args:
            base_model_name: Name of the base model to fine-tune
            dataset_path: Path to the training dataset
            output_name: Name for the fine-tuned model
            config: Training configuration
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
            logger.info(f"Starting LoRA training: {base_model_name} -> {output_name}")
            
            # Step 1: Load base model
            logger.info("Loading base model...")
            model, tokenizer = self._load_base_model(base_model_name)
            if model is None:
                logger.error("Failed to load base model")
                return False
            
            # Step 2: Apply LoRA layers
            logger.info(f"Applying LoRA with rank={config.lora_r}")
            model = self._apply_lora(model, config)
            
            # Step 3: Load and prepare dataset
            logger.info("Loading dataset...")
            train_dataset = self._load_dataset(dataset_path, tokenizer, config)
            if train_dataset is None:
                logger.error("Failed to load dataset")
                return False
            
            # Step 4: Setup optimizer
            optimizer = self._setup_optimizer(model, config)
            
            # Step 5: Training loop
            logger.info(f"Starting training for {config.epochs} epochs...")
            trained_model = self._training_loop(
                model=model,
                dataset=train_dataset,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                progress_callback=progress_callback
            )
            
            # Step 6: Save fine-tuned model
            logger.info(f"Saving fine-tuned model as {output_name}...")
            success = self._save_model(trained_model, tokenizer, output_name, base_model_name)
            
            if success:
                logger.info(f"Successfully fine-tuned model saved as {output_name}")
                return True
            else:
                logger.error("Failed to save fine-tuned model")
                return False
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def _load_base_model(self, model_name: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load the base model and tokenizer."""
        try:
            # The model should already be loaded by the ModelManager
            # We just need to get it from the cache
            
            # Try all possible cache keys
            possible_keys = [
                model_name,
                self.model_manager.current_model,
                # Sometimes the model is stored with path as key
                str(Path.home() / ".cortex" / "mlx_models" / model_name),
            ]
            
            model = None
            tokenizer = None
            
            for key in possible_keys:
                if key and not model:
                    model = self.model_manager.model_cache.get(key)
                if key and not tokenizer:
                    tokenizer = self.model_manager.tokenizers.get(key)
                    
                if model and tokenizer:
                    logger.info(f"Using loaded model from cache (key: {key})")
                    break
            
            if model and tokenizer:
                return model, tokenizer
            
            # If not in cache, this is unexpected since the wizard confirmed the model is loaded
            logger.error(f"Model {model_name} not found in cache. Available keys: {list(self.model_manager.model_cache.keys())}")
            logger.error(f"Current model: {self.model_manager.current_model}")
            
            # As a fallback, try to load it (but this shouldn't happen)
            logger.warning(f"Attempting to reload model {model_name}")
            
            # First check if it's already an MLX model to avoid re-conversion
            mlx_path = Path.home() / ".cortex" / "mlx_models" / model_name
            if mlx_path.exists():
                # It's already converted, load it directly
                success, message = self.model_manager.load_model(str(mlx_path), model_name=model_name)
            else:
                # Try loading from original location
                success, message = self.model_manager.load_model(model_name)
            
            if not success:
                logger.error(f"Failed to load model: {message}")
                return None, None
            
            # Try to get it from cache again
            model = self.model_manager.model_cache.get(model_name) or self.model_manager.model_cache.get(self.model_manager.current_model)
            tokenizer = self.model_manager.tokenizers.get(model_name) or self.model_manager.tokenizers.get(self.model_manager.current_model)
            
            if not model or not tokenizer:
                logger.error(f"Model or tokenizer still not available after reload")
                return None, None
                
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            return None, None
    
    def _apply_lora(self, model: Any, config: TrainingConfig) -> Any:
        """Apply LoRA layers to the model."""
        if LoRALinear is None:
            # Fallback: Simple LoRA implementation
            logger.warning("mlx_lm LoRA not available, using basic implementation")
            return self._apply_basic_lora(model, config)
        
        # Use mlx_lm's LoRA implementation
        lora_layers = 0
        
        def apply_lora_to_linear(layer):
            nonlocal lora_layers
            if isinstance(layer, nn.Linear):
                # Check if this is a target module
                for target in config.target_modules:
                    if hasattr(layer, '__name__') and target in str(layer.__name__):
                        # Replace with LoRA layer
                        lora_layers += 1
                        return LoRALinear(
                            in_features=layer.weight.shape[1],
                            out_features=layer.weight.shape[0],
                            r=config.lora_r,
                            alpha=config.lora_alpha,
                            dropout=config.lora_dropout
                        )
                return layer
            return layer
        
        # Apply LoRA to all linear layers in target modules
        model = tree_map(apply_lora_to_linear, model)
        logger.info(f"Applied LoRA to {lora_layers} layers")
        
        return model
    
    def _apply_basic_lora(self, model: Any, config: TrainingConfig) -> Any:
        """Apply basic LoRA implementation."""
        class BasicLoRALinear(nn.Module):
            def __init__(self, linear_layer, r=16, alpha=32):
                super().__init__()
                self.linear = linear_layer
                self.r = r
                self.alpha = alpha
                
                # LoRA parameters
                in_features = linear_layer.weight.shape[1]
                out_features = linear_layer.weight.shape[0]
                
                # Low-rank matrices
                self.lora_a = mx.random.normal((r, in_features)) * 0.01
                self.lora_b = mx.zeros((out_features, r))
                
                # Scaling factor
                self.scaling = alpha / r
                
            def __call__(self, x):
                # Original forward pass
                result = self.linear(x)
                
                # Add LoRA contribution
                lora_out = x @ self.lora_a.T @ self.lora_b.T * self.scaling
                
                return result + lora_out
        
        # Apply to target modules
        def apply_basic_lora_to_layer(layer):
            if isinstance(layer, nn.Linear):
                return BasicLoRALinear(layer, r=config.lora_r, alpha=config.lora_alpha)
            return layer
        
        model = tree_map(apply_basic_lora_to_layer, model)
        return model
    
    def _load_dataset(self, dataset_path: Path, tokenizer: Any, config: TrainingConfig) -> Optional[Any]:
        """Load and prepare the dataset."""
        try:
            # Load JSONL dataset
            examples = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    examples.append(data)
            
            # Tokenize examples
            tokenized_examples = []
            max_seq_len = getattr(config, "max_sequence_length", None)
            for example in examples:
                # Format as conversation
                if 'prompt' in example and 'response' in example:
                    text = f"User: {example['prompt']}\nAssistant: {example['response']}"
                elif 'text' in example:
                    text = example['text']
                else:
                    continue
                
                # Tokenize
                tokens = tokenizer.encode(text)
                if max_seq_len and len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                tokenized_examples.append({
                    'input_ids': mx.array(tokens),
                    'labels': mx.array(tokens)  # For causal LM
                })
            
            logger.info(f"Loaded {len(tokenized_examples)} training examples")
            return tokenized_examples
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def _setup_optimizer(self, model: Any, config: TrainingConfig) -> Any:
        """Setup the optimizer."""
        # Get trainable parameters (LoRA parameters only)
        trainable_params = []
        
        def get_lora_params(module, prefix=""):
            # Check for LoRA parameters
            if hasattr(module, 'lora_a'):
                trainable_params.append(module.lora_a)
            if hasattr(module, 'lora_b'):
                trainable_params.append(module.lora_b)
            
            # Try to iterate over child modules
            try:
                # Try vars() first (for regular Python objects)
                children = vars(module).items()
            except TypeError:
                # If vars() doesn't work, try __dict__ directly
                if hasattr(module, '__dict__'):
                    children = module.__dict__.items()
                else:
                    # For MLX modules, try to get children differently
                    children = []
                    if hasattr(module, 'children'):
                        for child in module.children():
                            children.append(('', child))
            
            for name, child in children:
                if isinstance(child, nn.Module):
                    get_lora_params(child, f"{prefix}.{name}")
        
        # Only try to extract LoRA params if model is a Module
        if isinstance(model, nn.Module):
            get_lora_params(model)
        
        if not trainable_params:
            # If no LoRA params found, train all parameters (fallback)
            logger.warning("No LoRA parameters found, training all parameters")
            # For MLX models, we need to get parameters differently
            if hasattr(model, 'parameters'):
                trainable_params = list(model.parameters())
            else:
                logger.error("Model has no parameters() method")
                trainable_params = []
        
        # Create optimizer
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize optimizer state
        optimizer.init(trainable_params)
        
        logger.info(f"Initialized optimizer with {len(trainable_params)} trainable parameters")
        return optimizer
    
    def _training_loop(
        self,
        model: Any,
        dataset: list,
        optimizer: Any,
        config: TrainingConfig,
        tokenizer: Any,
        progress_callback: Optional[Callable] = None
    ) -> Any:
        """Main training loop."""
        model.train()
        
        total_steps = len(dataset) * config.epochs
        current_step = 0
        
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            batch_loss = 0.0
            
            for i, batch in enumerate(dataset):
                # Forward pass
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # Compute loss
                logits = model(input_ids[None, :])  # Add batch dimension
                
                # Cross-entropy loss
                loss = mx.mean(
                    nn.losses.cross_entropy(
                        logits[0, :-1],  # All but last prediction
                        labels[1:],  # All but first token
                        reduction='none'
                    )
                )
                
                # Backward pass
                loss_value, grads = mx.value_and_grad(lambda m: loss)(model)
                
                # Gradient accumulation
                batch_loss += loss_value.item()
                
                if (i + 1) % config.gradient_accumulation_steps == 0:
                    # Update weights
                    optimizer.update(model, grads)
                    
                    # Clear accumulated loss
                    avg_loss = batch_loss / config.gradient_accumulation_steps
                    epoch_loss += avg_loss
                    batch_loss = 0.0
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(epoch, i, avg_loss)
                
                current_step += 1
                
                # Evaluate to ensure computation
                mx.eval(model.parameters())
            
            # Log epoch statistics
            avg_epoch_loss = epoch_loss / (len(dataset) / config.gradient_accumulation_steps)
            logger.info(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_epoch_loss:.4f}")
        
        return model
    
    def _save_model(
        self,
        model: Any,
        tokenizer: Any,
        output_name: str,
        base_model_name: str
    ) -> bool:
        """Save the fine-tuned model."""
        try:
            # Create output directory in MLX models folder for consistency
            output_dir = Path.home() / ".cortex" / "mlx_models" / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model weights
            weights_path = output_dir / "model.safetensors"
            
            # Get model state dict
            state_dict = {}
            
            def extract_weights(module, prefix=""):
                for name, param in vars(module).items():
                    if isinstance(param, mx.array):
                        state_dict[f"{prefix}.{name}"] = param
                    elif isinstance(param, nn.Module):
                        extract_weights(param, f"{prefix}.{name}")
            
            extract_weights(model)
            
            # Save using safetensors format (or numpy for simplicity)
            import numpy as np
            np_state_dict = {k: v.tolist() for k, v in state_dict.items()}
            
            with open(weights_path, 'w') as f:
                json.dump(np_state_dict, f)
            
            # Save tokenizer
            if hasattr(tokenizer, 'save_pretrained'):
                tokenizer.save_pretrained(output_dir)
            
            # Save config
            config_data = {
                "base_model": base_model_name,
                "model_type": "fine-tuned",
                "fine_tuning_method": "LoRA",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(output_dir / "config.json", 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Copy any additional files from base model
            base_model_path = Path.home() / ".cortex" / "models" / base_model_name
            if base_model_path.exists():
                for file in ['tokenizer_config.json', 'special_tokens_map.json', 'vocab.json']:
                    src = base_model_path / file
                    if src.exists():
                        shutil.copy2(src, output_dir / file)
            
            logger.info(f"Model saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
