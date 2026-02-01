"""Model management for GPU-accelerated inference."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import shutil
import struct
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Import MLX LM functions safely
try:
    from mlx_lm import load as mlx_load
except ImportError:
    mlx_load = None

from cortex.config import Config
from cortex.gpu_validator import GPUValidator
from cortex.metal.memory_pool import MemoryPool, AllocationStrategy
from cortex.metal.mlx_converter import MLXConverter, ConversionConfig, QuantizationRecipe, ConversionFormat
from cortex.metal.mlx_accelerator import MLXAccelerator, MLXConfig
from cortex.quantization.dynamic_quantizer import DynamicQuantizer, QuantizationConfig, QuantizationMode

# Configure tokenizer parallelism for optimal performance
# Enable parallelism for better tokenization speed
# This is safe with threading (not multiprocessing)
if os.environ.get('TOKENIZERS_PARALLELISM') is None:
    # Enable tokenizer parallelism for better performance
    # Safe with threading, improves tokenization speed
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Set optimal number of threads for tokenizer
if os.environ.get('RAYON_NUM_THREADS') is None:
    # Use half of available CPU cores for tokenizer threads
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    os.environ['RAYON_NUM_THREADS'] = str(max(1, num_cores // 2))


class ModelFormat(Enum):
    """Supported model formats."""
    GGUF = "gguf"
    MLX = "mlx"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    QUANTIZED = "quantized"  # For GPTQ/AWQ/etc
    UNKNOWN = "unknown"


class QuantizationType(Enum):
    """Supported quantization types."""
    NONE = "none"
    INT4 = "int4"
    INT8 = "int8"
    GPTQ = "gptq"
    AWQ = "awq"
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    FP16 = "FP16"
    FP32 = "FP32"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    path: Path
    format: ModelFormat
    quantization: QuantizationType
    size_bytes: int
    parameters: int
    context_length: int
    loaded_at: datetime
    gpu_memory_used: int
    tokenizer_path: Optional[Path]
    config: Dict[str, Any]
    
    @property
    def size_gb(self) -> float:
        """Get model size in GB."""
        return self.size_bytes / (1024 ** 3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'path': str(self.path),
            'format': self.format.value,
            'quantization': self.quantization.value,
            'size_gb': self.size_gb,
            'parameters': self.parameters,
            'context_length': self.context_length,
            'loaded_at': self.loaded_at.isoformat(),
            'gpu_memory_used': self.gpu_memory_used
        }


class ModelManager:
    """Manage model loading and GPU memory allocation."""
    
    def __init__(
        self,
        config: Config,
        gpu_validator: GPUValidator,
        memory_pool: Optional[MemoryPool] = None
    ):
        """Initialize model manager."""
        self.config = config
        self.gpu_validator = gpu_validator
        self.memory_pool = memory_pool
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.current_model: Optional[str] = None
        self.model_cache: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        # Initialize quantizer for memory-efficient loading
        self.quantizer = DynamicQuantizer(QuantizationConfig(
            mode=QuantizationMode.DYNAMIC,
            per_channel=True,
            cache_quantized=True,
            cache_dir=self.config.model.quantization_cache
        ))
        
        # Initialize MLX converter for native conversion
        # Use a consistent cache directory
        mlx_cache_dir = Path.home() / ".cortex" / "mlx_models"
        self.mlx_converter = MLXConverter(
            cache_dir=mlx_cache_dir
        )
        
        # Initialize MLX accelerator for optimizations
        self.mlx_accelerator = None
        self._mlx_init_error: Optional[str] = None
        try:
            self.mlx_accelerator = MLXAccelerator(MLXConfig(
                compile_model=True,
                use_amx=True,
                fuse_operations=True,
                rotating_kv_cache=True,
                quantization_bits=4
            ))
        except Exception as e:
            self._mlx_init_error = str(e)
            logger.warning("MLX accelerator initialization failed: %s", e, exc_info=True)
        
        self._setup_directories()
        self._initialize_memory_pool()
    
    def __del__(self):
        """Clean up resources on deletion."""
        try:
            # Unload all models properly
            for model_name in list(self.loaded_models.keys()):
                self.unload_model(model_name)
        except:
            pass  # Ignore errors during cleanup
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        self.config.model.model_path.expanduser().mkdir(parents=True, exist_ok=True)
        self.config.model.model_cache_dir.expanduser().mkdir(parents=True, exist_ok=True)
        self.config.model.quantization_cache.expanduser().mkdir(parents=True, exist_ok=True)
    
    def _initialize_memory_pool(self) -> None:
        """Initialize memory pool if needed."""
        # Skip if already provided by InferenceEngine to avoid duplication
        if self.memory_pool is not None:
            return
            
        if self.config.gpu.force_gpu:
            try:
                # Only create if not already provided
                self.memory_pool = MemoryPool(
                    pool_size=None,
                    strategy=AllocationStrategy.UNIFIED,
                    device="mps" if torch.backends.mps.is_available() else "mlx",
                    auto_size=True,
                    silent=True  # Suppress message since InferenceEngine also creates a pool
                )
            except Exception as e:
                # InferenceEngine likely has its own pool; log for visibility.
                logger.debug("Memory pool initialization skipped: %s", e)

    def _prefer_speed_quantization(self) -> bool:
        """Return True when config prefers maximum speed over quality."""
        level = getattr(self.config.gpu, "gpu_optimization_level", "maximum")
        level = str(level).lower().strip()
        return level in {"maximum", "max", "speed", "fast", "performance"}

    def _get_default_quant_recipe(self) -> Optional[QuantizationRecipe]:
        """Map configured default_quantization to an MLX quantization recipe."""
        raw = getattr(self.config.model, "default_quantization", "") or ""
        value = str(raw).strip().lower()
        if not value or value == "auto":
            return None

        mapping = {
            "q4_k_m": QuantizationRecipe.SPEED_4BIT,
            "q5_k_m": QuantizationRecipe.BALANCED_5BIT,
            "q6_k": QuantizationRecipe.QUALITY_8BIT,  # closest available MLX recipe
            "q8_0": QuantizationRecipe.QUALITY_8BIT,
            "4bit": QuantizationRecipe.SPEED_4BIT,
            "5bit": QuantizationRecipe.BALANCED_5BIT,
            "8bit": QuantizationRecipe.QUALITY_8BIT,
            "mixed": QuantizationRecipe.MIXED_PRECISION,
            "none": QuantizationRecipe.NONE,
        }

        recipe = mapping.get(value)
        if recipe is None:
            logger.warning("Unknown default_quantization value: %s", raw)
            return None

        supported = getattr(self.config.model, "supported_quantizations", None)
        if supported:
            supported_norm = {str(s).strip().lower() for s in supported}
            if value.startswith("q") and value not in supported_norm:
                logger.warning("default_quantization '%s' not in supported_quantizations", raw)
                return None

        return recipe
    
    def load_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        force_reload: bool = False,
        convert_to_mlx: bool = False,
        quantization: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Load a model to GPU memory with optional MLX conversion.
        
        Args:
            model_path: Path to model file or directory (or HF repo ID)
            model_name: Optional name for the model
            force_reload: Force reload even if already loaded
            convert_to_mlx: Convert to MLX format for better performance
            quantization: Quantization recipe ('4bit', '5bit', '8bit', 'mixed')
            
        Returns:
            Tuple of (success, message)
        """
        # Check if it's a HuggingFace repo ID
        is_hf_repo = "/" in model_path and not Path(model_path).exists()
        
        # Auto-enable MLX conversion if MLX backend is enabled in config
        if hasattr(self.config, 'gpu') and hasattr(self.config.gpu, 'mlx_backend'):
            if self.config.gpu.mlx_backend:
                logger.info("MLX backend enabled in config, auto-converting models to MLX format")
                convert_to_mlx = True
        
        # Handle MLX conversion for HF models or local models
        if convert_to_mlx or is_hf_repo:
            # Check if this is a cached MLX model name
            if "_4bit" in model_path or "_5bit" in model_path or "_8bit" in model_path or "_none" in model_path:
                # This might be a cached MLX model, check if it exists
                mlx_cache_dir = Path.home() / ".cortex" / "mlx_models"
                cache_path = mlx_cache_dir / Path(model_path).name
                
                if cache_path.exists() and cache_path.is_dir():
                    logger.info(f"Loading cached MLX model from {cache_path}")
                    success, result = self._load_mlx(cache_path, model_name or Path(model_path).name, {
                        'format': ModelFormat.MLX,
                        'quantization': QuantizationType.INT4 if "_4bit" in model_path else QuantizationType.INT8,
                        'reason': 'Cached MLX model'
                    })
                    
                    if success:
                        self.current_model = model_name or Path(model_path).name
                        # When loading from cache, don't update the config - it already has the right path
                        return True, f"Successfully loaded MLX model '{model_name or Path(model_path).name}'"
                    else:
                        return False, f"Failed to load MLX model: {result}"
                else:
                    # Cached model not found, need to reconvert from original
                    # Try to extract original path from the cached name
                    base_name = model_path.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "").replace("_none", "")
                    
                    # Try to find the original model
                    if base_name.startswith("_Users_"):
                        # This is a local model path encoded in the name
                        original_path = "/" + base_name[1:].replace("_", "/")
                        original_path = Path(original_path).expanduser()
                        
                        if original_path.exists():
                            logger.info(f"Found original model at {original_path}, will convert")
                            model_path = str(original_path)
                            # Continue with normal conversion flow
                        else:
                            return False, f"Cached MLX model not found and original model not found at {original_path}"
                    else:
                        return False, f"Cached MLX model not found at {cache_path}"
            
            # Check if model is already in MLX format by looking ahead
            test_path = Path(model_path).expanduser().resolve()
            if test_path.exists() and test_path.is_dir():
                # Check if it's in the mlx_models directory - if so, it's already converted
                mlx_models_dir = Path.home() / ".cortex" / "mlx_models"
                # Use proper path comparison
                try:
                    is_in_mlx_dir = test_path.is_relative_to(mlx_models_dir)
                except (ValueError, AttributeError):
                    # Fallback for older Python versions
                    is_in_mlx_dir = str(mlx_models_dir.resolve()) in str(test_path.resolve())
                
                # Check for MLX format markers - include adapter files for fine-tuned models
                has_mlx_weights = (test_path / 'weights.npz').exists() or (test_path / 'model.safetensors').exists()
                has_config = (test_path / 'config.json').exists()
                has_adapter = (test_path / 'adapter.safetensors').exists()
                has_fine_tuned_marker = (test_path / 'fine_tuned.marker').exists()
                
                # A model is MLX format if:
                # 1. It's in the mlx_models directory, OR
                # 2. It has MLX weights and config, OR  
                # 3. It's a fine-tuned model with adapters
                if is_in_mlx_dir or (has_mlx_weights and has_config) or has_fine_tuned_marker or has_adapter:
                    # Already MLX format, skip conversion
                    logger.info(f"Model at {model_path} is already in MLX format, skipping conversion")
                    path = test_path
                    
                    # Check if it's a fine-tuned model
                    format_info = {
                        'format': ModelFormat.MLX,
                        'quantization': QuantizationType.NONE,  # Will be detected from model
                        'reason': 'Existing MLX model'
                    }
                    
                    # Check config for fine-tuning markers
                    config_path = path / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                if config.get('fine_tuned') or config.get('lora_adapter'):
                                    format_info['is_fine_tuned'] = True
                                    format_info['has_lora_adapter'] = True
                                    logger.info("Detected fine-tuned model with LoRA adapters")
                        except Exception as e:
                            logger.warning(f"Could not read config to check fine-tuning status: {e}")
                    
                    # Check for adapter files
                    if (path / "adapter.safetensors").exists() or (path / "adapter_config.json").exists():
                        format_info['has_lora_adapter'] = True
                        logger.info("Found LoRA adapter files")
                    
                    # Load the existing MLX model directly
                    success, result = self._load_mlx(path, model_name or path.name, format_info)
                    
                    if success:
                        self.current_model = model_name or path.name
                        # Store the original model path
                        self.config.update_last_used_model(str(model_path))
                        return True, f"Successfully loaded MLX model '{model_name or path.name}'"
                    else:
                        return False, f"Failed to load MLX model: {result}"
                else:
                    # Needs conversion
                    # Determine quantization recipe - smart default based on model size
                    # First check model name for size hints
                    model_name_lower = str(test_path).lower()
                    if any(size in model_name_lower for size in ['270m', '350m', '500m']):
                        quant_recipe = QuantizationRecipe.NONE
                        logger.info(f"Very small model detected from name ({test_path.name}), skipping quantization")
                        print(f"Note: Very small model detected ({test_path.name}), skipping quantization for quality")
                    elif any(size in model_name_lower for size in ['1b', '2b', '3b']):
                        quant_recipe = QuantizationRecipe.QUALITY_8BIT
                        logger.info(f"Small model detected from name ({test_path.name}), using 8-bit quantization")
                        print(f"Note: Small model detected ({test_path.name}), using 8-bit quantization")
                    else:
                        # Use smart parameter detection for quantization decisions
                        try:
                            model_size_gb = self._get_model_size(test_path) / (1024**3)
                            
                            # Use accurate parameter detection (returns billions)
                            actual_params_b = self.get_model_parameters_smart(test_path)
                            params_billions = float(actual_params_b) if actual_params_b is not None else (model_size_gb / 2.2)
                            
                            logger.info(f"Model size: {model_size_gb:.2f}GB, parameters: {params_billions:.2f}B")
                            
                            # For very small models like Gemma-270M, avoid quantization
                            if params_billions < 0.5:
                                quant_recipe = QuantizationRecipe.NONE
                                logger.info(f"Very small model detected ({self._format_param_count(params_billions)} params), skipping quantization")
                                print(f"Note: Very small model detected ({self._format_param_count(params_billions)} params), skipping quantization")
                            elif params_billions < 1.0:
                                quant_recipe = QuantizationRecipe.QUALITY_8BIT  # Prefer 8bit for small models
                                logger.info(f"Small model detected ({self._format_param_count(params_billions)} params), using 8-bit quantization")
                                print(f"Note: Small model detected ({self._format_param_count(params_billions)} params), using 8-bit quantization")
                            else:
                                quant_recipe = QuantizationRecipe.SPEED_4BIT  # Default for larger models
                        except Exception as e:
                            logger.warning(f"Could not estimate model parameters: {e}, defaulting to 4-bit")
                            quant_recipe = QuantizationRecipe.SPEED_4BIT  # Fallback

                    default_recipe = self._get_default_quant_recipe()
                    if default_recipe is not None:
                        quant_recipe = default_recipe
                    
                    if quantization:
                        quant_map = {
                            "4bit": QuantizationRecipe.SPEED_4BIT,
                            "5bit": QuantizationRecipe.BALANCED_5BIT,
                            "8bit": QuantizationRecipe.QUALITY_8BIT,
                            "mixed": QuantizationRecipe.MIXED_PRECISION,
                            "none": QuantizationRecipe.NONE
                        }
                        quant_recipe = quant_map.get(quantization, quant_recipe)  # Use smart default if invalid
                    elif self._prefer_speed_quantization() and quant_recipe != QuantizationRecipe.NONE:
                        if quant_recipe != QuantizationRecipe.SPEED_4BIT:
                            logger.info("Max optimization enabled, using 4-bit quantization for best tokens/sec")
                            print("Note: Max optimization enabled, using 4-bit quantization for best tokens/sec")
                        quant_recipe = QuantizationRecipe.SPEED_4BIT
                    
                    # Determine source format
                    source_format = ConversionFormat.HUGGINGFACE if is_hf_repo else ConversionFormat.SAFETENSORS
                    
                    # For local SafeTensors models, ensure correct format detection
                    if not is_hf_repo and test_path.exists():
                        if any(f.suffix == '.safetensors' for f in test_path.glob('*.safetensors')):
                            source_format = ConversionFormat.SAFETENSORS
                            logger.info("Detected SafeTensors format for conversion to MLX")
                    
                    # Convert to MLX
                    conversion_config = ConversionConfig(
                        source_format=source_format,
                        quantization=quant_recipe,
                        use_amx=True,
                        compile_model=True
                    )
                    
                    logger.info(f"Converting model to MLX format with {quant_recipe.name} quantization...")
                    print(f"Converting model to MLX format for optimal performance...")
                    
                    success, msg, mlx_path = self.mlx_converter.convert_model(
                        model_path,
                        output_name=model_name,
                        config=conversion_config
                    )
                    
                    if not success:
                        return False, f"MLX conversion failed: {msg}"
                    
                    # Update path to converted model
                    path = mlx_path
                    print(f"✓ Model converted to MLX format at {mlx_path}")
                    logger.info(f"Successfully converted model to MLX format at {mlx_path}")
                    
                    # Now load the converted MLX model directly
                    success, result = self._load_mlx(path, model_name or path.name, {
                        'format': ModelFormat.MLX,
                        'quantization': self._get_quantization_type_from_recipe(quant_recipe),
                        'reason': 'MLX converted model'
                    })
                    
                    if success:
                        self.current_model = model_name or path.name
                        # Store the original model path
                        self.config.update_last_used_model(str(model_path))
                        return True, f"Successfully loaded MLX-converted model '{model_name or path.name}'"
                    else:
                        return False, f"Failed to load converted MLX model: {result}"
            else:
                # HuggingFace repo or non-existent path - needs conversion
                # Determine quantization recipe - smart default for HF models too
                # For HF models, we don't know the exact size, but we can use heuristics
                model_name_lower = model_path.lower()
                if any(size in model_name_lower for size in ['270m', '350m', '500m']):
                    quant_recipe = QuantizationRecipe.NONE  # Very small models
                    logger.info(f"Very small model detected from name ({model_path}), skipping quantization")
                elif any(size in model_name_lower for size in ['1b', '2b', '3b']):
                    quant_recipe = QuantizationRecipe.QUALITY_8BIT  # Small models
                    logger.info(f"Small model detected from name ({model_path}), using 8-bit quantization")
                else:
                    quant_recipe = QuantizationRecipe.SPEED_4BIT  # Default for larger models
                
                default_recipe = self._get_default_quant_recipe()
                if default_recipe is not None:
                    quant_recipe = default_recipe
                
                if quantization:
                    quant_map = {
                        "4bit": QuantizationRecipe.SPEED_4BIT,
                        "5bit": QuantizationRecipe.BALANCED_5BIT,
                        "8bit": QuantizationRecipe.QUALITY_8BIT,
                        "mixed": QuantizationRecipe.MIXED_PRECISION,
                        "none": QuantizationRecipe.NONE
                    }
                    quant_recipe = quant_map.get(quantization, quant_recipe)
                elif self._prefer_speed_quantization() and quant_recipe != QuantizationRecipe.NONE:
                    if quant_recipe != QuantizationRecipe.SPEED_4BIT:
                        logger.info("Max optimization enabled, using 4-bit quantization for best tokens/sec")
                        print("Note: Max optimization enabled, using 4-bit quantization for best tokens/sec")
                    quant_recipe = QuantizationRecipe.SPEED_4BIT
                
                # Convert to MLX
                conversion_config = ConversionConfig(
                    source_format=ConversionFormat.HUGGINGFACE if is_hf_repo else ConversionFormat.SAFETENSORS,
                    quantization=quant_recipe,
                    use_amx=True,
                    compile_model=True
                )
                
                logger.info(f"Converting HF model to MLX format with {quant_recipe.name} quantization...")
                print(f"Downloading and converting model to MLX format...")
                
                success, msg, mlx_path = self.mlx_converter.convert_model(
                    model_path,
                    output_name=model_name,
                    config=conversion_config
                )
                
                if not success:
                    return False, f"MLX conversion failed: {msg}"
                
                # Update path to converted model
                path = mlx_path
                print(f"✓ Model converted to MLX format at {mlx_path}")
                logger.info(f"Successfully converted model to MLX format at {mlx_path}")
                
                # Now load the converted MLX model directly
                success, result = self._load_mlx(path, model_name or path.name, {
                    'format': ModelFormat.MLX,
                    'quantization': self._get_quantization_type_from_recipe(quant_recipe),
                    'reason': 'MLX converted model'
                })
                
                if success:
                    self.current_model = model_name or path.name
                    # Store the original model path
                    self.config.update_last_used_model(str(model_path))
                    return True, f"Successfully loaded MLX-converted model '{model_name or path.name}'"
                else:
                    return False, f"Failed to load converted MLX model: {result}"
        else:
            # Validate inputs - properly expand home directory
            path = Path(model_path).expanduser().resolve()
            if not path.exists():
                # Try mlx-community models
                if model_path.startswith("mlx-community/"):
                    # Download from HuggingFace mlx-community
                    success, msg, mlx_path = self.mlx_converter.convert_model(
                        model_path,
                        output_name=model_name,
                        config=ConversionConfig(quantization=QuantizationRecipe.NONE)
                    )
                    if success:
                        path = mlx_path
                    else:
                        return False, f"Failed to download MLX model: {msg}"
                else:
                    return False, f"Model path does not exist: {model_path}"
        
        # Use the full name for directories, stem for files
        if path.is_dir():
            model_name = model_name or path.name
        else:
            model_name = model_name or path.stem
        
        # Check if already loaded
        if model_name in self.loaded_models and not force_reload:
            self.current_model = model_name
            # Still update last used model even if already loaded
            self.config.update_last_used_model(model_name)
            return True, f"Model '{model_name}' already loaded"
        
        # Check memory constraints
        if len(self.loaded_models) >= self.config.model.max_loaded_models:
            oldest = min(self.loaded_models.items(), key=lambda x: x[1].loaded_at)[0]
            self.unload_model(oldest)
        
        # Detect format and load
        format_info = self._detect_format(path)
        if format_info['format'] == ModelFormat.UNKNOWN:
            return False, f"Unknown model format: {format_info['reason']}"
        
        # Check GPU compatibility and determine if quantization is needed
        model_size_gb = self._get_model_size(path) / (1024**3)
        can_load, message = self.gpu_validator.verify_model_compatibility(model_size_gb)
        
        # Determine if we need quantization (only for non-quantized models)
        needs_quantization = False
        quantization_mode = None
        
        # Only apply dynamic quantization to non-quantized SafeTensors/PyTorch models
        can_apply_quantization = (
            format_info['format'] in [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH] and
            format_info['quantization'] == QuantizationType.NONE
        )
        
        if not can_load and can_apply_quantization:
            if not getattr(self.config.model, "auto_quantize", True):
                return False, f"GPU incompatible: {message} (auto_quantize disabled)"
            # Check if quantization would help
            gpu_status = self.gpu_validator.get_gpu_memory_status()
            available_gb = gpu_status['available_gb']
            
            # DEBUG: Uncomment to see memory calculations for quantization decisions
            # print(f"DEBUG: Model size on disk: {model_size_gb:.1f}GB, Available memory: {available_gb:.1f}GB")
            
            # Estimate if INT8 quantization would fit
            # Use same 3.5x multiplier as gpu_validator for consistency
            # INT8 is more stable than INT4, so prefer it when possible
            estimated_int8_size = model_size_gb * 0.5 * 2.5  # 50% reduction + 150% overhead (less conservative for INT8)
            
            # DEBUG: Uncomment to see quantization size estimates
            # print(f"DEBUG: INT8 estimated size: {estimated_int8_size:.1f}GB")
            
            if estimated_int8_size <= available_gb:
                needs_quantization = True
                quantization_mode = 'int8'
                required_with_overhead = model_size_gb * 3.5
                print(f"Model requires {required_with_overhead:.1f}GB (including overhead), only {available_gb:.1f}GB available.")
                print(f"Will apply INT8 quantization to reduce to ~{estimated_int8_size:.1f}GB")
            else:
                # Try INT4 as last resort
                estimated_int4_size = model_size_gb * 0.25 * 3.5  # 75% reduction + 250% overhead
                
                # DEBUG: Uncomment to see INT4 quantization estimates
                # print(f"DEBUG: INT4 estimated size: {estimated_int4_size:.1f}GB")
                
                if estimated_int4_size <= available_gb:
                    needs_quantization = True
                    quantization_mode = 'int4'
                    required_with_overhead = model_size_gb * 3.5
                    print(f"Model requires {required_with_overhead:.1f}GB (including overhead), only {available_gb:.1f}GB available.")
                    print(f"Will apply INT4 quantization to reduce to ~{estimated_int4_size:.1f}GB")
                else:
                    return False, f"Model too large even with quantization: {message}"
        elif not can_load:
            # Can't apply quantization to this format
            return False, f"GPU incompatible: {message}"
        
        # Load based on format
        loader_map = {
            ModelFormat.MLX: self._load_mlx,
            ModelFormat.GGUF: self._load_gguf,
            ModelFormat.SAFETENSORS: self._load_safetensors,
            ModelFormat.PYTORCH: self._load_pytorch,
            ModelFormat.QUANTIZED: self._load_quantized
        }
        
        loader = loader_map.get(format_info['format'])
        if not loader:
            return False, f"No loader for format: {format_info['format'].value}"
        
        try:
            # Pass quantization info to loaders that support it
            if format_info['format'] in [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH]:
                success, result = loader(path, model_name, format_info, needs_quantization, quantization_mode)
            else:
                success, result = loader(path, model_name, format_info)
            
            if success:
                self.current_model = model_name
                # Save the last used model to config
                self.config.update_last_used_model(model_name)
                return True, f"Successfully loaded '{model_name}'"
            else:
                return False, result
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def _detect_format(self, path: Path) -> Dict[str, Any]:
        """
        Detect model format from path.
        
        Returns:
            Dict with 'format', 'quantization', and 'reason'
        """
        # Check for specific file types
        if path.is_file():
            if path.suffix.lower() == '.gguf':
                return {
                    'format': ModelFormat.GGUF,
                    'quantization': QuantizationType.Q4_K_M,
                    'reason': 'GGUF file'
                }
        
        # Check for directory-based formats
        if path.is_dir():
            # Check for MLX format - support both regular MLX and fine-tuned MLX models
            has_config = (path / 'config.json').exists()
            has_weights_npz = (path / 'weights.npz').exists()
            has_safetensors = any(path.glob('*.safetensors'))
            has_fine_tuned_marker = (path / 'fine_tuned.marker').exists()
            
            if has_config and (has_weights_npz or has_safetensors):
                # Detect if it's a fine-tuned model with LoRA adapters
                has_adapter = (path / 'adapter.safetensors').exists()
                
                return {
                    'format': ModelFormat.MLX,
                    'quantization': QuantizationType.NONE,
                    'reason': 'MLX model with LoRA adapters' if has_adapter else 'MLX model (weights + config)',
                    'has_lora_adapter': has_adapter,
                    'is_fine_tuned': has_fine_tuned_marker
                }
            
            # Check for SafeTensors format
            safetensor_files = list(path.glob('*.safetensors'))
            if safetensor_files and (path / 'config.json').exists():
                # Check if it's quantized by looking at the config
                quantization = self._detect_quantization(path)
                if quantization in [QuantizationType.GPTQ, QuantizationType.AWQ, QuantizationType.INT4, QuantizationType.INT8]:
                    return {
                        'format': ModelFormat.QUANTIZED,
                        'quantization': quantization,
                        'reason': f'Quantized model ({quantization.value})'
                    }
                else:
                    return {
                        'format': ModelFormat.SAFETENSORS,
                        'quantization': quantization,
                        'reason': 'SafeTensors model'
                    }
            
            # Check for PyTorch format
            if (path / 'pytorch_model.bin').exists() or list(path.glob('pytorch_model*.bin')):
                return {
                    'format': ModelFormat.PYTORCH,
                    'quantization': QuantizationType.NONE,
                    'reason': 'PyTorch model'
                }
        
        return {
            'format': ModelFormat.UNKNOWN,
            'quantization': QuantizationType.NONE,
            'reason': 'No recognized model files found'
        }
    
    def _detect_quantization(self, path: Path) -> QuantizationType:
        """Detect quantization type from model files."""
        # Check config.json for quantization info
        config_path = path / 'config.json'
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                
                # Check for quantization config
                if 'quantization_config' in config:
                    quant_config = config['quantization_config']
                    if 'quant_method' in quant_config:
                        method = quant_config['quant_method'].upper()
                        if 'GPTQ' in method:
                            return QuantizationType.GPTQ
                        elif 'AWQ' in method:
                            return QuantizationType.AWQ
                    if 'bits' in quant_config:
                        bits = quant_config['bits']
                        if bits == 4:
                            return QuantizationType.INT4
                        elif bits == 8:
                            return QuantizationType.INT8
                
                # Check model name for hints (be careful with model size indicators like 4B = 4 billion)
                model_name = str(path.name).upper()
                # Only detect as quantized if explicitly mentioned
                if '4BIT' in model_name or 'INT4' in model_name or 'GPTQ-4' in model_name:
                    return QuantizationType.INT4
                elif '8BIT' in model_name or 'INT8' in model_name or 'GPTQ-8' in model_name:
                    return QuantizationType.INT8
                elif 'GPTQ' in model_name:
                    return QuantizationType.GPTQ
                elif 'AWQ' in model_name:
                    return QuantizationType.AWQ
            except:
                pass
        
        # Check for quantization-specific files
        safetensor_files = list(path.glob('*.safetensors'))
        if safetensor_files:
            # Load one file to check for quantization tensors
            try:
                from safetensors.torch import load_file
                sample = load_file(safetensor_files[0], device='cpu')
                # Check for GPTQ/AWQ specific tensors
                has_scales = any('.scales' in k for k in sample.keys())
                has_qweight = any('.qweight' in k for k in sample.keys())
                
                if has_scales or has_qweight:
                    return QuantizationType.GPTQ
            except:
                pass
        
        return QuantizationType.NONE
    
    def _get_quantization_type_from_recipe(self, recipe: QuantizationRecipe) -> QuantizationType:
        """Convert MLX quantization recipe to QuantizationType."""
        recipe_to_type = {
            QuantizationRecipe.SPEED_4BIT: QuantizationType.INT4,
            QuantizationRecipe.BALANCED_5BIT: QuantizationType.INT4,  # Closest match
            QuantizationRecipe.QUALITY_8BIT: QuantizationType.INT8,
            QuantizationRecipe.MIXED_PRECISION: QuantizationType.INT4,
            QuantizationRecipe.NONE: QuantizationType.NONE
        }
        return recipe_to_type.get(recipe, QuantizationType.INT4)

    def _format_param_count(self, params_b: Optional[float]) -> str:
        """Format a parameter count in billions as a human-readable string (M/B/T)."""
        try:
            if params_b is None:
                return "unknown"
            # Trillions
            if params_b >= 1000:
                return f"{params_b / 1000:.1f}T"
            # Billions
            if params_b >= 1:
                return f"{params_b:.1f}B"
            # Millions (10M - 999M)
            if params_b >= 0.01:
                return f"{params_b * 1000:.0f}M"
            # Low millions (1M - 9.9M)
            if params_b >= 0.001:
                return f"{params_b * 1000:.1f}M"
            # Thousands
            if params_b > 0:
                return f"{params_b * 1e6:.0f}K"
            return "0"
        except Exception:
            # Fallback formatting
            try:
                return f"{float(params_b):.2f}B"
            except Exception:
                return "unknown"
    
    def _get_model_size(self, path: Path) -> int:
        """Get total size of model files in bytes."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            # Check if this is a fine-tuned model with LoRA adapters
            is_finetuned = (path / 'fine_tuned.marker').exists() or (path / 'adapter.safetensors').exists()
            
            for file in path.rglob('*'):
                if file.is_file():
                    # Skip training checkpoint files for fine-tuned models
                    if is_finetuned and file.name.endswith('_adapters.safetensors'):
                        # These are intermediate checkpoints, not needed for inference
                        continue
                    # Skip cache and git files
                    if '/.cache/' in str(file) or '/.git/' in str(file):
                        continue
                    total += file.stat().st_size
            return total
        return 0
    
    def _load_mlx(self, path: Path, model_name: str, format_info: Dict) -> Tuple[bool, str]:
        """Load MLX format model with optimizations and LoRA adapter support."""
        try:
            if mlx_load is None:
                return False, "MLX LM library not available. Install with: pip install mlx-lm"
            
            # Check if this is a fine-tuned model with LoRA adapters
            has_adapter = format_info.get('has_lora_adapter', False)
            is_fine_tuned = format_info.get('is_fine_tuned', False)
            
            if has_adapter or is_fine_tuned:
                logger.info(f"Loading fine-tuned MLX model with LoRA adapters: {model_name}")
                
                # For fine-tuned models, we need to load the base model and apply adapters
                try:
                    # Try to load with adapter integration
                    from mlx_lm.tuner.utils import apply_lora_layers
                    
                    # Load the model (should include merged weights)
                    model, tokenizer = mlx_load(str(path))
                    
                    # The model should already have adapters merged since we saved it that way
                    logger.info("Fine-tuned MLX model loaded with integrated LoRA weights")
                    
                except ImportError:
                    # Fallback to regular loading
                    logger.warning("MLX LoRA utilities not available, loading as regular MLX model")
                    model, tokenizer = mlx_load(str(path))
            else:
                # Regular MLX model loading
                model, tokenizer = mlx_load(str(path))
            
            # Apply MLX accelerator optimizations if available
            if self.mlx_accelerator:
                # Silently apply optimizations - details shown in CLI
                logger.info("Applying MLX optimizations (AMX, operation fusion)...")
                model = self.mlx_accelerator.optimize_model(model)
                
                # MLX LM models are already quantized during conversion
                # No need to apply additional quantization
                logger.info("MLX model already optimized with quantization")
            
            # Evaluate parameters if they exist
            if hasattr(model, 'parameters'):
                mx.eval(model.parameters())
            
            self.model_cache[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Load config
            config = {}
            config_path = path / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            
            # Create model info with accurate parameter detection
            parameters = self.get_model_parameters_smart(path)
            
            model_info = ModelInfo(
                name=model_name,
                path=path,
                format=ModelFormat.MLX,
                quantization=format_info['quantization'],
                size_bytes=self._get_model_size(path),
                parameters=parameters,
                context_length=config.get('max_position_embeddings', 4096),
                loaded_at=datetime.now(),
                gpu_memory_used=self._estimate_mlx_memory(model),
                tokenizer_path=path / 'tokenizer.json',
                config=config
            )
            
            self.loaded_models[model_name] = model_info
            return True, "MLX model loaded successfully"
            
        except Exception as e:
            return False, f"Failed to load MLX model: {str(e)}"
    
    def _load_gguf(self, path: Path, model_name: str, format_info: Dict) -> Tuple[bool, str]:
        """Load GGUF format model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            print("Loading GGUF model with llama.cpp...")
            
            # Determine optimal parameters for Apple Silicon
            n_gpu_layers = -1  # Use all layers on GPU
            n_ctx = 4096  # Context size
            n_batch = 512  # Batch size for prompt processing
            
            # Load the model
            model = Llama(
                model_path=str(path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=8,  # Use 8 threads for CPU operations
                use_mlock=True,  # Lock model in RAM
                verbose=False
            )
            
            # Create a simple tokenizer wrapper for compatibility
            class GGUFTokenizer:
                def __init__(self, model):
                    self.model = model
                    self.pad_token = None
                    self.eos_token = None
                
                def encode(self, text):
                    return self.model.tokenize(text.encode('utf-8'))
                
                def decode(self, tokens):
                    return self.model.detokenize(tokens).decode('utf-8')
            
            tokenizer = GGUFTokenizer(model)
            
            self.model_cache[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Create model info
            # Get model parameters from the model object if available
            try:
                # Try to get parameters from model metadata
                n_params = getattr(model, 'n_params', 0)
                if n_params == 0:
                    # Estimate based on model size
                    n_params = self._get_model_size(path) // 2  # Rough estimate
            except:
                n_params = self._get_model_size(path) // 2
            
            model_info = ModelInfo(
                name=model_name,
                path=path,
                format=ModelFormat.GGUF,
                quantization=format_info['quantization'],
                size_bytes=self._get_model_size(path),
                parameters=n_params,
                context_length=n_ctx,
                loaded_at=datetime.now(),
                gpu_memory_used=self._get_model_size(path),  # GGUF loads full model
                tokenizer_path=None,
                config={'n_ctx': n_ctx, 'n_gpu_layers': n_gpu_layers}
            )
            
            self.loaded_models[model_name] = model_info
            return True, "GGUF model loaded successfully"
            
        except ImportError:
            return False, "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        except Exception as e:
            return False, f"Failed to load GGUF model: {str(e)}"
    
    def _load_safetensors(
        self, 
        path: Path, 
        model_name: str, 
        format_info: Dict,
        needs_quantization: bool = False,
        quantization_mode: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Load standard SafeTensors model with optional quantization."""
        try:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            # Check for cached quantized model first BEFORE loading anything
            if needs_quantization and self.quantizer.config.cache_quantized:
                cached = self.quantizer.load_cached_model(path, self.quantizer.config)
                if cached:
                    print(f"Loading cached quantized model...")
                    # Load to CPU first with minimal memory usage
                    print(f"Creating model structure...")
                    with torch.device('cpu'):
                        model = AutoModelForCausalLM.from_pretrained(
                            str(path),
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map={'': 'cpu'}  # Force CPU loading
                        )
                    
                    print(f"Applying cached quantized weights...")
                    model.load_state_dict(cached[0])
                    
                    # Now move to target device
                    print(f"Moving to {device}...")
                    model = model.to(device)
                    quantization_info = cached[1]
                    print(f"Quantized model loaded from cache")
                else:
                    # Load and quantize
                    print(f"Loading model for quantization...")
                    model = AutoModelForCausalLM.from_pretrained(
                        str(path),
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    
                    if needs_quantization:
                        print(f"Applying {quantization_mode} quantization...")
                        gpu_status = self.gpu_validator.get_gpu_memory_status()
                        model, quantization_info = self.quantizer.quantize_model(
                            model,
                            target_dtype=quantization_mode,
                            available_memory_gb=gpu_status['available_gb'],
                            model_size_gb=self._get_model_size(path) / (1024**3),
                            target_device=device  # Pass the target device (MPS)
                        )
                        
                        # Cache the quantized model
                        if self.quantizer.config.cache_quantized:
                            cache_path = self.quantizer.cache_quantized_model(model, path, quantization_info)
                            print(f"Cached quantized model for faster future loads")
                    
                    model = model.to(device)
            else:
                # Normal loading without quantization
                model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                model = model.to(device)
                model.eval()  # Set model to evaluation mode
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.model_cache[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Load config
            config = {}
            config_path = path / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            
            # Create model info with quantization details if applicable
            if needs_quantization:
                # Update quantization type based on what was applied
                actual_quantization = QuantizationType.INT8 if quantization_mode == 'int8' else QuantizationType.INT4
                # Calculate actual memory used after quantization
                memory_used = sum(
                    p.numel() * 1 if hasattr(p, 'numel') else 0  # Quantized uses less bytes per element
                    for p in model.parameters()
                )
            else:
                actual_quantization = format_info['quantization']
                memory_used = sum(p.element_size() * p.numel() for p in model.parameters())
            
            # Use smart parameter detection instead of counting loaded model parameters
            parameters = self.get_model_parameters_smart(path)
            
            model_info = ModelInfo(
                name=model_name,
                path=path,
                format=ModelFormat.SAFETENSORS,
                quantization=actual_quantization,
                size_bytes=self._get_model_size(path),
                parameters=parameters,
                context_length=config.get('max_position_embeddings', 4096),
                loaded_at=datetime.now(),
                gpu_memory_used=memory_used,
                tokenizer_path=path / 'tokenizer.json',
                config=config
            )
            
            self.loaded_models[model_name] = model_info
            return True, "SafeTensors model loaded successfully"
            
        except Exception as e:
            return False, f"Failed to load SafeTensors model: {str(e)}"
    
    def _load_pytorch(
        self,
        path: Path,
        model_name: str,
        format_info: Dict,
        needs_quantization: bool = False,
        quantization_mode: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Load PyTorch format model with optional quantization."""
        try:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            # Check for cached quantized model first
            if needs_quantization and self.quantizer.config.cache_quantized:
                cached = self.quantizer.load_cached_model(path, self.quantizer.config)
                if cached:
                    print(f"Loading cached quantized model...")
                    model = AutoModelForCausalLM.from_pretrained(
                        str(path),
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    model.load_state_dict(cached[0])
                    model = model.to(device)
                    quantization_info = cached[1]
                else:
                    # Load and quantize
                    print(f"Loading model for quantization...")
                    model = AutoModelForCausalLM.from_pretrained(
                        str(path),
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    
                    if needs_quantization:
                        print(f"Applying {quantization_mode} quantization...")
                        gpu_status = self.gpu_validator.get_gpu_memory_status()
                        model, quantization_info = self.quantizer.quantize_model(
                            model,
                            target_dtype=quantization_mode,
                            available_memory_gb=gpu_status['available_gb'],
                            model_size_gb=self._get_model_size(path) / (1024**3),
                            target_device=device  # Pass the target device (MPS)
                        )
                        
                        # Cache the quantized model
                        if self.quantizer.config.cache_quantized:
                            cache_path = self.quantizer.cache_quantized_model(model, path, quantization_info)
                            print(f"Cached quantized model for faster future loads")
                    
                    model = model.to(device)
            else:
                # Normal loading without quantization
                model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                model = model.to(device)
                model.eval()  # Set model to evaluation mode
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.model_cache[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Get config
            config = model.config.to_dict() if hasattr(model, 'config') else {}
            
            # Create model info with quantization details if applicable
            if needs_quantization:
                # Update quantization type based on what was applied
                actual_quantization = QuantizationType.INT8 if quantization_mode == 'int8' else QuantizationType.INT4
                # Calculate actual memory used after quantization
                memory_used = sum(
                    p.numel() * 1 if hasattr(p, 'numel') else 0  # Quantized uses less bytes per element
                    for p in model.parameters()
                )
            else:
                actual_quantization = format_info['quantization']
                memory_used = sum(p.element_size() * p.numel() for p in model.parameters())
            
            # Use smart parameter detection instead of counting loaded model parameters
            parameters = self.get_model_parameters_smart(path)
            
            model_info = ModelInfo(
                name=model_name,
                path=path,
                format=ModelFormat.PYTORCH,
                quantization=actual_quantization,
                size_bytes=self._get_model_size(path),
                parameters=parameters,
                context_length=config.get('max_position_embeddings', 4096),
                loaded_at=datetime.now(),
                gpu_memory_used=memory_used,
                tokenizer_path=None,
                config=config
            )
            
            self.loaded_models[model_name] = model_info
            return True, "PyTorch model loaded successfully"
            
        except Exception as e:
            return False, f"Failed to load PyTorch model: {str(e)}"
    
    def _load_quantized(self, path: Path, model_name: str, format_info: Dict) -> Tuple[bool, str]:
        """Load quantized model (GPTQ/AWQ/etc) using appropriate libraries."""
        quant_type = format_info['quantization']
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Loading {quant_type.value} quantized model...")
        
        # Try different quantization libraries based on detected type
        if quant_type in [QuantizationType.GPTQ, QuantizationType.INT4]:
            # Try GPTQ loader first
            try:
                from auto_gptq import AutoGPTQForCausalLM
                
                model = AutoGPTQForCausalLM.from_quantized(
                    str(path),
                    device="cuda:0" if torch.cuda.is_available() else "cpu",  # GPTQ doesn't support MPS directly
                    use_safetensors=True,
                    trust_remote_code=True,
                    inject_fused_attention=False,  # Disable for compatibility
                    inject_fused_mlp=False
                )
                
                # Move to MPS if needed
                if device.type == "mps" and not torch.cuda.is_available():
                    model = model.cpu()  # GPTQ models may need CPU fallback on Mac
                    print("Note: GPTQ model loaded on CPU (MPS not fully supported)")
                
                tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self.model_cache[model_name] = model
                self.tokenizers[model_name] = tokenizer
                
                config = self._load_config(path)
                model_info = self._create_model_info(
                    model_name, path, ModelFormat.QUANTIZED, quant_type,
                    model, config
                )
                
                self.loaded_models[model_name] = model_info
                return True, f"GPTQ quantized model loaded successfully"
                
            except ImportError:
                print("GPTQ library not available, trying alternative methods...")
            except Exception as e:
                print(f"GPTQ loading failed: {str(e)[:100]}")
        
        if quant_type == QuantizationType.AWQ:
            # Try AWQ loader
            try:
                from awq import AutoAWQForCausalLM
                
                model = AutoAWQForCausalLM.from_quantized(
                    str(path),
                    fuse_layers=False,  # Disable for compatibility
                    trust_remote_code=True
                )
                
                tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self.model_cache[model_name] = model
                self.tokenizers[model_name] = tokenizer
                
                config = self._load_config(path)
                model_info = self._create_model_info(
                    model_name, path, ModelFormat.QUANTIZED, quant_type,
                    model, config
                )
                
                self.loaded_models[model_name] = model_info
                return True, f"AWQ quantized model loaded successfully"
                
            except ImportError:
                print("AWQ library not available, trying alternative methods...")
            except Exception as e:
                print(f"AWQ loading failed: {str(e)[:100]}")
        
        # Try using accelerate for general quantized models
        try:
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            from transformers import AutoConfig
            
            print("Attempting to load with accelerate library...")
            
            # Load config first
            config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
            
            # Initialize model with empty weights
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(
                    config,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            # Determine the checkpoint files
            checkpoint_files = list(path.glob("*.safetensors"))
            if not checkpoint_files:
                checkpoint_files = list(path.glob("pytorch_model*.bin"))
            
            if not checkpoint_files:
                raise ValueError("No model files found")
            
            # Create proper device map for MPS
            if device.type == "mps":
                device_map = {"": "cpu"}  # Load to CPU first for MPS compatibility
            else:
                device_map = "auto"
            
            # Load and dispatch to device
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=str(path),  # Directory containing the model files
                device_map=device_map,
                dtype=torch.float16,
                offload_folder=str(self.config.model.model_cache_dir / "offload")
            )
            
            # Move to MPS if needed
            if device.type == "mps" and device_map == {"": "cpu"}:
                model = model.to(device)
            
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.model_cache[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            config_dict = self._load_config(path)
            model_info = self._create_model_info(
                model_name, path, ModelFormat.QUANTIZED, quant_type,
                model, config_dict
            )
            
            self.loaded_models[model_name] = model_info
            return True, f"Quantized model loaded with accelerate"
            
        except Exception as e:
            print(f"Accelerate loading failed: {str(e)[:100]}")
        
        # Try bitsandbytes for 4-bit/8-bit models
        try:
            from transformers import BitsAndBytesConfig
            
            print("Attempting to load with bitsandbytes quantization...")
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True if quant_type == QuantizationType.INT4 else False,
                load_in_8bit=True if quant_type == QuantizationType.INT8 else False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                str(path),
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else {"": device}
            )
            
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.model_cache[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            config = self._load_config(path)
            model_info = self._create_model_info(
                model_name, path, ModelFormat.QUANTIZED, quant_type,
                model, config
            )
            
            self.loaded_models[model_name] = model_info
            return True, f"Quantized model loaded with bitsandbytes"
            
        except Exception as e:
            print(f"Bitsandbytes loading failed: {str(e)[:100]}")
        
        # If all methods fail, provide guidance
        return False, f"Failed to load {quant_type.value} quantized model. The model format may not be compatible with Apple Silicon."
    
    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load config.json from model path."""
        config_path = path / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
    
    def _create_model_info(
        self,
        model_name: str,
        path: Path,
        format: ModelFormat,
        quantization: QuantizationType,
        model: Any,
        config: Dict[str, Any]
    ) -> ModelInfo:
        """Create ModelInfo object for a loaded model."""
        # Use smart parameter detection instead of loading model parameters
        parameters = self.get_model_parameters_smart(path)
        
        # Calculate memory usage
        if hasattr(model, 'parameters'):
            try:
                memory_used = sum(p.element_size() * p.numel() for p in model.parameters())
            except:
                memory_used = self._get_model_size(path)
        else:
            memory_used = self._get_model_size(path)
        
        return ModelInfo(
            name=model_name,
            path=path,
            format=format,
            quantization=quantization,
            size_bytes=self._get_model_size(path),
            parameters=parameters,
            context_length=config.get('max_position_embeddings', 4096),
            loaded_at=datetime.now(),
            gpu_memory_used=memory_used,
            tokenizer_path=path / 'tokenizer.json' if (path / 'tokenizer.json').exists() else None,
            config=config
        )
    
    def _count_mlx_parameters(self, model: Any) -> int:
        """Count parameters in MLX model."""
        try:
            if hasattr(model, 'num_parameters'):
                return model.num_parameters()
            elif hasattr(model, 'parameters'):
                params = model.parameters()
                if isinstance(params, dict):
                    return sum(p.size for p in params.values())
            return 0
        except:
            return 0
    
    def _estimate_mlx_memory(self, model: Any) -> int:
        """Estimate memory usage of MLX model."""
        try:
            if hasattr(model, 'parameters'):
                params = model.parameters()
                if isinstance(params, dict):
                    return sum(p.nbytes if hasattr(p, 'nbytes') else 0 for p in params.values())
            return 0
        except:
            return 0
    
    def unload_model(self, model_name: str) -> Tuple[bool, str]:
        """Unload a model from memory."""
        if model_name not in self.loaded_models:
            return False, f"Model '{model_name}' not loaded"
        
        try:
            # Special cleanup for GGUF models (llama-cpp-python)
            if model_name in self.model_cache:
                model = self.model_cache[model_name]
                model_info = self.loaded_models.get(model_name)
                
                # Clean up GGUF models properly to avoid memory leaks
                if model_info and model_info.format == ModelFormat.GGUF:
                    try:
                        # llama-cpp-python models have a close() method
                        if hasattr(model, 'close'):
                            model.close()
                        # Also try to explicitly delete the model
                        del model
                    except Exception as e:
                        print(f"Warning: Error closing GGUF model: {e}")
                
                # Remove from cache
                del self.model_cache[model_name]
            
            # Remove tokenizer
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            # Remove model info
            del self.loaded_models[model_name]
            
            # Update current model
            if self.current_model == model_name:
                self.current_model = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                # Clear MPS cache on Apple Silicon
                try:
                    torch.mps.empty_cache()
                except:
                    pass  # MPS cache clearing might not be available in all versions
            
            # Force garbage collection for thorough cleanup
            import gc
            gc.collect()
            
            return True, f"Model '{model_name}' unloaded"
            
        except Exception as e:
            return False, f"Error unloading model: {str(e)}"
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        return [model.to_dict() for model in self.loaded_models.values()]
    
    def discover_available_models(self) -> List[Dict[str, Any]]:
        """Discover all available models including MLX converted ones."""
        available = []
        model_path = self.config.model.model_path.expanduser().resolve()
        
        if not model_path.exists():
            return available
        
        # Also check MLX models directory for fine-tuned models
        mlx_path = Path.home() / ".cortex" / "mlx_models"
        
        # First, get all MLX converted models to check for optimized versions
        mlx_models = self.mlx_converter.list_converted_models()
        mlx_cache_map = {}  # Map original paths to MLX versions
        
        # Build a map of original model paths to their MLX versions
        for name, info in mlx_models.items():
            # Extract original path from MLX model name
            if name.startswith("_Users_") and ("_4bit" in name or "_5bit" in name or "_8bit" in name):
                base_name = name.replace("_4bit", "").replace("_5bit", "").replace("_8bit", "")
                original_path = "/" + base_name[1:].replace("_", "/")
                mlx_cache_map[original_path] = {
                    'mlx_name': name,
                    'mlx_path': info['path'],
                    'quantization': info.get('quantization', 4),
                    'size_gb': info.get('size_gb', 0)
                }
        
        # Search for models in the models directory
        for item in model_path.iterdir():
            if item.is_file() and item.suffix == '.gguf':
                # GGUF file
                size_gb = item.stat().st_size / (1024**3)
                available.append({
                    'name': item.stem,
                    'path': str(item),
                    'relative_path': item.name,
                    'format': 'GGUF',
                    'size_gb': round(size_gb, 2),
                    'mlx_optimized': False,
                    'mlx_available': False
                })
            elif item.is_dir():
                # Check if it's a model directory
                format_info = self._detect_format(item)
                if format_info['format'] != ModelFormat.UNKNOWN:
                    # Check if this model has an MLX optimized version
                    full_path = str(item.resolve())
                    has_mlx = full_path in mlx_cache_map
                    
                    if has_mlx:
                        # Use the MLX version's info
                        mlx_info = mlx_cache_map[full_path]
                        available.append({
                            'name': item.name,
                            'path': str(item),  # Keep original path for compatibility
                            'relative_path': item.name,
                            'format': 'MLX-optimized',
                            'size_gb': round(mlx_info['size_gb'], 2) if mlx_info['size_gb'] > 0 else round(self._get_model_size(item) / (1024**3), 2),
                            'mlx_optimized': True,
                            'mlx_path': mlx_info['mlx_path'],
                            'mlx_name': mlx_info['mlx_name'],
                            'quantization': f"{mlx_info['quantization']}-bit",
                            'original_format': format_info['format'].value.upper()
                        })
                    else:
                        # Regular model without MLX optimization
                        size_gb = self._get_model_size(item) / (1024**3)
                        format_str = format_info['format'].value.upper()
                        if format_info['quantization'] != QuantizationType.NONE:
                            format_str = f"{format_str} ({format_info['quantization'].value})"
                        
                        available.append({
                            'name': item.name,
                            'path': str(item),
                            'relative_path': item.name,
                            'format': format_str,
                            'size_gb': round(size_gb, 2),
                            'mlx_optimized': False,
                            'mlx_available': format_info['format'] in [
                                ModelFormat.SAFETENSORS,
                                ModelFormat.PYTORCH
                            ]
                        })
        
        # Add fine-tuned models from MLX directory that aren't already included
        if mlx_path.exists():
            for item in mlx_path.iterdir():
                if item.is_dir():
                    # Check if it's a fine-tuned model that's not already in the list
                    if (item / 'fine_tuned.marker').exists():
                        # This is a fine-tuned model
                        already_added = any(model['name'] == item.name for model in available)
                        if not already_added:
                            size_gb = self._get_model_size(item) / (1024**3)
                            
                            # Read metadata from marker file
                            base_model = "Unknown"
                            try:
                                with open(item / 'fine_tuned.marker', 'r') as f:
                                    content = f.read()
                                    for line in content.split('\n'):
                                        if 'LoRA fine-tuned version of' in line:
                                            base_model = line.split('LoRA fine-tuned version of ')[-1].strip()
                                            break
                            except:
                                pass
                            
                            available.append({
                                'name': item.name,
                                'path': str(item),
                                'relative_path': item.name,
                                'format': 'MLX Fine-tuned',
                                'size_gb': round(size_gb, 2),
                                'mlx_optimized': True,
                                'is_fine_tuned': True,
                                'base_model': base_model,
                                'fine_tuning_method': 'LoRA'
                            })
        
        # Sort by name
        available.sort(key=lambda x: x['name'].lower())
        return available
    
    def get_current_model(self) -> Optional[ModelInfo]:
        """Get currently active model."""
        if self.current_model:
            return self.loaded_models.get(self.current_model)
        return None
    
    def switch_model(self, model_name: str) -> Tuple[bool, str]:
        """Switch to a different loaded model."""
        if model_name not in self.loaded_models:
            return False, f"Model '{model_name}' not loaded"
        
        self.current_model = model_name
        return True, f"Switched to model '{model_name}'"
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get GPU memory status with MLX details."""
        status = self.gpu_validator.get_gpu_memory_status()
        
        total_model_memory = sum(
            model.gpu_memory_used for model in self.loaded_models.values()
        )
        
        status['models_loaded'] = len(self.loaded_models)
        status['model_memory_gb'] = total_model_memory / (1024**3)
        status['current_model'] = self.current_model
        
        # Add MLX-specific info
        mlx_models = [m for m in self.loaded_models.values() if m.format == ModelFormat.MLX]
        if mlx_models:
            status['mlx_models'] = len(mlx_models)
            status['mlx_memory_gb'] = sum(m.gpu_memory_used for m in mlx_models) / (1024**3)
        
        if self.memory_pool:
            pool_stats = self.memory_pool.get_stats()
            status['memory_pool'] = {
                'allocated_gb': pool_stats['allocated_memory'] / (1024**3),
                'free_gb': pool_stats['free_memory'] / (1024**3),
                'fragmentation': pool_stats['fragmentation'],
                'total_blocks': pool_stats['total_blocks'],
                'zero_copy_enabled': pool_stats.get('zero_copy', False)
            }
        
        # Add MLX accelerator status
        if self.mlx_accelerator:
            status['mlx_acceleration'] = {
                'amx_enabled': self.mlx_accelerator.config.use_amx,
                'operation_fusion': self.mlx_accelerator.config.fuse_operations,
                'kv_cache_size': self.mlx_accelerator.config.kv_cache_size,
                'quantization_bits': self.mlx_accelerator.config.quantization_bits
            }
        
        return status
    
    def detect_model_parameters(self, model_path: Path) -> Optional[int]:
        """
        Detect the actual number of parameters in a model.
        
        Uses proper parameter counting that handles quantization, LoRA adapters,
        and non-weight files correctly.
        
        Returns:
            Number of parameters, or None if detection fails
        """
        try:
            # Check cache first
            cache_key = f"{model_path.resolve()}:{model_path.stat().st_mtime}"
            cached_result = self._get_cached_parameter_count(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached parameter count: {cached_result:,}")
                return cached_result
            
            # Detect model format and apply appropriate detection method
            format_info = self._detect_format(model_path)
            param_count = None
            
            if format_info['format'] == ModelFormat.SAFETENSORS:
                param_count = self._detect_safetensors_parameters(model_path)
            elif format_info['format'] == ModelFormat.MLX:
                param_count = self._detect_mlx_parameters(model_path)
            elif format_info['format'] == ModelFormat.PYTORCH:
                param_count = self._detect_pytorch_parameters(model_path)
            
            # Fallback to config.json analysis
            if param_count is None:
                param_count = self._detect_config_parameters(model_path)
            
            # Cache the result if successful
            if param_count is not None:
                self._cache_parameter_count(cache_key, param_count)
                logger.info(f"Detected {param_count:,} parameters in {model_path.name}")
            else:
                logger.warning(f"Could not detect parameters for {model_path.name}")
            
            return param_count
            
        except Exception as e:
            logger.warning(f"Parameter detection failed for {model_path}: {e}")
            return None
    
    def _get_cached_parameter_count(self, cache_key: str) -> Optional[int]:
        """Get cached parameter count."""
        cache_file = self.config.model.model_cache_dir / "parameter_counts.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            return cache.get(cache_key)
        except:
            return None
    
    def _cache_parameter_count(self, cache_key: str, param_count: int) -> None:
        """Cache parameter count for faster future lookups."""
        cache_file = self.config.model.model_cache_dir / "parameter_counts.json"
        
        # Load existing cache
        cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except:
                pass
        
        # Update cache
        cache[cache_key] = param_count
        
        # Keep only recent entries (last 100)
        if len(cache) > 100:
            sorted_items = sorted(cache.items(), key=lambda x: x[0])[-100:]
            cache = dict(sorted_items)
        
        # Save cache
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning(f"Failed to cache parameter count: {e}")
    
    def _detect_safetensors_parameters(self, model_path: Path) -> Optional[int]:
        """Detect parameters by reading SafeTensors headers."""
        try:
            safetensor_files = list(model_path.glob("*.safetensors"))
            if not safetensor_files:
                return None
            
            total_params = 0
            
            for st_file in safetensor_files:
                # Skip adapter files for base model parameter counting
                if "adapter" in st_file.name.lower():
                    continue
                    
                # Read SafeTensors header to get tensor shapes without loading weights
                params = self._read_safetensors_header(st_file)
                if params is not None:
                    total_params += params
                else:
                    logger.warning(f"Could not read SafeTensors header from {st_file.name}")
                    return None
            
            return total_params if total_params > 0 else None
            
        except Exception as e:
            logger.warning(f"SafeTensors parameter detection failed: {e}")
            return None
    
    def _read_safetensors_header(self, file_path: Path) -> Optional[int]:
        """Read parameter count from SafeTensors file header without loading the full file."""
        try:
            with open(file_path, 'rb') as f:
                # Read the header length (first 8 bytes)
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    return None
                
                header_size = struct.unpack('<Q', header_size_bytes)[0]
                
                # Read the header JSON
                header_json = f.read(header_size).decode('utf-8')
                header = json.loads(header_json)
                
                # Count parameters from tensor shapes
                total_params = 0
                for tensor_name, tensor_info in header.items():
                    if tensor_name == "__metadata__":
                        continue
                    
                    # Skip non-parameter tensors (buffers, etc.)
                    if self._is_parameter_tensor(tensor_name):
                        shape = tensor_info.get('shape', [])
                        if shape:
                            tensor_params = 1
                            for dim in shape:
                                tensor_params *= dim
                            total_params += tensor_params
                
                return total_params
                
        except Exception as e:
            logger.debug(f"Failed to read SafeTensors header from {file_path}: {e}")
            return None
    
    def _is_parameter_tensor(self, tensor_name: str) -> bool:
        """Check if a tensor name represents a model parameter (not a buffer)."""
        # Common parameter patterns
        param_patterns = [
            'weight', 'bias', 'embeddings', 'lm_head',
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            'fc1', 'fc2', 'mlp', 'attention'
        ]
        
        # Common non-parameter patterns (buffers)
        non_param_patterns = [
            'position_ids', 'attention_mask', 'token_type_ids',
            'freqs_cos', 'freqs_sin', 'inv_freq'
        ]
        
        tensor_lower = tensor_name.lower()
        
        # Check non-parameter patterns first
        for pattern in non_param_patterns:
            if pattern in tensor_lower:
                return False
        
        # Check parameter patterns
        for pattern in param_patterns:
            if pattern in tensor_lower:
                return True
        
        # Default: assume it's a parameter if it contains common layer indicators
        return any(indicator in tensor_lower for indicator in ['layer', 'block', 'transformer'])
    
    def _detect_mlx_parameters(self, model_path: Path) -> Optional[int]:
        """Detect parameters in MLX models by inspecting weights.npz or using config."""
        try:
            # First try to read from weights.npz directly
            weights_file = model_path / "weights.npz"
            if weights_file.exists():
                import numpy as np
                
                # Load the weights file
                weights = np.load(weights_file)
                total_params = 0
                
                for array_name in weights.files:
                    if self._is_parameter_tensor(array_name):
                        array = weights[array_name]
                        total_params += array.size
                
                return total_params if total_params > 0 else None
            
            # Fallback to checking for SafeTensors in MLX directory
            safetensor_files = list(model_path.glob("*.safetensors"))
            if safetensor_files:
                return self._detect_safetensors_parameters(model_path)
            
            return None
            
        except Exception as e:
            logger.warning(f"MLX parameter detection failed: {e}")
            return None
    
    def _detect_pytorch_parameters(self, model_path: Path) -> Optional[int]:
        """Detect parameters in PyTorch models."""
        try:
            # For PyTorch models, we need to load the config to get architecture info
            # as loading the full model would be too expensive
            return self._detect_config_parameters(model_path)
            
        except Exception as e:
            logger.warning(f"PyTorch parameter detection failed: {e}")
            return None
    
    def _detect_config_parameters(self, model_path: Path) -> Optional[int]:
        """Detect parameters by analyzing config.json and calculating from architecture."""
        try:
            config_path = model_path / "config.json"
            if not config_path.exists():
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check for directly specified parameter count
            if 'num_parameters' in config:
                return int(config['num_parameters'])
            
            # Calculate from architecture parameters
            model_type = config.get('model_type', '').lower()
            
            if model_type in ['llama', 'gemma', 'mistral', 'qwen']:
                return self._calculate_llama_parameters(config)
            elif model_type in ['gpt', 'gpt2', 'gpt_neo', 'gpt_neox']:
                return self._calculate_gpt_parameters(config)
            elif model_type in ['bert', 'roberta', 'distilbert']:
                return self._calculate_bert_parameters(config)
            else:
                # Generic calculation for transformer models
                return self._calculate_generic_transformer_parameters(config)
                
        except Exception as e:
            logger.warning(f"Config parameter detection failed: {e}")
            return None
    
    def _calculate_llama_parameters(self, config: Dict) -> Optional[int]:
        """Calculate parameters for Llama-style models (including Gemma)."""
        try:
            vocab_size = config.get('vocab_size', 32000)
            hidden_size = config.get('hidden_size', 4096)
            intermediate_size = config.get('intermediate_size', 11008)
            num_layers = config.get('num_hidden_layers', 32)
            num_attention_heads = config.get('num_attention_heads', 32)
            
            # Check if this is a Gemma model for special handling
            is_gemma = config.get('model_type', '').lower() == 'gemma'
            
            # Embedding layer
            embedding_params = vocab_size * hidden_size
            
            # Each transformer layer:
            if is_gemma:
                # Gemma uses grouped query attention with fewer k/v heads
                num_key_value_heads = config.get('num_key_value_heads', num_attention_heads // 4)
                head_dim = hidden_size // num_attention_heads
                
                # Attention projections
                q_proj_params = hidden_size * hidden_size  # Full size
                k_proj_params = hidden_size * (num_key_value_heads * head_dim)  # Reduced
                v_proj_params = hidden_size * (num_key_value_heads * head_dim)  # Reduced
                o_proj_params = hidden_size * hidden_size  # Full size
                
                attention_params = q_proj_params + k_proj_params + v_proj_params + o_proj_params
            else:
                # Standard Llama: q, k, v, o projections all full size
                attention_params = 4 * (hidden_size * hidden_size)
            
            # Feed-forward: gate_proj, up_proj, down_proj
            ff_params = 2 * (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
            
            # Layer norms (2 per layer)
            ln_params = 2 * hidden_size
            
            layer_params = attention_params + ff_params + ln_params
            transformer_params = num_layers * layer_params
            
            # Final layer norm
            final_ln_params = hidden_size
            
            # LM head - check if tied to embeddings (common in smaller models)
            tie_word_embeddings = config.get('tie_word_embeddings', True)  # Default True for most models
            if tie_word_embeddings:
                lm_head_params = 0  # Tied to embeddings, don't double count
            else:
                lm_head_params = vocab_size * hidden_size
            
            total = embedding_params + transformer_params + final_ln_params + lm_head_params
            
            return total
            
        except Exception as e:
            logger.warning(f"Llama parameter calculation failed: {e}")
            return None
    
    def _calculate_gpt_parameters(self, config: Dict) -> Optional[int]:
        """Calculate parameters for GPT-style models."""
        try:
            vocab_size = config.get('vocab_size', 50257)
            n_embd = config.get('n_embd', config.get('hidden_size', 768))
            n_layer = config.get('n_layer', config.get('num_hidden_layers', 12))
            n_head = config.get('n_head', config.get('num_attention_heads', 12))
            
            # Token + position embeddings
            max_position_embeddings = config.get('n_positions', config.get('max_position_embeddings', 1024))
            embedding_params = vocab_size * n_embd + max_position_embeddings * n_embd
            
            # Each transformer block
            # - Attention: qkv projection + output projection
            attention_params = 4 * (n_embd * n_embd)
            
            # - MLP: typically 4x expansion
            mlp_size = config.get('n_inner', 4 * n_embd)
            mlp_params = n_embd * mlp_size + mlp_size * n_embd
            
            # - Layer norms
            ln_params = 2 * n_embd
            
            block_params = attention_params + mlp_params + ln_params
            transformer_params = n_layer * block_params
            
            # Final layer norm + LM head
            final_ln_params = n_embd
            lm_head_params = vocab_size * n_embd
            
            total = embedding_params + transformer_params + final_ln_params + lm_head_params
            
            return total
            
        except Exception as e:
            logger.warning(f"GPT parameter calculation failed: {e}")
            return None
    
    def _calculate_bert_parameters(self, config: Dict) -> Optional[int]:
        """Calculate parameters for BERT-style models."""
        try:
            vocab_size = config.get('vocab_size', 30522)
            hidden_size = config.get('hidden_size', 768)
            num_hidden_layers = config.get('num_hidden_layers', 12)
            intermediate_size = config.get('intermediate_size', 3072)
            max_position_embeddings = config.get('max_position_embeddings', 512)
            type_vocab_size = config.get('type_vocab_size', 2)
            
            # Embeddings: token + position + token_type
            embedding_params = (vocab_size * hidden_size + 
                              max_position_embeddings * hidden_size + 
                              type_vocab_size * hidden_size)
            
            # Each encoder layer
            # - Self-attention
            attention_params = 4 * (hidden_size * hidden_size)
            
            # - Feed-forward
            ff_params = hidden_size * intermediate_size + intermediate_size * hidden_size
            
            # - Layer norms
            ln_params = 2 * hidden_size
            
            layer_params = attention_params + ff_params + ln_params
            encoder_params = num_hidden_layers * layer_params
            
            # Pooler (optional)
            pooler_params = hidden_size * hidden_size
            
            total = embedding_params + encoder_params + pooler_params
            
            return total
            
        except Exception as e:
            logger.warning(f"BERT parameter calculation failed: {e}")
            return None
    
    def _calculate_generic_transformer_parameters(self, config: Dict) -> Optional[int]:
        """Generic parameter calculation for transformer models."""
        try:
            # Try to extract common parameters
            vocab_size = config.get('vocab_size', 32000)
            hidden_size = config.get('hidden_size', config.get('n_embd', config.get('d_model', 512)))
            num_layers = config.get('num_hidden_layers', config.get('n_layer', config.get('num_layers', 6)))
            
            if hidden_size is None or num_layers is None:
                return None
            
            # Very rough estimation for generic transformers
            # Embeddings + layers + head
            embedding_params = vocab_size * hidden_size
            
            # Each layer: attention + ffn + norms (rough 6x hidden_size^2 per layer)
            layer_params = 6 * (hidden_size * hidden_size)
            transformer_params = num_layers * layer_params
            
            # Output head
            head_params = vocab_size * hidden_size
            
            total = embedding_params + transformer_params + head_params
            
            logger.info(f"Generic parameter estimation: {total:,} parameters")
            return total
            
        except Exception as e:
            logger.warning(f"Generic parameter calculation failed: {e}")
            return None
    
    def get_model_parameters_smart(self, model_path: Path) -> float:
        """Get model parameters in billions with smart detection, fallback to size estimation."""
        # Try accurate parameter detection first
        param_count = self.detect_model_parameters(model_path)
        
        if param_count is not None:
            return param_count / 1e9  # Convert to billions
        
        # Fallback to improved size-based estimation
        logger.warning(f"Falling back to size-based parameter estimation for {model_path.name}")
        
        size_bytes = self._get_model_size(model_path)
        size_gb = size_bytes / (1024**3)
        
        # Improved estimation that considers file overhead
        # Only count actual weight files, not tokenizer configs etc.
        weight_size = self._estimate_weight_file_size(model_path)
        weight_size_gb = weight_size / (1024**3)
        
        # Use weight size if significantly different from total size
        if weight_size_gb < size_gb * 0.8:  # If weight files are <80% of total
            size_gb = weight_size_gb
            logger.info(f"Using weight-only size: {size_gb:.2f}GB (total: {size_bytes / (1024**3):.2f}GB)")
        
        # Better default estimation: 2.2 bytes per parameter (accounts for some overhead)
        estimated_params_b = size_gb / 2.2  # Already in billions
        
        logger.info(f"Estimated {estimated_params_b:.2f}B parameters from {size_gb:.2f}GB model size")
        return estimated_params_b
    
    def _estimate_weight_file_size(self, model_path: Path) -> int:
        """Estimate size of actual weight files, excluding configs and tokenizers."""
        if model_path.is_file():
            return model_path.stat().st_size
        
        weight_patterns = [
            '*.safetensors', '*.bin', '*.npz',
            'pytorch_model*.bin', 'model*.safetensors'
        ]
        
        non_weight_patterns = [
            'tokenizer*', 'vocab*', 'merges.txt', 'config.json',
            'generation_config.json', 'special_tokens_map.json',
            'tokenizer_config.json', 'added_tokens.json'
        ]
        
        total_weight_size = 0
        
        for file_path in model_path.rglob('*'):
            if file_path.is_file():
                # Check if it matches weight patterns
                is_weight = any(file_path.match(pattern) for pattern in weight_patterns)
                
                # Exclude non-weight files
                is_non_weight = any(file_path.match(pattern) for pattern in non_weight_patterns)
                
                if is_weight and not is_non_weight:
                    total_weight_size += file_path.stat().st_size
                elif not is_non_weight and file_path.suffix in ['.safetensors', '.bin', '.npz']:
                    # Include other tensor files that don't match specific patterns
                    total_weight_size += file_path.stat().st_size
        
        return total_weight_size
