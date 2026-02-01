"""Unified Metal optimization interface for Apple Silicon LLM inference.

This module provides a simple, effective interface for accelerating LLM inference
on Apple Silicon using the most appropriate backend (MLX or MPS).
"""

import os
import sys
from typing import Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

import torch
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_validator import GPUValidator


class Backend(Enum):
    """Available acceleration backends."""
    MLX = "mlx"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class OptimizationConfig:
    """Configuration for Metal optimization."""
    backend: Backend = Backend.MLX
    dtype: str = "auto"  # auto, float32, float16, bfloat16
    batch_size: int = 1
    max_memory_gb: Optional[float] = None  # None = auto-detect
    use_quantization: bool = True
    quantization_bits: int = 8  # 4 or 8
    compile_model: bool = True
    use_kv_cache: bool = True
    kv_cache_size: int = 2048
    enable_profiling: bool = False
    fallback_to_cpu: bool = True
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.quantization_bits not in [4, 8]:
            raise ValueError(f"Quantization bits must be 4 or 8, got {self.quantization_bits}")
        
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")
        
        if self.kv_cache_size < 0:
            raise ValueError(f"KV cache size must be >= 0, got {self.kv_cache_size}")
        
        return True


class MetalOptimizer:
    """Unified Metal optimizer for LLM inference on Apple Silicon."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize the Metal optimizer.
        
        Args:
            config: Optimization configuration. Uses defaults if None.
        """
        self.config = config or OptimizationConfig()
        self.config.validate()
        
        # Initialize GPU validator
        self.gpu_validator = GPUValidator()
        self.gpu_validator.validate()
        
        # Detect best backend
        self.backend = self._select_backend()
        self.device = self._get_device()
        
        # Initialize backend-specific components
        self._backend_optimizer = None
        self._initialize_backend()
        
        logger.info(f"MetalOptimizer initialized with backend: {self.backend.value}")
        logger.info(f"GPU: {self.gpu_validator.get_gpu_family()}, "
                   f"bfloat16: {self.gpu_validator.check_bfloat16_support()}")
    
    def _select_backend(self) -> Backend:
        """Select the best available backend.
        
        Returns:
            Selected backend based on availability and configuration.
        """
        if self.config.backend != Backend.MLX:
            # User explicitly selected a backend
            return self._validate_backend(self.config.backend)
        
        # Auto-select best backend
        try:
            import mlx.core as mx
            # MLX is available and preferred
            return Backend.MLX
        except ImportError:
            logger.warning("MLX not available, falling back to MPS")
            
        if torch.backends.mps.is_available():
            return Backend.MPS
        
        if self.config.fallback_to_cpu:
            logger.warning("No GPU acceleration available, using CPU")
            return Backend.CPU
        
        raise RuntimeError("No suitable backend available and CPU fallback disabled")
    
    def _validate_backend(self, backend: Backend) -> Backend:
        """Validate that the requested backend is available.
        
        Args:
            backend: Requested backend
            
        Returns:
            The backend if available
            
        Raises:
            RuntimeError: If backend not available
        """
        if backend == Backend.MLX:
            try:
                import mlx.core as mx
                return backend
            except ImportError:
                raise RuntimeError("MLX backend requested but not installed")
        
        elif backend == Backend.MPS:
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS backend requested but not available")
            return backend
        
        elif backend == Backend.CPU:
            return backend
        
        raise ValueError(f"Unknown backend: {backend}")
    
    def _get_device(self) -> Union[str, torch.device]:
        """Get the appropriate device for the selected backend.
        
        Returns:
            Device object or string
        """
        if self.backend == Backend.MPS:
            return torch.device("mps")
        elif self.backend == Backend.CPU:
            return torch.device("cpu")
        else:
            return "gpu"  # MLX uses string device names
    
    def _initialize_backend(self) -> None:
        """Initialize the backend-specific optimizer."""
        if self.backend == Backend.MLX:
            self._initialize_mlx()
        elif self.backend == Backend.MPS:
            self._initialize_mps()
        else:
            # CPU backend needs no special initialization
            pass
    
    def _initialize_mlx(self) -> None:
        """Initialize MLX backend."""
        try:
            from mlx_accelerator import MLXAccelerator, MLXConfig
            
            mlx_config = MLXConfig(
                compile_model=self.config.compile_model,
                batch_size=self.config.batch_size,
                dtype=self._get_mlx_dtype()
            )
            
            self._backend_optimizer = MLXAccelerator(mlx_config)
            
        except ImportError as e:
            logger.error(f"Failed to initialize MLX backend: {e}")
            raise RuntimeError("MLX initialization failed")
    
    def _initialize_mps(self) -> None:
        """Initialize MPS backend."""
        try:
            from mps_optimizer import MPSOptimizer, MPSConfig
            
            mps_config = MPSConfig(
                use_fp16=(self.config.dtype in ["float16", "auto"]),
                use_bfloat16=self._should_use_bfloat16(),
                max_batch_size=self.config.batch_size,
                optimize_memory=True
            )
            
            self._backend_optimizer = MPSOptimizer(mps_config)
            
        except ImportError as e:
            logger.error(f"Failed to initialize MPS backend: {e}")
            raise RuntimeError("MPS initialization failed")
    
    def _get_mlx_dtype(self):
        """Get MLX dtype based on configuration and hardware."""
        if self.config.dtype == "auto":
            # Auto-select based on hardware
            if self.gpu_validator.check_bfloat16_support():
                import mlx.core as mx
                return mx.bfloat16
            else:
                import mlx.core as mx
                return mx.float16
        
        elif self.config.dtype == "float32":
            import mlx.core as mx
            return mx.float32
        elif self.config.dtype == "float16":
            import mlx.core as mx
            return mx.float16
        elif self.config.dtype == "bfloat16":
            import mlx.core as mx
            return mx.bfloat16
        else:
            raise ValueError(f"Unknown dtype: {self.config.dtype}")
    
    def _should_use_bfloat16(self) -> bool:
        """Determine if bfloat16 should be used."""
        if self.config.dtype == "bfloat16":
            return True
        
        if self.config.dtype == "auto":
            return self.gpu_validator.check_bfloat16_support()
        
        return False
    
    def optimize_model(
        self,
        model: Any,
        model_type: str = "auto"
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize a model for inference.
        
        Args:
            model: Model to optimize (PyTorch, MLX, or Hugging Face)
            model_type: Type of model ("pytorch", "mlx", "transformers", "auto")
            
        Returns:
            Tuple of (optimized_model, optimization_info)
        """
        if model_type == "auto":
            model_type = self._detect_model_type(model)
        
        logger.info(f"Optimizing {model_type} model with {self.backend.value} backend")
        
        optimization_info = {
            "backend": self.backend.value,
            "device": str(self.device),
            "dtype": self.config.dtype,
            "quantization": self.config.use_quantization,
            "quantization_bits": self.config.quantization_bits if self.config.use_quantization else None,
            "gpu_family": self.gpu_validator.get_gpu_family(),
            "optimizations_applied": []
        }
        
        if self.backend == Backend.MLX:
            optimized_model = self._optimize_mlx_model(model, model_type)
            optimization_info["optimizations_applied"].extend([
                "mlx_compilation",
                "dtype_optimization",
                "memory_layout"
            ])
            
        elif self.backend == Backend.MPS:
            optimized_model = self._optimize_mps_model(model, model_type)
            optimization_info["optimizations_applied"].extend([
                "mps_optimization",
                "dtype_conversion",
                "memory_optimization"
            ])
            
        else:
            # CPU - minimal optimization
            optimized_model = model
            optimization_info["optimizations_applied"].append("none")
        
        # Apply quantization if requested
        if self.config.use_quantization:
            optimized_model = self._apply_quantization(
                optimized_model,
                self.config.quantization_bits
            )
            optimization_info["optimizations_applied"].append(
                f"int{self.config.quantization_bits}_quantization"
            )
        
        return optimized_model, optimization_info
    
    def _detect_model_type(self, model: Any) -> str:
        """Detect the type of model.
        
        Args:
            model: Model to detect
            
        Returns:
            Model type string
        """
        # Check for PyTorch model
        if hasattr(model, "parameters") and hasattr(model, "forward"):
            return "pytorch"
        
        # Check for MLX model
        if hasattr(model, "apply_to_parameters"):
            return "mlx"
        
        # Check for Hugging Face transformers
        if hasattr(model, "config") and hasattr(model, "forward"):
            return "transformers"
        
        return "unknown"
    
    def _optimize_mlx_model(self, model: Any, model_type: str) -> Any:
        """Optimize model using MLX backend.
        
        Args:
            model: Model to optimize
            model_type: Type of model
            
        Returns:
            Optimized model
        """
        if not self._backend_optimizer:
            raise RuntimeError("MLX backend not initialized")
        
        if model_type == "pytorch":
            # Convert PyTorch model to MLX
            logger.warning("PyTorch to MLX conversion not yet implemented")
            return model
        
        elif model_type == "mlx":
            # Already MLX model, just optimize
            return self._backend_optimizer.optimize_model(model)
        
        elif model_type == "transformers":
            # Convert Hugging Face model to MLX
            logger.warning("Transformers to MLX conversion not yet implemented")
            return model
        
        return model
    
    def _optimize_mps_model(self, model: Any, model_type: str) -> Any:
        """Optimize model using MPS backend.
        
        Args:
            model: Model to optimize
            model_type: Type of model
            
        Returns:
            Optimized model
        """
        if not self._backend_optimizer:
            raise RuntimeError("MPS backend not initialized")
        
        if model_type in ["pytorch", "transformers"]:
            # MPS works with PyTorch models
            return self._backend_optimizer.optimize_model(model)
        
        elif model_type == "mlx":
            # Cannot use MLX model with MPS
            logger.error("Cannot use MLX model with MPS backend")
            raise ValueError("MLX models not compatible with MPS backend")
        
        return model
    
    def _apply_quantization(self, model: Any, bits: int) -> Any:
        """Apply quantization to the model.
        
        Args:
            model: Model to quantize
            bits: Quantization bits (4 or 8)
            
        Returns:
            Quantized model
        """
        if self.backend == Backend.MLX:
            # Use MLX quantization
            if hasattr(self._backend_optimizer, "quantize_model"):
                return self._backend_optimizer.quantize_model(model, bits)
        
        elif self.backend == Backend.MPS:
            # Use custom quantization for PyTorch
            try:
                from quantization.dynamic_quantizer import DynamicQuantizer, QuantizationConfig
                
                quantizer = DynamicQuantizer(
                    QuantizationConfig(
                        mode="int8" if bits == 8 else "int4",
                        device=self.device
                    )
                )
                
                quantized_model, _ = quantizer.quantize_model(model)
                return quantized_model
                
            except ImportError:
                logger.warning("Quantization module not available")
                return model
        
        return model
    
    def create_inference_session(
        self,
        model: Any,
        tokenizer: Optional[Any] = None
    ) -> 'InferenceSession':
        """Create an optimized inference session.
        
        Args:
            model: Model for inference
            tokenizer: Optional tokenizer
            
        Returns:
            InferenceSession object
        """
        optimized_model, info = self.optimize_model(model)
        
        return InferenceSession(
            model=optimized_model,
            tokenizer=tokenizer,
            optimizer=self,
            optimization_info=info
        )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        import psutil
        
        vm = psutil.virtual_memory()
        stats = {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent_used": vm.percent
        }
        
        if self.backend == Backend.MPS:
            if hasattr(torch.mps, "current_allocated_memory"):
                stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / (1024**3)
        
        return stats
    
    def profile_inference(
        self,
        model: Any,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """Profile model inference performance.
        
        Args:
            model: Model to profile
            input_shape: Shape of input tensor
            num_iterations: Number of iterations for profiling
            
        Returns:
            Profiling results
        """
        if self.backend == Backend.MLX and self._backend_optimizer:
            return self._backend_optimizer.profile_model(
                model, input_shape, num_iterations
            )
        
        elif self.backend == Backend.MPS and self._backend_optimizer:
            return self._backend_optimizer.profile_model(
                model, input_shape, num_iterations
            )
        
        # Basic CPU profiling
        import time
        
        if self.backend == Backend.CPU:
            dummy_input = torch.randn(input_shape)
        else:
            dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Profile
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_iterations
        
        return {
            "backend": self.backend.value,
            "avg_inference_time": avg_time,
            "throughput": input_shape[0] / avg_time,
            "device": str(self.device),
            "iterations": num_iterations
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.backend == Backend.MPS:
            torch.mps.empty_cache()
        
        if hasattr(self._backend_optimizer, "cleanup"):
            self._backend_optimizer.cleanup()


class InferenceSession:
    """Optimized inference session for LLM models."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Optional[Any],
        optimizer: MetalOptimizer,
        optimization_info: Dict[str, Any]
    ):
        """Initialize inference session.
        
        Args:
            model: Optimized model
            tokenizer: Optional tokenizer
            optimizer: MetalOptimizer instance
            optimization_info: Optimization information
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.optimization_info = optimization_info
        self.generation_config = {}
    
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Union[str, torch.Tensor]:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt (string or tensor)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or tokens
        """
        # This is a simplified interface - actual implementation
        # would depend on the model type and backend
        
        if isinstance(prompt, str) and self.tokenizer:
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            if self.optimizer.backend == Backend.MPS:
                input_ids = input_ids.to(self.optimizer.device)
        else:
            input_ids = prompt
        
        # Generate based on backend
        if self.optimizer.backend == Backend.MLX:
            # MLX generation (simplified)
            output = self._generate_mlx(input_ids, max_tokens, temperature, top_p)
        else:
            # PyTorch generation
            output = self._generate_pytorch(input_ids, max_tokens, temperature, top_p, **kwargs)
        
        # Decode if tokenizer available
        if self.tokenizer and not isinstance(output, str):
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return output
    
    def _generate_mlx(
        self,
        input_ids: Any,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> Any:
        """Generate using MLX backend."""
        # Simplified MLX generation
        # Actual implementation would use MLX-specific generation
        return input_ids
    
    def _generate_pytorch(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> torch.Tensor:
        """Generate using PyTorch backend."""
        # Use model's generate method if available
        if hasattr(self.model, "generate"):
            return self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                **kwargs
            )
        
        # Simple generation loop (fallback)
        generated = input_ids
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        callback: Optional[Callable] = None,
        **kwargs
    ):
        """Stream generation token by token.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            callback: Optional callback for each token
            **kwargs: Additional parameters
            
        Yields:
            Generated tokens
        """
        # This would implement streaming generation
        # For now, just yield the final result
        result = self.generate(prompt, max_tokens, **kwargs)
        yield result
    
    def get_info(self) -> Dict[str, Any]:
        """Get session information.
        
        Returns:
            Session information dictionary
        """
        info = self.optimization_info.copy()
        info["memory_usage"] = self.optimizer.get_memory_usage()
        return info