"""GPU-only inference engine for Cortex."""

import sys
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Generator, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import mlx.core as mx
import mlx.nn as nn

# Import MLX LM functions safely
try:
    from mlx_lm import generate as mlx_generate, stream_generate as mlx_stream_generate
except ImportError:
    mlx_generate = None
    mlx_stream_generate = None
from cortex.metal.mlx_compat import patch_mlx_lm_device_info
patch_mlx_lm_device_info()

from cortex.config import Config
from cortex.model_manager import ModelManager, ModelFormat
from cortex.metal.memory_pool import MemoryPool, AllocationStrategy
from cortex.metal.mps_optimizer import MPSOptimizer, MPSConfig
from cortex.metal.mlx_accelerator import MLXAccelerator, MLXConfig
from cortex.metal.performance_profiler import PerformanceProfiler

class InferenceStatus(Enum):
    """Status of inference operation."""
    IDLE = "idle"
    LOADING = "loading"
    GENERATING = "generating"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class GenerationMetrics:
    """Metrics for generation performance."""
    tokens_generated: int
    time_elapsed: float
    tokens_per_second: float
    gpu_utilization: float
    memory_used_gb: float
    first_token_latency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tokens_generated': self.tokens_generated,
            'time_elapsed': self.time_elapsed,
            'tokens_per_second': self.tokens_per_second,
            'gpu_utilization': self.gpu_utilization,
            'memory_used_gb': self.memory_used_gb,
            'first_token_latency': self.first_token_latency
        }

@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = None
    stream: bool = True
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


class StreamDeltaNormalizer:
    """Normalize streaming chunks to deltas, handling cumulative or overlapping output."""

    def __init__(self, max_overlap: int = 4096, min_cumulative_length: int = 32) -> None:
        self._total_text = ""
        self._max_overlap = max_overlap
        self._min_cumulative_length = min_cumulative_length
        self._cumulative_mode = False

    def normalize(self, chunk: Any) -> str:
        if chunk is None:
            return ""
        if not isinstance(chunk, str):
            chunk = str(chunk)
        if not chunk:
            return ""

        if not self._total_text:
            self._total_text = chunk
            return chunk

        if not self._cumulative_mode:
            if len(chunk) > len(self._total_text) and chunk.startswith(self._total_text):
                self._cumulative_mode = True
                delta = chunk[len(self._total_text):]
                self._total_text = chunk
                return delta
            if chunk == self._total_text and len(chunk) >= self._min_cumulative_length:
                # Likely a cumulative stream repeating the full text; don't re-emit.
                self._cumulative_mode = True
                return ""
            # Default to delta mode to avoid dropping legitimate repeats.
            self._total_text += chunk
            return chunk

        # Cumulative mode: emit only new suffix.
        if chunk.startswith(self._total_text):
            delta = chunk[len(self._total_text):]
            self._total_text = chunk
            return delta

        # Handle partial overlap in cumulative streams.
        max_overlap = min(len(self._total_text), len(chunk), self._max_overlap)
        if max_overlap > 0:
            tail = self._total_text[-max_overlap:]
            for i in range(max_overlap, 0, -1):
                if tail[-i:] == chunk[:i]:
                    delta = chunk[i:]
                    self._total_text += delta
                    return delta

        # Fallback: treat as fresh delta to avoid loss.
        self._total_text += chunk
        return chunk

class InferenceEngine:
    """GPU-accelerated inference engine."""
    
    def __init__(self, config: Config, model_manager: ModelManager):
        """Initialize inference engine."""
        self.config = config
        self.model_manager = model_manager
        self.status = InferenceStatus.IDLE
        self.current_metrics: Optional[GenerationMetrics] = None
        self._cancel_event = threading.Event()
        self._generation_lock = threading.Lock()
        
        # Initialize Metal optimizations
        self.memory_pool: Optional[MemoryPool] = None
        self.mps_optimizer: Optional[MPSOptimizer] = None
        self.mlx_accelerator: Optional[MLXAccelerator] = None
        self.profiler = PerformanceProfiler(sample_interval=0.1)
        
        self._ensure_gpu_backend()
        self._initialize_metal_optimizations()
    
    def _ensure_gpu_backend(self) -> None:
        """Ensure GPU backend is available."""
        if not torch.backends.mps.is_available():
            print("❌ MPS backend not available. GPU acceleration required.")
            sys.exit(1)
        
        try:
            mx.default_device()
        except Exception as e:
            print(f"❌ MLX not available: {e}")
            print("GPU acceleration via MLX is required.")
            sys.exit(1)
    
    def _initialize_metal_optimizations(self) -> None:
        """Initialize Metal-specific optimizations."""
        # Initialize shared memory pool with auto-sizing
        if self.config.gpu.force_gpu and self.memory_pool is None:
            # Create a single shared memory pool to avoid duplication
            self.memory_pool = MemoryPool(
                pool_size=None,  # Will auto-size based on available memory
                strategy=AllocationStrategy.UNIFIED,
                device="mps" if torch.backends.mps.is_available() else "mlx",
                auto_size=True  # Enable auto-sizing
            )
            
            # Share the pool with model manager to avoid duplication
            if hasattr(self.model_manager, 'memory_pool') and self.model_manager.memory_pool is None:
                self.model_manager.memory_pool = self.memory_pool
        
        # Initialize MPS optimizer
        if torch.backends.mps.is_available():
            mps_config = MPSConfig(
                use_fp16=True,
                use_channels_last=True,
                optimize_memory=True,
                max_batch_size=self.config.performance.max_batch_size
            )
            self.mps_optimizer = MPSOptimizer(mps_config)
        
        # Initialize MLX accelerator with AMX and advanced features
        try:
            mlx_config = MLXConfig(
                compile_model=True,
                use_graph=True,
                batch_size=self.config.performance.batch_size,
                dtype=mx.bfloat16 if self._supports_bfloat16() else mx.float16,
                use_amx=True,
                fuse_operations=True,
                lazy_evaluation=True,
                rotating_kv_cache=True,
                kv_cache_size=self.config.performance.context_length,
                quantization_bits=4
            )
            self.mlx_accelerator = MLXAccelerator(mlx_config)
            print("✓ MLX accelerator initialized with AMX support")
        except Exception as e:
            print(f"Warning: MLX accelerator initialization failed: {e}")
            self.mlx_accelerator = None
        
        # GPU acceleration handled by MLX and MPS backends
    
    def generate(
        self,
        request: GenerationRequest
    ) -> Generator[str, None, GenerationMetrics]:
        """
        Generate text using GPU-accelerated inference.
        
        Args:
            request: Generation request parameters
            
        Yields:
            Generated text tokens
            
        Returns:
            Generation metrics
        """
        with self._generation_lock:
            if self.status == InferenceStatus.GENERATING:
                raise RuntimeError("Generation already in progress")
            
            self.status = InferenceStatus.GENERATING
            self._cancel_event.clear()
            
            try:
                model_info = self.model_manager.get_current_model()
                if not model_info:
                    raise RuntimeError("No model loaded")
                
                model = self.model_manager.model_cache.get(model_info.name)
                tokenizer = self.model_manager.tokenizers.get(model_info.name)
                
                if not model or not tokenizer:
                    raise RuntimeError(f"Model '{model_info.name}' not properly loaded")
                
                if model_info.format == ModelFormat.MLX:
                    yield from self._generate_mlx(model, tokenizer, request)
                elif model_info.format == ModelFormat.PYTORCH:
                    yield from self._generate_pytorch(model, tokenizer, request)
                elif model_info.format == ModelFormat.SAFETENSORS:
                    yield from self._generate_safetensors(model, tokenizer, request)
                elif model_info.format == ModelFormat.QUANTIZED:
                    # Quantized models are loaded as PyTorch-compatible modules
                    yield from self._generate_pytorch(model, tokenizer, request)
                elif model_info.format == ModelFormat.GGUF:
                    yield from self._generate_gguf(model, tokenizer, request)
                else:
                    raise RuntimeError(f"Unsupported format: {model_info.format}")
                
                return self.current_metrics
                
            except Exception as e:
                self.status = InferenceStatus.ERROR
                raise e
            finally:
                if self.status != InferenceStatus.CANCELLED:
                    self.status = InferenceStatus.COMPLETED
    
    def _generate_mlx(
        self,
        model: Any,
        tokenizer: Any,
        request: GenerationRequest
    ) -> Generator[str, None, None]:
        """Generate using MLX model on GPU with Metal optimizations."""
        # Apply MLX optimizations if available
        if self.mlx_accelerator:
            logger.info("Applying MLX accelerator optimizations to model")
            model = self.mlx_accelerator.optimize_model(model)
        
        # Start profiling
        self.profiler.start_profiling("mlx_generation", {
            "model_type": "mlx",
            "max_tokens": request.max_tokens
        })
        
        start_time = time.time()
        tokens_generated = 0
        first_token_time = None
        last_metrics_update = time.time()
        normalizer = StreamDeltaNormalizer() if request.stream else None
        
        try:
            # Use MLX accelerator's optimized generation if available
            if self.mlx_accelerator and request.stream:
                logger.info("Using MLX accelerator optimized generation with AMX")
                for token in self.mlx_accelerator.generate_optimized(
                    model,
                    tokenizer,
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    stream=True,
                    stop_sequences=request.stop_sequences
                ):
                    if self._cancel_event.is_set():
                        self.status = InferenceStatus.CANCELLED
                        break

                    delta = normalizer.normalize(token) if normalizer else str(token)
                    if not delta:
                        continue

                    if first_token_time is None:
                        first_token_time = time.time() - start_time

                    tokens_generated += 1
                    
                    # Update metrics less frequently
                    current_time = time.time()
                    if current_time - last_metrics_update > 1.0 or tokens_generated % 50 == 0:
                        elapsed_time = current_time - start_time
                        
                        self.current_metrics = GenerationMetrics(
                            tokens_generated=tokens_generated,
                            time_elapsed=elapsed_time,
                            tokens_per_second=tokens_generated / elapsed_time if elapsed_time > 0 else 0,
                            gpu_utilization=0.0,
                            memory_used_gb=0.0,
                            first_token_latency=first_token_time or 0
                        )
                        last_metrics_update = current_time
                    
                    # Token is already a string from generate_optimized
                    yield delta
                    
                    if any(stop in token for stop in request.stop_sequences):
                        break
            elif mlx_generate:
                # Fallback to standard MLX generation
                if request.stream and mlx_stream_generate:
                    logger.info("Using MLX streaming generation")
                    generate_fn = mlx_stream_generate
                else:
                    logger.info("Using standard MLX generation")
                    generate_fn = mlx_generate
                
                # Import sample_utils for creating sampler
                try:
                    from mlx_lm.sample_utils import make_sampler
                    # Create sampler with temperature and top_p
                    sampler = make_sampler(request.temperature, top_p=request.top_p)
                    logger.debug(f"Created sampler with temp={request.temperature}, top_p={request.top_p}")
                except ImportError:
                    sampler = None
                    logger.warning("mlx_lm.sample_utils not available, using default sampler")
                
                # Build generation kwargs
                generation_kwargs = {
                    'prompt': request.prompt,
                    'max_tokens': request.max_tokens,
                }
                
                if sampler is not None:
                    generation_kwargs['sampler'] = sampler
                
                if request.seed is not None and request.seed >= 0:
                    mx.random.seed(request.seed)
                
                for response in generate_fn(
                    model,
                    tokenizer,
                    **generation_kwargs
                ):
                    if self._cancel_event.is_set():
                        self.status = InferenceStatus.CANCELLED
                        break
                    
                    # Extract text from GenerationResponse
                    if hasattr(response, 'text'):
                        token = response.text
                    else:
                        token = str(response)

                    delta = normalizer.normalize(token) if normalizer else token
                    if request.stream and not delta:
                        continue

                    if first_token_time is None:
                        first_token_time = time.time() - start_time

                    tokens_generated += 1
                    
                    # Update metrics less frequently to reduce overhead
                    # Only update every 50 tokens or 1 second for better performance
                    current_time = time.time()
                    if current_time - last_metrics_update > 1.0 or tokens_generated % 50 == 0:
                        elapsed_time = current_time - start_time
                        
                        # Skip expensive GPU queries during generation for better performance
                        # These will be calculated once at the end
                        self.current_metrics = GenerationMetrics(
                            tokens_generated=tokens_generated,
                            time_elapsed=elapsed_time,
                            tokens_per_second=tokens_generated / elapsed_time if elapsed_time > 0 else 0,
                            gpu_utilization=0.0,  # Skip during generation
                            memory_used_gb=0.0,   # Skip during generation
                            first_token_latency=first_token_time or 0
                        )
                        last_metrics_update = current_time
                    
                    yield delta
                    
                    if any(stop in token for stop in request.stop_sequences):
                        break
            else:
                # No MLX generation available
                logger.error("MLX generation functions not available")
                raise RuntimeError("MLX generation not available. Please install mlx-lm.")
            
            elapsed_time = time.time() - start_time
            
            # Stop profiling and get final results
            profile_result = self.profiler.stop_profiling()
            
            # Update final metrics
            self.current_metrics = GenerationMetrics(
                tokens_generated=tokens_generated,
                time_elapsed=elapsed_time,
                tokens_per_second=tokens_generated / elapsed_time if elapsed_time > 0 else 0,
                gpu_utilization=profile_result.gpu_utilization,
                memory_used_gb=profile_result.memory_used_mb / 1024,
                first_token_latency=first_token_time or 0
            )
            
        except Exception as e:
            self.status = InferenceStatus.ERROR
            self.profiler.stop_profiling()
            raise e
    
    def _generate_pytorch(
        self,
        model: Any,
        tokenizer: Any,
        request: GenerationRequest
    ) -> Generator[str, None, None]:
        """Generate using PyTorch model on MPS with Metal optimizations."""
        # Apply MPS optimizations if available
        if self.mps_optimizer:
            model = self.mps_optimizer.optimize_model(model)
        
        # Start profiling
        self.profiler.start_profiling("pytorch_generation", {
            "model_type": "pytorch",
            "max_tokens": request.max_tokens
        })
        
        start_time = time.time()
        tokens_generated = 0
        first_token_time = None
        last_metrics_update = time.time()
        
        try:
            # Use the model's device when available (quantized models may be CPU-only on macOS)
            device = None
            try:
                first_param = next(model.parameters())
                device = first_param.device
            except Exception:
                device = None

            if device is None or str(device) == "meta":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            elif device.type == "mps" and not torch.backends.mps.is_available():
                device = torch.device("cpu")
            
            inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
            
            generation_config = {
                'max_new_tokens': request.max_tokens,
                'temperature': request.temperature,
                'top_p': request.top_p,
                'top_k': request.top_k,
                'repetition_penalty': request.repetition_penalty,
                'do_sample': request.temperature > 0,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
            }
            
            if request.seed is not None and request.seed >= 0:
                torch.manual_seed(request.seed)
            
            with torch.no_grad():
                if request.stream:
                    from transformers import TextIteratorStreamer
                    
                    normalizer = StreamDeltaNormalizer()
                    streamer = TextIteratorStreamer(
                        tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True
                    )
                    
                    generation_kwargs = dict(
                        inputs,
                        streamer=streamer,
                        **generation_config
                    )
                    
                    thread = threading.Thread(
                        target=model.generate,
                        kwargs=generation_kwargs
                    )
                    thread.start()
                    
                    for token in streamer:
                        if self._cancel_event.is_set():
                            self.status = InferenceStatus.CANCELLED
                            break

                        delta = normalizer.normalize(token)
                        if not delta:
                            continue
                        
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                        
                        tokens_generated += 1
                        
                        # Update metrics less frequently to reduce overhead
                        # Only update every 50 tokens or 1 second for better performance
                        current_time = time.time()
                        if current_time - last_metrics_update > 1.0 or tokens_generated % 50 == 0:
                            elapsed_time = current_time - start_time
                            
                            # Skip expensive GPU queries during generation for better performance
                            # These will be calculated once at the end
                            self.current_metrics = GenerationMetrics(
                                tokens_generated=tokens_generated,
                                time_elapsed=elapsed_time,
                                tokens_per_second=tokens_generated / elapsed_time if elapsed_time > 0 else 0,
                                gpu_utilization=0.0,  # Skip during generation
                                memory_used_gb=0.0,   # Skip during generation
                                first_token_latency=first_token_time or 0
                            )
                            last_metrics_update = current_time
                        
                        yield delta
                        
                        if any(stop in token for stop in request.stop_sequences):
                            break
                    
                    thread.join()
                    
                else:
                    outputs = model.generate(
                        **inputs,
                        **generation_config
                    )
                    
                    generated_text = tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
                    first_token_time = (time.time() - start_time) / tokens_generated if tokens_generated > 0 else 0
                    
                    yield generated_text
            
            elapsed_time = time.time() - start_time
            
            # Stop profiling and get final results
            profile_result = self.profiler.stop_profiling()
            
            # Update final metrics
            self.current_metrics = GenerationMetrics(
                tokens_generated=tokens_generated,
                time_elapsed=elapsed_time,
                tokens_per_second=tokens_generated / elapsed_time if elapsed_time > 0 else 0,
                gpu_utilization=profile_result.gpu_utilization,
                memory_used_gb=profile_result.memory_used_mb / 1024,
                first_token_latency=first_token_time or 0
            )
            
        except Exception as e:
            self.status = InferenceStatus.ERROR
            self.profiler.stop_profiling()
            raise e
    
    def _generate_safetensors(
        self,
        model: Any,
        tokenizer: Any,
        request: GenerationRequest
    ) -> Generator[str, None, None]:
        """Generate using SafeTensors model (loaded as PyTorch) on MPS."""
        # SafeTensors models are loaded as PyTorch models, so use the same generation logic
        yield from self._generate_pytorch(model, tokenizer, request)
    
    def _generate_gguf(
        self,
        model: Any,
        tokenizer: Any,
        request: GenerationRequest
    ) -> Generator[str, None, None]:
        """Generate using GGUF model with llama-cpp-python."""
        start_time = time.time()
        first_token_time = None
        tokens_generated = 0
        
        try:
            # GGUF models use llama-cpp-python which has its own generation method
            # The model is a Llama object from llama-cpp-python
            
            # Generate response using llama-cpp's native method
            response = model(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repetition_penalty,
                stream=request.stream
            )
            
            if request.stream:
                normalizer = StreamDeltaNormalizer()
                # Stream tokens
                for chunk in response:
                    if self._cancel_event.is_set():
                        break
                    
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        token = chunk['choices'][0].get('text', '')
                        delta = normalizer.normalize(token)
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.time()
                            tokens_generated += 1
                            yield delta
            else:
                # Return full response
                if 'choices' in response and len(response['choices']) > 0:
                    text = response['choices'][0].get('text', '')
                    tokens_generated = len(text.split())  # Rough estimate
                    yield text
            
            # Calculate metrics
            end_time = time.time()
            time_elapsed = end_time - start_time
            
            self.current_metrics = GenerationMetrics(
                tokens_generated=tokens_generated,
                time_elapsed=time_elapsed,
                tokens_per_second=tokens_generated / time_elapsed if time_elapsed > 0 else 0,
                gpu_utilization=0.0,  # GGUF doesn't provide GPU metrics directly
                memory_used_gb=self.model_manager.get_memory_status().get('model_memory_gb', 0),
                first_token_latency=first_token_time - start_time if first_token_time else 0
            )
            
        except Exception as e:
            self.status = InferenceStatus.ERROR
            raise e
    
    async def generate_async(
        self,
        request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """
        Async generator for text generation.
        
        Args:
            request: Generation request parameters
            
        Yields:
            Generated text tokens
        """
        loop = asyncio.get_event_loop()
        queue = Queue()
        
        def generate_worker():
            try:
                for token in self.generate(request):
                    queue.put(token)
            except Exception as e:
                queue.put(e)
            finally:
                queue.put(None)
        
        thread = threading.Thread(target=generate_worker)
        thread.start()
        
        while True:
            result = await loop.run_in_executor(None, queue.get)
            
            if result is None:
                break
            elif isinstance(result, Exception):
                raise result
            else:
                yield result
        
        thread.join()
    
    def _supports_bfloat16(self) -> bool:
        """Check if system supports bfloat16."""
        try:
            test = mx.array([1.0], dtype=mx.bfloat16)
            mx.eval(test)
            return True
        except:
            return False
    
    def cancel_generation(self) -> None:
        """Cancel ongoing generation."""
        self._cancel_event.set()
        self.status = InferenceStatus.CANCELLED
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            import psutil
            process = psutil.Process()
            return min(process.cpu_percent() * 2, 100.0)
        except:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        try:
            if torch.backends.mps.is_available():
                allocated = torch.mps.current_allocated_memory()
                return allocated / (1024**3)
            return 0.0
        except:
            return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current inference status with MLX details."""
        status = {
            'status': self.status.value,
            'model': self.model_manager.current_model,
            'metrics': self.current_metrics.to_dict() if self.current_metrics else None,
            'gpu_memory_gb': self._get_memory_usage(),
            'gpu_utilization': self._get_gpu_utilization()
        }
        
        # Add MLX accelerator status
        if self.mlx_accelerator:
            status['mlx_accelerator'] = {
                'enabled': True,
                'amx': self.mlx_accelerator.config.use_amx,
                'operation_fusion': self.mlx_accelerator.config.fuse_operations,
                'lazy_evaluation': self.mlx_accelerator.config.lazy_evaluation,
                'kv_cache': self.mlx_accelerator.config.rotating_kv_cache,
                'kv_cache_size': self.mlx_accelerator.config.kv_cache_size,
                'quantization_bits': self.mlx_accelerator.config.quantization_bits
            }
        else:
            status['mlx_accelerator'] = {'enabled': False}
        
        return status
    
    def benchmark(
        self,
        prompt: str = "Once upon a time",
        num_tokens: int = 100
    ) -> GenerationMetrics:
        """
        Run a benchmark test.
        
        Args:
            prompt: Prompt to use for benchmark
            num_tokens: Number of tokens to generate
            
        Returns:
            Benchmark metrics
        """
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=num_tokens,
            temperature=0.7,
            stream=True
        )
        
        tokens = []
        for token in self.generate(request):
            tokens.append(token)
        
        return self.current_metrics
    
    def warmup(self) -> None:
        """Warm up the GPU with a small generation."""
        try:
            request = GenerationRequest(
                prompt="Hello",
                max_tokens=1,
                temperature=0.0,
                stream=False
            )
            
            for _ in self.generate(request):
                pass
            
        except Exception as e:
            print(f"Warning: GPU warmup failed: {e}")
