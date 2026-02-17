# MLX Acceleration Guide

## Overview

Cortex leverages MLX (Apple's machine learning framework) as its primary acceleration backend, providing native Metal GPU support with exceptional performance on Apple Silicon. This guide covers the MLX integration, features, and optimization techniques.

## Key Features

### AMX Coprocessor Support

The Apple Matrix coprocessor (AMX) is utilized for matrix operations:
- **32x32 tile operations** for high throughput
- **Automatic optimization** for matrix dimensions
- **Zero configuration** required

### Advanced Quantization

MLX supports multiple quantization strategies for optimal performance vs quality tradeoffs:

#### Quantization Recipes

1. **4-bit Quantization** (`SPEED_4BIT`)
   - Maximum inference speed
   - Large model size reduction
   - Best for interactive applications
   - Actual throughput varies by model and chip

2. **5-bit Quantization** (`BALANCED_5BIT`)
   - Balanced speed and quality
   - Solid size reduction
   - Good for general use

3. **8-bit Quantization** (`QUALITY_8BIT`)
   - Higher quality output
   - Moderate size reduction
   - Best for quality-critical applications

4. **Mixed Precision** (`MIXED_PRECISION`)
   - Per-layer quantization
   - Critical layers (embeddings, output) at higher precision
   - FFN and attention layers at lower precision
   - Optimal quality/performance balance

#### Group-wise Quantization

- **64-group quantization** for better quality
- **Per-group scaling** factors
- **Minimal quality loss** compared to uniform quantization

### Rotating KV Cache

Efficient handling of long context windows:
- **Configurable cache size** (wired to `context_length` in `config.yaml`, default 8192)
- **Automatic rotation** when context exceeds limit
- **Memory efficient** for extended conversations
- **Minimal overhead** on short contexts

### Operation Fusion

MLX automatically fuses operations for efficiency:
- **Fused attention** mechanisms
- **Combined matmul-activation** operations
- **Reduced kernel launches**
- **Lower memory bandwidth** usage

### JIT Compilation

Models are JIT compiled for optimal performance:
- **mx.compile** decorator for hot paths
- **Graph optimization** for complex models
- **Cached compilation** results
- **Performance improves** after warmup due to compilation and caching

## Model Conversion

### Converting HuggingFace Models

```python
from cortex.metal.mlx_converter import MLXConverter, ConversionConfig, QuantizationRecipe

# Initialize converter
converter = MLXConverter()

# Configure conversion
config = ConversionConfig(
    quantization=QuantizationRecipe.SPEED_4BIT,
    use_amx=True,
    compile_model=True,
    group_size=64  # Quantization group size
)

# Convert model
success, message, output_path = converter.convert_model(
    "microsoft/DialoGPT-medium",
    config=config
)
```

### Mixed Precision Configuration

```python
# Custom mixed precision setup
mixed_config = {
    "critical_layers": ["lm_head", "embed_tokens", "wte", "wpe"],
    "critical_bits": 6,  # Higher precision for critical layers
    "standard_bits": 4   # Standard precision for other layers
}

config = ConversionConfig(
    quantization=QuantizationRecipe.MIXED_PRECISION,
    mixed_precision_config=mixed_config
)
```

### Chip-Specific Optimization

Models can be optimized for specific Apple Silicon chips:

```python
# Optimize for specific chip
converter.optimize_for_chip(output_path, "m3_pro")
```

Supported chips:
- M1, M1 Pro, M1 Max, M1 Ultra
- M2, M2 Pro, M2 Max, M2 Ultra
- M3, M3 Pro, M3 Max
- M4, M4 Pro, M4 Max

Note: `optimize_for_chip(...)` writes chip metadata into `config.json`. It does not alter weights; it is used as a hint for downstream tooling.

## Performance Optimization

### MLX Accelerator Configuration

```python
from cortex.metal.mlx_accelerator import MLXAccelerator, MLXConfig

# Configure acceleration
config = MLXConfig(
    compile_model=True,        # JIT compilation
    use_graph=True,           # Graph optimization
    batch_size=8,             # Inference batch size
    stream_parallel=True,     # Stream parallelism
    use_amx=True,            # AMX coprocessor
    fuse_operations=True,    # Operation fusion
    rotating_kv_cache=True,  # Rotating cache
    kv_cache_size=4096,      # Max cache size
    quantization_bits=4,     # Default quantization
    dtype=mx.bfloat16       # Precision (bfloat16 or float16)
)

# Initialize accelerator
accelerator = MLXAccelerator(config)

# Optimize model
optimized_model = accelerator.optimize_model(model)
```

### Optimization Presets

```python
# Use preset configurations
accelerator = MLXAccelerator(
    MLXConfig(**MLXAccelerator.OPTIMIZATION_PRESETS["speed"])
)
```

Available presets:
- **"speed"**: Maximum performance, bfloat16 precision
- **"memory"**: Memory efficient, reduced features
- **"balanced"**: Balanced features, float32 precision

### Transformer-Specific Optimizations

```python
# Apply transformer optimizations
optimized_model = accelerator.accelerate_transformer(
    model,
    num_heads=12,
    head_dim=64
)
```

Features:
- Fused attention with AMX acceleration
- Optimized KV caching
- Efficient softmax computation
- Reduced memory transfers

## Benchmarking

### Performance Profiling

```python
# Profile model performance
profile = accelerator.profile_model(
    model,
    input_shape=(1, 512),  # Batch size, sequence length
    num_iterations=100
)

print(f"Average inference time: {profile['avg_inference_time']:.3f}s")
print(f"Throughput: {profile['throughput']:.1f} samples/s")
print(f"Parameters: {profile['num_parameters']:,}")
```

### Operation Benchmarking

```python
# Benchmark specific operations
import mlx.core as mx

def matmul_op(x):
    return mx.matmul(x, x.T)

results = MLXAccelerator.benchmark_operation(
    matmul_op,
    input_shape=(1024, 1024),
    num_iterations=1000,
    use_amx=True
)

print(f"GFLOPS: {results['throughput_gflops']:.1f}")
```

## Memory Management

### Optimized Memory Allocation

```python
# Configure memory usage
config = MLXConfig(
    memory_fraction=0.85,  # Use 85% of available memory
    fusion_threshold=1024  # Minimum size for operation fusion
)
```

### Memory Optimization for Large Models

```python
# Optimize memory for large models
optimized_model = accelerator.optimize_memory(model)
```

Features:
- Weight sharding for large tensors
- Automatic memory pool management
- Zero-copy operations where possible

## Debugging and Monitoring

### Check MLX Device

```python
import mlx.core as mx

# Verify GPU is being used
device = mx.default_device()
print(f"Device: {device}")  # Should show Device(gpu, 0)

# Get device info
info = MLXAccelerator.get_device_info()
print(f"Is GPU: {info['is_gpu']}")
```

### Monitor Performance

```python
# Real-time performance monitoring
for token in accelerator.generate_optimized(
    model, tokenizer, prompt,
    max_tokens=100,
    stream=True
):
    # Token generation with performance metrics
    print(token, end="", flush=True)
```

## Best Practices

### 1. Model Selection
- Use MLX-native models when available
- Convert HuggingFace models to MLX format
- Choose appropriate quantization for your use case

### 2. Memory Management
- Monitor memory usage with Activity Monitor
- Adjust `memory_fraction` if experiencing issues
- Use quantization for large models

### 3. Performance Tuning
- Enable JIT compilation for production
- Use operation fusion for complex models
- Profile and benchmark your specific workload

### 4. Context Management
- Configure KV cache size based on expected context
- Use rotating cache for long conversations
- Monitor context usage to prevent overflow

## Troubleshooting

### Common Issues

**MLX not using GPU:**
```python
# Force GPU usage
import mlx.core as mx
mx.set_default_device(mx.gpu)
```

**Memory errors with large models:**
```python
# Reduce memory allocation
config = MLXConfig(memory_fraction=0.7)
```

**Slow first inference:**
- This is normal due to JIT compilation
- Subsequent inferences will be faster
- Run warmup iterations in production

**Quantization quality issues:**
- Try higher bit quantization (5-bit or 8-bit)
- Use mixed precision for critical layers
- Adjust group_size parameter (32 or 128)

## Advanced Topics

### Custom Quantization Predicates

```python
def custom_quantization_predicate(layer_path, layer, model_config):
    """Define custom quantization per layer."""
    if "attention" in layer_path:
        return {"bits": 6, "group_size": 64}
    elif "mlp" in layer_path:
        return {"bits": 4, "group_size": 64}
    else:
        return False  # No quantization

config = ConversionConfig(
    quantization=QuantizationRecipe.MIXED_PRECISION,
    mixed_precision_config={"predicate": custom_quantization_predicate}
)
```

### Pipeline Creation

```python
# Create optimized inference pipeline
pipeline = accelerator.create_pipeline(
    [model1, model2, model3],
    batch_size=4
)

# Run pipeline
output = pipeline(input_data)
```

### Stream Parallelism

```python
# Enable stream parallelism for concurrent operations
config = MLXConfig(
    stream_parallel=True,
    prefetch_size=2
)
```

## Performance Expectations

Actual performance varies by model size, quantization, context length, and chip. Use `/benchmark` inside Cortex or `MLXAccelerator.profile_model()` to measure on your machine.

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [Apple Developer - Metal](https://developer.apple.com/metal/)
- [Cortex GitHub](https://github.com/faisalmumtaz89/Cortex)
