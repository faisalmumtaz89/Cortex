"""Tests for Metal optimization components."""

import pytest
import sys
import time
from pathlib import Path
import torch
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortex.metal import (
    MetalCapabilities,
    check_metal_support,
)
from cortex.metal.memory_pool import MemoryPool, AllocationStrategy
from cortex.metal.mps_optimizer import MPSOptimizer, MPSConfig
from cortex.metal.mlx_accelerator import MLXAccelerator, MLXConfig
from cortex.metal.performance_profiler import PerformanceProfiler

class TestMetalCapabilities:
    """Test Metal capabilities detection."""
    
    def test_detect_capabilities(self):
        """Test capability detection."""
        capabilities = MetalCapabilities.detect_capabilities()
        
        if sys.platform == "darwin":
            assert capabilities["supported"] is True
            assert "version" in capabilities
            assert "features" in capabilities
            assert "gpu_family" in capabilities
        else:
            assert capabilities["supported"] is False
    
    def test_optimization_profile(self):
        """Test optimization profile retrieval."""
        profile = MetalCapabilities.get_optimization_profile("apple8")
        
        assert profile["max_threads_per_threadgroup"] == 1024
        assert profile["simd_width"] == 32
        assert profile["preferred_batch_size"] == 8
        assert profile["use_fused_operations"] is True
    
    def test_check_metal_support(self):
        """Test Metal support check."""
        if sys.platform == "darwin":
            assert check_metal_support() is True
        else:
            assert check_metal_support() is False


class TestMemoryPool:
    """Test GPU memory pool management."""
    
    def setup_method(self):
        """Setup test memory pool."""
        if torch.backends.mps.is_available():
            self.pool = MemoryPool(
                pool_size=1024 * 1024 * 100,  # 100MB
                strategy=AllocationStrategy.UNIFIED,
                device="mps"
            )
        else:
            pytest.skip("MPS not available")
    
    def test_memory_allocation(self):
        """Test memory allocation from pool."""
        tensor = self.pool.allocate(1024 * 1024)  # 1MB
        
        assert tensor is not None
        stats = self.pool.get_stats()
        assert stats["allocated_memory"] > 0
    
    def test_memory_deallocation(self):
        """Test memory deallocation."""
        tensor = self.pool.allocate(1024 * 1024)
        initial_allocated = self.pool.allocated_memory
        
        success = self.pool.deallocate(tensor)
        assert success is True
        assert self.pool.allocated_memory < initial_allocated
    
    def test_allocation_strategies(self):
        """Test different allocation strategies."""
        for strategy in [AllocationStrategy.BEST_FIT, AllocationStrategy.FIRST_FIT]:
            pool = MemoryPool(
                pool_size=1024 * 1024 * 10,
                strategy=strategy,
                device="mps"
            )
            
            tensor = pool.allocate(1024 * 100)
            assert tensor is not None
    
    def test_memory_stats(self):
        """Test memory statistics."""
        stats = self.pool.get_stats()
        
        assert "pool_size" in stats
        assert "allocated_memory" in stats
        assert "free_memory" in stats
        assert "fragmentation" in stats

class TestMPSOptimizer:
    """Test MPS backend optimization."""
    
    def setup_method(self):
        """Setup test MPS optimizer."""
        if torch.backends.mps.is_available():
            self.optimizer = MPSOptimizer()
        else:
            pytest.skip("MPS not available")
    
    def test_optimize_model(self):
        """Test model optimization."""
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        )
        
        optimized = self.optimizer.optimize_model(model)
        
        assert optimized is not None
        assert next(optimized.parameters()).device.type == "mps"
    
    def test_fp16_conversion(self):
        """Test FP16 conversion."""
        config = MPSConfig(use_fp16=True)
        optimizer = MPSOptimizer(config)
        
        model = torch.nn.Linear(10, 10)
        optimized = optimizer.optimize_model(model)
        
        assert next(optimized.parameters()).dtype == torch.float16
    
    def test_channels_last_conversion(self):
        """Test channels_last memory format."""
        model = torch.nn.Conv2d(3, 16, 3)
        optimized = self.optimizer.optimize_model(model)
        
        assert optimized is not None
    
    def test_mps_info(self):
        """Test MPS information retrieval."""
        info = MPSOptimizer.get_mps_info()
        
        assert info["available"] is True
        assert info["built"] is True

class TestMLXAccelerator:
    """Test MLX framework acceleration."""
    
    def setup_method(self):
        """Setup test MLX accelerator."""
        try:
            self.accelerator = MLXAccelerator()
        except RuntimeError:
            pytest.skip("MLX not available on GPU")
    
    def test_optimize_model(self):
        """Test MLX model optimization."""
        import mlx.nn as nn
        
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU()
        )
        
        optimized = self.accelerator.optimize_model(model)
        assert optimized is not None
    
    def test_dtype_optimization(self):
        """Test dtype optimization."""
        config = MLXConfig(dtype=mx.float16)
        accelerator = MLXAccelerator(config)
        
        assert accelerator.config.dtype == mx.float16
    
    def test_device_info(self):
        """Test device information."""
        info = MLXAccelerator.get_device_info()
        
        assert "device" in info
        assert "is_gpu" in info


class TestPerformanceProfiler:
    """Test performance profiling."""
    
    def setup_method(self):
        """Setup test profiler."""
        self.profiler = PerformanceProfiler(sample_interval=0.01)
    
    def test_profile_operation(self):
        """Test operation profiling."""
        def test_op():
            time.sleep(0.01)
            return sum(range(1000))
        
        result = self.profiler.profile_operation(
            test_op,
            "test_operation",
            warmup_runs=1,
            profile_runs=3
        )
        
        assert result is not None
        assert result.operation_name == "test_operation"
        assert result.duration_ms > 0
    
    def test_compare_operations(self):
        """Test operation comparison."""
        def op1():
            return sum(range(100))
        
        def op2():
            return sum(range(1000))
        
        results = self.profiler.compare_operations([
            (op1, "small_sum"),
            (op2, "large_sum")
        ])
        
        assert "small_sum" in results
        assert "large_sum" in results
    
    def test_generate_report(self):
        """Test report generation."""
        self.profiler.profile_operation(
            lambda: sum(range(100)),
            "test_op",
            warmup_runs=1,
            profile_runs=1
        )
        
        report = self.profiler.generate_report()
        
        assert "total_operations" in report
        assert report["total_operations"] == 1
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        suggestions = self.profiler.get_optimization_suggestions()
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

@pytest.mark.integration
class TestMetalIntegration:
    """Integration tests for Metal optimization components."""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        if sys.platform != "darwin":
            pytest.skip("Metal only available on macOS")
        
        # GPU acceleration uses MLX/MPS backends
        
        # Test memory pool
        if torch.backends.mps.is_available():
            pool = MemoryPool(
                pool_size=1024 * 1024 * 100,
                device="mps"
            )
            tensor = pool.allocate(1024 * 1024)
            assert tensor is not None
        
        # Test performance profiling
        profiler = PerformanceProfiler()
        result = profiler.profile_operation(
            lambda: sum(range(1000)),
            "integration_test"
        )
        assert result.duration_ms > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])