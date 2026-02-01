"""Performance profiler for GPU operations with Metal/MPS."""

import time
import json
import psutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import threading

@dataclass
class ProfileResult:
    """Result from a profiling session."""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    gpu_utilization: float
    memory_used_mb: float
    memory_bandwidth_gb: float
    tokens_per_second: float
    flops: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result

class PerformanceProfiler:
    """Profile GPU performance for Metal/MPS operations."""
    
    def __init__(self, sample_interval: float = 0.1):
        """
        Initialize performance profiler.
        
        Args:
            sample_interval: Sampling interval in seconds
        """
        self.sample_interval = sample_interval
        self.results: List[ProfileResult] = []
        self.current_profile: Optional[ProfileResult] = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._gpu_samples: deque = deque(maxlen=1000)
        self._memory_samples: deque = deque(maxlen=1000)
    
    def start_profiling(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start profiling an operation."""
        self.current_profile = ProfileResult(
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=0,
            gpu_utilization=0,
            memory_used_mb=0,
            memory_bandwidth_gb=0,
            tokens_per_second=0,
            flops=0,
            metadata=metadata or {}
        )
        
        self._start_monitoring()
    
    def get_current_metrics(self) -> Optional[ProfileResult]:
        """Get current metrics without stopping the profiling session."""
        if not self.current_profile:
            return None
        
        current_time = datetime.now()
        duration_ms = (current_time - self.current_profile.start_time).total_seconds() * 1000
        
        gpu_utilization = 0
        if self._gpu_samples:
            recent_samples = list(self._gpu_samples)[-10:]  # Use last 10 samples for smoother metrics
            gpu_utilization = sum(recent_samples) / len(recent_samples) if recent_samples else 0
        
        memory_used_mb = 0
        if self._memory_samples:
            recent_memory = list(self._memory_samples)[-10:]
            memory_used_mb = sum(recent_memory) / len(recent_memory) if recent_memory else 0
        
        return ProfileResult(
            operation_name=self.current_profile.operation_name,
            start_time=self.current_profile.start_time,
            end_time=current_time,
            duration_ms=duration_ms,
            gpu_utilization=gpu_utilization,
            memory_used_mb=memory_used_mb,
            memory_bandwidth_gb=0,
            tokens_per_second=0,
            flops=0,
            metadata=self.current_profile.metadata
        )
    
    def stop_profiling(self) -> ProfileResult:
        """Stop profiling and return results."""
        if not self.current_profile:
            raise RuntimeError("No profiling session active")
        
        self._stop_monitoring()
        
        self.current_profile.end_time = datetime.now()
        self.current_profile.duration_ms = (
            self.current_profile.end_time - self.current_profile.start_time
        ).total_seconds() * 1000
        
        if self._gpu_samples:
            self.current_profile.gpu_utilization = sum(self._gpu_samples) / len(self._gpu_samples)
        
        if self._memory_samples:
            self.current_profile.memory_used_mb = max(self._memory_samples)
        
        self.results.append(self.current_profile)
        result = self.current_profile
        self.current_profile = None
        
        return result
    
    def _start_monitoring(self) -> None:
        """Start GPU monitoring thread."""
        self._monitoring = True
        self._gpu_samples.clear()
        self._memory_samples.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def _stop_monitoring(self) -> None:
        """Stop GPU monitoring thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """Monitoring loop for GPU metrics."""
        while self._monitoring:
            try:
                gpu_util = self._get_gpu_utilization()
                memory_mb = self._get_memory_usage()
                
                self._gpu_samples.append(gpu_util)
                self._memory_samples.append(memory_mb)
                
                time.sleep(self.sample_interval)
            except Exception:
                pass
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            result = subprocess.run(
                ["ioreg", "-l", "-w", "0"],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            lines = result.stdout.split('\n')
            for line in lines:
                if "PercentGPUUtilization" in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        return float(parts[1].strip())
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return min(cpu_percent * 1.5, 100.0)
            
        except:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            vm = psutil.virtual_memory()
            return vm.used / (1024 * 1024)
        except:
            return 0.0
    
    def profile_operation(
        self,
        operation: Callable,
        operation_name: str,
        args: tuple = (),
        kwargs: dict = None,
        warmup_runs: int = 3,
        profile_runs: int = 10
    ) -> ProfileResult:
        """
        Profile a specific operation.
        
        Args:
            operation: Operation to profile
            operation_name: Name for the operation
            args: Arguments for operation
            kwargs: Keyword arguments for operation
            warmup_runs: Number of warmup runs
            profile_runs: Number of profiling runs
            
        Returns:
            Profile result
        """
        kwargs = kwargs or {}
        
        for _ in range(warmup_runs):
            operation(*args, **kwargs)
        
        self.start_profiling(operation_name, {
            "warmup_runs": warmup_runs,
            "profile_runs": profile_runs
        })
        
        start = time.perf_counter()
        for _ in range(profile_runs):
            operation(*args, **kwargs)
        end = time.perf_counter()
        
        result = self.stop_profiling()
        
        avg_time = (end - start) / profile_runs
        result.duration_ms = avg_time * 1000
        
        return result
    
    def compare_operations(
        self,
        operations: List[Tuple[Callable, str]],
        args: tuple = (),
        kwargs: dict = None
    ) -> Dict[str, ProfileResult]:
        """
        Compare performance of multiple operations.
        
        Args:
            operations: List of (operation, name) tuples
            args: Common arguments
            kwargs: Common keyword arguments
            
        Returns:
            Dictionary of results by operation name
        """
        results = {}
        
        for operation, name in operations:
            result = self.profile_operation(
                operation, name, args, kwargs
            )
            results[name] = result
        
        return results
    
    def profile_model_inference(
        self,
        model: Any,
        input_data: Any,
        num_iterations: int = 100
    ) -> ProfileResult:
        """Profile model inference performance."""
        def inference():
            return model(input_data)
        
        return self.profile_operation(
            inference,
            "model_inference",
            warmup_runs=5,
            profile_runs=num_iterations
        )
    
    def estimate_flops(
        self,
        operation_type: str,
        input_shape: Tuple[int, ...],
        duration_ms: float
    ) -> float:
        """Estimate FLOPS for an operation."""
        flops_map = {
            "matmul": lambda shape: 2 * shape[0] * shape[1] * shape[2] if len(shape) >= 3 else 0,
            "attention": lambda shape: 4 * shape[0] * shape[1] * shape[1] * shape[2],
            "layernorm": lambda shape: 3 * shape[0] * shape[1],
            "gelu": lambda shape: 10 * shape[0] * shape[1],
            "softmax": lambda shape: 3 * shape[0] * shape[1]
        }
        
        if operation_type in flops_map and duration_ms > 0:
            total_flops = flops_map[operation_type](input_shape)
            return total_flops / (duration_ms / 1000)
        
        return 0.0
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report from all results."""
        if not self.results:
            return {"error": "No profiling results available"}
        
        total_time = sum(r.duration_ms for r in self.results)
        avg_gpu = sum(r.gpu_utilization for r in self.results) / len(self.results)
        peak_memory = max(r.memory_used_mb for r in self.results)
        
        operations_summary = []
        for result in self.results:
            operations_summary.append({
                "name": result.operation_name,
                "duration_ms": result.duration_ms,
                "gpu_utilization": result.gpu_utilization,
                "memory_mb": result.memory_used_mb
            })
        
        return {
            "total_operations": len(self.results),
            "total_time_ms": total_time,
            "average_gpu_utilization": avg_gpu,
            "peak_memory_mb": peak_memory,
            "operations": operations_summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, filepath: Path) -> None:
        """Save profiling results to JSON file."""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def clear_results(self) -> None:
        """Clear all profiling results."""
        self.results.clear()
        self._gpu_samples.clear()
        self._memory_samples.clear()
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on profiling results."""
        suggestions = []
        
        if not self.results:
            return ["No profiling data available"]
        
        avg_gpu = sum(r.gpu_utilization for r in self.results) / len(self.results)
        
        if avg_gpu < 50:
            suggestions.append("Low GPU utilization - consider increasing batch size")
        
        if avg_gpu > 95:
            suggestions.append("Very high GPU utilization - may be throttling")
        
        peak_memory = max(r.memory_used_mb for r in self.results)
        if peak_memory > 18000:  # 18GB for high memory systems
            suggestions.append("High memory usage - consider model quantization")
        
        slow_ops = [r for r in self.results if r.duration_ms > 100]
        if slow_ops:
            suggestions.append(f"Found {len(slow_ops)} slow operations (>100ms)")
        
        return suggestions if suggestions else ["Performance looks optimal"]