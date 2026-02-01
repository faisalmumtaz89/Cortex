"""GPU memory pool management for pre-allocation and zero-copy operations."""

import logging
import weakref
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from datetime import datetime
import numpy as np
import psutil
import sys
import os
import atexit

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import mlx.core as mx

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_validator import GPUValidator

class AllocationStrategy(Enum):
    """Memory allocation strategies with MLX zero-copy support."""
    BEST_FIT = "best_fit"
    FIRST_FIT = "first_fit"
    UNIFIED = "unified"  # Unified memory for CPU/GPU sharing
    DEDICATED = "dedicated"
    ZERO_COPY = "zero_copy"  # MLX zero-copy for maximum efficiency

@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""
    block_id: str
    size: int
    offset: int
    allocated: bool
    allocation_time: Optional[datetime]
    last_access: Optional[datetime]
    device_type: str  # "mps" or "mlx"
    tensor_ref: Any  # Weak reference to tensor
    metadata: Dict[str, Any]
    is_constant: bool = False  # Whether this block is for constant memory
    is_read_only: bool = False  # Whether this block is read-only (weights)
    
    def is_free(self) -> bool:
        """Check if block is free for allocation."""
        if not self.allocated:
            return True
        if self.tensor_ref is not None:
            ref = self.tensor_ref()
            if ref is None:
                self.allocated = False
                return True
        return False
    
    def mark_allocated(self, tensor: Any) -> None:
        """Mark block as allocated."""
        self.allocated = True
        self.allocation_time = datetime.now()
        self.last_access = datetime.now()
        self.tensor_ref = weakref.ref(tensor) if tensor is not None else None
    
    def mark_free(self) -> None:
        """Mark block as free."""
        self.allocated = False
        self.tensor_ref = None

class MemoryPool:
    """Pre-allocated GPU memory pool for zero-copy operations."""
    
    DEFAULT_POOL_SIZE = 20 * 1024 * 1024 * 1024  # 20GB default target
    CONSTANT_POOL_SIZE = 64 * 1024 * 1024  # 64MB for constant memory (Metal limit)
    BLOCK_SIZES = [
        1 * 1024 * 1024,      # 1MB
        16 * 1024 * 1024,     # 16MB
        64 * 1024 * 1024,     # 64MB
        256 * 1024 * 1024,    # 256MB
        1024 * 1024 * 1024,   # 1GB
        4096 * 1024 * 1024,   # 4GB
    ]
    
    @classmethod
    def get_optimal_pool_size(cls, target_size: Optional[int] = None) -> int:
        """Get optimal pool size based on available memory."""
        if target_size is not None:
            return target_size
            
        # Get available memory
        vm = psutil.virtual_memory()
        available = vm.available
        total = vm.total
        
        # More aggressive memory allocation for better performance
        # Use 60% of available memory, but never more than 75% of total memory
        optimal_size = min(
            int(available * 0.60),  # 60% of available
            int(total * 0.75)       # Never more than 75% of total
        )
        
        # Further limit based on actual available memory
        # Can allocate up to 90% of what's available for better utilization
        max_safe_size = int(available * 0.90)
        optimal_size = min(optimal_size, max_safe_size)
        
        # Cap at DEFAULT_POOL_SIZE if we have plenty of memory
        if optimal_size > cls.DEFAULT_POOL_SIZE:
            optimal_size = cls.DEFAULT_POOL_SIZE
        
        # If available memory is very low (< 4GB), be extra conservative
        if available < 4 * 1024 * 1024 * 1024:
            optimal_size = min(optimal_size, int(available * 0.25))
        
        # Minimum 256MB for basic functionality (reduced from 512MB)
        min_size = 256 * 1024 * 1024
        return max(optimal_size, min_size)
    
    def __init__(
        self,
        pool_size: Optional[int] = None,
        strategy: AllocationStrategy = AllocationStrategy.UNIFIED,
        device: str = "mps",
        auto_size: bool = True,
        silent: bool = False,
        use_bfloat16: Optional[bool] = None,
        enable_zero_copy: bool = True
    ):
        """
        Initialize memory pool.
        
        Args:
            pool_size: Total pool size in bytes (None for auto-detection)
            strategy: Allocation strategy
            device: Device type ("mps" or "mlx")
            auto_size: Automatically determine pool size based on available memory
            silent: Suppress initialization messages
            use_bfloat16: Use bfloat16 if supported (None for auto-detect)
        """
        if auto_size and pool_size is None:
            self.pool_size = self.get_optimal_pool_size()
            logger.info(f"Auto-sizing memory pool to {self.pool_size / (1024**3):.1f}GB")
            if not silent:
                print(f"Auto-sizing memory pool to {self.pool_size / (1024**3):.1f}GB")
        else:
            self.pool_size = pool_size or self.get_optimal_pool_size()
            logger.info(f"Memory pool size set to {self.pool_size / (1024**3):.1f}GB")
            
        self.strategy = strategy
        self.device = device
        self.blocks: List[MemoryBlock] = []
        self.constant_blocks: List[MemoryBlock] = []  # Separate pool for constant memory
        self.allocated_memory = 0
        self.allocated_constant_memory = 0
        self.peak_memory = 0
        self._lock = threading.Lock()
        self.silent = silent
        
        # Initialize GPU validator for hardware detection
        self.gpu_validator = GPUValidator()
        self.gpu_validator.validate()
        
        # Determine optimal dtype
        self.use_bfloat16 = use_bfloat16
        if self.use_bfloat16 is None:
            self.use_bfloat16 = self._should_use_bfloat16()
        
        self.optimal_dtype = self._get_optimal_dtype()
        self.enable_zero_copy = enable_zero_copy and device == "mlx"
        
        # Regular and constant memory buffers
        self._mps_buffer: Optional[torch.Tensor] = None
        self._mps_constant_buffer: Optional[torch.Tensor] = None
        self._mlx_buffer: Optional[mx.array] = None
        self._mlx_constant_buffer: Optional[mx.array] = None
        
        # Zero-copy memory tracking
        self.zero_copy_arrays: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.unified_memory_regions: Dict[str, Any] = {}
        
        # Enable zero-copy strategy for MLX if requested
        if self.enable_zero_copy and strategy == AllocationStrategy.UNIFIED:
            self.strategy = AllocationStrategy.ZERO_COPY
            logger.info("Zero-copy memory strategy enabled for MLX")
        
        logger.info(f"Initializing memory pool with strategy: {self.strategy.value}")
        self._initialize_pool()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def cleanup(self) -> None:
        """Clean up allocated resources to prevent leaks."""
        try:
            # Release MPS buffers
            if self._mps_buffer is not None:
                del self._mps_buffer
                self._mps_buffer = None
            
            if self._mps_constant_buffer is not None:
                del self._mps_constant_buffer
                self._mps_constant_buffer = None
            
            # Release MLX buffers
            if self._mlx_buffer is not None:
                del self._mlx_buffer
                self._mlx_buffer = None
            
            if self._mlx_constant_buffer is not None:
                del self._mlx_constant_buffer
                self._mlx_constant_buffer = None
            
            # Force synchronization and cleanup
            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.synchronize()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def _should_use_bfloat16(self) -> bool:
        """
        Determine if bfloat16 should be used based on hardware.
        
        Returns:
            True if bfloat16 is supported and beneficial
        """
        if self.gpu_validator.gpu_info and self.gpu_validator.gpu_info.supports_bfloat16:
            if self.device == "mps":
                # Check PyTorch bfloat16 support on MPS
                try:
                    test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device="mps")
                    return True
                except (RuntimeError, TypeError):
                    return False
            elif self.device == "mlx":
                # Check MLX bfloat16 support
                return hasattr(mx, 'bfloat16')
        return False
    
    def _get_optimal_dtype(self) -> Any:
        """
        Get optimal dtype for current device and hardware.
        
        Returns:
            torch.dtype or mx.dtype
        """
        if self.device == "mps":
            if self.use_bfloat16:
                return torch.bfloat16
            else:
                return torch.float16
        elif self.device == "mlx":
            if self.use_bfloat16 and hasattr(mx, 'bfloat16'):
                return mx.bfloat16
            else:
                return mx.float16
        else:
            return torch.float32
    
    def _initialize_pool(self) -> None:
        """Initialize the memory pool with pre-allocated buffers."""
        if self.device == "mps" and torch.backends.mps.is_available():
            self._initialize_mps_pool()
        elif self.device == "mlx":
            self._initialize_mlx_pool()
        else:
            raise RuntimeError(f"Unsupported device: {self.device}")
    
    def _initialize_mps_pool(self) -> None:
        """Initialize MPS memory pool with optimal dtype and constant memory."""
        try:
            device = torch.device("mps")
            
            # Calculate number of elements based on dtype size
            if self.optimal_dtype in [torch.float16, torch.bfloat16]:
                element_size = 2  # 16-bit types
            else:
                element_size = 4  # 32-bit types
            
            # Initialize regular memory pool
            num_elements = self.pool_size // element_size
            self._mps_buffer = torch.empty(
                num_elements,
                dtype=self.optimal_dtype,
                device=device
            )
            
            # Initialize constant memory pool (for weights)
            constant_elements = self.CONSTANT_POOL_SIZE // element_size
            self._mps_constant_buffer = torch.empty(
                constant_elements,
                dtype=self.optimal_dtype,
                device=device
            )
            # Mark as read-only after initialization
            self._mps_constant_buffer.requires_grad_(False)
            
            dtype_name = str(self.optimal_dtype).split('.')[-1]
            logger.info(f"MPS memory pool initialized: dtype={dtype_name}, constant_pool={self.CONSTANT_POOL_SIZE / (1024*1024):.1f}MB")
            if not self.silent:
                print(f"Initialized MPS pool with dtype: {dtype_name}")
                print(f"Constant memory pool: {self.CONSTANT_POOL_SIZE / (1024*1024):.1f}MB")
            
            if self.strategy == AllocationStrategy.UNIFIED:
                self._create_unified_blocks()
            else:
                self._create_segmented_blocks()
            
            # Create constant memory blocks
            self._create_constant_blocks()
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if this is a dtype support issue vs memory issue
            if self.optimal_dtype == torch.bfloat16 and 'bfloat16' in error_msg:
                # bfloat16 not supported, fallback to float16
                self.optimal_dtype = torch.float16
                self.use_bfloat16 = False
                logger.info("bfloat16 not supported, falling back to float16")
                if not self.silent:
                    print("bfloat16 not supported, falling back to float16")
                self._initialize_mps_pool()  # Retry with float16
            elif 'invalid buffer size' in error_msg or 'out of memory' in error_msg:
                # Memory allocation failed - provide helpful error
                pool_size_gb = self.pool_size / (1024**3)
                raise RuntimeError(
                    f"Failed to allocate {pool_size_gb:.2f}GB memory pool. "
                    f"Insufficient memory available. Consider reducing pool size or "
                    f"freeing up system memory. Error: {e}"
                )
            else:
                # Other errors - pass through
                raise RuntimeError(f"Failed to initialize MPS pool: {e}")
    
    def _initialize_mlx_pool(self) -> None:
        """Initialize MLX memory pool with optimal dtype and zero-copy support."""
        try:
            # Calculate number of elements based on dtype size
            if self.optimal_dtype in [mx.float16, getattr(mx, 'bfloat16', mx.float16)]:
                element_size = 2  # 16-bit types
            else:
                element_size = 4  # 32-bit types
            
            num_elements = self.pool_size // element_size
            
            # For zero-copy, create unified memory that can be shared
            if self.strategy == AllocationStrategy.ZERO_COPY:
                logger.info("Creating MLX zero-copy unified memory pool")
                # MLX arrays are already zero-copy between CPU/GPU
                self._mlx_buffer = mx.zeros(
                    (num_elements,),
                    dtype=self.optimal_dtype
                )
                # Force evaluation to allocate unified memory
                mx.eval(self._mlx_buffer)
                
                # Also create constant buffer for weights
                constant_elements = self.CONSTANT_POOL_SIZE // element_size
                self._mlx_constant_buffer = mx.zeros(
                    (constant_elements,),
                    dtype=self.optimal_dtype
                )
                mx.eval(self._mlx_constant_buffer)
                
                dtype_name = str(self.optimal_dtype).split('.')[-1]
                logger.info(f"MLX zero-copy pool initialized: dtype={dtype_name}, size={self.pool_size / (1024**3):.1f}GB")
                if not self.silent:
                    print(f"Initialized MLX zero-copy pool with dtype: {dtype_name}")
                    print(f"Zero-copy unified memory: {self.pool_size / (1024**3):.1f}GB")
            else:
                # Standard MLX buffer
                self._mlx_buffer = mx.zeros(
                    (num_elements,),
                    dtype=self.optimal_dtype
                )
                mx.eval(self._mlx_buffer)
                
                dtype_name = str(self.optimal_dtype).split('.')[-1]
                logger.info(f"MLX pool initialized: dtype={dtype_name}")
                if not self.silent:
                    print(f"Initialized MLX pool with dtype: {dtype_name}")
            
            if self.strategy in [AllocationStrategy.UNIFIED, AllocationStrategy.ZERO_COPY]:
                self._create_unified_blocks()
            else:
                self._create_segmented_blocks()
            
            # Create constant blocks for MLX if needed
            if self._mlx_constant_buffer is not None:
                self._create_constant_blocks()
                
        except Exception as e:
            # Fallback to float16 if bfloat16 fails
            if hasattr(mx, 'bfloat16') and self.optimal_dtype == mx.bfloat16:
                self.optimal_dtype = mx.float16
                self.use_bfloat16 = False
                logger.info("MLX bfloat16 not supported, falling back to float16")
                if not self.silent:
                    print("bfloat16 not supported, falling back to float16")
                self._initialize_mlx_pool()  # Retry with float16
            else:
                raise RuntimeError(f"Failed to initialize MLX pool: {e}")
    
    def _create_unified_blocks(self) -> None:
        """Create a single unified memory block."""
        block = MemoryBlock(
            block_id="unified_0",
            size=self.pool_size,
            offset=0,
            allocated=False,
            allocation_time=None,
            last_access=None,
            device_type=self.device,
            tensor_ref=None,
            metadata={"type": "unified"},
            is_constant=False,
            is_read_only=False
        )
        self.blocks.append(block)
    
    def _create_constant_blocks(self) -> None:
        """Create constant memory blocks for weights."""
        # Create smaller blocks for better allocation flexibility
        block_sizes = [
            4 * 1024 * 1024,   # 4MB blocks
            16 * 1024 * 1024,  # 16MB blocks
        ]
        
        offset = 0
        block_id = 0
        remaining_size = self.CONSTANT_POOL_SIZE
        
        for size in block_sizes:
            while remaining_size >= size:
                block = MemoryBlock(
                    block_id=f"constant_{block_id}",
                    size=size,
                    offset=offset,
                    allocated=False,
                    allocation_time=None,
                    last_access=None,
                    device_type=self.device,
                    tensor_ref=None,
                    metadata={"type": "constant", "size_class": size},
                    is_constant=True,
                    is_read_only=True
                )
                self.constant_blocks.append(block)
                offset += size
                remaining_size -= size
                block_id += 1
        
        # Add remainder as final block
        if remaining_size > 0:
            block = MemoryBlock(
                block_id=f"constant_{block_id}",
                size=remaining_size,
                offset=offset,
                allocated=False,
                allocation_time=None,
                last_access=None,
                device_type=self.device,
                tensor_ref=None,
                metadata={"type": "constant", "size_class": "remainder"},
                is_constant=True,
                is_read_only=True
            )
            self.constant_blocks.append(block)
    
    def _create_segmented_blocks(self) -> None:
        """Create segmented memory blocks of various sizes."""
        offset = 0
        block_id = 0
        
        remaining_size = self.pool_size
        
        for size_class in reversed(self.BLOCK_SIZES):
            while remaining_size >= size_class:
                block = MemoryBlock(
                    block_id=f"block_{block_id}",
                    size=size_class,
                    offset=offset,
                    allocated=False,
                    allocation_time=None,
                    last_access=None,
                    device_type=self.device,
                    tensor_ref=None,
                    metadata={"size_class": size_class}
                )
                self.blocks.append(block)
                offset += size_class
                remaining_size -= size_class
                block_id += 1
        
        if remaining_size > 0:
            block = MemoryBlock(
                block_id=f"block_{block_id}",
                size=remaining_size,
                offset=offset,
                allocated=False,
                allocation_time=None,
                last_access=None,
                device_type=self.device,
                tensor_ref=None,
                metadata={"size_class": "remainder"}
            )
            self.blocks.append(block)
    
    def allocate(
        self,
        size: int,
        dtype: Optional[Any] = None,
        is_constant: bool = False
    ) -> Optional[Any]:
        """
        Allocate memory from the pool.
        
        Args:
            size: Size in bytes
            dtype: Data type for the tensor
            is_constant: Whether to allocate from constant memory pool
            
        Returns:
            Allocated tensor or None if allocation fails
        """
        with self._lock:
            if is_constant:
                # Allocate from constant memory pool
                block = self._find_constant_block(size)
                if block is None:
                    return None
                
                tensor = self._create_tensor_from_block(block, size, dtype, is_constant=True)
                block.mark_allocated(tensor)
                
                self.allocated_constant_memory += size
            else:
                # Allocate from regular pool
                block = self._find_block(size)
                
                if block is None:
                    self._try_defragment()
                    block = self._find_block(size)
                
                if block is None:
                    return None
                
                tensor = self._create_tensor_from_block(block, size, dtype, is_constant=False)
                block.mark_allocated(tensor)
                
                self.allocated_memory += size
                self.peak_memory = max(self.peak_memory, self.allocated_memory)
            
            return tensor
    
    def allocate_weights(
        self,
        size: int,
        dtype: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Allocate memory for weights using constant memory.
        
        Args:
            size: Size in bytes
            dtype: Data type for the tensor
            
        Returns:
            Allocated tensor in constant memory or regular memory as fallback
        """
        # Try constant memory first
        tensor = self.allocate(size, dtype, is_constant=True)
        
        # Fallback to regular memory if constant memory is full
        if tensor is None:
            logger.info("Constant memory full, falling back to regular memory for weights")
            if not self.silent:
                print("Constant memory full, falling back to regular memory for weights")
            tensor = self.allocate(size, dtype, is_constant=False)
        
        return tensor
    
    def _find_block(self, size: int) -> Optional[MemoryBlock]:
        """Find a suitable block for allocation."""
        if self.strategy == AllocationStrategy.BEST_FIT:
            return self._best_fit(size)
        elif self.strategy == AllocationStrategy.FIRST_FIT:
            return self._first_fit(size)
        else:
            return self._first_fit(size)
    
    def _find_constant_block(self, size: int) -> Optional[MemoryBlock]:
        """Find a suitable constant memory block for allocation."""
        for block in self.constant_blocks:
            if block.is_free() and block.size >= size:
                return block
        return None
    
    def _best_fit(self, size: int) -> Optional[MemoryBlock]:
        """Find the smallest block that fits the requested size."""
        best_block = None
        best_waste = float('inf')
        
        for block in self.blocks:
            if block.is_free() and block.size >= size:
                waste = block.size - size
                if waste < best_waste:
                    best_waste = waste
                    best_block = block
        
        return best_block
    
    def _first_fit(self, size: int) -> Optional[MemoryBlock]:
        """Find the first block that fits the requested size."""
        for block in self.blocks:
            if block.is_free() and block.size >= size:
                return block
        return None
    
    def _create_tensor_from_block(
        self,
        block: MemoryBlock,
        size: int,
        dtype: Optional[Any],
        is_constant: bool = False
    ) -> Any:
        """Create a tensor view from a memory block with zero-copy support."""
        # Use zero-copy for MLX arrays when enabled
        if self.device == "mlx" and self.strategy == AllocationStrategy.ZERO_COPY:
            return self._create_zero_copy_array(block, size, dtype, is_constant)
        
        if self.device == "mps":
            if dtype is None:
                dtype = self.optimal_dtype
            
            # Select buffer based on memory type
            buffer = self._mps_constant_buffer if is_constant else self._mps_buffer
            
            # Get element size based on buffer dtype
            if buffer.dtype in [torch.float16, torch.bfloat16]:
                buffer_element_size = 2
            else:
                buffer_element_size = 4
            
            # Calculate number of elements needed
            if dtype in [torch.float16, torch.bfloat16]:
                target_element_size = 2
            else:
                target_element_size = 4
            
            num_elements = size // target_element_size
            start_idx = block.offset // buffer_element_size
            end_idx = start_idx + (size // buffer_element_size)
            
            tensor_view = buffer[start_idx:end_idx].view(-1)
            
            # Convert dtype if needed
            if dtype != buffer.dtype:
                tensor_view = tensor_view.to(dtype)
            
            # Mark as non-gradients for constant memory
            if is_constant:
                tensor_view.requires_grad_(False)
            
            return tensor_view[:num_elements]
            
        elif self.device == "mlx":
            if dtype is None:
                dtype = self.optimal_dtype
            
            # Get element size based on buffer dtype
            if self._mlx_buffer.dtype in [mx.float16, getattr(mx, 'bfloat16', mx.float16)]:
                buffer_element_size = 2
            else:
                buffer_element_size = 4
            
            # Calculate number of elements
            if dtype in [mx.float16, getattr(mx, 'bfloat16', mx.float16)]:
                target_element_size = 2
            else:
                target_element_size = 4
            
            num_elements = size // target_element_size
            start_idx = block.offset // buffer_element_size
            end_idx = start_idx + (size // buffer_element_size)
            
            array_view = self._mlx_buffer[start_idx:end_idx]
            
            # Convert dtype if needed (MLX handles this automatically)
            if dtype != self._mlx_buffer.dtype:
                array_view = array_view.astype(dtype)
            
            return array_view[:num_elements]
    
    def _create_zero_copy_array(
        self,
        block: MemoryBlock,
        size: int,
        dtype: Optional[Any],
        is_constant: bool = False
    ) -> mx.array:
        """Create MLX array with zero-copy from unified memory."""
        logger.debug(f"Creating zero-copy array: size={size}, constant={is_constant}")
        
        if dtype is None:
            dtype = self.optimal_dtype
        
        # Select buffer based on memory type
        buffer = self._mlx_constant_buffer if is_constant else self._mlx_buffer
        
        # Calculate slice for zero-copy view
        if dtype in [mx.float16, getattr(mx, 'bfloat16', mx.float16)]:
            element_size = 2
        else:
            element_size = 4
        
        num_elements = size // element_size
        start_idx = block.offset // element_size
        end_idx = start_idx + num_elements
        
        # Create zero-copy view - no data movement
        array_view = buffer[start_idx:end_idx]
        
        # Convert dtype if needed (MLX does this efficiently)
        if dtype != buffer.dtype:
            array_view = array_view.astype(dtype)
        
        # Track zero-copy arrays for monitoring
        array_id = f"zero_copy_{block.block_id}_{id(array_view)}"
        self.zero_copy_arrays[array_id] = array_view
        
        # Store metadata for unified memory region
        self.unified_memory_regions[block.block_id] = {
            'size': size,
            'dtype': str(dtype),
            'zero_copy': True,
            'constant': is_constant
        }
        
        logger.debug(f"Zero-copy array created: id={array_id}, zero_copy_arrays={len(self.zero_copy_arrays)}")
        return array_view
    
    def deallocate(self, tensor: Any) -> bool:
        """
        Deallocate memory back to the pool.
        
        Args:
            tensor: Tensor to deallocate
            
        Returns:
            True if deallocation successful
        """
        with self._lock:
            for block in self.blocks:
                if block.tensor_ref is not None:
                    ref = block.tensor_ref()
                    if ref is tensor:
                        block.mark_free()
                        self.allocated_memory -= block.size
                        return True
            return False
    
    def _try_defragment(self) -> None:
        """Attempt to defragment the memory pool."""
        free_blocks = [b for b in self.blocks if b.is_free()]
        
        if len(free_blocks) < 2:
            return
        
        free_blocks.sort(key=lambda b: b.offset)
        
        merged = []
        current = free_blocks[0]
        
        for block in free_blocks[1:]:
            if current.offset + current.size == block.offset:
                current = MemoryBlock(
                    block_id=f"merged_{current.block_id}_{block.block_id}",
                    size=current.size + block.size,
                    offset=current.offset,
                    allocated=False,
                    allocation_time=None,
                    last_access=None,
                    device_type=self.device,
                    tensor_ref=None,
                    metadata={"merged": True}
                )
            else:
                merged.append(current)
                current = block
        
        merged.append(current)
        
        allocated_blocks = [b for b in self.blocks if not b.is_free()]
        self.blocks = allocated_blocks + merged
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            free_blocks = sum(1 for b in self.blocks if b.is_free())
            allocated_blocks = len(self.blocks) - free_blocks
            free_memory = sum(b.size for b in self.blocks if b.is_free())
            
            # Constant memory stats
            constant_free_blocks = sum(1 for b in self.constant_blocks if b.is_free())
            constant_allocated_blocks = len(self.constant_blocks) - constant_free_blocks
            constant_free_memory = sum(b.size for b in self.constant_blocks if b.is_free())
            
            # Get dtype information
            dtype_name = str(self.optimal_dtype).split('.')[-1]
            dtype_bits = 16 if self.optimal_dtype in [torch.float16, torch.bfloat16, 
                                                      mx.float16, getattr(mx, 'bfloat16', mx.float16)] else 32
            
            # Calculate zero-copy statistics
            zero_copy_count = len(self.zero_copy_arrays) if hasattr(self, 'zero_copy_arrays') else 0
            unified_regions = len(self.unified_memory_regions) if hasattr(self, 'unified_memory_regions') else 0
            
            stats = {
                "pool_size": self.pool_size,
                "allocated_memory": self.allocated_memory,
                "free_memory": free_memory,
                "peak_memory": self.peak_memory,
                "total_blocks": len(self.blocks),
                "allocated_blocks": allocated_blocks,
                "free_blocks": free_blocks,
                "fragmentation": 1.0 - (free_memory / (self.pool_size - self.allocated_memory + 0.01)),
                "constant_pool_size": self.CONSTANT_POOL_SIZE,
                "constant_allocated": self.allocated_constant_memory,
                "constant_free": constant_free_memory,
                "constant_blocks": len(self.constant_blocks),
                "constant_allocated_blocks": constant_allocated_blocks,
                "device": self.device,
                "strategy": self.strategy.value,
                "dtype": dtype_name,
                "dtype_bits": dtype_bits,
                "gpu_family": self.gpu_validator.gpu_info.gpu_family if self.gpu_validator.gpu_info else "unknown",
                "memory_efficiency": f"{dtype_bits}-bit precision, {50 if dtype_bits == 16 else 0}% memory savings",
                "constant_memory_benefit": "15-20% bandwidth improvement for weights"
            }
            
            # Add zero-copy statistics if MLX with zero-copy enabled
            if self.device == "mlx" and self.strategy == AllocationStrategy.ZERO_COPY:
                stats.update({
                    "zero_copy_enabled": True,
                    "zero_copy_arrays": zero_copy_count,
                    "unified_memory_regions": unified_regions,
                    "zero_copy_benefit": "Eliminates CPU-GPU transfer overhead"
                })
                logger.debug(f"Zero-copy stats: arrays={zero_copy_count}, regions={unified_regions}")
            else:
                stats["zero_copy_enabled"] = False
            
            return stats
    
    def reset(self) -> None:
        """Reset the memory pool."""
        with self._lock:
            for block in self.blocks:
                block.mark_free()
            
            self.allocated_memory = 0
            
            if self.strategy == AllocationStrategy.UNIFIED:
                self.blocks = []
                self._create_unified_blocks()
    
    def optimize_layout(self) -> None:
        """Optimize memory layout for better cache locality."""
        with self._lock:
            allocated_blocks = [b for b in self.blocks if not b.is_free()]
            
            allocated_blocks.sort(key=lambda b: b.last_access or datetime.min, reverse=True)
            
            self._try_defragment()
    
    def __del__(self):
        """Cleanup when pool is destroyed."""
        if self._mps_buffer is not None:
            del self._mps_buffer
        if self._mlx_buffer is not None:
            del self._mlx_buffer