"""
Memory management utilities for GPU-efficient model evaluation.

This module provides utilities for managing GPU memory during model evaluation,
including context managers, dynamic batch sizing, and memory monitoring.
"""

import gc
from contextlib import contextmanager
from typing import Generator

import torch


@contextmanager
def memory_efficient_context(
    clear_before: bool = True,
    clear_after: bool = True,
    aggressive: bool = False,
    sync_before: bool = True,
) -> Generator[None, None, None]:
    """
    Context manager for memory-efficient operations.

    Args:
        clear_before: Clear GPU memory before operation
        clear_after: Clear GPU memory after operation
        aggressive: Use aggressive memory clearing
        sync_before: Synchronize CUDA streams before operation

    Example:
        with memory_efficient_context(aggressive=True):
            embeddings = model.encode(sentences, batch_size=1)
    """
    try:
        # Pre-operation cleanup
        if clear_before:
            clear_gpu_memory(aggressive=aggressive, sync=sync_before)

        yield

    finally:
        # Post-operation cleanup
        if clear_after:
            clear_gpu_memory(aggressive=aggressive, sync=True)


def clear_gpu_memory(aggressive: bool = False, sync: bool = True) -> None:
    """
    Clear GPU memory and run garbage collection.

    Args:
        aggressive: If True, performs multiple rounds of cleanup
        sync: If True, synchronizes CUDA streams before clearing
    """
    # Force garbage collection (multiple rounds for aggressive cleanup)
    for _ in range(3 if aggressive else 1):
        gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()

        # Clear cache multiple times for aggressive cleanup
        for _ in range(2 if aggressive else 1):
            torch.cuda.empty_cache()

        if aggressive:
            torch.cuda.reset_peak_memory_stats()


def calculate_optimal_batch_size(
    base_batch_size: int, memory_threshold_gb: float = 2.0, min_batch_size: int = 1
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    Args:
        base_batch_size: Desired batch size
        memory_threshold_gb: Minimum free memory to maintain (GB)
        min_batch_size: Minimum allowed batch size

    Returns:
        Adjusted batch size
    """
    if not torch.cuda.is_available():
        return base_batch_size

    # Calculate free GPU memory in GB
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_memory = total - allocated

    if free_memory < memory_threshold_gb:
        # Reduce batch size aggressively if memory is low
        if free_memory < 0.5:  # Less than 500MB
            return min_batch_size
        elif free_memory < 1.0:  # Less than 1GB
            return max(min_batch_size, base_batch_size // 4)
        else:  # Less than threshold but above 1GB
            return max(min_batch_size, base_batch_size // 2)

    return base_batch_size
