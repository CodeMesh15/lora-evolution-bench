"""
Memory tracking and profiling utilities
"""

from typing import Dict, Optional
import torch
import psutil
import time
from contextlib import contextmanager


def get_gpu_memory() -> Dict[str, float]:
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {}
    
    memory_stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        memory_stats[f"gpu_{i}_allocated_gb"] = allocated
        memory_stats[f"gpu_{i}_reserved_gb"] = reserved
    
    return memory_stats


def get_cpu_memory() -> Dict[str, float]:
    """Get current CPU memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "cpu_memory_rss_gb": memory_info.rss / (1024 ** 3),
        "cpu_memory_vms_gb": memory_info.vms / (1024 ** 3),
    }


def print_memory_stats(prefix: str = ""):
    """Print current memory statistics"""
    stats = {**get_gpu_memory(), **get_cpu_memory()}
    
    print(f"\n{'='*60}")
    if prefix:
        print(f"{prefix}")
    print(f"MEMORY STATISTICS")
    print(f"{'='*60}")
    
    for key, value in stats.items():
        print(f"{key:30s}: {value:8.2f} GB")
    
    print(f"{'='*60}\n")


@contextmanager
def track_memory(name: str = "Operation"):
    """
    Context manager to track memory usage
    
    Usage:
        with track_memory("Training"):
            train_model()
    """
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Get initial memory
    initial_gpu = get_gpu_memory()
    initial_cpu = get_cpu_memory()
    start_time = time.time()
    
    print(f"\n[{name}] Starting...")
    print_memory_stats(f"Initial Memory - {name}")
    
    try:
        yield
    finally:
        # Get final memory
        final_gpu = get_gpu_memory()
        final_cpu = get_cpu_memory()
        elapsed_time = time.time() - start_time
        
        print(f"\n[{name}] Completed in {elapsed_time:.2f}s")
        print_memory_stats(f"Final Memory - {name}")
        
        # Print delta
        print(f"{'='*60}")
        print(f"Memory Delta - {name}")
        print(f"{'='*60}")
        
        for key in initial_gpu:
            delta = final_gpu[key] - initial_gpu[key]
            print(f"{key:30s}: +{delta:8.2f} GB")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                peak = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                print(f"gpu_{i}_peak_allocated_gb{' '*8}: {peak:8.2f} GB")
        
        print(f"{'='*60}\n")
