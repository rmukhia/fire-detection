"""
Performance and system optimization utilities for fire detection system.
"""

import os
import sys
import psutil
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def get_optimal_device() -> str:
    """
    Automatically detect the best available device for PyTorch operations.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    # Check for CUDA
    if torch.cuda.is_available():
        # Check CUDA memory
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            if memory_gb >= 4.0:  # Require at least 4GB GPU memory
                return f"cuda:{i}"
        return "cuda:0"  # Default to first CUDA device
    
    # Check for Apple Silicon MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"


def get_optimal_worker_count(task_type: str = "cpu_bound") -> int:
    """
    Calculate optimal number of workers based on system resources.
    
    Args:
        task_type: Type of task - 'cpu_bound', 'io_bound', or 'mixed'
        
    Returns:
        int: Optimal worker count
    """
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    logical_count = psutil.cpu_count(logical=True)  # Logical cores
    memory_gb = psutil.virtual_memory().total / 1e9
    
    if task_type == "cpu_bound":
        # For CPU-bound tasks, use physical cores
        return max(1, min(cpu_count, int(memory_gb // 2)))
    elif task_type == "io_bound":
        # For I/O-bound tasks, can use more workers
        return max(1, min(logical_count * 2, int(memory_gb // 1)))
    else:  # mixed
        # Conservative approach for mixed workloads
        return max(1, min(cpu_count, int(memory_gb // 3)))


def optimize_pytorch_settings():
    """Configure PyTorch for optimal performance."""
    # Enable optimized attention (if available)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Set number of threads for CPU operations
    cpu_cores = psutil.cpu_count(logical=False)
    torch.set_num_threads(cpu_cores)
    
    # Enable CUDNN optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_memory_info() -> Dict[str, Any]:
    """Get system memory information."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    info = {
        "total_ram_gb": memory.total / 1e9,
        "available_ram_gb": memory.available / 1e9,
        "ram_usage_percent": memory.percent,
        "total_swap_gb": swap.total / 1e9,
        "swap_usage_percent": swap.percent
    }
    
    # Add GPU memory info if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}_name"] = props.name
            info[f"gpu_{i}_memory_gb"] = props.total_memory / 1e9
    
    return info


def configure_environment_variables():
    """Set optimal environment variables for performance."""
    env_vars = {
        # PyTorch optimizations
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
        "OMP_NUM_THREADS": str(psutil.cpu_count(logical=False)),
        "MKL_NUM_THREADS": str(psutil.cpu_count(logical=False)),
        
        # Dask optimizations
        "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
        "DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING": "True",
        
        # General optimizations
        "PYTHONHASHSEED": "0",  # For reproducibility
        "CUDA_LAUNCH_BLOCKING": "0",  # Async CUDA operations
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


def check_system_requirements() -> Dict[str, bool]:
    """Check if system meets minimum requirements."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    requirements = {
        "memory_sufficient": memory.total >= 8e9,  # 8GB RAM
        "disk_sufficient": disk.free >= 10e9,  # 10GB free space
        "python_version_ok": sys.version_info >= (3, 8),
        "torch_available": torch.__version__ is not None,
        "cuda_available": torch.cuda.is_available() if torch.cuda.is_available() else True,
    }
    
    return requirements


def log_system_info(logger: Optional[logging.Logger] = None):
    """Log detailed system information."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # System info
    memory_info = get_memory_info()
    requirements = check_system_requirements()
    
    logger.info("=== System Information ===")
    logger.info(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    logger.info(f"Memory: {memory_info['total_ram_gb']:.1f}GB total, {memory_info['available_ram_gb']:.1f}GB available")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {memory_info.get(f'gpu_{i}_name', 'Unknown')} ({memory_info.get(f'gpu_{i}_memory_gb', 0):.1f}GB)")
    
    # Requirements check
    logger.info("=== Requirements Check ===")
    for req, met in requirements.items():
        status = "✓" if met else "✗"
        logger.info(f"{status} {req}: {met}")
    
    # Recommendations
    logger.info("=== Recommendations ===")
    optimal_device = get_optimal_device()
    optimal_workers = get_optimal_worker_count("mixed")
    logger.info(f"Recommended device: {optimal_device}")
    logger.info(f"Recommended workers: {optimal_workers}")


def create_performance_config() -> Dict[str, Any]:
    """Create optimized configuration based on system capabilities."""
    memory_gb = psutil.virtual_memory().total / 1e9
    
    config = {
        "device": get_optimal_device(),
        "num_workers": get_optimal_worker_count("mixed"),
        "batch_size": min(64, max(8, int(memory_gb // 2))),
        "use_mixed_precision": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7,
        "dataloader_num_workers": min(4, psutil.cpu_count(logical=False)),
        "pin_memory": torch.cuda.is_available(),
    }
    
    return config


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configure environment
    configure_environment_variables()
    optimize_pytorch_settings()
    
    # Log system info
    log_system_info(logger)
    
    # Show performance config
    perf_config = create_performance_config()
    logger.info("=== Performance Configuration ===")
    for key, value in perf_config.items():
        logger.info(f"{key}: {value}")