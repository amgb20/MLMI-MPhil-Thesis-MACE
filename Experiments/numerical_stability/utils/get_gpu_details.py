import torch
from utils.get_logging_profile import logger

def get_gpu_with_least_memory():
    """
    Automatically select the GPU with the least memory usage (most free memory).
    Uses CUDA's native memory query functions to get actual GPU memory usage.
    
    Returns:
        str: Device string like "cuda:0", "cuda:1", etc., or "cpu" if no GPUs available
    """
    if not torch.cuda.is_available():
        logger.info("No CUDA devices available, using CPU")
        return "cpu"
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} CUDA device(s)")
    
    if num_gpus == 1:
        logger.info("Only one GPU available, using cuda:0")
        return "cuda:0"
    
    # Check memory usage for each GPU using CUDA's native functions
    gpu_memory_info = []
    for i in range(num_gpus):
        try:
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory
            
            # Use CUDA's native memory query (this gets actual GPU memory usage)
            memory_info = torch.cuda.memory_stats(i)
            
            # Get actual memory usage from CUDA
            memory_allocated = memory_info.get('allocated_bytes.all.current', 0)
            memory_reserved = memory_info.get('reserved_bytes.all.current', 0)
            
            # Calculate free memory
            free_memory = total_memory - memory_reserved
            
            gpu_memory_info.append({
                'device_id': i,
                'memory_allocated': memory_allocated,
                'memory_reserved': memory_reserved,
                'memory_total': total_memory,
                'free_memory': free_memory,
                'usage_percent': (memory_reserved / total_memory) * 100
            })
            
            logger.info(f"GPU {i}: {memory_allocated/1024**2:.1f}MB allocated, "
                       f"{memory_reserved/1024**2:.1f}MB reserved ({memory_reserved/total_memory*100:.1f}% used), "
                       f"{free_memory/1024**2:.1f}MB free out of {total_memory/1024**2:.1f}MB total")
            
        except Exception as e:
            logger.warning(f"Could not get memory info for GPU {i}: {e}")
            # If we can't get memory info, assume it's heavily used
            gpu_memory_info.append({
                'device_id': i,
                'memory_allocated': 0,
                'memory_reserved': 0,
                'memory_total': 0,
                'free_memory': 0,
                'usage_percent': 100
            })
    
    # Find GPU with most free memory
    best_gpu = max(gpu_memory_info, key=lambda x: x['free_memory'])
    selected_device = f"cuda:{best_gpu['device_id']}"
    
    logger.info(f"Selected GPU {best_gpu['device_id']} with {best_gpu['free_memory']/1024**2:.1f}MB free memory "
               f"({best_gpu['usage_percent']:.1f}% used)")
    
    return selected_device
