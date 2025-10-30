"""
Model utility functions
"""

from typing import Dict, Any
import torch
from transformers import PreTrainedModel


def get_model_info(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Get comprehensive model information
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Memory estimation (rough)
    param_memory_mb = (total_params * 4) / (1024 ** 2)  # Assuming float32
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
        "estimated_memory_mb": param_memory_mb,
        "model_dtype": next(model.parameters()).dtype,
        "device": next(model.parameters()).device,
    }


def estimate_memory(
    model_size: int,
    batch_size: int = 1,
    sequence_length: int = 2048,
    gradient_checkpointing: bool = False,
    quantization: Optional[str] = None
) -> Dict[str, float]:
    """
    Estimate memory requirements for training
    
    Args:
        model_size: Number of parameters
        batch_size: Batch size
        sequence_length: Sequence length
        gradient_checkpointing: Whether gradient checkpointing is enabled
        quantization: Quantization type
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Base model memory
    if quantization == "4bit":
        model_memory = (model_size * 0.5) / (1024 ** 3)  # 4-bit = 0.5 bytes
    elif quantization == "8bit":
        model_memory = model_size / (1024 ** 3)  # 8-bit = 1 byte
    else:
        model_memory = (model_size * 4) / (1024 ** 3)  # float32 = 4 bytes
    
    # Activation memory (rough estimate)
    hidden_size = int((model_size / 125e6) * 4096)  # Rough approximation
    activation_memory = (batch_size * sequence_length * hidden_size * 4) / (1024 ** 3)
    
    if gradient_checkpointing:
        activation_memory *= 0.5  # Approximate reduction
    
    # Optimizer states (AdamW has 2 states per parameter)
    trainable_ratio = 0.01  # Typical for PEFT
    optimizer_memory = (model_size * trainable_ratio * 8) / (1024 ** 3)
    
    # Gradient memory
    gradient_memory = (model_size * trainable_ratio * 4) / (1024 ** 3)
    
    total_memory = model_memory + activation_memory + optimizer_memory + gradient_memory
    
    return {
        "model_memory_gb": model_memory,
        "activation_memory_gb": activation_memory,
        "optimizer_memory_gb": optimizer_memory,
        "gradient_memory_gb": gradient_memory,
        "total_memory_gb": total_memory,
        "recommended_gpu_memory_gb": total_memory * 1.2,  # Add 20% buffer
    }


def print_model_summary(model: PreTrainedModel) -> None:
    """Print detailed model summary"""
    info = get_model_info(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total Parameters:       {info['total_params']:>15,}")
    print(f"Trainable Parameters:   {info['trainable_params']:>15,}")
    print(f"Non-trainable Params:   {info['non_trainable_params']:>15,}")
    print(f"Trainable Percentage:   {info['trainable_percent']:>15.4f}%")
    print(f"Estimated Memory:       {info['estimated_memory_mb']:>15.2f} MB")
    print(f"Model dtype:            {str(info['model_dtype']):>15}")
    print(f"Device:                 {str(info['device']):>15}")
    print("="*60 + "\n")
