"""
Base class for all fine-tuning methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseMethod(ABC):
    """Abstract base class for fine-tuning methods"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
    ):
        """
        Initialize the fine-tuning method
        
        Args:
            model: Pretrained model to fine-tune
            tokenizer: Tokenizer for the model
            config: Configuration dictionary for the method
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    @abstractmethod
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare the model for fine-tuning with the specific method
        
        Returns:
            Prepared model ready for training
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get information about trainable parameters
        
        Returns:
            Dictionary with trainable parameter statistics
        """
        pass
    
    def print_trainable_parameters(self) -> None:
        """Print trainable parameter statistics"""
        stats = self.get_trainable_parameters()
        print(f"Trainable params: {stats['trainable_params']:,}")
        print(f"All params: {stats['all_params']:,}")
        print(f"Trainable %: {stats['trainable_percent']:.4f}%")
    
    @abstractmethod
    def save_adapter(self, output_dir: str) -> None:
        """
        Save the trained adapter/intervention
        
        Args:
            output_dir: Directory to save the adapter
        """
        pass
    
    @abstractmethod
    def load_adapter(self, adapter_path: str) -> None:
        """
        Load a trained adapter/intervention
        
        Args:
            adapter_path: Path to the adapter directory
        """
        pass
