"""
ReFT: Representation Fine-Tuning for Language Models
Implementation using pyreft library
"""

from typing import Dict, Any, Optional, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaseMethod

try:
    import pyreft
    PYREFT_AVAILABLE = True
except ImportError:
    PYREFT_AVAILABLE = False
    print("Warning: pyreft not installed. Install with: pip install pyreft")


class ReFTMethod(BaseMethod):
    """
    ReFT implementation for representation-level fine-tuning
    
    Features:
    - 10-50x fewer parameters than LoRA
    - Modifies hidden representations instead of weights
    - Fast training with minimal examples
    - LoReFT intervention on specific layers
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
    ):
        super().__init__(model, tokenizer, config)
        
        if not PYREFT_AVAILABLE:
            raise ImportError("pyreft is required for ReFT. Install with: pip install pyreft")
        
        # ReFT configurations
        self.rank = config.get("rank", 4)
        self.layers = config.get("layers", None)  # None means auto-select
        self.component = config.get("component", "block_output")
        self.intervention_type = config.get("intervention_type", "LoreftIntervention")
        self.position = config.get("position", "last")  # Position to intervene: last, first, all
        
        self.reft_model = None
        
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model with ReFT interventions
        
        Returns:
            ReFT model ready for training
        """
        # Auto-select layers if not specified
        if self.layers is None:
            num_layers = self.model.config.num_hidden_layers
            # Target middle-to-late layers (typically more effective)
            start_layer = num_layers // 2
            self.layers = list(range(start_layer, min(start_layer + 6, num_layers)))
        
        # Create ReFT configuration
        representations = []
        for layer_idx in self.layers:
            representations.append({
                "layer": layer_idx,
                "component": self.component,
                "low_rank_dimension": self.rank,
                "intervention": getattr(pyreft, self.intervention_type)(
                    embed_dim=self.model.config.hidden_size,
                    low_rank_dimension=self.rank
                )
            })
        
        reft_config = pyreft.ReftConfig(representations=representations)
        
        # Get ReFT model
        self.reft_model = pyreft.get_reft_model(self.model, reft_config)
        
        # Set device
        device = next(self.model.parameters()).device
        self.reft_model.set_device(str(device))
        
        return self.reft_model
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get trainable parameter statistics"""
        if self.reft_model is None:
            return {"trainable_params": 0, "all_params": 0, "trainable_percent": 0.0}
        
        trainable_params = 0
        all_params = 0
        
        for _, param in self.reft_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percent": 100 * trainable_params / all_params
        }
    
    def save_adapter(self, output_dir: str) -> None:
        """Save ReFT intervention"""
        if self.reft_model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        
        self.reft_model.save_intervention(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"ReFT intervention saved to {output_dir}")
    
    def load_adapter(self, adapter_path: str) -> None:
        """Load ReFT intervention"""
        if self.reft_model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        
        self.reft_model.load_intervention(adapter_path)
        print(f"ReFT intervention loaded from {adapter_path}")
    
    @staticmethod
    def create_from_pretrained(
        model_name: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto"
    ) -> "ReFTMethod":
        """
        Create ReFT method from pretrained model
        
        Args:
            model_name: Model name or path
            tokenizer_name: Tokenizer name (defaults to model_name)
            config: Configuration dictionary
            device_map: Device map for model loading
            
        Returns:
            ReFTMethod instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if tokenizer_name is None:
            tokenizer_name = model_name
        
        if config is None:
            config = {}
        
        # Load model (typically without quantization for ReFT)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=config.get("torch_dtype", torch.bfloat16)
        )
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return ReFTMethod(model, tokenizer, config)
