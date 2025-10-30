"""
LoftQ: LoRA-Fine-Tuning-Aware Quantization
Implementation for better quantized initialization
"""

from typing import Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoftQConfig

from .base import BaseMethod


class LoftQMethod(BaseMethod):
    """
    LoftQ implementation for optimized quantization
    
    Features:
    - Optimizes LoRA initialization for quantized models
    - Minimizes quantization error
    - Better convergence than standard QLoRA
    - Compatible with 4-bit and 8-bit quantization
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
    ):
        super().__init__(model, tokenizer, config)
        
        # LoftQ/LoRA configurations
        self.lora_r = config.get("lora_r", 64)
        self.lora_alpha = config.get("lora_alpha", 16)
        self.lora_dropout = config.get("lora_dropout", 0.05)
        self.target_modules = config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        self.bias = config.get("bias", "none")
        self.task_type = config.get("task_type", "CAUSAL_LM")
        
        # LoftQ-specific configurations
        self.loftq_bits = config.get("loftq_bits", 4)
        self.loftq_iter = config.get("loftq_iter", 1)
        
        # Quantization config
        self.use_double_quant = config.get("use_double_quant", True)
        self.compute_dtype = config.get("compute_dtype", torch.bfloat16)
        
    @staticmethod
    def get_quantization_config(
        compute_dtype: torch.dtype = torch.bfloat16,
        use_double_quant: bool = True
    ) -> BitsAndBytesConfig:
        """Create quantization config for LoftQ"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model with LoftQ initialization
        
        Returns:
            Model with LoRA adapters using LoftQ initialization
        """
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.get("use_gradient_checkpointing", True)
        )
        
        # Create LoftQ configuration
        loftq_config = LoftQConfig(
            loftq_bits=self.loftq_bits,
            loftq_iter=self.loftq_iter
        )
        
        # Create LoRA configuration with LoftQ initialization
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
            init_lora_weights="loftq",  # Use LoftQ initialization
            loftq_config=loftq_config,
        )
        
        # Apply LoRA with LoftQ
        self.model = get_peft_model(self.model, peft_config)
        
        return self.model
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get trainable parameter statistics"""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percent": 100 * trainable_params / all_params
        }
    
    def save_adapter(self, output_dir: str) -> None:
        """Save LoftQ adapter"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"LoftQ adapter saved to {output_dir}")
    
    def load_adapter(self, adapter_path: str) -> None:
        """Load LoftQ adapter"""
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"LoftQ adapter loaded from {adapter_path}")
    
    @staticmethod
    def create_from_pretrained(
        model_name: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto"
    ) -> "LoftQMethod":
        """
        Create LoftQ method from pretrained model
        
        Args:
            model_name: Model name or path
            tokenizer_name: Tokenizer name (defaults to model_name)
            config: Configuration dictionary
            device_map: Device map for model loading
            
        Returns:
            LoftQMethod instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if tokenizer_name is None:
            tokenizer_name = model_name
        
        if config is None:
            config = {}
        
        # Get quantization config
        quant_config = LoftQMethod.get_quantization_config(
            compute_dtype=config.get("compute_dtype", torch.bfloat16),
            use_double_quant=config.get("use_double_quant", True)
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return LoftQMethod(model, tokenizer, config)
