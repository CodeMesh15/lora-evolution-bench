"""
DoRA: Weight-Decomposed Low-Rank Adaptation
Implementation using HuggingFace PEFT
"""

from typing import Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from .base import BaseMethod


class DoRAMethod(BaseMethod):
    """
    DoRA implementation with weight decomposition
    
    Features:
    - Decomposes weights into magnitude and direction
    - Applies LoRA to directional component
    - Better accuracy than standard LoRA
    - Same memory footprint as LoRA
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
    ):
        super().__init__(model, tokenizer, config)
        
        # DoRA/LoRA configurations
        self.lora_r = config.get("lora_r", 64)
        self.lora_alpha = config.get("lora_alpha", 16)
        self.lora_dropout = config.get("lora_dropout", 0.05)
        self.target_modules = config.get("target_modules", "all-linear")
        self.bias = config.get("bias", "none")
        self.task_type = config.get("task_type", "CAUSAL_LM")
        self.use_rslora = config.get("use_rslora", False)
        
        # Quantization config
        self.use_quantization = config.get("use_quantization", True)
        self.use_double_quant = config.get("use_double_quant", True)
        self.compute_dtype = config.get("compute_dtype", torch.bfloat16)
        
    @staticmethod
    def get_quantization_config(
        compute_dtype: torch.dtype = torch.bfloat16,
        use_double_quant: bool = True
    ) -> BitsAndBytesConfig:
        """Create quantization config for DoRA"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model with DoRA
        
        Returns:
            Model with DoRA adapters applied
        """
        # Prepare model for k-bit training if quantized
        if self.use_quantization:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.get("use_gradient_checkpointing", True)
            )
        
        # Create DoRA configuration (uses LoraConfig with use_dora=True)
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
            use_dora=True,  # Enable DoRA weight decomposition
            use_rslora=self.use_rslora,  # Optional: rank-stabilized scaling
        )
        
        # Apply DoRA
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
        """Save DoRA adapter"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"DoRA adapter saved to {output_dir}")
    
    def load_adapter(self, adapter_path: str) -> None:
        """Load DoRA adapter"""
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"DoRA adapter loaded from {adapter_path}")
    
    @staticmethod
    def create_from_pretrained(
        model_name: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto"
    ) -> "DoRAMethod":
        """
        Create DoRA method from pretrained model
        
        Args:
            model_name: Model name or path
            tokenizer_name: Tokenizer name (defaults to model_name)
            config: Configuration dictionary
            device_map: Device map for model loading
            
        Returns:
            DoRAMethod instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if tokenizer_name is None:
            tokenizer_name = model_name
        
        if config is None:
            config = {}
        
        use_quantization = config.get("use_quantization", True)
        
        # Load model
        if use_quantization:
            quant_config = DoRAMethod.get_quantization_config(
                compute_dtype=config.get("compute_dtype", torch.bfloat16),
                use_double_quant=config.get("use_double_quant", True)
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=config.get("compute_dtype", torch.bfloat16)
            )
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return DoRAMethod(model, tokenizer, config)
