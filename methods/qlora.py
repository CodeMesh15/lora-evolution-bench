"""
QLoRA: Efficient Fine-tuning of Quantized LLMs
Implementation using HuggingFace PEFT and bitsandbytes
"""

from typing import Dict, Any, Optional, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from .base import BaseMethod


class QLoRAMethod(BaseMethod):
    """
    QLoRA implementation with 4-bit quantization
    
    Features:
    - 4-bit NormalFloat (NF4) quantization
    - Double quantization for extra memory savings
    - Paged optimizers for memory spikes
    - LoRA adapters on specific modules
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
    ):
        super().__init__(model, tokenizer, config)
        
        # QLoRA-specific configurations
        self.lora_r = config.get("lora_r", 64)
        self.lora_alpha = config.get("lora_alpha", 16)
        self.lora_dropout = config.get("lora_dropout", 0.05)
        self.target_modules = config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        self.bias = config.get("bias", "none")
        self.task_type = config.get("task_type", "CAUSAL_LM")
        
        # Quantization config
        self.use_double_quant = config.get("use_double_quant", True)
        self.compute_dtype = config.get("compute_dtype", torch.bfloat16)
        
    @staticmethod
    def get_quantization_config(
        compute_dtype: torch.dtype = torch.bfloat16,
        use_double_quant: bool = True
    ) -> BitsAndBytesConfig:
        """
        Create BitsAndBytesConfig for 4-bit quantization
        
        Args:
            compute_dtype: Compute dtype for quantization
            use_double_quant: Whether to use double quantization
            
        Returns:
            BitsAndBytesConfig for model loading
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model with QLoRA
        
        Returns:
            Model with LoRA adapters applied
        """
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.get("use_gradient_checkpointing", True)
        )
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
        )
        
        # Apply LoRA
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
        """Save LoRA adapter"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"QLoRA adapter saved to {output_dir}")
    
    def load_adapter(self, adapter_path: str) -> None:
        """Load LoRA adapter"""
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"QLoRA adapter loaded from {adapter_path}")
    
    @staticmethod
    def create_from_pretrained(
        model_name: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto"
    ) -> "QLoRAMethod":
        """
        Create QLoRA method from pretrained model
        
        Args:
            model_name: Model name or path
            tokenizer_name: Tokenizer name (defaults to model_name)
            config: Configuration dictionary
            device_map: Device map for model loading
            
        Returns:
            QLoRAMethod instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if tokenizer_name is None:
            tokenizer_name = model_name
        
        if config is None:
            config = {}
        
        # Get quantization config
        quant_config = QLoRAMethod.get_quantization_config(
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
        
        return QLoRAMethod(model, tokenizer, config)
