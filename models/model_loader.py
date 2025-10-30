"""
Centralized model loading utilities
"""

from typing import Optional, Dict, Any, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)


class ModelLoader:
    """Unified model loading interface for different configurations"""
    
    SUPPORTED_MODELS = {
        "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
        "mistral-7b": "mistralai/Mistral-7B-v0.3",
        "qwen-7b": "Qwen/Qwen2-7B",
        "gemma-7b": "google/gemma-7b",
    }
    
    @staticmethod
    def load_model_and_tokenizer(
        model_name: str,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        use_flash_attention: bool = True,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer with specified configuration
        
        Args:
            model_name: Model name or path
            quantization: Quantization type (None, "4bit", "8bit")
            device_map: Device mapping strategy
            torch_dtype: Torch data type for model
            trust_remote_code: Whether to trust remote code
            use_flash_attention: Whether to use flash attention 2
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Resolve model alias
        if model_name in ModelLoader.SUPPORTED_MODELS:
            model_name = ModelLoader.SUPPORTED_MODELS[model_name]
        
        # Prepare quantization config
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        
        # Prepare model kwargs
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            **kwargs
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Load tokenizer
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return model, tokenizer
    
    @staticmethod
    def get_model_config(model_name: str) -> Dict[str, Any]:
        """Get recommended configuration for a model"""
        configs = {
            "llama-3-8b": {
                "max_length": 4096,
                "quantization": "4bit",
                "lora_r": 64,
                "lora_alpha": 16,
            },
            "llama-3-70b": {
                "max_length": 4096,
                "quantization": "4bit",
                "lora_r": 32,
                "lora_alpha": 16,
            },
            "mistral-7b": {
                "max_length": 8192,
                "quantization": "4bit",
                "lora_r": 64,
                "lora_alpha": 16,
            },
        }
        
        return configs.get(model_name, {
            "max_length": 2048,
            "quantization": "4bit",
            "lora_r": 64,
            "lora_alpha": 16,
        })
