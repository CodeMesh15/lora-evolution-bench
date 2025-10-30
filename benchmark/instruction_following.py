"""
Instruction following evaluation
"""

from typing import Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset


def evaluate_instruction_following(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "tatsu-lab/alpaca_eval",
    num_samples: int = 500,
    max_new_tokens: int = 512
) -> Dict[str, float]:
    """
    Evaluate instruction following capability
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset_name: Instruction dataset name
        num_samples: Number of samples
        max_new_tokens: Max tokens to generate
        
    Returns:
        Dictionary with results
    """
    dataset = load_dataset(dataset_name, split="eval")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    total_length = 0
    total_samples = 0
    
    model.eval()
    responses = []
    
    with torch.no_grad():
        for example in dataset:
            instruction = example["instruction"]
            
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated.split("### Response:")[-1].strip()
            
            responses.append(response)
            total_length += len(response.split())
            total_samples += 1
    
    avg_length = total_length / total_samples if total_samples > 0 else 0
    
    return {
        "instruction_following_samples": total_samples,
        "avg_response_length": avg_length,
        "responses": responses[:10],  # Sample responses
    }
