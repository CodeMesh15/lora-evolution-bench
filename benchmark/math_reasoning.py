"""
Math reasoning benchmark (GSM8K)
"""

from typing import Dict, Any
import torch
import re
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset


def extract_answer(text: str) -> str:
    """Extract numerical answer from text"""
    # Look for patterns like "#### 123" or "The answer is 123"
    patterns = [
        r"####\s*([0-9,]+)",
        r"answer is\s*([0-9,]+)",
        r"=\s*([0-9,]+)\s*$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")
    
    # Fallback: last number in text
    numbers = re.findall(r"([0-9,]+)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return ""


def evaluate_math_reasoning(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 1000,
    max_new_tokens: int = 512
) -> Dict[str, float]:
    """
    Evaluate on GSM8K math reasoning benchmark
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        num_samples: Number of samples to evaluate
        max_new_tokens: Max tokens to generate
        
    Returns:
        Dictionary with results
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for example in dataset:
            question = example["question"]
            answer = example["answer"]
            
            # Extract ground truth answer
            gt_answer = extract_answer(answer)
            
            # Create prompt
            prompt = f"Question: {question}\nLet's solve this step by step:\n"
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_answer = extract_answer(generated)
            
            # Check if correct
            if pred_answer == gt_answer:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100
    
    return {
        "gsm8k_accuracy": accuracy,
        "gsm8k_samples": total,
        "gsm8k_correct": correct,
    }
