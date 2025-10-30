"""
Evaluation utilities
"""

from typing import Dict, Any, List, Optional
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate model on dataset
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Encode inputs
            inputs = tokenizer(
                batch["input"],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
            references.extend(batch["output"])
    
    # Compute metrics
    metrics = compute_metrics(predictions, references)
    
    return metrics


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        predictions: Model predictions
        references: Ground truth references
        
    Returns:
        Dictionary of metrics
    """
    # Exact match
    exact_match = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    exact_match = exact_match / len(predictions) * 100
    
    # Token-level accuracy (rough approximation)
    token_matches = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if len(ref_tokens) > 0:
            matches = sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
            token_matches.append(matches / len(ref_tokens))
    
    token_accuracy = np.mean(token_matches) * 100 if token_matches else 0.0
    
    return {
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "num_samples": len(predictions),
    }
