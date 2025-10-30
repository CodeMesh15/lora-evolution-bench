"""
Commonsense reasoning benchmarks (BoolQ, PIQA, HellaSwag, ARC)
"""

from typing import Dict, Any, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
import numpy as np


def evaluate_boolq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 1000
) -> Dict[str, float]:
    """Evaluate on BoolQ dataset"""
    dataset = load_dataset("google/boolq", split="validation")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for example in dataset:
            question = example["question"]
            passage = example["passage"]
            label = example["answer"]
            
            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (True/False):"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Get logits for True/False tokens
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            true_id = tokenizer.encode("True", add_special_tokens=False)[0]
            false_id = tokenizer.encode("False", add_special_tokens=False)[0]
            
            prediction = logits[true_id] > logits[false_id]
            
            if prediction == label:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100
    return {"boolq_accuracy": accuracy, "boolq_samples": total}


def evaluate_piqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 1000
) -> Dict[str, float]:
    """Evaluate on PIQA dataset"""
    dataset = load_dataset("piqa", split="validation")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for example in dataset:
            goal = example["goal"]
            sol1 = example["sol1"]
            sol2 = example["sol2"]
            label = example["label"]
            
            # Compute perplexity for each solution
            prompt1 = f"{goal} {sol1}"
            prompt2 = f"{goal} {sol2}"
            
            inputs1 = tokenizer(prompt1, return_tensors="pt").to(model.device)
            inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
            
            loss1 = model(**inputs1, labels=inputs1["input_ids"]).loss
            loss2 = model(**inputs2, labels=inputs2["input_ids"]).loss
            
            prediction = 0 if loss1 < loss2 else 1
            
            if prediction == label:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100
    return {"piqa_accuracy": accuracy, "piqa_samples": total}


def evaluate_hellaswag(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 1000
) -> Dict[str, float]:
    """Evaluate on HellaSwag dataset"""
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for example in dataset:
            ctx = example["ctx"]
            endings = example["endings"]
            label = int(example["label"])
            
            # Compute loss for each ending
            losses = []
            for ending in endings:
                prompt = ctx + " " + ending
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                loss = model(**inputs, labels=inputs["input_ids"]).loss
                losses.append(loss.item())
            
            prediction = np.argmin(losses)
            
            if prediction == label:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100
    return {"hellaswag_accuracy": accuracy, "hellaswag_samples": total}


def evaluate_arc(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    difficulty: str = "easy",
    num_samples: int = 1000
) -> Dict[str, float]:
    """Evaluate on ARC dataset"""
    split_name = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"
    dataset = load_dataset("ai2_arc", split_name, split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for example in dataset:
            question = example["question"]
            choices = example["choices"]
            label = example["answerKey"]
            
            # Map label to index
            label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
            label_idx = label_map.get(label, 0)
            
            # Compute loss for each choice
            losses = []
            for choice in choices["text"]:
                prompt = f"Question: {question}\nAnswer: {choice}"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                loss = model(**inputs, labels=inputs["input_ids"]).loss
                losses.append(loss.item())
            
            prediction = np.argmin(losses)
            
            if prediction == label_idx:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100
    prefix = "arc_easy" if difficulty == "easy" else "arc_challenge"
    return {f"{prefix}_accuracy": accuracy, f"{prefix}_samples": total}


def evaluate_commonsense(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    benchmarks: List[str] = ["boolq", "piqa", "hellaswag", "arc-e", "arc-c"],
    num_samples: int = 1000
) -> Dict[str, float]:
    """
    Run all commonsense reasoning benchmarks
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        benchmarks: List of benchmarks to run
        num_samples: Number of samples per benchmark
        
    Returns:
        Dictionary with all results
    """
    results = {}
    
    if "boolq" in benchmarks:
        print("Evaluating BoolQ...")
        results.update(evaluate_boolq(model, tokenizer, num_samples))
    
    if "piqa" in benchmarks:
        print("Evaluating PIQA...")
        results.update(evaluate_piqa(model, tokenizer, num_samples))
    
    if "hellaswag" in benchmarks:
        print("Evaluating HellaSwag...")
        results.update(evaluate_hellaswag(model, tokenizer, num_samples))
    
    if "arc-e" in benchmarks:
        print("Evaluating ARC-Easy...")
        results.update(evaluate_arc(model, tokenizer, "easy", num_samples))
    
    if "arc-c" in benchmarks:
        print("Evaluating ARC-Challenge...")
        results.update(evaluate_arc(model, tokenizer, "challenge", num_samples))
    
    # Compute average
    accuracies = [v for k, v in results.items() if "accuracy" in k]
    if accuracies:
        results["average_accuracy"] = np.mean(accuracies)
    
    return results
