"""
Complete benchmark suite runner
"""

from typing import Dict, Any, List, Optional
import time
import json
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer

from .commonsense import evaluate_commonsense
from .math_reasoning import evaluate_math_reasoning
from .instruction_following import evaluate_instruction_following
from ..utils.memory_utils import track_memory, get_gpu_memory
from ..utils.logging_utils import log_metrics


def run_full_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str = "./benchmark_results",
    benchmarks: List[str] = ["commonsense", "math", "instruction"],
    num_samples: int = 1000,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run complete benchmark suite
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        output_dir: Directory to save results
        benchmarks: List of benchmark categories
        num_samples: Number of samples per benchmark
        save_results: Whether to save results to file
        
    Returns:
        Dictionary with all results
    """
    results = {
        "model_name": model.config._name_or_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": num_samples,
    }
    
    # Commonsense reasoning
    if "commonsense" in benchmarks:
        print("\n" + "="*60)
        print("COMMONSENSE REASONING BENCHMARKS")
        print("="*60)
        
        with track_memory("Commonsense Evaluation"):
            commonsense_results = evaluate_commonsense(
                model, tokenizer, num_samples=num_samples
            )
        
        results["commonsense"] = commonsense_results
        log_metrics(commonsense_results, "Commonsense Results")
    
    # Math reasoning
    if "math" in benchmarks:
        print("\n" + "="*60)
        print("MATH REASONING BENCHMARK")
        print("="*60)
        
        with track_memory("Math Evaluation"):
            math_results = evaluate_math_reasoning(
                model, tokenizer, num_samples=num_samples
            )
        
        results["math"] = math_results
        log_metrics(math_results, "Math Results")
    
    # Instruction following
    if "instruction" in benchmarks:
        print("\n" + "="*60)
        print("INSTRUCTION FOLLOWING BENCHMARK")
        print("="*60)
        
        with track_memory("Instruction Evaluation"):
            instruction_results = evaluate_instruction_following(
                model, tokenizer, num_samples=min(500, num_samples)
            )
        
        # Don't include full responses in summary
        summary = {k: v for k, v in instruction_results.items() if k != "responses"}
        results["instruction"] = summary
        log_metrics(summary, "Instruction Following Results")
    
    # Add memory stats
    results["memory"] = get_gpu_memory()
    
    # Save results
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"benchmark_results_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for category, metrics in results.items():
        if isinstance(metrics, dict) and category not in ["memory"]:
            print(f"\n{category.upper()}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and "accuracy" in key:
                    print(f"  {key:30s}: {value:8.2f}%")
    print("="*60 + "\n")
    
    return results
