"""
Data loading and preprocessing utilities
"""

from typing import Dict, Any, Optional, List, Callable
from datasets import load_dataset as hf_load_dataset, Dataset
from transformers import PreTrainedTokenizer


DATASET_CONFIGS = {
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "split": "train",
        "instruction_key": "instruction",
        "input_key": "input",
        "output_key": "output",
    },
    "dolly": {
        "path": "databricks/databricks-dolly-15k",
        "split": "train",
        "instruction_key": "instruction",
        "input_key": "context",
        "output_key": "response",
    },
    "gsm8k": {
        "path": "gsm8k",
        "name": "main",
        "split": "train",
        "instruction_key": "question",
        "output_key": "answer",
    },
}


def load_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    streaming: bool = False,
    num_samples: Optional[int] = None,
) -> Dataset:
    """
    Load a dataset by name
    
    Args:
        dataset_name: Name of dataset or path
        split: Split to load
        streaming: Whether to stream dataset
        num_samples: Limit number of samples
        
    Returns:
        Loaded dataset
    """
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        dataset = hf_load_dataset(
            config["path"],
            name=config.get("name"),
            split=split or config["split"],
            streaming=streaming
        )
    else:
        dataset = hf_load_dataset(dataset_name, split=split, streaming=streaming)
    
    if num_samples and not streaming:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    return dataset


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str = "",
    template: str = "alpaca"
) -> str:
    """
    Format instruction following specific template
    
    Args:
        instruction: Instruction text
        input_text: Optional input context
        output: Optional output/response
        template: Template name
        
    Returns:
        Formatted text
    """
    if template == "alpaca":
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        if output:
            return prompt + output
        return prompt
    
    elif template == "simple":
        prompt = f"Question: {instruction}\n"
        if input_text:
            prompt += f"Context: {input_text}\n"
        prompt += "Answer: "
        
        if output:
            return prompt + output
        return prompt
    
    else:
        raise ValueError(f"Unknown template: {template}")


def prepare_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    template: str = "alpaca",
    dataset_name: Optional[str] = None,
) -> Dataset:
    """
    Prepare dataset for training
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        template: Formatting template
        dataset_name: Name of dataset for config lookup
        
    Returns:
        Tokenized dataset
    """
    # Get column names from config
    if dataset_name and dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        instruction_key = config.get("instruction_key", "instruction")
        input_key = config.get("input_key", "input")
        output_key = config.get("output_key", "output")
    else:
        instruction_key = "instruction"
        input_key = "input"
        output_key = "output"
    
    def tokenize_function(examples):
        """Tokenize examples"""
        texts = []
        
        for i in range(len(examples[instruction_key])):
            instruction = examples[instruction_key][i]
            input_text = examples.get(input_key, [""] * len(examples[instruction_key]))[i]
            output = examples[output_key][i]
            
            formatted = format_instruction(
                instruction=instruction,
                input_text=input_text,
                output=output,
                template=template
            )
            texts.append(formatted)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Set labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset
