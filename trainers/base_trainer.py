"""
Base trainer class for all fine-tuning methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import torch
from transformers import TrainingArguments, Trainer
from datasets import Dataset


@dataclass
class TrainerConfig:
    """Configuration for training"""
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    group_by_length: bool = True
    report_to: str = "tensorboard"
    

class BaseTrainer(ABC):
    """Abstract base trainer for all methods"""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        data_collator: Optional[Callable] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainerConfig()
        self.data_collator = data_collator
        self.trainer = None
        
    def get_training_arguments(self) -> TrainingArguments:
        """Create TrainingArguments from config"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if self.eval_dataset else None,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            group_by_length=self.config.group_by_length,
            report_to=self.config.report_to,
            load_best_model_at_end=True if self.eval_dataset else False,
        )
    
    @abstractmethod
    def create_trainer(self) -> Trainer:
        """Create method-specific trainer"""
        pass
    
    def train(self) -> Dict[str, Any]:
        """Train the model"""
        if self.trainer is None:
            self.trainer = self.create_trainer()
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        train_result = self.trainer.train()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        return train_result
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        if self.eval_dataset is None:
            raise ValueError("No evaluation dataset provided.")
        
        return self.trainer.evaluate()
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the trained model"""
        save_dir = output_dir or self.config.output_dir
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"\nModel saved to {save_dir}")
