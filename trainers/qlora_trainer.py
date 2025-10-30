"""
QLoRA-specific trainer implementation
"""

from typing import Optional, Callable
from transformers import Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

from .base_trainer import BaseTrainer, TrainerConfig


class QLoRATrainer(BaseTrainer):
    """Trainer for QLoRA method"""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        data_collator: Optional[Callable] = None,
    ):
        super().__init__(model, tokenizer, train_dataset, eval_dataset, config, data_collator)
        
        # Default data collator for causal LM
        if self.data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
    
    def create_trainer(self) -> Trainer:
        """Create HuggingFace Trainer for QLoRA"""
        training_args = self.get_training_arguments()
        
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
        )
