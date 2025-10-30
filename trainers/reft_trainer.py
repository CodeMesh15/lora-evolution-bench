"""
ReFT-specific trainer implementation  
"""

from typing import Optional, Callable
from datasets import Dataset

from .base_trainer import BaseTrainer, TrainerConfig

try:
    import pyreft
    PYREFT_AVAILABLE = True
except ImportError:
    PYREFT_AVAILABLE = False


class ReFTTrainer(BaseTrainer):
    """Trainer for ReFT method"""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        data_collator: Optional[Callable] = None,
    ):
        if not PYREFT_AVAILABLE:
            raise ImportError("pyreft required for ReFT training")
        
        super().__init__(model, tokenizer, train_dataset, eval_dataset, config, data_collator)
    
    def create_trainer(self):
        """Create pyreft trainer"""
        training_args = self.get_training_arguments()
        
        # ReFT uses its own trainer from pyreft
        return pyreft.ReftTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
        )
