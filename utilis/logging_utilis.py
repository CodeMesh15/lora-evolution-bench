"""
Logging utilities
"""

import logging
from typing import Dict, Any
from pathlib import Path


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("eft_arena")
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(metrics: Dict[str, Any], prefix: str = ""):
    """Log metrics in a formatted way"""
    logger = logging.getLogger("eft_arena")
    
    logger.info(f"\n{'='*60}")
    if prefix:
        logger.info(f"{prefix}")
    logger.info(f"METRICS")
    logger.info(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key:30s}: {value:>12.4f}")
        else:
            logger.info(f"{key:30s}: {value:>12}")
    
    logger.info(f"{'='*60}\n")
