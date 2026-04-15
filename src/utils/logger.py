import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path

class Logger:
    """Centralized logging system for the project"""
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ):
        """
        Initialize logger with console and file handlers
        
        Args:
            name: Name of the logger (usually __name__ or specific component name)
            log_dir: Directory to store log files
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(message)s'  # Simplified console output
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler - one file per day
        today = datetime.now().strftime('%Y-%m-%d')
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{today}.log')
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Store metrics
        self.metrics: Dict[str, Any] = {}
        self.results_dir = "results"

    @property
    def handlers(self):
        """Expose the underlying logger's handlers for compatibility."""
        return self.logger.handlers
    
    def debug(self, msg: str) -> None:
        """Log debug message"""
        self.logger.debug(msg)
    
    def info(self, msg: str) -> None:
        """Log info message"""
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log error message"""
        self.logger.error(msg)
    
    def critical(self, msg: str) -> None:
        """Log critical message"""
        self.logger.critical(msg)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics/statistics with optional step number
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number or iteration
        """
        self.metrics.update(metrics)
        
        # Format metrics for logging
        metrics_str = " - ".join([f"{k}: {v}" for k, v in metrics.items()])
        if step is not None:
            metrics_str = f"Step {step} - {metrics_str}"
        
        self.info(metrics_str)
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save accumulated metrics to JSON file
        
        Args:
            filepath: Path to save metrics JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.info(f"Saved metrics to {filepath}")

# Global logger instance
def get_logger(
    name: str = __name__,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> Logger:
    """Get or create a logger instance"""
    return Logger(name, log_dir, console_level, file_level)