import logging
import os
import sys
from datetime import datetime

def get_logger(name: str, log_dir: str = None, level=logging.INFO):
    """
    Get a configured logger.
    Args:
        name: Name of the logger
        log_dir: Directory to save log files
        level: Logging level
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        # Use date in filename to avoid overwriting
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{date_str}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger
