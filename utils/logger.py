# Logger utility for Kolam AI
import logging
import os
from datetime import datetime

def get_logger(name='kolam_ai'):
    """Get a simple console logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def setup_logger(name='kolam_ai', log_dir='logs', log_to_console=True, log_level=logging.INFO):
    """
    Set up a logger with file and optional console output
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        log_to_console: Whether to also log to console
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    )
    
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create file handler with a timestamp in the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Optionally add console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger
