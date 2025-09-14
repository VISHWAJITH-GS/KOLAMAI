# Config loader utility for Kolam AI
import os
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

def load_config(config_path, default_config=None):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        default_config: Default configuration to use if file not found
        
    Returns:
        Dictionary containing configuration data
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        else:
            logger.warning(f"Configuration file not found at {config_path}, using default config")
            return default_config if default_config is not None else {}
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return default_config if default_config is not None else {}
