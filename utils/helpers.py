"""
Common Utility Functions and Constants for Kolam AI Application

This module provides shared utilities for file handling, validation, logging,
configuration management, and other common operations used across the application.
"""

import os
import re
import uuid
import json
import logging
import hashlib
import mimetypes
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Application Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
MAX_IMAGE_DIMENSION = 2048  # Maximum width/height in pixels
MIN_IMAGE_DIMENSION = 100   # Minimum width/height in pixels

# File paths and directories
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated'
MODELS_FOLDER = 'models/saved'
DATA_FOLDER = 'data'
LOGS_FOLDER = 'logs'
TEMP_FOLDER = 'temp'

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = logging.INFO

# Cultural constants
TRADITIONAL_PATTERNS = {
    'pulli_kolam': {
        'min_dots': 9,
        'max_dots': 225,  # 15x15 grid
        'typical_grid_sizes': [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11), (13, 13)],
        'dot_spacing_range': (15, 60)  # pixels
    },
    'sikku_kolam': {
        'min_lines': 1,
        'max_lines': 50,
        'continuity_threshold': 0.8,
        'symmetry_requirement': True
    },
    'rangoli': {
        'min_elements': 3,
        'max_elements': 100,
        'color_flexibility': True,
        'shape_flexibility': True
    }
}

# Image processing constants
IMAGE_PREPROCESSING = {
    'resize_dimensions': (512, 512),
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    'edge_detection_thresholds': (50, 150),
    'contour_min_area': 100,
    'blur_kernel_size': (5, 5)
}

# Model constants
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'batch_size': 32,
    'confidence_threshold': 0.6,
    'classification_classes': [
        'pulli_kolam', 'sikku_kolam', 'rangoli', 'festival_special', 'other'
    ]
}

# Session configuration
SESSION_CONFIG = {
    'max_uploads_per_session': 10,
    'session_timeout_hours': 24,
    'max_generation_requests': 5
}

class KolamError(Exception):
    """Base exception class for Kolam AI application"""
    pass

class FileValidationError(KolamError):
    """Exception for file validation errors"""
    pass

class ProcessingError(KolamError):
    """Exception for processing errors"""
    pass

class ConfigurationError(KolamError):
    """Exception for configuration errors"""
    pass

def setup_logging(log_level: int = LOG_LEVEL, log_file: str = None) -> logging.Logger:
    """
    Set up application logging with consistent formatting
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )
    
    logger = logging.getLogger('kolam_ai')
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = os.path.join(LOGS_FOLDER, f'kolam_ai_{datetime.now().strftime("%Y%m%d")}.log')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info("Logging initialized successfully")
    return logger

def allowed_file(filename: str) -> bool:
    """
    Check if uploaded file has allowed extension
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    if not filename or '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def validate_image_file(file_path: str) -> Dict[str, Any]:
    """
    Comprehensive validation of uploaded image file
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary with validation results
        
    Raises:
        FileValidationError: If file fails validation
    """
    validation_result = {
        'valid': False,
        'file_size': 0,
        'dimensions': (0, 0),
        'format': None,
        'errors': []
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileValidationError(f"File does not exist: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        validation_result['file_size'] = file_size
        
        if file_size > MAX_FILE_SIZE:
            validation_result['errors'].append(
                f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE / 1024 / 1024:.1f}MB)"
            )
        
        if file_size == 0:
            validation_result['errors'].append("File is empty")
        
        # Validate image using PIL
        try:
            with Image.open(file_path) as img:
                validation_result['dimensions'] = img.size
                validation_result['format'] = img.format
                
                width, height = img.size
                
                # Check dimensions
                if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                    validation_result['errors'].append(
                        f"Image dimensions ({width}x{height}) are too small. Minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}"
                    )
                
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    validation_result['errors'].append(
                        f"Image dimensions ({width}x{height}) are too large. Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
                    )
                
                # Check if image is corrupted
                img.verify()
                
        except Exception as e:
            validation_result['errors'].append(f"Invalid image file: {str(e)}")
        
        # Set validation status
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        return validation_result
        
    except Exception as e:
        raise FileValidationError(f"File validation failed: {str(e)}")

def generate_filename(original_filename: str, prefix: str = "", suffix: str = "") -> str:
    """
    Generate secure, unique filename with timestamp
    
    Args:
        original_filename: Original uploaded filename
        prefix: Optional prefix to add
        suffix: Optional suffix to add before extension
        
    Returns:
        Generated secure filename
    """
    # Get file extension
    if '.' in original_filename:
        name, ext = original_filename.rsplit('.', 1)
        ext = '.' + ext.lower()
    else:
        name = original_filename
        ext = ''
    
    # Create secure base name
    secure_name = secure_filename(name)
    if not secure_name:
        secure_name = 'upload'
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate short UUID for uniqueness
    unique_id = str(uuid.uuid4())[:8]
    
    # Combine components
    parts = []
    if prefix:
        parts.append(prefix)
    parts.extend([secure_name, timestamp, unique_id])
    if suffix:
        parts.append(suffix)
    
    return '_'.join(parts) + ext

def cleanup_files(directory: str, max_age_hours: int = 24, pattern: str = "*") -> int:
    """
    Clean up old files in specified directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep (in hours)
        pattern: File pattern to match (glob pattern)
        
    Returns:
        Number of files deleted
    """
    if not os.path.exists(directory):
        return 0
    
    deleted_count = 0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    try:
        for file_path in Path(directory).glob(pattern):
            if file_path.is_file():
                # Check file modification time
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mod_time < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    
    except Exception as e:
        logger = logging.getLogger('kolam_ai')
        logger.error(f"Error during cleanup: {str(e)}")
    
    return deleted_count

def cleanup_temp_files() -> int:
    """
    Clean up temporary files created during processing
    
    Returns:
        Number of files cleaned up
    """
    total_cleaned = 0
    
    # Clean upload directory
    total_cleaned += cleanup_files(UPLOAD_FOLDER, max_age_hours=24)
    
    # Clean generated files
    total_cleaned += cleanup_files(GENERATED_FOLDER, max_age_hours=72)
    
    # Clean temp directory
    if os.path.exists(TEMP_FOLDER):
        total_cleaned += cleanup_files(TEMP_FOLDER, max_age_hours=1)
    
    return total_cleaned

def log_activity(activity_type: str, details: Dict[str, Any], user_id: str = None) -> None:
    """
    Log user activity with structured format
    
    Args:
        activity_type: Type of activity (upload, classify, generate, etc.)
        details: Activity details dictionary
        user_id: Optional user identifier
    """
    logger = logging.getLogger('kolam_ai')
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'activity_type': activity_type,
        'user_id': user_id or 'anonymous',
        'details': details
    }
    
    logger.info(f"Activity: {json.dumps(log_entry)}")

def create_directory_structure() -> None:
    """
    Create necessary directory structure for the application
    """
    directories = [
        UPLOAD_FOLDER,
        GENERATED_FOLDER,
        LOGS_FOLDER,
        TEMP_FOLDER,
        f"{DATA_FOLDER}/sample_patterns",
        f"{MODELS_FOLDER}"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_config(config_file: str = 'config.json') -> Dict[str, Any]:
    """
    Load application configuration from JSON file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If config file cannot be loaded
    """
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # Return default configuration
            config = get_default_config()
            
        # Validate required keys
        validate_config(config)
        return config
        
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {str(e)}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")

def get_default_config() -> Dict[str, Any]:
    """
    Get default application configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'app': {
            'name': 'Kolam AI',
            'version': '1.0.0',
            'debug': False,
            'secret_key': 'your-secret-key-here'
        },
        'upload': {
            'max_file_size': MAX_FILE_SIZE,
            'allowed_extensions': list(ALLOWED_EXTENSIONS),
            'upload_folder': UPLOAD_FOLDER
        },
        'processing': {
            'max_image_dimension': MAX_IMAGE_DIMENSION,
            'min_image_dimension': MIN_IMAGE_DIMENSION,
            'resize_dimensions': IMAGE_PREPROCESSING['resize_dimensions']
        },
        'models': {
            'classifier_path': 'models/saved/kolam_classifier.h5',
            'confidence_threshold': MODEL_CONFIG['confidence_threshold']
        },
        'cultural': {
            'authenticity_threshold': 0.7,
            'traditional_patterns': TRADITIONAL_PATTERNS
        },
        'logging': {
            'level': 'INFO',
            'format': LOG_FORMAT,
            'file_rotation': True
        }
    }

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_sections = ['app', 'upload', 'processing', 'models']
    
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required configuration section: {section}")
    
    # Validate specific values
    if config['upload']['max_file_size'] <= 0:
        raise ConfigurationError("Invalid max_file_size in configuration")
    
    if not config['upload']['allowed_extensions']:
        raise ConfigurationError("No allowed extensions specified in configuration")

def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file for integrity checking
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def resize_image(image_path: str, target_size: Tuple[int, int], 
                 output_path: str = None, maintain_aspect: bool = True) -> str:
    """
    Resize image to target dimensions
    
    Args:
        image_path: Input image path
        target_size: Target (width, height)
        output_path: Output path (if None, overwrites original)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Path to resized image
    """
    output_path = output_path or image_path
    
    with Image.open(image_path) as img:
        if maintain_aspect:
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
        else:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        img.save(output_path, 'JPEG', quality=90, optimize=True)
    
    return output_path

def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about an image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                'filename': os.path.basename(image_path),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.size[0],
                'height': img.size[1],
                'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                'file_size': os.path.getsize(image_path),
                'mime_type': mimetypes.guess_type(image_path)[0]
            }
    except Exception as e:
        return {'error': str(e)}

def sanitize_input(input_string: str, max_length: int = 255) -> str:
    """
    Sanitize user input string
    
    Args:
        input_string: Input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not input_string:
        return ""
    
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', str(input_string))
    
    # Limit length
    sanitized = sanitized[:max_length]
    
    # Remove leading/trailing whitespace
    sanitized = sanitized.strip()
    
    return sanitized

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def validate_pattern_data(pattern_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate pattern data structure
    
    Args:
        pattern_data: Pattern data dictionary
        
    Returns:
        Validation result dictionary
    """
    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required structure
    if not isinstance(pattern_data, dict):
        validation['errors'].append("Pattern data must be a dictionary")
        validation['valid'] = False
        return validation
    
    # Check for at least one pattern element
    pattern_elements = ['dots', 'lines', 'curves', 'contours']
    has_elements = any(key in pattern_data for key in pattern_elements)
    
    if not has_elements:
        validation['errors'].append("Pattern data must contain at least one element (dots, lines, curves, or contours)")
        validation['valid'] = False
    
    # Validate individual elements
    if 'dots' in pattern_data:
        dots = pattern_data['dots']
        if not isinstance(dots, list):
            validation['errors'].append("Dots must be a list")
            validation['valid'] = False
        elif len(dots) > TRADITIONAL_PATTERNS['pulli_kolam']['max_dots']:
            validation['warnings'].append(f"Large number of dots ({len(dots)}) may not be traditional")
    
    if 'lines' in pattern_data:
        lines = pattern_data['lines']
        if not isinstance(lines, list):
            validation['errors'].append("Lines must be a list")
            validation['valid'] = False
    
    return validation

def create_backup(file_path: str, backup_dir: str = None) -> str:
    """
    Create backup copy of a file
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory for backup (default: same directory with .backup suffix)
        
    Returns:
        Path to backup file
    """
    if backup_dir is None:
        backup_path = f"{file_path}.backup"
    else:
        os.makedirs(backup_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        backup_path = os.path.join(backup_dir, filename)
    
    shutil.copy2(file_path, backup_path)
    return backup_path

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': dict(psutil.disk_usage('/')),
        'timestamp': datetime.now().isoformat()
    }

def emergency_cleanup() -> Dict[str, int]:
    """
    Emergency cleanup function to free up space
    
    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        'files_deleted': 0,
        'space_freed': 0,
        'directories_cleaned': 0
    }
    
    # Clean temporary files aggressively
    temp_dirs = [UPLOAD_FOLDER, TEMP_FOLDER, GENERATED_FOLDER]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            initial_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(temp_dir)
                for filename in filenames
            )
            
            # Clean files older than 1 hour
            deleted = cleanup_files(temp_dir, max_age_hours=1)
            
            final_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(temp_dir)
                for filename in filenames
            )
            
            stats['files_deleted'] += deleted
            stats['space_freed'] += initial_size - final_size
            stats['directories_cleaned'] += 1
    
    return stats

# Initialize logging when module is imported
logger = setup_logging()
logger.info("Helpers module initialized successfully")