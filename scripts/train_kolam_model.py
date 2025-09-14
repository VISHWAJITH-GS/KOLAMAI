"""
Kolam AI - Training Script
This script demonstrates how to train the Kolam classifier model.
"""

import os
import sys
import numpy as np
import logging
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import local modules
from models.model_trainer import train_model
from models.model_evaluator import evaluate_model
from utils.image_processor import ImageProcessor
from utils.logger import setup_logger

# Configure logging
logger = setup_logger("kolam_training")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Kolam AI model")
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/datasets",
        help="Directory containing training data"
    )
    
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["classifier", "generator"], 
        default="classifier",
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--save-path", 
        type=str, 
        default="models/saved/kolam_classifier.h5",
        help="Path to save the trained model"
    )
    
    parser.add_argument(
        "--config-path", 
        type=str, 
        default=None,
        help="Path to custom configuration file (optional)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--eval", 
        action="store_true",
        help="Evaluate model after training"
    )
    
    parser.add_argument(
        "--test-split", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for testing (if --eval is specified)"
    )
    
    return parser.parse_args()

def load_training_data(data_dir, test_split=None):
    """
    Load training data from the specified directory.
    
    This function assumes data is organized in subdirectories by class:
    - data_dir/
        - class1/
            - image1.jpg
            - image2.jpg
            - ...
        - class2/
            - image1.jpg
            - ...
        ...
    
    Args:
        data_dir: Path to directory containing training data
        test_split: If specified, split data into train and test sets
        
    Returns:
        tuple of (data, labels) or (train_data, train_labels, test_data, test_labels)
    """
    logger.info(f"Loading data from {data_dir}")
    
    data_dir = Path(data_dir)
    image_processor = ImageProcessor()
    
    # Get class folders
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    class_names = [d.name for d in class_dirs]
    
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    # Load images and labels
    images = []
    labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        logger.info(f"Processing class {class_names[class_idx]}")
        
        # Get image files in this class directory
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = image_processor.load_image(str(img_path))
                img = image_processor.resize_image(img, target_size=(224, 224))
                
                # Add to dataset
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    logger.info(f"Loaded {len(images)} images with shape {images.shape}")
    
    # Split data if test_split is specified
    if test_split is not None and 0 < test_split < 1:
        # Shuffle indices
        indices = np.random.permutation(len(images))
        test_size = int(len(images) * test_split)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        # Split data
        train_data = images[train_indices]
        train_labels = labels[train_indices]
        test_data = images[test_indices]
        test_labels = labels[test_indices]
        
        logger.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
        return train_data, train_labels, test_data, test_labels
    else:
        return images, labels

def main():
    """Main function to train the model"""
    args = parse_arguments()
    
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Load data
    logger.info("Loading training data...")
    if args.eval:
        train_data, train_labels, test_data, test_labels = load_training_data(
            args.data_dir, test_split=args.test_split
        )
    else:
        data, labels = load_training_data(args.data_dir)
        train_data, train_labels = data, labels
    
    # Create custom config if specific parameters are provided
    if args.batch_size != 32 or args.epochs != 50:
        import json
        import tempfile
        
        config = {
            "batch_size": args.batch_size,
            "epochs": args.epochs
        }
        
        if args.config_path:
            # Load existing config and update with our parameters
            with open(args.config_path, 'r') as f:
                loaded_config = json.load(f)
                loaded_config.update(config)
                config = loaded_config
        
        # Write to a temporary file
        temp_config = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        with open(temp_config.name, 'w') as f:
            json.dump(config, f)
        
        config_path = temp_config.name
        logger.info(f"Created temporary config at {config_path}")
    else:
        config_path = args.config_path
    
    # Train the model
    logger.info(f"Starting {args.model_type} training...")
    train_model(
        data=train_data,
        labels=train_labels,
        save_path=args.save_path,
        model_type=args.model_type,
        config_path=config_path
    )
    logger.info(f"Model saved to {args.save_path}")
    
    # Evaluate model if requested
    if args.eval:
        logger.info("Evaluating model on test data...")
        
        # Load the model
        if args.model_type == "classifier":
            from models.kolam_classifier import KolamClassifier
            model = KolamClassifier(model_path=args.save_path)
        else:
            from models.pattern_generator import PatternGenerator
            model = PatternGenerator()
        
        # Set up evaluation directory
        eval_dir = os.path.join(os.path.dirname(args.save_path), "evaluation")
        
        # Evaluate
        metrics = evaluate_model(
            model=model,
            test_data=test_data,
            test_labels=test_labels,
            output_dir=eval_dir,
            model_type=args.model_type
        )
        
        # Print key metrics
        if args.model_type == "classifier":
            logger.info(f"Evaluation results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"Detailed results saved to {eval_dir}")
    
    # Cleanup temporary config file
    if 'temp_config' in locals() and os.path.exists(temp_config.name):
        os.unlink(temp_config.name)

if __name__ == "__main__":
    main()