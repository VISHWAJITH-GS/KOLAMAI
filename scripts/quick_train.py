"""
Kolam AI - Quick Training Script
This script provides a simple way to train a Kolam classifier model with sensible defaults.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import local modules
from models.model_trainer import train_model
from models.model_evaluator import evaluate_model
from models.kolam_classifier import KolamClassifier
from utils.image_processor import ImageProcessor

def train_kolam_classifier(data_path="data/datasets", save_path="models/saved/kolam_classifier.h5"):
    """
    Train a Kolam classifier model with the given data.
    
    Args:
        data_path: Path to dataset directory organized by class folders
        save_path: Path to save the trained model
    """
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    
    # Load and preprocess the data
    image_processor = ImageProcessor()
    data_dir = Path(data_path)
    
    # Get class folders
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    class_names = [d.name for d in class_dirs]
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Load images and labels
    images = []
    labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        print(f"Processing class {class_names[class_idx]}...")
        
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
                print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    
    # Split data into training and validation sets
    indices = np.random.permutation(len(images))
    val_size = int(len(images) * 0.2)  # 20% for validation
    test_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_data = images[train_indices]
    train_labels = labels[train_indices]
    test_data = images[test_indices]
    test_labels = labels[test_indices]
    
    print(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
    
    # Train the model
    print("Training model...")
    train_model(
        data=train_data,
        labels=train_labels,
        save_path=save_path,
        model_type="classifier"
    )
    print(f"Model saved to {save_path}")
    
    # Evaluate the model
    print("Evaluating model...")
    model = KolamClassifier(model_path=save_path)
    
    # Set up evaluation directory
    eval_dir = os.path.join(os.path.dirname(save_path), "evaluation")
    
    metrics = evaluate_model(
        model=model,
        test_data=test_data,
        test_labels=test_labels,
        output_dir=eval_dir,
        model_type="classifier"
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nDetailed results saved to {eval_dir}")
    
    return model, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick train Kolam classifier")
    parser.add_argument("--data", type=str, default="data/datasets", 
                        help="Path to dataset directory")
    parser.add_argument("--save", type=str, default="models/saved/kolam_classifier.h5",
                        help="Path to save the model")
    
    args = parser.parse_args()
    train_kolam_classifier(args.data, args.save)