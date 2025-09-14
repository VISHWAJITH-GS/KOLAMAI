"""
Kolam AI - Example Training
This is a simple example of how to train the Kolam AI model.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

print("Training Kolam AI Model")
print("======================")

# Check for sample data
data_path = Path("data/datasets")
if not data_path.exists() or not any(data_path.iterdir()):
    print(f"No data found in {data_path}. Let's set up sample data folders...")
    
    # Create sample data directories
    sample_classes = ["pulli_kolam", "sikku_kolam", "kambi_kolam", "padi_kolam", "rangoli"]
    for cls in sample_classes:
        os.makedirs(data_path / cls, exist_ok=True)
    
    print("Created sample data folders. Please add your images to the following directories:")
    for cls in sample_classes:
        print(f"  - {data_path / cls}")
    
    print("\nEach folder should contain training images for that specific class.")
    print("After adding your images, run this script again.")
    sys.exit(0)

# Check if there are enough images for training
total_images = sum(len(list(folder.glob("*.jpg")) + list(folder.glob("*.png"))) 
                  for folder in data_path.iterdir() if folder.is_dir())

if total_images < 10:
    print(f"Found only {total_images} images in total. Need at least 10 for training.")
    print("Please add more training images to the class folders.")
    sys.exit(0)

print(f"Found {total_images} images. Starting training...")

# Choose a script to use
use_quick = input("\nUse quick training script? (y/n) [y]: ").lower() != 'n'

if use_quick:
    # Use the quick training script
    from scripts.quick_train import train_kolam_classifier
    
    print("\nStarting quick training...")
    model, metrics = train_kolam_classifier(
        data_path=str(data_path),
        save_path="models/saved/kolam_classifier.h5"
    )
    
    print("\nTraining complete!")
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    
else:
    # Use the full training script with command line arguments
    import subprocess
    
    cmd = [
        "python", "scripts/train_kolam_model.py",
        "--data-dir", str(data_path),
        "--save-path", "models/saved/kolam_classifier.h5",
        "--epochs", "30",
        "--batch-size", "16",
        "--eval"
    ]
    
    print("\nRunning full training script:")
    print(" ".join(cmd))
    print("\n")
    
    subprocess.run(cmd)
    
print("\nTraining complete! You can now use the model for classification.")
print("The model is saved at: models/saved/kolam_classifier.h5")
print("Evaluation results are saved at: models/saved/evaluation/")