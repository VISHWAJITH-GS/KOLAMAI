"""
Organize Kolam Images
This script organizes the kolam images into class folders for training.
"""

import os
import shutil
from pathlib import Path
import random
import sys

def organize_images():
    """Organize images from the dataset folder into class subfolders."""
    dataset_path = Path("data/datasets")
    
    # Define class folders
    class_folders = [
        "pulli_kolam",
        "sikku_kolam", 
        "kambi_kolam",
        "padi_kolam",
        "rangoli"
    ]
    
    # Ensure class folders exist
    for folder in class_folders:
        os.makedirs(dataset_path / folder, exist_ok=True)
    
    # Get all images in the datasets folder (not in subfolders)
    image_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        image_files.extend([f for f in dataset_path.glob(ext) 
                          if f.is_file() and not any(class_name in str(f) for class_name in class_folders)])
    
    print(f"Found {len(image_files)} images to organize")
    
    if not image_files:
        print("No images found to organize.")
        return
    
    # Group images by prefix (e.g., kolam19, kolam29, kolam109)
    image_groups = {}
    for img in image_files:
        # Extract the prefix (everything before the dash)
        prefix = img.name.split('-')[0]
        if prefix not in image_groups:
            image_groups[prefix] = []
        image_groups[prefix].append(img)
    
    print(f"Found {len(image_groups)} image groups: {list(image_groups.keys())}")
    
    # Distribute each group to a class
    for i, (prefix, files) in enumerate(image_groups.items()):
        # Assign each prefix to a class (cycling through them)
        target_class = class_folders[i % len(class_folders)]
        print(f"Moving {len(files)} images with prefix '{prefix}' to {target_class}")
        
        for file in files:
            # Create hard link instead of copying to save space
            try:
                target_path = dataset_path / target_class / file.name
                if not target_path.exists():
                    # Create hard link instead of moving/copying
                    os.link(file, target_path)
                    print(f"Created link for {file.name} to {target_class}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    organize_images()
    print("Done organizing images!")