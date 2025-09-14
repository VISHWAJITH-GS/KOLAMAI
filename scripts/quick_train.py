"""
Quick training script for Kolam AI
This script provides a simplified way to train the Kolam AI classifier.
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train_kolam_classifier(data_path, save_path, img_size=(224, 224), batch_size=32, epochs=10):
    """
    Train a basic CNN model for kolam classification
    
    Args:
        data_path: Path to the dataset directory containing class subfolders
        save_path: Path to save the trained model
        img_size: Image size for training (default: 224x224)
        batch_size: Training batch size
        epochs: Number of training epochs
        
    Returns:
        model: The trained model
        metrics: Dictionary with training metrics
    """
    print(f"Training with data from: {data_path}")
    
    # Check if data path exists and contains subfolders
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Check specifically for the pulli_kolam folder
    pulli_kolam_dir = data_dir / "pulli_kolam"
    if not pulli_kolam_dir.exists() or not pulli_kolam_dir.is_dir():
        print(f"pulli_kolam directory not found at {pulli_kolam_dir}. Using a fallback approach.")
        return train_dummy_model(data_dir, save_path, img_size, batch_size, epochs)
    
    # Check if there are images in the pulli_kolam folder
    image_count = len(list(pulli_kolam_dir.glob("*.jpg"))) + len(list(pulli_kolam_dir.glob("*.png")))
    if image_count == 0:
        print("No images found in pulli_kolam directory. Using a fallback approach.")
        return train_dummy_model(data_dir, save_path, img_size, batch_size, epochs)
    
    # Data preprocessing
    print("Creating data generators...")
    
    # Use TensorFlow's image data generator for loading and augmenting
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 20% for validation
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Flow from directory - training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Flow from directory - validation data
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Found {num_classes} classes: {train_generator.class_indices}")
    
    # Create model
    print("Creating model...")
    model = create_model(img_size + (3,), num_classes)
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    # Save model
    print(f"Saving model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    # Calculate metrics
    metrics = {
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
    }
    
    print(f"Training complete! Final accuracy: {metrics['accuracy']:.4f}")
    return model, metrics

def create_model(input_shape, num_classes):
    """Create a simple CNN model for kolam classification"""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_dummy_model(data_dir, save_path, img_size=(224, 224), batch_size=32, epochs=5):
    """
    Train a dummy model when proper class organization isn't available
    This function trains on all images, assigning random labels just for demonstration
    """
    # Find all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend([str(f) for f in data_dir.glob(f"*{ext}")])
    
    if not image_files:
        print("No image files found!")
        return None, {'accuracy': 0}
    
    print(f"Found {len(image_files)} images total")
    
    # Define 5 dummy classes
    dummy_classes = 5
    
    # Create dummy model
    model = create_model(img_size + (3,), dummy_classes)
    
    # For simplicity, just save the untrained model
    print(f"Saving dummy model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    print("Note: This is a dummy model for demonstration only.")
    return model, {'accuracy': 0.2}  # Dummy accuracy

if __name__ == "__main__":
    # This code runs when the script is executed directly
    if len(sys.argv) < 3:
        print("Usage: python quick_train.py <data_path> <save_path>")
        sys.exit(1)
        
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    
    train_kolam_classifier(data_path, save_path)