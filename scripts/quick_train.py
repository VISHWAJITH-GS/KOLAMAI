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
    """
    # Check if data path exists
    if not os.path.exists(data_path):
        raise ValueError(f"Data path not found: {data_path}")
    
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get list of class folders
    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if not class_folders:
        raise ValueError(f"No class folders found in {data_path}")
    
    print(f"Found {len(class_folders)} classes: {', '.join(class_folders)}")
    
    # Prepare data arrays
    images = []
    labels = []
    
    # Load images from each class
    for i, class_name in enumerate(class_folders):
        class_path = os.path.join(data_path, class_name)
        print(f"Loading images from {class_path}")
        
        # Get all image files
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files[:500]:  # Limit to 500 per class for quick training
            try:
                img_path = os.path.join(class_path, img_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(i)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize images
    images = images / 255.0
    
    # Convert labels to categorical
    num_classes = len(class_folders)
    labels = tf.keras.utils.to_categorical(labels, num_classes)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training with {len(X_train)} images, validating with {len(X_val)} images")
    
    # Build model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Save class mapping
    class_mapping = {i: class_name for i, class_name in enumerate(class_folders)}
    config_path = os.path.join(os.path.dirname(save_path), 'classifier_config.json')
    
    import json
    with open(config_path, 'w') as f:
        json.dump({
            'classes': class_mapping,
            'input_shape': img_size + (3,),
            'version': '1.0'
        }, f, indent=2)
    
    print(f"Class mapping saved to {config_path}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Loss plot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(save_path), 'training_history.png'))
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    return model

def evaluate_kolam_classifier(model_path, data_path, img_size=(224, 224)):
    """
    Evaluate a trained model on test data
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load class mapping
    config_path = os.path.join(os.path.dirname(model_path), 'classifier_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    class_mapping = config['classes']
    
    # Get list of class folders
    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    # Prepare confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    y_true = []
    y_pred = []
    
    # Process each class
    for class_name in class_folders:
        class_path = os.path.join(data_path, class_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files[:100]:  # Test on 100 images per class
            try:
                img_path = os.path.join(class_path, img_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                # Get true class index
                true_class_idx = [k for k, v in class_mapping.items() if v == class_name][0]
                y_true.append(int(true_class_idx))
                
                # Predict
                pred = model.predict(img_array)[0]
                pred_class_idx = np.argmax(pred)
                y_pred.append(pred_class_idx)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=[class_mapping[str(i)] for i in range(len(class_mapping))]))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = [class_mapping[str(i)] for i in range(len(class_mapping))]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(model_path), 'confusion_matrix.png'))

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick train Kolam classifier")
    parser.add_argument("--data", type=str, default="data/datasets", 
                        help="Path to dataset directory")
    parser.add_argument("--save", type=str, default="models/saved/kolam_classifier.h5",
                        help="Path to save the model")
    
    args = parser.parse_args()
    train_kolam_classifier(args.data, args.save)