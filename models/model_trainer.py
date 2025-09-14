# Model Trainer for Kolam AI

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import time

# Local imports
from models.kolam_classifier import KolamClassifier
from utils.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='kolam_ai.log'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training Kolam AI models (classifier and generator)"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the model trainer
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.image_processor = ImageProcessor()
        
    def _load_config(self) -> Dict:
        """Load training configuration from file or use defaults"""
        default_config = {
            # Model parameters
            "input_shape": (224, 224, 3),
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.0001,
            "dropout_rate": 0.3,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
            
            # Data augmentation
            "augmentation": {
                "enabled": True,
                "rotation_range": 20,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2,
                "shear_range": 0.2,
                "zoom_range": 0.2,
                "horizontal_flip": True,
                "vertical_flip": False,
                "fill_mode": "nearest"
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with default config
                    for key, value in loaded_config.items():
                        if key in default_config and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config
        
    def _save_config(self, model_type: str, config: Dict, save_dir: str) -> None:
        """Save configuration to JSON file"""
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, f"{model_type}_config.json")
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            
    def _save_training_history(self, history: Any, save_dir: str, model_type: str) -> None:
        """Save training history and generate plots"""
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save history as pickle
        history_path = os.path.join(save_dir, f"{model_type}_history.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        
        # Generate and save accuracy plot
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{model_type.capitalize()} Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_type.capitalize()} Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_type}_training_history.png"))
        plt.close()
        
        logger.info(f"Training history and plots saved to {save_dir}")

    def _preprocess_data(self, data: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess input data for model training
        
        Args:
            data: Input image data or features
            labels: Class labels (optional for unsupervised training)
            
        Returns:
            Preprocessed data and labels
        """
        # If data is a list of file paths, load the images
        if isinstance(data[0], (str, Path)):
            processed_data = []
            for img_path in data:
                try:
                    # Load and preprocess image
                    img = self.image_processor.load_image(img_path)
                    img = self.image_processor.resize_image(img, target_size=self.config["input_shape"][:2])
                    processed_data.append(img)
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    continue
            
            if not processed_data:
                raise ValueError("No valid images found in the provided data")
                
            data = np.array(processed_data)
        
        # Ensure correct dimensions
        if len(data.shape) == 3 and self.config["input_shape"][-1] == 3:
            # Convert grayscale to RGB if needed
            data = np.repeat(data[..., np.newaxis], 3, axis=-1)
            
        # Normalize pixel values to [0, 1]
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
            
        return data, labels
            
    def train_classifier(self, data: np.ndarray, labels: np.ndarray, save_path: str) -> KolamClassifier:
        """
        Train a Kolam classifier model
        
        Args:
            data: Training image data (numpy array or list of image paths)
            labels: Class labels (one-hot encoded or class indices)
            save_path: Path to save the trained model
            
        Returns:
            Trained KolamClassifier model
        """
        # Create save directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize classifier
        classifier = KolamClassifier()
        
        # Preprocess data
        logger.info("Preprocessing training data...")
        data, labels = self._preprocess_data(data, labels)
        
        # Convert labels to one-hot if needed
        if len(labels.shape) == 1:
            num_classes = len(classifier.CLASS_LABELS)
            labels = keras.utils.to_categorical(labels, num_classes)
        
        # Split data into training and validation sets
        val_split = self.config["validation_split"]
        val_idx = int(len(data) * (1 - val_split))
        
        train_data, val_data = data[:val_idx], data[val_idx:]
        train_labels, val_labels = labels[:val_idx], labels[val_idx:]
        
        logger.info(f"Training set: {train_data.shape}, Validation set: {val_data.shape}")
        
        # Setup data augmentation
        if self.config["augmentation"]["enabled"]:
            datagen = ImageDataGenerator(
                rotation_range=self.config["augmentation"]["rotation_range"],
                width_shift_range=self.config["augmentation"]["width_shift_range"],
                height_shift_range=self.config["augmentation"]["height_shift_range"],
                shear_range=self.config["augmentation"]["shear_range"],
                zoom_range=self.config["augmentation"]["zoom_range"],
                horizontal_flip=self.config["augmentation"]["horizontal_flip"],
                vertical_flip=self.config["augmentation"]["vertical_flip"],
                fill_mode=self.config["augmentation"]["fill_mode"],
            )
            datagen.fit(train_data)
            
        # Build and compile model
        logger.info("Building classification model...")
        classifier.build_model(num_classes=labels.shape[1])
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config["early_stopping_patience"],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        logger.info("Starting model training...")
        start_time = time.time()
        
        if self.config["augmentation"]["enabled"]:
            # Train with data augmentation
            history = classifier.model.fit(
                datagen.flow(train_data, train_labels, batch_size=self.config["batch_size"]),
                steps_per_epoch=len(train_data) // self.config["batch_size"],
                epochs=self.config["epochs"],
                validation_data=(val_data, val_labels),
                callbacks=callbacks
            )
        else:
            # Train without data augmentation
            history = classifier.model.fit(
                train_data, 
                train_labels,
                batch_size=self.config["batch_size"],
                epochs=self.config["epochs"],
                validation_data=(val_data, val_labels),
                callbacks=callbacks
            )
            
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Save configuration
        config = self.config.copy()
        config.update({
            "training_samples": len(train_data),
            "validation_samples": len(val_data),
            "training_time": training_time,
            "model_path": save_path,
            "input_shape": list(self.config["input_shape"]),
            "num_classes": labels.shape[1],
            "class_labels": classifier.CLASS_LABELS
        })
        
        self._save_config("classifier", config, save_dir)
        self._save_training_history(history, save_dir, "classifier")
        
        # Save model and return
        classifier.save_model(save_path)
        logger.info(f"Classifier model saved to {save_path}")
        
        return classifier
        
    def train_generator(self, data: np.ndarray, save_path: str) -> None:
        """
        Train a Kolam pattern generator model
        
        Args:
            data: Training pattern data
            save_path: Path to save the trained model
        """
        # For the generator, we would implement appropriate training logic
        # This could involve training a GAN, VAE, or other generative models
        # or updating the rule-based parameters of the PatternGenerator
        
        logger.info("Pattern generator training not fully implemented yet")
        logger.info("Current implementation uses rule-based generation rather than ML training")
        
        # Save a placeholder file to indicate this was called
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, "generator_info.json"), 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "training_samples": len(data) if hasattr(data, "__len__") else "unknown",
                "model_type": "rule_based_generator",
                "note": "Pattern generator uses rule-based generation rather than ML training"
            }, f, indent=4)
        
        logger.info(f"Generator information saved to {save_dir}")

def train_model(data: np.ndarray, labels: np.ndarray = None, save_path: str = "models/saved/model.h5", 
                model_type: str = "classifier", config_path: str = None) -> None:
    """
    Train a Kolam classifier or generator model and save it.
    
    Args:
        data: Training data (images or features)
        labels: Class labels (required for classifier, optional for generator)
        save_path: Path to save the trained model
        model_type: Type of model to train ('classifier' or 'generator')
        config_path: Path to configuration file (optional)
    """
    # Create trainer instance
    trainer = ModelTrainer(config_path)
    
    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Train appropriate model type
    if model_type.lower() == "classifier":
        if labels is None:
            raise ValueError("Labels are required for training a classifier")
        trainer.train_classifier(data, labels, save_path)
    elif model_type.lower() == "generator":
        trainer.train_generator(data, save_path)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'classifier' or 'generator'")
