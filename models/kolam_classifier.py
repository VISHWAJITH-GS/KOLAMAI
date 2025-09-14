"""
Kolam AI - CNN-based Kolam Classification Model
This module implements a deep learning model for classifying different types of Kolam patterns
using TensorFlow/Keras with MobileNet backbone for efficient inference.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2
from PIL import Image
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KolamClassifier:
    """
    CNN-based Kolam pattern classifier using MobileNet backbone
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize the Kolam Classifier
        
        Args:
            model_path: Path to saved model file
            config_path: Path to configuration file
        """
        # Model configuration constants
        self.INPUT_SHAPE = (224, 224, 3)
        self.MODEL_PATH = model_path or "models/saved/kolam_classifier.h5"
        self.CONFIG_PATH = config_path or "models/saved/classifier_config.json"
        
        # Class labels for different Kolam types
        self.CLASS_LABELS = [
            "pulli_kolam",      # Dot Kolam - புள்ளி கோலம்
            "sikku_kolam",      # Line Kolam - சிக்கு கோலம்
            "kambi_kolam",      # Wire Kolam - கம்பி கோலம்
            "padi_kolam",       # Step Kolam - படி கோலம்
            "rangoli",          # Rangoli - ரங்கோலி
            "festival_special", # Festival Special - பண்டிகை சிறப்பு
            "geometric",        # Geometric Patterns
            "traditional",      # Traditional Patterns
            "modern",           # Modern Adaptations
            "regional"          # Regional Variations
        ]
        
        # Tamil names mapping
        self.TAMIL_NAMES = {
            "pulli_kolam": "புள்ளி கோலம்",
            "sikku_kolam": "சிக்கு கோலம்",
            "kambi_kolam": "கம்பி கோலம்",
            "padi_kolam": "படி கோலம்",
            "rangoli": "ரங்கோலி",
            "festival_special": "பண்டிகை சிறப்பு கோலம்",
            "geometric": "வடிவியல் கோலம்",
            "traditional": "பாரம்பரிய கோலம்",
            "modern": "நவீன கோலம்",
            "regional": "பிராந்திய கோலம்"
        }
        
        # Model and feature extractor
        self.model = None
        self.feature_extractor = None
        self.config = {}
        
        # Load configuration and model if available
        self._load_config()
        if os.path.exists(self.MODEL_PATH):
            self.load_model()
    
    def _load_config(self):
        """Load model configuration"""
        try:
            if os.path.exists(self.CONFIG_PATH):
                with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load configuration: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "input_shape": self.INPUT_SHAPE,
            "num_classes": len(self.CLASS_LABELS),
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.0001,
            "dropout_rate": 0.3,
            "l2_regularization": 0.001,
            "data_augmentation": True,
            "use_mixed_precision": True,
            "freeze_base_layers": 100
        }
    
    def create_model(self) -> keras.Model:
        """
        Create CNN model with MobileNet backbone
        
        Returns:
            Compiled Keras model
        """
        try:
            # Enable mixed precision for faster training
            if self.config.get("use_mixed_precision", True):
                policy = keras.mixed_precision.Policy('mixed_float16')
                keras.mixed_precision.set_global_policy(policy)
            
            # Input layer
            inputs = keras.Input(shape=self.INPUT_SHAPE, name='input_image')
            
            # Preprocessing layer
            x = keras.applications.mobilenet_v2.preprocess_input(inputs)
            
            # MobileNetV2 backbone
            base_model = applications.MobileNetV2(
                input_shape=self.INPUT_SHAPE,
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            
            # Freeze initial layers
            freeze_layers = self.config.get("freeze_base_layers", 100)
            for layer in base_model.layers[:freeze_layers]:
                layer.trainable = False
            
            # Feature extraction from base model
            x = base_model(x, training=False)
            
            # Custom classification head
            x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x) if base_model.pooling != 'avg' else x
            x = layers.BatchNormalization(name='bn_features')(x)
            x = layers.Dropout(self.config.get("dropout_rate", 0.3), name='dropout_1')(x)
            
            # Dense layers for classification
            x = layers.Dense(
                512, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.config.get("l2_regularization", 0.001)),
                name='dense_512'
            )(x)
            x = layers.BatchNormalization(name='bn_dense_512')(x)
            x = layers.Dropout(0.2, name='dropout_2')(x)
            
            x = layers.Dense(
                256, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.config.get("l2_regularization", 0.001)),
                name='dense_256'
            )(x)
            x = layers.BatchNormalization(name='bn_dense_256')(x)
            x = layers.Dropout(0.1, name='dropout_3')(x)
            
            # Feature extraction layer (for feature_extractor)
            features = layers.Dense(128, activation='relu', name='feature_layer')(x)
            
            # Output layer
            predictions = layers.Dense(
                len(self.CLASS_LABELS),
                activation='softmax',
                dtype='float32',  # Ensure float32 output for mixed precision
                name='predictions'
            )(features)
            
            # Create model
            model = keras.Model(inputs=inputs, outputs=predictions, name='kolam_classifier')
            
            # Create feature extractor model
            self.feature_extractor = keras.Model(inputs=inputs, outputs=features, name='feature_extractor')
            
            # Compile model
            optimizer = keras.optimizers.Adam(
                learning_rate=self.config.get("learning_rate", 0.0001),
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )
            
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_3_accuracy']
            )
            
            logger.info(f"Model created successfully with {model.count_params():,} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model prediction
        
        Args:
            image: Image path, numpy array, or PIL Image
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                img = np.array(image.convert('RGB'))
            else:
                img = image.copy()
            
            # Resize to model input shape
            img = cv2.resize(img, (self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]))
            
            # Normalize pixel values
            img = img.astype(np.float32)
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Predict Kolam pattern type for given image
        
        Args:
            image: Input image (path, array, or PIL Image)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            logger.warning("Model not loaded. Attempting to load model first.")
            try:
                self.load_model()
            except:
                # If load fails, create dummy model
                self._create_dummy_model()
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            confidence_scores = predictions[0]
            
            # Get top predictions
            top_indices = np.argsort(confidence_scores)[::-1]
            
            # Extract features if feature extractor is available
            features = None
            if self.feature_extractor is not None:
                features = self.feature_extractor.predict(processed_img, verbose=0)[0]
            
            # Prepare results
            results = {
                'predicted_class': self.CLASS_LABELS[top_indices[0]],
                'predicted_class_tamil': self.TAMIL_NAMES.get(self.CLASS_LABELS[top_indices[0]], ''),
                'confidence': float(confidence_scores[top_indices[0]]),
                'top_3_predictions': [
                    {
                        'class': self.CLASS_LABELS[idx],
                        'class_tamil': self.TAMIL_NAMES.get(self.CLASS_LABELS[idx], ''),
                        'confidence': float(confidence_scores[idx])
                    }
                    for idx in top_indices[:3]
                ],
                'all_probabilities': {
                    self.CLASS_LABELS[i]: float(confidence_scores[i])
                    for i in range(len(self.CLASS_LABELS))
                },
                'features': features.tolist() if features is not None else None,
                'prediction_metadata': {
                    'model_version': self.config.get('version', '1.0'),
                    'timestamp': np.datetime64('now').astype(str),
                    'input_shape': self.INPUT_SHAPE
                }
            }
            
            logger.info(f"Prediction completed: {results['predicted_class']} ({results['confidence']:.3f})")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def extract_features(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract feature vector from image
        
        Args:
            image: Input image
            
        Returns:
            Feature vector as numpy array
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not available")
        
        try:
            processed_img = self.preprocess_image(image)
            features = self.feature_extractor.predict(processed_img, verbose=0)
            return features[0]
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def train_model(self, train_data_dir: str, validation_data_dir: str = None, 
                   val_split: float = 0.2) -> keras.callbacks.History:
        """
        Train the Kolam classifier model
        
        Args:
            train_data_dir: Directory containing training images
            validation_data_dir: Directory containing validation images
            val_split: Validation split ratio if validation_data_dir not provided
            
        Returns:
            Training history
        """
        try:
            # Create model if not exists
            if self.model is None:
                self.model = self.create_model()
            
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='nearest',
                validation_split=val_split if validation_data_dir is None else 0.0
            )
            
            # Training generator
            train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]),
                batch_size=self.config.get("batch_size", 32),
                class_mode='categorical',
                classes=self.CLASS_LABELS,
                subset='training' if validation_data_dir is None else None,
                shuffle=True
            )
            
            # Validation generator
            if validation_data_dir is not None:
                val_datagen = ImageDataGenerator()
                validation_generator = val_datagen.flow_from_directory(
                    validation_data_dir,
                    target_size=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]),
                    batch_size=self.config.get("batch_size", 32),
                    class_mode='categorical',
                    classes=self.CLASS_LABELS,
                    shuffle=False
                )
            else:
                validation_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    target_size=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]),
                    batch_size=self.config.get("batch_size", 32),
                    class_mode='categorical',
                    classes=self.CLASS_LABELS,
                    subset='validation',
                    shuffle=False
                )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    self.MODEL_PATH,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_generator,
                epochs=self.config.get("epochs", 100),
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save configuration
            self._save_config()
            
            logger.info("Model training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def load_model(self, model_path: str = None):
        """
        Load trained model from file
        
        Args:
            model_path: Path to model file
        """
        try:
            path = model_path or self.MODEL_PATH
            if not os.path.exists(path):
                logger.warning(f"Model file not found: {path}. Creating a dummy model for development.")
                # Create a simple dummy model for development
                self._create_dummy_model()
                return
            
            self.model = keras.models.load_model(path, compile=False)
            
            # Recreate feature extractor
            if 'feature_layer' in [layer.name for layer in self.model.layers]:
                feature_layer = self.model.get_layer('feature_layer')
                self.feature_extractor = keras.Model(
                    inputs=self.model.input,
                    outputs=feature_layer.output,
                    name='feature_extractor'
                )
            
            # Recompile model
            optimizer = keras.optimizers.Adam(
                learning_rate=self.config.get("learning_rate", 0.0001)
            )
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_3_accuracy']
            )
            
            logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create a dummy model for development
            self._create_dummy_model()
            
    def _create_dummy_model(self):
        """Create a dummy model for development purposes"""
        logger.info("Creating a dummy model for development")
        
        # Create a simple model that returns random predictions
        inputs = keras.Input(shape=self.INPUT_SHAPE)
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        feature_layer = layers.Dense(64, activation='relu', name='feature_layer')(x)
        outputs = layers.Dense(len(self.CLASS_LABELS), activation='softmax')(feature_layer)
        
        self.model = keras.Model(inputs, outputs)
        self.feature_extractor = keras.Model(
            inputs=inputs,
            outputs=feature_layer
        )
        
        # Override predict method to return random predictions for development
        original_predict = self.model.predict
        def mock_predict(*args, **kwargs):
            batch_size = args[0].shape[0]
            return np.random.rand(batch_size, len(self.CLASS_LABELS))
        
        self.model.predict = mock_predict
        logger.warning("Using MOCK predictions - model is not trained!")
    
    def save_model(self, model_path: str = None):
        """
        Save trained model to file
        
        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            path = model_path or self.MODEL_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            self.model.save(path)
            self._save_config()
            
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def _save_config(self):
        """Save model configuration"""
        try:
            os.makedirs(os.path.dirname(self.CONFIG_PATH), exist_ok=True)
            with open(self.CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save configuration: {e}")
    
    def evaluate_model(self, test_data_dir: str) -> Dict:
        """
        Evaluate model performance on test dataset
        
        Args:
            test_data_dir: Directory containing test images
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            test_datagen = ImageDataGenerator()
            test_generator = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]),
                batch_size=self.config.get("batch_size", 32),
                class_mode='categorical',
                classes=self.CLASS_LABELS,
                shuffle=False
            )
            
            # Evaluate
            results = self.model.evaluate(test_generator, verbose=1)
            
            # Get predictions for detailed metrics
            predictions = self.model.predict(test_generator, verbose=1)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = test_generator.classes
            
            # Calculate per-class accuracy
            class_accuracies = {}
            for i, class_name in enumerate(self.CLASS_LABELS):
                class_mask = (true_classes == i)
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(predicted_classes[class_mask] == i)
                    class_accuracies[class_name] = float(class_acc)
            
            evaluation_results = {
                'test_loss': float(results[0]),
                'test_accuracy': float(results[1]),
                'test_top_3_accuracy': float(results[2]) if len(results) > 2 else None,
                'class_accuracies': class_accuracies,
                'total_samples': len(true_classes),
                'num_classes': len(self.CLASS_LABELS)
            }
            
            logger.info(f"Model evaluation completed: {evaluation_results['test_accuracy']:.3f} accuracy")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not loaded"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            self.model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary


class FeatureExtractor:
    """
    Standalone feature extractor for Kolam patterns
    """
    
    def __init__(self, classifier: KolamClassifier = None):
        """Initialize feature extractor"""
        self.classifier = classifier
        self.feature_cache = {}
    
    def extract_visual_features(self, image: Union[str, np.ndarray]) -> Dict:
        """
        Extract visual features from Kolam pattern
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Load and preprocess image
            if isinstance(image, str):
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Extract various features
            features = {
                'geometric_features': self._extract_geometric_features(img),
                'texture_features': self._extract_texture_features(img),
                'symmetry_features': self._extract_symmetry_features(img),
                'contour_features': self._extract_contour_features(img)
            }
            
            # Extract deep learning features if classifier available
            if self.classifier is not None and self.classifier.feature_extractor is not None:
                try:
                    dl_features = self.classifier.extract_features(image)
                    features['deep_features'] = dl_features.tolist()
                except Exception as e:
                    logger.warning(f"Could not extract deep features: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _extract_geometric_features(self, image: np.ndarray) -> Dict:
        """Extract geometric features"""
        # Placeholder for geometric feature extraction
        return {
            'aspect_ratio': image.shape[1] / image.shape[0],
            'area_ratio': np.sum(image > 0) / (image.shape[0] * image.shape[1]),
            'centroid_x': 0.5,  # Normalized centroid
            'centroid_y': 0.5
        }
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict:
        """Extract texture features"""
        # Placeholder for texture feature extraction
        return {
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
            'entropy': 0.0  # Placeholder
        }
    
    def _extract_symmetry_features(self, image: np.ndarray) -> Dict:
        """Extract symmetry features"""
        # Placeholder for symmetry analysis
        return {
            'horizontal_symmetry': 0.0,
            'vertical_symmetry': 0.0,
            'rotational_symmetry': 0.0,
            'radial_symmetry': 0.0
        }
    
    def _extract_contour_features(self, image: np.ndarray) -> Dict:
        """Extract contour-based features"""
        # Placeholder for contour analysis
        return {
            'num_contours': 0,
            'total_perimeter': 0.0,
            'complexity_index': 0.0
        }


# Utility functions
def load_trained_classifier(model_path: str = None) -> KolamClassifier:
    """
    Load a pre-trained Kolam classifier
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded KolamClassifier instance
    """
    classifier = KolamClassifier(model_path=model_path)
    return classifier


def batch_predict(classifier: KolamClassifier, image_paths: List[str]) -> List[Dict]:
    """
    Perform batch prediction on multiple images
    
    Args:
        classifier: Trained KolamClassifier
        image_paths: List of image file paths
        
    Returns:
        List of prediction results
    """
    results = []
    for image_path in image_paths:
        try:
            result = classifier.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results


if __name__ == "__main__":
    # Example usage
    classifier = KolamClassifier()
    
    # Create and train model (uncomment for training)
    # classifier.create_model()
    # history = classifier.train_model("data/train", "data/validation")
    
    # Load pre-trained model and make prediction
    # classifier.load_model()
    # result = classifier.predict("path/to/kolam_image.jpg")
    # print(f"Prediction: {result['predicted_class']} ({result['confidence']:.3f})")