# Model Evaluator for Kolam AI

import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import logging
import pickle
from typing import Dict, List, Tuple, Union, Any, Optional
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

class ModelEvaluator:
    """Class for evaluating Kolam AI models (classifier and generator)"""
    
    def __init__(self, model=None, model_path: str = None):
        """
        Initialize the model evaluator
        
        Args:
            model: Trained model instance (optional)
            model_path: Path to saved model file (optional)
        """
        self.model = model
        self.model_path = model_path
        self.image_processor = ImageProcessor()
        
        if model is None and model_path is not None:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load model from file path"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Determine model type based on file extension or other indicators
        if self.model_path.endswith('.h5') or self.model_path.endswith('.keras'):
            try:
                # Load as Kolam classifier
                self.model = KolamClassifier(model_path=self.model_path)
                logger.info(f"Loaded KolamClassifier model from {self.model_path}")
            except Exception as e:
                # Fallback to standard Keras model
                logger.warning(f"Failed to load as KolamClassifier, trying standard keras.models.load_model: {e}")
                try:
                    self.model = keras.models.load_model(self.model_path)
                    logger.info(f"Loaded standard Keras model from {self.model_path}")
                except Exception as e2:
                    logger.error(f"Failed to load model: {e2}")
                    raise ValueError(f"Could not load model: {e2}")
        else:
            # Handle other model types (e.g., pickle files for scikit-learn models)
            logger.warning(f"Unsupported model format for automatic loading: {self.model_path}")
            raise ValueError(f"Unsupported model format: {self.model_path}")
    
    def _preprocess_data(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess input data for model evaluation
        
        Args:
            data: Input image data or features
            labels: Class labels (optional)
            
        Returns:
            Preprocessed data and labels
        """
        # If data is a list of file paths, load the images
        if isinstance(data[0], str):
            processed_data = []
            for img_path in data:
                try:
                    # Load and preprocess image
                    img = self.image_processor.load_image(img_path)
                    img = self.image_processor.resize_image(img, target_size=(224, 224))  # Standard size
                    processed_data.append(img)
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    continue
            
            if not processed_data:
                raise ValueError("No valid images found in the provided data")
                
            data = np.array(processed_data)
        
        # Ensure correct dimensions
        if len(data.shape) == 3 and data.shape[2] == 1:
            # Convert grayscale to RGB if needed
            data = np.repeat(data, 3, axis=2)
        elif len(data.shape) == 3 and data.shape[2] != 3:
            # Add channel dimension if needed
            data = np.expand_dims(data, axis=-1)
            data = np.repeat(data, 3, axis=-1)
            
        # Normalize pixel values to [0, 1]
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
            
        return data, labels
    
    def evaluate_classifier(self, test_data: np.ndarray, test_labels: np.ndarray, 
                           output_dir: str = "models/evaluation") -> Dict[str, Any]:
        """
        Evaluate a Kolam classifier model on test data
        
        Args:
            test_data: Test image data (numpy array or list of image paths)
            test_labels: Ground truth labels (one-hot encoded or class indices)
            output_dir: Directory to save evaluation results and visualizations
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model available for evaluation. Please provide a model or model_path.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess data
        logger.info("Preprocessing test data...")
        test_data, test_labels = self._preprocess_data(test_data, test_labels)
        
        # Convert labels from one-hot encoding to class indices if needed
        if len(test_labels.shape) > 1 and test_labels.shape[1] > 1:
            test_labels_indices = np.argmax(test_labels, axis=1)
        else:
            test_labels_indices = test_labels
        
        # Get class labels for the classifier
        if hasattr(self.model, 'CLASS_LABELS'):
            class_labels = self.model.CLASS_LABELS
        else:
            # Use generic class labels if unavailable
            num_classes = test_labels.shape[1] if len(test_labels.shape) > 1 else len(np.unique(test_labels))
            class_labels = [f"Class {i}" for i in range(num_classes)]
        
        # Make predictions
        logger.info("Running predictions on test data...")
        start_time = time.time()
        
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(test_data)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'predict'):
            predictions = self.model.model.predict(test_data)
        else:
            raise ValueError("Model does not have a 'predict' method")
            
        inference_time = time.time() - start_time
        logger.info(f"Prediction completed in {inference_time:.2f} seconds")
        
        # Convert predictions to class indices
        pred_indices = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = skmetrics.accuracy_score(test_labels_indices, pred_indices)
        precision = skmetrics.precision_score(test_labels_indices, pred_indices, average='weighted', zero_division=0)
        recall = skmetrics.recall_score(test_labels_indices, pred_indices, average='weighted', zero_division=0)
        f1 = skmetrics.f1_score(test_labels_indices, pred_indices, average='weighted', zero_division=0)
        
        # Calculate confusion matrix
        conf_matrix = skmetrics.confusion_matrix(test_labels_indices, pred_indices)
        
        # Calculate per-class metrics
        classification_report = skmetrics.classification_report(
            test_labels_indices, pred_indices, target_names=class_labels, output_dict=True
        )
        
        # Compile all metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "inference_time": inference_time,
            "samples_per_second": len(test_data) / inference_time,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": classification_report
        }
        
        # Save metrics to JSON file
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        
        # Create visualizations
        self._create_visualizations(metrics, output_dir, class_labels)
        
        # Save sample predictions
        self._save_sample_predictions(test_data, test_labels_indices, pred_indices, 
                                      predictions, class_labels, output_dir)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        return metrics
    
    def _create_visualizations(self, metrics: Dict[str, Any], output_dir: str, 
                              class_labels: List[str]) -> None:
        """Create and save evaluation visualizations"""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        conf_matrix = np.array(metrics["confusion_matrix"])
        
        # Normalize confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Replace NaNs with zeros
        
        sns.heatmap(
            conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels
        )
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()
        
        # Plot per-class metrics
        class_metrics = {}
        for cls in class_labels:
            if cls in metrics["classification_report"]:
                class_metrics[cls] = {
                    "precision": metrics["classification_report"][cls]["precision"],
                    "recall": metrics["classification_report"][cls]["recall"],
                    "f1-score": metrics["classification_report"][cls]["f1-score"]
                }
        
        # Convert to dataframe for easier plotting
        df = pd.DataFrame(class_metrics).T
        
        plt.figure(figsize=(12, 6))
        df.plot(kind='bar', ax=plt.gca())
        plt.title('Per-Class Performance Metrics')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=300)
        plt.close()
        
        # Overall metrics summary
        plt.figure(figsize=(8, 6))
        overall_metrics = {
            'Accuracy': metrics['accuracy'], 
            'Precision': metrics['precision'],
            'Recall': metrics['recall'], 
            'F1 Score': metrics['f1_score']
        }
        
        bars = plt.bar(overall_metrics.keys(), overall_metrics.values(), color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom'
            )
            
        plt.title('Overall Model Performance')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_metrics.png"), dpi=300)
        plt.close()
    
    def _save_sample_predictions(self, test_data: np.ndarray, true_labels: np.ndarray,
                               pred_indices: np.ndarray, pred_probs: np.ndarray,
                               class_labels: List[str], output_dir: str, num_samples: int = 10) -> None:
        """Save visualizations of sample predictions"""
        # Determine number of samples to visualize
        n_samples = min(num_samples, len(test_data))
        
        # Select random samples
        indices = np.random.choice(len(test_data), n_samples, replace=False)
        
        # Create a grid of images with predictions
        fig, axes = plt.subplots(2, 5, figsize=(15, 6)) if n_samples >= 10 else plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            if i < len(axes):
                img = test_data[idx]
                true_label = true_labels[idx]
                pred_label = pred_indices[idx]
                probability = pred_probs[idx][pred_label]
                
                # Display image
                axes[i].imshow(img)
                
                # Set title with prediction info
                title = f"True: {class_labels[true_label]}\nPred: {class_labels[pred_label]}\nConf: {probability:.2f}"
                axes[i].set_title(title, fontsize=8)
                axes[i].axis('off')
                
                # Color border based on correctness
                if true_label == pred_label:
                    axes[i].patch.set_edgecolor('green')
                else:
                    axes[i].patch.set_edgecolor('red')
                axes[i].patch.set_linewidth(2)
        
        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sample_predictions.png"), dpi=300)
        plt.close()
    
    def evaluate_generator(self, validation_data: Optional[np.ndarray] = None, 
                         output_dir: str = "models/evaluation") -> Dict[str, Any]:
        """
        Evaluate a Kolam generator model
        
        Args:
            validation_data: Data for validating generated patterns (optional)
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generator evaluation is limited to qualitative assessment.")
        
        # Since the generator is rule-based, we'll just create a placeholder
        metrics = {
            "note": "Generator model is rule-based and primarily evaluated qualitatively",
            "timestamp": time.time()
        }
        
        # Save metrics to JSON file
        metrics_path = os.path.join(output_dir, "generator_evaluation.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Generator evaluation info saved to {metrics_path}")
        return metrics

def evaluate_model(model: Any, test_data: np.ndarray, test_labels: Optional[np.ndarray] = None, 
                  output_dir: str = "models/evaluation", model_type: str = "classifier") -> Dict[str, Any]:
    """
    Evaluate a Kolam model and return metrics.
    
    Args:
        model: Trained model to evaluate
        test_data: Test data for evaluation
        test_labels: Ground truth labels (required for classifier)
        output_dir: Directory to save evaluation results and visualizations
        model_type: Type of model to evaluate ('classifier' or 'generator')
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator(model=model)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate appropriate model type
    if model_type.lower() == "classifier":
        if test_labels is None:
            raise ValueError("test_labels is required for evaluating a classifier")
        return evaluator.evaluate_classifier(test_data, test_labels, output_dir)
    elif model_type.lower() == "generator":
        return evaluator.evaluate_generator(test_data, output_dir)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'classifier' or 'generator'")
