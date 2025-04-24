"""
Classification model functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import importlib
from core.constants import CLASSIFICATION_ALGORITHMS
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import streamlit as st

def get_classifier(algorithm: str, **kwargs) -> Any:
    """
    Get a classifier instance based on algorithm name.
    
    Args:
        algorithm: Algorithm name or class path
        **kwargs: Algorithm parameters
    
    Returns:
        Classifier instance
    """
    # Get the class path from the mapping or use directly
    class_path = CLASSIFICATION_ALGORITHMS.get(algorithm, algorithm)
    
    # Import the class
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        classifier_class = getattr(module, class_name)
        
        # Create instance with parameters
        classifier = classifier_class(**kwargs)
        return classifier
    except Exception as e:
        raise ValueError(f"Error creating classifier '{algorithm}': {str(e)}")

def train_classifier(X: Union[pd.DataFrame, np.ndarray], 
                    y: Union[pd.Series, np.ndarray],
                    algorithm: str,
                    **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a classification model.
    
    Args:
        X: Features
        y: Target labels
        algorithm: Algorithm name or class path
        **kwargs: Algorithm parameters
    
    Returns:
        Tuple of (trained classifier, training metadata)
    """
    # Get classifier
    classifier = get_classifier(algorithm, **kwargs)
    
    # Train the model
    classifier.fit(X, y)
    
    # Get training metadata
    metadata = {
        'algorithm': algorithm,
        'parameters': kwargs,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'classes': np.unique(y).tolist()
    }
    
    # Add feature names if available
    if isinstance(X, pd.DataFrame):
        metadata['feature_names'] = X.columns.tolist()
    
    # Add feature importances if available
    try:
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            if isinstance(X, pd.DataFrame):
                metadata['feature_importance'] = dict(zip(X.columns, importances))
            else:
                metadata['feature_importance'] = importances.tolist()
        elif hasattr(classifier, 'coef_'):
            coefficients = classifier.coef_[0] if len(classifier.coef_.shape) > 1 else classifier.coef_
            if isinstance(X, pd.DataFrame):
                metadata['feature_importance'] = dict(zip(X.columns, np.abs(coefficients)))
            else:
                metadata['feature_importance'] = np.abs(coefficients).tolist()
    except:
        pass
    
    return classifier, metadata

def evaluate_classifier(classifier: Any, 
                      X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray],
                      metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate a classification model.
    
    Args:
        classifier: Trained classifier
        X: Features
        y: True labels
        metrics: List of metrics to calculate
    
    Returns:
        Dictionary of metric names and values
    """
    # Default metrics
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']
    
    # Make predictions
    y_pred = classifier.predict(X)
    
    # Get predicted probabilities if available
    try:
        y_prob = classifier.predict_proba(X)
    except:
        y_prob = None
    
    # Calculate metrics
    results = {}
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y, y_pred)
    
    if 'precision' in metrics:
        if len(np.unique(y)) > 2:
            # Multiclass
            results['precision'] = precision_score(y, y_pred, average='weighted')
        else:
            results['precision'] = precision_score(y, y_pred)
    
    if 'recall' in metrics:
        if len(np.unique(y)) > 2:
            # Multiclass
            results['recall'] = recall_score(y, y_pred, average='weighted')
        else:
            results['recall'] = recall_score(y, y_pred)
    
    if 'f1' in metrics:
        if len(np.unique(y)) > 2:
            # Multiclass
            results['f1'] = f1_score(y, y_pred, average='weighted')
        else:
            results['f1'] = f1_score(y, y_pred)
    
    if 'roc_auc' in metrics and y_prob is not None:
        try:
            if len(np.unique(y)) > 2:
                # Multiclass
                results['roc_auc'] = roc_auc_score(y, y_prob, multi_class='ovr')
            else:
                # Binary classification
                results['roc_auc'] = roc_auc_score(y, y_prob[:, 1])
        except:
            # Skip if ROC AUC can't be calculated
            pass
    
    if 'log_loss' in metrics and y_prob is not None:
        try:
            results['log_loss'] = log_loss(y, y_prob)
        except:
            # Skip if log loss can't be calculated
            pass
    
    return results

def predict_classifier(classifier: Any, 
                     X: Union[pd.DataFrame, np.ndarray],
                     return_proba: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions with a classification model.
    
    Args:
        classifier: Trained classifier
        X: Features
        return_proba: Whether to return class probabilities
    
    Returns:
        Predictions or tuple of (predictions, probabilities)
    """
    # Make class predictions
    y_pred = classifier.predict(X)
    
    if return_proba:
        try:
            # Get predicted probabilities
            y_prob = classifier.predict_proba(X)
            return y_pred, y_prob
        except:
            # Return predictions only if probabilities aren't available
            return y_pred
    else:
        return y_pred

def save_classifier(classifier: Any, 
                  filename: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a trained classifier to a file.
    
    Args:
        classifier: Trained classifier
        filename: Output filename
        metadata: Optional metadata to save with the model
    
    Returns:
        Path to the saved model file
    """
    # Create a model package with metadata
    model_package = {
        'model': classifier,
        'type': 'classifier',
        'timestamp': pd.Timestamp.now().isoformat(),
        'metadata': metadata or {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save the model package
    with open(filename, 'wb') as f:
        pickle.dump(model_package, f)
    
    return filename

def load_classifier(filename: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a classifier from a file.
    
    Args:
        filename: Path to the model file
    
    Returns:
        Tuple of (classifier, metadata)
    """
    # Load the model package
    with open(filename, 'rb') as f:
        model_package = pickle.load(f)
    
    # Verify model type
    if model_package.get('type') != 'classifier':
        raise ValueError(f"Model in {filename} is not a classifier")
    
    # Extract model and metadata
    classifier = model_package['model']
    metadata = model_package.get('metadata', {})
    
    return classifier, metadata
