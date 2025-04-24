"""
Regression model functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import importlib
from core.constants import REGRESSION_ALGORITHMS
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import streamlit as st

def get_regressor(algorithm: str, **kwargs) -> Any:
    """
    Get a regressor instance based on algorithm name.
    
    Args:
        algorithm: Algorithm name or class path
        **kwargs: Algorithm parameters
    
    Returns:
        Regressor instance
    """
    # Get the class path from the mapping or use directly
    class_path = REGRESSION_ALGORITHMS.get(algorithm, algorithm)
    
    # Import the class
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        regressor_class = getattr(module, class_name)
        
        # Create instance with parameters
        regressor = regressor_class(**kwargs)
        return regressor
    except Exception as e:
        raise ValueError(f"Error creating regressor '{algorithm}': {str(e)}")

def train_regressor(X: Union[pd.DataFrame, np.ndarray], 
                   y: Union[pd.Series, np.ndarray],
                   algorithm: str,
                   **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a regression model.
    
    Args:
        X: Features
        y: Target values
        algorithm: Algorithm name or class path
        **kwargs: Algorithm parameters
    
    Returns:
        Tuple of (trained regressor, training metadata)
    """
    # Get regressor
    regressor = get_regressor(algorithm, **kwargs)
    
    # Train the model
    regressor.fit(X, y)
    
    # Get training metadata
    metadata = {
        'algorithm': algorithm,
        'parameters': kwargs,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'target_min': float(np.min(y)),
        'target_max': float(np.max(y)),
        'target_mean': float(np.mean(y)),
        'target_std': float(np.std(y))
    }
    
    # Add feature names if available
    if isinstance(X, pd.DataFrame):
        metadata['feature_names'] = X.columns.tolist()
    
    # Add feature importances if available
    try:
        if hasattr(regressor, 'feature_importances_'):
            importances = regressor.feature_importances_
            if isinstance(X, pd.DataFrame):
                metadata['feature_importance'] = dict(zip(X.columns, importances))
            else:
                metadata['feature_importance'] = importances.tolist()
        elif hasattr(regressor, 'coef_'):
            coefficients = regressor.coef_
            if isinstance(X, pd.DataFrame):
                metadata['feature_importance'] = dict(zip(X.columns, np.abs(coefficients)))
            else:
                metadata['feature_importance'] = np.abs(coefficients).tolist()
    except:
        pass
    
    return regressor, metadata

def evaluate_regressor(regressor: Any, 
                     X: Union[pd.DataFrame, np.ndarray], 
                     y: Union[pd.Series, np.ndarray],
                     metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate a regression model.
    
    Args:
        regressor: Trained regressor
        X: Features
        y: True values
        metrics: List of metrics to calculate
    
    Returns:
        Dictionary of metric names and values
    """
    # Default metrics
    if metrics is None:
        metrics = ['mae', 'mse', 'rmse', 'r2', 'mape']
    
    # Make predictions
    y_pred = regressor.predict(X)
    
    # Calculate metrics
    results = {}
    
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y, y_pred)
    
    if 'mse' in metrics:
        results['mse'] = mean_squared_error(y, y_pred)
    
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
    
    if 'r2' in metrics:
        results['r2'] = r2_score(y, y_pred)
    
    if 'mape' in metrics:
        # Handle zeros in actual values
        mask = y != 0
        if mask.any():
            results['mape'] = mean_absolute_percentage_error(y[mask], y_pred[mask])
        else:
            results['mape'] = np.nan
    
    return results

def predict_regressor(regressor: Any, 
                    X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Make predictions with a regression model.
    
    Args:
        regressor: Trained regressor
        X: Features
    
    Returns:
        Predicted values
    """
    # Make predictions
    return regressor.predict(X)

def save_regressor(regressor: Any, 
                 filename: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a trained regressor to a file.
    
    Args:
        regressor: Trained regressor
        filename: Output filename
        metadata: Optional metadata to save with the model
    
    Returns:
        Path to the saved model file
    """
    # Create a model package with metadata
    model_package = {
        'model': regressor,
        'type': 'regressor',
        'timestamp': pd.Timestamp.now().isoformat(),
        'metadata': metadata or {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save the model package
    with open(filename, 'wb') as f:
        pickle.dump(model_package, f)
    
    return filename

def load_regressor(filename: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a regressor from a file.
    
    Args:
        filename: Path to the model file
    
    Returns:
        Tuple of (regressor, metadata)
    """
    # Load the model package
    with open(filename, 'rb') as f:
        model_package = pickle.load(f)
    
    # Verify model type
    if model_package.get('type') != 'regressor':
        raise ValueError(f"Model in {filename} is not a regressor")
    
    # Extract model and metadata
    regressor = model_package['model']
    metadata = model_package.get('metadata', {})
    
    return regressor, metadata
