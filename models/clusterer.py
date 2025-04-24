"""
Clustering model functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import importlib
from core.constants import CLUSTERING_ALGORITHMS
import pickle
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import streamlit as st

def get_clusterer(algorithm: str, **kwargs) -> Any:
    """
    Get a clusterer instance based on algorithm name.
    
    Args:
        algorithm: Algorithm name or class path
        **kwargs: Algorithm parameters
    
    Returns:
        Clusterer instance
    """
    # Get the class path from the mapping or use directly
    class_path = CLUSTERING_ALGORITHMS.get(algorithm, algorithm)
    
    # Import the class
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        clusterer_class = getattr(module, class_name)
        
        # Create instance with parameters
        clusterer = clusterer_class(**kwargs)
        return clusterer
    except Exception as e:
        raise ValueError(f"Error creating clusterer '{algorithm}': {str(e)}")

def train_clusterer(X: Union[pd.DataFrame, np.ndarray],
                  algorithm: str,
                  **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a clustering model.
    
    Args:
        X: Features
        algorithm: Algorithm name or class path
        **kwargs: Algorithm parameters
    
    Returns:
        Tuple of (trained clusterer, training metadata)
    """
    # Get clusterer
    clusterer = get_clusterer(algorithm, **kwargs)
    
    # Train the model (fit)
    clusterer.fit(X)
    
    # Get training metadata
    metadata = {
        'algorithm': algorithm,
        'parameters': kwargs,
        'n_features': X.shape[1],
        'n_samples': X.shape[0]
    }
    
    # Add feature names if available
    if isinstance(X, pd.DataFrame):
        metadata['feature_names'] = X.columns.tolist()
    
    # Add cluster information if available
    try:
        if hasattr(clusterer, 'labels_'):
            metadata['n_clusters'] = len(np.unique(clusterer.labels_))
            metadata['cluster_sizes'] = np.bincount(clusterer.labels_[clusterer.labels_ >= 0]).tolist()
    except:
        pass
    
    # Add cluster centers if available
    try:
        if hasattr(clusterer, 'cluster_centers_'):
            metadata['cluster_centers'] = clusterer.cluster_centers_.tolist()
    except:
        pass
    
    return clusterer, metadata

def evaluate_clusterer(clusterer: Any, 
                     X: Union[pd.DataFrame, np.ndarray],
                     metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate a clustering model.
    
    Args:
        clusterer: Trained clusterer
        X: Features
        metrics: List of metrics to calculate
    
    Returns:
        Dictionary of metric names and values
    """
    # Default metrics
    if metrics is None:
        metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    
    # Get cluster labels
    if hasattr(clusterer, 'labels_'):
        labels = clusterer.labels_
    else:
        labels = clusterer.predict(X)
    
    # Calculate metrics
    results = {}
    
    # Only calculate metrics if there are at least 2 clusters and no noise points
    unique_labels = np.unique(labels)
    valid_clusters = len(unique_labels) >= 2 and -1 not in unique_labels
    
    if valid_clusters:
        if 'silhouette' in metrics:
            try:
                results['silhouette'] = silhouette_score(X, labels)
            except:
                results['silhouette'] = np.nan
        
        if 'davies_bouldin' in metrics:
            try:
                results['davies_bouldin'] = davies_bouldin_score(X, labels)
            except:
                results['davies_bouldin'] = np.nan
        
        if 'calinski_harabasz' in metrics:
            try:
                results['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            except:
                results['calinski_harabasz'] = np.nan
    else:
        # Set metrics to NaN if we can't calculate them
        for metric in metrics:
            results[metric] = np.nan
    
    # Add cluster distribution
    results['n_clusters'] = len(unique_labels)
    results['cluster_distribution'] = np.bincount(labels[labels >= 0]).tolist()
    
    return results

def predict_clusterer(clusterer: Any, 
                    X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Predict cluster labels for new data.
    
    Args:
        clusterer: Trained clusterer
        X: Features
    
    Returns:
        Cluster labels
    """
    # Check if the model supports predict
    if hasattr(clusterer, 'predict'):
        return clusterer.predict(X)
    elif hasattr(clusterer, 'labels_'):
        # For models that don't have a predict method
        clusterer.fit(X)
        return clusterer.labels_
    else:
        raise ValueError("Clusterer doesn't support prediction")

def save_clusterer(clusterer: Any, 
                 filename: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a trained clusterer to a file.
    
    Args:
        clusterer: Trained clusterer
        filename: Output filename
        metadata: Optional metadata to save with the model
    
    Returns:
        Path to the saved model file
    """
    # Create a model package with metadata
    model_package = {
        'model': clusterer,
        'type': 'clusterer',
        'timestamp': pd.Timestamp.now().isoformat(),
        'metadata': metadata or {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save the model package
    with open(filename, 'wb') as f:
        pickle.dump(model_package, f)
    
    return filename

def load_clusterer(filename: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a clusterer from a file.
    
    Args:
        filename: Path to the model file
    
    Returns:
        Tuple of (clusterer, metadata)
    """
    # Load the model package
    with open(filename, 'rb') as f:
        model_package = pickle.load(f)
    
    # Verify model type
    if model_package.get('type') != 'clusterer':
        raise ValueError(f"Model in {filename} is not a clusterer")
    
    # Extract model and metadata
    clusterer = model_package['model']
    metadata = model_package.get('metadata', {})
    
    return clusterer, metadata
