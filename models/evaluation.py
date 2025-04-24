"""
Model evaluation functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    classification_report as sk_classification_report,
    roc_curve as sk_roc_curve,
    precision_recall_curve as sk_precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

def split_data(X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None,
              test_size: float = 0.2,
              random_state: int = 42,
              stratify: bool = True) -> Tuple:
    """
    Split data into training and test sets.
    
    Args:
        X: Features
        y: Target variable (optional for unsupervised learning)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify the split based on target
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) if y is provided,
        otherwise (X_train, X_test)
    """
    if y is None:
        # Unsupervised learning (no target)
        return train_test_split(X, test_size=test_size, random_state=random_state)
    else:
        # Supervised learning (with target)
        stratify_param = y if stratify else None
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )

def cross_validate(model: Any, 
                 X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray],
                 cv: int = 5,
                 scoring: Union[str, List[str]] = None,
                 return_estimator: bool = False) -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target variable
        cv: Number of cross-validation folds
        scoring: Scoring metric(s) to use
        return_estimator: Whether to return trained estimators
    
    Returns:
        Dictionary with cross-validation results
    """
    # Set default scoring based on model type
    if scoring is None:
        # Try to detect problem type
        if hasattr(model, '_estimator_type'):
            if model._estimator_type == 'classifier':
                scoring = ['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
            elif model._estimator_type == 'regressor':
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            else:
                scoring = None
        else:
            scoring = None
    
    # Set return_estimator parameter
    cv_params = {
        'cv': cv,
        'scoring': scoring,
        'return_train_score': True,
        'return_estimator': return_estimator
    }
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, **cv_params)
    
    # Process results
    processed_results = {}
    
    # Calculate mean and std for each metric
    for key in cv_results:
        if isinstance(cv_results[key], np.ndarray) and key != 'estimator':
            processed_results[key] = {
                'mean': float(np.mean(cv_results[key])),
                'std': float(np.std(cv_results[key])),
                'values': cv_results[key].tolist()
            }
    
    # Add trained estimators if requested
    if return_estimator and 'estimator' in cv_results:
        processed_results['estimators'] = cv_results['estimator']
    
    return processed_results

def confusion_matrix(y_true: Union[pd.Series, np.ndarray], 
                   y_pred: Union[pd.Series, np.ndarray],
                   normalize: bool = False,
                   labels: Optional[List] = None) -> Dict[str, Any]:
    """
    Calculate confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize the counts
        labels: List of label values to include
    
    Returns:
        Dictionary with confusion matrix data
    """
    # Calculate confusion matrix
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
    
    # Get class labels
    if labels is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        unique_labels = np.array(labels)
    
    # Create result dictionary
    result = {
        'matrix': cm.tolist(),
        'labels': unique_labels.tolist(),
        'normalized': normalize
    }
    
    return result

def classification_report(y_true: Union[pd.Series, np.ndarray], 
                        y_pred: Union[pd.Series, np.ndarray],
                        labels: Optional[List] = None,
                        target_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate a classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label values to include
        target_names: Display names of the labels
    
    Returns:
        Dictionary with classification metrics
    """
    # Get the report as a dictionary
    report = sk_classification_report(
        y_true, 
        y_pred, 
        labels=labels, 
        target_names=target_names, 
        output_dict=True
    )
    
    return report

def regression_metrics(y_true: Union[pd.Series, np.ndarray], 
                     y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with regression metrics
    """
    from sklearn.metrics import (
        mean_squared_error, 
        mean_absolute_error,
        r2_score, 
        median_absolute_error,
        mean_absolute_percentage_error
    )
    
    # Calculate metrics
    metrics = {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'median_ae': float(median_absolute_error(y_true, y_pred))
    }
    
    # Calculate MAPE (handle zeros in actual values)
    mask = y_true != 0
    if mask.any():
        metrics['mape'] = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))
    else:
        metrics['mape'] = np.nan
    
    # Calculate additional statistics
    metrics['residuals'] = {
        'mean': float(np.mean(y_true - y_pred)),
        'std': float(np.std(y_true - y_pred)),
        'min': float(np.min(y_true - y_pred)),
        'max': float(np.max(y_true - y_pred))
    }
    
    return metrics

def clustering_metrics(X: Union[pd.DataFrame, np.ndarray], 
                     labels: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Calculate clustering evaluation metrics.
    
    Args:
        X: Features
        labels: Cluster labels
    
    Returns:
        Dictionary with clustering metrics
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score
    )
    
    # Only calculate metrics if there are at least 2 clusters and no noise points
    unique_labels = np.unique(labels)
    metrics = {
        'n_clusters': len(unique_labels[unique_labels >= 0]),
        'cluster_sizes': np.bincount(labels[labels >= 0]).tolist()
    }
    
    valid_clusters = len(unique_labels) >= 2 and -1 not in unique_labels
    
    if valid_clusters:
        try:
            metrics['silhouette'] = float(silhouette_score(X, labels))
        except:
            metrics['silhouette'] = np.nan
            
        try:
            metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
        except:
            metrics['davies_bouldin'] = np.nan
            
        try:
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
        except:
            metrics['calinski_harabasz'] = np.nan
    else:
        metrics['silhouette'] = np.nan
        metrics['davies_bouldin'] = np.nan
        metrics['calinski_harabasz'] = np.nan
    
    return metrics

def learning_curve(model: Any, 
                 X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray],
                 cv: int = 5,
                 train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
                 scoring: str = None) -> Dict[str, Any]:
    """
    Calculate learning curve data for a model.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target variable
        cv: Number of cross-validation folds
        train_sizes: Array of training set sizes to evaluate
        scoring: Scoring metric to use
    
    Returns:
        Dictionary with learning curve data
    """
    # Set default scoring based on model type
    if scoring is None:
        # Try to detect problem type
        if hasattr(model, '_estimator_type'):
            if model._estimator_type == 'classifier':
                scoring = 'accuracy'
            elif model._estimator_type == 'regressor':
                scoring = 'r2'
            else:
                scoring = None
        else:
            scoring = None
    
    # Calculate learning curve
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create result dictionary
    result = {
        'train_sizes': train_sizes_abs.tolist(),
        'train_scores': {
            'mean': train_mean.tolist(),
            'std': train_std.tolist(),
            'raw': train_scores.tolist()
        },
        'test_scores': {
            'mean': test_mean.tolist(),
            'std': test_std.tolist(),
            'raw': test_scores.tolist()
        },
        'scoring': scoring
    }
    
    return result

def roc_curve(y_true: Union[pd.Series, np.ndarray], 
            y_score: Union[pd.Series, np.ndarray],
            pos_label: Optional[Any] = None,
            multi_class: bool = False) -> Dict[str, Any]:
    """
    Calculate ROC curve data.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities or scores
        pos_label: Label of the positive class
        multi_class: Whether to calculate curves for multiclass classification
    
    Returns:
        Dictionary with ROC curve data
    """
    # Handle multiclass case
    if multi_class and len(np.unique(y_true)) > 2:
        # Get unique classes
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # Ensure y_score is proper shape for multiclass
        if y_score.ndim == 1:
            raise ValueError("y_score must have shape (n_samples, n_classes) for multiclass ROC curve")
        
        # Calculate ROC curve for each class
        results = {
            'classes': classes.tolist(),
            'curves': {}
        }
        
        for i, cls in enumerate(classes):
            fpr, tpr, thresholds = sk_roc_curve(y_true_bin[:, i], y_score[:, i])
            results['curves'][str(cls)] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        
        return results
    else:
        # Binary classification
        fpr, tpr, thresholds = sk_roc_curve(y_true, y_score, pos_label=pos_label)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }

def precision_recall_curve(y_true: Union[pd.Series, np.ndarray], 
                         y_score: Union[pd.Series, np.ndarray],
                         pos_label: Optional[Any] = None,
                         multi_class: bool = False) -> Dict[str, Any]:
    """
    Calculate precision-recall curve data.
    
    Args:
        y_true: True labels
        y_score: Predicted probabilities or scores
        pos_label: Label of the positive class
        multi_class: Whether to calculate curves for multiclass classification
    
    Returns:
        Dictionary with precision-recall curve data
    """
    # Handle multiclass case
    if multi_class and len(np.unique(y_true)) > 2:
        # Get unique classes
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # Ensure y_score is proper shape for multiclass
        if y_score.ndim == 1:
            raise ValueError("y_score must have shape (n_samples, n_classes) for multiclass precision-recall curve")
        
        # Calculate precision-recall curve for each class
        results = {
            'classes': classes.tolist(),
            'curves': {}
        }
        
        for i, cls in enumerate(classes):
            precision, recall, thresholds = sk_precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            results['curves'][str(cls)] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist() if len(thresholds) > 0 else []
            }
        
        return results
    else:
        # Binary classification
        precision, recall, thresholds = sk_precision_recall_curve(y_true, y_score, pos_label=pos_label)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist() if len(thresholds) > 0 else []
        }

def feature_importance(model: Any, 
                     feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extract feature importance from a model.
    
    Args:
        model: Trained model
        feature_names: Names of the features
    
    Returns:
        Dictionary mapping feature names to importance values
    """
    # Try different methods to get feature importance
    importance_values = None
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance_values = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        if model.coef_.ndim > 1 and model.coef_.shape[0] == 1:
            # Handle shaped like (1, n_features)
            importance_values = np.abs(model.coef_[0])
        else:
            # Handle shaped like (n_features,)
            importance_values = np.abs(model.coef_)
    
    if importance_values is None:
        return {}
    
    # If feature names were not provided, use indices
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance_values))]
    
    # Create dictionary
    importance_dict = {}
    for i, name in enumerate(feature_names):
        if i < len(importance_values):
            importance_dict[name] = float(importance_values[i])
    
    # Sort by importance (descending)
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict
