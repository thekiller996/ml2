"""
Feature selection functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    chi2, 
    f_classif, 
    f_regression, 
    mutual_info_classif, 
    mutual_info_regression,
    RFE,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression

def select_features(df: pd.DataFrame, method: str, target: Optional[pd.Series] = None,
                   columns: Optional[List[str]] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a feature selection method.
    
    Args:
        df: DataFrame with features
        method: Selection method name
        target: Target variable (required for some methods)
        columns: List of columns to consider (default: all numeric columns)
        **kwargs: Additional arguments for the specific method
    
    Returns:
        Tuple of (DataFrame with selected features, selection metadata)
    """
    # Map method name to function
    method_map = {
        'variance': select_by_variance,
        'correlation': select_by_correlation,
        'mutual_info': select_by_mutual_info,
        'model_based': select_by_model,
        'k_best': select_best_k,
        'rfe': recursive_feature_elimination
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Call the appropriate function
    return method_map[method](df, target, columns, **kwargs)

def select_by_variance(df: pd.DataFrame, target: Optional[pd.Series] = None,
                      columns: Optional[List[str]] = None, 
                      threshold: float = 0.01, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select features based on variance threshold.
    
    Args:
        df: DataFrame with features
        target: Not used for this method
        columns: List of columns to consider (default: all numeric columns)
        threshold: Variance threshold for feature selection
        
    Returns:
        Tuple of (DataFrame with selected features, selection metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'selected_features': []}
    
    # Create selector
    selector = VarianceThreshold(threshold=threshold)
    
    # Fit and transform
    X = df[columns]
    X_selected = selector.fit_transform(X)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = [columns[i] for i in selected_indices]
    
    # Create output DataFrame
    if len(selected_features) == 0:
        # No features selected
        result_df = df.drop(columns=columns)
    else:
        # Keep selected features and non-numeric columns
        non_selected = [col for col in columns if col not in selected_features]
        result_df = df.drop(columns=non_selected)
    
    # Create metadata
    feature_variances = pd.Series({col: selector.variances_[i] for i, col in enumerate(columns)})
    
    metadata = {
        'selected_features': selected_features,
        'variance_threshold': threshold,
        'feature_variances': feature_variances,
        'selector': selector
    }
    
    return result_df, metadata

def select_by_correlation(df: pd.DataFrame, target: Optional[pd.Series] = None,
                         columns: Optional[List[str]] = None, 
                         threshold: float = 0.7, 
                         target_threshold: Optional[float] = None, 
                         method: str = 'pearson', **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select features based on correlation.
    
    Args:
        df: DataFrame with features
        target: Target variable (optional)
        columns: List of columns to consider (default: all numeric columns)
        threshold: Correlation threshold for dropping features
        target_threshold: Minimum correlation with target to keep feature
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Tuple of (DataFrame with selected features, selection metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'selected_features': []}
    
    # Calculate correlation matrix
    X = df[columns]
    corr_matrix = X.corr(method=method).abs()
    
    # Create upper triangle mask
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    # Find pairs with correlation above threshold
    pairs = []
    for i, row in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            if upper_tri[i, j] and corr_matrix.iloc[i, j] > threshold:
                pairs.append((row, col, corr_matrix.iloc[i, j]))
    
    # Sort pairs by correlation (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Track features to drop
    to_drop = set()
    
    # If target is provided, calculate correlations with target
    if target is not None and target_threshold is not None:
        # Calculate correlation with target
        target_corr = pd.DataFrame()
        target_corr['correlation'] = X.corrwith(target, method=method).abs()
        
        # Create set of features with low correlation to target
        low_corr_features = set(target_corr[target_corr['correlation'] < target_threshold].index)
        to_drop.update(low_corr_features)
        
        # Update pairs to only consider non-dropped features
        pairs = [(col1, col2, corr) for col1, col2, corr in pairs 
                if col1 not in low_corr_features and col2 not in low_corr_features]
    
    # For each pair, drop the feature with lower correlation to target
    for col1, col2, corr in pairs:
        # Skip if both features are already marked for dropping
        if col1 in to_drop and col2 in to_drop:
            continue
        
        # If one feature is already marked, drop the other
        if col1 in to_drop:
            to_drop.add(col2)
            continue
        if col2 in to_drop:
            to_drop.add(col1)
            continue
        
        # Neither feature is marked yet
        if target is not None:
            # Drop feature with lower correlation to target
            corr1 = abs(X[col1].corr(target, method=method))
            corr2 = abs(X[col2].corr(target, method=method))
            
            if corr1 < corr2:
                to_drop.add(col1)
            else:
                to_drop.add(col2)
        else:
            # Without target, drop the second feature
            to_drop.add(col2)
    
    # Convert set to list for consistency
    to_drop = list(to_drop)
    
    # Get selected features
    selected_features = [col for col in columns if col not in to_drop]
    
    # Create output DataFrame
    result_df = df.drop(columns=to_drop)
    
    # Create metadata
    metadata = {
        'selected_features': selected_features,
        'correlation_threshold': threshold,
        'target_threshold': target_threshold,
        'dropped_features': to_drop,
        'correlation_method': method,
        'correlation_matrix': corr_matrix
    }
    
    if target is not None and target_threshold is not None:
        metadata['target_correlations'] = target_corr
    
    return result_df, metadata

def select_by_mutual_info(df: pd.DataFrame, target: pd.Series,
                         columns: Optional[List[str]] = None, 
                         k: Optional[int] = None,
                         percentile: Optional[int] = None,
                         task: str = 'auto', **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select features based on mutual information.
    
    Args:
        df: DataFrame with features
        target: Target variable
        columns: List of columns to consider (default: all numeric columns)
        k: Number of features to select
        percentile: Percentile of features to select
        task: ML task ('classification', 'regression', or 'auto')
        
    Returns:
        Tuple of (DataFrame with selected features, selection metadata)
    """
    if target is None:
        raise ValueError("Target variable is required for mutual information feature selection")
    
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'selected_features': []}
    
    # Determine task type if auto
    if task == 'auto':
        if pd.api.types.is_numeric_dtype(target) and len(np.unique(target)) > 10:
            task = 'regression'
        else:
            task = 'classification'
    
    # Determine score function
    if task == 'classification':
        score_func = mutual_info_classif
    elif task == 'regression':
        score_func = mutual_info_regression
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    # Create selector parameters
    selector_params = {'score_func': score_func}
    
    if k is not None:
        selector_params['k'] = min(k, len(columns))
    elif percentile is not None:
        selector_params['percentile'] = percentile
    else:
        # Default to selecting half of features
        selector_params['k'] = max(1, len(columns) // 2)
    
    # Create selector
    selector = SelectKBest(**selector_params)
    
    # Fit and transform
    X = df[columns]
    X_selected = selector.fit_transform(X, target)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = [columns[i] for i in selected_indices]
    
    # Get feature scores
    feature_scores = pd.Series(selector.scores_, index=columns)
    
    # Create output DataFrame
    result_df = df.drop(columns=[col for col in columns if col not in selected_features])
    
    # Create metadata
    metadata = {
        'selected_features': selected_features,
        'feature_scores': feature_scores,
        'task': task,
        'selector': selector
    }
    
    return result_df, metadata

def select_by_model(df: pd.DataFrame, target: pd.Series,
                   columns: Optional[List[str]] = None, 
                   model: Optional[Any] = None,
                   task: str = 'auto',
                   threshold: Optional[float] = None,
                   max_features: Optional[int] = None,
                   **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select features based on model feature importance.
    
    Args:
        df: DataFrame with features
        target: Target variable
        columns: List of columns to consider (default: all numeric columns)
        model: Model to use for feature importance
        task: ML task ('classification', 'regression', or 'auto')
        threshold: Importance threshold for feature selection
        max_features: Maximum number of features to select
        
    Returns:
        Tuple of (DataFrame with selected features, selection metadata)
    """
    if target is None:
        raise ValueError("Target variable is required for model-based feature selection")
    
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'selected_features': []}
    
    # Determine task type if auto
    if task == 'auto':
        if pd.api.types.is_numeric_dtype(target) and len(np.unique(target)) > 10:
            task = 'regression'
        else:
            task = 'classification'
    
    # Create default model if not provided
    if model is None:
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif task == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown task type: {task}")
    
    # Create selector parameters
    selector_params = {'estimator': model}
    
    if threshold is not None:
        selector_params['threshold'] = threshold
    elif max_features is not None:
        selector_params['max_features'] = min(max_features, len(columns))
    
    # Create selector
    selector = SelectFromModel(**selector_params)
    
    # Fit and transform
    X = df[columns]
    X_selected = selector.fit_transform(X, target)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = [columns[i] for i in selected_indices]
    
    # Get feature importances
    ###
    # Get feature importances
    try:
        importances = selector.estimator_.feature_importances_
    except:
        try:
            importances = selector.estimator_.coef_[0]
        except:
            importances = None
    
    if importances is not None:
        feature_importances = pd.Series(importances, index=columns)
    else:
        # Create dummy importances based on selection
        feature_importances = pd.Series(
            [1.0 if f in selected_features else 0.0 for f in columns],
            index=columns
        )
    
    # Create output DataFrame
    result_df = df.drop(columns=[col for col in columns if col not in selected_features])
    
    # Create metadata
    metadata = {
        'selected_features': selected_features,
        'feature_importances': feature_importances,
        'task': task,
        'selector': selector,
        'model': model
    }
    
    return result_df, metadata

def select_best_k(df: pd.DataFrame, target: pd.Series,
                 columns: Optional[List[str]] = None, 
                 k: int = 10,
                 task: str = 'auto',
                 score_func: Optional[Callable] = None,
                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select k best features based on statistical tests.
    
    Args:
        df: DataFrame with features
        target: Target variable
        columns: List of columns to consider (default: all numeric columns)
        k: Number of features to select
        task: ML task ('classification', 'regression', or 'auto')
        score_func: Scoring function to use
        
    Returns:
        Tuple of (DataFrame with selected features, selection metadata)
    """
    if target is None:
        raise ValueError("Target variable is required for SelectKBest feature selection")
    
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'selected_features': []}
    
    # Determine task type if auto
    if task == 'auto':
        if pd.api.types.is_numeric_dtype(target) and len(np.unique(target)) > 10:
            task = 'regression'
        else:
            task = 'classification'
    
    # Determine score function if not provided
    if score_func is None:
        if task == 'classification':
            # For classification, use chi2 for non-negative data, otherwise f_classif
            X = df[columns]
            if (X < 0).any().any():
                score_func = f_classif
            else:
                score_func = chi2
        elif task == 'regression':
            score_func = f_regression
        else:
            raise ValueError(f"Unknown task type: {task}")
    
    # Create selector
    k = min(k, len(columns))  # Ensure k is not greater than number of features
    selector = SelectKBest(score_func=score_func, k=k)
    
    # Fit and transform
    X = df[columns]
    X_selected = selector.fit_transform(X, target)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = [columns[i] for i in selected_indices]
    
    # Get feature scores
    feature_scores = pd.Series(selector.scores_, index=columns)
    
    # Create output DataFrame
    result_df = df.drop(columns=[col for col in columns if col not in selected_features])
    
    # Create metadata
    metadata = {
        'selected_features': selected_features,
        'feature_scores': feature_scores,
        'task': task,
        'selector': selector,
        'score_func': score_func.__name__ if hasattr(score_func, '__name__') else str(score_func)
    }
    
    return result_df, metadata

def recursive_feature_elimination(df: pd.DataFrame, target: pd.Series,
                                 columns: Optional[List[str]] = None, 
                                 model: Optional[Any] = None,
                                 task: str = 'auto',
                                 n_features_to_select: Optional[int] = None,
                                 step: int = 1,
                                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select features using Recursive Feature Elimination.
    
    Args:
        df: DataFrame with features
        target: Target variable
        columns: List of columns to consider (default: all numeric columns)
        model: Model to use for feature importance
        task: ML task ('classification', 'regression', or 'auto')
        n_features_to_select: Number of features to select
        step: Number of features to remove at each iteration
        
    Returns:
        Tuple of (DataFrame with selected features, selection metadata)
    """
    if target is None:
        raise ValueError("Target variable is required for RFE feature selection")
    
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'selected_features': []}
    
    # Determine task type if auto
    if task == 'auto':
        if pd.api.types.is_numeric_dtype(target) and len(np.unique(target)) > 10:
            task = 'regression'
        else:
            task = 'classification'
    
    # Create default model if not provided
    if model is None:
        if task == 'classification':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif task == 'regression':
            model = Lasso(random_state=42)
        else:
            raise ValueError(f"Unknown task type: {task}")
    
    # Set default n_features_to_select if not provided
    if n_features_to_select is None:
        n_features_to_select = max(1, len(columns) // 2)
    else:
        n_features_to_select = min(n_features_to_select, len(columns))
    
    # Create RFE
    rfe = RFE(
        estimator=model,
        n_features_to_select=n_features_to_select,
        step=step,
        verbose=kwargs.get('verbose', 0)
    )
    
    # Fit and transform
    X = df[columns]
    X_selected = rfe.fit_transform(X, target)
    
    # Get selected feature indices
    selected_indices = rfe.get_support(indices=True)
    selected_features = [columns[i] for i in selected_indices]
    
    # Get feature rankings
    feature_rankings = pd.Series(rfe.ranking_, index=columns)
    
    # Create output DataFrame
    result_df = df.drop(columns=[col for col in columns if col not in selected_features])
    
    # Create metadata
    metadata = {
        'selected_features': selected_features,
        'feature_rankings': feature_rankings,
        'task': task,
        'selector': rfe,
        'model': model
    }
    
    return result_df, metadata