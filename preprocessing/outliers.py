"""
Outlier detection and handling functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                    method: str = 'zscore', threshold: float = 3.0, **kwargs) -> Dict[str, np.ndarray]:
    """
    Detect outliers in numeric columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to check (default: all numeric columns)
        method: Method for outlier detection ('zscore', 'iqr', 'isolation_forest', 'lof')
        threshold: Threshold for outlier detection
        **kwargs: Additional arguments for the specific method
        
    Returns:
        Dictionary mapping column names to arrays of outlier indices
    """
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Map method name to function
    method_map = {
        'zscore': _detect_outliers_zscore,
        'iqr': _detect_outliers_iqr,
        'isolation_forest': _detect_outliers_isolation_forest,
        'lof': _detect_outliers_lof
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Call the appropriate function
    return method_map[method](df, columns, threshold, **kwargs)

def _detect_outliers_zscore(df: pd.DataFrame, columns: List[str], 
                            threshold: float = 3.0, **kwargs) -> Dict[str, np.ndarray]:
    """
    Detect outliers using Z-score method.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to check
        threshold: Z-score threshold
        
    Returns:
        Dictionary mapping column names to arrays of outlier indices
    """
    outliers = {}
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Get non-missing values
        values = df[col].dropna()
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(values))
        
        # Find outliers
        outlier_idx = np.where(z_scores > threshold)[0]
        if len(outlier_idx) > 0:
            outliers[col] = values.iloc[outlier_idx].index.values
    
    return outliers

def _detect_outliers_iqr(df: pd.DataFrame, columns: List[str], 
                         threshold: float = 1.5, **kwargs) -> Dict[str, np.ndarray]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to check
        threshold: IQR multiplier
        
    Returns:
        Dictionary mapping column names to arrays of outlier indices
    """
    outliers = {}
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Get non-missing values
        values = df[col].dropna()
        
        # Calculate quartiles and IQR
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Find outliers
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_idx = values[outlier_mask].index.values
        
        if len(outlier_idx) > 0:
            outliers[col] = outlier_idx
    
    return outliers

def _detect_outliers_isolation_forest(df: pd.DataFrame, columns: List[str], 
                                     threshold: float = 0.1, **kwargs) -> Dict[str, np.ndarray]:
    """
    Detect outliers using Isolation Forest algorithm.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to check
        threshold: Contamination parameter
        
    Returns:
        Dictionary mapping column names to arrays of outlier indices
    """
    # Filter columns
    valid_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not valid_columns:
        return {}
    
    # Extract features
    X = df[valid_columns].copy()
    
    # Fill missing values temporarily for algorithm
    X = X.fillna(X.mean())
    
    # Apply Isolation Forest
    model = IsolationForest(
        contamination=threshold,
        random_state=kwargs.get('random_state', 42)
    )
    
    # Predict outliers (-1 for outliers, 1 for inliers)
    y_pred = model.fit_predict(X)
    outlier_idx = np.where(y_pred == -1)[0]
    
    # Return all affected columns
    return {col: df.index.values[outlier_idx] for col in valid_columns}

def _detect_outliers_lof(df: pd.DataFrame, columns: List[str], 
                        threshold: float = 0.1, **kwargs) -> Dict[str, np.ndarray]:
    """
    Detect outliers using Local Outlier Factor (LOF) algorithm.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to check
        threshold: Contamination parameter
        
    Returns:
        Dictionary mapping column names to arrays of outlier indices
    """
    # Filter columns
    valid_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not valid_columns:
        return {}
    
    # Extract features
    X = df[valid_columns].copy()
    
    # Fill missing values temporarily for algorithm
    X = X.fillna(X.mean())
    
    # Apply LOF
    n_neighbors = kwargs.get('n_neighbors', 20)
    model = LocalOutlierFactor(
        n_neighbors=min(n_neighbors, len(X) - 1),
        contamination=threshold
    )
    
    # Predict outliers (-1 for outliers, 1 for inliers)
    y_pred = model.fit_predict(X)
    outlier_idx = np.where(y_pred == -1)[0]
    
    # Return all affected columns
    return {col: df.index.values[outlier_idx] for col in valid_columns}

def remove_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                   method: str = 'zscore', threshold: float = 3.0, **kwargs) -> pd.DataFrame:
    """
    Remove rows containing outliers.
    
    Args:
        df: DataFrame to process
        columns: List of columns to check (default: all numeric columns)
        method: Method for outlier detection
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    # Detect outliers
    outliers = detect_outliers(df, columns, method, threshold, **kwargs)
    
    if not outliers:
        return df.copy()
    
    # Collect all unique indices
    all_outlier_indices = set()
    for indices in outliers.values():
        all_outlier_indices.update(indices)
    
    # Remove rows with outliers
    return df.drop(index=list(all_outlier_indices))

def cap_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                method: str = 'iqr', threshold: float = 1.5, **kwargs) -> pd.DataFrame:
    """
    Cap outliers at specified quantiles.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all numeric columns)
        method: Method for outlier detection
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with capped outliers
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        if method == 'iqr':
            # Calculate quartiles and IQR
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Cap values
            df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'quantile':
            # Use quantiles directly
            lower_quantile = kwargs.get('lower_quantile', 0.01)
            upper_quantile = kwargs.get('upper_quantile', 0.99)
            
            lower_bound = df[col].quantile(lower_quantile)
            upper_bound = df[col].quantile(upper_quantile)
            
            # Cap values
            df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            # Get non-missing values
            values = df[col].dropna()
            
            # Calculate mean and std
            mean = values.mean()
            std = values.std()
            
            # Define bounds
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Cap values
            df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_copy

def replace_outliers_mean(df: pd.DataFrame, columns: Optional[List[str]] = None,
                         method: str = 'zscore', threshold: float = 3.0, **kwargs) -> pd.DataFrame:
    """
    Replace outliers with column mean.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all numeric columns)
        method: Method for outlier detection
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers replaced
    """
    # Detect outliers
    outliers = detect_outliers(df, columns, method, threshold, **kwargs)
    
    if not outliers:
        return df.copy()
    
    df_copy = df.copy()
    
    # Replace outliers with mean for each column
    for col, indices in outliers.items():
        mean_val = df[col].mean()
        df_copy.loc[indices, col] = mean_val
    
    return df_copy

def replace_outliers_median(df: pd.DataFrame, columns: Optional[List[str]] = None,
                           method: str = 'zscore', threshold: float = 3.0, **kwargs) -> pd.DataFrame:
    """
    Replace outliers with column median.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all numeric columns)
        method: Method for outlier detection
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers replaced
    """
    # Detect outliers
    outliers = detect_outliers(df, columns, method, threshold, **kwargs)
    
    if not outliers:
        return df.copy()
    
    df_copy = df.copy()
    
    # Replace outliers with median for each column
    for col, indices in outliers.items():
        median_val = df[col].median()
        df_copy.loc[indices, col] = median_val
    
    return df_copy
