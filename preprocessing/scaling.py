"""
Feature scaling functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

def scale_features(df: pd.DataFrame, method: str, columns: Optional[List[str]] = None, 
                  **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a scaling method to numeric features.
    
    Args:
        df: DataFrame to process
        method: Scaling method ('standard', 'minmax', 'robust', 'normalizer')
        columns: List of columns to scale (default: all numeric columns)
        **kwargs: Additional arguments for the specific scaling method
    
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    # Map method name to function
    method_map = {
        'standard': standardize,
        'minmax': minmax_scale,
        'robust': robust_scale,
        'normalizer': normalize
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Call the appropriate function
    return method_map[method](df, columns, **kwargs)

def standardize(df: pd.DataFrame, columns: Optional[List[str]] = None, 
               **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply standardization (z-score normalization) to features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to scale (default: all numeric columns)
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out columns that don't exist or aren't numeric
    columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return df_copy, {}
    
    # Create scaler
    scaler = StandardScaler(**kwargs)
    
    # Fit and transform
    df_copy[columns] = scaler.fit_transform(df[columns])
    
    return df_copy, {'scaler': scaler, 'columns': columns}

def minmax_scale(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                feature_range: Tuple[float, float] = (0, 1), 
                **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply min-max scaling to features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to scale (default: all numeric columns)
        feature_range: Range of transformed values
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out columns that don't exist or aren't numeric
    columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return df_copy, {}
    
    # Create scaler
    scaler = MinMaxScaler(feature_range=feature_range, **kwargs)
    
    # Fit and transform
    df_copy[columns] = scaler.fit_transform(df[columns])
    
    return df_copy, {'scaler': scaler, 'columns': columns}

def robust_scale(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                quantile_range: Tuple[float, float] = (25.0, 75.0), 
                **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply robust scaling to features (using median and quantiles).
    
    Args:
        df: DataFrame to process
        columns: List of columns to scale (default: all numeric columns)
        quantile_range: Quantile range to use
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out columns that don't exist or aren't numeric
    columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return df_copy, {}
    
    # Create scaler
    scaler = RobustScaler(quantile_range=quantile_range, **kwargs)
    
    # Fit and transform
    df_copy[columns] = scaler.fit_transform(df[columns])
    
    return df_copy, {'scaler': scaler, 'columns': columns}

def normalize(df: pd.DataFrame, columns: Optional[List[str]] = None, 
             norm: str = 'l2', **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply normalization to features (scale samples to unit norm).
    
    Args:
        df: DataFrame to process
        columns: List of columns to normalize (default: all numeric columns)
        norm: Norm to use ('l1', 'l2', 'max')
        
    Returns:
        Tuple of (normalized DataFrame, normalizer object)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out columns that don't exist or aren't numeric
    columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return df_copy, {}
    
    # Create normalizer
    normalizer = Normalizer(norm=norm, **kwargs)
    
    # Fit and transform
    # Note: Normalizer works on rows, not columns
    df_copy[columns] = normalizer.fit_transform(df[columns])
    
    return df_copy, {'normalizer': normalizer, 'columns': columns}

def log_transform(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                 base: Union[float, str] = 'e', epsilon: float = 1e-8) -> pd.DataFrame:
    """
    Apply logarithmic transformation to features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to transform (default: all numeric columns)
        base: Log base ('e' for natural log, 2 for log2, 10 for log10)
        epsilon: Small value to add to avoid log(0)
        
    Returns:
        DataFrame with transformed features
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out columns that don't exist or aren't numeric
    columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return df_copy
    
    # Apply transformation
    for col in columns:
        # Add epsilon to avoid log(0)
        values = df[col].values + epsilon
        
        if base == 'e':
            df_copy[col] = np.log(values)
        elif base == 2:
            df_copy[col] = np.log2(values)
        elif base == 10:
            df_copy[col] = np.log10(values)
        else:
            try:
                base_val = float(base)
                df_copy[col] = np.log(values) / np.log(base_val)
            except:
                raise ValueError(f"Invalid log base: {base}")
    
    return df_copy

def power_transform(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                   method: str = 'yeo-johnson') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply power transformation to make data more Gaussian-like.
    
    Args:
        df: DataFrame to process
        columns: List of columns to transform (default: all numeric columns)
        method: Transformation method ('yeo-johnson' or 'box-cox')
        
    Returns:
        Tuple of (transformed DataFrame, transformer object)
    """
    from sklearn.preprocessing import PowerTransformer
    
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out columns that don't exist or aren't numeric
    columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not columns:
        return df_copy, {}
    
    # Check if using Box-Cox (data must be positive)
    if method == 'box-cox':
        # Ensure data is positive
        for col in columns:
            if (df[col] <= 0).any():
                raise ValueError(f"Box-Cox transformation requires positive values, but column {col} contains non-positive values")
    
    # Create transformer
    transformer = PowerTransformer(method=method)
    
    # Fit and transform
    df_copy[columns] = transformer.fit_transform(df[columns])
    
    return df_copy, {'transformer': transformer, 'columns': columns}
