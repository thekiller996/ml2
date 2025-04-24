"""
Feature transformation functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import scipy.stats as stats
import scipy.fftpack as fftpack

def apply_transformation(df: pd.DataFrame, method: str, columns: Optional[List[str]] = None,
                        **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a transformation to features.
    
    Args:
        df: DataFrame with features
        method: Transformation method name
        columns: List of columns to transform (default: all numeric columns)
        **kwargs: Additional arguments for the specific method
    
    Returns:
        Tuple of (DataFrame with transformed features, transformation metadata)
    """
    # Map method name to function
    method_map = {
        'log': lambda x, **kw: apply_math_func(x, np.log, columns, suffix='_log', epsilon=1e-10, **kw),
        'sqrt': lambda x, **kw: apply_math_func(x, np.sqrt, columns, suffix='_sqrt', min_value=0, **kw),
        'square': lambda x, **kw: apply_math_func(x, np.square, columns, suffix='_squared', **kw),
        'cube': lambda x, **kw: apply_math_func(x, lambda a: np.power(a, 3), columns, suffix='_cubed', **kw),
        'exp': lambda x, **kw: apply_math_func(x, np.exp, columns, suffix='_exp', **kw),
        'inverse': lambda x, **kw: apply_math_func(x, lambda a: 1.0 / a, columns, suffix='_inverse', epsilon=1e-10, **kw),
        'lag': lambda x, **kw: create_lag_features(x, columns, **kw),
        'window': lambda x, **kw: create_window_features(x, columns, **kw),
        'fft': lambda x, **kw: apply_spectral_transformation(x, columns, method='fft', **kw)
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown transformation method: {method}")
    
    # Call the appropriate function
    return method_map[method](df, **kwargs)

def apply_math_func(df: pd.DataFrame, func: Callable, columns: Optional[List[str]] = None,
                   suffix: str = '_transformed', inplace: bool = False,
                   min_value: Optional[float] = None, max_value: Optional[float] = None,
                   epsilon: float = 0.0, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a mathematical function to features.
    
    Args:
        df: DataFrame with features
        func: Function to apply
        columns: List of columns to transform (default: all numeric columns)
        suffix: Suffix to add to transformed column names
        inplace: Whether to replace original columns
        min_value: Minimum value to allow (clip values below)
        max_value: Maximum value to allow (clip values above)
        epsilon: Small value to add to prevent issues (e.g. log(0))
        
    Returns:
        Tuple of (DataFrame with transformed features, transformation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'transformed_features': []}
    
    # Create output DataFrame
    if inplace:
        result_df = df.copy()
        new_columns = columns
    else:
        result_df = df.copy()
        new_columns = [f"{col}{suffix}" for col in columns]
    
    # Apply transformation to each column
    for i, col in enumerate(columns):
        # Get values
        values = df[col].values.copy()
        
        # Apply clipping if needed
        if min_value is not None:
            values = np.maximum(values, min_value)
        if max_value is not None:
            values = np.minimum(values, max_value)
        
        # Add epsilon if needed
        if epsilon != 0.0:
            values = values + epsilon
        
        # Apply function
        try:
            transformed_values = func(values)
            result_df[new_columns[i]] = transformed_values
        except Exception as e:
            raise ValueError(f"Error transforming column '{col}': {str(e)}")
    
    # Create metadata
    metadata = {
        'transformed_features': new_columns,
        'source_columns': columns,
        'function': func.__name__ if hasattr(func, '__name__') else str(func),
        'inplace': inplace,
        'suffix': suffix
    }
    
    return result_df, metadata

def create_lag_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       lags: List[int] = [1, 2, 3], group_col: Optional[str] = None,
                       sort_col: Optional[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create lag features for time series data.
    
    Args:
        df: DataFrame with features
        columns: List of columns to transform (default: all numeric columns)
        lags: List of lag values to create
        group_col: Column to group by (e.g. ID column)
        sort_col: Column to sort by (e.g. time column)
        
    Returns:
        Tuple of (DataFrame with lag features, transformation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'transformed_features': []}
    
    # Ensure DataFrame is sorted
    result_df = df.copy()
    if sort_col is not None:
        if group_col is not None:
            # Sort within groups
            result_df = result_df.sort_values([group_col, sort_col])
        else:
            # Sort entire DataFrame
            result_df = result_df.sort_values(sort_col)
    
    # Create lag features
    new_columns = []
    
    for col in columns:
        for lag in lags:
            feature_name = f"{col}_lag_{lag}"
            
            if group_col is not None:
                # Create lag within groups
                result_df[feature_name] = result_df.groupby(group_col)[col].shift(lag)
            else:
                # Create lag for entire DataFrame
                result_df[feature_name] = result_df[col].shift(lag)
            
            new_columns.append(feature_name)
    
    # Create metadata
    metadata = {
        'transformed_features': new_columns,
        'source_columns': columns,
        'lags': lags,
        'group_column': group_col,
        'sort_column': sort_col
    }
    
    return result_df, metadata

def create_window_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                         window_size: int = 3, functions: List[str] = ['mean', 'std'],
                         group_col: Optional[str] = None, sort_col: Optional[str] = None,
                         **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create rolling window features for time series data.
    
    Args:
        df: DataFrame with features
        columns: List of columns to transform (default: all numeric columns)
        window_size: Size of the rolling window
        functions: List of functions to apply on window ('mean', 'std', 'min', 'max', 'sum', 'median')
        group_col: Column to group by (e.g. ID column)
        sort_col: Column to sort by (e.g. time column)
        
    Returns:
        Tuple of (DataFrame with window features, transformation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'transformed_features': []}
    
    # Ensure DataFrame is sorted
    result_df = df.copy()
    if sort_col is not None:
        if group_col is not None:
            # Sort within groups
            result_df = result_df.sort_values([group_col, sort_col])
        else:
            # Sort entire DataFrame
            result_df = result_df.sort_values(sort_col)
    
    # Map function names to actual functions
    function_map = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'sum': np.sum,
        'median': np.median,
        'count': len,
        'var': np.var,
        'skew': stats.skew,
        'kurt': stats.kurtosis
    }
    
    # Create window features
    new_columns = []
    
    for col in columns:
        for func_name in functions:
            feature_name = f"{col}_window_{window_size}_{func_name}"
            
            if func_name not in function_map:
                continue
                
            func = function_map[func_name]
            
            if group_col is not None:
                # Create window within groups
                result_df[feature_name] = result_df.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window_size, min_periods=1).apply(func, raw=True)
                )
            else:
                # Create window for entire DataFrame
                result_df[feature_name] = result_df[col].rolling(window_size, min_periods=1).apply(
                    func, raw=True
                )
            
            new_columns.append(feature_name)
    
    # Create metadata
    metadata = {
        'transformed_features': new_columns,
        'source_columns': columns,
        'window_size': window_size,
        'functions': functions,
        'group_column': group_col,
        'sort_column': sort_col
    }
    
    return result_df, metadata

def apply_spectral_transformation(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                 method: str = 'fft', n_components: int = 5,
                                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply spectral transformation (FFT, DCT) to features.
    
    Args:
        df: DataFrame with features
        columns: List of columns to transform (default: all numeric columns)
        method: Spectral method ('fft', 'dct')
        n_components: Number of frequency components to keep
        
    Returns:
        Tuple of (DataFrame with spectral features, transformation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'transformed_features': []}
    
    result_df = df.copy()
    new_columns = []
    
    for col in columns:
        values = df[col].values
        
        if method == 'fft':
            # Apply Fast Fourier Transform
            fft_result = np.fft.fft(values)
            # Get magnitudes of the first n components
            magnitudes = np.abs(fft_result)[:n_components]
            
            # Add magnitude features
            for i in range(min(n_components, len(magnitudes))):
                feature_name = f"{col}_fft_mag_{i+1}"
                result_df[feature_name] = magnitudes[i]
                new_columns.append(feature_name)
                
        elif method == 'dct':
            # Apply Discrete Cosine Transform
            dct_result = fftpack.dct(values, type=2, norm='ortho')
            # Keep first n components
            components = dct_result[:n_components]
            
            # Add DCT features
            for i in range(min(n_components, len(components))):
                feature_name = f"{col}_dct_{i+1}"
                result_df[feature_name] = components[i]
                new_columns.append(feature_name)
                
        else:
            raise ValueError(f"Unknown spectral method: {method}")
    
    # Create metadata
    metadata = {
        'transformed_features': new_columns,
        'source_columns': columns,
        'method': method,
        'n_components': n_components
    }
    
    return result_df, metadata
