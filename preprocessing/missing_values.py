"""
Missing value handling functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
import streamlit as st
from sklearn.impute import KNNImputer, SimpleImputer

def handle_missing_values(df: pd.DataFrame, method: str, columns: Optional[List[str]] = None, 
                         **kwargs) -> pd.DataFrame:
    """
    Apply a missing value handling method to a DataFrame.
    
    Args:
        df: DataFrame to process
        method: Method to use ('remove_rows', 'remove_columns', etc.)
        columns: List of columns to process (default: all columns)
        **kwargs: Additional arguments for the specific method
    
    Returns:
        Processed DataFrame
    """
    # Map method name to function
    method_map = {
        'remove_rows': remove_missing_rows,
        'remove_columns': remove_missing_columns,
        'fill_mean': fill_missing_mean,
        'fill_median': fill_missing_median,
        'fill_mode': fill_missing_mode,
        'fill_constant': fill_missing_constant,
        'fill_interpolation': fill_missing_interpolation,
        'fill_knn': fill_missing_knn
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown missing value handling method: {method}")
    
    # Call the appropriate function
    return method_map[method](df, columns, **kwargs)

def remove_missing_rows(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                        threshold: float = 1.0, **kwargs) -> pd.DataFrame:
    """
    Remove rows with missing values.
    
    Args:
        df: DataFrame to process
        columns: List of columns to check (default: all columns)
        threshold: Proportion of non-NA values required to keep a row (0.0 to 1.0)
        
    Returns:
        DataFrame with rows removed
    """
    if columns is None:
        # Check all columns
        if threshold == 1.0:
            return df.dropna()
        else:
            return df.dropna(thresh=int(threshold * df.shape[1]))
    else:
        # Check only specified columns
        return df.dropna(subset=columns)
    
def remove_missing_columns(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                           threshold: float = 0.5, **kwargs) -> pd.DataFrame:
    """
    Remove columns with too many missing values.
    
    Args:
        df: DataFrame to process
        columns: List of columns to check (default: all columns)
        threshold: Maximum proportion of missing values allowed (0.0 to 1.0)
        
    Returns:
        DataFrame with columns removed
    """
    if columns is None:
        columns = df.columns
        
    # Calculate missing value proportion for each column
    missing_prop = df[columns].isnull().mean()
    
    # Get columns to drop
    cols_to_drop = missing_prop[missing_prop > threshold].index.tolist()
    
    # Return DataFrame without dropped columns
    return df.drop(columns=cols_to_drop)

def fill_missing_mean(df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """
    Fill missing values with column mean.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all numeric columns)
        
    Returns:
        DataFrame with filled values
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create the imputer
    imputer = SimpleImputer(strategy='mean')
    
    # Apply to each column separately
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df_copy[col] = imputer.fit_transform(df[[col]])
    
    return df_copy

def fill_missing_median(df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """
    Fill missing values with column median.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all numeric columns)
        
    Returns:
        DataFrame with filled values
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create the imputer
    imputer = SimpleImputer(strategy='median')
    
    # Apply to each column separately
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df_copy[col] = imputer.fit_transform(df[[col]])
    
    return df_copy

def fill_missing_mode(df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """
    Fill missing values with column mode (most frequent value).
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all columns)
        
    Returns:
        DataFrame with filled values
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all columns
        columns = df.columns.tolist()
    
    # Create the imputer
    imputer = SimpleImputer(strategy='most_frequent')
    
    # Apply to each column separately
    for col in columns:
        if col in df.columns:
            df_copy[col] = imputer.fit_transform(df[[col]])
    
    return df_copy

def fill_missing_constant(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                          value: Any = 0, **kwargs) -> pd.DataFrame:
    """
    Fill missing values with a constant value.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all columns)
        value: Value to fill with
        
    Returns:
        DataFrame with filled values
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all columns
        columns = df.columns.tolist()
    
    # Apply to each column separately
    for col in columns:
        if col in df.columns:
            df_copy[col] = df[col].fillna(value)
    
    return df_copy

def fill_missing_interpolation(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                               method: str = 'linear', **kwargs) -> pd.DataFrame:
    """
    Fill missing values using interpolation.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all numeric columns)
        method: Interpolation method ('linear', 'time', 'quadratic', etc.)
        
    Returns:
        DataFrame with filled values
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Apply to each column separately
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df_copy[col] = df[col].interpolate(method=method)
    
    # Handle edge cases (start/end of series)
    df_copy = df_copy.ffill().bfill()
    
    return df_copy

def fill_missing_knn(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                     n_neighbors: int = 5, **kwargs) -> pd.DataFrame:
    """
    Fill missing values using K-Nearest Neighbors imputation.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (default: all numeric columns)
        n_neighbors: Number of neighbors to use
        
    Returns:
        DataFrame with filled values
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Filter to numeric columns from the provided list
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return df_copy
    
    # Create the imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Apply imputation to all numeric columns at once
    df_copy[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df_copy
