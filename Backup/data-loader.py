"""
Data loading functions for the ML Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from typing import Optional, Union, Tuple, Dict, List, Any
import pickle
from pathlib import Path
import config

def load_data(file_obj: Union[io.BytesIO, Path], file_type: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load data from a file object based on file type.
    
    Args:
        file_obj: File object or path to load
        file_type: Type of file ('csv', 'excel', etc.)
    
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        if file_type == 'csv':
            return load_csv(file_obj), None
        elif file_type in ['xlsx', 'xls']:
            return load_excel(file_obj), None
        elif file_type == 'parquet':
            return load_parquet(file_obj), None
        elif file_type == 'json':
            return load_json(file_obj), None
        elif file_type == 'pkl':
            return load_pickle(file_obj), None
        else:
            return None, f"Unsupported file type: {file_type}"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def load_csv(file_obj: Union[io.BytesIO, Path], **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    
    Args:
        file_obj: File object or path to load
        **kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        DataFrame with loaded data
    """
    # Set some sensible defaults
    kwargs.setdefault('sep', ',')
    kwargs.setdefault('encoding', 'utf-8')
    kwargs.setdefault('na_values', ['NA', 'N/A', '', 'null', 'None'])
    
    return pd.read_csv(file_obj, **kwargs)

def load_excel(file_obj: Union[io.BytesIO, Path], **kwargs) -> pd.DataFrame:
    """
    Load an Excel file into a DataFrame.
    
    Args:
        file_obj: File object or path to load
        **kwargs: Additional arguments to pass to pd.read_excel
    
    Returns:
        DataFrame with loaded data
    """
    # Set some sensible defaults
    kwargs.setdefault('sheet_name', 0)
    kwargs.setdefault('na_values', ['NA', 'N/A', '', 'null', 'None'])
    
    return pd.read_excel(file_obj, **kwargs)

def load_parquet(file_obj: Union[io.BytesIO, Path], **kwargs) -> pd.DataFrame:
    """
    Load a Parquet file into a DataFrame.
    
    Args:
        file_obj: File object or path to load
        **kwargs: Additional arguments to pass to pd.read_parquet
    
    Returns:
        DataFrame with loaded data
    """
    return pd.read_parquet(file_obj, **kwargs)

def load_json(file_obj: Union[io.BytesIO, Path], **kwargs) -> pd.DataFrame:
    """
    Load a JSON file into a DataFrame.
    
    Args:
        file_obj: File object or path to load
        **kwargs: Additional arguments to pass to pd.read_json
    
    Returns:
        DataFrame with loaded data
    """
    return pd.read_json(file_obj, **kwargs)

def load_pickle(file_obj: Union[io.BytesIO, Path]) -> pd.DataFrame:
    """
    Load a pickled DataFrame.
    
    Args:
        file_obj: File object or path to load
    
    Returns:
        DataFrame with loaded data
    """
    return pd.read_pickle(file_obj)

def load_sample_data(dataset_name: str) -> pd.DataFrame:
    """
    Load a sample dataset.
    
    Args:
        dataset_name: Name of the sample dataset to load
    
    Returns:
        DataFrame with loaded sample data
    """
    if dataset_name == "iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    elif dataset_name == "boston":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    elif dataset_name == "wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    elif dataset_name == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    elif dataset_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
        
    else:
        raise ValueError(f"Unknown sample dataset: {dataset_name}")

def auto_detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect column types in a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with column types and corresponding column names
    """
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'id': []
    }
    
    # Check for ID columns
    id_patterns = ['id', 'identifier', 'key', 'code']
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column name contains ID patterns
        if any(pattern in col_lower for pattern in id_patterns) and df[col].nunique() > df.shape[0] * 0.9:
            column_types['id'].append(col)
            continue
        
        # Check data type
        dtype = df[col].dtype
        
        # Numeric columns
        if np.issubdtype(dtype, np.number):
            column_types['numeric'].append(col)
            continue
            
        # Datetime columns
        if pd.api.types.is_datetime64_any_dtype(dtype):
            column_types['datetime'].append(col)
            continue
            
        # Try to convert to datetime if it's a string
        if pd.api.types.is_string_dtype(dtype):
            try:
                pd.to_datetime(df[col], errors='raise')
                column_types['datetime'].append(col)
                continue
            except:
                pass
        
        # Categorical columns (few unique values)
        if df[col].nunique() < min(20, df.shape[0] * 0.1):
            column_types['categorical'].append(col)
            continue
            
        # Text columns (many unique values)
        if pd.api.types.is_string_dtype(dtype):
            column_types['text'].append(col)
            continue
            
        # Default to categorical
        column_types['categorical'].append(col)
    
    return column_types
