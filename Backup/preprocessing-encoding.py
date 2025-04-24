"""
Feature encoding functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import category_encoders as ce

def encode_features(df: pd.DataFrame, method: str, columns: Optional[List[str]] = None, 
                   **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply an encoding method to categorical features.
    
    Args:
        df: DataFrame to process
        method: Encoding method ('onehot', 'label', 'ordinal', 'target', 'binary', 'frequency')
        columns: List of columns to encode (default: all object and category columns)
        **kwargs: Additional arguments for the specific encoding method
    
    Returns:
        Tuple of (encoded DataFrame, encoder objects)
    """
    # Map method name to function
    method_map = {
        'onehot': one_hot_encode,
        'label': label_encode,
        'ordinal': ordinal_encode,
        'target': target_encode,
        'binary': binary_encode,
        'frequency': frequency_encode
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown encoding method: {method}")
    
    # Call the appropriate function
    return method_map[method](df, columns, **kwargs)

def one_hot_encode(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                  drop_first: bool = False, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply one-hot encoding to categorical features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to encode (default: all object and category columns)
        drop_first: Whether to drop first category
        
    Returns:
        Tuple of (encoded DataFrame, encoder object)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all categorical columns
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out columns that don't exist
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df_copy, {}
    
    # Create encoder
    encoder = OneHotEncoder(drop='first' if drop_first else None, sparse=False)
    
    # Fit encoder
    encoded_array = encoder.fit_transform(df[columns])
    
    # Get feature names
    feature_names = encoder.get_feature_names_out(columns)
    
    # Create DataFrame with encoded values
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    
    # Drop original columns and add encoded columns
    df_copy = df_copy.drop(columns=columns)
    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    
    return df_copy, {'encoder': encoder, 'columns': columns}

def label_encode(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply label encoding to categorical features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to encode (default: all object and category columns)
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all categorical columns
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out columns that don't exist
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df_copy, {}
    
    # Create encoders and transform each column
    encoders = {}
    for col in columns:
        encoder = LabelEncoder()
        df_copy[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    
    return df_copy, {'encoders': encoders, 'columns': columns}

def ordinal_encode(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                  categories: Optional[Dict[str, List[Any]]] = None, 
                  **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply ordinal encoding to categorical features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to encode (default: all object and category columns)
        categories: Dictionary mapping column names to category orders
        
    Returns:
        Tuple of (encoded DataFrame, encoder object)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all categorical columns
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out columns that don't exist
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df_copy, {}
    
    # Prepare categories list for encoder
    if categories is not None:
        category_list = [categories.get(col, None) for col in columns]
    else:
        category_list = 'auto'
    
    # Create encoder
    encoder = OrdinalEncoder(categories=category_list)
    
    # Fit and transform
    df_copy[columns] = encoder.fit_transform(df[columns].astype(str))
    
    return df_copy, {'encoder': encoder, 'columns': columns}

def target_encode(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                 target_column: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply target encoding to categorical features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to encode (default: all object and category columns)
        target_column: Name of the target column
        
    Returns:
        Tuple of (encoded DataFrame, encoder object)
    """
    if target_column is None:
        raise ValueError("Target column must be specified for target encoding")
    
    df_copy = df.copy()
    
    if columns is None:
        # Process all categorical columns except target
        all_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        columns = [col for col in all_cat_cols if col != target_column]
    
    # Filter out columns that don't exist
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df_copy, {}
    
    # Create encoder
    encoder = ce.TargetEncoder(cols=columns)
    
    # Fit and transform
    df_copy[columns] = encoder.fit_transform(df[columns], df[target_column])
    
    return df_copy, {'encoder': encoder, 'columns': columns}

def binary_encode(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply binary encoding to categorical features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to encode (default: all object and category columns)
        
    Returns:
        Tuple of (encoded DataFrame, encoder object)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all categorical columns
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out columns that don't exist
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df_copy, {}
    
    # Create encoder
    encoder = ce.BinaryEncoder(cols=columns)
    
    # Fit and transform
    encoded_df = encoder.fit_transform(df[columns])
    
    # Drop original columns
    df_copy = df_copy.drop(columns=columns)
    
    # Add encoded columns
    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    
    return df_copy, {'encoder': encoder, 'columns': columns}

def frequency_encode(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                    **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply frequency encoding to categorical features.
    
    Args:
        df: DataFrame to process
        columns: List of columns to encode (default: all object and category columns)
        
    Returns:
        Tuple of (encoded DataFrame, frequency maps)
    """
    df_copy = df.copy()
    
    if columns is None:
        # Process all categorical columns
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out columns that don't exist
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df_copy, {}
    
    # Create frequency maps
    freq_maps = {}
    
    for col in columns:
        # Calculate frequency map
        freq_map = df[col].value_counts(normalize=True)
        freq_maps[col] = freq_map
        
        # Replace values with frequencies
        df_copy[col] = df_copy[col].map(freq_map)
    
    return df_copy, {'freq_maps': freq_maps, 'columns': columns}
