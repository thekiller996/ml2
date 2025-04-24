"""
Data exporting functions for the ML Platform.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path
import io
import streamlit as st
import config

def export_to_csv(df: pd.DataFrame, path: Optional[Union[str, Path]] = None, 
                  index: bool = False, **kwargs) -> Optional[io.StringIO]:
    """
    Export DataFrame to CSV.
    
    Args:
        df: DataFrame to export
        path: Path to save file (if None, returns StringIO object)
        index: Whether to include index in output
        **kwargs: Additional arguments to pass to df.to_csv()
    
    Returns:
        StringIO object with CSV data if path is None, otherwise None
    """
    if path is None:
        # Return as StringIO
        buffer = io.StringIO()
        df.to_csv(buffer, index=index, **kwargs)
        buffer.seek(0)
        return buffer
    else:
        # Save to file
        df.to_csv(path, index=index, **kwargs)
        return None

def export_to_excel(df: pd.DataFrame, path: Union[str, Path], 
                    sheet_name: str = 'Data', index: bool = False, **kwargs) -> None:
    """
    Export DataFrame to Excel.
    
    Args:
        df: DataFrame to export
        path: Path to save file
        sheet_name: Name of the worksheet
        index: Whether to include index in output
        **kwargs: Additional arguments to pass to df.to_excel()
    """
    df.to_excel(path, sheet_name=sheet_name, index=index, **kwargs)

def export_to_parquet(df: pd.DataFrame, path: Optional[Union[str, Path]] = None, 
                      index: bool = False, **kwargs) -> Optional[io.BytesIO]:
    """
    Export DataFrame to Parquet.
    
    Args:
        df: DataFrame to export
        path: Path to save file (if None, returns BytesIO object)
        index: Whether to include index in output
        **kwargs: Additional arguments to pass to df.to_parquet()
    
    Returns:
        BytesIO object with Parquet data if path is None, otherwise None
    """
    if path is None:
        # Return as BytesIO
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=index, **kwargs)
        buffer.seek(0)
        return buffer
    else:
        # Save to file
        df.to_parquet(path, index=index, **kwargs)
        return None

def export_to_json(df: pd.DataFrame, path: Optional[Union[str, Path]] = None, 
                   orient: str = 'records', **kwargs) -> Optional[str]:
    """
    Export DataFrame to JSON.
    
    Args:
        df: DataFrame to export
        path: Path to save file (if None, returns JSON string)
        orient: JSON format orientation
        **kwargs: Additional arguments to pass to df.to_json()
    
    Returns:
        JSON string if path is None, otherwise None
    """
    if path is None:
        # Return as string
        return df.to_json(orient=orient, **kwargs)
    else:
        # Save to file
        df.to_json(path, orient=orient, **kwargs)
        return None

def export_to_pickle(obj: Any, path: Optional[Union[str, Path]] = None) -> Optional[io.BytesIO]:
    """
    Export object to pickle format.
    
    Args:
        obj: Object to export
        path: Path to save file (if None, returns BytesIO object)
    
    Returns:
        BytesIO object with pickled data if path is None, otherwise None
    """
    if path is None:
        # Return as BytesIO
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)
        return buffer
    else:
        # Save to file
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        return None

def export_model(model: Any, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Export a trained model with metadata.
    
    Args:
        model: Model object to export
        model_name: Name to use for the model file
        metadata: Dictionary with model metadata
    
    Returns:
        Path to the saved model file
    """
    # Ensure models directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Create a safe filename
    safe_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
    filename = f"{safe_name}.pkl"
    model_path = os.path.join(config.MODELS_DIR, filename)
    
    # Create a model package with metadata
    model_package = {
        'model': model,
        'name': model_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'metadata': metadata or {}
    }
    
    # Save the model package
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    return model_path

def get_downloadable_dataframe(df: pd.DataFrame, format: str = 'csv') -> Dict[str, Any]:
    """
    Convert DataFrame to downloadable data in various formats.
    
    Args:
        df: DataFrame to convert
        format: Output format ('csv', 'excel', 'json', 'parquet')
    
    Returns:
        Dictionary with file content, MIME type, and filename
    """
    if format == 'csv':
        buffer = export_to_csv(df)
        return {
            'data': buffer.getvalue(),
            'mime': 'text/csv',
            'filename': 'data.csv'
        }
    elif format == 'excel':
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return {
            'data': buffer.getvalue(),
            'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'filename': 'data.xlsx'
        }
    elif format == 'json':
        json_str = export_to_json(df)
        return {
            'data': json_str,
            'mime': 'application/json',
            'filename': 'data.json'
        }
    elif format == 'parquet':
        buffer = export_to_parquet(df)
        return {
            'data': buffer.getvalue(),
            'mime': 'application/octet-stream',
            'filename': 'data.parquet'
        }
    else:
        raise ValueError(f"Unsupported format: {format}")
