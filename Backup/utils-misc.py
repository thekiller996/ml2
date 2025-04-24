"""
Miscellaneous utility functions for the ML tool.
Contains helper functions that don't fit in other categories.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import time
import datetime
import re
import hashlib
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def timer(func):
    """
    Decorator for timing function execution.
    
    Parameters:
    -----------
    func : callable
        Function to time
    
    Returns:
    --------
    callable
        Wrapped function that prints execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility across multiple libraries.
    
    Parameters:
    -----------
    seed : int
        Seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def format_bytes(bytes: int) -> str:
    """
    Format bytes to human-readable string.
    
    Parameters:
    -----------
    bytes : int
        Number of bytes
    
    Returns:
    --------
    str
        Formatted string (e.g. "3.45 MB")
    """
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024**2:
        return f"{bytes/1024:.2f} KB"
    elif bytes < 1024**3:
        return f"{bytes/1024**2:.2f} MB"
    else:
        return f"{bytes/1024**3:.2f} GB"

def get_memory_usage(df: pd.DataFrame) -> Dict[str, Union[int, str]]:
    """
    Get memory usage of a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns:
    --------
    dict
        Dictionary with memory usage information
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    return {
        'total_bytes': memory_bytes,
        'formatted': format_bytes(memory_bytes),
        'per_row': format_bytes(memory_bytes / len(df)) if len(df) > 0 else "0 B"
    }

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by downcasting numeric types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    verbose : bool
        Whether to print memory savings information
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with optimized dtypes
    """
    start_memory = df.memory_usage(deep=True).sum()
    
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    # Process all numeric columns
    for col in result.select_dtypes(include=['int', 'float']).columns:
        col_data = result[col]
        col_min, col_max = col_data.min(), col_data.max()
        
        # Integer columns
        if pd.api.types.is_integer_dtype(col_data):
            if col_min >= 0:  # Unsigned integers
                if col_max < np.iinfo(np.uint8).max:
                    result[col] = col_data.astype(np.uint8)
                elif col_max < np.iinfo(np.uint16).max:
                    result[col] = col_data.astype(np.uint16)
                elif col_max < np.iinfo(np.uint32).max:
                    result[col] = col_data.astype(np.uint32)
                else:
                    result[col] = col_data.astype(np.uint64)
            else:  # Signed integers
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    result[col] = col_data.astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    result[col] = col_data.astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    result[col] = col_data.astype(np.int32)
                else:
                    result[col] = col_data.astype(np.int64)
        
        # Float columns
        else:
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                result[col] = col_data.astype(np.float32)
            else:
                result[col] = col_data.astype(np.float64)
    
    # Calculate memory savings
    end_memory = result.memory_usage(deep=True).sum()
    reduction = 100 * (1 - end_memory/start_memory)
    
    if verbose:
        logger.info(f"Memory reduced from {format_bytes(start_memory)} to {format_bytes(end_memory)} ({reduction:.2f}% reduction)")
    
    return result

def save_model(model: Any, filepath: str, include_timestamp: bool = True):
    """
    Save a model to disk.
    
    Parameters:
    -----------
    model : object
        Model to save
    filepath : str
        Path to save the model
    include_timestamp : bool
        Whether to include a timestamp in the filename
    """
    if include_timestamp:
        # Add timestamp to filename
        path = Path(filepath)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filepath = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        filepath = str(new_filepath)
    
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")

def load_model(filepath: str) -> Any:
    """
    Load a model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the model file
    
    Returns:
    --------
    object
        Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load the model using pickle
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model

def save_pipeline(pipeline: Any, filepath: str, include_timestamp: bool = True):
    """
    Save a data processing pipeline to disk.
    
    Parameters:
    -----------
    pipeline : object
        Pipeline to save
    filepath : str
        Path to save the pipeline
    include_timestamp : bool
        Whether to include a timestamp in the filename
    """
    save_model(pipeline, filepath, include_timestamp)
    logger.info(f"Pipeline saved to {filepath}")

def load_pipeline(filepath: str) -> Any:
    """
    Load a data processing pipeline from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the pipeline file
    
    Returns:
    --------
    object
        Loaded pipeline
    """
    return load_model(filepath)

def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to a JSON file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    filepath : str
        Path to save the configuration
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict, tuple)) or value is None:
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    # Save as JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    logger.info(f"Configuration saved to {filepath}")

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the configuration file
    
    Returns:
    --------
    dict
        Loaded configuration
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    # Load JSON file
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Configuration loaded from {filepath}")
    return config

def copy_to_clipboard(text: str):
    """
    Copy text to clipboard (for Streamlit).
    
    Parameters:
    -----------
    text : str
        Text to copy
    """
    # Use streamlit to create a button that copies text to clipboard
    st.code(text)
    st.button(
        "Copy to clipboard",
        on_click=lambda: st.write(
            f'<span id="copy_text" style="display:none">{text}</span>'
            '<script>navigator.clipboard.writeText('
            'document.getElementById("copy_text").innerText);'
            '</script>',
            unsafe_allow_html=True
        )
    )

def create_hash(obj: Any) -> str:
    """
    Create a hash string from an object.
    
    Parameters:
    -----------
    obj : object
        Object to hash
    
    Returns:
    --------
    str
        Hash string
    """
    if isinstance(obj, pd.DataFrame):
        # For DataFrames, use a subset of data to create a hash
        data = obj.head(1000).to_csv()
    elif isinstance(obj, np.ndarray):
        # For NumPy arrays, use a subset of data
        data = obj.flatten()[:1000].tobytes()
    else:
        # For other objects, use pickle to serialize
        try:
            data = pickle.dumps(obj)
        except:
            data = str(obj).encode()
    
    # Create hash
    return hashlib.md5(data).hexdigest()

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
    
    Returns:
    --------
    str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)} minutes {int(seconds)} seconds"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename.
    
    Parameters:
    -----------
    filename : str
        Input filename
    
    Returns:
    --------
    str
        Sanitized filename
    """
    # Replace invalid characters with underscore
    s = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Remove leading/trailing whitespace and dots
    s = s.strip('. ')
    # Replace multiple spaces with single underscore
    s = re.sub(r'\s+', '_', s)
    # Ensure the name is not empty
    if not s:
        s = "unnamed"
    return s

def generate_id(prefix: str = '') -> str:
    """
    Generate a unique ID.
    
    Parameters:
    -----------
    prefix : str
        Prefix for the ID
    
    Returns:
    --------
    str
        Unique ID
    """
    import uuid
    return f"{prefix}{uuid.uuid4().hex[:8]}"

def chunker(seq: List[Any], size: int) -> List[List[Any]]:
    """
    Split a sequence into chunks of specified size.
    
    Parameters:
    -----------
    seq : list
        Input sequence
    size : int
        Chunk size
    
    Returns:
    --------
    list of list
        List of chunks
    """
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list.
    
    Parameters:
    -----------
    nested_list : list
        Nested list
    
    Returns:
    --------
    list
        Flattened list
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cleaned column names
    """
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    # Clean column names
    result.columns = [
        re.sub(r'[^\w\s]', '', str(col))  # Remove special characters
        .strip()  # Remove leading/trailing whitespace
        .lower()  # Convert to lowercase
        .replace(' ', '_')  # Replace spaces with underscores
        for col in result.columns
    ]
    
    return result

def get_pandas_dtypes(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get column names grouped by data type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns:
    --------
    dict
        Dictionary with data types as keys and lists of column names as values
    """
    dtypes = {}
    for dtype in ['int', 'float', 'object', 'datetime', 'category', 'bool']:
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        if cols:
            dtypes[dtype] = cols
    
    return dtypes

def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer and convert data types in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with inferred data types
    """
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    # Try to convert object columns to datetime
    for col in result.select_dtypes(include=['object']).columns:
        try:
            # Check if column can be converted to datetime
            datetime_series = pd.to_datetime(result[col], errors='raise')
            result[col] = datetime_series
            logger.info(f"Column '{col}' converted to datetime")
        except:
            # Check if column can be converted to numeric
            try:
                numeric_series = pd.to_numeric(result[col], errors='raise')
                result[col] = numeric_series
                logger.info(f"Column '{col}' converted to numeric")
            except:
                pass
    
    # Try to convert numeric columns with few unique values to category
    for col in result.select_dtypes(include=['int', 'float']).columns:
        unique_count = result[col].nunique()
        if unique_count <= 20 and unique_count / len(result) < 0.05:
            result[col] = result[col].astype('category')
            logger.info(f"Column '{col}' converted to category")
    
    return result

def create_chunks(df: pd.DataFrame, chunksize: int = 10000) -> List[pd.DataFrame]:
    """
    Split a DataFrame into chunks of specified size.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    chunksize : int
        Number of rows per chunk
    
    Returns:
    --------
    list of pandas.DataFrame
        List of DataFrame chunks
    """
    return [df.iloc[i:i + chunksize] for i in range(0, len(df), chunksize)]

def parallel_apply(df: pd.DataFrame, func: Callable, axis: int = 0, n_jobs: int = -1) -> pd.Series:
    """
    Apply a function to a DataFrame in parallel.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    func : callable
        Function to apply
    axis : int
        Apply function to each row (0) or column (1)
    n_jobs : int
        Number of jobs to run in parallel
    
    Returns:
    --------
    pandas.Series
        Result of applying the function
    """
    from joblib import Parallel, delayed
    
    # Split the DataFrame into chunks
    chunks = np.array_split(df, n_jobs)
    
    # Apply function to each chunk in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(lambda chunk: chunk.apply(func, axis=axis))(chunk) for chunk in chunks)
    
    # Combine results
    return pd.concat(results)

def get_duplicates_summary(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get summary of duplicate rows in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    subset : list of str, optional
        Column names to consider for identifying duplicates
    
    Returns:
    --------
    dict
        Dictionary with duplicate summary information
    """
    # Identify duplicates
    is_duplicate = df.duplicated(subset=subset, keep=False)
    duplicates = df[is_duplicate]
    
    # Count duplicates by group
    if not duplicates.empty:
        if subset:
            dup_counts = duplicates.groupby(subset).size().sort_values(ascending=False)
        else:
            dup_counts = duplicates.groupby(list(df.columns)).size().sort_values(ascending=False)
    else:
        dup_counts = pd.Series(dtype=int)
    
    return {
        'total_duplicates': is_duplicate.sum(),
        'duplicate_percent': (is_duplicate.sum() / len(df)) * 100 if len(df) > 0 else 0,
        'unique_duplicate_groups': len(dup_counts),
        'max_duplicates_in_group': dup_counts.max() if not dup_counts.empty else 0,
        'most_common_duplicates': dup_counts.head(5).to_dict() if not dup_counts.empty else {}
    }

def auto_detect_categorical(df: pd.DataFrame, max_unique: int = 20, 
                           min_freq: float = 0.01) -> List[str]:
    """
    Automatically detect categorical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    max_unique : int
        Maximum number of unique values for a column to be considered categorical
    min_freq : float
        Minimum frequency of values to not be considered outliers
    
    Returns:
    --------
    list
        List of likely categorical column names
    """
    categorical_cols = []
    
    # Check all columns
    for col in df.columns:
        n_unique = df[col].nunique()
        n_samples = len(df)
        
        # Skip if too many unique values relative to dataframe size
        if n_unique > max(max_unique, n_samples * min_freq):
            continue
        
        # For numeric columns, check distribution
        if pd.api.types.is_numeric_dtype(df[col]):
            # If values are mostly integers
            if (df[col].dropna() == df[col].dropna().astype(int)).all():
                value_counts = df[col].value_counts(normalize=True)
                # Check if the distribution is mostly discrete values
                if (value_counts >= min_freq).sum() <= max_unique:
                    categorical_cols.append(col)
        
        # For object columns with few unique values
        elif pd.api.types.is_object_dtype(df[col]) and n_unique <= max_unique:
            categorical_cols.append(col)
        
        # Already categorical
        elif pd.api.types.is_categorical_dtype(df[col]):
            categorical_cols.append(col)
    
    return categorical_cols

def save_df_as_parquet(df: pd.DataFrame, filepath: str, compression: str = 'snappy'):
    """
    Save DataFrame to parquet format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    filepath : str
        Path to save the file
    compression : str
        Compression method
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save as parquet
    df.to_parquet(filepath, compression=compression)
    
    # Get file size
    file_size = os.path.getsize(filepath)
    logger.info(f"DataFrame saved to {filepath} ({format_bytes(file_size)})")

def create_data_sample(df: pd.DataFrame, size: int = 1000, 
                      stratify_column: Optional[str] = None,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Create a sample of a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    size : int
        Sample size
    stratify_column : str, optional
        Column to use for stratified sampling
    random_state : int
        Random seed
    
    Returns:
    --------
    pandas.DataFrame
        Sampled DataFrame
    """
    # Cap the sample size to the DataFrame size
    sample_size = min(size, len(df))
    
    # Stratified sampling
    if stratify_column and stratify_column in df.columns:
        try:
            from sklearn.model_selection import train_test_split
            _, sample = train_test_split(
                df, 
                test_size=sample_size/len(df),
                stratify=df[stratify_column],
                random_state=random_state
            )
            return sample
        except:
            logger.warning(f"Stratified sampling failed, falling back to random sampling")
            return df.sample(n=sample_size, random_state=random_state)
    else:
        # Random sampling
        return df.sample(n=sample_size, random_state=random_state)

def suggest_batch_size(data_size: int, model_complexity: str = 'medium', 
                      available_memory: str = 'medium') -> int:
    """
    Suggest batch size based on data size and model complexity.
    
    Parameters:
    -----------
    data_size : int
        Number of samples in dataset
    model_complexity : str
        Model complexity: 'low', 'medium', or 'high'
    available_memory : str
        Available memory: 'low', 'medium', or 'high'
    
    Returns:
    --------
    int
        Suggested batch size
    """
    # Base batch sizes for different memory levels
    if available_memory == 'low':
        base_batch = 16
    elif available_memory == 'medium':
        base_batch = 32
    else:  # high
        base_batch = 64
    
    # Adjust for model complexity
    if model_complexity == 'low':
        complexity_factor = 2.0
    elif model_complexity == 'medium':
        complexity_factor = 1.0
    else:  # high
        complexity_factor = 0.5
    
    # Adjust for dataset size (larger datasets might need smaller batches)
    if data_size < 1000:
        size_factor = 2.0
    elif data_size < 10000:
        size_factor = 1.5
    elif data_size < 100000:
        size_factor = 1.0
    else:
        size_factor = 0.5
    
    # Calculate batch size
    batch_size = int(base_batch * complexity_factor * size_factor)
    
    # Make sure batch size is a power of 2
    batch_size = 2 ** int(np.log2(batch_size))
    
    # Make sure batch size is at least 8 and at most 512
    batch_size = max(8, min(512, batch_size))
    
    return batch_size

def detect_outliers_iqr(df: pd.DataFrame, column: str, 
                       threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a column using the IQR method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column : str
        Column name to check for outliers
    threshold : float
        IQR multiplier to determine outliers
    
    Returns:
    --------
    pandas.Series
        Boolean series indicating outliers
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def detect_outliers_zscore(df: pd.DataFrame, column: str, 
                          threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a column using the Z-score method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column : str
        Column name to check for outliers
    threshold : float
        Z-score threshold to determine outliers
    
    Returns:
    --------
    pandas.Series
        Boolean series indicating outliers
    """
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    return pd.Series(z_scores > threshold, index=df[column].dropna().index)

def generate_random_id(length: int = 8) -> str:
    """
    Generate a random alphanumeric ID.
    
    Parameters:
    -----------
    length : int
        Length of the ID
    
    Returns:
    --------
    str
        Random ID
    """
    import random
    import string
    
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def check_gpu_availability():
    """
    Check if GPU is available for training.
    
    Returns:
    --------
    bool
        True if GPU is available, False otherwise
    """
    # Check TensorFlow
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        if gpu_available:
            logger.info("TensorFlow GPU is available")
            return True
    except:
        pass
    
    # Check PyTorch
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            logger.info(f"PyTorch GPU is available: {torch.cuda.get_device_name(0)}")
            return True
    except:
        pass
    
    logger.info("No GPU available, using CPU")
    return False

def create_timestamp_filename(prefix: str, extension: str) -> str:
    """
    Create a filename with a timestamp.
    
    Parameters:
    -----------
    prefix : str
        Prefix for the filename
    extension : str
        File extension
    
    Returns:
    --------
    str
        Filename with timestamp
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension.lstrip('.')}"

def is_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook.
    
    Returns:
    --------
    bool
        True if running in a notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal IPython
        else:
            return False  # Other type
    except NameError:
        return False  # Not running in IPython

def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to parse string columns into datetime.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with parsed datetime columns
    """
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    # Try to convert object columns to datetime
    for col in result.select_dtypes(include=['object']).columns:
        # Check if column contains date-like strings
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        # Sample the column to check for date patterns
        sample = result[col].dropna().astype(str).sample(min(100, len(result[col].dropna())))
        
        is_date_column = False
        for pattern in date_patterns:
            if sample.str.contains(pattern).mean() > 0.8:  # If more than 80% match the pattern
                is_date_column = True
                break
        
        if is_date_column:
            try:
                result[col] = pd.to_datetime(result[col], errors='coerce')
                logger.info(f"Column '{col}' converted to datetime")
            except:
                pass
    
    return result

def get_ram_usage() -> Dict[str, str]:
    """
    Get RAM usage information.
    
    Returns:
    --------
    dict
        Dictionary with RAM usage information
    """
    import psutil
    
    # Get memory information
    memory = psutil.virtual_memory()
    
    return {
        'total': format_bytes(memory.total),
        'available': format_bytes(memory.available),
        'used': format_bytes(memory.used),
        'percent': f"{memory.percent}%"
    }

def suggest_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Suggest appropriate data types for columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns:
    --------
    dict
        Dictionary with column names as keys and suggested types as values
    """
    suggestions = {}
    
    for col in df.columns:
        # Skip columns that are already non-object types
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        
        # Sample values
        values = df[col].dropna().sample(min(100, len(df[col].dropna())))
        
        # Check if it could be numeric
        numeric_ratio = pd.to_numeric(values, errors='coerce').notna().mean()
        
        # Check if it could be datetime
        datetime_ratio = pd.to_datetime(values, errors='coerce').notna().mean()
        
        # Check if it's a categorical with few unique values
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1.0
        
        # Determine the most likely type
        if numeric_ratio > 0.8:
            if all((df[col].dropna().astype(str).str.contains(r'\.') == False)):
                suggestions[col] = 'int'
            else:
                suggestions[col] = 'float'
        elif datetime_ratio > 0.8:
            suggestions[col] = 'datetime'
        elif unique_ratio < 0.05 and df[col].nunique() <= 100:
            suggestions[col] = 'category'
        else:
            suggestions[col] = 'object'
    
    return suggestions

def get_sql_schema(df: pd.DataFrame) -> str:
    """
    Generate SQL CREATE TABLE schema for DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns:
    --------
    str
        SQL CREATE TABLE statement
    """
    # Map pandas dtypes to SQL types
    dtype_map = {
        'int64': 'INTEGER',
        'int32': 'INTEGER',
        'int16': 'SMALLINT',
        'int8': 'SMALLINT',
        'uint64': 'BIGINT',
        'uint32': 'INTEGER',
        'uint16': 'INTEGER',
        'uint8': 'SMALLINT',
        'float64': 'FLOAT',
        'float32': 'FLOAT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
        'timedelta64[ns]': 'VARCHAR(255)',
        'category': 'VARCHAR(255)',
        'object': 'VARCHAR(255)'
    }
    
    # Generate column definitions
    column_defs = []
    for col_name, dtype in df.dtypes.items():
        sql_type = dtype_map.get(str(dtype), 'VARCHAR(255)')
        column_defs.append(f'    "{col_name}" {sql_type}')
    
    # Build CREATE TABLE statement
    table_name = "table_name"
    sql = f"CREATE TABLE {table_name} (\n"
    sql += ",\n".join(column_defs)
    sql += "\n);"
    
    return sql

def create_code_snippet(language: str, code: str) -> str:
    """
    Format code for display in Streamlit.
    
    Parameters:
    -----------
    language : str
        Programming language
    code : str
        Code to format
    
    Returns:
    --------
    str
        Formatted code for display
    """
    return f"```{language}\n{code}\n```"

def calculate_execution_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Calculate execution time of a function.
    
    Parameters:
    -----------
    func : callable
        Function to measure
    *args, **kwargs
        Arguments to pass to the function
    
    Returns:
    --------
    tuple
        (function result, execution time in seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Parameters:
    -----------
    filename : str
        Input filename
    
    Returns:
    --------
    str
        File extension (lowercase, without dot)
    """
    return os.path.splitext(filename)[1].lower().lstrip('.')

def check_dependencies(packages: List[str]) -> Dict[str, bool]:
    """
    Check if required packages are installed.
    
    Parameters:
    -----------
    packages : list of str
        List of package names to check
    
    Returns:
    --------
    dict
        Dictionary with package names as keys and boolean availability as values
    """
    import importlib
    
    results = {}
    for package in packages:
        try:
            importlib.import_module(package)
            results[package] = True
        except ImportError:
            results[package] = False
    
    return results

def install_dependencies(packages: List[str], upgrade: bool = False):
    """
    Install required packages using pip.
    
    Parameters:
    -----------
    packages : list of str
        List of package names to install
    upgrade : bool
        Whether to upgrade existing packages
    """
    import subprocess
    import sys
    
    for package in packages:
        try:
            if upgrade:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to install {package}")
