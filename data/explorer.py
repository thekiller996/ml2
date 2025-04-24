"""
Data exploration functions for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get general information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with information about the DataFrame
    """
    info = {
        'shape': df.shape,
        'size': df.size,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'columns': df.columns.tolist(),
        'has_nulls': df.isnull().any().any(),
        'null_counts': df.isnull().sum().to_dict() if df.isnull().any().any() else None,
        'duplicated_rows': df.duplicated().sum()
    }
    return info

def get_numeric_summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get summary statistics for numeric columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to include (default: all numeric columns)
    
    Returns:
        DataFrame with summary statistics
    """
    if columns is None:
        # Get all numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        # Filter to specified columns
        numeric_df = df[columns].select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    # Basic statistics
    summary = numeric_df.describe()
    
    # Additional statistics
    summary.loc['skew'] = numeric_df.skew()
    summary.loc['kurtosis'] = numeric_df.kurtosis()
    summary.loc['median'] = numeric_df.median()
    summary.loc['missing'] = numeric_df.isnull().sum()
    summary.loc['missing_pct'] = numeric_df.isnull().mean() * 100
    
    return summary.transpose()

def get_categorical_summary(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                            max_categories: int = 20) -> Dict[str, pd.DataFrame]:
    """
    Get summary statistics for categorical columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to include (default: all object and category columns)
        max_categories: Maximum number of categories to display in frequency table
    
    Returns:
        Dictionary of DataFrames with summary statistics for each column
    """
    if columns is None:
        # Get all categorical columns
        cat_df = df.select_dtypes(include=['object', 'category'])
    else:
        # Filter to specified columns
        cat_df = df[columns]
    
    if cat_df.empty:
        return {}
    
    result = {}
    for col in cat_df.columns:
        # Value counts
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']
        value_counts['percentage'] = (value_counts['count'] / len(df) * 100).round(2)
        
        # Limit to max_categories
        if len(value_counts) > max_categories:
            top_values = value_counts.iloc[:max_categories-1]
            other_values = value_counts.iloc[max_categories-1:]
            other_row = pd.DataFrame({
                col: ['Other'],
                'count': [other_values['count'].sum()],
                'percentage': [other_values['percentage'].sum()]
            })
            value_counts = pd.concat([top_values, other_row], ignore_index=True)
        
        # Column statistics
        stats = pd.DataFrame({
            'statistic': ['unique_values', 'null_count', 'null_percentage', 'mode'],
            'value': [
                df[col].nunique(),
                df[col].isnull().sum(),
                round(df[col].isnull().mean() * 100, 2),
                df[col].mode()[0] if not df[col].mode().empty else None
            ]
        })
        
        result[col] = {
            'frequency': value_counts,
            'statistics': stats
        }
    
    return result

def analyze_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze missing values in a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Tuple of (column_missing_summary, row_missing_summary)
    """
    # Column missing summary
    col_missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('missing_percentage', ascending=False)
    
    # Row missing summary
    row_missing_count = df.isnull().sum(axis=1)
    bins = [0, 1, 2, 5, 10, 20, df.shape[1]]
    labels = ['0', '1', '2-5', '6-10', '11-20', f'>{20}']
    row_missing = pd.DataFrame({
        'missing_columns': pd.cut(row_missing_count, bins=bins, labels=labels).value_counts().sort_index()
    })
    row_missing['percentage'] = (row_missing['missing_columns'] / len(df) * 100).round(2)
    
    return col_missing, row_missing

def analyze_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                     method: str = 'zscore', threshold: float = 3.0) -> Dict[str, Any]:
    """
    Analyze outliers in numeric columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to check (default: all numeric columns)
        method: Method to use for outlier detection ('zscore', 'iqr')
        threshold: Threshold for outlier detection
    
    Returns:
        Dictionary with outlier information
    """
    if columns is None:
        # Get all numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        columns = numeric_df.columns.tolist()
    else:
        # Filter to specified columns
        numeric_df = df[columns].select_dtypes(include=[np.number])
    
    outliers = {}
    summary = pd.DataFrame(index=columns)
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        values = df[col].dropna()
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            outlier_idx = np.where(z_scores > threshold)[0]
            outlier_values = values.iloc[outlier_idx]
        elif method == 'iqr':
            # IQR method
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_idx = values[(values < lower_bound) | (values > upper_bound)].index
            outlier_values = values.loc[outlier_idx]
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        outliers[col] = outlier_values
        
        # Add to summary
        summary.loc[col, 'count'] = len(outlier_values)
        summary.loc[col, 'percentage'] = (len(outlier_values) / len(values) * 100).round(2)
        if not outlier_values.empty:
            summary.loc[col, 'min'] = outlier_values.min()
            summary.loc[col, 'max'] = outlier_values.max()
        else:
            summary.loc[col, 'min'] = None
            summary.loc[col, 'max'] = None
    
    return {
        'summary': summary,
        'details': outliers
    }

def analyze_correlations(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                         method: str = 'pearson', threshold: float = 0.7) -> pd.DataFrame:
    """
    Analyze correlations between numeric columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to include (default: all numeric columns)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        threshold: Correlation coefficient threshold to highlight
    
    Returns:
        DataFrame with correlation results
    """
    if columns is None:
        # Get all numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        columns = numeric_df.columns.tolist()
    else:
        # Filter to specified columns
        numeric_df = df[columns].select_dtypes(include=[np.number])
    
    if len(columns) < 2:
        return pd.DataFrame()
    
    # Calculate correlations
    corr = numeric_df.corr(method=method)
    
    # Create a DataFrame for strong correlations
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            col1, col2 = corr.columns[i], corr.columns[j]
            corr_value = corr.iloc[i, j]
            if abs(corr_value) >= threshold:
                corr_pairs.append({
                    'column_1': col1,
                    'column_2': col2,
                    'correlation': corr_value
                })
    
    return pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False)

def get_column_distribution(df: pd.DataFrame, column: str, bins: int = 30) -> Dict[str, Any]:
    """
    Get distribution statistics for a single column.
    
    Args:
        df: DataFrame containing the column
        column: Column name to analyze
        bins: Number of bins for histogram
    
    Returns:
        Dictionary with distribution information
    """
    if column not in df.columns:
        return None
        
    values = df[column].dropna()
    
    if pd.api.types.is_numeric_dtype(values):
        # Numeric column
        stats = {
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'skew': values.skew(),
            'kurtosis': values.kurtosis(),
            'type': 'numeric'
        }
        
        # Calculate histogram
        hist, bin_edges = np.histogram(values, bins=bins)
        stats['histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
    else:
        # Categorical column
        value_counts = values.value_counts().reset_index()
        value_counts.columns = ['value', 'count']
        value_counts['percentage'] = (value_counts['count'] / len(values) * 100).round(2)
        
        stats = {
            'unique_count': values.nunique(),
            'mode': values.mode()[0] if not values.mode().empty else None,
            'frequency': value_counts.to_dict(orient='records'),
            'type': 'categorical'
        }
    
    return stats
