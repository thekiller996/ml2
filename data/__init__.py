"""
Data handling functionality for the ML Platform.
"""

from data.loader import (
    load_data, 
    load_csv, 
    load_excel, 
    load_parquet, 
    load_json,
    load_pickle,
    load_sample_data
)
from data.explorer import (
    get_dataframe_info, 
    get_numeric_summary, 
    get_categorical_summary,
    analyze_missing_values,
    analyze_outliers,
    analyze_correlations
)
from data.exporter import (
    export_to_csv,
    export_to_excel,
    export_to_parquet,
    export_to_json,
    export_to_pickle,
    export_model
)

__all__ = [
    'load_data',
    'load_csv',
    'load_excel',
    'load_parquet',
    'load_json',
    'load_pickle',
    'load_sample_data',
    'get_dataframe_info',
    'get_numeric_summary',
    'get_categorical_summary',
    'analyze_missing_values',
    'analyze_outliers',
    'analyze_correlations',
    'export_to_csv',
    'export_to_excel',
    'export_to_parquet',
    'export_to_json',
    'export_to_pickle',
    'export_model'
]
