"""
Common UI elements for the ML Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Callable
import base64
import io

def show_header(title: str, description: str = None):
    """
    Display a header with optional description.
    
    Args:
        title: Header title text
        description: Optional description text
    """
    st.title(title)
    if description:
        st.markdown(description)
    st.divider()

def show_info(message: str):
    """
    Display an info message.
    
    Args:
        message: Message to display
    """
    st.info(message)

def show_success(message: str):
    """
    Display a success message.
    
    Args:
        message: Message to display
    """
    st.success(message)

def show_warning(message: str):
    """
    Display a warning message.
    
    Args:
        message: Message to display
    """
    st.warning(message)

def show_error(message: str):
    """
    Display an error message.
    
    Args:
        message: Message to display
    """
    st.error(message)

def show_progress(iterable, message: str = "Processing..."):
    """
    Display a progress bar while iterating.
    
    Args:
        iterable: Iterable to track progress for
        message: Message to display with progress bar
        
    Returns:
        Generator that yields items from the iterable
    """
    progress_bar = st.progress(0)
    total = len(iterable)
    
    for i, item in enumerate(iterable):
        yield item
        progress_bar.progress((i + 1) / total)
    
    progress_bar.empty()

def display_dataframe(df: pd.DataFrame, max_rows: int = 10, max_cols: int = None):
    """
    Display a dataframe with additional information.
    
    Args:
        df: DataFrame to display
        max_rows: Maximum number of rows to display
        max_cols: Maximum number of columns to display
    """
    if df is None or df.empty:
        st.warning("No data to display.")
        return
    
    # Display shape information
    rows, cols = df.shape
    st.caption(f"Shape: {rows} rows × {cols} columns")
    
    # Display dataframe
    st.dataframe(df.head(max_rows), use_container_width=True)
    
    # Display column info
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (10, 8)):
    """
    Plot correlation matrix for numeric columns.
    
    Args:
        df: DataFrame to analyze
        figsize: Figure size (width, height) in inches
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns to create a correlation matrix.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix")
    st.pyplot(fig)

def plot_feature_importance(importance: pd.Series, title: str = "Feature Importance"):
    """
    Plot feature importance.
    
    Args:
        importance: Series with feature names as index and importance as values
        title: Plot title
    """
    if importance is None or len(importance) == 0:
        st.warning("No feature importance data available.")
        return
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance) * 0.3)))
    importance.sort_values().plot(kind='barh', ax=ax)
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

def download_button(object_to_download, download_filename, button_text):
    """
    Create a download button for any object that can be pickled.
    
    Args:
        object_to_download: Object to download
        download_filename: Filename for the downloaded file
        button_text: Text to display on the button
        
    Returns:
        Download button
    """
    if isinstance(object_to_download, pd.DataFrame):
        # If DataFrame, convert to CSV
        buffer = io.StringIO()
        object_to_download.to_csv(buffer, index=False)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue().encode()).decode()
        file_type = 'text/csv'
    else:
        # Otherwise, pickle the object
        buffer = io.BytesIO()
        pd.to_pickle(object_to_download, buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        file_type = 'application/octet-stream'
    
    # Create download button
    href = f'<a href="data:{file_type};base64,{b64}" download="{download_filename}" class="download-button">{button_text}</a>'
    return st.markdown(href, unsafe_allow_html=True)

def create_tabs(tab_names: List[str]):
    """
    Create a set of tabs.
    
    Args:
        tab_names: List of tab names
        
    Returns:
        List of tab objects
    """
    return st.tabs(tab_names)

def create_expander(title: str, expanded: bool = False):
    """
    Create an expander section.
    
    Args:
        title: Expander title
        expanded: Whether the expander is initially expanded
        
    Returns:
        Expander object
    """
    return st.expander(title, expanded=expanded)
