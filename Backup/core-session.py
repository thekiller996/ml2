"""
Session state management for the ML Platform.
"""

import streamlit as st
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np

def initialize_session():
    """
    Initialize the session state with default values if not already set.
    """
    if 'initialized' not in st.session_state:
        # Basic navigation state
        st.session_state.current_page = "Project Setup"
        st.session_state.previous_page = None
        
        # Project metadata
        st.session_state.project_name = ""
        st.session_state.project_description = ""
        st.session_state.ml_task = None  # 'classification', 'regression', 'clustering', etc.
        
        # Data state
        st.session_state.df = None
        st.session_state.original_df = None
        st.session_state.file_name = None
        st.session_state.file_path = None
        st.session_state.target_column = None
        st.session_state.id_column = None
        st.session_state.numeric_columns = []
        st.session_state.categorical_columns = []
        st.session_state.datetime_columns = []
        st.session_state.image_columns = []
        st.session_state.text_columns = []
        
        # Feature engineering state
        st.session_state.feature_importance = None
        st.session_state.selected_features = []
        st.session_state.engineered_features = []
        
        # Preprocessing state
        st.session_state.preprocessing_steps = []
        st.session_state.applied_preprocessing = {}
        
        # Model state
        st.session_state.models = {}
        st.session_state.current_model = None
        st.session_state.best_model = None
        st.session_state.model_results = {}
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        
        # Predictions state
        st.session_state.predictions = None
        st.session_state.prediction_df = None
        
        # Plugin state
        st.session_state.plugins = {}
        st.session_state.plugin_data = {}
        
        # Set initialized flag
        st.session_state.initialized = True

def get_session_state(key: str, default: Any = None) -> Any:
    """
    Get a value from the session state with a fallback default.
    
    Args:
        key: Key to retrieve from session state
        default: Default value if key doesn't exist
        
    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)

def update_session_state(key: str, value: Any):
    """
    Update a value in the session state.
    
    Args:
        key: Key to update in session state
        value: New value to set
    """
    st.session_state[key] = value

def navigate_to(page: str):
    """
    Navigate to a different page and update navigation history.
    
    Args:
        page: Page name to navigate to
    """
    st.session_state.previous_page = st.session_state.get('current_page')
    st.session_state.current_page = page
    
def save_dataframe(df: pd.DataFrame, name: str = 'df'):
    """
    Save a dataframe to session state and update column type information.
    
    Args:
        df: DataFrame to save
        name: Name to use in session state (default: 'df')
    """
    st.session_state[name] = df
    
    if name == 'df':
        # Update column type information
        st.session_state.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        st.session_state.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Find datetime columns
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
        st.session_state.datetime_columns = datetime_cols
