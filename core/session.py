import streamlit as st
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SessionState:
    """Class for managing application session state"""
    
    def __init__(self, **kwargs):
        """Initialize session state with optional default values.
        
        Args:
            **kwargs: Initial state values
        """
        # Core application state
        self.current_page = kwargs.get('current_page', 'Project Setup')
        self.data = kwargs.get('data', None)  # Main DataFrame
        self.dataset_name = kwargs.get('dataset_name', None)
        self.model = kwargs.get('model', None)  # Trained model
        self.model_type = kwargs.get('model_type', None)
        self.problem_type = kwargs.get('problem_type', 'Classification')
        
        # Data related state
        self.target_column = kwargs.get('target_column', None)
        self.feature_columns = kwargs.get('feature_columns', [])
        self.categorical_columns = kwargs.get('categorical_columns', [])
        self.numerical_columns = kwargs.get('numerical_columns', [])
        self.datetime_columns = kwargs.get('datetime_columns', [])
        self.text_columns = kwargs.get('text_columns', [])
        
        # UI related state
        self.theme = kwargs.get('theme', 'Light')
        self.custom_theme = kwargs.get('custom_theme', {})
        self.show_dataset_sample = kwargs.get('show_dataset_sample', False)
        self.sidebar_collapsed = kwargs.get('sidebar_collapsed', False)
        self.debug_mode = kwargs.get('debug_mode', False)
        
        # Processing state
        self.use_cache = kwargs.get('use_cache', True)
        self.cache = kwargs.get('cache', {})
        self.temp_storage = kwargs.get('temp_storage', {})
        self.history = kwargs.get('history', [])
        self.export_format = kwargs.get('export_format', 'CSV')
        
        # Project state
        self.project_id = kwargs.get('project_id', str(uuid.uuid4()))
        self.project_name = kwargs.get('project_name', f"ML Project {datetime.now().strftime('%Y-%m-%d')}")
        self.project_description = kwargs.get('project_description', '')
        self.project_path = kwargs.get('project_path', '')
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def update(self, **kwargs):
        """Update session state with new values.
        
        Args:
            **kwargs: State values to update
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def reset(self):
        """Reset session state to defaults."""
        self.__init__()
    
    def save_to_file(self, filepath: str = 'session_state.json'):
        """Save session state to a file.
        
        Args:
            filepath: Path to save the state
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Create a dict of serializable state
            state_dict = {}
            
            for key, value in self.__dict__.items():
                # Skip non-serializable objects
                if key in ['data', 'model']:
                    continue
                
                # Try to serialize the value
                try:
                    json.dumps({key: value})
                    state_dict[key] = value
                except (TypeError, OverflowError):
                    logger.warning(f"Could not serialize {key}, skipping")
            
            # Save to file
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state_dict, f, indent=2)
            
            logger.info(f"Session state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session state: {str(e)}")
            return False
    
    def load_from_file(self, filepath: str = 'session_state.json'):
        """Load session state from a file.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            True if load was successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Session state file not found: {filepath}")
                return False
            
            # Load from file
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
            
            # Update state
            self.update(**state_dict)
            
            logger.info(f"Session state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session state: {str(e)}")
            return False
    
    def set_data(self, data: pd.DataFrame, dataset_name: str, detect_column_types: bool = True):
        """Set the main data and automatically detect column types.
        
        Args:
            data: DataFrame to set
            dataset_name: Name of the dataset
            detect_column_types: Whether to automatically detect column types
        """
        self.data = data
        self.dataset_name = dataset_name
        
        if detect_column_types:
            self._detect_column_types()
    
    def _detect_column_types(self):
        """Automatically detect column types in the data."""
        if self.data is None:
            return
        
        # Reset column lists
        self.numerical_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
        
        # Detect column types
        for col in self.data.columns:
            # Check if datetime
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.datetime_columns.append(col)
            # Check if numeric
            elif pd.api.types.is_numeric_dtype(self.data[col]):
                # Check if it looks like a categorical variable
                if self.data[col].nunique() < 20 and self.data[col].nunique() / len(self.data) < 0.05:
                    self.categorical_columns.append(col)
                else:
                    self.numerical_columns.append(col)
            # Check if categorical
            elif pd.api.types.is_categorical_dtype(self.data[col]) or self.data[col].nunique() < 20:
                self.categorical_columns.append(col)
            # Check if text
            elif self.data[col].dtype == 'object' and self.data[col].str.len().mean() > 20:
                self.text_columns.append(col)
            # Default to categorical
            else:
                self.categorical_columns.append(col)
        
        # Log detection results
        logger.info(f"Detected {len(self.numerical_columns)} numerical columns, "
                   f"{len(self.categorical_columns)} categorical columns, "
                   f"{len(self.datetime_columns)} datetime columns, "
                   f"{len(self.text_columns)} text columns")

def initialize_session():
    """Initialize or get existing session state using Streamlit's session state.
    
    Returns:
        SessionState object
    """
    # Check if session_state already exists in st.session_state
    if 'session_state' not in st.session_state:
        # Create new session state
        st.session_state.session_state = SessionState()
        logger.info("Initialized new session state")
    
    return st.session_state.session_state

def get_session():
    """Get current session state.
    
    Returns:
        SessionState object
    """
    if 'session_state' not in st.session_state:
        return initialize_session()
    return st.session_state.session_state

def save_session_state(filepath: str = 'session_state.json'):
    """Save current session state to file.
    
    Args:
        filepath: Path to save the state
        
    Returns:
        True if save was successful, False otherwise
    """
    session = get_session()
    return session.save_to_file(filepath)

def load_session_state(filepath: str = 'session_state.json'):
    """Load session state from file.
    
    Args:
        filepath: Path to load the state from
        
    Returns:
        True if load was successful, False otherwise
    """
    session = get_session()
    return session.load_from_file(filepath)