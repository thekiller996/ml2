"""
Data upload page for the ML Platform.
"""

import streamlit as st
import pandas as pd
import io
import os
from core.session import update_session_state, get_session_state
from ui.common import show_header, show_info, show_success, show_error, display_dataframe
from data.loader import load_data, load_sample_data, auto_detect_column_types
from plugins.plugin_manager import PluginManager
import config

def render():
    """
    Render the data upload page.
    """
    show_header(
        "Data Upload",
        "Upload your dataset or use a sample dataset."
    )
    
    # Data source selection
    st.subheader("Data Source")
    
    data_source = st.radio(
        "Select Data Source",
        options=["Upload File", "Sample Dataset", "Previous Data"],
        horizontal=True,
        help="Choose where to load data from"
    )
    
    # Get plugin data sources
    plugin_manager = PluginManager()
    plugin_data_sources = plugin_manager.execute_hook('get_data_sources')
    
    # File uploader
    if data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload your dataset file",
            type=["csv", "xlsx", "parquet", "json", "pkl"],
            help=f"Maximum file size: {config.MAX_UPLOAD_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            # Get file type from extension
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Load data from file
            with st.spinner("Loading data..."):
                df, error = load_data(uploaded_file, file_type)
                
                if error:
                    show_error(f"Error loading data: {error}")
                elif df is not None:
                    # Display preview
                    st.subheader("Data Preview")
                    display_dataframe(df)
                    
                    # Save button
                    if st.button("Use This Dataset", type="primary", use_container_width=True):
                        # Save to session state
                        update_session_state('df', df)
                        update_session_state('original_df', df.copy())
                        update_session_state('file_name', uploaded_file.name)
                        
                        # Auto-detect column types
                        column_types = auto_detect_column_types(df)
                        update_session_state('numeric_columns', column_types['numeric'])
                        update_session_state('categorical_columns', column_types['categorical'])
                        update_session_state('datetime_columns', column_types['datetime'])
                        update_session_state('text_columns', column_types['text'])
                        update_session_state('id_column', column_types['id'][0] if column_types['id'] else None)
                        
                        # Show success message
                        show_success(f"Dataset '{uploaded_file.name}' loaded successfully!")
                        
                        # Navigate to exploratory analysis
                        update_session_state('current_page', "Exploratory Analysis")
                        st.experimental_rerun()
                        
    # Sample dataset
    elif data_source == "Sample Dataset":
        sample_options = ["iris", "boston", "wine", "diabetes", "breast_cancer"]
        
        # Get additional sample datasets from plugins
        plugin_samples = plugin_manager.execute_hook('get_sample_datasets')
        for samples in plugin_samples:
            if samples and isinstance(samples, list):
                sample_options.extend(samples)
        
        sample_dataset = st.selectbox(
            "Select a sample dataset",
            options=sample_options,
            help="Choose from available sample datasets"
        )
        
        if st.button("Load Sample Dataset", type="primary", use_container_width=True):
            with st.spinner("Loading sample data..."):
                try:
                    # Check if this is a plugin dataset
                    plugin_df = None
                    for hook_result in plugin_manager.execute_hook('load_sample_dataset', dataset_name=sample_dataset):
                        if isinstance(hook_result, pd.DataFrame):
                            plugin_df = hook_result
                            break
                    
                    # Load built-in sample or plugin sample
                    if plugin_df is not None:
                        df = plugin_df
                    else:
                        df = load_sample_data(sample_dataset)
                    
                    # Display preview
                    st.subheader("Data Preview")
                    display_dataframe(df)
                    
                    # Save to session state
                    update_session_state('df', df)
                    update_session_state('original_df', df.copy())
                    update_session_state('file_name', f"{sample_dataset}_sample.csv")
                    
                    # Auto-detect column types
                    column_types = auto_detect_column_types(df)
                    update_session_state('numeric_columns', column_types['numeric'])
                    update_session_state('categorical_columns', column_types['categorical'])
                    update_session_state('datetime_columns', column_types['datetime'])
                    update_session_state('text_columns', column_types['text'])
                    update_session_state('id_column', column_types['id'][0] if column_types['id'] else None)
                    
                    # Show success message
                    show_success(f"Sample dataset '{sample_dataset}' loaded successfully!")
                    
                    # Navigate to exploratory analysis
                    update_session_state('current_page', "Exploratory Analysis")
                    st.experimental_rerun()
                except Exception as e:
                    show_error(f"Error loading sample data: {str(e)}")
    
    # Previous data
    elif data_source == "Previous Data":
        if get_session_state('original_df') is not None:
            # Display preview of original data
            st.subheader("Original Data Preview")
            display_dataframe(get_session_state('original_df'))
            
            if st.button("Restore Original Data", type="primary", use_container_width=True):
                # Restore original data
                update_session_state('df', get_session_state('original_df').copy())
                
                # Refresh column types
                df = get_session_state('original_df')
                column_types = auto_detect_column_types(df)
                update_session_state('numeric_columns', column_types['numeric'])
                update_session_state('categorical_columns', column_types['categorical'])
                update_session_state('datetime_columns', column_types['datetime'])
                update_session_state('text_columns', column_types['text'])
                update_session_state('id_column', column_types['id'][0] if column_types['id'] else None)
                
                # Reset preprocessing steps
                update_session_state('preprocessing_steps', [])
                update_session_state('applied_preprocessing', {})
                
                # Show success message
                show_success("Original data restored successfully!")
                
                # Navigate to exploratory analysis
                update_session_state('current_page', "Exploratory Analysis")
                st.experimental_rerun()
        else:
            show_info("No previous data available. Please upload a file or use a sample dataset.")
    
    # Plugin data sources
    else:
        # Let plugins render their data source UI
        for hook_result in plugin_manager.execute_hook('render_data_source_ui', source_name=data_source):
            if isinstance(hook_result, pd.DataFrame):
                # Plugin returned a DataFrame
                df = hook_result
                
                # Display preview
                st.subheader("Data Preview")
                display_dataframe(df)
                
                if st.button("Use This Dataset", type="primary", use_container_width=True):
                    # Save to session state
                    update_session_state('df', df)
                    update_session_state('original_df', df.copy())
                    update_session_state('file_name', f"plugin_data_{data_source}.csv")
                    
                    # Auto-detect column types
                    column_types = auto_detect_column_types(df)
                    update_session_state('numeric_columns', column_types['numeric'])
                    update_session_state('categorical_columns', column_types['categorical'])
                    update_session_state('datetime_columns', column_types['datetime'])
                    update_session_state('text_columns', column_types['text'])
                    update_session_state('id_column', column_types['id'][0] if column_types['id'] else None)
                    
                    # Show success message
                    show_success(f"Dataset from '{data_source}' loaded successfully!")
                    
                    # Navigate to exploratory analysis
                    update_session_state('current_page', "Exploratory Analysis")
                    st.experimental_rerun()
