"""
Data preprocessing page for the ML Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.session import update_session_state, get_session_state
from ui.common import (
    show_header, 
    show_info, 
    show_success, 
    show_warning,
    show_error,
    create_tabs, 
    create_expander,
    display_dataframe
)
from preprocessing.missing_values import (
    handle_missing_values,
    remove_missing_rows,
    remove_missing_columns,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
    fill_missing_constant,
    fill_missing_interpolation,
    fill_missing_knn
)
from preprocessing.outliers import (
    detect_outliers,
    remove_outliers,
    cap_outliers,
    replace_outliers_mean,
    replace_outliers_median
)
from preprocessing.encoding import (
    encode_features,
    one_hot_encode,
    label_encode,
    ordinal_encode,
    target_encode,
    binary_encode,
    frequency_encode
)
from preprocessing.scaling import (
    scale_features,
    standardize,
    minmax_scale,
    robust_scale,
    normalize
)
from data.explorer import analyze_missing_values, analyze_outliers
from plugins.plugin_manager import PluginManager

def render():
    """
    Render the data preprocessing page.
    """
    show_header(
        "Data Preprocessing",
        "Clean and transform your data for modeling."
    )
    
    # Get DataFrame from session state
    df = get_session_state('df')
    
    if df is None:
        show_info("No data available. Please upload a dataset first.")
        return
    
    # Create tabs for different preprocessing steps
    tabs = create_tabs([
        "Overview", 
        "Missing Values", 
        "Outliers", 
        "Encoding", 
        "Scaling",
        "Custom Preprocessing"
    ])
    
    # Overview tab - shows current data and preprocessing steps applied
    with tabs[0]:
        render_overview_tab(df)
    
    # Missing values tab
    with tabs[1]:
        render_missing_values_tab(df)
    
    # Outliers tab
    with tabs[2]:
        render_outliers_tab(df)
    
    # Encoding tab
    with tabs[3]:
        render_encoding_tab(df)
    
    # Scaling tab
    with tabs[4]:
        render_scaling_tab(df)
    
    # Custom preprocessing tab - for plugins and custom functions
    with tabs[5]:
        render_custom_preprocessing_tab(df)
    
    # Let plugins add their own preprocessing tabs
    plugin_manager = PluginManager()
    plugin_manager.execute_hook('render_preprocessing_tabs', df=df)

def render_overview_tab(df):
    """
    Render the overview tab with current data and preprocessing steps.
    """
    st.subheader("Current Dataset")
    display_dataframe(df)
    
    # Show preprocessing steps
    preprocessing_steps = get_session_state('preprocessing_steps', [])
    
    st.subheader("Applied Preprocessing Steps")
    
    if not preprocessing_steps:
        st.info("No preprocessing steps have been applied yet.")
    else:
        for i, step in enumerate(preprocessing_steps):
            with create_expander(f"Step {i+1}: {step['name']}", expanded=False):
                st.write(f"**Type:** {step['type']}")
                st.write(f"**Applied to:** {', '.join(step['columns']) if step['columns'] else 'All suitable columns'}")
                
                # Show parameters
                if 'params' in step and step['params']:
                    st.write("**Parameters:**")
                    for param, value in step['params'].items():
                        st.write(f"- {param}: {value}")
                
                # Show results if available
                if 'results' in step and step['results']:
                    st.write("**Results:**")
                    for key, value in step['results'].items():
                        if isinstance(value, (list, tuple)) and len(value) > 10:
                            st.write(f"- {key}: {value[:10]} (truncated...)")
                        else:
                            st.write(f"- {key}: {value}")
    
    # Revert button
    if preprocessing_steps:
        if st.button("Revert Last Preprocessing Step"):
            # Remove last step
            preprocessing_steps.pop()
            update_session_state('preprocessing_steps', preprocessing_steps)
            
            # Restore data from original and re-apply remaining steps
            original_df = get_session_state('original_df')
            
            if original_df is not None:
                current_df = original_df.copy()
                
                # Re-apply all remaining steps
                for step in preprocessing_steps:
                    current_df = apply_preprocessing_step(current_df, step)
                
                # Update session state
                update_session_state('df', current_df)
                
                show_success("Reverted last preprocessing step.")
                st.experimental_rerun()

def render_missing_values_tab(df):
    """
    Render the missing values handling tab.
    """
    st.subheader("Handle Missing Values")
    
    # Analyze missing values
    missing_cols, missing_rows = analyze_missing_values(df)
    
    # Display columns with missing values
    st.write("**Columns with Missing Values:**")
    
    missing_df = missing_cols[missing_cols['missing_count'] > 0]
    if missing_df.empty:
        st.success("No missing values found in the dataset.")
        return
    else:
        st.dataframe(missing_df, use_container_width=True)
    
    # Method selection
    method = st.selectbox(
        "Select Method",
        options=[
            "Remove Rows",
            "Remove Columns",
            "Fill with Mean",
            "Fill with Median",
            "Fill with Mode",
            "Fill with Constant",
            "Interpolation",
            "KNN Imputation"
        ],
        help="Choose a method to handle missing values"
    )
    
    # Column selection
    columns_with_missing = missing_df['column'].tolist()
    selected_columns = st.multiselect(
        "Select Columns to Process",
        options=columns_with_missing,
        default=columns_with_missing,
        help="Choose which columns to apply the method to"
    )
    
    # Method-specific parameters
    params = {}
    
    if method == "Remove Rows":
        threshold = st.slider(
            "Row Removal Threshold",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Minimum proportion of non-NA values to keep a row (1.0 = remove all rows with any missing value)"
        )
        params['threshold'] = threshold
    
    elif method == "Remove Columns":
        threshold = st.slider(
            "Column Removal Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Maximum proportion of missing values allowed (columns with more will be removed)"
        )
        params['threshold'] = threshold
    
    elif method == "Fill with Constant":
        value_type = st.radio(
            "Value Type",
            options=["Numeric", "Text"],
            horizontal=True
        )
        
        if value_type == "Numeric":
            fill_value = st.number_input("Fill Value", value=0)
        else:
            fill_value = st.text_input("Fill Value", value="missing")
        
        params['value'] = fill_value
    
    elif method == "Interpolation":
        interp_method = st.selectbox(
            "Interpolation Method",
            options=["linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic", "barycentric"],
            index=0
        )
        params['method'] = interp_method
    
    elif method == "KNN Imputation":
        n_neighbors = st.slider(
            "Number of Neighbors",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Number of neighbors to use for imputation"
        )
        params['n_neighbors'] = n_neighbors
    
    # Apply button
    if st.button("Apply Missing Value Handling", type="primary", use_container_width=True):
        if not selected_columns and method not in ["Remove Rows"]:
            show_warning("Please select at least one column to process.")
            return
        
        # Map method name to function name
        method_map = {
            "Remove Rows": "remove_rows",
            "Remove Columns": "remove_columns",
            "Fill with Mean": "fill_mean",
            "Fill with Median": "fill_median",
            "Fill with Mode": "fill_mode",
            "Fill with Constant": "fill_constant",
            "Interpolation": "fill_interpolation",
            "KNN Imputation": "fill_knn"
        }
        
        try:
            # Apply the method
            result_df = handle_missing_values(
                df,
                method=method_map[method],
                columns=selected_columns if selected_columns else None,
                **params
            )
            
            # Record the preprocessing step
            preprocessing_step = {
                'type': 'missing_values',
                'name': method,
                'columns': selected_columns,
                'params': params,
                'method': method_map[method]
            }
            
            # Check if any changes were made
            if df.equals(result_df):
                show_info("No changes were made to the dataset.")
            else:
                # Update session state
                preprocessing_steps = get_session_state('preprocessing_steps', [])
                preprocessing_steps.append(preprocessing_step)
                update_session_state('preprocessing_steps', preprocessing_steps)
                update_session_state('df', result_df)
                
                # Show success message
                show_success(f"Applied {method} to {len(selected_columns) if selected_columns else 'all'} columns.")
                st.experimental_rerun()
                
        except Exception as e:
            show_error(f"Error applying missing value handling: {str(e)}")

def render_outliers_tab(df):
    """
    Render the outliers handling tab.
    """
    st.subheader("Handle Outliers")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns found in the dataset.")
        return
    
    # Column selection
    selected_columns = st.multiselect(
        "Select Columns to Process",
        options=numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
        help="Choose which columns to check for outliers"
    )
    
    if not selected_columns:
        st.info("Please select at least one column to process.")
        return
    
    # Outlier detection method
    detection_method = st.selectbox(
        "Outlier Detection Method",
        options=[
            "Z-Score",
            "IQR Method",
            "Isolation Forest",
            "Local Outlier Factor"
        ],
        help="Choose a method to detect outliers"
    )
    
    # Method-specific parameters
    params = {}
    
    if detection_method == "Z-Score":
        threshold = st.slider(
            "Z-Score Threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Number of standard deviations to use as threshold"
        )
        params['threshold'] = threshold
    
    elif detection_method == "IQR Method":
        threshold = st.slider(
            "IQR Factor",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Multiplier for IQR to determine outlier boundary"
        )
        params['threshold'] = threshold
    
    elif detection_method in ["Isolation Forest", "Local Outlier Factor"]:
        contamination = st.slider(
            "Contamination",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Expected proportion of outliers in the data"
        )
        params['threshold'] = contamination
    
    # Detect outliers button
    if st.button("Detect Outliers"):
        # Map method name to function name
        method_map = {
            "Z-Score": "zscore",
            "IQR Method": "iqr",
            "Isolation Forest": "isolation_forest",
            "Local Outlier Factor": "lof"
        }
        
        try:
            # Detect outliers
            outliers = detect_outliers(
                df,
                columns=selected_columns,
                method=method_map[detection_method].lower(),
                **params
            )
            
            # Display results
            st.write("**Detected Outliers:**")
            
            outlier_counts = {col: len(indices) for col, indices in outliers.items()}
            
            if not outlier_counts:
                st.success("No outliers detected with the current settings.")
            else:
                # Create a summary DataFrame
                summary_data = []
                for col, count in outlier_counts.items():
                    summary_data.append({
                        'Column': col,
                        'Outliers Count': count,
                        'Percentage': f"{(count / len(df)) * 100:.2f}%"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Plot outliers for one column
                if outlier_counts:
                    col_to_plot = st.selectbox(
                        "Select Column to Visualize Outliers",
                        options=list(outlier_counts.keys())
                    )
                    
                    if col_to_plot in outliers:
                        # Create plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot full data
                        ax.scatter(range(len(df)), df[col_to_plot], s=3, label='Normal')
                        
                        # Highlight outliers
                        outlier_indices = outliers[col_to_plot]
                        if len(outlier_indices) > 0:
                            ax.scatter(outlier_indices, df.loc[outlier_indices, col_to_plot], 
                                      color='red', s=20, label='Outliers')
                        
                        ax.set_xlabel('Data Point Index')
                        ax.set_ylabel(col_to_plot)
                        ax.set_title(f'Outliers in {col_to_plot}')
                        ax.legend()
                        
                        st.pyplot(fig)
                        
                        # Show box plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(y=df[col_to_plot], ax=ax)
                        ax.set_title(f'Box Plot of {col_to_plot}')
                        
                        st.pyplot(fig)
            
            # Store outliers in session state for handling
            update_session_state('detected_outliers', outliers)
            update_session_state('outlier_detection_method', detection_method)
            update_session_state('outlier_detection_params', params)
            
        except Exception as e:
            show_error(f"Error detecting outliers: {str(e)}")
    
    # Handling method (only shown if outliers have been detected)
    if get_session_state('detected_outliers'):
        st.subheader("Handle Detected Outliers")
        
        handling_method = st.selectbox(
            "Outlier Handling Method",
            options=[
                "Remove Outliers",
                "Cap Outliers",
                "Replace with Mean",
                "Replace with Median"
            ],
            help="Choose a method to handle the detected outliers"
        )
        
        # Apply outlier handling
        if st.button("Apply Outlier Handling", type="primary", use_container_width=True):
            # Get detected outliers
            outliers = get_session_state('detected_outliers')
            detection_method = get_session_state('outlier_detection_method')
            detection_params = get_session_state('outlier_detection_params')
            
            # Map method name to function
            handling_method_map = {
                "Remove Outliers": remove_outliers,
                "Cap Outliers": cap_outliers,
                "Replace with Mean": replace_outliers_mean,
                "Replace with Median": replace_outliers_median
            }
            
            # Map detection method name to function name
            detection_method_map = {
                "Z-Score": "zscore",
                "IQR Method": "iqr",
                "Isolation Forest": "isolation_forest",
                "Local Outlier Factor": "lof"
            }
            
            try:
                # Apply the handling method
                result_df = handling_method_map[handling_method](
                    df,
                    columns=selected_columns,
                    method=detection_method_map[detection_method].lower(),
                    **detection_params
                )
                
                # Record the preprocessing step
                preprocessing_step = {
                    'type': 'outliers',
                    'name': f"{detection_method} Detection + {handling_method}",
                    'columns': selected_columns,
                    'params': detection_params,
                    'detection_method': detection_method_map[detection_method].lower(),
                    'handling_method': handling_method
                }
                
                # Check if any changes were made
                if df.equals(result_df):
                    show_info("No changes were made to the dataset.")
                else:
                    # Update session state
                    preprocessing_steps = get_session_state('preprocessing_steps', [])
                    preprocessing_steps.append(preprocessing_step)
                    update_session_state('preprocessing_steps', preprocessing_steps)
                    update_session_state('df', result_df)
                    
                    # Clear detected outliers
                    update_session_state('detected_outliers', None)
                    
                    # Show success message
                    show_success(f"Applied {handling_method} to {len(selected_columns)} columns.")
                    st.experimental_rerun()
                
            except Exception as e:
                show_error(f"Error handling outliers: {str(e)}")

def render_encoding_tab(df):
    """
    Render the feature encoding tab.
    """
    st.subheader("Encode Categorical Features")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        st.info("No categorical columns found in the dataset.")
        return
    
    # Column selection
    selected_columns = st.multiselect(
        "Select Columns to Encode",
        options=cat_cols,
        default=cat_cols,
        help="Choose which categorical columns to encode"
    )
    
    if not selected_columns:
        st.info("Please select at least one column to encode.")
        return
    
    # Encoding method
    encoding_method = st.selectbox(
        "Encoding Method",
        options=[
            "One-Hot Encoding",
            "Label Encoding",
            "Ordinal Encoding",
            "Target Encoding",
            "Binary Encoding",
            "Frequency Encoding"
        ],
        help="Choose a method to encode categorical features"
    )
    
    # Method-specific parameters
    params = {}
    
    if encoding_method == "One-Hot Encoding":
        drop_first = st.checkbox(
            "Drop First Category",
            value=False,
            help="Drop the first category to avoid multicollinearity (dummy encoding)"
        )
        params['drop_first'] = drop_first
    
    elif encoding_method == "Ordinal Encoding":
        st.info("Ordinal encoding will assign integers based on alphabetical order of categories. "
               "If you need custom ordering, use a plugin or custom preprocessing.")
    
    elif encoding_method == "Target Encoding":
        target_column = get_session_state('target_column')
        
        if target_column is None:
            st.warning("Target encoding requires a target column. Please set one in the Exploratory Analysis page.")
            return
        
        st.info(f"Will use '{target_column}' as the target for encoding.")
        params['target_column'] = target_column
    
    # Apply button
    if st.button("Apply Encoding", type="primary", use_container_width=True):
        # Map method name to function name
        method_map = {
            "One-Hot Encoding": "onehot",
            "Label Encoding": "label",
            "Ordinal Encoding": "ordinal",
            "Target Encoding": "target",
            "Binary Encoding": "binary",
            "Frequency Encoding": "frequency"
        }
        
        try:
            # Apply encoding
            result_df, encoders = encode_features(
                df,
                method=method_map[encoding_method],
                columns=selected_columns,
                **params
            )
            
            # Record the preprocessing step
            preprocessing_step = {
                'type': 'encoding',
                'name': encoding_method,
                'columns': selected_columns,
                'params': params,
                'method': method_map[encoding_method]
            }
            
            # Check if any changes were made
            if df.equals(result_df):
                show_info("No changes were made to the dataset.")
            else:
                # Update session state
                preprocessing_steps = get_session_state('preprocessing_steps', [])
                preprocessing_steps.append(preprocessing_step)
                update_session_state('preprocessing_steps', preprocessing_steps)
                update_session_state('df', result_df)
                
                # Also store encoders for inference
                applied_preprocessing = get_session_state('applied_preprocessing', {})
                applied_preprocessing['encoders'] = encoders
                update_session_state('applied_preprocessing', applied_preprocessing)
                
                # Show success message
                new_cols = set(result_df.columns) - set(df.columns)
                show_success(f"Applied {encoding_method} to {len(selected_columns)} columns. "
                            f"Added {len(new_cols)} new encoded columns.")
                st.experimental_rerun()
            
        except Exception as e:
            show_error(f"Error applying encoding: {str(e)}")

def render_scaling_tab(df):
    """
    Render the feature scaling tab.
    """
    st.subheader("Scale Numeric Features")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns found in the dataset.")
        return
    
    # Column selection
    selected_columns = st.multiselect(
        "Select Columns to Scale",
        options=numeric_cols,
        default=numeric_cols,
        help="Choose which numeric columns to scale"
    )
    
    if not selected_columns:
        st.info("Please select at least one column to scale.")
        return
    
    # Scaling method
    scaling_method = st.selectbox(
        "Scaling Method",
        options=[
            "Standard Scaler (Z-score)",
            "Min-Max Scaler",
            "Robust Scaler",
            "Normalizer"
        ],
        help="Choose a method to scale numeric features"
    )
    
    # Method-specific parameters
    params = {}
    
    if scaling_method == "Min-Max Scaler":
        min_val = st.number_input("Minimum Value", value=0.0)
        max_val = st.number_input("Maximum Value", value=1.0)
        params['feature_range'] = (min_val, max_val)
    
    elif scaling_method == "Robust Scaler":
        q_min = st.number_input("Lower Quantile", value=25.0, min_value=0.0, max_value=50.0)
        q_max = st.number_input("Upper Quantile", value=75.0, min_value=50.0, max_value=100.0)
        params['quantile_range'] = (q_min, q_max)
    
    elif scaling_method == "Normalizer":
        norm_type = st.selectbox(
            "Norm Type",
            options=["l1", "l2", "max"],
            index=1
        )
        params['norm'] = norm_type
    
    # Apply button
    if st.button("Apply Scaling", type="primary", use_container_width=True):
        # Map method name to function name
        method_map = {
            "Standard Scaler (Z-score)": "standard",
            "Min-Max Scaler": "minmax",
            "Robust Scaler": "robust",
            "Normalizer": "normalizer"
        }
        
        try:
            # Apply scaling
            result_df, scaler = scale_features(
                df,
                method=method_map[scaling_method],
                columns=selected_columns,
                **params
            )
            
            # Record the preprocessing step
            preprocessing_step = {
                'type': 'scaling',
                'name': scaling_method,
                'columns': selected_columns,
                'params': params,
                'method': method_map[scaling_method]
            }
            
            # Check if any changes were made
            if df.equals(result_df):
                show_info("No changes were made to the dataset.")
            else:
                # Update session state
                preprocessing_steps = get_session_state('preprocessing_steps', [])
                preprocessing_steps.append(preprocessing_step)
                update_session_state('preprocessing_steps', preprocessing_steps)
                update_session_state('df', result_df)
                
                # Also store scaler for inference
                applied_preprocessing = get_session_state('applied_preprocessing', {})
                applied_preprocessing['scalers'] = scaler
                update_session_state('applied_preprocessing', applied_preprocessing)
                
                # Show success message
                show_success(f"Applied {scaling_method} to {len(selected_columns)} columns.")
                st.experimental_rerun()
            
        except Exception as e:
            show_error(f"Error applying scaling: {str(e)}")

def render_custom_preprocessing_tab(df):
    """
    Render the custom preprocessing tab for plugins and custom functions.
    """
    st.subheader("Custom Preprocessing")
    
    # Get available preprocessors from plugins
    plugin_manager = PluginManager()
    custom_preprocessors = plugin_manager.execute_hook('get_preprocessors')
    
    # Flatten the list of preprocessors
    all_preprocessors = []
    for preprocessors in custom_preprocessors:
        if isinstance(preprocessors, list):
            all_preprocessors.extend(preprocessors)
    
    if not all_preprocessors:
        st.info("No custom preprocessors available. Install plugins that provide preprocessing functionality.")
        return
    
    # Select preprocessor
    selected_preprocessor = st.selectbox(
        "Select Preprocessor",
        options=[p['name'] for p in all_preprocessors]
    )
    
    # Get the selected preprocessor
    preprocessor = next((p for p in all_preprocessors if p['name'] == selected_preprocessor), None)
    
    if preprocessor:
        # Show description
        if 'description' in preprocessor:
            st.write(preprocessor['description'])
        
        # Let the plugin render UI for its preprocessor
        result = plugin_manager.execute_hook('render_preprocessor_ui', 
                                            preprocessor_name=preprocessor['name'],
                                            df=df)
        
        # Check if the hook returned a DataFrame (preprocessed data)
        for hook_result in result:
            if isinstance(hook_result, tuple) and len(hook_result) == 2:
                result_df, metadata = hook_result
                
                if isinstance(result_df, pd.DataFrame):
                    # Record the preprocessing step
                    preprocessing_step = {
                        'type': 'custom',
                        'name': f"Custom: {preprocessor['name']}",
                        'plugin': preprocessor.get('plugin', 'unknown'),
                        'params': metadata.get('params', {}),
                        'results': metadata.get('results', {})
                    }
                    
                    # Check if any changes were made
                    if df.equals(result_df):
                        show_info("No changes were made to the dataset.")
                    else:
                        # Update session state
                        preprocessing_steps = get_session_state('preprocessing_steps', [])
                        preprocessing_steps.append(preprocessing_step)
                        update_session_state('preprocessing_steps', preprocessing_steps)
                        update_session_state('df', result_df)
                        
                        # Also store metadata for inference
                        if 'artifacts' in metadata:
                            applied_preprocessing = get_session_state('applied_preprocessing', {})
                            if 'custom' not in applied_preprocessing:
                                applied_preprocessing['custom'] = {}
                            
                            applied_preprocessing['custom'][preprocessor['name']] = metadata['artifacts']
                            update_session_state('applied_preprocessing', applied_preprocessing)
                        
                        # Show success message
                        show_success(f"Applied custom preprocessor: {preprocessor['name']}")
                        st.experimental_rerun()

def apply_preprocessing_step(df, step):
    """
    Apply a preprocessing step to a DataFrame.
    
    Args:
        df: DataFrame to process
        step: Preprocessing step dictionary
    
    Returns:
        Processed DataFrame
    """
    try:
        if step['type'] == 'missing_values':
            return handle_missing_values(
                df,
                method=step['method'],
                columns=step['columns'],
                **(step.get('params', {}))
            )
        
        elif step['type'] == 'outliers':
            handling_method_map = {
                "Remove Outliers": remove_outliers,
                "Cap Outliers": cap_outliers,
                "Replace with Mean": replace_outliers_mean,
                "Replace with Median": replace_outliers_median
            }
            
            return handling_method_map[step['handling_method']](
                df,
                columns=step['columns'],
                method=step['detection_method'],
                **(step.get('params', {}))
            )
        
        elif step['type'] == 'encoding':
            result_df, _ = encode_features(
                df,
                method=step['method'],
                columns=step['columns'],
                **(step.get('params', {}))
            )
            return result_df
        
        elif step['type'] == 'scaling':
            result_df, _ = scale_features(
                df,
                method=step['method'],
                columns=step['columns'],
                **(step.get('params', {}))
            )
            return result_df
        
        elif step['type'] == 'custom':
            # Custom steps need to be re-applied through plugins
            plugin_manager = PluginManager()
            result = plugin_manager.execute_hook('apply_preprocessor', 
                                              preprocessor_name=step['name'].replace('Custom: ', ''),
                                              df=df,
                                              params=step.get('params', {}))
            
            # Check if any hook returned a DataFrame
            for hook_result in result:
                if isinstance(hook_result, pd.DataFrame):
                    return hook_result
            
            # If no plugin handled it, return original DataFrame
            return df
        
        else:
            # Unknown step type
            return df
    
    except Exception as e:
        # Log error but continue with original DataFrame
        print(f"Error applying preprocessing step: {str(e)}")
        return df
