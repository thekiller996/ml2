"""
Feature engineering page for the ML Platform.
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
    display_dataframe,
    plot_feature_importance
)
from feature_engineering.feature_selection import (
    select_features,
    select_by_variance,
    select_by_correlation,
    select_by_mutual_info,
    select_by_model,
    select_best_k,
    recursive_feature_elimination
)
from feature_engineering.feature_creation import (
    create_polynomial_features,
    create_interaction_features,
    create_binned_features,
    create_datetime_features,
    create_text_features,
    create_pca_features,
    create_cluster_features
)
from feature_engineering.dim_reduction import (
    reduce_dimensions,
    apply_pca,
    apply_tsne,
    apply_umap,
    apply_lda,
    apply_kernel_pca
)
from feature_engineering.feature_transform import (
    apply_transformation,
    apply_math_func,
    create_lag_features,
    create_window_features,
    apply_spectral_transformation
)
from plugins.plugin_manager import PluginManager

def render():
    """
    Render the feature engineering page.
    """
    show_header(
        "Feature Engineering",
        "Create, select, and transform features for your model."
    )
    
    # Get DataFrame from session state
    df = get_session_state('df')
    
    if df is None:
        show_info("No data available. Please upload a dataset first.")
        return
    
    # Create tabs for different feature engineering steps
    tabs = create_tabs([
        "Overview", 
        "Feature Selection", 
        "Feature Creation", 
        "Dimensionality Reduction", 
        "Feature Transformation",
        "Custom Feature Engineering"
    ])
    
    # Overview tab
    with tabs[0]:
        render_overview_tab(df)
    
    # Feature selection tab
    with tabs[1]:
        render_feature_selection_tab(df)
    
    # Feature creation tab
    with tabs[2]:
        render_feature_creation_tab(df)
    
    # Dimensionality reduction tab
    with tabs[3]:
        render_dimensionality_reduction_tab(df)
    
    # Feature transformation tab
    with tabs[4]:
        render_feature_transformation_tab(df)
    
    # Custom feature engineering tab
    with tabs[5]:
        render_custom_feature_engineering_tab(df)
    
    # Let plugins add their own feature engineering tabs
    plugin_manager = PluginManager()
    plugin_manager.execute_hook('render_feature_engineering_tabs', df=df)

def render_overview_tab(df):
    """
    Render the overview tab with current data and feature information.
    """
    st.subheader("Dataset Overview")
    display_dataframe(df)
    
    # Feature statistics
    st.subheader("Feature Statistics")
    
    # Get column types
    numeric_cols = get_session_state('numeric_columns', [])
    categorical_cols = get_session_state('categorical_columns', [])
    datetime_cols = get_session_state('datetime_columns', [])
    
    # Display column type counts
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Features", len(df.columns))
    with col2:
        st.metric("Numeric Features", len(numeric_cols))
    with col3:
        st.metric("Categorical Features", len(categorical_cols))
    with col4:
        st.metric("Datetime Features", len(datetime_cols))
    
    # Feature engineering history
    st.subheader("Applied Feature Engineering Steps")
    
    # Get feature engineering steps
    feature_engineering_steps = get_session_state('feature_engineering_steps', [])
    
    if not feature_engineering_steps:
        st.info("No feature engineering steps have been applied yet.")
    else:
        for i, step in enumerate(feature_engineering_steps):
            with st.expander(f"Step {i+1}: {step['name']}", expanded=False):
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
    if feature_engineering_steps:
        if st.button("Revert Last Feature Engineering Step"):
            # Remove last step
            feature_engineering_steps.pop()
            update_session_state('feature_engineering_steps', feature_engineering_steps)
            
            # Restore data from after preprocessing and re-apply remaining steps
            preprocessed_df = get_session_state('preprocessed_df')
            
            if preprocessed_df is not None:
                current_df = preprocessed_df.copy()
                
                # Re-apply all remaining steps
                for step in feature_engineering_steps:
                    current_df = apply_feature_engineering_step(current_df, step)
                
                # Update session state
                update_session_state('df', current_df)
                
                show_success("Reverted last feature engineering step.")
                st.experimental_rerun()

def render_feature_selection_tab(df):
    """
    Render the feature selection tab.
    """
    st.subheader("Feature Selection")
    
    # Method selection
    method = st.selectbox(
        "Select Method",
        options=[
            "Variance Threshold",
            "Correlation Selection",
            "Mutual Information",
            "Model-Based Selection",
            "K Best Features",
            "Recursive Feature Elimination"
        ],
        help="Choose a method to select relevant features"
    )
    
    # Target column required for some methods
    target_column = get_session_state('target_column')
    target_required = method in [
        "Mutual Information", 
        "Model-Based Selection", 
        "K Best Features", 
        "Recursive Feature Elimination"
    ]
    
    if target_required and target_column is None:
        st.warning("This method requires a target column. Please set one in the Exploratory Analysis page.")
        return
    
    # Column selection
    if method == "Variance Threshold":
        # For variance threshold, only numeric columns are relevant
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # For other methods, allow selection from all columns
        available_cols = df.columns.tolist()
        if target_column in available_cols:
            available_cols.remove(target_column)
    
    if not available_cols:
        st.warning("No suitable columns found for this method.")
        return
    
    selected_columns = st.multiselect(
        "Select Columns",
        options=available_cols,
        default=available_cols,
        help="Choose which columns to consider for feature selection"
    )
    
    if not selected_columns:
        st.info("Please select at least one column.")
        return
    
    # Method-specific parameters
    params = {}
    
    if method == "Variance Threshold":
        threshold = st.slider(
            "Variance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            help="Features with variance below this threshold will be removed"
        )
        params['threshold'] = threshold
    
    elif method == "Correlation Selection":
        threshold = st.slider(
            "Correlation Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Features with correlation above this threshold will be considered redundant"
        )
        params['threshold'] = threshold
        
        if target_column:
            target_threshold = st.slider(
                "Target Correlation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                help="Minimum correlation with target to keep a feature"
            )
            params['target_threshold'] = target_threshold
    
    elif method == "Mutual Information":
        if target_column:
            k = st.slider(
                "Number of Features to Select",
                min_value=1,
                max_value=len(selected_columns),
                value=min(10, len(selected_columns)),
                step=1,
                help="Number of top features to select"
            )
            params['k'] = k
            
            task = get_session_state('ml_task')
            if task in ["Classification", "Regression"]:
                params['task'] = task.lower()
            else:
                params['task'] = 'auto'
    
    elif method == "Model-Based Selection":
        if target_column:
            model_type = st.selectbox(
                "Model Type",
                options=["Random Forest", "Lasso", "XGBoost"],
                help="Model to use for feature importance"
            )
            params['model_type'] = model_type
            
            threshold = st.slider(
                "Importance Threshold",
                min_value=0.001,
                max_value=0.5,
                value=0.01,
                step=0.005,
                help="Feature importance threshold"
            )
            params['threshold'] = threshold
    
    elif method == "K Best Features":
        if target_column:
            k = st.slider(
                "Number of Features to Select",
                min_value=1,
                max_value=len(selected_columns),
                value=min(10, len(selected_columns)),
                step=1,
                help="Number of top features to select"
            )
            params['k'] = k
            
            score_func = st.selectbox(
                "Scoring Function",
                options=["F-test", "Chi2", "Mutual Information"],
                help="Statistical test for feature selection"
            )
            params['score_func'] = score_func
    
    elif method == "Recursive Feature Elimination":
        if target_column:
            n_features = st.slider(
                "Number of Features to Select",
                min_value=1,
                max_value=len(selected_columns),
                value=min(10, len(selected_columns)),
                step=1,
                help="Number of features to select"
            )
            params['n_features_to_select'] = n_features
            
            model_type = st.selectbox(
                "Model Type",
                options=["Logistic Regression", "Linear Regression", "Random Forest"],
                help="Model to use for feature elimination"
            )
            params['model_type'] = model_type
            
            step = st.slider(
                "Step Size",
                min_value=1,
                max_value=max(1, len(selected_columns) // 2),
                value=1,
                step=1,
                help="Number of features to remove at each iteration"
            )
            params['step'] = step
    
    # Apply button
    if st.button("Apply Feature Selection", type="primary", use_container_width=True):
        try:
            # Map method name to function name
            method_map = {
                "Variance Threshold": "variance",
                "Correlation Selection": "correlation",
                "Mutual Information": "mutual_info",
                "Model-Based Selection": "model_based",
                "K Best Features": "k_best",
                "Recursive Feature Elimination": "rfe"
            }
            
            # Add target to params if needed
            if target_required and target_column:
                target = df[target_column]
            else:
                target = None
            
            # Apply feature selection
            result_df, metadata = select_features(
                df,
                method=method_map[method],
                target=target,
                columns=selected_columns,
                **params
            )
            
            # Record the feature engineering step
            feature_engineering_step = {
                'type': 'feature_selection',
                'name': method,
                'columns': selected_columns,
                'params': params,
                'method': method_map[method],
                'results': {
                    'selected_features': metadata.get('selected_features', []),
                    'num_selected': len(metadata.get('selected_features', []))
                }
            }
            
            # Check if any changes were made
            if df.equals(result_df):
                show_info("No changes were made to the dataset.")
            else:
                # Save current data if this is the first feature engineering step
                if not get_session_state('feature_engineering_steps'):
                    update_session_state('preprocessed_df', df.copy())
                
                # Update session state
                feature_engineering_steps = get_session_state('feature_engineering_steps', [])
                feature_engineering_steps.append(feature_engineering_step)
                update_session_state('feature_engineering_steps', feature_engineering_steps)
                update_session_state('df', result_df)
                
                # Store feature importance if available
                if 'feature_importance' in metadata or 'feature_scores' in metadata:
                    feature_importance = metadata.get('feature_importance', metadata.get('feature_scores', {}))
                    update_session_state('feature_importance', feature_importance)
                
                # Show success message
                original_cols = set(df.columns)
                new_cols = set(result_df.columns)
                removed_cols = original_cols - new_cols
                
                show_success(f"Applied {method}. Kept {len(new_cols)} features, removed {len(removed_cols)} features.")
                st.experimental_rerun()
        
        except Exception as e:
            show_error(f"Error applying feature selection: {str(e)}")

def render_feature_creation_tab(df):
    """
    Render the feature creation tab.
    """
    st.subheader("Feature Creation")
    
    # Method selection
    method = st.selectbox(
        "Select Method",
        options=[
            "Polynomial Features",
            "Interaction Features",
            "Binned Features",
            "Datetime Features",
            "Text Features",
            "PCA Features",
            "Cluster Features"
        ],
        help="Choose a method to create new features"
    )
    
    # Column selection based on method
    if method == "Polynomial Features" or method == "Interaction Features":
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        column_type = "numeric"
    elif method == "Binned Features":
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        column_type = "numeric"
    elif method == "Datetime Features":
        available_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                available_cols.append(col)
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                try:
                    # Try to convert to datetime
                    pd.to_datetime(df[col], errors='raise')
                    available_cols.append(col)
                except:
                    pass
        column_type = "datetime"
    elif method == "Text Features":
        available_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        column_type = "text"
    elif method == "PCA Features" or method == "Cluster Features":
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        column_type = "numeric"
    else:
        available_cols = df.columns.tolist()
        column_type = "any"
    
    if not available_cols:
        st.warning(f"No suitable {column_type} columns found for this method.")
        return
    
    selected_columns = st.multiselect(
        "Select Columns",
        options=available_cols,
        default=available_cols[:min(5, len(available_cols))],
        help=f"Choose which {column_type} columns to use for feature creation"
    )
    
    if not selected_columns:
        st.info("Please select at least one column.")
        return
    
    # Method-specific parameters
    params = {}
    
    if method == "Polynomial Features":
        degree = st.slider(
            "Polynomial Degree",
            min_value=2,
            max_value=5,
            value=2,
            step=1,
            help="Degree of polynomial features to create"
        )
        params['degree'] = degree
        
        interaction_only = st.checkbox(
            "Interaction Terms Only",
            value=False,
            help="If checked, only interaction terms will be created (no higher-order terms)"
        )
        params['interaction_only'] = interaction_only
    
    elif method == "Interaction Features":
        if len(selected_columns) < 2:
            st.warning("Please select at least 2 columns for interaction features.")
            return
            
        # Let user specify which pairs to use
        use_all_pairs = st.checkbox(
            "Use All Possible Pairs",
            value=True,
            help="If checked, all possible pairs of selected columns will be used"
        )
        
        if not use_all_pairs:
            # Create a multi-select for each column to select which other columns to pair with
            pairs = []
            for i, col1 in enumerate(selected_columns):
                other_cols = st.multiselect(
                    f"Select columns to pair with {col1}",
                    options=[col for col in selected_columns if col != col1],
                    key=f"pair_{col1}"
                )
                for col2 in other_cols:
                    if (col1, col2) not in pairs and (col2, col1) not in pairs:
                        pairs.append((col1, col2))
            
            if not pairs:
                st.warning("Please select at least one pair of columns.")
                return
                
            params['pairs'] = pairs
    
    elif method == "Binned Features":
        bins = st.slider(
            "Number of Bins",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Number of bins to create"
        )
        params['bins'] = bins
        
        strategy = st.selectbox(
            "Binning Strategy",
            options=["uniform", "quantile", "kmeans"],
            help="Method to determine bin edges"
        )
        params['strategy'] = strategy
    
    elif method == "Datetime Features":
        features = st.multiselect(
            "Features to Extract",
            options=[
                "year", "month", "day", "hour", "minute", "second",
                "weekday", "quarter", "is_weekend", "is_month_start",
                "is_month_end", "dayofyear"
            ],
            default=["year", "month", "day", "weekday", "quarter"],
            help="Datetime components to extract as features"
        )
        params['features'] = features
        
        drop_original = st.checkbox(
            "Drop Original Columns",
            value=False,
            help="If checked, original datetime columns will be removed"
        )
        params['drop_original'] = drop_original
    
    elif method == "Text Features":
        features = st.multiselect(
            "Features to Extract",
            options=[
                "char_count", "word_count", "unique_word_count",
                "word_density", "punctuation_count", "stopword_count",
                "avg_word_length", "sentiment"
            ],
            default=["char_count", "word_count", "unique_word_count"],
            help="Text features to extract"
        )
        params['features'] = features
        
        language = st.selectbox(
            "Language",
            options=["english", "spanish", "french", "german", "italian", "portuguese"],
            index=0,
            help="Language of the text for stopwords and sentiment analysis"
        )
        params['language'] = language
        
        drop_original = st.checkbox(
            "Drop Original Columns",
            value=False,
            help="If checked, original text columns will be removed"
        )
        params['drop_original'] = drop_original
    
    elif method == "PCA Features":
        n_components = st.slider(
            "Number of Components",
            min_value=1,
            max_value=min(len(selected_columns), 20),
            value=min(3, len(selected_columns)),
            step=1,
            help="Number of PCA components to create"
        )
        params['n_components'] = n_components
        
        variance_threshold = st.slider(
            "Variance Threshold",
            min_value=0.5,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Minimum explained variance to keep"
        )
        params['variance_threshold'] = variance_threshold
    
    elif method == "Cluster Features":
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=20,
            value=3,
            step=1,
            help="Number of clusters to create"
        )
        params['n_clusters'] = n_clusters
        
        cluster_method = st.selectbox(
            "Clustering Method",
            options=["kmeans", "dbscan", "hierarchical", "gmm"],
            help="Algorithm to use for clustering"
        )
        params['method'] = cluster_method
        
        if cluster_method == "dbscan":
            eps = st.slider(
                "DBSCAN Epsilon",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Maximum distance between samples for DBSCAN"
            )
            params['eps'] = eps
            
            min_samples = st.slider(
                "DBSCAN Min Samples",
                min_value=2,
                max_value=20,
                value=5,
                step=1,
                help="Minimum number of samples in a neighborhood for DBSCAN"
            )
            params['min_samples'] = min_samples
    
    # Apply button
    if st.button("Create Features", type="primary", use_container_width=True):
        try:
            # Map method name to function
            method_map = {
                "Polynomial Features": create_polynomial_features,
                "Interaction Features": create_interaction_features,
                "Binned Features": create_binned_features,
                "Datetime Features": create_datetime_features,
                "Text Features": create_text_features,
                "PCA Features": create_pca_features,
                "Cluster Features": create_cluster_features
            }
            
            # Apply feature creation
            result_df, metadata = method_map[method](
                df,
                columns=selected_columns,
                **params
            )
            
            # Record the feature engineering step
            feature_engineering_step = {
                'type': 'feature_creation',
                'name': method,
                'columns': selected_columns,
                'params': params,
                'results': {
                    'created_features': metadata.get('created_features', []),
                    'num_created': len(metadata.get('created_features', []))
                }
            }
            
            # Check if any changes were made
            if df.equals(result_df):
                show_info("No new features were created.")
            else:
                # Save current data if this is the first feature engineering step
                if not get_session_state('feature_engineering_steps'):
                    update_session_state('preprocessed_df', df.copy())
                
                # Update session state
                feature_engineering_steps = get_session_state('feature_engineering_steps', [])
                feature_engineering_steps.append(feature_engineering_step)
                update_session_state('feature_engineering_steps', feature_engineering_steps)
                update_session_state('df', result_df)
                
                # Show success message
                created_features = metadata.get('created_features', [])
                show_success(f"Created {len(created_features)} new features using {method}.")
                st.experimental_rerun()
        
        except Exception as e:
            show_error(f"Error creating features: {str(e)}")

def render_dimensionality_reduction_tab(df):
    """
    Render the dimensionality reduction tab.
    """
    st.subheader("Dimensionality Reduction")
    
    # Method selection
    method = st.selectbox(
        "Select Method",
        options=[
            "PCA",
            "t-SNE",
            "UMAP",
            "LDA (supervised)",
            "Kernel PCA"
        ],
        help="Choose a method to reduce dimensionality"
    )
    
    # Check if target is required
    target_required = method == "LDA (supervised)"
    target_column = get_session_state('target_column')
    
    if target_required and target_column is None:
        st.warning("LDA requires a target column. Please set one in the Exploratory Analysis page.")
        return
    
    # Column selection (only numeric columns)
    available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from available columns
    if target_column in available_cols:
        available_cols.remove(target_column)
    
    if not available_cols:
        st.warning("No numeric columns found for dimensionality reduction.")
        return
    
    selected_columns = st.multiselect(
        "Select Columns",
        options=available_cols,
        default=available_cols,
        help="Choose which numeric columns to use for dimensionality reduction"
    )
    
    if len(selected_columns) < 2:
        st.warning("Please select at least 2 columns for dimensionality reduction.")
        return
    
    # Number of components
    n_components = st.slider(
        "Number of Components",
        min_value=1,
        max_value=min(len(selected_columns), 10),
        value=2,
        step=1,
        help="Number of dimensions to reduce to"
    )
    
    # Method-specific parameters
    params = {'n_components': n_components}
    
    if method == "t-SNE":
        perplexity = st.slider(
            "Perplexity",
            min_value=5,
            max_value=50,
            value=30,
            step=5,
            help="Related to the number of nearest neighbors in t-SNE"
        )
        params['perplexity'] = perplexity
        
        learning_rate = st.slider(
            "Learning Rate",
            min_value=10.0,
            max_value=1000.0,
            value=200.0,
            step=10.0,
            help="Learning rate for t-SNE"
        )
        params['learning_rate'] = learning_rate
    
    elif method == "UMAP":
        n_neighbors = st.slider(
            "Number of Neighbors",
            min_value=2,
            max_value=100,
            value=15,
            step=1,
            help="Size of local neighborhood in UMAP"
        )
        params['n_neighbors'] = n_neighbors
        
        min_dist = st.slider(
            "Minimum Distance",
            min_value=0.0,
            max_value=0.99,
            value=0.1,
            step=0.05,
            help="Minimum distance between points in the embedding"
        )
        params['min_dist'] = min_dist
    
    elif method == "Kernel PCA":
        kernel = st.selectbox(
            "Kernel",
            options=["linear", "poly", "rbf", "sigmoid", "cosine"],
            index=2,
            help="Kernel function to use"
        )
        params['kernel'] = kernel
    
    # Apply button
    if st.button("Apply Dimensionality Reduction", type="primary", use_container_width=True):
        try:
            # Map method name to function name
            method_map = {
                "PCA": "pca",
                "t-SNE": "tsne",
                "UMAP": "umap",
                "LDA (supervised)": "lda",
                "Kernel PCA": "kernel_pca"
            }
            
            # Get target if required
            target = df[target_column] if target_required and target_column else None
            
            # Apply dimensionality reduction
            result_df, metadata = reduce_dimensions(
                df,
                method=method_map[method],
                columns=selected_columns,
                n_components=n_components,
                target=target,
                **params
            )
            
            # Record the feature engineering step
            feature_engineering_step = {
                'type': 'dimensionality_reduction',
                'name': method,
                'columns': selected_columns,
                'params': params,
                'method': method_map[method],
                'results': {
                    'reduced_features': metadata.get('reduced_features', []),
                    'num_components': n_components
                }
            }
            
            # Check if any changes were made
            if df.equals(result_df):
                show_info("No changes were made to the dataset.")
            else:
                # Save current data if this is the first feature engineering step
                if not get_session_state('feature_engineering_steps'):
                    update_session_state('preprocessed_df', df.copy())
                
                # Update session state
                feature_engineering_steps = get_session_state('feature_engineering_steps', [])
                feature_engineering_steps.append(feature_engineering_step)
                update_session_state('feature_engineering_steps', feature_engineering_steps)
                update_session_state('df', result_df)
                
                # Show success message
                reduced_features = metadata.get('reduced_features', [])
                show_success(f"Applied {method} and created {len(reduced_features)} new components.")
                st.experimental_rerun()
        
        except Exception as e:
            show_error(f"Error applying dimensionality reduction: {str(e)}")

def render_feature_transformation_tab(df):
    """
    Render the feature transformation tab.
    """
    st.subheader("Feature Transformation")
    
    # Method selection
    method = st.selectbox(
        "Select Method",
        options=[
            "Log Transform",
            "Square Root Transform",
            "Square Transform",
            "Cube Transform",
            "Exponential Transform",
            "Inverse Transform",
            "Lag Features",
            "Window Features",
            "Fourier Transform"
        ],
        help="Choose a method to transform features"
    )
    
    # Column selection based on method
    if method in ["Lag Features", "Window Features"]:
        # These methods require a time series
        time_col = st.selectbox(
            "Time Column",
            options=df.columns.tolist(),
            help="Column representing time order"
        )
        
        # ID column for grouped time series (optional)
        id_columns = ["None"] + df.columns.tolist()
        group_col = st.selectbox(
            "Group Column (optional)",
            options=id_columns,
            help="Column to group time series by (e.g., ID column)"
        )
        if group_col == "None":
            group_col = None
            
        # Value columns (numeric only)
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # For other transforms, use numeric columns
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not available_cols:
        st.warning("No numeric columns found for transformation.")
        return
    
    selected_columns = st.multiselect(
        "Select Columns",
        options=available_cols,
        default=available_cols[:min(3, len(available_cols))],
        help="Choose which numeric columns to transform"
    )
    
    if not selected_columns:
        st.info("Please select at least one column.")
        return
    
    # Method-specific parameters
    params = {}
    
    if method == "Log Transform":
        base = st.selectbox(
            "Log Base",
            options=["e (natural)", "2", "10"],
            help="Base for logarithm"
        )
        params['base'] = 'e' if base == "e (natural)" else int(base)
        
        # Add small constant to avoid log(0)
        epsilon = st.number_input(
            "Epsilon (added to avoid log(0))",
            value=1e-8,
            format="%.8f",
            help="Small constant added to values before taking log"
        )
        params['epsilon'] = epsilon
    
    elif method == "Square Root Transform":
        # No additional parameters needed
        pass
    
    elif method == "Inverse Transform":
        # Add small constant to avoid division by zero
        epsilon = st.number_input(
            "Epsilon (added to avoid division by zero)",
            value=1e-8,
            format="%.8f",
            help="Small constant added to values before taking inverse"
        )
        params['epsilon'] = epsilon
    
    elif method == "Lag Features":
        lags = st.multiselect(
            "Lag Values",
            options=list(range(1, 31)),
            default=[1, 2, 3],
            help="Number of time steps to lag"
        )
        params['lags'] = lags
        params['sort_col'] = time_col
        if group_col:
            params['group_col'] = group_col
    
    elif method == "Window Features":
        window_size = st.slider(
            "Window Size",
            min_value=2,
            max_value=30,
            value=3,
            step=1,
            help="Size of the rolling window"
        )
        params['window_size'] = window_size
        
        functions = st.multiselect(
            "Window Functions",
            options=["mean", "std", "min", "max", "sum", "median", "count", "var"],
            default=["mean", "std"],
            help="Functions to apply to the rolling window"
        )
        params['functions'] = functions
        params['sort_col'] = time_col
        if group_col:
            params['group_col'] = group_col
    
    elif method == "Fourier Transform":
        n_components = st.slider(
            "Number of Components",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Number of frequency components to keep"
        )
        params['n_components'] = n_components
    
    # Apply button
    if st.button("Apply Transformation", type="primary", use_container_width=True):
        try:
            # Map method name to function name
            method_map = {
                "Log Transform": "log",
                "Square Root Transform": "sqrt",
                "Square Transform": "square",
                "Cube Transform": "cube",
                "Exponential Transform": "exp",
                "Inverse Transform": "inverse",
                "Lag Features": "lag",
                "Window Features": "window",
                "Fourier Transform": "fft"
            }
            
            # Apply transformation
            result_df, metadata = apply_transformation(
                df,
                method=method_map[method],
                columns=selected_columns,
                **params
            )
            
            # Record the feature engineering step
            feature_engineering_step = {
                'type': 'feature_transformation',
                'name': method,
                'columns': selected_columns,
                'params': params,
                'method': method_map[method],
                'results': {
                    'transformed_features': metadata.get('transformed_features', []),
                    'num_transformed': len(metadata.get('transformed_features', []))
                }
            }
            
            # Check if any changes were made
            if df.equals(result_df):
                show_info("No changes were made to the dataset.")
            else:
                # Save current data if this is the first feature engineering step
                if not get_session_state('feature_engineering_steps'):
                    update_session_state('preprocessed_df', df.copy())
                
                # Update session state
                feature_engineering_steps = get_session_state('feature_engineering_steps', [])
                feature_engineering_steps.append(feature_engineering_step)
                update_session_state('feature_engineering_steps', feature_engineering_steps)
                update_session_state('df', result_df)
                
                # Show success message
                transformed_features = metadata.get('transformed_features', [])
                show_success(f"Applied {method} to {len(selected_columns)} columns and created {len(transformed_features)} transformed features.")
                st.experimental_rerun()
        
        except Exception as e:
            show_error(f"Error applying transformation: {str(e)}")

def render_custom_feature_engineering_tab(df):
    """
    Render the custom feature engineering tab for plugins.
    """
    st.subheader("Custom Feature Engineering")
    
    # Get available feature engineering methods from plugins
    plugin_manager = PluginManager()
    custom_methods = plugin_manager.execute_hook('get_feature_engineering_methods')
    
    # Flatten the list of methods
    all_methods = []
    for methods in custom_methods:
        if isinstance(methods, list):
            all_methods.extend(methods)
    
    if not all_methods:
        st.info("No custom feature engineering methods available. Install plugins that provide feature engineering functionality.")
        return
    
    # Select method
    selected_method = st.selectbox(
        "Select Method",
        options=[m['name'] for m in all_methods]
    )
    
    # Get the selected method
    method = next((m for m in all_methods if m['name'] == selected_method), None)
    
    if method:
        # Show description
        if 'description' in method:
            st.write(method['description'])
        
        # Let the plugin render UI for its method
        result = plugin_manager.execute_hook('render_feature_engineering_ui', 
                                           method_name=method['name'],
                                           df=df)
        
        # Check if the hook returned a DataFrame (processed data)
        for hook_result in result:
            if isinstance(hook_result, tuple) and len(hook_result) == 2:
                result_df, metadata = hook_result
                
                if isinstance(result_df, pd.DataFrame):
                    # Record the feature engineering step
                    feature_engineering_step = {
                        'type': 'custom',
                        'name': f"Custom: {method['name']}",
                        'plugin': method.get('plugin', 'unknown'),
                        'params': metadata.get('params', {}),
                        'results': metadata.get('results', {})
                    }
                    
                    # Check if any changes were made
                    if df.equals(result_df):
                        show_info("No changes were made to the dataset.")
                    else:
                        # Save current data if this is the first feature engineering step
                        if not get_session_state('feature_engineering_steps'):
                            update_session_state('preprocessed_df', df.copy())
                        
                        # Update session state
                        feature_engineering_steps = get_session_state('feature_engineering_steps', [])
                        feature_engineering_steps.append(feature_engineering_step)
                        update_session_state('feature_engineering_steps', feature_engineering_steps)
                        update_session_state('df', result_df)
                        
                        # Show success message
                        show_success(f"Applied custom feature engineering: {method['name']}")
                        st.experimental_rerun()

def apply_feature_engineering_step(df, step):
    """
    Apply a feature engineering step to a DataFrame.
    
    Args:
        df: DataFrame to process
        step: Feature engineering step dictionary
    
    Returns:
        Processed DataFrame
    """
    try:
        if step['type'] == 'feature_selection':
            result_df, _ = select_features(
                df,
                method=step['method'],
                columns=step['columns'],
                **(step.get('params', {}))
            )
            return result_df
        
        elif step['type'] == 'feature_creation':
            method_map = {
                "Polynomial Features": create_polynomial_features,
                "Interaction Features": create_interaction_features,
                "Binned Features": create_binned_features,
                "Datetime Features": create_datetime_features,
                "Text Features": create_text_features,
                "PCA Features": create_pca_features,
                "Cluster Features": create_cluster_features
            }
            
            result_df, _ = method_map[step['name']](
                df,
                columns=step['columns'],
                **(step.get('params', {}))
            )
            return result_df
        
        elif step['type'] == 'dimensionality_reduction':
            result_df, _ = reduce_dimensions(
                df,
                method=step['method'],
                columns=step['columns'],
                **(step.get('params', {}))
            )
            return result_df
        
        elif step['type'] == 'feature_transformation':
            result_df, _ = apply_transformation(
                df,
                method=step['method'],
                columns=step['columns'],
                **(step.get('params', {}))
            )
            return result_df
        
        elif step['type'] == 'custom':
            # Custom steps need to be re-applied through plugins
            plugin_manager = PluginManager()
            result = plugin_manager.execute_hook('apply_feature_engineering', 
                                              method_name=step['name'].replace('Custom: ', ''),
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
        print(f"Error applying feature engineering step: {str(e)}")
        return df
