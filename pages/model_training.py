"""
Model training page for the ML Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from core.session import update_session_state, get_session_state
from core.constants import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    CLUSTERING_ALGORITHMS,
    CV_METHODS
)
from ui.common import (
    show_header, 
    show_info, 
    show_success, 
    show_warning,
    show_error,
    create_tabs, 
    create_expander,
    plot_feature_importance
)
from models.classifier import (
    get_classifier,
    train_classifier,
    evaluate_classifier,
    predict_classifier
)
from models.regressor import (
    get_regressor,
    train_regressor,
    evaluate_regressor,
    predict_regressor
)
from models.clusterer import (
    get_clusterer,
    train_clusterer,
    evaluate_clusterer,
    predict_clusterer
)
from models.evaluation import (
    split_data,
    cross_validate,
    learning_curve
)
from models.tuning import (
    tune_hyperparameters,
    grid_search,
    random_search,
    bayesian_optimization
)
from plugins.plugin_manager import PluginManager

def render():
    """
    Render the model training page.
    """
    show_header(
        "Model Training",
        "Train and tune machine learning models on your data."
    )
    
    # Get DataFrame from session state
    df = get_session_state('df')
    
    if df is None:
        show_info("No data available. Please upload a dataset first.")
        return
    
    # Check for target column
    target_column = get_session_state('target_column')
    ml_task = get_session_state('ml_task')
    
    if ml_task in ["Classification", "Regression"] and target_column is None:
        show_warning("Target column is not set. Please set a target column in the Exploratory Analysis page.")
        return
    
    # Create tabs for different model training steps
    tabs = create_tabs([
        "Overview", 
        "Data Splitting", 
        "Model Selection", 
        "Training", 
        "Cross-Validation",
        "Hyperparameter Tuning"
    ])
    
    # Overview tab
    with tabs[0]:
        render_overview_tab(df, target_column, ml_task)
    
    # Data splitting tab
    with tabs[1]:
        render_data_splitting_tab(df, target_column, ml_task)
    
    # Model selection tab
    with tabs[2]:
        render_model_selection_tab(df, target_column, ml_task)
    
    # Training tab
    with tabs[3]:
        render_training_tab(df, target_column, ml_task)
    
    # Cross-validation tab
    with tabs[4]:
        render_cross_validation_tab(df, target_column, ml_task)
    
    # Hyperparameter tuning tab
    with tabs[5]:
        render_hyperparameter_tuning_tab(df, target_column, ml_task)
    
    # Let plugins add their own training tabs
    plugin_manager = PluginManager()
    plugin_manager.execute_hook('render_model_training_tabs', df=df, task=ml_task)

def render_overview_tab(df, target_column, ml_task):
    """
    Render the overview tab with data and model information.
    """
    st.subheader("Training Overview")
    
    # Display task type
    st.write(f"**ML Task:** {ml_task or 'Not set'}")
    if target_column:
        st.write(f"**Target Column:** {target_column}")
    
    # Display feature information
    st.subheader("Feature Summary")
    
    # Count columns by type
    n_features = len(df.columns) - (1 if target_column in df.columns else 0)
    st.write(f"**Total Features:** {n_features}")
    
    # Get feature types
    numeric_cols = get_session_state('numeric_columns', [])
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
        
    categorical_cols = get_session_state('categorical_columns', [])
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
        
    datetime_cols = get_session_state('datetime_columns', [])
    if target_column in datetime_cols:
        datetime_cols.remove(target_column)
        
    # Display feature type counts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Numeric Features", len(numeric_cols))
    with col2:
        st.metric("Categorical Features", len(categorical_cols))
    with col3:
        st.metric("Datetime Features", len(datetime_cols))
    
    # Display target distribution
    if target_column:
        st.subheader("Target Distribution")
        
        # Check if classification or regression
        if ml_task == "Classification":
            # Show class distribution
            target_counts = df[target_column].value_counts().reset_index()
            target_counts.columns = ['Class', 'Count']
            target_counts['Percentage'] = (target_counts['Count'] / len(df) * 100).round(2)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x='Class', y='Count', data=target_counts, ax=ax)
                ax.set_title(f"Distribution of {target_column}")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                st.dataframe(target_counts, use_container_width=True)
                
                # Show class balance
                if len(target_counts) > 1:
                    class_balance = target_counts['Count'].min() / target_counts['Count'].max()
                    st.write(f"**Class Balance Ratio:** {class_balance:.3f}")
                    
                    if class_balance < 0.2:
                        st.warning("Significant class imbalance detected. Consider using class weights, resampling, or SMOTE.")
                
        elif ml_task == "Regression":
            # Show histogram of target
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[target_column], ax=ax, kde=True)
            ax.set_title(f"Distribution of {target_column}")
            st.pyplot(fig)
            
            # Show target statistics
            target_stats = df[target_column].describe().to_frame().T
            st.dataframe(target_stats, use_container_width=True)
    
    # Display model status
    st.subheader("Model Status")
    
    # Get trained models
    models = get_session_state('models', {})
    
    if not models:
        st.info("No models have been trained yet. Use the Training tab to train a model.")
    else:
        # Create a model summary table
        model_data = []
        for model_name, model_info in models.items():
            model_data.append({
                'Model': model_name,
                'Algorithm': model_info.get('algorithm', 'Unknown'),
                'Metrics': ', '.join([f"{k}: {v:.4f}" for k, v in model_info.get('metrics', {}).items() if isinstance(v, (int, float))])
            })
        
        model_df = pd.DataFrame(model_data)
        st.dataframe(model_df, use_container_width=True)
        
        # Show best model
        best_model = get_session_state('best_model')
        if best_model:
            st.write(f"**Best Model:** {best_model}")

def render_data_splitting_tab(df, target_column, ml_task):
    """
    Render the data splitting tab.
    """
    st.subheader("Train-Test Split")
    
    # Test size selection
    test_size = st.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Proportion of data to use for testing"
    )
    
    # Random state
    random_state = st.number_input(
        "Random State",
        value=42,
        help="Random seed for reproducibility"
    )
    
    # Stratification for classification
    stratify = True
    if ml_task == "Classification" and target_column:
        stratify = st.checkbox(
            "Stratified Split",
            value=True,
            help="Maintain the same class distribution in train and test sets"
        )
    
    # Column selection (features to use)
    st.subheader("Feature Selection")
    
    # Get all potential feature columns
    all_columns = df.columns.tolist()
    if target_column in all_columns:
        all_columns.remove(target_column)
    
    # Let user select features
    selected_features = st.multiselect(
        "Select Features to Use",
        options=all_columns,
        default=all_columns,
        help="Choose which columns to use as features for the model"
    )
    
    if not selected_features:
        st.warning("Please select at least one feature column.")
        return
    
    # Apply split button
    if st.button("Apply Data Split", type="primary", use_container_width=True):
        try:
            # Extract features and target
            X = df[selected_features]
            
            if ml_task in ["Classification", "Regression"] and target_column:
                y = df[target_column]
                
                # Apply train-test split
                X_train, X_test, y_train, y_test = split_data(
                    X, y, 
                    test_size=test_size, 
                    random_state=random_state, 
                    stratify=stratify
                )
                
                # Save to session state
                update_session_state('X_train', X_train)
                update_session_state('X_test', X_test)
                update_session_state('y_train', y_train)
                update_session_state('y_test', y_test)
                update_session_state('selected_features', selected_features)
                update_session_state('test_size', test_size)
                update_session_state('random_state', random_state)
                update_session_state('stratify', stratify)
                
                # Show split information
                st.success(f"Data split completed: {len(X_train)} training samples, {len(X_test)} test samples")
                
                # Show class distribution for classification
                if ml_task == "Classification":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Training Set Class Distribution**")
                        train_dist = pd.DataFrame(y_train.value_counts()).reset_index()
                        train_dist.columns = ['Class', 'Count']
                        train_dist['Percentage'] = (train_dist['Count'] / len(y_train) * 100).round(2)
                        st.dataframe(train_dist, use_container_width=True)
                    
                    with col2:
                        st.write("**Test Set Class Distribution**")
                        test_dist = pd.DataFrame(y_test.value_counts()).reset_index()
                        test_dist.columns = ['Class', 'Count']
                        test_dist['Percentage'] = (test_dist['Count'] / len(y_test) * 100).round(2)
                        st.dataframe(test_dist, use_container_width=True)
                
            elif ml_task == "Clustering":
                # For clustering, no target is needed
                X_train, X_test = split_data(
                    X, 
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Save to session state
                update_session_state('X_train', X_train)
                update_session_state('X_test', X_test)
                update_session_state('selected_features', selected_features)
                update_session_state('test_size', test_size)
                update_session_state('random_state', random_state)
                
                # Show split information
                st.success(f"Data split completed: {len(X_train)} training samples, {len(X_test)} test samples")
            
            else:
                st.error("Unable to determine ML task type. Please set a task type and target column first.")
                
        except Exception as e:
            show_error(f"Error applying data split: {str(e)}")

def render_model_selection_tab(df, target_column, ml_task):
    """
    Render the model selection tab.
    """
    st.subheader("Model Selection")
    
    # Get model algorithms based on ML task
    if ml_task == "Classification":
        algorithms = list(CLASSIFICATION_ALGORITHMS.keys())
        
        # Get additional algorithms from plugins
        plugin_manager = PluginManager()
        additional_algorithms = plugin_manager.execute_hook('get_additional_models', task='classification')
        
        for plugin_algorithms in additional_algorithms:
            if isinstance(plugin_algorithms, dict):
                for name in plugin_algorithms.keys():
                    if name not in algorithms:
                        algorithms.append(name)
        
    elif ml_task == "Regression":
        algorithms = list(REGRESSION_ALGORITHMS.keys())
        
        # Get additional algorithms from plugins
        plugin_manager = PluginManager()
        additional_algorithms = plugin_manager.execute_hook('get_additional_models', task='regression')
        
        for plugin_algorithms in additional_algorithms:
            if isinstance(plugin_algorithms, dict):
                for name in plugin_algorithms.keys():
                    if name not in algorithms:
                        algorithms.append(name)
        
    elif ml_task == "Clustering":
        algorithms = list(CLUSTERING_ALGORITHMS.keys())
        
        # Get additional algorithms from plugins
        plugin_manager = PluginManager()
        additional_algorithms = plugin_manager.execute_hook('get_additional_models', task='clustering')
        
        for plugin_algorithms in additional_algorithms:
            if isinstance(plugin_algorithms, dict):
                for name in plugin_algorithms.keys():
                    if name not in algorithms:
                        algorithms.append(name)
        
    else:
        algorithms = []
    
    if not algorithms:
        st.warning(f"No algorithms available for {ml_task} task. Please set a valid ML task.")
        return
    
    # Algorithm selection
    selected_algorithm = st.selectbox(
        "Select Algorithm",
        options=algorithms,
        help="Choose a machine learning algorithm to train"
    )
    
    # Get common parameters for the selected algorithm
    st.subheader("Model Parameters")
    
    # Get parameter options from plugins
    plugin_manager = PluginManager()
    param_options = plugin_manager.execute_hook('get_model_params', model_name=selected_algorithm, task=ml_task)
    
    # Initialize parameters dictionary
    params = {}
    
    # Handle different algorithms
    if ml_task == "Classification":
        if selected_algorithm == "Logistic Regression":
            # Logistic Regression parameters
            C = st.number_input("Regularization (C)", min_value=0.001, max_value=10.0, value=1.0, step=0.1)
            params['C'] = C
            
            solver = st.selectbox(
                "Solver",
                options=['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                index=1
            )
            params['solver'] = solver
            
            max_iter = st.number_input("Max Iterations", min_value=100, max_value=10000, value=1000, step=100)
            params['max_iter'] = max_iter
            
            class_weight = st.selectbox(
                "Class Weight",
                options=['None', 'balanced'],
                index=0
            )
            params['class_weight'] = None if class_weight == 'None' else class_weight
            
        elif selected_algorithm == "Decision Tree":
            # Decision Tree parameters
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=None, step=1)
            params['max_depth'] = max_depth if max_depth else None
            
            min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
            params['min_samples_split'] = min_samples_split
            
            min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
            params['min_samples_leaf'] = min_samples_leaf
            
            criterion = st.selectbox(
                "Criterion",
                options=['gini', 'entropy'],
                index=0
            )
            params['criterion'] = criterion
            
        elif selected_algorithm == "Random Forest":
            # Random Forest parameters
            n_estimators = st.number_input("Number of Trees", min_value=10, max_value=1000, value=100, step=10)
            params['n_estimators'] = n_estimators
            
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=None, step=1)
            params['max_depth'] = max_depth if max_depth else None
            
            min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
            params['min_samples_split'] = min_samples_split
            
            criterion = st.selectbox(
                "Criterion",
                options=['gini', 'entropy'],
                index=0
            )
            params['criterion'] = criterion
            
            class_weight = st.selectbox(
                "Class Weight",
                options=['None', 'balanced', 'balanced_subsample'],
                index=0
            )
            params['class_weight'] = None if class_weight == 'None' else class_weight
            
        elif selected_algorithm == "Gradient Boosting":
            # Gradient Boosting parameters
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
            params['n_estimators'] = n_estimators
            
            learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
            params['learning_rate'] = learning_rate
            
            max_depth = st.number_input("Max Depth", min_value=1, max_value=20, value=3, step=1)
            params['max_depth'] = max_depth
            
            subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
            params['subsample'] = subsample
            
        elif selected_algorithm == "XGBoost":
            # XGBoost parameters
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
            params['n_estimators'] = n_estimators
            
            learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
            params['learning_rate'] = learning_rate
            
            max_depth = st.number_input("Max Depth", min_value=1, max_value=20, value=3, step=1)
            params['max_depth'] = max_depth
            
            subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
            params['subsample'] = subsample
            
            colsample_bytree = st.number_input("Column Sample By Tree", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
            params['colsample_bytree'] = colsample_bytree
            
        elif selected_algorithm == "Support Vector Machine":
            # SVM parameters
            C = st.number_input("Regularization (C)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            params['C'] = C
            
            kernel = st.selectbox(
                "Kernel",
                options=['linear', 'poly', 'rbf', 'sigmoid'],
                index=2
            )
            params['kernel'] = kernel
            
            if kernel in ['poly', 'rbf', 'sigmoid']:
                gamma = st.selectbox(
                    "Gamma",
                    options=['scale', 'auto', 'value'],
                    index=0
                )
                
                if gamma == 'value':
                    gamma_val = st.number_input("Gamma Value", min_value=0.001, max_value=10.0, value=0.1, step=0.01)
                    params['gamma'] = gamma_val
                else:
                    params['gamma'] = gamma
                    
            class_weight = st.selectbox(
                "Class Weight",
                options=['None', 'balanced'],
                index=0
            )
            params['class_weight'] = None if class_weight == 'None' else class_weight
            
        elif selected_algorithm == "K-Nearest Neighbors":
            # KNN parameters
            n_neighbors = st.number_input("Number of Neighbors", min_value=1, max_value=50, value=5, step=1)
            params['n_neighbors'] = n_neighbors
            
            weights = st.selectbox(
                "Weight Function",
                options=['uniform', 'distance'],
                index=0
            )
            params['weights'] = weights
            
            metric = st.selectbox(
                "Distance Metric",
                options=['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
                index=0
            )
            params['metric'] = metric
            
        elif selected_algorithm == "Naive Bayes":
            # Naive Bayes parameters
            var_smoothing = st.number_input(
                "Variance Smoothing",
                min_value=1e-12,
                max_value=1.0,
                value=1e-9,
                format="%.2e"
            )
            params['var_smoothing'] = var_smoothing
    
    elif ml_task == "Regression":
        if selected_algorithm == "Linear Regression":
            # Linear Regression parameters
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            params['fit_intercept'] = fit_intercept
            
            normalize = st.checkbox("Normalize", value=False)
            params['normalize'] = normalize
            
        elif selected_algorithm == "Ridge Regression":
            # Ridge parameters
            alpha = st.number_input("Alpha (Regularization)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            params['alpha'] = alpha
            
            solver = st.selectbox(
                "Solver",
                options=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                index=0
            )
            params['solver'] = solver
            
        elif selected_algorithm == "Lasso Regression":
            # Lasso parameters
            alpha = st.number_input("Alpha (Regularization)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            params['alpha'] = alpha
            
            max_iter = st.number_input("Max Iterations", min_value=100, max_value=10000, value=1000, step=100)
            params['max_iter'] = max_iter
            
            selection = st.selectbox(
                "Selection",
                options=['cyclic', 'random'],
                index=0
            )
            params['selection'] = selection
            
        elif selected_algorithm == "Decision Tree":
            # Decision Tree parameters
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=None, step=1)
            params['max_depth'] = max_depth if max_depth else None
            
            min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
            params['min_samples_split'] = min_samples_split
            
            min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
            params['min_samples_leaf'] = min_samples_leaf
            
            criterion = st.selectbox(
                "Criterion",
                options=['mse', 'friedman_mse', 'mae'],
                index=0
            )
            params['criterion'] = criterion
            
        elif selected_algorithm == "Random Forest":
            # Random Forest parameters
            n_estimators = st.number_input("Number of Trees", min_value=10, max_value=1000, value=100, step=10)
            params['n_estimators'] = n_estimators
            
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=None, step=1)
            params['max_depth'] = max_depth if max_depth else None
            
            min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
            params['min_samples_split'] = min_samples_split
            
            criterion = st.selectbox(
                "Criterion",
                options=['mse', 'mae'],
                index=0
            )
            params['criterion'] = criterion
            
        elif selected_algorithm == "Gradient Boosting":
            # Gradient Boosting parameters
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
            params['n_estimators'] = n_estimators
            
            learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
            params['learning_rate'] = learning_rate
            
            max_depth = st.number_input("Max Depth", min_value=1, max_value=20, value=3, step=1)
            params['max_depth'] = max_depth
            
            loss = st.selectbox(
                "Loss Function",
                options=['ls', 'lad', 'huber', 'quantile'],
                index=0
            )
            params['loss'] = loss
            
        elif selected_algorithm == "XGBoost":
            # XGBoost parameters
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
            params['n_estimators'] = n_estimators
            
            learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
            params['learning_rate'] = learning_rate
            
            max_depth = st.number_input("Max Depth", min_value=1, max_value=20, value=3, step=1)
            params['max_depth'] = max_depth
            
            subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
            params['subsample'] = subsample
            
            colsample_bytree = st.number_input("Column Sample By Tree", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
            params['colsample_bytree'] = colsample_bytree
            
        elif selected_algorithm == "Support Vector Regression":
            # SVR parameters
            C = st.number_input("Regularization (C)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            params['C'] = C
            
            kernel = st.selectbox(
                "Kernel",
                options=['linear', 'poly', 'rbf', 'sigmoid'],
                index=2
            )
            params['kernel'] = kernel
            
            if kernel in ['poly', 'rbf', 'sigmoid']:
                gamma = st.selectbox(
                    "Gamma",
                    options=['scale', 'auto', 'value'],
                    index=0
                )
                
                if gamma == 'value':
                    gamma_val = st.number_input("Gamma Value", min_value=0.001, max_value=10.0, value=0.1, step=0.01)
                    params['gamma'] = gamma_val
                else:
                    params['gamma'] = gamma
                    
            epsilon = st.number_input("Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            params['epsilon'] = epsilon
    
    elif ml_task == "Clustering":
        if selected_algorithm == "K-Means":
            # K-means parameters
            n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=20, value=3, step=1)
            params['n_clusters'] = n_clusters
            
            init = st.selectbox(
                "Initialization Method",
                options=['k-means++', 'random'],
                index=0
            )
            params['init'] = init
            
            n_init = st.number_input("Number of Initializations", min_value=1, max_value=50, value=10, step=1)
            params['n_init'] = n_init
            
            max_iter = st.number_input("Maximum Iterations", min_value=10, max_value=1000, value=300, step=10)
            params['max_iter'] = max_iter
            
        elif selected_algorithm == "DBSCAN":
            # DBSCAN parameters
            eps = st.number_input("Epsilon (Neighborhood Size)", min_value=0.01, max_value=10.0, value=0.5, step=0.01)
            params['eps'] = eps
            
            min_samples = st.number_input("Minimum Samples", min_value=1, max_value=100, value=5, step=1)
            params['min_samples'] = min_samples
            
            metric = st.selectbox(
                "Distance Metric",
                options=['euclidean', 'manhattan', 'cosine', 'l1', 'l2'],
                index=0
            )
            params['metric'] = metric
            
        elif selected_algorithm == "Hierarchical Clustering":
            # Hierarchical clustering parameters
            n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=20, value=3, step=1)
            params['n_clusters'] = n_clusters
            
            linkage = st.selectbox(
                "Linkage",
                options=['ward', 'complete', 'average', 'single'],
                index=0
            )
            params['linkage'] = linkage
            
            distance_threshold = st.number_input("Distance Threshold", min_value=None, max_value=None, value=None)
            if distance_threshold:
                params['distance_threshold'] = distance_threshold
                
        elif selected_algorithm == "Mean Shift":
            # Mean Shift parameters
            bandwidth = st.number_input("Bandwidth", min_value=None, max_value=None, value=None)
            if bandwidth:
                params['bandwidth'] = bandwidth
                
            bin_seeding = st.checkbox("Bin Seeding", value=False)
            params['bin_seeding'] = bin_seeding
            
            min_bin_freq = st.number_input("Minimum Bin Frequency", min_value=1, max_value=100, value=1, step=1)
            params['min_bin_freq'] = min_bin_freq
            
        elif selected_algorithm == "Gaussian Mixture":
            # Gaussian Mixture parameters
            n_components = st.number_input("Number of Components", min_value=1, max_value=20, value=3, step=1)
            params['n_components'] = n_components
            
            covariance_type = st.selectbox(
                "Covariance Type",
                options=['full', 'tied', 'diag', 'spherical'],
                index=0
            )
            params['covariance_type'] = covariance_type
            
            max_iter = st.number_input("Maximum Iterations", min_value=10, max_value=1000, value=100, step=10)
            params['max_iter'] = max_iter
            
            init_params = st.selectbox(
                "Initialization Method",
                options=['kmeans', 'random'],
                index=0
            )
            params['init_params'] = init_params
    
    # Add any plugin parameters
    for plugin_params in param_options:
        if isinstance(plugin_params, dict):
            for key, value in plugin_params.items():
                # Only add if not already set
                if key not in params:
                    params[key] = value
    
    # Common parameters
    st.subheader("Common Parameters")
    
    random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility")
    params['random_state'] = random_state
    
    # Save parameters
    if st.button("Save Model Configuration", type="primary", use_container_width=True):
        # Save to session state
        update_session_state('model_algorithm', selected_algorithm)
        update_session_state('model_params', params)
        
        show_success(f"Model configuration for {selected_algorithm} saved successfully!")

def render_training_tab(df, target_column, ml_task):
    """
    Render the model training tab.
    """
    st.subheader("Model Training")
    
    # Check if data is split
    X_train = get_session_state('X_train')
    y_train = get_session_state('y_train')
    
    if X_train is None:
        show_warning("Data has not been split yet. Please use the Data Splitting tab first.")
        return
    
    # Get model configuration
    model_algorithm = get_session_state('model_algorithm')
    model_params = get_session_state('model_params')
    
    if model_algorithm is None or model_params is None:
        show_warning("Model configuration is not set. Please use the Model Selection tab first.")
        return
    
    # Display model configuration
    st.write(f"**Selected Algorithm:** {model_algorithm}")
    
    with create_expander("Model Parameters", expanded=False):
        params_df = pd.DataFrame({
            'Parameter': list(model_params.keys()),
            'Value': list(model_params.values())
        })
        st.dataframe(params_df, use_container_width=True)
    
    # Model name
    model_name = st.text_input(
        "Model Name",
        value=f"{model_algorithm}_{int(time.time())}",
        help="A unique name to identify this model"
    )
    
    # Train button
    if st.button("Train Model", type="primary", use_container_width=True):
        try:
            with st.spinner("Training model..."):
                # Train based on task type
                if ml_task == "Classification":
                    model, metadata = train_classifier(
                        X_train, y_train,
                        algorithm=model_algorithm,
                        **model_params
                    )
                    
                    # Evaluate on training set
                    train_metrics = evaluate_classifier(model, X_train, y_train)
                    
                    # Evaluate on test set
                    X_test = get_session_state('X_test')
                    y_test = get_session_state('y_test')
                    
                    test_metrics = evaluate_classifier(model, X_test, y_test)
                    
                elif ml_task == "Regression":
                    model, metadata = train_regressor(
                        X_train, y_train,
                        algorithm=model_algorithm,
                        **model_params
                    )
                    
                    # Evaluate on training set
                    train_metrics = evaluate_regressor(model, X_train, y_train)
                    
                    # Evaluate on test set
                    X_test = get_session_state('X_test')
                    y_test = get_session_state('y_test')
                    
                    test_metrics = evaluate_regressor(model, X_test, y_test)
                    
                elif ml_task == "Clustering":
                    model, metadata = train_clusterer(
                        X_train,
                        algorithm=model_algorithm,
                        **model_params
                    )
                    
                    # Evaluate on training set
                    train_metrics = evaluate_clusterer(model, X_train)
                    
                    # Evaluate on test set
                    X_test = get_session_state('X_test')
                    
                    test_metrics = evaluate_clusterer(model, X_test)
                    
                else:
                    show_error(f"Unsupported ML task: {ml_task}")
                    return
                
                # Store model and metrics
                models = get_session_state('models', {})
                
                models[model_name] = {
                    'model': model,
                    'algorithm': model_algorithm,
                    'params': model_params,
                    'metadata': metadata,
                    'metrics': test_metrics,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                }
                
                update_session_state('models', models)
                update_session_state('current_model', model_name)
                
                # Update best model if this is the first or better than current best
                best_model = get_session_state('best_model')
                if best_model is None:
                    update_session_state('best_model', model_name)
                else:
                    # Compare key metric based on task
                    if ml_task == "Classification":
                        current_metric = test_metrics.get('accuracy', 0)
                        best_metric = models[best_model]['metrics'].get('accuracy', 0)
                        
                        if current_metric > best_metric:
                            update_session_state('best_model', model_name)
                            
                    elif ml_task == "Regression":
                        current_metric = test_metrics.get('r2', 0)
                        best_metric = models[best_model]['metrics'].get('r2', 0)
                        
                        if current_metric > best_metric:
                            update_session_state('best_model', model_name)
                            
                    elif ml_task == "Clustering":
                        current_metric = test_metrics.get('silhouette', 0)
                        best_metric = models[best_model]['metrics'].get('silhouette', 0)
                        
                        if current_metric > best_metric:
                            update_session_state('best_model', model_name)
                
                # Let plugins handle post-training actions
                plugin_manager = PluginManager()
                plugin_manager.execute_hook('after_model_training', model=model, metadata=metadata)
                
                # Show success message
                show_success(f"Model '{model_name}' trained successfully!")
            
            # Display evaluation metrics
            st.subheader("Evaluation Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Training Set Metrics**")
                train_metrics_df = pd.DataFrame({
                    'Metric': list(train_metrics.keys()),
                    'Value': [f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in train_metrics.values()]
                })
                st.dataframe(train_metrics_df, use_container_width=True)
            
            with col2:
                st.write("**Test Set Metrics**")
                test_metrics_df = pd.DataFrame({
                    'Metric': list(test_metrics.keys()),
                    'Value': [f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in test_metrics.values()]
                })
                st.dataframe(test_metrics_df, use_container_width=True)
            
            # Show feature importance if available
            if 'feature_importance' in metadata:
                st.subheader("Feature Importance")
                
                if isinstance(metadata['feature_importance'], dict):
                    # Convert to series
                    feature_importance = pd.Series(metadata['feature_importance'])
                else:
                    # Create series with feature names
                    feature_names = get_session_state('selected_features')
                    feature_importance = pd.Series(
                        metadata['feature_importance'],
                        index=feature_names[:len(metadata['feature_importance'])]
                    )
                
                # Sort and plot
                feature_importance = feature_importance.sort_values(ascending=False)
                
                # Store in session state
                update_session_state('feature_importance', feature_importance)
                
                # Plot
                plot_feature_importance(feature_importance)
            
        except Exception as e:
            show_error(f"Error training model: {str(e)}")

def render_cross_validation_tab(df, target_column, ml_task):
    """
    Render the cross-validation tab.
    """
    st.subheader("Cross-Validation")
    
    # Check if data is available
    if df is None:
        show_warning("No data available. Please upload a dataset first.")
        return
    
    # Check if model is configured
    model_algorithm = get_session_state('model_algorithm')
    model_params = get_session_state('model_params')
    
    if model_algorithm is None or model_params is None:
        show_warning("Model configuration is not set. Please use the Model Selection tab first.")
        return
    
    # CV parameters
    cv_method = st.selectbox(
        "Cross-Validation Method",
        options=["K-Fold", "Stratified K-Fold", "Time Series Split", "Leave One Out"],
        index=0,
        help="Method to use for cross-validation"
    )
    
    # Number of folds
    if cv_method in ["K-Fold", "Stratified K-Fold", "Time Series Split"]:
        n_splits = st.slider(
            "Number of Splits (Folds)",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Number of folds for cross-validation"
        )
    else:
        n_splits = None
    
    # Shuffle option
    if cv_method in ["K-Fold", "Stratified K-Fold"]:
        shuffle = st.checkbox(
            "Shuffle Data",
            value=True,
            help="Whether to shuffle data before splitting"
        )
    else:
        shuffle = None
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Get all potential feature columns
    all_columns = df.columns.tolist()
    if target_column in all_columns:
        all_columns.remove(target_column)
    
    # Let user select features
    selected_features = st.multiselect(
        "Select Features to Use",
        options=all_columns,
        default=all_columns,
        help="Choose which columns to use as features for the model"
    )
    
    if not selected_features:
        st.warning("Please select at least one feature column.")
        return
    
    # Run cross-validation button
    if st.button("Run Cross-Validation", type="primary", use_container_width=True):
        try:
            with st.spinner("Running cross-validation..."):
                # Extract features and target
                X = df[selected_features]
                
                if ml_task in ["Classification", "Regression"] and target_column:
                    y = df[target_column]
                    
                    # Create model instance
                    if ml_task == "Classification":
                        model = get_classifier(model_algorithm, **model_params)
                    else:  # Regression
                        model = get_regressor(model_algorithm, **model_params)
                    
                    # Create CV parameters
                    cv_params = {}
                    
                    if n_splits:
                        cv_params['n_splits'] = n_splits
                    
                    if shuffle is not None:
                        cv_params['shuffle'] = shuffle
                        
                        if shuffle:
                            cv_params['random_state'] = model_params.get('random_state', 42)
                    
                    # Create CV object
                    if cv_method == "K-Fold":
                        from sklearn.model_selection import KFold
                        cv = KFold(**cv_params)
                    elif cv_method == "Stratified K-Fold":
                        from sklearn.model_selection import StratifiedKFold
                        cv = StratifiedKFold(**cv_params)
                    elif cv_method == "Time Series Split":
                        from sklearn.model_selection import TimeSeriesSplit
                        cv = TimeSeriesSplit(**cv_params)
                    else:  # Leave One Out
                        from sklearn.model_selection import LeaveOneOut
                        cv = LeaveOneOut()
                    
                    # Perform cross-validation
                    cv_results = cross_validate(
                        model, X, y,
                        cv=cv,
                        return_estimator=True
                    )
                    
                    # Store results in session state
                    update_session_state('cv_results', cv_results)
                    
                    # Display CV results
                    st.subheader("Cross-Validation Results")
                    
                    # Get key metrics
                    if ml_task == "Classification":
                        train_scores = cv_results.get('mean_train_accuracy', 0)
                        test_scores = cv_results.get('mean_test_accuracy', 0)
                        metric_name = "Accuracy"
                    else:  # Regression
                        train_scores = cv_results.get('mean_train_r2', 0)
                        test_scores = cv_results.get('mean_test_r2', 0)
                        metric_name = "R² Score"
                    
                    # Create metrics summary
                    metrics_data = []
                    for key, value in cv_results.items():
                        if key.startswith('mean_'):
                            metric_name = key.replace('mean_', '')
                            metrics_data.append({
                                'Metric': metric_name,
                                'Mean': f"{value['mean']:.4f}",
                                'Std Dev': f"{value['std']:.4f}",
                                'Min': f"{min(value['values']):.4f}",
                                'Max': f"{max(value['values']):.4f}"
                            })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Plot scores by fold
                    if 'test_accuracy' in cv_results or 'test_r2' in cv_results:
                        st.subheader("Scores by Fold")
                        
                        if ml_task == "Classification":
                            scores = cv_results.get('test_accuracy', {}).get('values', [])
                            metric = "Accuracy"
                        else:  # Regression
                            scores = cv_results.get('test_r2', {}).get('values', [])
                            metric = "R² Score"
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(range(1, len(scores) + 1), scores, marker='o')
                        ax.set_xlabel('Fold')
                        ax.set_ylabel(metric)
                        ax.set_title(f'{metric} by Fold')
                        ax.grid(True)
                        st.pyplot(fig)
                    
                    # Plot learning curve
                    st.subheader("Learning Curve")
                    
                    with st.spinner("Generating learning curve..."):
                        # Get learning curve data
                        lc_data = learning_curve(
                            model, X, y,
                            cv=cv,
                            scoring='accuracy' if ml_task == 'Classification' else 'r2'
                        )
                        
                        # Plot learning curve
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        train_sizes = lc_data['train_sizes']
                        train_scores_mean = lc_data['train_scores']['mean']
                        train_scores_std = lc_data['train_scores']['std']
                        test_scores_mean = lc_data['test_scores']['mean']
                        test_scores_std = lc_data['test_scores']['std']
                        
                        ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
                        ax.fill_between(train_sizes, 
                                        train_scores_mean - train_scores_std,
                                        train_scores_mean + train_scores_std, 
                                        alpha=0.1, color='r')
                        
                        ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
                        ax.fill_between(train_sizes, 
                                        test_scores_mean - test_scores_std,
                                        test_scores_mean + test_scores_std, 
                                        alpha=0.1, color='g')
                        
                        ax.set_xlabel('Training examples')
                        ax.set_ylabel('Score')
                        ax.set_title('Learning Curve')
                        ax.grid(True)
                        ax.legend(loc='best')
                        
                        st.pyplot(fig)
                    
                else:
                    show_error(f"Cross-validation not supported for {ml_task} task.")
                    return
                
            # Show success message
            show_success(f"Cross-validation completed with {cv_method}!")
            
        except Exception as e:
            show_error(f"Error running cross-validation: {str(e)}")

def render_hyperparameter_tuning_tab(df, target_column, ml_task):
    """
    Render the hyperparameter tuning tab.
    """
    st.subheader("Hyperparameter Tuning")
    
    # Check if data is available
    if df is None:
        show_warning("No data available. Please upload a dataset first.")
        return
    
    # Check if model is configured
    model_algorithm = get_session_state('model_algorithm')
    base_params = get_session_state('model_params')
    
    if model_algorithm is None or base_params is None:
        show_warning("Model configuration is not set. Please use the Model Selection tab first.")
        return
    
    # Tuning method
    tuning_method = st.selectbox(
        "Tuning Method",
        options=["Grid Search", "Random Search", "Bayesian Optimization"],
        index=0,
        help="Method to use for hyperparameter tuning"
    )
    
    # Number of CV folds
    cv_folds = st.slider(
        "Cross-Validation Folds",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Number of cross-validation folds"
    )
    
    # Scoring metric
    if ml_task == "Classification":
        scoring_options = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        default_scoring = "accuracy"
    elif ml_task == "Regression":
        scoring_options = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
        default_scoring = "r2"
    else:
        scoring_options = ["silhouette", "calinski_harabasz", "davies_bouldin"]
        default_scoring = "silhouette"
    
    scoring = st.selectbox(
        "Scoring Metric",
        options=scoring_options,
        index=scoring_options.index(default_scoring) if default_scoring in scoring_options else 0,
        help="Metric to optimize during tuning"
    )
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Get all potential feature columns
    all_columns = df.columns.tolist()
    if target_column in all_columns:
        all_columns.remove(target_column)
    
    # Let user select features
    selected_features = st.multiselect(
        "Select Features to Use",
        options=all_columns,
        default=all_columns,
        help="Choose which columns to use as features for the model"
    )
    
    if not selected_features:
        st.warning("Please select at least one feature column.")
        return
    
    # Parameter grid definition
    st.subheader("Parameter Grid")
    
    # Preset parameter grids based on algorithm
    param_grid = {}
    
    if ml_task == "Classification":
        if model_algorithm == "Logistic Regression":
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs', 'saga'],
                'max_iter': [100, 500, 1000]
            }
        elif model_algorithm == "Decision Tree":
            param_grid = {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        elif model_algorithm == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_algorithm == "Gradient Boosting":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_algorithm == "XGBoost":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif model_algorithm == "Support Vector Machine":
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1.0]
            }
        elif model_algorithm == "K-Nearest Neighbors":
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
            }
    
    elif ml_task == "Regression":
        if model_algorithm == "Linear Regression":
            param_grid = {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        elif model_algorithm == "Ridge Regression":
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
            }
        elif model_algorithm == "Lasso Regression":
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000, 5000],
                'selection': ['cyclic', 'random']
            }
        elif model_algorithm == "Decision Tree":
            param_grid = {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['mse', 'mae']
            }
        elif model_algorithm == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_algorithm == "Gradient Boosting":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'loss': ['ls', 'lad', 'huber']
            }
        elif model_algorithm == "XGBoost":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif model_algorithm == "Support Vector Regression":
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1.0],
                'epsilon': [0.1, 0.2, 0.5]
            }
    
    elif ml_task == "Clustering":
        if model_algorithm == "K-Means":
            param_grid = {
                'n_clusters': [2, 3, 4, 5, 6],
                'init': ['k-means++', 'random'],
                'n_init': [10, 20],
                'max_iter': [200, 300, 500]
            }
        elif model_algorithm == "DBSCAN":
            param_grid = {
                'eps': [0.1, 0.3, 0.5, 0.7, 0.9],
                'min_samples': [3, 5, 10],
                'metric': ['euclidean', 'manhattan']
            }
        elif model_algorithm == "Hierarchical Clustering":
            param_grid = {
                'n_clusters': [2, 3, 4, 5, 6],
                'linkage': ['ward', 'complete', 'average', 'single']
            }
        elif model_algorithm == "Gaussian Mixture":
            param_grid = {
                'n_components': [2, 3, 4, 5, 6],
                'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                'max_iter': [100, 200]
            }
    
    # Display parameter grid
    with create_expander("Edit Parameter Grid", expanded=True):
        # Convert the parameter grid to a string for editing
        param_grid_str = str(param_grid).replace('{', '{\n  ').replace('}', '\n}').replace('], ', '],\n  ')
        edited_param_grid_str = st.text_area("Parameter Grid (Python Dictionary)", value=param_grid_str, height=200)
        
        # Parse the edited parameter grid
        try:
            import ast
            edited_param_grid = ast.literal_eval(edited_param_grid_str)
            if isinstance(edited_param_grid, dict):
                param_grid = edited_param_grid
            else:
                st.error("Invalid parameter grid. Please provide a valid Python dictionary.")
        except:
            st.error("Error parsing parameter grid. Please check the syntax.")
    
    # Number of iterations for random/bayesian search
    if tuning_method in ["Random Search", "Bayesian Optimization"]:
        n_iter = st.slider(
            "Number of Iterations",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of parameter settings to try"
        )
    else:
        n_iter = None
    
    # Run tuning button
    if st.button("Run Hyperparameter Tuning", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Running {tuning_method}..."):
                # Extract features and target
                X = df[selected_features]
                
                if ml_task in ["Classification", "Regression"] and target_column:
                    y = df[target_column]
                    
                    # Create model instance
                    if ml_task == "Classification":
                        model = get_classifier(model_algorithm, **{k: v for k, v in base_params.items() if k not in param_grid})
                    else:  # Regression
                        model = get_regressor(model_algorithm, **{k: v for k, v in base_params.items() if k not in param_grid})
                    
                    # Create tuning params
                    tuning_params = {
                        'cv': cv_folds,
                        'scoring': scoring
                    }
                    
                    if n_iter:
                        tuning_params['n_iter'] = n_iter
                    
                    # Map method name to function name
                    method_map = {
                        "Grid Search": "grid",
                        "Random Search": "random",
                        "Bayesian Optimization": "bayesian"
                    }
                    
                    # Run hyperparameter tuning
                    best_model, results = tune_hyperparameters(
                        model, X, y,
                        param_grid=param_grid,
                        method=method_map[tuning_method],
                        **tuning_params
                    )
                    
                    # Store results in session state
                    update_session_state('tuning_results', results)
                    
                    # Create model name for the best model
                    model_name = f"{model_algorithm}_tuned_{int(time.time())}"
                    
                    # Evaluate on the whole dataset
                    if ml_task == "Classification":
                        metrics = evaluate_classifier(best_model, X, y)
                    else:  # Regression
                        metrics = evaluate_regressor(best_model, X, y)
                    
                    # Store model and metrics
                    models = get_session_state('models', {})
                    
                    models[model_name] = {
                        'model': best_model,
                        'algorithm': model_algorithm,
                        'params': dict(best_model.get_params()),
                        'metrics': metrics,
                        'tuning_method': tuning_method,
                        'tuning_results': results
                    }
                    
                    update_session_state('models', models)
                    update_session_state('current_model', model_name)
                    
                    # Update best model
                    update_session_state('best_model', model_name)
                    
                    # Display tuning results
                    st.subheader("Tuning Results")
                    
                    # Display best parameters
                    st.write("**Best Parameters:**")
                    best_params_df = pd.DataFrame({
                        'Parameter': list(results['best_params'].keys()),
                        'Value': list(results['best_params'].values())
                    })
                    st.dataframe(best_params_df, use_container_width=True)
                    
                    # Display best score
                    st.write(f"**Best Score ({scoring}):** {results['best_score']:.4f}")
                    
                    # Display CV results summary
                    if 'cv_results' in results:
                        cv_results = results['cv_results']
                        
                        # Get top 10 parameter combinations
                        param_combinations = min(10, len(cv_results['params']))
                        
                        top_results = []
                        for i in range(param_combinations):
                            row = {'Rank': cv_results['rank_test_score'][i]}
                            row.update({f"param_{k}": v for k, v in cv_results['params'][i].items()})
                            row['Mean Test Score'] = cv_results['mean_test_score'][i]
                            row['Std Test Score'] = cv_results['std_test_score'][i]
                            top_results.append(row)
                        
                        top_df = pd.DataFrame(top_results)
                        st.write("**Top Parameter Combinations:**")
                        st.dataframe(top_df, use_container_width=True)
                        
                        # Plot the distribution of scores
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(cv_results['mean_test_score'], bins=20)
                        ax.set_xlabel(f'{scoring} Score')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {scoring} Scores')
                        ax.grid(True)
                        st.pyplot(fig)
                    
                    # Show success message
                    show_success(f"Hyperparameter tuning completed with {tuning_method}!")
                    
                else:
                    show_error(f"Hyperparameter tuning not supported for {ml_task} task.")
                    return
                
        except Exception as e:
            show_error(f"Error running hyperparameter tuning: {str(e)}")
