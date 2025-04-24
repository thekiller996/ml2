"""
Prediction page for the ML Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from core.session import update_session_state, get_session_state
from ui.common import (
    show_header, 
    show_info, 
    show_success, 
    show_warning,
    show_error,
    create_tabs, 
    display_dataframe
)
from data.loader import load_data
from data.exporter import export_to_csv, export_to_excel, export_to_parquet, export_to_json
from models.classifier import predict_classifier
from models.regressor import predict_regressor
from models.clusterer import predict_clusterer
from plugins.plugin_manager import PluginManager

def render():
    """
    Render the prediction page.
    """
    show_header(
        "Prediction",
        "Make predictions on new data using your trained model."
    )
    
    # Check if model is available
    best_model_name = get_session_state('best_model')
    models = get_session_state('models', {})
    
    if not best_model_name or best_model_name not in models:
        show_warning("No best model selected. Please train and evaluate models first.")
        return
    
    # Get ML task type
    ml_task = get_session_state('ml_task')
    
    # Create tabs for different prediction options
    tabs = create_tabs([
        "Upload Data", 
        "Manual Input", 
        "Batch Prediction", 
        "Save Predictions"
    ])
    
    # Upload data tab
    with tabs[0]:
        render_upload_tab(models, best_model_name, ml_task)
    
    # Manual input tab
    with tabs[1]:
        render_manual_input_tab(models, best_model_name, ml_task)
    
    # Batch prediction tab
    with tabs[2]:
        render_batch_prediction_tab(models, best_model_name, ml_task)
    
    # Save predictions tab
    with tabs[3]:
        render_save_predictions_tab(models, best_model_name, ml_task)
    
    # Let plugins add their own prediction tabs
    plugin_manager = PluginManager()
    plugin_manager.execute_hook('render_prediction_tabs', model=models.get(best_model_name))

def render_upload_tab(models, best_model_name, ml_task):
    """
    Render the upload data tab.
    """
    st.subheader("Upload Prediction Data")
    
    # Get model info
    model_info = models[best_model_name]
    model = model_info.get('model')
    
    # Get required features
    selected_features = get_session_state('selected_features', [])
    
    if not selected_features:
        show_warning("No feature information available. Please check model training setup.")
        return
    
    # Show feature info
    st.write("**Required Features:**")
    st.write(", ".join(selected_features))
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "parquet", "json"],
        help="File should contain all required features"
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
                
                # Check if all required features are present
                missing_features = [feature for feature in selected_features if feature not in df.columns]
                
                if missing_features:
                    show_error(f"Missing required features: {', '.join(missing_features)}")
                else:
                    # Make predictions button
                    if st.button("Make Predictions", type="primary", use_container_width=True):
                        try:
                            # Get features for prediction
                            X = df[selected_features]
                            
                            # Run prediction
                            results_df = make_predictions(model, X, df, ml_task)
                            
                            # Show results
                            st.subheader("Prediction Results")
                            display_dataframe(results_df)
                            
                            # Save to session state for export
                            update_session_state('prediction_df', results_df)
                            
                            # Show success message
                            show_success("Predictions generated successfully!")
                        
                        except Exception as e:
                            show_error(f"Error making predictions: {str(e)}")

def render_manual_input_tab(models, best_model_name, ml_task):
    """
    Render the manual input tab.
    """
    st.subheader("Manual Feature Input")
    
    # Get model info
    model_info = models[best_model_name]
    model = model_info.get('model')
    
    # Get required features
    selected_features = get_session_state('selected_features', [])
    
    if not selected_features:
        show_warning("No feature information available. Please check model training setup.")
        return
    
    # Show feature info
    st.write("**Enter values for each feature:**")
    
    # Get original data for reference
    df = get_session_state('df')
    
    # Create input fields for each feature
    feature_values = {}
    
    for feature in selected_features:
        # Determine input type based on data type
        if df is not None and feature in df.columns:
            col_type = df[feature].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                # For numeric features, get min and max for reference
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                feature_values[feature] = st.number_input(
                    f"{feature} (min: {min_val:.2f}, max: {max_val:.2f}, mean: {mean_val:.2f})",
                    value=mean_val,
                    step=0.01 if col_type == 'float64' else 1
                )
            elif pd.api.types.is_categorical_dtype(col_type) or col_type == 'object':
                # For categorical features, create a dropdown
                categories = df[feature].dropna().unique().tolist()
                feature_values[feature] = st.selectbox(
                    feature,
                    options=categories,
                    index=0
                )
            else:
                # Default to text input for other types
                feature_values[feature] = st.text_input(feature, value="")
        else:
            # If no reference data, default to text input
            feature_values[feature] = st.text_input(feature, value="")
    
    # Make prediction button
    if st.button("Predict", type="primary", use_container_width=True):
        try:
            # Create DataFrame from input values
            input_df = pd.DataFrame([feature_values])
            
            # Make prediction
            if ml_task == "Classification":
                prediction, probabilities = predict_classifier(model, input_df, return_proba=True)
                
                # Display prediction
                st.subheader("Prediction Result")
                st.write(f"**Predicted Class:** {prediction[0]}")
                
                # Display class probabilities
                st.write("**Class Probabilities:**")
                
                # Create probabilities DataFrame
                prob_data = {}
                for i in range(probabilities.shape[1]):
                    prob_data[f"Class {i}"] = f"{probabilities[0, i]:.4f}"
                
                prob_df = pd.DataFrame([prob_data])
                st.dataframe(prob_df, use_container_width=True)
                
                # Create bar chart of probabilities
                fig, ax = plt.subplots(figsize=(10, 5))
                x = [f"Class {i}" for i in range(probabilities.shape[1])]
                ax.bar(x, probabilities[0])
                ax.set_xlabel("Class")
                ax.set_ylabel("Probability")
                ax.set_title("Class Probabilities")
                
                st.pyplot(fig)
                
            elif ml_task == "Regression":
                prediction = predict_regressor(model, input_df)
                
                # Display prediction
                st.subheader("Prediction Result")
                st.metric("Predicted Value", f"{prediction[0]:.4f}")
                
            elif ml_task == "Clustering":
                prediction = predict_clusterer(model, input_df)
                
                # Display prediction
                st.subheader("Prediction Result")
                st.write(f"**Assigned Cluster:** {prediction[0]}")
                
                # If cluster centers are available, show distance to each center
                if hasattr(model, 'cluster_centers_'):
                    centers = model.cluster_centers_
                    
                    # Calculate distances to each cluster center
                    distances = []
                    for i, center in enumerate(centers):
                        dist = np.sqrt(np.sum((input_df.values[0] - center) ** 2))
                        distances.append((i, dist))
                    
                    # Sort by distance
                    distances.sort(key=lambda x: x[1])
                    
                    # Show distances
                    st.write("**Distance to Cluster Centers:**")
                    
                    dist_data = {
                        'Cluster': [d[0] for d in distances],
                        'Distance': [f"{d[1]:.4f}" for d in distances]
                    }
                    
                    dist_df = pd.DataFrame(dist_data)
                    st.dataframe(dist_df, use_container_width=True)
            
            else:
                st.error(f"Unsupported ML task: {ml_task}")
        
        except Exception as e:
            show_error(f"Error making prediction: {str(e)}")

def render_batch_prediction_tab(models, best_model_name, ml_task):
    """
    Render the batch prediction tab.
    """
    st.subheader("Batch Prediction")
    
    # Get original dataframe
    df = get_session_state('df')
    
    if df is None:
        show_warning("No data available. Please upload a dataset first.")
        return
    
    # Get model info
    model_info = models[best_model_name]
    model = model_info.get('model')
    
    # Get required features
    selected_features = get_session_state('selected_features', [])
    
    if not selected_features:
        show_warning("No feature information available. Please check model training setup.")
        return
    
    # Data selection options
    data_option = st.radio(
        "Select Data Source",
        options=["Full Dataset", "Test Set", "Custom Sample"],
        horizontal=True,
        help="Choose which data to use for prediction"
    )
    
    if data_option == "Full Dataset":
        # Use the full dataset
        prediction_df = df.copy()
        st.write(f"Using full dataset: {len(prediction_df)} rows")
    
    elif data_option == "Test Set":
        # Use the test set
        X_test = get_session_state('X_test')
        
        if X_test is None:
            show_warning("Test set not available. Please split the data in the Model Training page.")
            return
        
        # Get the original df rows that correspond to X_test
        # This is approximate and assumes X_test rows came from the original df
        prediction_df = df.loc[X_test.index].copy() if hasattr(X_test, 'index') else df.head(len(X_test))
        
        st.write(f"Using test set: {len(prediction_df)} rows")
    
    else:  # Custom Sample
        # Let user choose sample size
        sample_size = st.slider(
            "Sample Size",
            min_value=1,
            max_value=min(100, len(df)),
            value=min(10, len(df)),
            step=1,
            help="Number of random rows to sample"
        )
        
        # Sample random rows
        prediction_df = df.sample(sample_size, random_state=42).copy()
        
        st.write(f"Using random sample: {len(prediction_df)} rows")
    
    # Display preview
    st.subheader("Data Preview")
    display_dataframe(prediction_df.head(10))
    
    # Make predictions button
    if st.button("Generate Batch Predictions", type="primary", use_container_width=True):
        try:
            # Get features for prediction
            X = prediction_df[selected_features]
            
            # Run prediction
            results_df = make_predictions(model, X, prediction_df, ml_task)
            
            # Show results
            st.subheader("Prediction Results")
            display_dataframe(results_df.head(10))
            
            # Save to session state for export
            update_session_state('prediction_df', results_df)
            
            # Show success message
            show_success(f"Predictions generated for {len(results_df)} rows!")
        
        except Exception as e:
            show_error(f"Error generating batch predictions: {str(e)}")

def render_save_predictions_tab(models, best_model_name, ml_task):
    """
    Render the save predictions tab.
    """
    st.subheader("Save Predictions")
    
    # Get predictions
    prediction_df = get_session_state('prediction_df')
    
    if prediction_df is None:
        show_info("No predictions available yet. Generate predictions first using the other tabs.")
        return
    
    # Show preview
    st.write("**Prediction Results Preview:**")
    display_dataframe(prediction_df.head(10))
    
    # Export format selection
    export_format = st.selectbox(
        "Export Format",
        options=["CSV", "Excel", "Parquet", "JSON"],
        index=0,
        help="Choose the export file format"
    )
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        include_input_features = st.checkbox(
            "Include Input Features",
            value=True,
            help="Include the original features in the export"
        )
    
    with col2:
        if ml_task == "Classification":
            include_probabilities = st.checkbox(
                "Include Class Probabilities",
                value=True,
                help="Include prediction probabilities for each class"
            )
    
    # Download button
    if st.button("Prepare Download", use_container_width=True):
        # Prepare export data
        export_df = prediction_df.copy()
        
        # Filter columns based on options
        if not include_input_features:
            # Keep only prediction columns
            if ml_task == "Classification":
                keep_cols = ['Predicted_Class']
                if 'Probability_Class_' in export_df.columns[0]:
                    prob_cols = [col for col in export_df.columns if 'Probability_Class_' in col]
                    if include_probabilities:
                        keep_cols.extend(prob_cols)
                export_df = export_df[keep_cols]
            elif ml_task == "Regression":
                export_df = export_df[['Predicted_Value']]
            elif ml_task == "Clustering":
                export_df = export_df[['Predicted_Cluster']]
        
        # Export file
        try:
            if export_format == "CSV":
                csv_data = export_to_csv(export_df)
                b64 = base64.b64encode(csv_data.getvalue().encode()).decode()
                href = f'<a href="data:text/csv;base64,{b64}" download="predictions.csv" class="btn">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            elif export_format == "Excel":
                buffer = io.BytesIO()
                export_df.to_excel(buffer, index=False)
                buffer.seek(0)
                b64 = base64.b64encode(buffer.read()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predictions.xlsx" class="btn">Download Excel</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            elif export_format == "Parquet":
                parquet_data = export_to_parquet(export_df)
                b64 = base64.b64encode(parquet_data.getvalue()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="predictions.parquet" class="btn">Download Parquet</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            elif export_format == "JSON":
                json_str = export_to_json(export_df)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="predictions.json" class="btn">Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            st.success("Download ready. Click the link above to download the file.")
        
        except Exception as e:
            show_error(f"Error preparing download: {str(e)}")

def make_predictions(model, X, original_df, ml_task):
    """
    Make predictions using the model and prepare results DataFrame.
    
    Args:
        model: Trained model
        X: Features for prediction
        original_df: Original DataFrame
        ml_task: ML task type
    
    Returns:
        DataFrame with predictions
    """
    # Create results DataFrame
    results_df = original_df.copy()
    
    # Let plugins modify features before prediction
    plugin_manager = PluginManager()
    modified_X = None
    
    for hook_result in plugin_manager.execute_hook('before_prediction', model=model, X=X):
        if isinstance(hook_result, (pd.DataFrame, np.ndarray)):
            modified_X = hook_result
            break
    
    # Use modified features if provided by plugins
    if modified_X is not None:
        X = modified_X
    
    # Make predictions based on task type
    if ml_task == "Classification":
        # Get predictions and probabilities
        predictions, probabilities = predict_classifier(model, X, return_proba=True)
        
        # Add predictions to results
        results_df['Predicted_Class'] = predictions
        
        # Add probabilities if available
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                results_df[f'Probability_Class_{i}'] = probabilities[:, i]
    
    elif ml_task == "Regression":
        # Get predictions
        predictions = predict_regressor(model, X)
        
        # Add predictions to results
        results_df['Predicted_Value'] = predictions
    
    elif ml_task == "Clustering":
        # Get cluster assignments
        clusters = predict_clusterer(model, X)
        
        # Add cluster assignments to results
        results_df['Predicted_Cluster'] = clusters
    
    else:
        raise ValueError(f"Unsupported ML task: {ml_task}")
    
    # Let plugins modify predictions
    modified_results = None
    
    for hook_result in plugin_manager.execute_hook('after_prediction', 
                                                 model=model, 
                                                 predictions=results_df, 
                                                 X=X):
        if isinstance(hook_result, pd.DataFrame):
            modified_results = hook_result
            break
    
    # Use modified results if provided by plugins
    if modified_results is not None:
        results_df = modified_results
    
    return results_df
