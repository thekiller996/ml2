"""
Model evaluation page for the ML Platform.
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
    display_dataframe,
    plot_feature_importance
)
from models.evaluation import (
    confusion_matrix,
    classification_report,
    regression_metrics,
    clustering_metrics,
    roc_curve,
    precision_recall_curve,
    feature_importance
)
from models.classifier import predict_classifier
from models.regressor import predict_regressor
from models.clusterer import predict_clusterer
from plugins.plugin_manager import PluginManager

def render():
    """
    Render the model evaluation page.
    """
    show_header(
        "Model Evaluation",
        "Evaluate and compare model performance."
    )
    
    # Get DataFrame from session state
    df = get_session_state('df')
    
    if df is None:
        show_info("No data available. Please upload a dataset first.")
        return
    
    # Check if models are available
    models = get_session_state('models', {})
    
    if not models:
        show_warning("No models available. Please train models in the Model Training page first.")
        return
    
    # Get ML task and target column
    ml_task = get_session_state('ml_task')
    target_column = get_session_state('target_column')
    
    # Create tabs for different evaluation aspects
    if ml_task == "Classification":
        tabs = create_tabs([
            "Model Comparison", 
            "Confusion Matrix", 
            "Classification Metrics", 
            "ROC Curve", 
            "Prediction Analysis",
            "Feature Importance"
        ])
    elif ml_task == "Regression":
        tabs = create_tabs([
            "Model Comparison", 
            "Regression Metrics", 
            "Residual Analysis", 
            "Prediction vs Actual", 
            "Feature Importance"
        ])
    elif ml_task == "Clustering":
        tabs = create_tabs([
            "Model Comparison", 
            "Clustering Metrics", 
            "Cluster Visualization", 
            "Silhouette Analysis"
        ])
    else:
        tabs = create_tabs(["Model Comparison"])
    
    # Model comparison tab
    with tabs[0]:
        render_model_comparison_tab(models, ml_task)
    
    # Task-specific tabs
    if ml_task == "Classification":
        # Confusion matrix tab
        with tabs[1]:
            render_confusion_matrix_tab(models, df, target_column)
        
        # Classification metrics tab
        with tabs[2]:
            render_classification_metrics_tab(models, df, target_column)
        
        # ROC curve tab
        with tabs[3]:
            render_roc_curve_tab(models, df, target_column)
        
        # Prediction analysis tab
        with tabs[4]:
            render_prediction_analysis_tab(models, df, target_column, ml_task)
        
        # Feature importance tab
        with tabs[5]:
            render_feature_importance_tab(models)
    
    elif ml_task == "Regression":
        # Regression metrics tab
        with tabs[1]:
            render_regression_metrics_tab(models, df, target_column)
        
        # Residual analysis tab
        with tabs[2]:
            render_residual_analysis_tab(models, df, target_column)
        
        # Prediction vs actual tab
        with tabs[3]:
            render_prediction_vs_actual_tab(models, df, target_column)
        
        # Feature importance tab
        with tabs[4]:
            render_feature_importance_tab(models)
    
    elif ml_task == "Clustering":
        # Clustering metrics tab
        with tabs[1]:
            render_clustering_metrics_tab(models, df)
        
        # Cluster visualization tab
        with tabs[2]:
            render_cluster_visualization_tab(models, df)
        
        # Silhouette analysis tab
        with tabs[3]:
            render_silhouette_analysis_tab(models, df)
    
    # Let plugins add their own evaluation tabs
    plugin_manager = PluginManager()
    plugin_manager.execute_hook('render_model_evaluation_tabs', model=models.get(get_session_state('current_model')), df=df)

def render_model_comparison_tab(models, ml_task):
    """
    Render the model comparison tab.
    """
    st.subheader("Model Comparison")
    
    # Create model comparison table
    comparison_data = []
    
    for model_name, model_info in models.items():
        # Get key metrics based on task
        if ml_task == "Classification":
            metrics = {
                'accuracy': model_info.get('metrics', {}).get('accuracy', None),
                'precision': model_info.get('metrics', {}).get('precision', None),
                'recall': model_info.get('metrics', {}).get('recall', None),
                'f1': model_info.get('metrics', {}).get('f1', None),
                'roc_auc': model_info.get('metrics', {}).get('roc_auc', None)
            }
        elif ml_task == "Regression":
            metrics = {
                'r2': model_info.get('metrics', {}).get('r2', None),
                'mse': model_info.get('metrics', {}).get('mse', None),
                'rmse': model_info.get('metrics', {}).get('rmse', None),
                'mae': model_info.get('metrics', {}).get('mae', None),
                'mape': model_info.get('metrics', {}).get('mape', None)
            }
        elif ml_task == "Clustering":
            metrics = {
                'silhouette': model_info.get('metrics', {}).get('silhouette', None),
                'davies_bouldin': model_info.get('metrics', {}).get('davies_bouldin', None),
                'calinski_harabasz': model_info.get('metrics', {}).get('calinski_harabasz', None),
                'n_clusters': model_info.get('metrics', {}).get('n_clusters', None)
            }
        else:
            metrics = {}
        
        # Add model to comparison data
        comparison_data.append({
            'Model': model_name,
            'Algorithm': model_info.get('algorithm', 'Unknown'),
            **{k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for k, v in metrics.items() if v is not None}
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Select current model
    current_model = get_session_state('current_model')
    best_model = get_session_state('best_model')
    
    # Show current model selection
    st.subheader("Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Current Model:** {current_model or 'None'}")
    
    with col2:
        st.write(f"**Best Model:** {best_model or 'None'}")
    
    # Model selector
    selected_model = st.selectbox(
        "Set Current Model",
        options=list(models.keys()),
        index=list(models.keys()).index(current_model) if current_model in models else 0,
        help="Select a model to evaluate in detail"
    )
    
    # Set as current model button
    if st.button("Set as Current Model", type="primary"):
        update_session_state('current_model', selected_model)
        show_success(f"Set '{selected_model}' as current model.")
    
    # Set as best model button
    if st.button("Set as Best Model"):
        update_session_state('best_model', selected_model)
        show_success(f"Set '{selected_model}' as best model.")
    
    # Show model details
    if selected_model in models:
        model_info = models[selected_model]
        
        with create_expander("Model Details", expanded=True):
            st.write(f"**Algorithm:** {model_info.get('algorithm', 'Unknown')}")
            
            # Show parameters
            params = model_info.get('params', {})
            if params:
                st.write("**Parameters:**")
                params_df = pd.DataFrame({
                    'Parameter': list(params.keys()),
                    'Value': [str(v) for v in params.values()]
                })
                st.dataframe(params_df, use_container_width=True)
            
            # Show metrics
            metrics = model_info.get('metrics', {})
            if metrics:
                st.write("**Test Metrics:**")
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': [f"{v:.4f}" if isinstance(v, (float, int)) else str(v) for v in metrics.values()]
                })
                st.dataframe(metrics_df, use_container_width=True)
            
            # Show training metrics if available
            train_metrics = model_info.get('train_metrics', {})
            if train_metrics:
                st.write("**Training Metrics:**")
                train_metrics_df = pd.DataFrame({
                    'Metric': list(train_metrics.keys()),
                    'Value': [f"{v:.4f}" if isinstance(v, (float, int)) else str(v) for v in train_metrics.values()]
                })
                st.dataframe(train_metrics_df, use_container_width=True)

def render_confusion_matrix_tab(models, df, target_column):
    """
    Render the confusion matrix tab.
    """
    st.subheader("Confusion Matrix")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to evaluate on
    dataset = st.radio(
        "Select Dataset",
        options=["Test Set", "Full Dataset"],
        horizontal=True,
        help="Choose which dataset to evaluate the model on"
    )
    
    # Get data based on selection
    if dataset == "Test Set":
        X_test = get_session_state('X_test')
        y_test = get_session_state('y_test')
        
        if X_test is None or y_test is None:
            show_warning("Test set not available. Please split the data in the Model Training page.")
            return
        
        X = X_test
        y_true = y_test
    else:
        # Use full dataset
        X = df[selected_features]
        y_true = df[target_column]
    
    # Make predictions
    try:
        y_pred = predict_classifier(model, X)
        
        # Calculate confusion matrix
        cm_data = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        st.write("**Confusion Matrix**")
        
        # Get class labels
        class_labels = cm_data['labels']
        
        # Create confusion matrix plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        if cm_data['normalized']:
            sns.heatmap(cm_data['matrix'], annot=True, fmt='.2f', cmap='Blues', 
                       xticklabels=class_labels, yticklabels=class_labels)
            ax.set_title('Normalized Confusion Matrix')
        else:
            sns.heatmap(cm_data['matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_labels, yticklabels=class_labels)
            ax.set_title('Confusion Matrix')
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)
        
        # Add option to normalize
        normalize = st.checkbox(
            "Normalize Confusion Matrix",
            value=cm_data['normalized'],
            help="Show proportions instead of counts"
        )
        
        # If normalize changed, recalculate and show
        if normalize != cm_data['normalized']:
            cm_data = confusion_matrix(y_true, y_pred, normalize=normalize)
            
            # Create new plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if normalize:
                sns.heatmap(cm_data['matrix'], annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=class_labels, yticklabels=class_labels)
                ax.set_title('Normalized Confusion Matrix')
            else:
                sns.heatmap(cm_data['matrix'], annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_labels, yticklabels=class_labels)
                ax.set_title('Confusion Matrix')
            
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)
        
        # Analysis of confusion matrix
        st.subheader("Confusion Matrix Analysis")
        
        if len(class_labels) == 2:
            # Binary classification
            ###
            # Binary classification
            tn, fp, fn, tp = cm_data['matrix'].ravel()
            
            metrics = {
                'Accuracy': (tp + tn) / (tp + tn + fp + fn),
                'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Recall (Sensitivity)': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'F1 Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'False Positive Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'False Negative Rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            }
            
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': [f"{v:.4f}" for v in metrics.values()]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Explanation
            with create_expander("Metric Explanations", expanded=False):
                st.markdown("""
                - **Accuracy**: Proportion of correctly classified instances
                - **Precision**: Proportion of positive identifications that were actually correct
                - **Recall (Sensitivity)**: Proportion of actual positives that were correctly identified
                - **Specificity**: Proportion of actual negatives that were correctly identified
                - **F1 Score**: Harmonic mean of precision and recall
                - **False Positive Rate**: Proportion of negatives incorrectly classified as positive
                - **False Negative Rate**: Proportion of positives incorrectly classified as negative
                """)
        else:
            # Multiclass classification
            st.write("Class-specific metrics are available in the Classification Metrics tab.")
    
    except Exception as e:
        show_error(f"Error generating confusion matrix: {str(e)}")

def render_classification_metrics_tab(models, df, target_column):
    """
    Render the classification metrics tab.
    """
    st.subheader("Classification Metrics")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to evaluate on
    dataset = st.radio(
        "Select Dataset",
        options=["Test Set", "Full Dataset"],
        horizontal=True,
        key="class_metrics_dataset",
        help="Choose which dataset to evaluate the model on"
    )
    
    # Get data based on selection
    if dataset == "Test Set":
        X_test = get_session_state('X_test')
        y_test = get_session_state('y_test')
        
        if X_test is None or y_test is None:
            show_warning("Test set not available. Please split the data in the Model Training page.")
            return
        
        X = X_test
        y_true = y_test
    else:
        # Use full dataset
        X = df[selected_features]
        y_true = df[target_column]
    
    # Make predictions
    try:
        y_pred = predict_classifier(model, X)
        
        # Generate classification report
        report = classification_report(y_true, y_pred)
        
        # Convert report to DataFrame for display
        if isinstance(report, dict):
            # Extract per-class metrics
            class_metrics = []
            
            for class_name, metrics in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg', 'samples avg']:
                    continue
                    
                if isinstance(metrics, dict):
                    metrics['class'] = class_name
                    class_metrics.append(metrics)
            
            # Create DataFrame
            if class_metrics:
                class_df = pd.DataFrame(class_metrics)
                if 'class' in class_df.columns:
                    class_df = class_df.set_index('class')
                
                # Format numbers
                for col in class_df.columns:
                    if class_df[col].dtype in [float, np.float64, np.float32]:
                        class_df[col] = class_df[col].round(4)
                
                st.write("**Per-Class Metrics:**")
                st.dataframe(class_df, use_container_width=True)
            
            # Extract overall metrics
            overall_metrics = {}
            
            if 'accuracy' in report:
                overall_metrics['accuracy'] = report['accuracy']
            
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    for metric, value in report[avg_type].items():
                        if metric != 'support':
                            overall_metrics[f"{avg_type} {metric}"] = value
            
            # Create DataFrame
            if overall_metrics:
                overall_df = pd.DataFrame({
                    'Metric': list(overall_metrics.keys()),
                    'Value': [f"{v:.4f}" if isinstance(v, (float, int)) else str(v) for v in overall_metrics.values()]
                })
                
                st.write("**Overall Metrics:**")
                st.dataframe(overall_df, use_container_width=True)
        
        # Metrics visualization
        st.subheader("Metrics Visualization")
        
        # Extract precision, recall, and f1-score for each class
        if isinstance(report, dict):
            classes = []
            precision = []
            recall = []
            f1 = []
            
            for class_name, metrics in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg', 'samples avg']:
                    continue
                    
                if isinstance(metrics, dict):
                    classes.append(class_name)
                    precision.append(metrics.get('precision', 0))
                    recall.append(metrics.get('recall', 0))
                    f1.append(metrics.get('f1-score', 0))
            
            if classes:
                # Plot metrics
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(classes))
                width = 0.25
                
                ax.bar(x - width, precision, width, label='Precision')
                ax.bar(x, recall, width, label='Recall')
                ax.bar(x + width, f1, width, label='F1-Score')
                
                ax.set_xlabel('Class')
                ax.set_ylabel('Score')
                ax.set_title('Precision, Recall, and F1-Score by Class')
                ax.set_xticks(x)
                ax.set_xticklabels(classes, rotation=45, ha='right')
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
    
    except Exception as e:
        show_error(f"Error generating classification metrics: {str(e)}")

def render_roc_curve_tab(models, df, target_column):
    """
    Render the ROC curve tab.
    """
    st.subheader("ROC Curve Analysis")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to evaluate on
    dataset = st.radio(
        "Select Dataset",
        options=["Test Set", "Full Dataset"],
        horizontal=True,
        key="roc_dataset",
        help="Choose which dataset to evaluate the model on"
    )
    
    # Get data based on selection
    if dataset == "Test Set":
        X_test = get_session_state('X_test')
        y_test = get_session_state('y_test')
        
        if X_test is None or y_test is None:
            show_warning("Test set not available. Please split the data in the Model Training page.")
            return
        
        X = X_test
        y_true = y_test
    else:
        # Use full dataset
        X = df[selected_features]
        y_true = df[target_column]
    
    # Make predictions
    try:
        # Get predicted probabilities
        y_pred, y_prob = predict_classifier(model, X, return_proba=True)
        
        # Check if binary or multiclass
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            # Binary classification - use second column of probability matrix
            if y_prob.shape[1] > 1:
                y_score = y_prob[:, 1]
            else:
                y_score = y_prob
                
            # Calculate ROC curve
            roc_data = roc_curve(y_true, y_score)
            
            # Plot ROC curve
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot ROC curve
            ax.plot(roc_data['fpr'], roc_data['tpr'], lw=2, label=f'ROC curve (AUC = {model_info["metrics"].get("roc_auc", 0):.4f})')
            
            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc='lower right')
            
            st.pyplot(fig)
            
            # Show optimal threshold
            # Find threshold that maximizes (TPR - FPR)
            optimal_idx = np.argmax(roc_data['tpr'] - roc_data['fpr'])
            optimal_threshold = roc_data['thresholds'][optimal_idx]
            
            st.write(f"**Optimal Threshold:** {optimal_threshold:.4f}")
            st.write(f"**At optimal threshold:**")
            st.write(f"- True Positive Rate: {roc_data['tpr'][optimal_idx]:.4f}")
            st.write(f"- False Positive Rate: {roc_data['fpr'][optimal_idx]:.4f}")
            
            # Precision-Recall curve
            st.subheader("Precision-Recall Curve")
            
            # Calculate precision-recall curve
            pr_data = precision_recall_curve(y_true, y_score)
            
            # Plot precision-recall curve
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.plot(pr_data['recall'], pr_data['precision'], lw=2)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            
            # Add average precision score if available
            if 'average_precision' in model_info.get('metrics', {}):
                ap = model_info['metrics']['average_precision']
                ax.set_title(f'Precision-Recall Curve (AP = {ap:.4f})')
            
            st.pyplot(fig)
            
        else:
            # Multiclass classification - show ROC curves for each class
            roc_data = roc_curve(y_true, y_prob, multi_class=True)
            
            # Plot ROC curves
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for i, cls in enumerate(roc_data['classes']):
                if str(cls) in roc_data['curves']:
                    curve_data = roc_data['curves'][str(cls)]
                    ax.plot(curve_data['fpr'], curve_data['tpr'], lw=2, 
                           label=f'Class {cls} (AUC = {model_info["metrics"].get(f"roc_auc_class_{cls}", 0):.4f})')
            
            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve - One vs Rest')
            ax.legend(loc='lower right')
            
            st.pyplot(fig)
            
            # Precision-Recall curves for each class
            st.subheader("Precision-Recall Curves")
            
            # Calculate precision-recall curves
            pr_data = precision_recall_curve(y_true, y_prob, multi_class=True)
            
            # Plot precision-recall curves
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for i, cls in enumerate(pr_data['classes']):
                if str(cls) in pr_data['curves']:
                    curve_data = pr_data['curves'][str(cls)]
                    ax.plot(curve_data['recall'], curve_data['precision'], lw=2, 
                           label=f'Class {cls}')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves - One vs Rest')
            ax.legend(loc='lower left')
            
            st.pyplot(fig)
    
    except Exception as e:
        show_error(f"Error generating ROC curve: {str(e)}")

def render_prediction_analysis_tab(models, df, target_column, ml_task):
    """
    Render the prediction analysis tab.
    """
    st.subheader("Prediction Analysis")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to evaluate on
    dataset = st.radio(
        "Select Dataset",
        options=["Test Set", "Full Dataset"],
        horizontal=True,
        key="pred_analysis_dataset",
        help="Choose which dataset to evaluate the model on"
    )
    
    # Get data based on selection
    if dataset == "Test Set":
        X_test = get_session_state('X_test')
        y_test = get_session_state('y_test')
        
        if X_test is None or y_test is None:
            show_warning("Test set not available. Please split the data in the Model Training page.")
            return
        
        X = X_test
        y_true = y_test
    else:
        # Use full dataset
        X = df[selected_features]
        y_true = df[target_column]
    
    # Make predictions
    try:
        if ml_task == "Classification":
            # Get predictions and probabilities
            y_pred, y_prob = predict_classifier(model, X, return_proba=True)
            
            # Create DataFrame with true labels, predictions, and probabilities
            results = pd.DataFrame({
                'True Label': y_true.values,
                'Predicted Label': y_pred
            })
            
            # Add class probabilities
            for i in range(y_prob.shape[1]):
                results[f'Probability Class {i}'] = y_prob[:, i]
            
            # Add correct prediction flag
            results['Correct'] = (results['True Label'] == results['Predicted Label']).astype(int)
            
            # Show results table
            st.write("**Prediction Results (sample):**")
            st.dataframe(results.head(100), use_container_width=True)
            
            # Show error analysis
            st.subheader("Error Analysis")
            
            # Count errors per class
            error_counts = results[results['Correct'] == 0].groupby(['True Label', 'Predicted Label']).size().reset_index()
            error_counts.columns = ['True Label', 'Predicted Label', 'Count']
            error_counts = error_counts.sort_values('Count', ascending=False)
            
            if not error_counts.empty:
                st.write("**Most Common Errors:**")
                st.dataframe(error_counts.head(10), use_container_width=True)
                
                # Plot errors by class
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group by true label and count errors
                class_errors = results[results['Correct'] == 0].groupby('True Label').size()
                class_totals = results.groupby('True Label').size()
                class_error_rates = (class_errors / class_totals).fillna(0)
                
                # Sort by error rate
                class_error_rates = class_error_rates.sort_values(ascending=False)
                
                # Plot error rates
                class_error_rates.plot(kind='bar', ax=ax)
                ax.set_xlabel('True Class')
                ax.set_ylabel('Error Rate')
                ax.set_title('Error Rate by Class')
                plt.xticks(rotation=45, ha='right')
                
                st.pyplot(fig)
                
                # Show examples of misclassified instances
                st.write("**Examples of Misclassified Instances:**")
                
                # Add index for selection
                results['index'] = results.index
                misclassified = results[results['Correct'] == 0]
                
                if not misclassified.empty:
                    # Select classes to examine
                    classes = sorted(results['True Label'].unique())
                    selected_class = st.selectbox(
                        "Select True Class to Examine",
                        options=classes,
                        index=0
                    )
                    
                    # Filter misclassified by selected class
                    class_errors = misclassified[misclassified['True Label'] == selected_class]
                    
                    if not class_errors.empty:
                        # Show examples
                        st.dataframe(class_errors.head(10), use_container_width=True)
                        
                        # If the original DataFrame is available, show feature values
                        if 'index' in class_errors.columns:
                            selected_row = st.selectbox(
                                "Select Row to Examine Features",
                                options=class_errors['index'].tolist(),
                                format_func=lambda x: f"Row {x}"
                            )
                            
                            if selected_row is not None:
                                st.write(f"**Feature Values for Misclassified Instance (Row {selected_row}):**")
                                features_df = X.iloc[selected_row].to_frame().reset_index()
                                features_df.columns = ['Feature', 'Value']
                                st.dataframe(features_df, use_container_width=True)
                    else:
                        st.info(f"No misclassifications found for class {selected_class}.")
            else:
                st.success("No errors found! The model classified all instances correctly.")
        
        elif ml_task == "Regression":
            # Get predictions
            y_pred = predict_regressor(model, X)
            
            # Create DataFrame with true values and predictions
            results = pd.DataFrame({
                'True Value': y_true.values,
                'Predicted Value': y_pred,
                'Error': y_true.values - y_pred,
                'Absolute Error': np.abs(y_true.values - y_pred),
                'Squared Error': (y_true.values - y_pred) ** 2
            })
            
            # Add percentage error (avoiding division by zero)
            non_zero_mask = (y_true.values != 0)
            results['Percentage Error'] = np.nan
            if non_zero_mask.any():
                results.loc[non_zero_mask, 'Percentage Error'] = (results.loc[non_zero_mask, 'Error'] / y_true.values[non_zero_mask]) * 100
            
            # Show results table
            st.write("**Prediction Results (sample):**")
            st.dataframe(results.head(100), use_container_width=True)
            
            # Error distribution
            st.subheader("Error Distribution")
            
            # Plot error histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.histplot(results['Error'], kde=True, ax=ax)
            ax.set_xlabel('Error (True - Predicted)')
            ax.set_ylabel('Frequency')
            ax.set_title('Error Distribution')
            
            st.pyplot(fig)
            
            # Plot absolute error vs true value
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(results['True Value'], results['Absolute Error'], alpha=0.5)
            ax.set_xlabel('True Value')
            ax.set_ylabel('Absolute Error')
            ax.set_title('Absolute Error vs. True Value')
            
            st.pyplot(fig)
            
            # Show error statistics
            st.subheader("Error Statistics")
            
            error_stats = {
                'Mean Error': results['Error'].mean(),
                'Mean Absolute Error': results['Absolute Error'].mean(),
                'Root Mean Squared Error': np.sqrt(results['Squared Error'].mean()),
                'Mean Percentage Error': results['Percentage Error'].mean(),
                'Error Standard Deviation': results['Error'].std(),
                'Min Error': results['Error'].min(),
                'Max Error': results['Error'].max()
            }
            
            error_stats_df = pd.DataFrame({
                'Metric': list(error_stats.keys()),
                'Value': [f"{v:.4f}" if not pd.isna(v) else "N/A" for v in error_stats.values()]
            })
            
            st.dataframe(error_stats_df, use_container_width=True)
            
            # Show worst predictions
            st.subheader("Worst Predictions")
            
            worst_predictions = results.nlargest(10, 'Absolute Error')
            st.dataframe(worst_predictions, use_container_width=True)
            
            # If the original DataFrame is available, show feature values for a selected row
            if not worst_predictions.empty:
                selected_row = st.selectbox(
                    "Select Row to Examine Features",
                    options=worst_predictions.index.tolist(),
                    format_func=lambda x: f"Row {x} (True: {results.loc[x, 'True Value']:.2f}, Pred: {results.loc[x, 'Predicted Value']:.2f}, Error: {results.loc[x, 'Error']:.2f})"
                )
                
                if selected_row is not None:
                    st.write(f"**Feature Values for Selected Instance (Row {selected_row}):**")
                    features_df = X.iloc[selected_row].to_frame().reset_index()
                    features_df.columns = ['Feature', 'Value']
                    st.dataframe(features_df, use_container_width=True)
    
    except Exception as e:
        show_error(f"Error analyzing predictions: {str(e)}")

def render_feature_importance_tab(models):
    """
    Render the feature importance tab.
    """
    st.subheader("Feature Importance Analysis")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Try to get feature importance from:
    # 1. Model metadata
    # 2. Current session state
    # 3. Extract from model
    
    feature_importance = None
    
    # Check model metadata
    if 'metadata' in model_info and 'feature_importance' in model_info['metadata']:
        feature_importance = model_info['metadata']['feature_importance']
    
    # Check session state
    if feature_importance is None:
        feature_importance = get_session_state('feature_importance')
    
    # Extract from model
    if feature_importance is None:
        selected_features = get_session_state('selected_features')
        if selected_features:
            feature_importance = feature_importance(model, selected_features)
    
    if feature_importance is None or not feature_importance:
        st.warning("Feature importance information not available for this model.")
        return
    
    # Convert to Series if it's a dictionary
    if isinstance(feature_importance, dict):
        feature_importance = pd.Series(feature_importance)
    
    # Sort and normalize
    feature_importance = feature_importance.sort_values(ascending=False)
    normalized_importance = feature_importance / feature_importance.sum()
    
    # Display feature importance
    st.write("**Feature Importance:**")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_importance.index,
        'Importance': feature_importance.values,
        'Normalized': normalized_importance.values
    })
    
    # Format values
    importance_df['Importance'] = importance_df['Importance'].apply(lambda x: f"{x:.6f}")
    importance_df['Normalized'] = importance_df['Normalized'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(importance_df, use_container_width=True)
    
    # Plot feature importance
    st.write("**Feature Importance Plot:**")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_importance) * 0.3)))
    
    feature_importance.sort_values().plot(kind='barh', ax=ax)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    
    st.pyplot(fig)
    
    # Pie chart for top features
    st.write("**Top 10 Features Contribution:**")
    
    top_n = min(10, len(feature_importance))
    top_features = feature_importance.iloc[:top_n]
    
    # If there are more features, create an "Other" category
    if len(feature_importance) > top_n:
        other_sum = feature_importance.iloc[top_n:].sum()
        top_features = pd.concat([top_features, pd.Series({'Other': other_sum})])
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.pie(top_features, labels=top_features.index, autopct='%1.1f%%')
    ax.set_title('Top Feature Contribution')
    ax.axis('equal')
    
    st.pyplot(fig)
    
    # Feature importance threshold analysis
    st.subheader("Feature Selection Threshold Analysis")
    
    # Allow user to set importance threshold
    threshold = st.slider(
        "Importance Threshold",
        min_value=0.0,
        max_value=float(feature_importance.max()),
        value=0.01,
        step=0.01,
        help="Minimum importance value to keep a feature"
    )
    
    # Apply threshold
    selected_features = feature_importance[feature_importance >= threshold].index.tolist()
    
    st.write(f"**Features Selected ({len(selected_features)}/{len(feature_importance)}):**")
    st.write(", ".join(selected_features))
    
    # Show accumulated importance
    selected_importance = feature_importance[feature_importance >= threshold].sum()
    total_importance = feature_importance.sum()
    
    st.write(f"**Cumulative Importance:** {selected_importance / total_importance:.2%}")

def render_regression_metrics_tab(models, df, target_column):
    """
    Render the regression metrics tab.
    """
    st.subheader("Regression Metrics")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to evaluate on
    dataset = st.radio(
        "Select Dataset",
        options=["Test Set", "Full Dataset"],
        horizontal=True,
        key="reg_metrics_dataset",
        help="Choose which dataset to evaluate the model on"
    )
    
    # Get data based on selection
    if dataset == "Test Set":
        X_test = get_session_state('X_test')
        y_test = get_session_state('y_test')
        
        if X_test is None or y_test is None:
            show_warning("Test set not available. Please split the data in the Model Training page.")
            return
        
        X = X_test
        y_true = y_test
    else:
        # Use full dataset
        X = df[selected_features]
        y_true = df[target_column]
    
    # Make predictions
    try:
        y_pred = predict_regressor(model, X)
        
        # Calculate regression metrics
        metrics = regression_metrics(y_true, y_pred)
        
        # Display metrics
        st.write("**Regression Metrics:**")
        
        # Create metrics DataFrame
        metrics_list = [
            ('Mean Squared Error (MSE)', metrics['mse']),
            ('Root Mean Squared Error (RMSE)', metrics['rmse']),
            ('Mean Absolute Error (MAE)', metrics['mae']),
            ('R² Score', metrics['r2']),
            ('Median Absolute Error', metrics['median_ae'])
        ]
        
        # Add MAPE if available
        if 'mape' in metrics and not pd.isna(metrics['mape']):
            metrics_list.append(('Mean Absolute Percentage Error (MAPE)', metrics['mape']))
        
        metrics_df = pd.DataFrame(metrics_list, columns=['Metric', 'Value'])
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.6f}")
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Display residual statistics
        if 'residuals' in metrics:
            st.write("**Residual Statistics:**")
            
            residual_stats = [
                ('Mean', metrics['residuals']['mean']),
                ('Standard Deviation', metrics['residuals']['std']),
                ('Minimum', metrics['residuals']['min']),
                ('Maximum', metrics['residuals']['max'])
            ]
            
            residual_df = pd.DataFrame(residual_stats, columns=['Statistic', 'Value'])
            residual_df['Value'] = residual_df['Value'].apply(lambda x: f"{x:.6f}")
            
            st.dataframe(residual_df, use_container_width=True)
        
        # Metrics explanations
        with create_expander("Metric Explanations", expanded=False):
            st.markdown("""
            - **Mean Squared Error (MSE)**: Average of squared errors between predicted and actual values
            - **Root Mean Squared Error (RMSE)**: Square root of MSE, in same units as the target
            - **Mean Absolute Error (MAE)**: Average of absolute errors between predicted and actual values
            - **R² Score**: Proportion of variance explained by the model (1.0 is perfect prediction)
            - **Median Absolute Error**: Median of absolute errors, robust to outliers
            - **Mean Absolute Percentage Error (MAPE)**: Average of absolute percentage errors
            """)
    
    except Exception as e:
        show_error(f"Error calculating regression metrics: {str(e)}")

def render_residual_analysis_tab(models, df, target_column):
    """
    Render the residual analysis tab.
    """
    st.subheader("Residual Analysis")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to evaluate on
    dataset = st.radio(
        "Select Dataset",
        options=["Test Set", "Full Dataset"],
        horizontal=True,
        key="residual_dataset",
        help="Choose which dataset to evaluate the model on"
    )
    
    # Get data based on selection
    if dataset == "Test Set":
        X_test = get_session_state('X_test')
        y_test = get_session_state('y_test')
        
        if X_test is None or y_test is None:
            show_warning("Test set not available. Please split the data in the Model Training page.")
            return
        
        X = X_test
        y_true = y_test
    else:
        # Use full dataset
        X = df[selected_features]
        y_true = df[target_column]
    
    # Make predictions
    try:
        y_pred = predict_regressor(model, X)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Display residual statistics
        st.write("**Residual Statistics:**")
        
        residual_stats = {
            'Mean': residuals.mean(),
            'Median': residuals.median(),
            'Standard Deviation': residuals.std(),
            'Minimum': residuals.min(),
            'Maximum': residuals.max(),
            'Skewness': residuals.skew(),
            'Kurtosis': residuals.kurtosis()
        }
        
        stats_df = pd.DataFrame({
            'Statistic': list(residual_stats.keys()),
            'Value': [f"{v:.6f}" for v in residual_stats.values()]
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Create residual plots
        st.subheader("Residual Plots")
        
        # Residual histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(residuals, kde=True, ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        
        st.pyplot(fig)
        
        # Residuals vs predicted values
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs. Predicted Values')
        
        st.pyplot(fig)
        
        # QQ plot for normality check
        from scipy import stats as scipy_stats
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create QQ plot
        scipy_stats.probplot(residuals, plot=ax)
        ax.set_title('QQ Plot of Residuals')
        
        st.pyplot(fig)
        
        # Normality test
        st.subheader("Normality Test")
        
        # Shapiro-Wilk test
        if len(residuals) <= 5000:  # Shapiro-Wilk limited to 5000 samples
            stat, p_value = scipy_stats.shapiro(residuals)
            st.write(f"**Shapiro-Wilk Test**")
            st.write(f"Statistic: {stat:.6f}")
            st.write(f"p-value: {p_value:.6f}")
            
            if p_value < 0.05:
                st.write("The residuals do not follow a normal distribution (p-value < 0.05).")
            else:
                st.write("The residuals appear to follow a normal distribution (p-value >= 0.05).")
        else:
            # For larger datasets, use D'Agostino K² test
            stat, p_value = scipy_stats.normaltest(residuals)
            st.write(f"**D'Agostino-Pearson Test**")
            st.write(f"Statistic: {stat:.6f}")
            st.write(f"p-value: {p_value:.6f}")
            
            if p_value < 0.05:
                st.write("The residuals do not follow a normal distribution (p-value < 0.05).")
            else:
                st.write("The residuals appear to follow a normal distribution (p-value >= 0.05).")
                
        # Autocorrelation analysis (for time series data)
        if st.checkbox("Show Autocorrelation Analysis (for time series data)"):
            from statsmodels.graphics.tsaplots import plot_acf
            
            # ACF plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            try:
                plot_acf(residuals, ax=ax, lags=min(50, len(residuals) // 5))
                ax.set_title('Autocorrelation Function (ACF) of Residuals')
                
                st.pyplot(fig)
                
                st.write("""
                **Interpreting the ACF Plot:**
                - Significant spikes at non-zero lags indicate autocorrelation in the residuals
                - For a good model, residuals should be random (white noise) with no autocorrelation
                - Autocorrelation suggests the model is missing important patterns in the data
                """)
            except Exception as e:
                st.warning(f"Error creating ACF plot: {str(e)}")
    
    except Exception as e:
        show_error(f"Error performing residual analysis: {str(e)}")

def render_prediction_vs_actual_tab(models, df, target_column):
    """
    Render the prediction vs actual tab.
    """
    st.subheader("Prediction vs. Actual")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to evaluate on
    dataset = st.radio(
        "Select Dataset",
        options=["Test Set", "Full Dataset"],
        horizontal=True,
        key="pred_actual_dataset",
        help="Choose which dataset to evaluate the model on"
    )
    
    # Get data based on selection
    if dataset == "Test Set":
        X_test = get_session_state('X_test')
        y_test = get_session_state('y_test')
        
        if X_test is None or y_test is None:
            show_warning("Test set not available. Please split the data in the Model Training page.")
            return
        
        X = X_test
        y_true = y_test
    else:
        # Use full dataset
        X = df[selected_features]
        y_true = df[target_column]
    
    # Make predictions
    try:
        y_pred = predict_regressor(model, X)
        
        # Scatter plot of predicted vs actual
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'r--')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs. True Values')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        st.pyplot(fig)
        
        # Prediction vs actual table
        st.subheader("Prediction vs. Actual Values")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'True Value': y_true.values,
            'Predicted Value': y_pred,
            'Error': y_true.values - y_pred,
            'Absolute Error': np.abs(y_true.values - y_pred),
            'Squared Error': (y_true.values - y_pred) ** 2
        })
        
        # Add percentage error (avoiding division by zero)
        non_zero_mask = (y_true.values != 0)
        results['Percentage Error'] = np.nan
        if non_zero_mask.any():
            results.loc[non_zero_mask, 'Percentage Error'] = (results.loc[non_zero_mask, 'Error'] / y_true.values[non_zero_mask]) * 100
        
        # Show results table
        st.dataframe(results.head(100), use_container_width=True)
        
        # Bin predictions and calculate average error by bin
        st.subheader("Error Analysis by Value Range")
        
        # Create bins of true values
        n_bins = st.slider("Number of Bins", min_value=5, max_value=50, value=10)
        
        # Create bin edges
        bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
        
        # Create bin labels
        bin_labels = [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" for i in range(n_bins)]
        
        # Assign bins
        results['Value Bin'] = pd.cut(y_true.values, bins=bin_edges, labels=bin_labels)
        
        # Group by bin and calculate average errors
        bin_analysis = results.groupby('Value Bin').agg({
            'True Value': 'count',
            'Error': 'mean',
            'Absolute Error': 'mean',
            'Percentage Error': 'mean'
        }).reset_index()
        
        bin_analysis = bin_analysis.rename(columns={
            'True Value': 'Count',
            'Error': 'Mean Error',
            'Absolute Error': 'Mean Absolute Error',
            'Percentage Error': 'Mean Percentage Error'
        })
        
        # Format numeric columns
        for col in ['Mean Error', 'Mean Absolute Error', 'Mean Percentage Error']:
            bin_analysis[col] = bin_analysis[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        st.dataframe(bin_analysis, use_container_width=True)
        
        # Plot error by bin
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate bin centers for plotting
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
        
        # Convert back to numeric
        numeric_errors = pd.to_numeric(bin_analysis['Mean Error'].str.replace('N/A', 'nan'), errors='coerce')
        numeric_abs_errors = pd.to_numeric(bin_analysis['Mean Absolute Error'].str.replace('N/A', 'nan'), errors='coerce')
        
        # Create plot
        ax.bar(range(len(bin_centers)), numeric_errors, width=0.4, label='Mean Error', alpha=0.7)
        ax.bar([x + 0.4 for x in range(len(bin_centers))], numeric_abs_errors, width=0.4, label='Mean Absolute Error', alpha=0.7)
        
        ax.set_xticks(range(len(bin_centers)))
        ax.set_xticklabels([f"{x:.2f}" for x in bin_centers], rotation=45, ha='right')
        ax.set_xlabel('Value Range (bin center)')
        ax.set_ylabel('Error')
        ax.set_title('Mean Error by Value Range')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    except Exception as e:
        show_error(f"Error comparing predictions to actual values: {str(e)}")

def render_clustering_metrics_tab(models, df):
    """
    Render the clustering metrics tab.
    """
    st.subheader("Clustering Metrics")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to evaluate on
    dataset = st.radio(
        "Select Dataset",
        options=["Training Set", "Full Dataset"],
        horizontal=True,
        key="cluster_metrics_dataset",
        help="Choose which dataset to evaluate the model on"
    )
    
    # Get data based on selection
    if dataset == "Training Set":
        X_train = get_session_state('X_train')
        
        if X_train is None:
            show_warning("Training set not available. Please split the data in the Model Training page.")
            return
        
        X = X_train
    else:
        # Use full dataset
        X = df[selected_features]
    
    # Evaluate clustering
    try:
        # Get cluster labels
        labels = predict_clusterer(model, X)
        
        # Calculate clustering metrics
        metrics = clustering_metrics(X, labels)
        
        # Display metrics
        st.write("**Clustering Metrics:**")
        
        # Create metrics DataFrame
        metrics_list = []
        
        if 'silhouette' in metrics and not pd.isna(metrics['silhouette']):
            metrics_list.append(('Silhouette Score', metrics['silhouette']))
        
        if 'davies_bouldin' in metrics and not pd.isna(metrics['davies_bouldin']):
            metrics_list.append(('Davies-Bouldin Index', metrics['davies_bouldin']))
        
        if 'calinski_harabasz' in metrics and not pd.isna(metrics['calinski_harabasz']):
            metrics_list.append(('Calinski-Harabasz Index', metrics['calinski_harabasz']))
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list, columns=['Metric', 'Value'])
            metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.6f}")
            
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("No metrics available. This may be because the clustering produced only one cluster or included noise points.")
        
        # Display cluster information
        st.subheader("Cluster Information")
        
        # Count instances per cluster
        unique_labels = np.unique(labels)
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        
        # Create cluster info DataFrame
        cluster_info = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Count': cluster_counts.values,
            'Percentage': (cluster_counts.values / len(labels) * 100).round(2)
        })
        
        st.dataframe(cluster_info, use_container_width=True)
        
        # Plot cluster distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(cluster_info['Cluster'].astype(str), cluster_info['Count'])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.set_title('Instances per Cluster')
        
        st.pyplot(fig)
        
        # Metrics explanations
        with create_expander("Metric Explanations", expanded=False):
            st.markdown("""
            - **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters (-1 to 1, higher is better)
            - **Davies-Bouldin Index**: Average similarity of each cluster with its most similar cluster (lower is better)
            - **Calinski-Harabasz Index**: Ratio of between-cluster variance to within-cluster variance (higher is better)
            """)
    
    except Exception as e:
        show_error(f"Error calculating clustering metrics: {str(e)}")

def render_cluster_visualization_tab(models, df):
    """
    Render the cluster visualization tab.
    """
    st.subheader("Cluster Visualization")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to visualize
    dataset = st.radio(
        "Select Dataset",
        options=["Training Set", "Full Dataset"],
        horizontal=True,
        key="cluster_viz_dataset",
        help="Choose which dataset to visualize"
    )
    
    # Get data based on selection
    if dataset == "Training Set":
        X_train = get_session_state('X_train')
        
        if X_train is None:
            show_warning("Training set not available. Please split the data in the Model Training page.")
            return
        
        X = X_train
    else:
        # Use full dataset
        X = df[selected_features]
    
    try:
        # Get cluster labels
        labels = predict_clusterer(model, X)
        
        # Handle high-dimensional data
        if X.shape[1] > 2:
            st.info(f"The dataset has {X.shape[1]} dimensions. Dimensionality reduction is needed for visualization.")
            
            # Dimensionality reduction method
            reduction_method = st.selectbox(
                "Dimensionality Reduction Method",
                options=["PCA", "t-SNE", "UMAP"],
                index=0,
                help="Method to reduce dimensions for visualization"
            )
            
            # Apply dimensionality reduction
            if reduction_method == "PCA":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                explained_var = reducer.explained_variance_ratio_.sum()
                st.write(f"**Explained variance:** {explained_var:.2%}")
            
            elif reduction_method == "t-SNE":
                from sklearn.manifold import TSNE
                with st.spinner("Running t-SNE (this may take a while)..."):
                    reducer = TSNE(n_components=2, random_state=42)
                    X_reduced = reducer.fit_transform(X)
            
            elif reduction_method == "UMAP":
                try:
                    import umap
                    with st.spinner("Running UMAP (this may take a while)..."):
                        reducer = umap.UMAP(n_components=2, random_state=42)
                        X_reduced = reducer.fit_transform(X)
                except ImportError:
                    st.error("UMAP is not installed. Please install it with 'pip install umap-learn'.")
                    return
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
            
            # Add cluster centers if available (for K-means)
            if hasattr(model, 'cluster_centers_'):
                try:
                    centers = model.cluster_centers_
                    centers_reduced = reducer.transform(centers)
                    ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                              marker='*', s=300, c='red', edgecolor='k')
                except:
                    pass
            
            ax.set_xlabel(f"{reduction_method} Component 1")
            ax.set_ylabel(f"{reduction_method} Component 2")
            ax.set_title(f"Cluster Visualization using {reduction_method}")
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Cluster')
            
            st.pyplot(fig)
            
        else:
            # Create scatter plot directly for 2D data
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.7)
            
            # Add cluster centers if available (for K-means)
            if hasattr(model, 'cluster_centers_'):
                centers = model.cluster_centers_
                ax.scatter(centers[:, 0], centers[:, 1], marker='*', s=300, c='red', edgecolor='k')
            
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.set_title("Cluster Visualization")
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Cluster')
            
            st.pyplot(fig)
        
        # Feature distribution by cluster
        st.subheader("Feature Distribution by Cluster")
        
        # Create DataFrame with cluster labels
        cluster_df = X.copy()
        cluster_df['Cluster'] = labels
        
        # Select feature to analyze
        feature = st.selectbox(
            "Select Feature to Analyze",
            options=selected_features,
            index=0
        )
        
        # Box plot of feature by cluster
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(x='Cluster', y=feature, data=cluster_df, ax=ax)
        ax.set_title(f"Distribution of {feature} by Cluster")
        
        st.pyplot(fig)
        
        # Feature statistics by cluster
        st.write("**Feature Statistics by Cluster:**")
        
        # Calculate statistics
        cluster_stats = cluster_df.groupby('Cluster')[feature].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        # Format values
        for col in ['mean', 'std', 'min', 'max']:
            cluster_stats[col] = cluster_stats[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Pairplot for selected features
        if len(selected_features) > 1:
            st.subheader("Pairwise Relationships")
            
            # Select features for pairplot
            pairplot_features = st.multiselect(
                "Select Features for Pairplot",
                options=selected_features,
                default=selected_features[:min(3, len(selected_features))],
                help="Choose 2-5 features for the best visualization"
            )
            
            if len(pairplot_features) >= 2:
                with st.spinner("Generating pairplot..."):
                    # Create pairplot
                    fig = sns.pairplot(
                        cluster_df, 
                        vars=pairplot_features, 
                        hue='Cluster', 
                        palette='viridis',
                        corner=True,
                        plot_kws={'alpha': 0.5}
                    )
                    
                    fig.fig.suptitle("Feature Relationships by Cluster", y=1.02)
                    st.pyplot(fig.fig)
    
    except Exception as e:
        show_error(f"Error visualizing clusters: {str(e)}")

def render_silhouette_analysis_tab(models, df):
    """
    Render the silhouette analysis tab.
    """
    st.subheader("Silhouette Analysis")
    
    # Get current model
    current_model = get_session_state('current_model')
    
    if current_model not in models:
        show_warning("No current model selected. Please select a model in the Model Comparison tab.")
        return
    
    model_info = models[current_model]
    model = model_info.get('model')
    
    # Get selected features
    selected_features = get_session_state('selected_features')
    
    if not selected_features:
        show_warning("No features selected. Please set up features in the Model Training page.")
        return
    
    # Select dataset to analyze
    dataset = st.radio(
        "Select Dataset",
        options=["Training Set", "Full Dataset"],
        horizontal=True,
        key="silhouette_dataset",
        help="Choose which dataset to analyze"
    )
    
    # Get data based on selection
    if dataset == "Training Set":
        X_train = get_session_state('X_train')
        
        if X_train is None:
            show_warning("Training set not available. Please split the data in the Model Training page.")
            return
        
        X = X_train
    else:
        # Use full dataset
        X = df[selected_features]
    
    try:
        # Get cluster labels
        labels = predict_clusterer(model, X)
        
        # Get unique clusters
        unique_labels = np.unique(labels)
        
        # Check if valid for silhouette analysis (at least 2 clusters, no noise points)
        if len(unique_labels) < 2 or -1 in unique_labels:
            st.warning("Silhouette analysis requires at least 2 clusters and no noise points (-1 labels).")
            return
        
        # Calculate silhouette scores
        from sklearn.metrics import silhouette_samples, silhouette_score
        
        # Overall silhouette score
        silhouette_avg = silhouette_score(X, labels)
        st.write(f"**Overall Silhouette Score:** {silhouette_avg:.4f}")
        
        # Sample-level silhouette values
        sample_silhouette_values = silhouette_samples(X, labels)
        
        # Create silhouette plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_lower = 10
        
        # Plot silhouette for each cluster
        for i in range(len(unique_labels)):
            # Get silhouette values for this cluster
            ith_cluster_values = sample_silhouette_values[labels == unique_labels[i]]
            ith_cluster_values.sort()
            
            size_cluster_i = ith_cluster_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(float(i) / len(unique_labels))
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(unique_labels[i]))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        ax.set_title("Silhouette Plot for Each Cluster")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster")
        
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        st.pyplot(fig)
        
        # Silhouette scores by cluster
        st.subheader("Silhouette Scores by Cluster")
        
        # Calculate mean silhouette score for each cluster
        cluster_silhouette = {}
        
        for label in unique_labels:
            cluster_values = sample_silhouette_values[labels == label]
            cluster_silhouette[label] = cluster_values.mean()
        
        # Create DataFrame
        silhouette_df = pd.DataFrame({
            'Cluster': list(cluster_silhouette.keys()),
            'Mean Silhouette Score': [f"{score:.4f}" for score in cluster_silhouette.values()],
            'Size': [np.sum(labels == label) for label in cluster_silhouette.keys()]
        })
        
        st.dataframe(silhouette_df, use_container_width=True)
        
        # Plot mean silhouette score by cluster
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cluster_labels = list(cluster_silhouette.keys())
        silhouette_values = list(cluster_silhouette.values())
        
        ax.bar(np.arange(len(cluster_labels)), silhouette_values, alpha=0.7)
        ax.axhline(y=silhouette_avg, color='r', linestyle='--', label=f'Average ({silhouette_avg:.4f})')
        
        ax.set_xticks(np.arange(len(cluster_labels)))
        ax.set_xticklabels(cluster_labels)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mean Silhouette Score')
        ax.set_title('Mean Silhouette Score by Cluster')
        ax.legend()
        
        st.pyplot(fig)
        
        # Silhouette analysis interpretation
        with create_expander("Silhouette Analysis Interpretation", expanded=False):
            st.markdown("""
            **Interpreting Silhouette Analysis:**
            
            - **Silhouette score** ranges from -1 to 1:
              - Values near +1 indicate the sample is far from neighboring clusters
              - Values near 0 indicate the sample is close to the decision boundary between clusters
              - Negative values indicate the sample may have been assigned to the wrong cluster
            
            - **Good clustering** typically shows:
              - Most samples have high silhouette values
              - Clusters have similar silhouette widths
              - Clusters have silhouette scores above the average
            
            - **Poor clustering** may show:
              - Wide variations in silhouette widths
              - Some clusters with below-average scores
              - Clusters with many negative silhouette values
            """)
    
    except Exception as e:
        show_error(f"Error performing silhouette analysis: {str(e)}")