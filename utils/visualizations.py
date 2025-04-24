"""
Visualization utilities for the ML tool.
Contains functions for creating various plots and visual representations of data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import streamlit as st

# Set default styling for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid")

def plot_distribution(data, column, plot_type='histogram', bins=30, kde=True):
    """
    Plot distribution of a column in the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    column : str
        Column name to visualize
    plot_type : str
        Type of plot: 'histogram', 'boxplot', 'violin', or 'density'
    bins : int
        Number of bins for histogram
    kde : bool
        Whether to include KDE in histogram
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == 'histogram':
        sns.histplot(data=data, x=column, bins=bins, kde=kde, ax=ax)
    elif plot_type == 'boxplot':
        sns.boxplot(data=data, x=column, ax=ax)
    elif plot_type == 'violin':
        sns.violinplot(data=data, x=column, ax=ax)
    elif plot_type == 'density':
        sns.kdeplot(data=data[column], fill=True, ax=ax)
    
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency' if plot_type != 'density' else 'Density')
    
    return fig

def plot_correlation_matrix(data, method='pearson', cmap='coolwarm', figsize=(12, 10)):
    """
    Plot correlation matrix of numeric columns in the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset (numeric columns only)
    method : str
        Correlation method: 'pearson', 'kendall', or 'spearman'
    cmap : str
        Colormap for heatmap
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Calculate correlation matrix
    corr_matrix = data.select_dtypes(include=['number']).corr(method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap=cmap,
        vmin=-1, 
        vmax=1, 
        center=0,
        fmt='.2f',
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(f'Correlation Matrix ({method.capitalize()})')
    fig.tight_layout()
    
    return fig

def plot_pairplot(data, hue=None, vars=None, diag_kind='kde', markers=None):
    """
    Create a pairplot of selected variables in the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    hue : str, optional
        Variable in data to map plot aspects to different colors
    vars : list, optional
        Variables to include in the plot
    diag_kind : {'hist', 'kde'}
        Kind of plot for the diagonal
    markers : str, optional
        Markers to use for scatter plots
        
    Returns:
    --------
    grid : seaborn.axisgrid.PairGrid
        The pairplot grid
    """
    # If vars is None, use all numeric columns (limited to avoid huge plots)
    if vars is None:
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 5:  # Limit to 5 variables to avoid performance issues
            vars = numeric_cols[:5]
            st.warning(f"Limiting pairplot to first 5 numeric columns. Selected: {vars}")
        else:
            vars = numeric_cols
    
    # Create pairplot
    grid = sns.pairplot(
        data=data,
        vars=vars,
        hue=hue,
        diag_kind=diag_kind,
        markers=markers
    )
    
    grid.fig.suptitle('Pairwise Relationships', y=1.02)
    
    return grid

def plot_feature_importance(importance, labels, title="Feature Importance", figsize=(10, 8)):
    """
    Plot feature importance from a model.
    
    Parameters:
    -----------
    importance : array-like
        Importance scores
    labels : array-like
        Feature names
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Sort features by importance
    indices = np.argsort(importance)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    ax.barh(range(len(indices)), importance[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([labels[i] for i in indices])
    ax.set_title(title)
    ax.set_xlabel('Importance')
    
    fig.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, cmap='Blues', figsize=(8, 6)):
    """
    Plot confusion matrix for classification results.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : array-like, optional
        List of label names
    normalize : bool
        Whether to normalize the confusion matrix
    cmap : str
        Colormap for heatmap
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap,
        xticklabels=labels, 
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    fig.tight_layout()
    return fig

def plot_roc_curve(y_true, y_score, figsize=(8, 6)):
    """
    Plot ROC curve for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    
    fig.tight_layout()
    return fig

def plot_learning_curve(train_sizes, train_scores, test_scores, figsize=(10, 6)):
    """
    Plot learning curve to show model performance as training set size increases.
    
    Parameters:
    -----------
    train_sizes : array-like
        Sizes of the training subsets
    train_scores : array-like of shape (n_ticks, n_cv_folds)
        Scores on training sets
    test_scores : array-like of shape (n_ticks, n_cv_folds)
        Scores on test set
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot learning curve
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    
    ax.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='red')
    
    ax.set_xlabel('Training samples')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    fig.tight_layout()
    return fig

def plot_precision_recall_curve(y_true, y_score, figsize=(8, 6)):
    """
    Plot precision-recall curve for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = np.mean(precision)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'Avg precision = {avg_precision:.2f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    
    fig.tight_layout()
    return fig

def plot_interactive_scatter(data, x, y, color=None, size=None, hover_name=None, title=None):
    """
    Create an interactive scatter plot using Plotly.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    color : str, optional
        Column name for color mapping
    size : str, optional
        Column name for size mapping
    hover_name : str, optional
        Column name for hover text
    title : str, optional
        Plot title
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The generated Plotly figure
    """
    fig = px.scatter(
        data_frame=data,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_name=hover_name,
        title=title or f'{y} vs {x}'
    )
    
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title=color if color else None,
        template='plotly_dark'
    )
    
    return fig

def plot_interactive_heatmap(data, x=None, y=None, z=None, title=None):
    """
    Create an interactive heatmap using Plotly.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    x : list, optional
        Column names for x-axis
    y : list, optional
        Column names for y-axis
    z : array-like, optional
        Values for heatmap, if None, correlation matrix is used
    title : str, optional
        Plot title
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The generated Plotly figure
    """
    if z is None:
        # Calculate correlation matrix
        z = data.select_dtypes(include=['number']).corr().values
        x = y = data.select_dtypes(include=['number']).columns
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        colorbar=dict(title='Correlation'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title or 'Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        template='plotly_dark'
    )
    
    return fig

def plot_time_series(data, date_column, value_columns, title=None, figsize=(12, 6)):
    """
    Plot time series data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset with datetime index or date column
    date_column : str
        Name of the date/time column
    value_columns : list
        List of columns to plot
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If date_column is not the index, set it as the index temporarily
    if date_column in data.columns:
        plot_data = data.set_index(date_column)
    else:
        plot_data = data.copy()
    
    # Plot each value column
    for column in value_columns:
        ax.plot(plot_data.index, plot_data[column], label=column)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title or 'Time Series Plot')
    ax.grid(True)
    ax.legend()
    
    fig.tight_layout()
    return fig

def plot_residuals(y_true, y_pred, figsize=(10, 6)):
    """
    Plot residuals from a regression model.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    residuals = y_true - y_pred
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot residuals vs predicted values
    ax1.scatter(y_pred, residuals)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted Values')
    ax1.grid(True)
    
    # Plot histogram of residuals
    ax2.hist(residuals, bins=30, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    ax2.grid(True)
    
    fig.tight_layout()
    return fig

def plot_cluster_results(data, x, y, labels, centroids=None, figsize=(10, 6)):
    """
    Plot clustering results in 2D.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    labels : array-like
        Cluster labels for each data point
    centroids : array-like, optional
        Centroids of clusters (if available)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with points colored by cluster
    scatter = ax.scatter(data[x], data[y], c=labels, cmap='viridis', alpha=0.7)
    
    # If centroids are provided, plot them
    if centroids is not None:
        ax.scatter(
            centroids[:, 0], centroids[:, 1], 
            marker='X', s=200, linewidths=2,
            color='red', edgecolor='black',
            label='Centroids'
        )
        ax.legend()
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('Cluster Results')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    fig.tight_layout()
    return fig

def plot_3d_scatter(data, x, y, z, color=None, size=None, title=None):
    """
    Create an interactive 3D scatter plot using Plotly.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    z : str
        Column name for z-axis
    color : str, optional
        Column name for color mapping
    size : str, optional
        Column name for size mapping
    title : str, optional
        Plot title
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The generated Plotly figure
    """
    fig = px.scatter_3d(
        data_frame=data,
        x=x,
        y=y,
        z=z,
        color=color,
        size=size,
        opacity=0.7,
        title=title or f'3D Scatter Plot: {x}, {y}, {z}'
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z
        ),
        template='plotly_dark'
    )
    
    return fig

def create_plot_grid(plot_functions, rows, cols, figsize=(15, 12), **kwargs):
    """
    Create a grid of plots.
    
    Parameters:
    -----------
    plot_functions : list of tuples
        List of (function, args_dict) tuples to create each subplot
    rows : int
        Number of rows in the grid
    cols : int
        Number of columns in the grid
    figsize : tuple
        Figure size
    **kwargs : dict
        Additional arguments to pass to plt.subplots
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure with subplots
    """
    if len(plot_functions) > rows * cols:
        raise ValueError(f"Too many plots ({len(plot_functions)}) for grid size ({rows}x{cols})")
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)
    axes = axes.flatten() if rows*cols > 1 else [axes]
    
    for i, (func, args) in enumerate(plot_functions):
        if i < len(axes):
            func(*args, ax=axes[i])
    
    # Hide unused axes
    for j in range(len(plot_functions), len(axes)):
        axes[j].set_visible(False)
    
    fig.tight_layout()
    return fig
