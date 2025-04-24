"""
Utility functions for the ML Platform.
"""

# Import utility modules for easier access
from utils.misc import (
    get_version_info,
    generate_unique_id,
    get_timestamp,
    format_file_size,
    format_duration,
    is_valid_json,
    memoize,
    timer,
    retry
)

# Import visualization functions
from utils.visualizations import Visualizer

# Create aliases for backward compatibility
create_bar_chart = Visualizer.plot_bar
create_line_chart = Visualizer.plot_line
create_scatter_plot = Visualizer.plot_scatter
create_histogram = Visualizer.plot_histogram
create_boxplot = Visualizer.plot_boxplot
create_heatmap = Visualizer.plot_heatmap
create_correlation_matrix = Visualizer.plot_correlation_matrix
create_3d_scatter = Visualizer.plot_3d_scatter
create_pca_plot = Visualizer.plot_pca_variance
create_confusion_matrix = Visualizer.plot_confusion_matrix
create_roc_curve = Visualizer.plot_roc_curve
create_precision_recall_curve = Visualizer.plot_precision_recall_curve
create_feature_importance_plot = Visualizer.plot_feature_importance

# Make these available at the package level
__all__ = [
    'get_version_info',
    'generate_unique_id',
    'get_timestamp',
    'format_file_size',
    'format_duration',
    'is_valid_json',
    'memoize',
    'timer',
    'retry',
    'Visualizer',
    'create_bar_chart',
    'create_line_chart',
    'create_scatter_plot',
    'create_histogram',
    'create_boxplot',
    'create_heatmap',
    'create_correlation_matrix',
    'create_3d_scatter',
    'create_pca_plot',
    'create_confusion_matrix',
    'create_roc_curve',
    'create_precision_recall_curve',
    'create_feature_importance_plot'
]