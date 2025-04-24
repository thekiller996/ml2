"""
Utility functions for the ML Platform.
"""

from utils.file_ops import (
    ensure_dir,
    save_file,
    load_file,
    get_file_extension,
    list_files,
    delete_file
)

from utils.visualizations import (
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_boxplot,
    create_heatmap,
    create_roc_curve,
    create_confusion_matrix,
    create_feature_importance_plot
)

from utils.stats import (
    describe_data,
    test_normality,
    calculate_correlation,
    calculate_vif,
    calculate_chi_square,
    bootstrap_statistic,
    confidence_interval
)

from utils.misc import (
    get_random_seed,
    format_time,
    mem_usage,
    truncate_text,
    is_notebook,
    generate_id
)

__all__ = [
    # File operations
    'ensure_dir',
    'save_file',
    'load_file',
    'get_file_extension',
    'list_files',
    'delete_file',
    
    # Visualizations
    'create_bar_chart',
    'create_line_chart',
    'create_scatter_plot',
    'create_boxplot',
    'create_heatmap',
    'create_roc_curve',
    'create_confusion_matrix',
    'create_feature_importance_plot',
    
    # Statistics
    'describe_data',
    'test_normality',
    'calculate_correlation',
    'calculate_vif',
    'calculate_chi_square',
    'bootstrap_statistic',
    'confidence_interval',
    
    # Miscellaneous
    'get_random_seed',
    'format_time',
    'mem_usage',
    'truncate_text',
    'is_notebook',
    'generate_id'
]
