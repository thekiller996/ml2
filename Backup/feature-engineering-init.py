"""
Feature engineering functionality for the ML Platform.
"""

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

__all__ = [
    # Feature selection
    'select_features',
    'select_by_variance',
    'select_by_correlation',
    'select_by_mutual_info',
    'select_by_model',
    'select_best_k',
    'recursive_feature_elimination',
    
    # Feature creation
    'create_polynomial_features',
    'create_interaction_features',
    'create_binned_features',
    'create_datetime_features',
    'create_text_features',
    'create_pca_features',
    'create_cluster_features',
    
    # Dimensionality reduction
    'reduce_dimensions',
    'apply_pca',
    'apply_tsne',
    'apply_umap',
    'apply_lda',
    'apply_kernel_pca',
    
    # Feature transformation
    'apply_transformation',
    'apply_math_func',
    'create_lag_features',
    'create_window_features',
    'apply_spectral_transformation'
]
