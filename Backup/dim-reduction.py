"""
Dimensionality reduction functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def reduce_dimensions(df: pd.DataFrame, method: str, columns: Optional[List[str]] = None,
                     n_components: int = 2, target: Optional[pd.Series] = None,
                     **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a dimensionality reduction method.
    
    Args:
        df: DataFrame with features
        method: Reduction method ('pca', 'tsne', 'umap', 'lda', 'kernel_pca')
        columns: List of columns to use (default: all numeric columns)
        n_components: Number of components to keep
        target: Target variable (required for LDA)
        **kwargs: Additional arguments for the specific method
    
    Returns:
        Tuple of (DataFrame with reduced features, reduction metadata)
    """
    # Map method name to function
    method_map = {
        'pca': apply_pca,
        'tsne': apply_tsne,
        'umap': apply_umap,
        'lda': apply_lda,
        'kernel_pca': apply_kernel_pca
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Call the appropriate function
    return method_map[method](df, columns, n_components, target, **kwargs)

def apply_pca(df: pd.DataFrame, columns: Optional[List[str]] = None,
            n_components: int = 2, target: Optional[pd.Series] = None,
            **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply Principal Component Analysis.
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        n_components: Number of components to keep
        target: Not used for PCA
        **kwargs: Additional PCA parameters
    
    Returns:
        Tuple of (DataFrame with reduced features, PCA metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'reduced_features': []}
    
    # Create PCA
    pca = PCA(n_components=min(n_components, len(columns), df.shape[0]),
             **kwargs)
    
    # Fit and transform
    X = df[columns]
    reduced_features = pca.fit_transform(X)
    
    # Create feature names
    feature_names = [f"pca_component_{i+1}" for i in range(reduced_features.shape[1])]
    
    # Create DataFrame with reduced features
    result_df = df.copy()
    for i, name in enumerate(feature_names):
        result_df[name] = reduced_features[:, i]
    
    # Create metadata
    metadata = {
        'reduced_features': feature_names,
        'transformer': pca,
        'source_columns': columns,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'components': pca.components_.tolist()
    }
    
    return result_df, metadata

def apply_tsne(df: pd.DataFrame, columns: Optional[List[str]] = None,
              n_components: int = 2, target: Optional[pd.Series] = None,
              **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply t-SNE (t-Distributed Stochastic Neighbor Embedding).
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        n_components: Number of components to keep
        target: Not used for t-SNE
        **kwargs: Additional t-SNE parameters
    
    Returns:
        Tuple of (DataFrame with reduced features, t-SNE metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'reduced_features': []}
    
    # Set defaults for t-SNE
    tsne_params = {
        'perplexity': min(30, max(5, df.shape[0] // 100)),
        'random_state': 42,
        'learning_rate': 'auto',
        'init': 'pca'
    }
    tsne_params.update(kwargs)
    
    # Create t-SNE
    tsne = TSNE(n_components=n_components, **tsne_params)
    
    # Fit and transform
    X = df[columns]
    reduced_features = tsne.fit_transform(X)
    
    # Create feature names
    feature_names = [f"tsne_component_{i+1}" for i in range(reduced_features.shape[1])]
    
    # Create DataFrame with reduced features
    result_df = df.copy()
    for i, name in enumerate(feature_names):
        result_df[name] = reduced_features[:, i]
    
    # Create metadata
    metadata = {
        'reduced_features': feature_names,
        'transformer': tsne,
        'source_columns': columns,
        'parameters': tsne_params
    }
    
    return result_df, metadata

def apply_umap(df: pd.DataFrame, columns: Optional[List[str]] = None,
             n_components: int = 2, target: Optional[pd.Series] = None,
             **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply UMAP (Uniform Manifold Approximation and Projection).
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        n_components: Number of components to keep
        target: Target variable for supervised UMAP (optional)
        **kwargs: Additional UMAP parameters
    
    Returns:
        Tuple of (DataFrame with reduced features, UMAP metadata)
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP is required for this function. Install it with 'pip install umap-learn'")
    
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'reduced_features': []}
    
    # Set defaults for UMAP
    umap_params = {
        'n_neighbors': min(15, max(2, df.shape[0] // 100)),
        'min_dist': 0.1,
        'metric': 'euclidean',
        'random_state': 42
    }
    umap_params.update(kwargs)
    
    # Create UMAP
    reducer = umap.UMAP(n_components=n_components, **umap_params)
    
    # Fit and transform
    X = df[columns]
    
    if target is not None:
        # Supervised UMAP
        reduced_features = reducer.fit_transform(X, target)
    else:
        # Unsupervised UMAP
        reduced_features = reducer.fit_transform(X)
    
    # Create feature names
    feature_names = [f"umap_component_{i+1}" for i in range(reduced_features.shape[1])]
    
    # Create DataFrame with reduced features
    result_df = df.copy()
    for i, name in enumerate(feature_names):
        result_df[name] = reduced_features[:, i]
    
    # Create metadata
    metadata = {
        'reduced_features': feature_names,
        'transformer': reducer,
        'source_columns': columns,
        'parameters': umap_params
    }
    
    return result_df, metadata

def apply_lda(df: pd.DataFrame, columns: Optional[List[str]] = None,
            n_components: int = 2, target: pd.Series = None,
            **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply Linear Discriminant Analysis.
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        n_components: Number of components to keep
        target: Target variable (required for LDA)
        **kwargs: Additional LDA parameters
    
    Returns:
        Tuple of (DataFrame with reduced features, LDA metadata)
    """
    if target is None:
        raise ValueError("Target variable is required for LDA")
    
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'reduced_features': []}
    
    # Get number of unique classes in target
    n_classes = len(np.unique(target))
    
    # LDA can produce at most min(n_features, n_classes - 1) components
    max_components = min(len(columns), n_classes - 1)
    n_components = min(n_components, max_components)
    
    if n_components < 1:
        return df.copy(), {'reduced_features': []}
    
    # Create LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)
    
    # Fit and transform
    X = df[columns]
    reduced_features = lda.fit_transform(X, target)
    
    # Create feature names
    feature_names = [f"lda_component_{i+1}" for i in range(reduced_features.shape[1])]
    
    # Create DataFrame with reduced features
    result_df = df.copy()
    for i, name in enumerate(feature_names):
        result_df[name] = reduced_features[:, i]
    
    # Create metadata
    metadata = {
        'reduced_features': feature_names,
        'transformer': lda,
        'source_columns': columns,
        'explained_variance_ratio': lda.explained_variance_ratio_.tolist() if hasattr(lda, 'explained_variance_ratio_') else None,
        'classes': lda.classes_.tolist()
    }
    
    return result_df, metadata

def apply_kernel_pca(df: pd.DataFrame, columns: Optional[List[str]] = None,
                   n_components: int = 2, target: Optional[pd.Series] = None,
                   kernel: str = 'rbf', **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply Kernel Principal Component Analysis.
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        n_components: Number of components to keep
        target: Not used for Kernel PCA
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid', etc.)
        **kwargs: Additional Kernel PCA parameters
    
    Returns:
        Tuple of (DataFrame with reduced features, Kernel PCA metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'reduced_features': []}
    
    # Create Kernel PCA
    kpca = KernelPCA(n_components=min(n_components, len(columns), df.shape[0]),
                   kernel=kernel, **kwargs)
    
    # Fit and transform
    X = df[columns]
    reduced_features = kpca.fit_transform(X)
    
    # Create feature names
    feature_names = [f"kpca_component_{i+1}" for i in range(reduced_features.shape[1])]
    
    # Create DataFrame with reduced features
    result_df = df.copy()
    for i, name in enumerate(feature_names):
        result_df[name] = reduced_features[:, i]
    
    # Create metadata
    metadata = {
        'reduced_features': feature_names,
        'transformer': kpca,
        'source_columns': columns,
        'kernel': kernel,
        'lambdas': kpca.lambdas_.tolist() if hasattr(kpca, 'lambdas_') else None
    }
    
    return result_df, metadata
