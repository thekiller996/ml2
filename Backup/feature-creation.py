"""
Feature creation functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime

def create_polynomial_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                              degree: int = 2, interaction_only: bool = False,
                              **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create polynomial features.
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        degree: Polynomial degree
        interaction_only: Whether to include only interaction features
        
    Returns:
        Tuple of (DataFrame with new features, feature creation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'created_features': []}
    
    # Create polynomial features
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False  # Don't include the bias term (constant)
    )
    
    # Fit and transform
    X = df[columns]
    poly_features = poly.fit_transform(X)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(columns)
    
    # Remove original feature names
    new_feature_names = [name for name in feature_names if name not in columns]
    
    # Create DataFrame with new features
    new_features_df = pd.DataFrame(
        poly_features[:, len(columns):],  # Exclude original features
        columns=new_feature_names,
        index=df.index
    )
    
    # Combine with original DataFrame
    result_df = pd.concat([df, new_features_df], axis=1)
    
    # Create metadata
    metadata = {
        'created_features': new_feature_names,
        'transformer': poly,
        'source_columns': columns
    }
    
    return result_df, metadata

def create_interaction_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                               pairs: Optional[List[Tuple[str, str]]] = None,
                               **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create interaction features (products of pairs of features).
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        pairs: List of column pairs to multiply (default: all combinations)
        
    Returns:
        Tuple of (DataFrame with new features, feature creation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns or len(columns) < 2:
        return df.copy(), {'created_features': []}
    
    result_df = df.copy()
    created_features = []
    
    # If pairs are not specified, create all combinations
    if pairs is None:
        pairs = []
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                pairs.append((col1, col2))
    
    # Create interaction features
    for col1, col2 in pairs:
        if col1 in df.columns and col2 in df.columns:
            feature_name = f"{col1}_x_{col2}"
            result_df[feature_name] = df[col1] * df[col2]
            created_features.append(feature_name)
    
    # Create metadata
    metadata = {
        'created_features': created_features,
        'pairs': pairs
    }
    
    return result_df, metadata

def create_binned_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                          bins: Union[int, List[float]] = 10,
                          strategy: str = 'uniform',
                          labels: Optional[List[str]] = None,
                          **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create binned features from numeric columns.
    
    Args:
        df: DataFrame with features
        columns: List of columns to bin (default: all numeric columns)
        bins: Number of bins or bin edges
        strategy: Binning strategy ('uniform', 'quantile', or 'kmeans')
        labels: Custom labels for bins
        
    Returns:
        Tuple of (DataFrame with new features, feature creation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'created_features': []}
    
    result_df = df.copy()
    created_features = []
    bin_edges = {}
    
    for col in columns:
        feature_name = f"{col}_bin"
        
        # Get values for binning
        values = df[col].dropna()
        
        if isinstance(bins, int):
            if strategy == 'uniform':
                # Create uniform bins
                bin_edge = np.linspace(values.min(), values.max(), bins + 1)
            elif strategy == 'quantile':
                # Create quantile-based bins
                bin_edge = np.percentile(values, np.linspace(0, 100, bins + 1))
            elif strategy == 'kmeans':
                # Create KMeans-based bins
                kmeans = KMeans(n_clusters=bins, random_state=42)
                kmeans.fit(values.values.reshape(-1, 1))
                centers = sorted(kmeans.cluster_centers_.flatten())
                # Create bin edges from cluster centers
                bin_edge = np.concatenate([[values.min()], 
                                          [(centers[i] + centers[i+1])/2 for i in range(len(centers)-1)],
                                          [values.max()]])
            else:
                raise ValueError(f"Unknown binning strategy: {strategy}")
        else:
            # Use provided bin edges
            bin_edge = bins
        
        # Store bin edges
        bin_edges[col] = bin_edge
        
        # Create binned feature
        result_df[feature_name] = pd.cut(
            df[col],
            bins=bin_edge,
            labels=labels,
            include_lowest=True
        )
        
        created_features.append(feature_name)
    
    # Create metadata
    metadata = {
        'created_features': created_features,
        'bin_edges': bin_edges,
        'strategy': strategy
    }
    
    return result_df, metadata

def create_datetime_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                            features: Optional[List[str]] = None,
                            drop_original: bool = False,
                            **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create features from datetime columns.
    
    Args:
        df: DataFrame with features
        columns: List of datetime columns to use
        features: List of datetime features to create
                 (default: all of ['year', 'month', 'day', 'hour', 'weekday', 'quarter'])
        drop_original: Whether to drop original datetime columns
        
    Returns:
        Tuple of (DataFrame with new features, feature creation metadata)
    """
    if columns is None:
        # Try to identify datetime columns
        columns = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                columns.append(col)
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                try:
                    # Try to convert to datetime
                    pd.to_datetime(df[col], errors='raise')
                    columns.append(col)
                except:
                    pass
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'created_features': []}
    
    # Default datetime features to extract
    default_features = ['year', 'month', 'day', 'hour', 'weekday', 'quarter']
    
    if features is None:
        features = default_features
    
    result_df = df.copy()
    created_features = []
    
    for col in columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            result_df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Extract datetime features
        dt_col = result_df[col].dt
        
        if 'year' in features:
            feature_name = f"{col}_year"
            result_df[feature_name] = dt_col.year
            created_features.append(feature_name)
        
        if 'month' in features:
            feature_name = f"{col}_month"
            result_df[feature_name] = dt_col.month
            created_features.append(feature_name)
        
        if 'day' in features:
            feature_name = f"{col}_day"
            result_df[feature_name] = dt_col.day
            created_features.append(feature_name)
        
        if 'hour' in features:
            feature_name = f"{col}_hour"
            result_df[feature_name] = dt_col.hour
            created_features.append(feature_name)
        
        if 'minute' in features:
            feature_name = f"{col}_minute"
            result_df[feature_name] = dt_col.minute
            created_features.append(feature_name)
        
        if 'second' in features:
            feature_name = f"{col}_second"
            result_df[feature_name] = dt_col.second
            created_features.append(feature_name)
        
        if 'weekday' in features:
            feature_name = f"{col}_weekday"
            result_df[feature_name] = dt_col.dayofweek
            created_features.append(feature_name)
        
        if 'quarter' in features:
            feature_name = f"{col}_quarter"
            result_df[feature_name] = dt_col.quarter
            created_features.append(feature_name)
        
        if 'is_weekend' in features:
            feature_name = f"{col}_is_weekend"
            result_df[feature_name] = dt_col.dayofweek.isin([5, 6]).astype(int)
            created_features.append(feature_name)
        
        if 'is_month_start' in features:
            feature_name = f"{col}_is_month_start"
            result_df[feature_name] = dt_col.is_month_start.astype(int)
            created_features.append(feature_name)
        
        if 'is_month_end' in features:
            feature_name = f"{col}_is_month_end"
            result_df[feature_name] = dt_col.is_month_end.astype(int)
            created_features.append(feature_name)
        
        if 'dayofyear' in features:
            feature_name = f"{col}_dayofyear"
            result_df[feature_name] = dt_col.dayofyear
            created_features.append(feature_name)
    
    # Drop original columns if requested
    if drop_original:
        result_df = result_df.drop(columns=columns)
    
    # Create metadata
    metadata = {
        'created_features': created_features,
        'source_columns': columns,
        'extracted_features': features
    }
    
    return result_df, metadata

def create_text_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                        features: Optional[List[str]] = None,
                        language: str = 'english',
                        drop_original: bool = False,
                        **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create features from text columns.
    
    Args:
        df: DataFrame with features
        columns: List of text columns to use
        features: List of text features to create
                 (default: all of ['char_count', 'word_count', 'unique_word_count', 
                                  'word_density', 'punctuation_count', 'stopword_count'])
        language: Language for stopwords
        drop_original: Whether to drop original text columns
        
    Returns:
        Tuple of (DataFrame with new features, feature creation metadata)
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    try:
        nltk.data.find(f'corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    if columns is None:
        # Try to identify text columns (object or string dtype)
        columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'created_features': []}
    
    # Default text features to extract
    default_features = [
        'char_count', 'word_count', 'unique_word_count', 
        'word_density', 'punctuation_count', 'stopword_count'
    ]
    
    if features is None:
        features = default_features
    
    result_df = df.copy()
    created_features = []
    
    # Get stopwords for the language
    stop_words = set(stopwords.words(language))
    
    for col in columns:
        # Ensure column is string type
        text_series = df[col].astype(str)
        
        if 'char_count' in features:
            feature_name = f"{col}_char_count"
            result_df[feature_name] = text_series.str.len()
            created_features.append(feature_name)
        
        if 'word_count' in features:
            feature_name = f"{col}_word_count"
            result_df[feature_name] = text_series.apply(lambda x: len(word_tokenize(x)))
            created_features.append(feature_name)
        
        if 'unique_word_count' in features:
            feature_name = f"{col}_unique_word_count"
            result_df[feature_name] = text_series.apply(lambda x: len(set(word_tokenize(x))))
            created_features.append(feature_name)
        
        if 'word_density' in features and 'word_count' in features and 'char_count' in features:
            feature_name = f"{col}_word_density"
            word_count_col = f"{col}_word_count"
            char_count_col = f"{col}_char_count"
            result_df[feature_name] = result_df[word_count_col] / (result_df[char_count_col] + 1)
            created_features.append(feature_name)
        
        if 'punctuation_count' in features:
            feature_name = f"{col}_punctuation_count"
            result_df[feature_name] = text_series.apply(lambda x: sum(1 for c in x if c in '.,;:!?()[]{}"\''))
            created_features.append(feature_name)
        
        if 'stopword_count' in features:
            feature_name = f"{col}_stopword_count"
            result_df[feature_name] = text_series.apply(lambda x: sum(1 for word in word_tokenize(x.lower()) if word in stop_words))
            created_features.append(feature_name)
        
        if 'avg_word_length' in features:
            feature_name = f"{col}_avg_word_length"
            result_df[feature_name] = text_series.apply(lambda x: np.mean([len(word) for word in word_tokenize(x)]) if word_tokenize(x) else 0)
            created_features.append(feature_name)
        
        if 'sentiment' in features:
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                
                sia = SentimentIntensityAnalyzer()
                
                feature_name_pos = f"{col}_sentiment_pos"
                feature_name_neg = f"{col}_sentiment_neg"
                feature_name_comp = f"{col}_sentiment_compound"
                
                sentiments = text_series.apply(lambda x: sia.polarity_scores(x))
                result_df[feature_name_pos] = sentiments.apply(lambda x: x['pos'])
                result_df[feature_name_neg] = sentiments.apply(lambda x: x['neg'])
                result_df[feature_name_comp] = sentiments.apply(lambda x: x['compound'])
                
                created_features.extend([feature_name_pos, feature_name_neg, feature_name_comp])
            except:
                pass  # Skip sentiment if not available
    
    # Drop original columns if requested
    if drop_original:
        result_df = result_df.drop(columns=columns)
    
    # Create metadata
    metadata = {
        'created_features': created_features,
        'source_columns': columns,
        'extracted_features': features,
        'language': language
    }
    
    return result_df, metadata

def create_pca_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       n_components: Optional[int] = None,
                       variance_threshold: Optional[float] = 0.95,
                       **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create features using PCA.
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        n_components: Number of components to keep
        variance_threshold: Minimum explained variance to keep
        
    Returns:
        Tuple of (DataFrame with new features, feature creation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'created_features': []}
    
    # Determine number of components
    if n_components is None and variance_threshold is not None:
        # Use variance threshold to determine components
        n_components = min(len(columns), df.shape[0])
        
    # Create PCA
    pca = PCA(n_components=n_components)
    
    # Fit and transform
    X = df[columns]
    pca_features = pca.fit_transform(X)
    
    # Determine number of components to keep based on variance
    if variance_threshold is not None and n_components > 1:
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_to_keep = np.argmax(cumulative_variance >= variance_threshold) + 1
        pca_features = pca_features[:, :n_components_to_keep]
    else:
        n_components_to_keep = pca_features.shape[1]
    
    # Create feature names
    feature_names = [f"pca_component_{i+1}" for i in range(n_components_to_keep)]
    
    # Create DataFrame with PCA features
    pca_df = pd.DataFrame(
        pca_features,
        columns=feature_names,
        index=df.index
    )
    
    # Combine with original DataFrame
    result_df = pd.concat([df, pca_df], axis=1)
    
    # Create metadata
    metadata = {
        'created_features': feature_names,
        'transformer': pca,
        'source_columns': columns,
        'explained_variance_ratio': pca.explained_variance_ratio_[:n_components_to_keep].tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)[:n_components_to_keep].tolist()
    }
    
    return result_df, metadata

def create_cluster_features(df: pd.DataFrame, columns: Optional[List[str]] = None,
                           n_clusters: int = 3,
                           method: str = 'kmeans',
                           **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create features using clustering.
    
    Args:
        df: DataFrame with features
        columns: List of columns to use (default: all numeric columns)
        n_clusters: Number of clusters
        method: Clustering method ('kmeans', 'dbscan', 'hierarchical', etc.)
        
    Returns:
        Tuple of (DataFrame with new features, feature creation metadata)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df.copy(), {'created_features': []}
    
    # Get data for clustering
    X = df[columns]
    
    # Create and fit cluster model
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        cluster_model = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'gmm':
        from sklearn.mixture import GaussianMixture
        cluster_model = GaussianMixture(n_components=n_clusters, random_state=42)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Fit and predict clusters
    cluster_labels = cluster_model.fit_predict(X)
    
    # Create feature name
    feature_name = f"cluster_{method}"
    
    # Add cluster feature to DataFrame
    result_df = df.copy()
    result_df[feature_name] = cluster_labels
    
    # Create metadata
    metadata = {
        'created_features': [feature_name],
        'model': cluster_model,
        'source_columns': columns,
        'method': method,
        'n_clusters': n_clusters
    }
    
    # Add additional metadata based on method
    if method == 'kmeans':
        metadata['cluster_centers'] = cluster_model.cluster_centers_.tolist()
        metadata['inertia'] = cluster_model.inertia_
    
    return result_df, metadata
