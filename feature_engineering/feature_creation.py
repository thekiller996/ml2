import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any, Optional, Tuple, Callable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureCreator:
    """Class for creating new features from existing data"""
    
    @staticmethod
    def create_interaction_features(
        X: Union[pd.DataFrame, np.ndarray],
        cols_to_interact: Optional[List[List[str]]] = None,
        interaction_type: str = 'product',
        include_all_pairs: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Create interaction features between columns.
        
        Args:
            X: Input features
            cols_to_interact: List of column pairs to interact
            interaction_type: Type of interaction ('product', 'sum', 'diff', 'ratio')
            include_all_pairs: If True, generate all possible pairs (ignores cols_to_interact)
            
        Returns:
            DataFrame or array with additional interaction features
        """
        # Working with pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
            numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
            
            # Determine which column pairs to interact
            pairs_to_interact = []
            if include_all_pairs:
                # Generate all possible pairs of numeric columns
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        pairs_to_interact.append([numeric_cols[i], numeric_cols[j]])
            elif cols_to_interact is not None:
                pairs_to_interact = cols_to_interact
            
            # Create interaction features
            for col_pair in pairs_to_interact:
                if len(col_pair) != 2:
                    logger.warning(f"Expected column pair, got {col_pair}. Skipping.")
                    continue
                
                col1, col2 = col_pair
                
                # Ensure columns exist in DataFrame
                if col1 not in X.columns or col2 not in X.columns:
                    logger.warning(f"Columns {col1} or {col2} not found in DataFrame. Skipping.")
                    continue
                
                # Create interaction feature based on specified type
                if interaction_type == 'product':
                    X_new[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                elif interaction_type == 'sum':
                    X_new[f"{col1}_plus_{col2}"] = X[col1] + X[col2]
                elif interaction_type == 'diff':
                    X_new[f"{col1}_minus_{col2}"] = X[col1] - X[col2]
                elif interaction_type == 'ratio':
                    # Avoid division by zero
                    X_new[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-8)
                else:
                    raise ValueError(f"Unknown interaction type: {interaction_type}")
            
            return X_new
            
        # Working with numpy array
        else:
            if cols_to_interact is None and not include_all_pairs:
                raise ValueError("For numpy arrays, either cols_to_interact or include_all_pairs=True must be provided")
            
            # Determine indices to interact
            idx_pairs = []
            if include_all_pairs:
                for i in range(X.shape[1]):
                    for j in range(i+1, X.shape[1]):
                        idx_pairs.append((i, j))
            else:
                for pair in cols_to_interact:
                    if len(pair) != 2 or not all(isinstance(i, int) for i in pair):
                        raise ValueError(f"Expected pair of column indices, got {pair}")
                    idx_pairs.append(tuple(pair))
            
            # Generate interactions
            interactions = []
            for i, j in idx_pairs:
                if interaction_type == 'product':
                    interactions.append(X[:, i] * X[:, j])
                elif interaction_type == 'sum':
                    interactions.append(X[:, i] + X[:, j])
                elif interaction_type == 'diff':
                    interactions.append(X[:, i] - X[:, j])
                elif interaction_type == 'ratio':
                    interactions.append(X[:, i] / (X[:, j] + 1e-8))
            
            if not interactions:
                return X
            
            return np.column_stack([X] + [interact.reshape(-1, 1) for interact in interactions])
    
    @staticmethod
    def extract_datetime_features(
        df: pd.DataFrame,
        datetime_cols: List[str],
        features: List[str] = ['year', 'month', 'day', 'hour', 'minute', 'second', 
                               'weekday', 'quarter', 'day_of_year', 'is_weekend', 
                               'is_month_start', 'is_month_end'],
        drop_original: bool = False
    ) -> pd.DataFrame:
        """Extract features from datetime columns.
        
        Args:
            df: Input DataFrame
            datetime_cols: List of datetime columns
            features: List of datetime features to extract
            drop_original: Whether to drop original datetime columns
            
        Returns:
            DataFrame with extracted datetime features
        """
        result_df = df.copy()
        
        for col in datetime_cols:
            # Ensure column exists
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame. Skipping.")
                continue
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    result_df[col] = pd.to_datetime(df[col])
                except Exception as e:
                    logger.error(f"Failed to convert {col} to datetime: {str(e)}. Skipping.")
                    continue
            
            # Extract requested features
            for feature in features:
                if feature == 'year':
                    result_df[f"{col}_year"] = result_df[col].dt.year
                elif feature == 'month':
                    result_df[f"{col}_month"] = result_df[col].dt.month
                elif feature == 'day':
                    result_df[f"{col}_day"] = result_df[col].dt.day
                elif feature == 'hour':
                    result_df[f"{col}_hour"] = result_df[col].dt.hour
                elif feature == 'minute':
                    result_df[f"{col}_minute"] = result_df[col].dt.minute
                elif feature == 'second':
                    result_df[f"{col}_second"] = result_df[col].dt.second
                elif feature == 'weekday':
                    result_df[f"{col}_weekday"] = result_df[col].dt.weekday
                elif feature == 'quarter':
                    result_df[f"{col}_quarter"] = result_df[col].dt.quarter
                elif feature == 'day_of_year':
                    result_df[f"{col}_day_of_year"] = result_df[col].dt.dayofyear
                elif feature == 'is_weekend':
                    result_df[f"{col}_is_weekend"] = result_df[col].dt.weekday.isin([5, 6]).astype(int)
                elif feature == 'is_month_start':
                    result_df[f"{col}_is_month_start"] = result_df[col].dt.is_month_start.astype(int)
                elif feature == 'is_month_end':
                    result_df[f"{col}_is_month_end"] = result_df[col].dt.is_month_end.astype(int)
                elif feature == 'week_of_year':
                    result_df[f"{col}_week_of_year"] = result_df[col].dt.isocalendar().week
                else:
                    logger.warning(f"Unknown datetime feature: {feature}. Skipping.")
        
        # Drop original columns if requested
        if drop_original:
            result_df = result_df.drop(columns=datetime_cols)
        
        return result_df
    
    @staticmethod
    def extract_text_features(
        df: pd.DataFrame,
        text_cols: List[str],
        features: List[str] = ['char_count', 'word_count', 'sentence_count', 
                              'avg_word_length', 'special_char_count'],
        drop_original: bool = False
    ) -> pd.DataFrame:
        """Extract features from text columns.
        
        Args:
            df: Input DataFrame
            text_cols: List of text columns
            features: List of text features to extract
            drop_original: Whether to drop original text columns
            
        Returns:
            DataFrame with extracted text features
        """
        result_df = df.copy()
        
        for col in text_cols:
            # Ensure column exists
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame. Skipping.")
                continue
            
            # Convert to string if needed
            result_df[col] = result_df[col].astype(str)
            
            # Extract requested features
            for feature in features:
                if feature == 'char_count':
                    result_df[f"{col}_char_count"] = result_df[col].str.len()
                elif feature == 'word_count':
                    result_df[f"{col}_word_count"] = result_df[col].str.split().str.len()
                elif feature == 'sentence_count':
                    result_df[f"{col}_sentence_count"] = result_df[col].str.count(r'[.!?]+')
                elif feature == 'avg_word_length':
                    # Calculate average word length
                    def avg_word_len(text):
                        words = text.split()
                        if not words:
                            return 0
                        return sum(len(word) for word in words) / len(words)
                    result_df[f"{col}_avg_word_length"] = result_df[col].apply(avg_word_len)
                elif feature == 'special_char_count':
                    result_df[f"{col}_special_char_count"] = result_df[col].str.count(r'[^a-zA-Z0-9\s]')
                elif feature == 'uppercase_count':
                    result_df[f"{col}_uppercase_count"] = result_df[col].str.count(r'[A-Z]')
                elif feature == 'lowercase_count':
                    result_df[f"{col}_lowercase_count"] = result_df[col].str.count(r'[a-z]')
                elif feature == 'digit_count':
                    result_df[f"{col}_digit_count"] = result_df[col].str.count(r'[0-9]')
                elif feature == 'unique_word_count':
                    result_df[f"{col}_unique_word_count"] = result_df[col].apply(
                        lambda x: len(set(x.lower().split())))
                else:
                    logger.warning(f"Unknown text feature: {feature}. Skipping.")
        
        # Drop original columns if requested
        if drop_original:
            result_df = result_df.drop(columns=text_cols)
        
        return result_df
    
    @staticmethod
    def create_aggregation_features(
        df: pd.DataFrame,
        group_cols: List[str],
        agg_cols: List[str],
        agg_funcs: List[str] = ['mean', 'min', 'max', 'std', 'count']
    ) -> pd.DataFrame:
        """Create aggregation features based on group columns.
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_cols: Columns to aggregate
            agg_funcs: Aggregation functions to apply
            
        Returns:
            DataFrame with aggregation features
        """
        result_df = df.copy()
        
        # Check if columns exist
        missing_cols = [col for col in group_cols + agg_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columns {missing_cols} not found in DataFrame. Skipping.")
            return result_df
        
        # Create aggregations
        agg_dict = {col: agg_funcs for col in agg_cols}
        agg_df = df.groupby(group_cols).agg(agg_dict)
        
        # Flatten multi-index columns
        agg_df.columns = [f"{col}_{func}" for col, func in agg_df.columns]
        
        # Reset index to join back with original df
        agg_df = agg_df.reset_index()
        
        # Join with original DataFrame
        result_df = result_df.merge(agg_df, on=group_cols, how='left')
        
        return result_df
    
    @staticmethod
    def create_window_features(
        df: pd.DataFrame,
        time_col: str,
        target_cols: List[str],
        windows: List[int] = [1, 3, 7, 14, 30],
        agg_funcs: List[str] = ['mean', 'min', 'max', 'std'],
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create rolling window features for time series data.
        
        Args:
            df: Input DataFrame
            time_col: Column containing time information
            target_cols: Columns to compute window features for
            windows: List of window sizes
            agg_funcs: Aggregation functions to apply
            group_cols: Optional columns to group by before computing windows
            
        Returns:
            DataFrame with window features
        """
        result_df = df.copy()
        
        # Check if columns exist
        all_cols = [time_col] + target_cols + (group_cols or [])
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columns {missing_cols} not found in DataFrame. Skipping.")
            return result_df
        
        # Sort by time column
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            try:
                result_df[time_col] = pd.to_datetime(df[time_col])
            except Exception as e:
                logger.error(f"Failed to convert {time_col} to datetime: {str(e)}. Skipping.")
                return result_df
        
        result_df = result_df.sort_values(time_col)
        
        # Create window features
        for window in windows:
            for col in target_cols:
                # Create rolling window objects, with or without groupby
                if group_cols:
                    grouped = result_df.groupby(group_cols)
                    for func in agg_funcs:
                        # Use custom apply function for grouped data
                        result_df[f"{col}_{window}_{func}"] = grouped[col].apply(
                            lambda x: x.rolling(window=window, min_periods=1).agg(func)).reset_index(level=0, drop=True)
                else:
                    # Direct rolling for non-grouped data
                    for func in agg_funcs:
                        result_df[f"{col}_{window}_{func}"] = result_df[col].rolling(
                            window=window, min_periods=1).agg(func)
        
        return result_df
    
    @staticmethod
    def create_lag_features(
        df: pd.DataFrame,
        time_col: str,
        target_cols: List[str],
        lags: List[int] = [1, 3, 7, 14, 30],
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create lag features for time series data.
        
        Args:
            df: Input DataFrame
            time_col: Column containing time information
            target_cols: Columns to create lags for
            lags: List of lag values
            group_cols: Optional columns to group by before computing lags
            
        Returns:
            DataFrame with lag features
        """
        result_df = df.copy()
        
        # Check if columns exist
        all_cols = [time_col] + target_cols + (group_cols or [])
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columns {missing_cols} not found in DataFrame. Skipping.")
            return result_df
        
        # Sort by time column
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            try:
                result_df[time_col] = pd.to_datetime(df[time_col])
            except Exception as e:
                logger.error(f"Failed to convert {time_col} to datetime: {str(e)}. Skipping.")
                return result_df
        
        result_df = result_df.sort_values(time_col)
        
        # Create lag features
        for lag in lags:
            for col in target_cols:
                # Create lag features, with or without groupby
                if group_cols:
                    result_df[f"{col}_lag_{lag}"] = result_df.groupby(group_cols)[col].shift(lag)
                else:
                    result_df[f"{col}_lag_{lag}"] = result_df[col].shift(lag)
        
        return result_df
    
    @staticmethod
    def create_diff_features(
        df: pd.DataFrame,
        time_col: str,
        target_cols: List[str],
        diffs: List[int] = [1, 2, 3, 7],
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create difference features for time series data.
        
        Args:
            df: Input DataFrame
            time_col: Column containing time information
            target_cols: Columns to create differences for
            diffs: List of difference periods
            group_cols: Optional columns to group by before computing differences
            
        Returns:
            DataFrame with difference features
        """
        result_df = df.copy()
        
        # Check if columns exist
        all_cols = [time_col] + target_cols + (group_cols or [])
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columns {missing_cols} not found in DataFrame. Skipping.")
            return result_df
        
        # Sort by time column
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            try:
                result_df[time_col] = pd.to_datetime(df[time_col])
            except Exception as e:
                logger.error(f"Failed to convert {time_col} to datetime: {str(e)}. Skipping.")
                return result_df
        
        result_df = result_df.sort_values(time_col)
        
        # Create difference features
        for diff in diffs:
            for col in target_cols:
                # Create lag features for differencing
                if group_cols:
                    lag_col = result_df.groupby(group_cols)[col].shift(diff)
                else:
                    lag_col = result_df[col].shift(diff)
                
                # Compute difference
                result_df[f"{col}_diff_{diff}"] = result_df[col] - lag_col
                
                # Compute percentage change
                result_df[f"{col}_pct_change_{diff}"] = result_df[col].pct_change(periods=diff)
        
        return result_df
    
    @staticmethod
    def create_binned_features(
        df: pd.DataFrame,
        cols_to_bin: List[str],
        num_bins: Union[int, Dict[str, int]] = 10,
        strategy: str = 'quantile',
        labels: Optional[List[str]] = None,
        return_numeric: bool = True
    ) -> pd.DataFrame:
        """Create binned (discretized) features.
        
        Args:
            df: Input DataFrame
            cols_to_bin: Columns to bin
            num_bins: Number of bins (int or dict mapping column to bin count)
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            labels: Optional labels for the bins
            return_numeric: Whether to return numeric bins (0 to n-1)
            
        Returns:
            DataFrame with binned features
        """
        from sklearn.preprocessing import KBinsDiscretizer
        
        result_df = df.copy()
        
        for col in cols_to_bin:
            # Check if column exists
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame. Skipping.")
                continue
            
            # Determine number of bins for this column
            n_bins = num_bins[col] if isinstance(num_bins, dict) else num_bins
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric. Skipping.")
                continue
            
            # Create binning transformer
            if strategy in ['uniform', 'quantile', 'kmeans']:
                # Use sklearn for these strategies
                encode = 'ordinal' if return_numeric else 'onehot-dense'
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
                
                # Extract non-null values for fitting
                valid_mask = ~df[col].isna()
                X_valid = df.loc[valid_mask, col].values.reshape(-1, 1)
                
                # Fit and transform
                binned_values = discretizer.fit_transform(X_valid)
                
                # Create result column with appropriate type
                if return_numeric:
                    result_col = pd.Series(np.nan, index=df.index)
                    result_col.loc[valid_mask] = binned_values.flatten()
                    result_df[f"{col}_bin"] = result_col.astype('Int64')  # nullable integer type
                else:
                    # One-hot encoded result
                    binned_df = pd.DataFrame(
                        binned_values, 
                        index=df.index[valid_mask],
                        columns=[f"{col}_bin_{i}" for i in range(n_bins)]
                    )
                    # Join with result DataFrame
                    result_df = pd.concat([result_df, binned_df], axis=1)
            
            elif strategy == 'custom':
                # Custom bins using pandas cut
                result_df[f"{col}_bin"] = pd.cut(
                    df[col], 
                    bins=n_bins, 
                    labels=labels if labels else False,
                    include_lowest=True
                )
                
                # Convert to numeric if requested
                if return_numeric and not labels:
                    result_df[f"{col}_bin"] = result_df[f"{col}_bin"].cat.codes
            
            else:
                logger.warning(f"Unknown binning strategy: {strategy}. Skipping.")
        
        return result_df
    
    @staticmethod
    def create_cyclical_features(
        df: pd.DataFrame,
        cols_to_transform: List[str],
        periods: Dict[str, float] = None
    ) -> pd.DataFrame:
        """Create cyclical features using sine and cosine transformations.
        
        Args:
            df: Input DataFrame
            cols_to_transform: Columns to transform
            periods: Dictionary mapping column names to their period length
            
        Returns:
            DataFrame with cyclical features
        """
        result_df = df.copy()
        
        # Auto-detect periods if not provided
        if periods is None:
            periods = {}
            
        for col in cols_to_transform:
            # Check if column exists
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame. Skipping.")
                continue
            
            # Determine period for this column
            period = periods.get(col)
            
            # Try to auto-detect period if not provided
            if period is None:
                # Check column name for common cyclic variables
                if any(term in col.lower() for term in ['hour', 'hr']):
                    period = 24
                elif any(term in col.lower() for term in ['day', 'weekday']):
                    period = 7
                elif any(term in col.lower() for term in ['month']):
                    period = 12
                elif any(term in col.lower() for term in ['quarter']):
                    period = 4
                elif any(term in col.lower() for term in ['year']):
                    period = 365
                elif any(term in col.lower() for term in ['minute', 'min']):
                    period = 60
                elif any(term in col.lower() for term in ['second', 'sec']):
                    period = 60
                elif any(term in col.lower() for term in ['week']):
                    period = 52
                else:
                    # Try to infer from data range
                    if pd.api.types.is_numeric_dtype(df[col]):
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if 0 <= min_val <= max_val <= 24:
                            period = 24  # Hours
                        elif 1 <= min_val <= max_val <= 7:
                            period = 7   # Days of week
                        elif 1 <= min_val <= max_val <= 12:
                            period = 12  # Months
                        elif 1 <= min_val <= max_val <= 31:
                            period = 31  # Days of month
                        elif 1 <= min_val <= max_val <= 366:
                            period = 366 # Days of year
                        else:
                            logger.warning(f"Could not determine period for {col}. Skipping.")
                            continue
            
            # Create sine and cosine features
            result_df[f"{col}_sin"] = np.sin(2 * np.pi * df[col].astype(float) / period)
            result_df[f"{col}_cos"] = np.cos(2 * np.pi * df[col].astype(float) / period)
        
        return result_df

    @staticmethod
    def create_ratio_features(
        df: pd.DataFrame,
        numerator_cols: List[str],
        denominator_cols: List[str],
        suffixes: Optional[List[str]] = None,
        epsilon: float = 1e-10,
        generate_all_combinations: bool = False
    ) -> pd.DataFrame:
        """Create ratio features between columns.
        
        Args:
            df: Input DataFrame
            numerator_cols: Columns to use as numerators
            denominator_cols: Columns to use as denominators
            suffixes: Optional suffixes for generated column names
            epsilon: Small value to add to denominator to avoid division by zero
            generate_all_combinations: Whether to generate all possible combinations
            
        Returns:
            DataFrame with ratio features
        """
        result_df = df.copy()
        
        # Generate all combinations of numerator and denominator columns
        if generate_all_combinations:
            pairs = [(num, den) for num in numerator_cols for den in denominator_cols 
                    if num != den]  # Avoid dividing by self
        else:
            # Use paired columns
            if len(numerator_cols) != len(denominator_cols):
                raise ValueError("numerator_cols and denominator_cols must have the same length "
                                "when generate_all_combinations is False")
            pairs = list(zip(numerator_cols, denominator_cols))
        
        # Create ratio features
        for i, (num_col, den_col) in enumerate(pairs):
            # Check if columns exist
            if num_col not in df.columns or den_col not in df.columns:
                logger.warning(f"Columns {num_col} or {den_col} not found. Skipping.")
                continue
            
            # Determine column name for the ratio
            if suffixes and i < len(suffixes):
                ratio_col = f"ratio_{suffixes[i]}"
            else:
                ratio_col = f"ratio_{num_col}_to_{den_col}"
            
            # Create ratio feature, adding epsilon to avoid division by zero
            result_df[ratio_col] = df[num_col] / (df[den_col] + epsilon)
        
        return result_df
    
    @staticmethod
    def create_polynomial_features(
        df: pd.DataFrame,
        cols_to_transform: List[str],
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """Create polynomial features from input columns.
        
        Args:
            df: Input DataFrame
            cols_to_transform: Columns to transform
            degree: Polynomial degree
            interaction_only: Whether to only include interaction terms
            include_bias: Whether to include a bias column (constant term)
            
        Returns:
            DataFrame with polynomial features
        """
        result_df = df.copy()
        
        # Check if columns exist
        missing_cols = [col for col in cols_to_transform if col not in df.columns]
        if missing_cols:
            logger.warning(f"Columns {missing_cols} not found in DataFrame. Skipping.")
            return result_df
        
        # Extract columns to transform
        X = df[cols_to_transform].values
        
        # Create polynomial transformer
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        
        # Fit and transform
        X_poly = poly.fit_transform(X)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(cols_to_transform)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        
        # Remove original features if they're duplicated
        if include_bias:
            # Skip constant term and original features
            poly_df = poly_df.iloc[:, degree+1:]
        else:
            # Skip original features
            poly_df = poly_df.iloc[:, len(cols_to_transform):]
        
        # Concatenate with original DataFrame
        return pd.concat([result_df, poly_df], axis=1)
    
    @staticmethod
    def create_custom_features(
        df: pd.DataFrame,
        feature_definitions: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create custom features based on definitions.
        
        Args:
            df: Input DataFrame
            feature_definitions: List of feature definitions with:
                - 'name': Name for the new feature
                - 'formula': Formula string or lambda function
                - 'columns': Columns used in formula (optional)
            
        Returns:
            DataFrame with custom features
        """
        result_df = df.copy()
        
        for feature_def in feature_definitions:
            name = feature_def.get('name')
            formula = feature_def.get('formula')
            columns = feature_def.get('columns', [])
            
            if not name or not formula:
                logger.warning("Feature definition missing name or formula. Skipping.")
                continue
            
            try:
                if callable(formula):
                    # Lambda function
                    if columns:
                        # Apply with specific columns
                        result_df[name] = df[columns].apply(formula, axis=1)
                    else:
                        # Apply to entire row
                        result_df[name] = df.apply(formula, axis=1)
                else:
                    # String formula - evaluate using pandas eval
                    result_df[name] = df.eval(formula)
            except Exception as e:
                logger.error(f"Error creating custom feature '{name}': {str(e)}")
                continue
        
        return result_df