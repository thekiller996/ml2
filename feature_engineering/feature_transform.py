import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats as stats
import math
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureTransformer:
    """Class for feature transformation operations"""
    
    @staticmethod
    def create_polynomial_features(
        X: Union[pd.DataFrame, np.ndarray],
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        columns: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Create polynomial features from a dataset.
        
        Args:
            X: Input features
            degree: Polynomial degree
            interaction_only: If True, only generate interaction features
            include_bias: If True, include a bias column
            columns: Specific columns to transform (if X is DataFrame)
            
        Returns:
            Feature matrix with polynomial features
        """
        # Create polynomial transformer
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        # If we're working with a DataFrame
        if isinstance(X, pd.DataFrame):
            if columns is not None:
                # Apply only to specified columns
                X_poly = X.copy()
                X_selected = X_poly[columns].values
                poly_features = poly.fit_transform(X_selected)
                
                # Create feature names
                if columns is not None:
                    feature_names = poly.get_feature_names_out(columns)
                else:
                    feature_names = poly.get_feature_names_out()
                
                # Drop the original bias column if it exists
                if include_bias:
                    poly_features = poly_features[:, 1:]
                    feature_names = feature_names[1:]
                
                # Add polynomial features to DataFrame
                poly_df = pd.DataFrame(
                    poly_features, 
                    columns=feature_names,
                    index=X.index
                )
                
                # Concatenate with original DataFrame
                return pd.concat([X_poly, poly_df], axis=1)
            else:
                # Apply to all columns
                X_array = X.values
                poly_features = poly.fit_transform(X_array)
                
                # Create feature names
                feature_names = poly.get_feature_names_out(X.columns)
                
                return pd.DataFrame(
                    poly_features,
                    columns=feature_names,
                    index=X.index
                )
        else:
            # Apply to numpy array
            return poly.fit_transform(X)
    
    @staticmethod
    def log_transform(
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None,
        base: float = np.e,
        epsilon: float = 1e-10,
        handle_zeros: str = 'epsilon'
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply logarithmic transformation to features.
        
        Args:
            X: Input features
            columns: Specific columns to transform (if X is DataFrame)
            base: Logarithm base
            epsilon: Small value to add to avoid log(0)
            handle_zeros: How to handle zeros ('epsilon' or 'drop')
            
        Returns:
            Log-transformed features
        """
        if handle_zeros not in ['epsilon', 'drop']:
            raise ValueError("handle_zeros must be 'epsilon' or 'drop'")
        
        # Define log transform function
        def _log_func(x):
            # Add epsilon to avoid log(0)
            if handle_zeros == 'epsilon':
                x = x + epsilon
            
            if base == np.e:
                return np.log(x)
            elif base == 10:
                return np.log10(x)
            elif base == 2:
                return np.log2(x)
            else:
                return np.log(x) / np.log(base)
        
        # Create transformer
        log_transformer = FunctionTransformer(_log_func)
        
        # If we're working with a DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            if columns is not None:
                # Apply only to specified columns
                for col in columns:
                    # Skip columns with negative values
                    if (X_transformed[col] <= 0).any():
                        logger.warning(f"Column {col} contains zero or negative values. "
                                      f"Adding epsilon={epsilon} before log transform.")
                    
                    X_transformed[f"log_{col}"] = log_transformer.transform(X_transformed[col].values.reshape(-1, 1)).flatten()
            else:
                # Apply to all numeric columns
                numeric_cols = X.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    # Skip columns with negative values
                    if (X_transformed[col] <= 0).any():
                        logger.warning(f"Column {col} contains zero or negative values. "
                                      f"Adding epsilon={epsilon} before log transform.")
                    
                    X_transformed[f"log_{col}"] = log_transformer.transform(X_transformed[col].values.reshape(-1, 1)).flatten()
            
            return X_transformed
        else:
            # Apply to numpy array
            return log_transformer.transform(X)
    
    @staticmethod
    def box_cox_transform(
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None,
        lmbda: Optional[Union[float, List[float]]] = None
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, List[float]]]:
        """Apply Box-Cox transformation to make data more normal-like.
        
        Args:
            X: Input features
            columns: Specific columns to transform (if X is DataFrame)
            lmbda: Lambda parameter(s) for Box-Cox transform
            
        Returns:
            Box-Cox transformed features and lambdas
        """
        # Function to perform Box-Cox transform
        def _box_cox_col(x, lam=None):
            # Box-Cox requires positive values
            if np.min(x) <= 0:
                x = x + abs(np.min(x)) + 1e-6
            
            if lam is not None:
                transformed, _ = stats.boxcox(x, lmbda=lam)
            else:
                transformed, lam = stats.boxcox(x)
            
            return transformed, lam
        
        lambdas = []
        
        # If we're working with a DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            if columns is not None:
                target_cols = columns
            else:
                # Apply to all numeric columns
                target_cols = X.select_dtypes(include=['number']).columns
            
            for i, col in enumerate(target_cols):
                # Skip columns with negative values
                if (X_transformed[col] <= 0).any():
                    logger.warning(f"Column {col} contains zero or negative values. "
                                  f"Adding offset before Box-Cox transform.")
                
                # Get lambda for this column
                col_lambda = None
                if lmbda is not None:
                    if isinstance(lmbda, list):
                        col_lambda = lmbda[i] if i < len(lmbda) else None
                    else:
                        col_lambda = lmbda
                
                transformed, lam = _box_cox_col(X_transformed[col].values, col_lambda)
                X_transformed[f"boxcox_{col}"] = transformed
                lambdas.append(lam)
            
            return X_transformed
        else:
            # Apply to numpy array
            if X.ndim == 1:
                return _box_cox_col(X, lmbda)
            else:
                X_transformed = np.zeros_like(X, dtype=float)
                for i in range(X.shape[1]):
                    col_lambda = None
                    if lmbda is not None:
                        if isinstance(lmbda, list):
                            col_lambda = lmbda[i] if i < len(lmbda) else None
                        else:
                            col_lambda = lmbda
                    
                    transformed, lam = _box_cox_col(X[:, i], col_lambda)
                    X_transformed[:, i] = transformed
                    lambdas.append(lam)
                
                return X_transformed, lambdas
    
    @staticmethod
    def yeo_johnson_transform(
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None,
        lmbda: Optional[Union[float, List[float]]] = None
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, List[float]]]:
        """Apply Yeo-Johnson transformation to make data more normal-like.
        
        Args:
            X: Input features
            columns: Specific columns to transform (if X is DataFrame)
            lmbda: Lambda parameter(s) for Yeo-Johnson transform
            
        Returns:
            Yeo-Johnson transformed features and lambdas
        """
        from scipy.stats import yeojohnson
        
        # Function to perform Yeo-Johnson transform
        def _yeo_johnson_col(x, lam=None):
            if lam is not None:
                transformed, _ = yeojohnson(x, lmbda=lam)
            else:
                transformed, lam = yeojohnson(x)
            
            return transformed, lam
        
        lambdas = []
        
        # If we're working with a DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            if columns is not None:
                target_cols = columns
            else:
                # Apply to all numeric columns
                target_cols = X.select_dtypes(include=['number']).columns
            
            for i, col in enumerate(target_cols):
                # Get lambda for this column
                col_lambda = None
                if lmbda is not None:
                    if isinstance(lmbda, list):
                        col_lambda = lmbda[i] if i < len(lmbda) else None
                    else:
                        col_lambda = lmbda
                
                transformed, lam = _yeo_johnson_col(X_transformed[col].values, col_lambda)
                X_transformed[f"yeojohnson_{col}"] = transformed
                lambdas.append(lam)
            
            return X_transformed
        else:
            # Apply to numpy array
            if X.ndim == 1:
                return _yeo_johnson_col(X, lmbda)
            else:
                X_transformed = np.zeros_like(X, dtype=float)
                for i in range(X.shape[1]):
                    col_lambda = None
                    if lmbda is not None:
                        if isinstance(lmbda, list):
                            col_lambda = lmbda[i] if i < len(lmbda) else None
                        else:
                            col_lambda = lmbda
                    
                    transformed, lam = _yeo_johnson_col(X[:, i], col_lambda)
                    X_transformed[:, i] = transformed
                    lambdas.append(lam)
                
                return X_transformed, lambdas
    
    @staticmethod
    def power_transform(
        X: Union[pd.DataFrame, np.ndarray],
        power: float,
        columns: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply a power transformation to features.
        
        Args:
            X: Input features
            power: Exponent for power transform
            columns: Specific columns to transform (if X is DataFrame)
            
        Returns:
            Power-transformed features
        """
        # Define power transform function
        def _power_func(x, p):
            # Handle different power scenarios
            if p == 0:
                return np.log(x)
            else:
                return np.sign(x) * np.abs(x) ** p
        
        # If we're working with a DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            if columns is not None:
                # Apply only to specified columns
                for col in columns:
                    X_transformed[f"power{power}_{col}"] = _power_func(X_transformed[col].values, power)
            else:
                # Apply to all numeric columns
                numeric_cols = X.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    X_transformed[f"power{power}_{col}"] = _power_func(X_transformed[col].values, power)
            
            return X_transformed
        else:
            # Apply to numpy array
            return _power_func(X, power)
    
    @staticmethod
    def quantile_transform(
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None,
        n_quantiles: int = 1000,
        output_distribution: str = 'uniform',
        random_state: int = 42
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply quantile transformation to features.
        
        Args:
            X: Input features
            columns: Specific columns to transform (if X is DataFrame)
            n_quantiles: Number of quantiles to use
            output_distribution: Target distribution ('uniform' or 'normal')
            random_state: Random seed
            
        Returns:
            Quantile-transformed features
        """
        from sklearn.preprocessing import QuantileTransformer
        
        # Create transformer
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            random_state=random_state
        )
        
        # If we're working with a DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            if columns is not None:
                # Apply only to specified columns
                for col in columns:
                    X_transformed[f"quantile_{col}"] = transformer.fit_transform(
                        X_transformed[col].values.reshape(-1, 1)).flatten()
            else:
                # Apply to all numeric columns
                numeric_cols = X.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    X_transformed[f"quantile_{col}"] = transformer.fit_transform(
                        X_transformed[col].values.reshape(-1, 1)).flatten()
            
            return X_transformed
        else:
            # Apply to numpy array
            return transformer.fit_transform(X)
    
    @staticmethod
    def sinusoidal_transform(
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None,
        period: Optional[Union[float, Dict[str, float]]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply sinusoidal transformation for cyclic features.
        
        Args:
            X: Input features
            columns: Specific columns to transform (if X is DataFrame)
            period: Period/cycle length for each feature
            
        Returns:
            Sinusoidal transformed features (sin and cos components)
        """
        # If we're working with a DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            if columns is not None:
                target_cols = columns
            else:
                # Default: apply to all numeric columns
                target_cols = X.select_dtypes(include=['number']).columns
            
            for col in target_cols:
                col_period = None
                
                # Determine period for this column
                if period is not None:
                    if isinstance(period, dict):
                        col_period = period.get(col)
                    else:
                        col_period = period
                
                # If no period specified, try to detect based on column name
                if col_period is None:
                    if 'hour' in col.lower():
                        col_period = 24
                    elif 'day' in col.lower() or 'weekday' in col.lower():
                        col_period = 7
                    elif 'month' in col.lower():
                        col_period = 12
                    elif 'quarter' in col.lower():
                        col_period = 4
                    elif 'minute' in col.lower():
                        col_period = 60
                    elif 'second' in col.lower():
                        col_period = 60
                    elif 'angle' in col.lower() or 'degree' in col.lower():
                        col_period = 360
                    elif 'day_of_year' in col.lower():
                        col_period = 365
                    else:
                        # Skip this column if we can't determine a period
                        continue
                
                # Apply sinusoidal transformation
                X_transformed[f"{col}_sin"] = np.sin(2 * np.pi * X_transformed[col] / col_period)
                X_transformed[f"{col}_cos"] = np.cos(2 * np.pi * X_transformed[col] / col_period)
            
            return X_transformed
        else:
            # For numpy array, require explicit period
            if period is None:
                raise ValueError("Period must be specified for numpy array input")
            
            if not isinstance(period, (list, tuple)):
                period = [period] * X.shape[1]
            
            # Apply to all columns
            X_sin = np.sin(2 * np.pi * X / np.array(period).reshape(1, -1))
            X_cos = np.cos(2 * np.pi * X / np.array(period).reshape(1, -1))
            
            # Combine original, sin, and cos features
            return np.hstack([X, X_sin, X_cos])
    
    @staticmethod
    def apply_custom_function(
        X: Union[pd.DataFrame, np.ndarray],
        func: Callable,
        columns: Optional[List[str]] = None,
        func_name: str = 'custom',
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply a custom function to features.
        
        Args:
            X: Input features
            func: Custom function to apply
            columns: Specific columns to transform (if X is DataFrame)
            func_name: Name of function (for column naming)
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            Transformed features
        """
        # Create transformer
        transformer = FunctionTransformer(func, kw_args=kwargs)
        
        # If we're working with a DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            if columns is not None:
                # Apply only to specified columns
                for col in columns:
                    X_transformed[f"{func_name}_{col}"] = transformer.transform(
                        X_transformed[col].values.reshape(-1, 1)).flatten()
            else:
                # Apply to all numeric columns
                numeric_cols = X.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    X_transformed[f"{func_name}_{col}"] = transformer.transform(
                        X_transformed[col].values.reshape(-1, 1)).flatten()
            
            return X_transformed
        else:
            # Apply to numpy array
            return transformer.transform(X)
    
    @staticmethod
    def sigmoid_transform(
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply sigmoid transformation to features.
        
        Args:
            X: Input features
            columns: Specific columns to transform (if X is DataFrame)
            
        Returns:
            Sigmoid-transformed features
        """
        # Define sigmoid function
        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        return FeatureTransformer.apply_custom_function(
            X, _sigmoid, columns, func_name='sigmoid'
        )
    
    @staticmethod
    def tanh_transform(
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply tanh transformation to features.
        
        Args:
            X: Input features
            columns: Specific columns to transform (if X is DataFrame)
            
        Returns:
            Tanh-transformed features
        """
        return FeatureTransformer.apply_custom_function(
            X, np.tanh, columns, func_name='tanh'
        )

class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for custom feature transformations"""
    
    def __init__(self, transformations: List[Dict[str, Any]]):
        """Initialize transformer with list of transformations.
        
        Args:
            transformations: List of dicts with keys:
                - 'type': Transformation type 
                - 'columns': Columns to transform (optional)
                - other parameters specific to the transformation
        """
        self.transformations = transformations
    
    def fit(self, X, y=None):
        """Fit the transformer (stateless).
        
        Args:
            X: Input features
            y: Target variable (unused)
        
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """Apply all transformations.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        X_result = X.copy() if isinstance(X, pd.DataFrame) else X.copy()
        
        for transform_params in self.transformations:
            transform_type = transform_params['type']
            columns = transform_params.get('columns')
            
            # Copy parameters and remove 'type' and 'columns'
            params = transform_params.copy()
            params.pop('type')
            if 'columns' in params:
                params.pop('columns')
            
            # Apply the specified transformation
            if transform_type == 'polynomial':
                X_result = FeatureTransformer.create_polynomial_features(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'log':
                X_result = FeatureTransformer.log_transform(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'box_cox':
                X_result = FeatureTransformer.box_cox_transform(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'yeo_johnson':
                X_result = FeatureTransformer.yeo_johnson_transform(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'power':
                X_result = FeatureTransformer.power_transform(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'quantile':
                X_result = FeatureTransformer.quantile_transform(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'sinusoidal':
                X_result = FeatureTransformer.sinusoidal_transform(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'sigmoid':
                X_result = FeatureTransformer.sigmoid_transform(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'tanh':
                X_result = FeatureTransformer.tanh_transform(
                    X_result, 
                    columns=columns, 
                    **params
                )
            elif transform_type == 'custom':
                if 'func' not in params:
                    raise ValueError("'func' parameter required for custom transformation")
                X_result = FeatureTransformer.apply_custom_function(
                    X_result, 
                    columns=columns, 
                    **params
                )
            else:
                raise ValueError(f"Unknown transformation type: {transform_type}")
        
        return X_result
        
    def fit_transform(self, X, y=None):
        """Fit and transform in one step.
        
        Args:
            X: Input features
            y: Target variable (unused)
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)