"""
Hyperparameter tuning functionality for the ML Platform.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_hyperparameters(model: Any, 
                       X: Union[pd.DataFrame, np.ndarray], 
                       y: Union[pd.Series, np.ndarray],
                       param_grid: Dict[str, List],
                       method: str = 'grid',
                       cv: int = 5,
                       scoring: Optional[str] = None,
                       **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune hyperparameters for a model.
    
    Args:
        model: Model to tune
        X: Features
        y: Target variable
        param_grid: Grid of hyperparameters to search
        method: Tuning method ('grid', 'random', 'bayesian')
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        **kwargs: Additional parameters for the search method
    
    Returns:
        Tuple of (best model, tuning results)
    """
    # Map method name to function
    method_map = {
        'grid': grid_search,
        'random': random_search,
        'bayesian': bayesian_optimization
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown tuning method: {method}")
    
    # Call the appropriate function
    return method_map[method](model, X, y, param_grid, cv, scoring, **kwargs)

def grid_search(model: Any, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray],
              param_grid: Dict[str, List],
              cv: int = 5,
              scoring: Optional[str] = None,
              **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        model: Model to tune
        X: Features
        y: Target variable
        param_grid: Grid of hyperparameters to search
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        **kwargs: Additional parameters for GridSearchCV
    
    Returns:
        Tuple of (best model, tuning results)
    """
    # Set default parameters
    search_params = {
        'n_jobs': -1,
        'verbose': 1,
        'return_train_score': True
    }
    search_params.update(kwargs)
    
    # Create grid search
    grid = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring=scoring,
        **search_params
    )
    
    # Fit the grid search
    grid.fit(X, y)
    
    # Process results
    results = {
        'best_params': grid.best_params_,
        'best_score': float(grid.best_score_),
        'best_index': int(grid.best_index_),
        'scoring': scoring or 'default',
        'cv': cv
    }
    
    # Add CV results
    cv_results = grid.cv_results_
    results['cv_results'] = {
        'params': cv_results['params'],
        'mean_test_score': cv_results['mean_test_score'].tolist(),
        'std_test_score': cv_results['std_test_score'].tolist(),
        'mean_train_score': cv_results['mean_train_score'].tolist() if 'mean_train_score' in cv_results else None,
        'std_train_score': cv_results['std_train_score'].tolist() if 'std_train_score' in cv_results else None,
        'rank_test_score': cv_results['rank_test_score'].tolist()
    }
    
    return grid.best_estimator_, results

def random_search(model: Any, 
                X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray],
                param_distributions: Dict[str, Any],
                cv: int = 5,
                scoring: Optional[str] = None,
                n_iter: int = 10,
                **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform randomized search for hyperparameter tuning.
    
    Args:
        model: Model to tune
        X: Features
        y: Target variable
        param_distributions: Distributions of hyperparameters to sample
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        n_iter: Number of parameter settings to sample
        **kwargs: Additional parameters for RandomizedSearchCV
    
    Returns:
        Tuple of (best model, tuning results)
    """
    # Set default parameters
    search_params = {
        'n_jobs': -1,
        'verbose': 1,
        'return_train_score': True,
        'random_state': 42
    }
    search_params.update(kwargs)
    
    # Create randomized search
    random_search = RandomizedSearchCV(
        model, 
        param_distributions, 
        n_iter=n_iter,
        cv=cv, 
        scoring=scoring,
        **search_params
    )
    
    # Fit the randomized search
    random_search.fit(X, y)
    
    # Process results
    results = {
        'best_params': random_search.best_params_,
        'best_score': float(random_search.best_score_),
        'best_index': int(random_search.best_index_),
        'scoring': scoring or 'default',
        'cv': cv,
        'n_iter': n_iter
    }
    
    # Add CV results
    cv_results = random_search.cv_results_
    results['cv_results'] = {
        'params': cv_results['params'],
        'mean_test_score': cv_results['mean_test_score'].tolist(),
        'std_test_score': cv_results['std_test_score'].tolist(),
        'mean_train_score': cv_results['mean_train_score'].tolist() if 'mean_train_score' in cv_results else None,
        'std_train_score': cv_results['std_train_score'].tolist() if 'std_train_score' in cv_results else None,
        'rank_test_score': cv_results['rank_test_score'].tolist()
    }
    
    return random_search.best_estimator_, results

def bayesian_optimization(model: Any, 
                        X: Union[pd.DataFrame, np.ndarray], 
                        y: Union[pd.Series, np.ndarray],
                        param_distributions: Dict[str, Any],
                        cv: int = 5,
                        scoring: Optional[str] = None,
                        n_iter: int = 10,
                        **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform Bayesian optimization for hyperparameter tuning.
    
    Args:
        model: Model to tune
        X: Features
        y: Target variable
        param_distributions: Distributions of hyperparameters to sample
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        n_iter: Number of parameter settings to sample
        **kwargs: Additional parameters for BayesSearchCV
    
    Returns:
        Tuple of (best model, tuning results)
    """
    try:
        from skopt import BayesSearchCV
    except ImportError:
        raise ImportError("Scikit-optimize is required for Bayesian optimization. Install it with 'pip install scikit-optimize'")
    
    # Set default parameters
    search_params = {
        'n_jobs': -1,
        'verbose': 1,
        'return_train_score': True,
        'random_state': 42
    }
    search_params.update(kwargs)
    
    # Create Bayesian search
    bayes_search = BayesSearchCV(
        model, 
        param_distributions, 
        n_iter=n_iter,
        cv=cv, 
        scoring=scoring,
        **search_params
    )
    
    # Fit the Bayesian search
    bayes_search.fit(X, y)
    
    # Process results
    results = {
        'best_params': dict(bayes_search.best_params_),
        'best_score': float(bayes_search.best_score_),
        'best_index': int(bayes_search.best_index_),
        'scoring': scoring or 'default',
        'cv': cv,
        'n_iter': n_iter
    }
    
    # Add CV results
    cv_results = bayes_search.cv_results_
    results['cv_results'] = {
        'params': [dict(p) for p in cv_results['params']],
        'mean_test_score': cv_results['mean_test_score'].tolist(),
        'std_test_score': cv_results['std_test_score'].tolist(),
        'mean_train_score': cv_results['mean_train_score'].tolist() if 'mean_train_score' in cv_results else None,
        'std_train_score': cv_results['std_train_score'].tolist() if 'std_train_score' in cv_results else None,
        'rank_test_score': cv_results['rank_test_score'].tolist()
    }
    
    # Add optimization details
    if hasattr(bayes_search, 'optimizer_results_'):
        opt_results = bayes_search.optimizer_results_
        results['optimization'] = {
            'x': [list(x) for x in opt_results.x_iters],
            'y': opt_results.func_vals.tolist()
        }
    
    return bayes_search.best_estimator_, results
