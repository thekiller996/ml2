import numpy as np
import pandas as pd
from typing import Dict, Any, List, Callable, Union, Tuple, Optional
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                    cross_val_score, KFold, StratifiedKFold)
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
import optuna
from optuna.samplers import TPESampler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTuner:
    """Class for hyperparameter tuning of machine learning models"""
    
    @staticmethod
    def grid_search(
        model: BaseEstimator,
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Union[str, Callable] = None,
        n_jobs: int = -1,
        verbose: int = 1,
        return_train_score: bool = False
    ) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame]:
        """Perform grid search for hyperparameter optimization.
        
        Args:
            model: Model to tune
            param_grid: Dictionary of parameter names and possible values
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            return_train_score: Whether to return training scores
            
        Returns:
            Tuple of (best model, best parameters, results dataframe)
        """
        # Create GridSearchCV instance
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=return_train_score
        )
        
        # Fit the grid search
        logger.info("Starting grid search...")
        start_time = time.time()
        grid_search.fit(X, y)
        end_time = time.time()
        logger.info(f"Grid search completed in {end_time - start_time:.2f} seconds")
        
        # Get results as dataframe
        results = pd.DataFrame(grid_search.cv_results_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, results
    
    @staticmethod
    def random_search(
        model: BaseEstimator,
        param_distributions: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 100,
        cv: int = 5,
        scoring: Union[str, Callable] = None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42,
        return_train_score: bool = False
    ) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame]:
        """Perform randomized search for hyperparameter optimization.
        
        Args:
            model: Model to tune
            param_distributions: Dictionary of parameter names and distributions
            X: Feature matrix
            y: Target variable
            n_iter: Number of parameter combinations to try
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random seed
            return_train_score: Whether to return training scores
            
        Returns:
            Tuple of (best model, best parameters, results dataframe)
        """
        # Create RandomizedSearchCV instance
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score=return_train_score
        )
        
        # Fit the random search
        logger.info("Starting randomized search...")
        start_time = time.time()
        random_search.fit(X, y)
        end_time = time.time()
        logger.info(f"Randomized search completed in {end_time - start_time:.2f} seconds")
        
        # Get results as dataframe
        results = pd.DataFrame(random_search.cv_results_)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, results
    
    @staticmethod
    def bayesian_optimization(
        model: BaseEstimator,
        search_spaces: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 50,
        cv: int = 5,
        scoring: Union[str, Callable] = None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42
    ) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame]:
        """Perform Bayesian optimization for hyperparameter tuning.
        
        Args:
            model: Model to tune
            search_spaces: Dictionary of parameter names and search spaces
            X: Feature matrix
            y: Target variable
            n_iter: Number of iterations
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random seed
            
        Returns:
            Tuple of (best model, best parameters, results dataframe)
        """
        # Convert search spaces to skopt format if needed
        formatted_spaces = {}
        for param, space in search_spaces.items():
            if isinstance(space, (list, tuple)):
                if all(isinstance(x, (int, float)) for x in space):
                    # Numerical range
                    if all(isinstance(x, int) for x in space):
                        formatted_spaces[param] = Integer(min(space), max(space))
                    else:
                        formatted_spaces[param] = Real(min(space), max(space))
                else:
                    # Categorical
                    formatted_spaces[param] = Categorical(space)
            else:
                # Already formatted or string
                formatted_spaces[param] = space
        
        # Create BayesSearchCV instance
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=formatted_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )
        
        # Fit the Bayesian search
        logger.info("Starting Bayesian optimization...")
        start_time = time.time()
        bayes_search.fit(X, y)
        end_time = time.time()
        logger.info(f"Bayesian optimization completed in {end_time - start_time:.2f} seconds")
        
        # Get results as dataframe
        results = pd.DataFrame(bayes_search.cv_results_)
        
        logger.info(f"Best parameters: {bayes_search.best_params_}")
        logger.info(f"Best score: {bayes_search.best_score_:.4f}")
        
        return bayes_search.best_estimator_, bayes_search.best_params_, results
    
    @staticmethod
    def optuna_optimization(
        model_creator: Callable[[Dict[str, Any]], BaseEstimator],
        param_spaces: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        objective: str = 'binary',
        n_trials: int = 100,
        cv: int = 5,
        scoring: Union[str, Callable] = 'accuracy',
        direction: str = 'maximize',
        n_jobs: int = 1,
        timeout: Optional[int] = None,
        random_state: int = 42
    ) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame]:
        """Perform hyperparameter optimization using Optuna.
        
        Args:
            model_creator: Function that creates model from parameters
            param_spaces: Dictionary of parameter names and spaces
            X: Feature matrix
            y: Target variable
            objective: Type of objective ('binary', 'multiclass', 'regression')
            n_trials: Number of trials
            cv: Number of cross-validation folds
            scoring: Scoring metric
            direction: Optimization direction ('maximize' or 'minimize')
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds (optional)
            random_state: Random seed
            
        Returns:
            Tuple of (best model, best parameters, results dataframe)
        """
        # Setup cross-validation
        if objective in ['binary', 'multiclass']:
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Store all results
        results_list = []
        
        # Create Optuna study
        sampler = TPESampler(seed=random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)
        
        # Define objective function
        def objective_func(trial):
            # Sample parameters
            params = {}
            for param, space in param_spaces.items():
                if isinstance(space, (list, tuple)):
                    if all(isinstance(x, int) for x in space):
                        params[param] = trial.suggest_int(param, min(space), max(space))
                    elif all(isinstance(x, float) for x in space):
                        params[param] = trial.suggest_float(param, min(space), max(space))
                    else:
                        params[param] = trial.suggest_categorical(param, space)
                elif isinstance(space, dict):
                    if space['type'] == 'int':
                        params[param] = trial.suggest_int(
                            param, space['low'], space['high'], step=space.get('step', 1)
                        )
                    elif space['type'] == 'float':
                        if space.get('log', False):
                            params[param] = trial.suggest_float(
                                param, space['low'], space['high'], log=True
                            )
                        else:
                            params[param] = trial.suggest_float(
                                param, space['low'], space['high'], step=space.get('step')
                            )
                    elif space['type'] == 'categorical':
                        params[param] = trial.suggest_categorical(param, space['choices'])
            
            # Create model with sampled parameters
            model = model_creator(params)
            
            # Calculate cross-validation score
            cv_scores = cross_val_score(
                model, X, y, cv=cv_obj, scoring=scoring, n_jobs=n_jobs
            )
            
            # Store results
            results_list.append({
                'trial_number': trial.number,
                'params': params,
                'value': cv_scores.mean(),
                'std': cv_scores.std()
            })
            
            return cv_scores.mean()
        
        # Run optimization
        logger.info("Starting Optuna optimization...")
        start_time = time.time()
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout, n_jobs=1)
        end_time = time.time()
        logger.info(f"Optuna optimization completed in {end_time - start_time:.2f} seconds")
        
        # Get best parameters and create best model
        best_params = study.best_params
        best_model = model_creator(best_params)
        
        # Fit the best model
        best_model.fit(X, y)
        
        # Create results dataframe
        results_df = pd.DataFrame(results_list)
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return best_model, best_params, results_df
    
    @staticmethod
    def cross_validate(
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Union[str, List[str], Dict[str, Callable]] = None,
        n_jobs: int = -1,
        verbose: int = 0,
        return_estimator: bool = False,
        return_indices: bool = False,
        objective: str = 'binary'
    ) -> Dict[str, np.ndarray]:
        """Perform detailed cross-validation of a model.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            scoring: Scoring metric(s)
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            return_estimator: Whether to return trained estimators
            return_indices: Whether to return indices of test folds
            objective: Type of task ('binary', 'multiclass', 'regression')
            
        Returns:
            Dictionary of cross-validation results
        """
        from sklearn.model_selection import cross_validate
        
        # Setup cross-validation
        if objective in ['binary', 'multiclass'] and len(np.unique(y)) > 1:
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv_obj, scoring=scoring, n_jobs=n_jobs,
            verbose=verbose, return_estimator=return_estimator,
            return_train_score=True
        )
        
        # Calculate summary statistics
        results_summary = {key: {} for key in cv_results.keys()}
        for key, values in cv_results.items():
            if key == 'estimator' or key == 'indices':
                results_summary[key] = values
            else:
                results_summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        return results_summary
    
    @staticmethod
    def nested_cross_validation(
        model_creator: Callable[[], BaseEstimator],
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        outer_cv: int = 5,
        inner_cv: int = 3,
        scoring: Union[str, Callable] = None,
        search_method: str = 'grid',
        n_jobs: int = -1,
        verbose: int = 0,
        objective: str = 'binary',
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Perform nested cross-validation for unbiased performance estimation.
        
        Args:
            model_creator: Function that creates a new model instance
            param_grid: Parameter grid for tuning
            X: Feature matrix
            y: Target variable
            outer_cv: Number of outer CV folds
            inner_cv: Number of inner CV folds
            scoring: Scoring metric
            search_method: Method for parameter search ('grid', 'random', 'bayesian')
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            objective: Type of task ('binary', 'multiclass', 'regression')
            random_state: Random seed
            
        Returns:
            Dictionary with nested CV results
        """
        # Setup outer cross-validation
        if objective in ['binary', 'multiclass'] and len(np.unique(y)) > 1:
            outer_cv_obj = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        else:
            outer_cv_obj = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        
        # Setup inner cross-validation
        if objective in ['binary', 'multiclass'] and len(np.unique(y)) > 1:
            inner_cv_obj = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
        else:
            inner_cv_obj = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
        
        # Lists to store results
        outer_scores = []
        best_params_list = []
        best_models = []
        
        # Outer loop
        for i, (train_idx, test_idx) in enumerate(outer_cv_obj.split(X, y)):
            logger.info(f"Outer fold {i+1}/{outer_cv}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create fresh model
            model = model_creator()
            
            # Inner loop (parameter tuning)
            if search_method == 'grid':
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=inner_cv_obj,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=verbose
                )
            elif search_method == 'random':
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=100,
                    cv=inner_cv_obj,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    random_state=random_state
                )
            elif search_method == 'bayesian':
                # Convert param_grid to skopt format
                search_spaces = {}
                for param, values in param_grid.items():
                    if all(isinstance(x, int) for x in values):
                        search_spaces[param] = Integer(min(values), max(values))
                    elif all(isinstance(x, float) for x in values):
                        search_spaces[param] = Real(min(values), max(values))
                    else:
                        search_spaces[param] = Categorical(values)
                
                search = BayesSearchCV(
                    estimator=model,
                    search_spaces=search_spaces,
                    n_iter=50,
                    cv=inner_cv_obj,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    random_state=random_state
                )
            
            # Perform parameter search
            search.fit(X_train, y_train)
            
            # Get best parameters and model
            best_params = search.best_params_
            best_params_list.append(best_params)
            
            # Train model with best parameters on full training set
            best_model = search.best_estimator_
            best_models.append(best_model)
            
            # Evaluate on test set
            score = search.score(X_test, y_test)
            outer_scores.append(score)
            
            logger.info(f"  Best parameters: {best_params}")
            logger.info(f"  Test score: {score:.4f}")
        
        # Calculate summary statistics
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        logger.info(f"Nested CV completed. Mean score: {mean_score:.4f} ± {std_score:.4f}")
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'outer_scores': outer_scores,
            'best_params_per_fold': best_params_list,
            'best_models': best_models
        }
    
    @staticmethod
    def save_tuning_results(
        results: Dict[str, Any],
        best_model: BaseEstimator,
        best_params: Dict[str, Any],
        results_df: pd.DataFrame,
        method: str,
        output_dir: str = 'tuning_results'
    ) -> None:
        """Save hyperparameter tuning results to disk.
        
        Args:
            results: Dictionary of tuning results
            best_model: Best model from tuning
            best_params: Best parameters from tuning
            results_df: DataFrame of all results
            method: Tuning method name
            output_dir: Directory to save results
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save model
        model_path = os.path.join(output_dir, f"{method}_best_model_{timestamp}.pkl")
        joblib.dump(best_model, model_path)
        
        # Save best parameters
        params_path = os.path.join(output_dir, f"{method}_best_params_{timestamp}.json")
        pd.Series(best_params).to_json(params_path)
        
        # Save full results DataFrame
        results_path = os.path.join(output_dir, f"{method}_full_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        # Save results summary
        if 'mean_score' in results:
            summary_path = os.path.join(output_dir, f"{method}_summary_{timestamp}.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Method: {method}\n")
                f.write(f"Mean score: {results['mean_score']:.4f}\n")
                f.write(f"Std score: {results['std_score']:.4f}\n")
                f.write(f"Best parameters: {best_params}\n")
        
        logger.info(f"Tuning results saved to {output_dir}")