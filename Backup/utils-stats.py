"""
Statistical analysis utilities for the ML tool.
Contains functions for statistical testing, analysis and data exploration.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import streamlit as st
from typing import Dict, List, Tuple, Union, Optional

def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed description of a DataFrame including additional statistics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to analyze
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with descriptive statistics
    """
    # Get basic statistics
    basic_stats = df.describe(include='all').T
    
    # Add additional statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if not numeric_cols.empty:
        # Calculate additional metrics for numeric columns
        additional_stats = pd.DataFrame(index=df.columns)
        additional_stats['skew'] = df[numeric_cols].skew()
        additional_stats['kurtosis'] = df[numeric_cols].kurtosis()
        additional_stats['median'] = df[numeric_cols].median()
        additional_stats['iqr'] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
        additional_stats['missing'] = df.isna().sum()
        additional_stats['missing_pct'] = (df.isna().sum() / len(df)) * 100
        additional_stats['unique'] = df.nunique()
        additional_stats['unique_pct'] = (df.nunique() / len(df)) * 100
        
        # Merge with basic stats
        result = pd.concat([basic_stats, additional_stats], axis=1)
    else:
        # Just add missing counts for non-numeric DataFrames
        result = basic_stats
        result['missing'] = df.isna().sum()
        result['missing_pct'] = (df.isna().sum() / len(df)) * 100
        result['unique'] = df.nunique()
        result['unique_pct'] = (df.nunique() / len(df)) * 100
    
    # Round numeric results for better display
    numeric_cols_result = result.select_dtypes(include=['number']).columns
    result[numeric_cols_result] = result[numeric_cols_result].round(2)
    
    return result

def calculate_correlations(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix with multiple methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame (numeric columns only)
    method : str
        Correlation method: 'pearson', 'kendall', 'spearman', or 'all'
    
    Returns:
    --------
    pandas.DataFrame or dict of pandas.DataFrame
        Correlation matrix or dictionary of correlation matrices
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    if method == 'all':
        return {
            'pearson': numeric_df.corr(method='pearson'),
            'kendall': numeric_df.corr(method='kendall'),
            'spearman': numeric_df.corr(method='spearman')
        }
    else:
        return numeric_df.corr(method=method)

def detect_outliers(df: pd.DataFrame, method: str = 'zscore', threshold: float = 3.0) -> Dict[str, np.ndarray]:
    """
    Detect outliers in numeric columns using different methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str
        Method to use: 'zscore', 'iqr', or 'quantile'
    threshold : float
        Threshold for z-score method or IQR multiplier
    
    Returns:
    --------
    dict
        Dictionary with column names as keys and boolean arrays as values
        (True for outliers, False for non-outliers)
    """
    numeric_df = df.select_dtypes(include=['number'])
    result = {}
    
    if numeric_df.empty:
        return result
    
    for column in numeric_df.columns:
        data = numeric_df[column].dropna()
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            result[column] = z_scores > threshold
            
        elif method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            result[column] = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'quantile':
            lower_bound = data.quantile(0.01)
            upper_bound = data.quantile(0.99)
            result[column] = (data < lower_bound) | (data > upper_bound)
    
    return result

def test_normality(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform normality tests on numeric columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    alpha : float
        Significance level
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with test statistics and p-values
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    results = []
    
    for column in numeric_df.columns:
        data = numeric_df[column].dropna()
        
        if len(data) < 3:  # Skip columns with insufficient data
            continue
            
        # Shapiro-Wilk test (better for smaller samples)
        if len(data) < 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            shapiro_normal = shapiro_p > alpha
        else:
            shapiro_stat, shapiro_p, shapiro_normal = np.nan, np.nan, np.nan
            
        # D'Agostino's K^2 test
        try:
            k2_stat, k2_p = stats.normaltest(data)
            k2_normal = k2_p > alpha
        except:
            k2_stat, k2_p, k2_normal = np.nan, np.nan, np.nan
            
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm')
        ks_normal = ks_p > alpha
        
        results.append({
            'column': column,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'shapiro_normal': shapiro_normal,
            'k2_stat': k2_stat,
            'k2_p': k2_p,
            'k2_normal': k2_normal,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'ks_normal': ks_normal,
        })
    
    return pd.DataFrame(results).set_index('column')

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    dict
        Dictionary with regression metrics
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': stats.explained_variance_score(y_true, y_pred),
        'median_absolute_error': stats.median_absolute_error(y_true, y_pred)
    }

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None, 
                           average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate classification evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    y_prob : array-like, optional
        Predicted probabilities
    average : str
        Averaging method for multi-class metrics
    
    Returns:
    --------
    dict
        Dictionary with classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Add ROC AUC if probabilities are provided
    if y_prob is not None:
        # Convert to binary if needed
        if y_prob.ndim > 1 and y_prob.shape[1] == 2:
            # For binary classification, use the probability of the positive class
            y_prob = y_prob[:, 1]
            
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr' if y_prob.ndim > 1 else 'raise')
        except:
            # ROC AUC might fail for certain cases
            metrics['roc_auc'] = np.nan
    
    return metrics

def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Generate a classification report as a DataFrame.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    
    Returns:
    --------
    pandas.DataFrame
        Classification report as a DataFrame
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame(report_dict).T

def feature_correlation_with_target(df: pd.DataFrame, target_col: str, method: str = 'pearson') -> pd.Series:
    """
    Calculate correlation of all features with the target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_col : str
        Target column name
    method : str
        Correlation method: 'pearson', 'kendall', or 'spearman'
    
    Returns:
    --------
    pandas.Series
        Series with correlation coefficients
    """
    if target_col not in df.columns:
        return pd.Series()
    
    numeric_df = df.select_dtypes(include=['number']).copy()
    
    if numeric_df.empty or target_col not in numeric_df.columns:
        return pd.Series()
    
    correlations = numeric_df.corr(method=method)[target_col].drop(target_col)
    return correlations.sort_values(ascending=False)

def anova_test(df: pd.DataFrame, feature_col: str, group_col: str) -> Dict[str, float]:
    """
    Perform one-way ANOVA test to determine if there are significant differences 
    between the means of groups defined by group_col.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    feature_col : str
        Numeric column to analyze
    group_col : str
        Categorical column defining groups
    
    Returns:
    --------
    dict
        Dictionary with F-statistic and p-value
    """
    if feature_col not in df.columns or group_col not in df.columns:
        return {'f_statistic': np.nan, 'p_value': np.nan}
    
    # Get groups
    groups = []
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][feature_col].dropna()
        if len(group_data) > 0:
            groups.append(group_data)
    
    if len(groups) < 2:
        return {'f_statistic': np.nan, 'p_value': np.nan}
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value
    }

def chi2_test(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, float]:
    """
    Perform Chi-square test for independence between two categorical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    col1 : str
        First categorical column
    col2 : str
        Second categorical column
    
    Returns:
    --------
    dict
        Dictionary with chi2-statistic, p-value, and degrees of freedom
    """
    if col1 not in df.columns or col2 not in df.columns:
        return {'chi2': np.nan, 'p_value': np.nan, 'dof': np.nan}
    
    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof
    }

def t_test(data1: np.ndarray, data2: np.ndarray, equal_var: bool = True) -> Dict[str, float]:
    """
    Perform Student's t-test to compare means of two samples.
    
    Parameters:
    -----------
    data1 : array-like
        First sample
    data2 : array-like
        Second sample
    equal_var : bool
        If True, perform a standard independent t-test. 
        If False, perform Welch's t-test.
    
    Returns:
    --------
    dict
        Dictionary with t-statistic and p-value
    """
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value
    }

def paired_t_test(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
    """
    Perform paired t-test to compare means of two related samples.
    
    Parameters:
    -----------
    data1 : array-like
        First sample
    data2 : array-like
        Second sample
    
    Returns:
    --------
    dict
        Dictionary with t-statistic and p-value
    """
    t_stat, p_value = stats.ttest_rel(data1, data2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value
    }

def mann_whitney_test(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test (non-parametric test for independent samples).
    
    Parameters:
    -----------
    data1 : array-like
        First sample
    data2 : array-like
        Second sample
    
    Returns:
    --------
    dict
        Dictionary with U-statistic and p-value
    """
    u_stat, p_value = stats.mannwhitneyu(data1, data2)
    
    return {
        'u_statistic': u_stat,
        'p_value': p_value
    }

def wilcoxon_test(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric test for paired samples).
    
    Parameters:
    -----------
    data1 : array-like
        First sample
    data2 : array-like
        Second sample
    
    Returns:
    --------
    dict
        Dictionary with W-statistic and p-value
    """
    w_stat, p_value = stats.wilcoxon(data1, data2)
    
    return {
        'w_statistic': w_stat,
        'p_value': p_value
    }

def kruskal_wallis_test(groups: List[np.ndarray]) -> Dict[str, float]:
    """
    Perform Kruskal-Wallis H test (non-parametric version of ANOVA).
    
    Parameters:
    -----------
    groups : list of array-like
        List of samples to compare
    
    Returns:
    --------
    dict
        Dictionary with H-statistic and p-value
    """
    h_stat, p_value = stats.kruskal(*groups)
    
    return {
        'h_statistic': h_stat,
        'p_value': p_value
    }

def calculate_sample_size(power: float = 0.8, alpha: float = 0.05, 
                         effect_size: float = 0.5) -> int:
    """
    Calculate required sample size for hypothesis testing.
    
    Parameters:
    -----------
    power : float
        Desired statistical power (1 - beta)
    alpha : float
        Significance level
    effect_size : float
        Cohen's d effect size
    
    Returns:
    --------
    int
        Required sample size per group
    """
    # Using the formula: n = (2 * (Z_alpha + Z_beta)^2) / d^2
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

def bootstrap_ci(data: np.ndarray, statistic: callable, 
                 n_bootstrap: int = 1000, 
                 ci: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters:
    -----------
    data : array-like
        Input data
    statistic : callable
        Function to compute the statistic of interest
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval level
    
    Returns:
    --------
    tuple
        Lower and upper bounds of the confidence interval
    """
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate statistic
        bootstrap_stats.append(statistic(bootstrap_sample))
    
    # Calculate confidence interval bounds
    alpha = (1 - ci) / 2
    lower_bound = np.percentile(bootstrap_stats, 100 * alpha)
    upper_bound = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    
    return lower_bound, upper_bound

def vif_scores(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Variance Inflation Factor (VIF) for each feature.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame (numeric columns only)
    
    Returns:
    --------
    pandas.Series
        Series with VIF scores for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return pd.Series()
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                       for i in range(numeric_df.shape[1])]
    
    return vif_data.set_index("feature")["VIF"]

def test_homogeneity_of_variance(groups: List[np.ndarray]) -> Dict[str, float]:
    """
    Test for homogeneity of variance using Levene's test.
    
    Parameters:
    -----------
    groups : list of array-like
        List of samples to compare
    
    Returns:
    --------
    dict
        Dictionary with test statistic and p-value
    """
    stat, p_value = stats.levene(*groups)
    
    return {
        'levene_statistic': stat,
        'p_value': p_value
    }

def summary_statistics_by_group(df: pd.DataFrame, 
                              feature_col: str, 
                              group_col: str) -> pd.DataFrame:
    """
    Calculate summary statistics for a feature grouped by a categorical variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    feature_col : str
        Feature column to analyze
    group_col : str
        Grouping column
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with summary statistics by group
    """
    if feature_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    
    # Group by the categorical column and calculate statistics
    return df.groupby(group_col)[feature_col].agg([
        'count', 'mean', 'std', 'min', 'max',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'skew', 'kurt'
    ]).rename(columns={
        '<lambda_0>': 'q1',
        '<lambda_1>': 'q3',
        'kurt': 'kurtosis'
    })

def cross_validate_regression_metrics(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: List[str] = None
) -> Dict[str, List[float]]:
    """
    Perform cross-validation for regression models and return various metrics.
    
    Parameters:
    -----------
    model : estimator object
        Scikit-learn estimator with fit and predict methods
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : int
        Number of cross-validation folds
    scoring : list of str, optional
        List of scoring metrics to compute
    
    Returns:
    --------
    dict
        Dictionary with cross-validation scores for each metric
    """
    from sklearn.model_selection import cross_validate
    
    if scoring is None:
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    # Extract metrics, remove 'test_' prefix and convert negative scores
    results = {}
    for metric in scoring:
        score_key = f'test_{metric}'
        metric_name = metric.replace('neg_', '')
        
        if score_key in scores:
            if metric.startswith('neg_'):
                results[metric_name] = -scores[score_key]
            else:
                results[metric_name] = scores[score_key]
    
    # Add RMSE if MSE is available
    if 'mean_squared_error' in results:
        results['root_mean_squared_error'] = np.sqrt(results['mean_squared_error'])
    
    return results

def cross_validate_classification_metrics(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: List[str] = None
) -> Dict[str, List[float]]:
    """
    Perform cross-validation for classification models and return various metrics.
    
    Parameters:
    -----------
    model : estimator object
        Scikit-learn estimator with fit and predict methods
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : int
        Number of cross-validation folds
    scoring : list of str, optional
        List of scoring metrics to compute
    
    Returns:
    --------
    dict
        Dictionary with cross-validation scores for each metric
    """
    from sklearn.model_selection import cross_validate
    
    if scoring is None:
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    # Extract metrics, remove 'test_' prefix
    results = {}
    for metric in scoring:
        score_key = f'test_{metric}'
        metric_name = metric
        
        if score_key in scores:
            results[metric_name] = scores[score_key]
    
    return results
