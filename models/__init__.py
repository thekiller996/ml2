"""
Machine learning models functionality for the ML Platform.
"""

from models.classifier import (
    get_classifier,
    train_classifier,
    evaluate_classifier,
    predict_classifier
)

from models.regressor import (
    get_regressor,
    train_regressor,
    evaluate_regressor,
    predict_regressor
)

from models.clusterer import (
    get_clusterer,
    train_clusterer,
    evaluate_clusterer,
    predict_clusterer
)

from models.evaluation import (
    split_data,
    cross_validate,
    confusion_matrix,
    classification_report,
    regression_metrics,
    clustering_metrics,
    learning_curve,
    roc_curve,
    precision_recall_curve,
    feature_importance
)

from models.tuning import (
    tune_hyperparameters,
    grid_search,
    random_search,
    bayesian_optimization
)

__all__ = [
    # Classifiers
    'get_classifier',
    'train_classifier',
    'evaluate_classifier',
    'predict_classifier',
    
    # Regressors
    'get_regressor',
    'train_regressor',
    'evaluate_regressor',
    'predict_regressor',
    
    # Clusterers
    'get_clusterer',
    'train_clusterer',
    'evaluate_clusterer',
    'predict_clusterer',
    
    # Evaluation
    'split_data',
    'cross_validate',
    'confusion_matrix',
    'classification_report',
    'regression_metrics',
    'clustering_metrics',
    'learning_curve',
    'roc_curve',
    'precision_recall_curve',
    'feature_importance',
    
    # Hyperparameter tuning
    'tune_hyperparameters',
    'grid_search',
    'random_search',
    'bayesian_optimization'
]
