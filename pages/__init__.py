"""
Pages for the ML Platform application.
"""

from pages.project_setup import render as project_setup
from pages.data_upload import render as data_upload
from pages.exploratory_analysis import render as exploratory_analysis
from pages.data_preprocessing import render as data_preprocessing
from pages.feature_engineering import render as feature_engineering
from pages.model_training import render as model_training
from pages.model_evaluation import render as model_evaluation
from pages.prediction import render as prediction

__all__ = [
    'project_setup',
    'data_upload',
    'exploratory_analysis',
    'data_preprocessing',
    'feature_engineering',
    'model_training',
    'model_evaluation',
    'prediction'
]
