"""
Core functionality for the ML Platform.
"""

# Import core components for easier access
from core.session import (
    SessionState, 
    initialize_session, 
    get_session, 
    save_session_state, 
    load_session_state
)

# Import constants
from core.constants import (
    PAGES,
    DATA_TYPES,
    MODEL_TYPES,
    PREPROCESSING_METHODS,
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    CLUSTERING_MODELS
)

# Rename functions to match expected imports elsewhere in the codebase
get_session_state = get_session
update_session_state = lambda **kwargs: get_session().update(**kwargs)

# Make these available at the package level
__all__ = [
    'SessionState',
    'initialize_session',
    'get_session_state',
    'update_session_state',
    'save_session_state',
    'load_session_state',
    'PAGES',
    'DATA_TYPES',
    'MODEL_TYPES',
    'PREPROCESSING_METHODS',
    'CLASSIFICATION_MODELS',
    'REGRESSION_MODELS',
    'CLUSTERING_MODELS'
]