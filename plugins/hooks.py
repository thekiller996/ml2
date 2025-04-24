"""
Hook definitions for the ML Platform plugin system.
"""

from typing import List, Dict, Any, Callable

def get_available_hooks() -> List[str]:
    """
    Get a list of all available hook names.
    
    Returns:
        List of hook names
    """
    return [
        # Application lifecycle hooks
        'on_app_start',
        'on_app_exit',
        
        # Page lifecycle hooks
        'before_page_render',
        'after_page_render',
        'render_custom_page',
        
        # UI extension hooks
        'render_project_setup_options',
        'save_project_setup_options',
        'extend_ml_tasks',
        
        # Data hooks
        'get_data_sources',
        'render_data_source_ui',
        'get_sample_datasets',
        'load_sample_dataset',
        
        # Analysis hooks
        'render_exploratory_analysis_tabs',
        'get_analysis_methods',
        'render_analysis_method_ui',
        
        # Preprocessing hooks
        'render_preprocessing_tabs',
        'get_preprocessors',
        'render_preprocessor_ui',
        'apply_preprocessor',
        
        # Feature engineering hooks
        'render_feature_engineering_tabs',
        'get_feature_engineering_methods',
        'render_feature_engineering_ui',
        'apply_feature_engineering',
        
        # Model hooks
        'get_additional_models',
        'render_model_training_tabs',
        'get_model_params',
        'after_model_training',
        'render_model_evaluation_tabs',
        
        # Prediction hooks
        'render_prediction_tabs',
        'before_prediction',
        'after_prediction',
        
        # Plugin system hooks
        'get_plugin_settings',
        'render_plugin_settings'
    ]

def register_hook_definitions():
    """
    Register hook function signatures and documentation.
    This is used for plugin development but not required at runtime.
    """
    hooks = {
        # Application lifecycle hooks
        'on_app_start': {
            'description': 'Called when the application starts',
            'params': {},
            'returns': 'None'
        },
        'on_app_exit': {
            'description': 'Called when the application exits',
            'params': {},
            'returns': 'None'
        },
        
        # Page lifecycle hooks
        'before_page_render': {
            'description': 'Called before a page is rendered',
            'params': {'page_name': 'Name of the page being rendered'},
            'returns': 'None'
        },
        'after_page_render': {
            'description': 'Called after a page is rendered',
            'params': {'page_name': 'Name of the page being rendered'},
            'returns': 'None'
        },
        'render_custom_page': {
            'description': 'Render a custom page',
            'params': {'page_name': 'Name of the page to render'},
            'returns': 'Boolean indicating if the page was rendered'
        },
        
        # UI extension hooks
        'render_project_setup_options': {
            'description': 'Render additional options in the project setup page',
            'params': {},
            'returns': 'None'
        },
        'save_project_setup_options': {
            'description': 'Save additional options from the project setup page',
            'params': {},
            'returns': 'None'
        },
        'extend_ml_tasks': {
            'description': 'Add additional ML task types',
            'params': {},
            'returns': 'List of task names to add'
        },
        
        # Data hooks
        'get_data_sources': {
            'description': 'Get additional data sources',
            'params': {},
            'returns': 'List of data source names'
        },
        'render_data_source_ui': {
            'description': 'Render UI for a data source',
            'params': {'source_name': 'Name of the data source'},
            'returns': 'DataFrame if data is loaded, None otherwise'
        },
        'get_sample_datasets': {
            'description': 'Get additional sample datasets',
            'params': {},
            'returns': 'List of sample dataset names'
        },
        'load_sample_dataset': {
            'description': 'Load a sample dataset',
            'params': {'dataset_name': 'Name of the dataset to load'},
            'returns': 'DataFrame with the loaded dataset'
        },
        
        # Analysis hooks
        'render_exploratory_analysis_tabs': {
            'description': 'Render additional tabs in exploratory analysis',
            'params': {'df': 'DataFrame being analyzed'},
            'returns': 'None'
        },
        'get_analysis_methods': {
            'description': 'Get additional analysis methods',
            'params': {},
            'returns': 'List of method descriptors'
        },
        'render_analysis_method_ui': {
            'description': 'Render UI for an analysis method',
            'params': {'method_name': 'Name of the method', 'df': 'DataFrame to analyze'},
            'returns': 'None'
        },
        
        # Preprocessing hooks
        'render_preprocessing_tabs': {
            'description': 'Render additional tabs in preprocessing',
            'params': {'df': 'DataFrame being processed'},
            'returns': 'None'
        },
        'get_preprocessors': {
            'description': 'Get additional preprocessors',
            'params': {},
            'returns': 'List of preprocessor descriptors'
        },
        'render_preprocessor_ui': {
            'description': 'Render UI for a preprocessor',
            'params': {'preprocessor_name': 'Name of the preprocessor', 'df': 'DataFrame to process'},
            'returns': '(DataFrame, metadata) tuple if processing is applied, None otherwise'
        },
        'apply_preprocessor': {
            'description': 'Apply a preprocessor',
            'params': {
                'preprocessor_name': 'Name of the preprocessor', 
                'df': 'DataFrame to process',
                'params': 'Parameters for the preprocessor'
            },
            'returns': 'Processed DataFrame'
        },
        
        # Feature engineering hooks
        'render_feature_engineering_tabs': {
            'description': 'Render additional tabs in feature engineering',
            'params': {'df': 'DataFrame being processed'},
            'returns': 'None'
        },
        'get_feature_engineering_methods': {
            'description': 'Get additional feature engineering methods',
            'params': {},
            'returns': 'List of method descriptors'
        },
        'render_feature_engineering_ui': {
            'description': 'Render UI for a feature engineering method',
            'params': {'method_name': 'Name of the method', 'df': 'DataFrame to process'},
            'returns': '(DataFrame, metadata) tuple if processing is applied, None otherwise'
        },
        'apply_feature_engineering': {
            'description': 'Apply a feature engineering method',
            'params': {
                'method_name': 'Name of the method', 
                'df': 'DataFrame to process',
                'params': 'Parameters for the method'
            },
            'returns': 'Processed DataFrame'
        },
        
        # Model hooks
        'get_additional_models': {
            'description': 'Get additional model types',
            'params': {'task': 'ML task type'},
            'returns': 'Dictionary mapping model names to class paths'
        },
        'render_model_training_tabs': {
            'description': 'Render additional tabs in model training',
            'params': {'df': 'DataFrame being used', 'task': 'ML task type'},
            'returns': 'None'
        },
        'get_model_params': {
            'description': 'Get parameter options for a model',
            'params': {'model_name': 'Name of the model', 'task': 'ML task type'},
            'returns': 'Dictionary of parameter options'
        },
        'after_model_training': {
            'description': 'Called after a model is trained',
            'params': {'model': 'Trained model', 'metadata': 'Training metadata'},
            'returns': 'None'
        },
        'render_model_evaluation_tabs': {
            'description': 'Render additional tabs in model evaluation',
            'params': {'model': 'Trained model', 'df': 'DataFrame being used'},
            'returns': 'None'
        },
        
        # Prediction hooks
        'render_prediction_tabs': {
            'description': 'Render additional tabs in prediction',
            'params': {'model': 'Trained model', 'df': 'DataFrame being used'},
            'returns': 'None'
        },
        'before_prediction': {
            'description': 'Called before making predictions',
            'params': {'model': 'Trained model', 'X': 'Features to predict on'},
            'returns': 'Modified features or None'
        },
        'after_prediction': {
            'description': 'Called after making predictions',
            'params': {'model': 'Trained model', 'predictions': 'Model predictions', 'X': 'Features used'},
            'returns': 'Modified predictions or None'
        },
        
        # Plugin system hooks
        'get_plugin_settings': {
            'description': 'Get settings for a plugin',
            'params': {},
            'returns': 'Dictionary of settings'
        },
        'render_plugin_settings': {
            'description': 'Render settings UI for a plugin',
            'params': {'plugin_name': 'Name of the plugin'},
            'returns': 'None'
        }
    }
    
    return hooks
