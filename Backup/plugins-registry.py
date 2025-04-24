"""
Plugin registry for the ML Platform.
"""

from typing import Dict, List, Any, Optional, Set, Callable
import logging

class PluginRegistry:
    """
    Registry for plugin information and capabilities.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Singleton pattern implementation.
        """
        if cls._instance is None:
            cls._instance = super(PluginRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the plugin registry.
        """
        if not self._initialized:
            self.plugins = {}
            self.custom_pages = {}
            self.preprocessors = {}
            self.feature_engineers = {}
            self.models = {}
            self.data_sources = {}
            self.sample_datasets = {}
            self.visualizers = {}
            self._initialized = True
    
    def register_plugin(self, plugin_name: str, metadata: Dict[str, Any]) -> None:
        """
        Register a plugin with metadata.
        
        Args:
            plugin_name: Name of the plugin
            metadata: Plugin metadata
        """
        self.plugins[plugin_name] = metadata
        logging.info(f"Registered plugin: {plugin_name}")
    
    def register_custom_page(self, page_name: str, plugin_name: str) -> None:
        """
        Register a custom page.
        
        Args:
            page_name: Name of the page
            plugin_name: Name of the plugin that provides the page
        """
        self.custom_pages[page_name] = plugin_name
        logging.debug(f"Registered custom page: {page_name} from plugin {plugin_name}")
    
    def register_preprocessor(self, preprocessor: Dict[str, Any]) -> None:
        """
        Register a preprocessor.
        
        Args:
            preprocessor: Preprocessor information
        """
        self.preprocessors[preprocessor['name']] = preprocessor
        logging.debug(f"Registered preprocessor: {preprocessor['name']}")
    
    def register_feature_engineer(self, feature_engineer: Dict[str, Any]) -> None:
        """
        Register a feature engineering method.
        
        Args:
            feature_engineer: Feature engineering method information
        """
        self.feature_engineers[feature_engineer['name']] = feature_engineer
        logging.debug(f"Registered feature engineer: {feature_engineer['name']}")
    
    def register_model(self, model_info: Dict[str, Any], task_type: str) -> None:
        """
        Register a model.
        
        Args:
            model_info: Model information
            task_type: Type of ML task ('classification', 'regression', 'clustering', etc.)
        """
        if task_type not in self.models:
            self.models[task_type] = {}
        
        self.models[task_type][model_info['name']] = model_info
        logging.debug(f"Registered {task_type} model: {model_info['name']}")
    
    def register_data_source(self, source_name: str, plugin_name: str, handler: Callable) -> None:
        """
        Register a data source.
        
        Args:
            source_name: Name of the data source
            plugin_name: Name of the plugin that provides the data source
            handler: Function to handle data loading
        """
        self.data_sources[source_name] = {
            'plugin': plugin_name,
            'handler': handler
        }
        logging.debug(f"Registered data source: {source_name} from plugin {plugin_name}")
    
    def register_sample_dataset(self, dataset_name: str, plugin_name: str, handler: Callable) -> None:
        """
        Register a sample dataset.
        
        Args:
            dataset_name: Name of the sample dataset
            plugin_name: Name of the plugin that provides the dataset
            handler: Function to load the dataset
        """
        self.sample_datasets[dataset_name] = {
            'plugin': plugin_name,
            'handler': handler
        }
        logging.debug(f"Registered sample dataset: {dataset_name} from plugin {plugin_name}")
    
    def register_visualizer(self, visualizer: Dict[str, Any]) -> None:
        """
        Register a visualization.
        
        Args:
            visualizer: Visualization information
        """
        self.visualizers[visualizer['name']] = visualizer
        logging.debug(f"Registered visualizer: {visualizer['name']}")
    
    def get_plugins(self) -> Dict[str, Any]:
        """
        Get all registered plugins.
        
        Returns:
            Dictionary of plugin names to metadata
        """
        return self.plugins
    
    def get_custom_pages(self) -> List[str]:
        """
        Get all registered custom pages.
        
        Returns:
            List of custom page names
        """
        return list(self.custom_pages.keys())
    
    def get_preprocessors(self) -> List[Dict[str, Any]]:
        """
        Get all registered preprocessors.
        
        Returns:
            List of preprocessor information
        """
        return list(self.preprocessors.values())
    
    def get_feature_engineers(self) -> List[Dict[str, Any]]:
        """
        Get all registered feature engineering methods.
        
        Returns:
            List of feature engineering method information
        """
        return list(self.feature_engineers.values())
    
    def get_models(self, task_type: str = None) -> Dict[str, Any]:
        """
        Get registered models.
        
        Args:
            task_type: Type of ML task to filter by (optional)
        
        Returns:
            Dictionary of model names to model information
        """
        if task_type:
            return self.models.get(task_type, {})
        return self.models
    
    def get_data_sources(self) -> List[str]:
        """
        Get all registered data sources.
        
        Returns:
            List of data source names
        """
        return list(self.data_sources.keys())
    
    def get_sample_datasets(self) -> List[str]:
        """
        Get all registered sample datasets.
        
        Returns:
            List of sample dataset names
        """
        return list(self.sample_datasets.keys())
    
    def get_visualizers(self) -> List[Dict[str, Any]]:
        """
        Get all registered visualizers.
        
        Returns:
            List of visualizer information
        """
        return list(self.visualizers.values())
