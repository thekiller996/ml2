"""
Plugin manager for the ML Platform.
"""

import os
import sys
import importlib
import pkgutil
import inspect
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from importlib.machinery import ModuleSpec
import config
from plugins.hooks import get_available_hooks

class PluginManager:
    """
    Manager for loading and managing plugins.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Singleton pattern to ensure only one instance exists.
        """
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the plugin manager.
        """
        if not self._initialized:
            self.plugins = {}
            self.hooks = {}
            self._initialize_hooks()
            self._initialized = True
    
    def _initialize_hooks(self):
        """
        Initialize hook dictionaries with empty lists for all available hooks.
        """
        for hook_name in get_available_hooks():
            self.hooks[hook_name] = []
    
    def load_plugins(self, plugins_dir: Optional[str] = None):
        """
        Load all plugins from the specified directory.
        
        Args:
            plugins_dir: Directory to load plugins from
        """
        if not config.PLUGIN_ENABLED:
            logging.info("Plugins are disabled in config.")
            return
        
        # Set plugins directory
        if plugins_dir is None:
            plugins_dir = config.PLUGINS_DIR
        
        # Ensure plugins directory exists
        os.makedirs(plugins_dir, exist_ok=True)
        
        # Add plugins directory to Python path
        if plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)
        
        # Clear existing plugins and hooks
        self.plugins = {}
        self._initialize_hooks()
        
        # Load plugins from the directory
        self._load_plugins_from_dir(plugins_dir)
        
        # Also load built-in example plugins
        builtin_plugins_dir = os.path.join(os.path.dirname(__file__), 'examples')
        if os.path.isdir(builtin_plugins_dir):
            self._load_plugins_from_dir(builtin_plugins_dir)
        
        logging.info(f"Loaded {len(self.plugins)} plugins.")
    
    def _load_plugins_from_dir(self, plugins_dir):
        """
        Load plugins from a directory.
        
        Args:
            plugins_dir: Directory to load plugins from
        """
        if not os.path.isdir(plugins_dir):
            return
        
        # Find all Python modules in the plugins directory
        for _, plugin_name, is_pkg in pkgutil.iter_modules([plugins_dir]):
            try:
                # Skip __pycache__ and other non-plugin directories
                if plugin_name.startswith('_'):
                    continue
                
                # Import the module or package
                if is_pkg:
                    # For packages, import the main module
                    plugin = importlib.import_module(f"{plugin_name}.plugin")
                else:
                    # For modules, import directly
                    plugin = importlib.import_module(plugin_name)
                
                # Register the plugin if it has the required attributes
                if hasattr(plugin, 'PLUGIN_NAME') and hasattr(plugin, 'register_plugin'):
                    plugin_name = plugin.PLUGIN_NAME
                    
                    # Call the register function
                    plugin.register_plugin(self)
                    
                    # Store the plugin
                    self.plugins[plugin_name] = plugin
                    
                    logging.info(f"Loaded plugin: {plugin_name}")
            except Exception as e:
                logging.error(f"Error loading plugin {plugin_name}: {str(e)}")
    
    def register_hook(self, hook_name: str, callback: Callable, plugin_name: str):
        """
        Register a hook callback.
        
        Args:
            hook_name: Name of the hook
            callback: Callback function
            plugin_name: Name of the plugin registering the hook
        """
        if hook_name not in self.hooks:
            logging.warning(f"Unknown hook name: {hook_name}")
            self.hooks[hook_name] = []
        
        # Add callback to the hook
        self.hooks[hook_name].append({
            'callback': callback,
            'plugin_name': plugin_name
        })
        
        logging.debug(f"Registered hook {hook_name} for plugin {plugin_name}")
    
    def execute_hook(self, hook_name: str, **kwargs) -> List[Any]:
        """
        Execute all callbacks for a hook.
        
        Args:
            hook_name: Name of the hook
            **kwargs: Arguments to pass to the hook callbacks
        
        Returns:
            List of results from all hook callbacks
        """
        if hook_name not in self.hooks:
            return []
        
        results = []
        for handler in self.hooks[hook_name]:
            try:
                result = handler['callback'](**kwargs)
                results.append(result)
            except Exception as e:
                logging.error(f"Error executing hook {hook_name} for plugin {handler['plugin_name']}: {str(e)}")
                results.append(None)
        
        return results
    
    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: Name of the plugin
        
        Returns:
            Plugin module or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def get_plugins(self) -> Dict[str, Any]:
        """
        Get all loaded plugins.
        
        Returns:
            Dictionary of plugin names to plugin modules
        """
        return self.plugins
    
    def get_custom_pages(self) -> List[str]:
        """
        Get list of custom pages provided by plugins.
        
        Returns:
            List of custom page names
        """
        custom_pages = []
        for result in self.execute_hook('get_custom_pages'):
            if isinstance(result, list):
                custom_pages.extend(result)
        return custom_pages
