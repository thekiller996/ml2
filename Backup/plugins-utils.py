"""
Utility functions for the ML Platform plugin system.
"""

import os
import sys
import importlib
import inspect
import pkgutil
import logging
from typing import List, Dict, Any, Optional, Set, Callable, Type
from pathlib import Path

def discover_plugins(plugins_dir: str) -> List[str]:
    """
    Discover available plugins in the plugins directory.
    
    Args:
        plugins_dir: Directory to scan for plugins
    
    Returns:
        List of plugin names
    """
    if not os.path.isdir(plugins_dir):
        return []
    
    plugin_names = []
    
    # Find all Python modules in the plugins directory
    for _, name, is_pkg in pkgutil.iter_modules([plugins_dir]):
        # Skip __pycache__ and other non-plugin directories
        if name.startswith('_'):
            continue
            
        if is_pkg:
            # Look for plugin.py in the package
            plugin_file = os.path.join(plugins_dir, name, 'plugin.py')
            if os.path.isfile(plugin_file):
                plugin_names.append(name)
        else:
            # Check if the module has PLUGIN_NAME and register_plugin
            try:
                spec = importlib.util.find_spec(name, [plugins_dir])
                if spec:
                    plugin_names.append(name)
            except Exception as e:
                logging.warning(f"Error inspecting plugin module {name}: {str(e)}")
    
    return plugin_names

def validate_plugin(plugin_module: Any) -> bool:
    """
    Validate that a module is a valid plugin.
    
    Args:
        plugin_module: Module to validate
    
    Returns:
        True if the module is a valid plugin, False otherwise
    """
    # Check for required attributes
    has_name = hasattr(plugin_module, 'PLUGIN_NAME')
    has_register = hasattr(plugin_module, 'register_plugin')
    
    # Check register_plugin is callable
    is_callable = callable(getattr(plugin_module, 'register_plugin', None))
    
    return has_name and has_register and is_callable

def load_plugin_module(plugin_name: str, plugins_dir: str) -> Optional[Any]:
    """
    Load a plugin module.
    
    Args:
        plugin_name: Name of the plugin to load
        plugins_dir: Directory containing plugins
    
    Returns:
        Plugin module or None if loading failed
    """
    try:
        # Check if plugin is a package or module
        plugin_path = os.path.join(plugins_dir, plugin_name)
        
        if os.path.isdir(plugin_path) and os.path.isfile(os.path.join(plugin_path, 'plugin.py')):
            # It's a package with plugin.py
            module_name = f"{plugin_name}.plugin"
        else:
            # It's a module
            module_name = plugin_name
        
        # Add plugins directory to path if not already there
        if plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)
        
        # Import the module
        plugin_module = importlib.import_module(module_name)
        
        # Validate the plugin
        if validate_plugin(plugin_module):
            return plugin_module
        else:
            logging.warning(f"Plugin {plugin_name} is not valid (missing PLUGIN_NAME or register_plugin)")
            return None
    
    except Exception as e:
        logging.error(f"Error loading plugin {plugin_name}: {str(e)}")
        return None

def get_plugin_info(plugin_module: Any) -> Dict[str, Any]:
    """
    Get information about a plugin.
    
    Args:
        plugin_module: Plugin module
    
    Returns:
        Dictionary with plugin information
    """
    info = {
        'name': getattr(plugin_module, 'PLUGIN_NAME', 'Unknown'),
        'version': getattr(plugin_module, 'PLUGIN_VERSION', '1.0.0'),
        'description': getattr(plugin_module, 'PLUGIN_DESCRIPTION', ''),
        'author': getattr(plugin_module, 'PLUGIN_AUTHOR', 'Unknown'),
        'hooks': []
    }
    
    # Find hooks in the module
    for name, obj in inspect.getmembers(plugin_module):
        if inspect.isfunction(obj) and hasattr(obj, '_hook_name'):
            info['hooks'].append(obj._hook_name)
    
    return info

def is_plugin_enabled(plugin_name: str, config: Dict[str, Any]) -> bool:
    """
    Check if a plugin is enabled in the configuration.
    
    Args:
        plugin_name: Name of the plugin
        config: Configuration dictionary
    
    Returns:
        True if the plugin is enabled, False otherwise
    """
    # Check global plugin setting
    if not config.get('PLUGIN_ENABLED', True):
        return False
    
    # Check plugin-specific setting
    disabled_plugins = config.get('DISABLED_PLUGINS', [])
    
    return plugin_name not in disabled_plugins
