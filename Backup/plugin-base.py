"""
Base class for plugins in the ML Platform.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

class PluginBase(ABC):
    """
    Base class for ML Platform plugins.
    
    Plugins should inherit from this class and implement the required methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the plugin.
        
        Returns:
            Plugin name
        """
        pass
    
    @property
    def version(self) -> str:
        """
        Get the version of the plugin.
        
        Returns:
            Plugin version
        """
        return "1.0.0"
    
    @property
    def description(self) -> str:
        """
        Get the description of the plugin.
        
        Returns:
            Plugin description
        """
        return "No description provided."
    
    @property
    def author(self) -> str:
        """
        Get the author of the plugin.
        
        Returns:
            Plugin author
        """
        return "Unknown"
    
    def register(self, manager) -> None:
        """
        Register hooks with the plugin manager.
        
        This method should be overridden by plugins to register their hook callbacks.
        
        Args:
            manager: Plugin manager instance
        """
        pass
    
    def get_hooks(self) -> Dict[str, Callable]:
        """
        Get all hooks provided by this plugin.
        
        Returns:
            Dictionary mapping hook names to callback functions
        """
        # Get all methods that are hook handlers
        hooks = {}
        
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_hook_name'):
                hooks[attr._hook_name] = attr
        
        return hooks
    
    def register_all_hooks(self, manager) -> None:
        """
        Register all hooks with the plugin manager.
        
        Args:
            manager: Plugin manager instance
        """
        for hook_name, callback in self.get_hooks().items():
            manager.register_hook(hook_name, callback, self.name)

# Decorator to mark methods as hook handlers
def hook(hook_name: str):
    """
    Decorator to mark a method as a hook handler.
    
    Args:
        hook_name: Name of the hook
    
    Returns:
        Decorated method
    """
    def decorator(func):
        func._hook_name = hook_name
        return func
    return decorator
