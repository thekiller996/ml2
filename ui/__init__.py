"""
UI components for the ML Platform.
"""

# Import UI components for easier access
from ui.common import (
    UIComponents,
    show_data_summary,
    create_filterable_dataframe
)

# Import sidebar functionality
from ui.sidebar import Sidebar, load_sidebar, register_sidebar_callbacks

# Import styling functions
from ui.styles import apply_styles, get_theme_colors

# Make these available at the package level
__all__ = [
    'UIComponents',
    'show_data_summary',
    'create_filterable_dataframe',
    'Sidebar',
    'load_sidebar',
    'register_sidebar_callbacks',
    'apply_styles',
    'get_theme_colors'
]