"""
UI styling utilities for the ML Platform.
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_styles(
    theme: str = 'Light',
    custom_theme: Optional[Dict[str, str]] = None
) -> None:
    """Apply global styling for the application.
    
    Args:
        theme: Theme name ('Light', 'Dark', 'Custom')
        custom_theme: Dictionary with custom theme colors
    """
    # Define theme colors
    colors = get_theme_colors(theme, custom_theme)
    
    # Apply theme using custom CSS
    st.markdown(f"""
        <style>
        /* Base styling */
        .stApp {{
            background-color: {colors['background']};
            color: {colors['text']};
        }}
        
        /* Sidebar styling */
        .css-1d391kg, .css-1p05t8e {{
            background-color: {colors['sidebar_bg']};
        }}
        
        /* Headers styling */
        h1, h2, h3 {{
            color: {colors['header']};
        }}
        
        /* Button styling */
        .stButton>button {{
            background-color: {colors['primary']};
            color: {colors['button_text']};
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            transition: all 0.2s;
        }}
        .stButton>button:hover {{
            background-color: {colors['primary_dark']};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        /* Widget styling */
        div.stSlider, div.stSelectbox, div.stMultiselect {{
            background-color: {colors['widget_bg']};
            border-radius: 4px;
            padding: 1px;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {colors['expander_bg']};
            border-radius: 4px;
        }}
        
        /* Dataframe styling */
        .dataframe {{
            border-collapse: collapse;
        }}
        .dataframe th {{
            background-color: {colors['table_header']};
            color: {colors['table_header_text']};
            padding: 8px;
        }}
        .dataframe td {{
            padding: 8px;
            border-bottom: 1px solid {colors['table_border']};
        }}
        .dataframe tr:nth-child(even) {{
            background-color: {colors['table_row_even']};
        }}
        .dataframe tr:hover {{
            background-color: {colors['table_row_hover']};
        }}
        
        /* Card styling */
        div.card {{
            background-color: {colors['card_bg']};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }}
        div.card h3 {{
            margin-top: 0;
            color: {colors['card_header']};
        }}
        
        /* Footer styling */
        footer {{
            background-color: {colors['footer_bg']};
            padding: 8px;
            text-align: center;
            color: {colors['footer_text']};
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Log the theme application
    logger.info(f"Applied {theme} theme to application")

def get_theme_colors(
    theme: str = 'Light',
    custom_theme: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Get color palette for a specific theme.
    
    Args:
        theme: Theme name ('Light', 'Dark', 'Custom')
        custom_theme: Dictionary with custom theme colors
        
    Returns:
        Dictionary with theme colors
    """
    # Default light theme
    light_theme = {
        'primary': '#4CAF50',
        'primary_dark': '#388E3C',
        'secondary': '#2196F3',
        'background': '#FFFFFF',
        'sidebar_bg': '#F0F2F6',
        'text': '#333333',
        'header': '#2E7D32',
        'button_text': '#FFFFFF',
        'widget_bg': '#F8F8F8',
        'expander_bg': '#F0F2F6',
        'table_header': '#4CAF50',
        'table_header_text': '#FFFFFF',
        'table_border': '#E0E0E0',
        'table_row_even': '#F9F9F9',
        'table_row_hover': '#F0F7F0',
        'card_bg': '#FFFFFF',
        'card_header': '#2E7D32',
        'footer_bg': '#F0F2F6',
        'footer_text': '#666666'
    }
    
    # Dark theme
    dark_theme = {
        'primary': '#4CAF50',
        'primary_dark': '#388E3C',
        'secondary': '#2196F3',
        'background': '#1E1E1E',
        'sidebar_bg': '#262730',
        'text': '#E0E0E0',
        'header': '#4CAF50',
        'button_text': '#FFFFFF',
        'widget_bg': '#2A2A2A',
        'expander_bg': '#262730',
        'table_header': '#388E3C',
        'table_header_text': '#FFFFFF',
        'table_border': '#444444',
        'table_row_even': '#2A2A2A',
        'table_row_hover': '#323932',
        'card_bg': '#2A2A2A',
        'card_header': '#4CAF50',
        'footer_bg': '#262730',
        'footer_text': '#AAAAAA'
    }
    
    # Get base theme
    if theme == 'Light':
        colors = light_theme
    elif theme == 'Dark':
        colors = dark_theme
    elif theme == 'Custom' and custom_theme:
        # Start with light theme as base for custom
        colors = light_theme.copy()
        
        # Update with custom colors
        if 'primary_color' in custom_theme:
            colors['primary'] = custom_theme['primary_color']
            # Darken primary color for hover states
            colors['primary_dark'] = darken_color(custom_theme['primary_color'])
            colors['header'] = custom_theme['primary_color']
            colors['table_header'] = custom_theme['primary_color']
            colors['card_header'] = custom_theme['primary_color']
        
        if 'background_color' in custom_theme:
            colors['background'] = custom_theme['background_color']
            # Adjust sidebar and widget backgrounds based on main background
            colors['sidebar_bg'] = adjust_lightness(custom_theme['background_color'], 0.95)
            colors['widget_bg'] = adjust_lightness(custom_theme['background_color'], 0.97)
            colors['card_bg'] = custom_theme['background_color']
        
        if 'text_color' in custom_theme:
            colors['text'] = custom_theme['text_color']
            # Adjust button text to ensure readability
            colors['button_text'] = get_contrast_color(colors['primary'])
            colors['table_header_text'] = get_contrast_color(colors['table_header'])
    else:
        # Default to light theme
        colors = light_theme
        
    return colors

def darken_color(color: str, factor: float = 0.8) -> str:
    """Darken a hex color by a factor.
    
    Args:
        color: Hex color code
        factor: Darkening factor (0-1)
        
    Returns:
        Darkened hex color
    """
    color = color.lstrip('#')
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f'#{r:02x}{g:02x}{b:02x}'

def lighten_color(color: str, factor: float = 1.2) -> str:
    """Lighten a hex color by a factor.
    
    Args:
        color: Hex color code
        factor: Lightening factor (>1)
        
    Returns:
        Lightened hex color
    """
    color = color.lstrip('#')
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    r = min(255, int(r * factor))
    g = min(255, int(g * factor))
    b = min(255, int(b * factor))
    return f'#{r:02x}{g:02x}{b:02x}'

def adjust_lightness(color: str, factor: float) -> str:
    """Adjust lightness of a color.
    
    Args:
        color: Hex color code
        factor: Adjustment factor (<1 darkens, >1 lightens)
        
    Returns:
        Adjusted hex color
    """
    if factor < 1:
        return darken_color(color, factor)
    else:
        return lighten_color(color, factor)

def get_contrast_color(color: str) -> str:
    """Get a contrasting color (black or white) for readability.
    
    Args:
        color: Hex color code
        
    Returns:
        '#FFFFFF' or '#000000' for best contrast
    """
    color = color.lstrip('#')
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    # Calculate relative luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    # Return black for light colors, white for dark colors
    return '#000000' if luminance > 0.5 else '#FFFFFF'

def create_gradient(
    color1: str,
    color2: str,
    direction: str = 'to right'
) -> str:
    """Create CSS gradient string.
    
    Args:
        color1: First hex color code
        color2: Second hex color code
        direction: Gradient direction
        
    Returns:
        CSS gradient string
    """
    return f'linear-gradient({direction}, {color1}, {color2})'

def apply_custom_component_style(
    component_type: str,
    style_props: Dict[str, str]
) -> None:
    """Apply custom style to a specific component type.
    
    Args:
        component_type: CSS selector for the component
        style_props: Dictionary of CSS properties and values
    """
    css = f"{component_type} {{"
    for prop, value in style_props.items():
        css += f"{prop}: {value};"
    css += "}"
    
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)