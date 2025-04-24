"""
CSS styles for the ML Platform UI.
"""

import streamlit as st

def apply_custom_styles():
    """
    Apply custom CSS styles to the Streamlit interface.
    """
    st.markdown("""
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header styling */
        h1 {
            color: #1E90FF;
            margin-bottom: 0.5rem;
        }
        
        h2 {
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        h3 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        
        /* Divider styling */
        hr {
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }
        
        /* Card-like sections */
        .card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #1E90FF;
        }
        
        /* Download button styling */
        .download-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #4CAF50;
            color: white;
            text-align: center;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
        }
        
        .download-button:hover {
            background-color: #45a049;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding-top: 2rem;
        }
        
        /* Table styling */
        .dataframe {
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_card_style(text: str):
    """
    Apply card-like styling to content.
    
    Args:
        text: HTML/markdown content to style
        
    Returns:
        Styled HTML
    """
    return f'<div class="card">{text}</div>'

def sidebar_title_style():
    """
    Get CSS for styling the sidebar title.
    
    Returns:
        CSS string
    """
    return """
    <style>
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #1E90FF;
        }
    </style>
    """
