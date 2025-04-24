"""
Main application file for the ML Unified Tool.
"""

import streamlit as st
from ui.sidebar import load_sidebar
from core.session import initialize_session
from pages import (
    project_setup,
    data_upload,
    data_preprocessing,
    exploratory_analysis,
    feature_engineering,
    model_training,
    model_evaluation,
    prediction
)
from ui.styles import apply_styles

# Set page config
st.set_page_config(
    page_title="ML Unified Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point."""
    # Initialize session
    session = initialize_session()
    
    # Apply theme styles
    apply_styles(session.theme, session.custom_theme)
    
    # Load sidebar
    sidebar = load_sidebar(session)
    
    # Define page callbacks
    page_callbacks = {
        "navigation_radio": lambda page: st.experimental_rerun(),
        "problem_type_radio": lambda problem_type: None,
        "theme_select": lambda theme: apply_styles(theme, session.custom_theme)
    }
    
    # Register callbacks
    from ui.sidebar import register_sidebar_callbacks
    register_sidebar_callbacks(sidebar, page_callbacks)
    
    # Render current page
    st.title(session.current_page)
    
    if session.current_page == "Project Setup":
        project_setup.render(session)
    elif session.current_page == "Data Upload":
        data_upload.render(session)
    elif session.current_page == "Data Preprocessing":
        data_preprocessing.render(session)
    elif session.current_page == "Exploratory Analysis":
        exploratory_analysis.render(session)
    elif session.current_page == "Feature Engineering":
        feature_engineering.render(session)
    elif session.current_page == "Model Training":
        model_training.render(session)
    elif session.current_page == "Model Evaluation":
        model_evaluation.render(session)
    elif session.current_page == "Prediction":
        prediction.render(session)
    
    # Add debug mode information if enabled
    if session.debug_mode:
        with st.expander("Debug Information", expanded=False):
            st.json({k: str(v) for k, v in session.__dict__.items() 
                    if k not in ['data', 'model', 'cache']})

if __name__ == "__main__":
    main()