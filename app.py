"""
Main entry point for the ML Platform application.
"""

import streamlit as st
from ui.sidebar import render_sidebar
from core.session import initialize_session
from pages import (
    project_setup,
    data_upload,
    exploratory_analysis,
    data_preprocessing,
    feature_engineering,
    model_training,
    model_evaluation,
    prediction
)
from plugins.plugin_manager import PluginManager

def main():
    # Set page config
    st.set_page_config(
        page_title="Unified ML Platform",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session()
    
    # Initialize plugin manager
    plugin_manager = PluginManager()
    plugin_manager.load_plugins()
    
    # Trigger plugin hook for application startup
    plugin_manager.execute_hook('on_app_start')
    
    # Render sidebar
    render_sidebar()
    
    # Determine which page to show based on session state
    page = st.session_state.get('current_page', 'Project Setup')
    
    # Trigger plugin hook before page render
    plugin_manager.execute_hook('before_page_render', page_name=page)
    
    # Render the appropriate page
    if page == "Project Setup":
        project_setup.render()
    elif page == "Data Upload":
        data_upload.render()
    elif page == "Exploratory Analysis":
        exploratory_analysis.render()
    elif page == "Data Preprocessing":
        data_preprocessing.render()
    elif page == "Feature Engineering":
        feature_engineering.render()
    elif page == "Model Training":
        model_training.render()
    elif page == "Model Evaluation":
        model_evaluation.render()
    elif page == "Prediction":
        prediction.render()
    else:
        # Check if any plugin has registered this page
        page_rendered = plugin_manager.execute_hook('render_custom_page', page_name=page)
        if not any(page_rendered):
            st.error(f"Unknown page: {page}")
    
    # Trigger plugin hook after page render
    plugin_manager.execute_hook('after_page_render', page_name=page)

if __name__ == "__main__":
    main()
