"""
Sidebar navigation for the ML Platform.
"""

import streamlit as st
from core.constants import PAGES
from core.session import get_session_state, update_session_state
from plugins.plugin_manager import PluginManager

def render_sidebar():
    """
    Render the application sidebar with navigation and current state information.
    """
    with st.sidebar:
        st.title("ML Platform")
        
        # Project info section
        if project_name := get_session_state('project_name'):
            st.markdown(f"### Project: {project_name}")
            
        if ml_task := get_session_state('ml_task'):
            st.markdown(f"**Task:** {ml_task}")
        
        st.divider()
        
        # Navigation section
        st.subheader("Navigation")
        
        # Standard pages navigation
        for page in PAGES:
            # Determine if page should be enabled based on application state
            disabled = _should_disable_page(page)
            
            if st.button(
                page, 
                disabled=disabled,
                key=f"nav_{page}",
                use_container_width=True
            ):
                update_session_state('current_page', page)
                st.experimental_rerun()
        
        # Plugin pages (if any)
        plugin_manager = PluginManager()
        plugin_pages = plugin_manager.get_custom_pages()
        
        if plugin_pages:
            st.divider()
            st.subheader("Plugin Pages")
            
            for page_name in plugin_pages:
                if st.button(
                    page_name,
                    key=f"nav_plugin_{page_name}",
                    use_container_width=True
                ):
                    update_session_state('current_page', page_name)
                    st.experimental_rerun()
        
        # Environment info
        st.divider()
        
        if df := get_session_state('df'):
            rows, cols = df.shape
            st.caption(f"Data: {rows} rows × {cols} columns")
            
        if model := get_session_state('current_model'):
            st.caption(f"Model: {model}")
            
        # Version info
        st.sidebar.markdown("---")
        st.sidebar.caption("ML Platform v2.0")

def _should_disable_page(page):
    """
    Determine if a page should be disabled based on the current state.
    
    Args:
        page: Page name to check
        
    Returns:
        Boolean indicating if the page should be disabled
    """
    # Initial page is always enabled
    if page == "Project Setup":
        return False
    
    # Data Upload requires Project Setup
    if page == "Data Upload":
        return not get_session_state('project_name')
    
    # All other pages require data to be loaded
    if get_session_state('df') is None:
        return True
        
    # Model Evaluation requires a trained model
    if page == "Model Evaluation":
        return not get_session_state('models')
    
    # Prediction requires at least one evaluated model
    if page == "Prediction":
        return not get_session_state('best_model')
        
    # By default, enable the page
    return False
