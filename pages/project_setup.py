"""
Project setup page for the ML Platform.
"""

import streamlit as st
from core.session import update_session_state
from core.constants import ML_TASKS
from ui.common import show_header, show_info, show_success
from plugins.plugin_manager import PluginManager

def render():
    """
    Render the project setup page.
    """
    show_header(
        "Project Setup",
        "Configure your machine learning project settings."
    )
    
    # Project information
    st.subheader("Project Information")
    
    project_name = st.text_input(
        "Project Name",
        value=st.session_state.get('project_name', ''),
        help="Enter a descriptive name for your project"
    )
    
    project_description = st.text_area(
        "Project Description",
        value=st.session_state.get('project_description', ''),
        help="Provide a brief description of the project"
    )
    
    # ML task selection
    st.subheader("Machine Learning Task")
    
    # Get extended tasks from plugins
    plugin_manager = PluginManager()
    extended_tasks = plugin_manager.execute_hook('extend_ml_tasks')
    
    # Combine built-in tasks with plugin tasks
    all_tasks = ML_TASKS.copy()
    for plugin_tasks in extended_tasks:
        if plugin_tasks:
            all_tasks.extend(plugin_tasks)
    
    ml_task = st.selectbox(
        "Select ML Task",
        options=all_tasks,
        index=all_tasks.index(st.session_state.get('ml_task', all_tasks[0])) if st.session_state.get('ml_task') in all_tasks else 0,
        help="Choose the type of machine learning task for this project"
    )
    
    # Plugins can add custom options to project setup
    plugin_manager.execute_hook('render_project_setup_options')
    
    # Save changes button
    if st.button("Save Project Settings", type="primary", use_container_width=True):
        update_session_state('project_name', project_name)
        update_session_state('project_description', project_description)
        update_session_state('ml_task', ml_task)
        
        # Let plugins save their custom settings
        plugin_manager.execute_hook('save_project_setup_options')
        
        # Show success message
        show_success("Project settings saved successfully!")
        
        # Navigate to data upload
        update_session_state('current_page', "Data Upload")
        st.experimental_rerun()
