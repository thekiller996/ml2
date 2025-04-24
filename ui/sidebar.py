import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
import os
import logging
from pathlib import Path
import json

# Import project modules
from core.session import SessionState
from core.constants import PAGES, DATA_TYPES, MODEL_TYPES
from ui.styles import apply_styles
from utils.misc import get_version_info

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Sidebar:
    """Class for managing the sidebar UI elements"""
    
    def __init__(self, session: SessionState):
        """Initialize the sidebar.
        
        Args:
            session: The application session state
        """
        self.session = session
        self.current_page = None
        self.callbacks = {}
    
    def register_callback(self, element_id: str, callback_func: Callable) -> None:
        """Register a callback function for a sidebar element.
        
        Args:
            element_id: The ID of the element
            callback_func: The callback function to register
        """
        self.callbacks[element_id] = callback_func
    
    def render(self) -> None:
        """Render the sidebar UI elements."""
        with st.sidebar:
            self._render_logo()
            self._render_navigation()
            self._render_dataset_selector()
            self._render_mode_selector()
            self._render_settings()
            self._render_help()
            self._render_about()
    
    def _render_logo(self) -> None:
        """Render the application logo."""
        st.image("assets/logo.png", width=200, use_column_width=True)
        
        # Get version info
        version_info = get_version_info()
        st.markdown(f"<h4 style='text-align: center;'>ML Unified Tool v{version_info['version']}</h4>", unsafe_allow_html=True)
        
        st.markdown("---")
    
    def _render_navigation(self) -> None:
        """Render the navigation menu."""
        st.subheader("Navigation")
        
        # Get available pages based on current state
        available_pages = self._get_available_pages()
        
        # Create radio buttons for navigation
        selected_page = st.radio(
            "Select Page",
            options=available_pages,
            index=self._get_page_index(self.session.current_page, available_pages),
            key="navigation_radio"
        )
        
        # Only change page if selection changed
        if selected_page != self.session.current_page:
            self.session.current_page = selected_page
            
            # Clear temporary storage for new page
            self.session.temp_storage = {}
            
            # Trigger callback if registered
            if "navigation_radio" in self.callbacks:
                self.callbacks["navigation_radio"](selected_page)
        
        st.markdown("---")
    
    def _get_available_pages(self) -> List[str]:
        """Get list of available pages based on current state.
        
        Returns:
            List of available page names
        """
        # All pages are available by default
        available_pages = list(PAGES.keys())
        
        # Restrict pages if no data is loaded
        if self.session.data is None:
            restricted_pages = [
                "Model Training", "Model Evaluation", "Feature Engineering",
                "Exploratory Analysis", "Prediction", "Data Preprocessing"
            ]
            available_pages = [page for page in available_pages if page not in restricted_pages]
        
        # Further restrict if no model is trained
        if self.session.model is None:
            restricted_pages = ["Model Evaluation", "Prediction"]
            available_pages = [page for page in available_pages if page not in restricted_pages]
        
        return available_pages
    
    def _get_page_index(self, current_page: str, available_pages: List[str]) -> int:
        """Get the index of the current page in the available pages list.
        
        Args:
            current_page: The current page name
            available_pages: List of available pages
            
        Returns:
            Index of the current page, or 0 if not found
        """
        try:
            return available_pages.index(current_page)
        except (ValueError, TypeError):
            return 0
    
    def _render_dataset_selector(self) -> None:
        """Render the dataset selector."""
        if self.session.current_page != "Project Setup":
            st.subheader("Dataset")
            
            # Show current dataset info if available
            if self.session.data is not None:
                df = self.session.data
                st.success(f"Loaded: {self.session.dataset_name}")
                st.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                
                # Add option to view sample
                if st.button("View Sample", key="view_sample_button"):
                    self.session.show_dataset_sample = not self.session.show_dataset_sample
                
                # Add option to change dataset
                if st.button("Change Dataset", key="change_dataset_button"):
                    # Trigger callback if registered
                    if "change_dataset_button" in self.callbacks:
                        self.callbacks["change_dataset_button"]()
                    
                    # Navigate to Data Upload page
                    self.session.current_page = "Data Upload"
                    # Clear data and model
                    self.session.data = None
                    self.session.model = None
                    self.session.dataset_name = None
                    self.session.target_column = None
                    self.session.feature_columns = []
                    st.experimental_rerun()
            else:
                st.warning("No dataset loaded")
                
                # Add button to load data
                if st.button("Load Dataset", key="load_dataset_button"):
                    # Navigate to Data Upload page
                    self.session.current_page = "Data Upload"
                    st.experimental_rerun()
            
            st.markdown("---")
    
    def _render_mode_selector(self) -> None:
        """Render the ML problem type selector."""
        if self.session.data is not None:
            st.subheader("ML Problem Type")
            
            # Create radio buttons for problem type
            selected_problem = st.radio(
                "Select problem type",
                options=list(MODEL_TYPES.keys()),
                index=list(MODEL_TYPES.keys()).index(self.session.problem_type),
                key="problem_type_radio"
            )
            
            # Only update if selection changed
            if selected_problem != self.session.problem_type:
                self.session.problem_type = selected_problem
                
                # Clear model when changing problem type
                self.session.model = None
                
                # Trigger callback if registered
                if "problem_type_radio" in self.callbacks:
                    self.callbacks["problem_type_radio"](selected_problem)
            
            st.markdown("---")
    
    def _render_settings(self) -> None:
        """Render the settings section."""
        with st.expander("Settings", expanded=False):
            # Theme selection
            st.subheader("UI Theme")
            themes = ["Light", "Dark", "Custom"]
            selected_theme = st.selectbox(
                "Select theme",
                options=themes,
                index=themes.index(self.session.theme),
                key="theme_select"
            )
            
            # Update theme if changed
            if selected_theme != self.session.theme:
                self.session.theme = selected_theme
                apply_styles(selected_theme)
                
                # Trigger callback if registered
                if "theme_select" in self.callbacks:
                    self.callbacks["theme_select"](selected_theme)
            
            # Custom theme settings
            if selected_theme == "Custom":
                self.session.custom_theme["primary_color"] = st.color_picker(
                    "Primary Color",
                    value=self.session.custom_theme.get("primary_color", "#4CAF50"),
                    key="primary_color_picker"
                )
                
                self.session.custom_theme["background_color"] = st.color_picker(
                    "Background Color",
                    value=self.session.custom_theme.get("background_color", "#FFFFFF"),
                    key="bg_color_picker"
                )
                
                self.session.custom_theme["text_color"] = st.color_picker(
                    "Text Color",
                    value=self.session.custom_theme.get("text_color", "#000000"),
                    key="text_color_picker"
                )
                
                # Apply custom theme
                apply_styles("Custom", self.session.custom_theme)
            
            # Performance settings
            st.subheader("Performance")
            self.session.use_cache = st.checkbox(
                "Use cache for computations",
                value=self.session.use_cache,
                key="cache_checkbox"
            )
            
            # Debug mode
            self.session.debug_mode = st.checkbox(
                "Debug mode",
                value=self.session.debug_mode,
                key="debug_checkbox"
            )
            
            # Export settings
            st.subheader("Export Settings")
            export_formats = ["CSV", "Excel", "JSON", "Pickle"]
            self.session.export_format = st.selectbox(
                "Default export format",
                options=export_formats,
                index=export_formats.index(self.session.export_format),
                key="export_format_select"
            )
            
            # Save settings button
            if st.button("Save Settings", key="save_settings_button"):
                self._save_settings()
                st.success("Settings saved successfully!")
    
    def _save_settings(self) -> None:
        """Save settings to file."""
        try:
            settings = {
                "theme": self.session.theme,
                "custom_theme": self.session.custom_theme,
                "use_cache": self.session.use_cache,
                "debug_mode": self.session.debug_mode,
                "export_format": self.session.export_format
            }
            
            # Create directory if it doesn't exist
            os.makedirs("settings", exist_ok=True)
            
            # Save settings to JSON file
            with open("settings/user_settings.json", "w") as f:
                json.dump(settings, f, indent=2)
            
            logger.info("Settings saved to file")
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            st.error(f"Error saving settings: {str(e)}")
    
    def _render_help(self) -> None:
        """Render the help section."""
        with st.expander("Help", expanded=False):
            st.markdown("""
            # Quick Help
            
            ## Navigation
            - Use the navigation menu to switch between different pages
            - Some pages require data to be loaded first
            
            ## Data Upload
            - Upload CSV, Excel, or other supported file formats
            - Select columns for features and target
            
            ## Preprocessing
            - Clean and transform your data before modeling
            - Handle missing values, outliers, and encoding
            
            ## Model Training
            - Select model type and algorithm
            - Configure hyperparameters
            - Train models on your data
            
            ## Need more help?
            Check the [Documentation](https://github.com/thekiller996/ml2/docs) or 
            [Report an Issue](https://github.com/thekiller996/ml2/issues)
            """)
    
    def _render_about(self) -> None:
        """Render the about section."""
        st.sidebar.markdown("---")
        
        # Show version info
        version_info = get_version_info()
        st.sidebar.markdown(f"v{version_info['version']} | {version_info['build_date']}")
        
        # Links to GitHub, etc.
        st.sidebar.markdown("[GitHub](https://github.com/thekiller996/ml2) | [Report Bug](https://github.com/thekiller996/ml2/issues)")
        
        # Add custom footer
        st.sidebar.markdown(
            "<div style='text-align: center; color: #888; font-size: 0.8em;'>"
            "© 2023 ML Unified Tool"
            "</div>",
            unsafe_allow_html=True
        )

# Helper functions
def load_sidebar(session: SessionState) -> Sidebar:
    """Load and render the sidebar.
    
    Args:
        session: The application session state
        
    Returns:
        Initialized sidebar object
    """
    sidebar = Sidebar(session)
    sidebar.render()
    return sidebar

def register_sidebar_callbacks(sidebar: Sidebar, callbacks: Dict[str, Callable]) -> None:
    """Register callbacks for sidebar elements.
    
    Args:
        sidebar: The sidebar instance
        callbacks: Dictionary mapping element IDs to callback functions
    """
    for element_id, callback_func in callbacks.items():
        sidebar.register_callback(element_id, callback_func)