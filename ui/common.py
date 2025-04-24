import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import tempfile
import uuid
import logging
import json
import time
from datetime import datetime

# Import project modules
from core.session import SessionState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UIComponents:
    """Class for common UI components used across the application"""
    
    @staticmethod
    def header(title: str, subtitle: Optional[str] = None, icon: Optional[str] = None) -> None:
        """Render a header with optional subtitle and icon.
        
        Args:
            title: The main title text
            subtitle: Optional subtitle text
            icon: Optional icon name (using Streamlit's icon set)
        """
        if icon:
            st.markdown(f"# :{icon}: {title}")
        else:
            st.markdown(f"# {title}")
        
        if subtitle:
            st.markdown(f"### {subtitle}")
        
        st.markdown("---")
    
    @staticmethod
    def card(
        title: str,
        content: Optional[str] = None,
        icon: Optional[str] = None,
        color: str = "#4CAF50",
        is_expanded: bool = False,
        key: Optional[str] = None
    ) -> bool:
        """Create an expandable card component.
        
        Args:
            title: Card title
            content: Card content (markdown)
            icon: Optional icon name
            color: Card color
            is_expanded: Whether the card is expanded by default
            key: Unique key for the component
            
        Returns:
            True if the card is expanded, False otherwise
        """
        # Generate key if not provided
        if key is None:
            key = f"card_{uuid.uuid4().hex[:8]}"
        
        # Create card header with colored style
        header = f"{icon} {title}" if icon else title
        card_header = f"<div style='padding: 10px; border-radius: 5px 5px 0 0; background-color: {color}; color: white;'><strong>{header}</strong></div>"
        
        st.markdown(card_header, unsafe_allow_html=True)
        
        # Create expandable content
        expanded = st.expander("", expanded=is_expanded)
        
        with expanded:
            if content:
                st.markdown(content)
        
        # Add some spacing after the card
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        
        return expanded.expanded
    
    @staticmethod
    def data_viewer(
        data: pd.DataFrame,
        title: str = "Data Preview",
        max_rows: int = 10,
        allow_column_config: bool = True,
        key: Optional[str] = None
    ) -> None:
        """Display a data preview component.
        
        Args:
            data: DataFrame to display
            title: Title for the data viewer
            max_rows: Maximum number of rows to display
            allow_column_config: Allow column configuration
            key: Unique key for the component
        """
        # Generate key if not provided
        if key is None:
            key = f"data_viewer_{uuid.uuid4().hex[:8]}"
        
        # Display data info
        st.subheader(title)
        st.markdown(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        
        # Display data statistics button
        show_stats = st.checkbox("Show data statistics", key=f"{key}_show_stats")
        if show_stats:
            with st.expander("Data Statistics", expanded=True):
                # Show general statistics
                st.subheader("General Statistics")
                buffer = io.StringIO()
                data.info(buf=buffer)
                st.text(buffer.getvalue())
                
                # Show numeric statistics
                st.subheader("Numeric Column Statistics")
                st.dataframe(data.describe())
                
                # Show memory usage
                st.subheader("Memory Usage")
                memory_usage = data.memory_usage(deep=True).sum()
                st.markdown(f"Total memory usage: **{memory_usage / (1024**2):.2f} MB**")
        
        # Add column selection for large dataframes
        columns_to_show = data.columns.tolist()
        if data.shape[1] > 10:
            with st.expander("Select columns to display", expanded=False):
                columns_to_show = st.multiselect(
                    "Columns",
                    options=data.columns.tolist(),
                    default=data.columns[:10].tolist(),
                    key=f"{key}_column_select"
                )
        
        # Display data with column configuration
        if allow_column_config:
            st.dataframe(
                data[columns_to_show].head(max_rows), 
                use_container_width=True,
                column_config={
                    col: st.column_config.Column(
                        f"{col} ({data[col].dtype})",
                        help=f"Column: {col}, Type: {data[col].dtype}"
                    )
                    for col in columns_to_show
                }
            )
        else:
            st.dataframe(data[columns_to_show].head(max_rows), use_container_width=True)
        
        # Add download button
        if st.button("Download data", key=f"{key}_download_btn"):
            UIComponents.download_dataframe(data, "data.csv")
    
    @staticmethod
    def notification(
        message: str,
        type: str = "info",
        icon: Optional[str] = None,
        dismissible: bool = True,
        key: Optional[str] = None
    ) -> None:
        """Display a notification message.
        
        Args:
            message: Notification message
            type: Notification type (info, success, warning, error)
            icon: Optional icon
            dismissible: Whether notification can be dismissed
            key: Unique key for the component
        """
        # Generate key if not provided
        if key is None:
            key = f"notification_{uuid.uuid4().hex[:8]}"
        
        # Set default icons based on type
        if icon is None:
            icon_map = {
                "info": "info",
                "success": "check_circle",
                "warning": "warning",
                "error": "error"
            }
            icon = icon_map.get(type, "info")
        
        # Create notification using the appropriate Streamlit method
        if type == "info":
            st.info(f":{icon}: {message}", icon=icon)
        elif type == "success":
            st.success(f":{icon}: {message}", icon=icon)
        elif type == "warning":
            st.warning(f":{icon}: {message}", icon=icon)
        elif type == "error":
            st.error(f":{icon}: {message}", icon=icon)
        else:
            st.info(f":{icon}: {message}", icon=icon)
    
    @staticmethod
    def interactive_chart(
        data: pd.DataFrame,
        chart_type: str = "scatter",
        title: str = "Interactive Chart",
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        color_col: Optional[str] = None,
        size_col: Optional[str] = None,
        category_col: Optional[str] = None,
        allow_config: bool = True,
        height: int = 500,
        key: Optional[str] = None
    ) -> Optional[go.Figure]:
        """Display an interactive chart with configuration options.
        
        Args:
            data: DataFrame with data to plot
            chart_type: Type of chart to display
            title: Chart title
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for color mapping
            size_col: Column for size mapping
            category_col: Column for categories/groups
            allow_config: Allow chart configuration
            height: Chart height
            key: Unique key for the component
            
        Returns:
            Plotly figure object if created, None otherwise
        """
        # Generate key if not provided
        if key is None:
            key = f"chart_{uuid.uuid4().hex[:8]}"
        
        st.subheader(title)
        
        # Show configuration if allowed
        if allow_config:
            with st.expander("Chart Configuration", expanded=True):
                # Create columns for config options
                col1, col2 = st.columns(2)
                
                # Chart type selection
                with col1:
                    chart_types = ["scatter", "line", "bar", "histogram", "box", "violin", "heatmap", "pie"]
                    chart_type = st.selectbox(
                        "Chart Type",
                        options=chart_types,
                        index=chart_types.index(chart_type) if chart_type in chart_types else 0,
                        key=f"{key}_chart_type"
                    )
                
                # Column selections based on chart type
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
                
                with col2:
                    if chart_type in ["scatter", "line", "bar"]:
                        x_col = st.selectbox(
                            "X-Axis",
                            options=data.columns.tolist(),
                            index=data.columns.get_loc(x_col) if x_col in data.columns else 0,
                            key=f"{key}_x_col"
                        )
                        
                    if chart_type in ["scatter", "line", "bar", "box", "violin"]:
                        y_col = st.selectbox(
                            "Y-Axis",
                            options=numeric_cols,
                            index=numeric_cols.index(y_col) if y_col in numeric_cols else 0,
                            key=f"{key}_y_col"
                        )
                
                # Additional options
                col3, col4 = st.columns(2)
                
                with col3:
                    if chart_type in ["scatter", "line", "bar", "box"]:
                        color_options = ["None"] + data.columns.tolist()
                        color_selected = st.selectbox(
                            "Color By",
                            options=color_options,
                            index=color_options.index(color_col) if color_col in color_options else 0,
                            key=f"{key}_color_col"
                        )
                        color_col = None if color_selected == "None" else color_selected
                    
                    if chart_type == "scatter":
                        size_options = ["None"] + numeric_cols
                        size_selected = st.selectbox(
                            "Size By",
                            options=size_options,
                            index=size_options.index(size_col) if size_col in size_options else 0,
                            key=f"{key}_size_col"
                        )
                        size_col = None if size_selected == "None" else size_selected
                
                with col4:
                    if chart_type in ["histogram", "box", "violin"]:
                        # Single column selection for these chart types
                        if chart_type == "histogram":
                            y_col = st.selectbox(
                                "Column",
                                options=numeric_cols,
                                index=numeric_cols.index(y_col) if y_col in numeric_cols else 0,
                                key=f"{key}_column"
                            )
                        
                        # Group by selection
                        category_options = ["None"] + categorical_cols
                        category_selected = st.selectbox(
                            "Group By",
                            options=category_options,
                            index=category_options.index(category_col) if category_col in category_options else 0,
                            key=f"{key}_category_col"
                        )
                        category_col = None if category_selected == "None" else category_selected
                
                # Additional chart specific options
                if chart_type == "histogram":
                    bins = st.slider("Bins", min_value=5, max_value=100, value=20, key=f"{key}_bins")
                
                if chart_type in ["scatter", "line"]:
                    show_trend = st.checkbox("Show Trend Line", value=False, key=f"{key}_trend")
                
                if chart_type == "heatmap":
                    # For heatmap, allow selection of rows, columns and values
                    x_col = st.selectbox(
                        "Rows",
                        options=data.columns.tolist(),
                        index=data.columns.get_loc(x_col) if x_col in data.columns else 0,
                        key=f"{key}_heatmap_rows"
                    )
                    
                    y_col = st.selectbox(
                        "Columns",
                        options=data.columns.tolist(),
                        index=data.columns.get_loc(y_col) if y_col in data.columns else min(1, len(data.columns)-1),
                        key=f"{key}_heatmap_cols"
                    )
                    
                    value_col = st.selectbox(
                        "Values",
                        options=numeric_cols,
                        index=0,
                        key=f"{key}_heatmap_values"
                    )
        
        # Create chart based on selected type and columns
        try:
            fig = None
            
            if chart_type == "scatter":
                fig = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    size=size_col,
                    title=title,
                    height=height,
                    trendline="ols" if "show_trend" in locals() and show_trend else None
                )
            
            elif chart_type == "line":
                fig = px.line(
                    data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=title,
                    height=height
                )
                
                if "show_trend" in locals() and show_trend:
                    fig.add_trace(
                        px.scatter(data, x=x_col, y=y_col, trendline="ols").data[1]
                    )
            
            elif chart_type == "bar":
                fig = px.bar(
                    data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=title,
                    height=height
                )
            
            elif chart_type == "histogram":
                fig = px.histogram(
                    data,
                    x=y_col,
                    color=category_col,
                    nbins=bins if "bins" in locals() else 20,
                    title=title,
                    height=height
                )
            
            elif chart_type == "box":
                fig = px.box(
                    data,
                    x=category_col,
                    y=y_col,
                    color=color_col,
                    title=title,
                    height=height
                )
            
            elif chart_type == "violin":
                fig = px.violin(
                    data,
                    x=category_col,
                    y=y_col,
                    color=category_col,
                    box=True,
                    title=title,
                    height=height
                )
            
            elif chart_type == "heatmap":
                # Create pivot table for heatmap
                pivot_data = data.pivot_table(
                    index=x_col,
                    columns=y_col,
                    values=value_col if "value_col" in locals() else numeric_cols[0],
                    aggfunc="mean"
                )
                
                fig = px.imshow(
                    pivot_data,
                    title=title,
                    height=height
                )
            
            elif chart_type == "pie":
                # For pie chart, use counts of categories
                counts = data[x_col].value_counts()
                fig = px.pie(
                    counts,
                    values=counts.values,
                    names=counts.index,
                    title=title,
                    height=height
                )
            
            # Display the figure
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add download button for the chart
                if st.button("Download Chart", key=f"{key}_download_chart"):
                    # Save chart as HTML and provide download link
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                    fig.write_html(temp_file.name)
                    
                    with open(temp_file.name, "rb") as file:
                        btn = st.download_button(
                            label="Download HTML",
                            data=file,
                            file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            key=f"{key}_download_html"
                        )
                    
                    # Clean up temporary file
                    os.unlink(temp_file.name)
                
                return fig
                
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            logger.error(f"Error creating chart: {str(e)}")
            return None
    
    @staticmethod
    def model_summary(
        model: Any,
        metrics: Dict[str, float],
        title: str = "Model Summary",
        include_params: bool = True,
        key: Optional[str] = None
    ) -> None:
        """Display a model summary component.
        
        Args:
            model: Trained model object
            metrics: Dictionary of metric names and values
            title: Title for the summary
            include_params: Whether to include model parameters
            key: Unique key for the component
        """
        # Generate key if not provided
        if key is None:
            key = f"model_summary_{uuid.uuid4().hex[:8]}"
        
        st.subheader(title)
        
        # Create columns for metrics and model info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance Metrics")
            
            # Create metrics display
            metrics_df = pd.DataFrame({
                "Metric": list(metrics.keys()),
                "Value": list(metrics.values())
            })
            
            # Format metric values
            metrics_df["Value"] = metrics_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
            
            # Display metrics table
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Add visual metrics display
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Determine color based on metric name
                    if "accuracy" in metric.lower() or "r2" in metric.lower():
                        # Accuracy-like metrics (higher is better)
                        color = "normal" if value >= 0.7 else ("off" if value < 0.5 else "warning")
                    elif "error" in metric.lower() or "loss" in metric.lower():
                        # Error-like metrics (lower is better)
                        color = "normal" if value <= 0.3 else ("off" if value > 0.5 else "warning")
                    else:
                        color = "normal"
                    
                    st.metric(
                        label=metric,
                        value=f"{value:.4f}" if isinstance(value, float) else value,
                        delta=None,
                        delta_color=color
                    )
        
        with col2:
            st.markdown("### Model Information")
            
            # Get model type
            model_type = type(model).__name__
            st.markdown(f"**Type:** {model_type}")
            
            # Show model parameters if requested
            if include_params:
                if hasattr(model, "get_params"):
                    params = model.get_params()
                    
                    with st.expander("Model Parameters", expanded=False):
                        # Convert parameters to DataFrame for display
                        params_df = pd.DataFrame({
                            "Parameter": list(params.keys()),
                            "Value": [str(v) for v in params.values()]
                        })
                        
                        st.dataframe(params_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Model parameters not available")
            
            # Show model timestamp if available
            if hasattr(model, "fit_time"):
                st.markdown(f"**Trained:** {model.fit_time}")
            elif hasattr(model, "_fit_time"):
                st.markdown(f"**Trained:** {model._fit_time}")
        
        # Add expandable section for additional information
        with st.expander("Additional Information", expanded=False):
            # Feature importances if available
            if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
                st.subheader("Feature Importances")
                
                # Create DataFrame of feature importances
                features = model.feature_names_in_
                importances = model.feature_importances_
                
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)
                
                # Display table of importances
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), ax=ax)
                ax.set_title("Top 15 Feature Importances")
                st.pyplot(fig)
            
            # Coefficients for linear models
            elif hasattr(model, "coef_") and hasattr(model, "feature_names_in_"):
                st.subheader("Model Coefficients")
                
                # Create DataFrame of coefficients
                features = model.feature_names_in_
                
                if len(model.coef_.shape) == 1:
                    # Single target case
                    coefs = model.coef_
                    coef_df = pd.DataFrame({
                        "Feature": features,
                        "Coefficient": coefs
                    }).sort_values("Coefficient", ascending=False)
                else:
                    # Multi-target case
                    coef_df = pd.DataFrame(
                        model.coef_,
                        columns=features
                    )
                    # Transpose for better display
                    coef_df = coef_df.T.reset_index()
                    coef_df.columns = ["Feature"] + [f"Target_{i}" for i in range(model.coef_.shape[0])]
                
                # Display coefficients table
                st.dataframe(coef_df, use_container_width=True, hide_index=True)
                
                # Plot coefficients
                if len(model.coef_.shape) == 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Coefficient", y="Feature", data=coef_df.head(15), ax=ax)
                    ax.set_title("Top 15 Feature Coefficients")
                    st.pyplot(fig)
    
    @staticmethod
    def file_uploader(
        allowed_types: List[str],
        label: str = "Upload File",
        help_text: Optional[str] = None,
        multiple_files: bool = False,
        key: Optional[str] = None
    ) -> Any:
        """Display a file uploader with enhanced UI.
        
        Args:
            allowed_types: List of allowed file extensions
            label: Label for the uploader
            help_text: Optional help text
            multiple_files: Allow multiple file uploads
            key: Unique key for the component
            
        Returns:
            Uploaded file(s) object
        """
        # Generate key if not provided
        if key is None:
            key = f"uploader_{uuid.uuid4().hex[:8]}"
        
        # Create a container with custom styling
        uploader_container = st.container()
        
        with uploader_container:
            # Add custom styling for the uploader area
            st.markdown(
                """
                <style>
                .uploadedFileData { display: none; }
                .stFileUploader > div:first-child { padding: 20px; border: 2px dashed #4CAF50; border-radius: 10px; }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Display label and help text
            st.markdown(f"### {label}")
            
            if help_text:
                st.markdown(help_text)
            
            # Format allowed types for display
            allowed_types_str = ", ".join(allowed_types)
            st.caption(f"Allowed file types: {allowed_types_str}")
            
            # Create the uploader
            uploaded_files = st.file_uploader(
                label="",
                type=allowed_types,
                accept_multiple_files=multiple_files,
                key=key
            )
            
            # Show upload information
            if uploaded_files:
                if multiple_files:
                    st.success(f"Uploaded {len(uploaded_files)} files")
                    
                    # Show file details
                    with st.expander("File Details", expanded=False):
                        for i, file in enumerate(uploaded_files):
                            st.text(f"File {i+1}: {file.name} ({file.size} bytes)")
                else:
                    st.success(f"Uploaded: {uploaded_files.name} ({uploaded_files.size} bytes)")
            
        return uploaded_files
    
    @staticmethod
    def progress_operation(
        function: Callable,
        message: str = "Processing...",
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None,
        success_message: str = "Operation completed successfully!",
        error_message: str = "An error occurred",
        key: Optional[str] = None
    ) -> Tuple[bool, Any]:
        """Execute a function with a progress indicator.
        
        Args:
            function: Function to execute
            message: Message to display during execution
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            success_message: Message to display on success
            error_message: Message to display on error
            key: Unique key for the component
            
        Returns:
            Tuple of (success boolean, function result or exception)
        """
        # Generate key if not provided
        if key is None:
            key = f"progress_{uuid.uuid4().hex[:8]}"
        
        # Initialize arguments if None
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        
        # Show progress message
        progress_container = st.empty()
        progress_container.info(message)
        
        # Show progress bar
        progress_bar = st.progress(0)
        
        result = None
        success = False
        error = None
        
        try:
            # Record start time
            start_time = time.time()
            
            # Execute function
            result = function(*args, **kwargs)
            
            # Update progress bar to complete
            progress_bar.progress(1.0)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Show success message
            progress_container.success(f"{success_message} ({execution_time:.2f} seconds)")
            
            success = True
            
        except Exception as e:
            # Show error message
            progress_container.error(f"{error_message}: {str(e)}")
            logger.error(f"Error in progress operation: {str(e)}", exc_info=True)
            
            error = e
        
        # Clear progress bar after completion
        time.sleep(0.5)  # Small delay to ensure messages are seen
        progress_bar.empty()
        
        # Return result or error
        return (success, result if success else error)
    
    @staticmethod
    def confirmation_dialog(
        title: str,
        message: str,
        confirm_button_text: str = "Confirm",
        cancel_button_text: str = "Cancel",
        key: Optional[str] = None
    ) -> bool:
        """Display a confirmation dialog.
        
        Args:
            title: Dialog title
            message: Dialog message
            confirm_button_text: Text for confirm button
            cancel_button_text: Text for cancel button
            key: Unique key for the component
            
        Returns:
            True if confirmed, False otherwise
        """
        # Generate key if not provided
        if key is None:
            key = f"confirm_{uuid.uuid4().hex[:8]}"
        
        st.markdown(f"### {title}")
        st.markdown(message)
        
        # Create columns for buttons
        col1, col2 = st.columns(2)
        
        # Confirm and cancel buttons
        with col1:
            confirm = st.button(confirm_button_text, key=f"{key}_confirm")
        
        with col2:
            cancel = st.button(cancel_button_text, key=f"{key}_cancel")
        
        # Return result
        if confirm:
            return True
        elif cancel:
            return False
        else:
            # No button pressed yet
            return None
    
    @staticmethod
    def tabs_container(
        tab_names: List[str],
        tab_icons: Optional[List[str]] = None,
        key: Optional[str] = None
    ) -> List[Any]:
        """Create a tabbed container.
        
        Args:
            tab_names: List of tab names
            tab_icons: Optional list of tab icons
            key: Unique key for the component
            
        Returns:
            List of tab objects
        """
        # Generate key if not provided
        if key is None:
            key = f"tabs_{uuid.uuid4().hex[:8]}"
        
        # Create tab labels with icons if provided
        if tab_icons and len(tab_icons) == len(tab_names):
            tab_labels = [f":{icon}: {name}" for name, icon in zip(tab_names, tab_icons)]
        else:
            tab_labels = tab_names
        
        # Create tabs
        tabs = st.tabs(tab_labels)
        
        return tabs
    
    @staticmethod
    def json_editor(
        data: Dict,
        title: str = "Edit JSON",
        height: int = 300,
        key: Optional[str] = None
    ) -> Dict:
        """Create an editable JSON editor.
        
        Args:
            data: Dictionary to edit
            title: Editor title
            height: Editor height
            key: Unique key for the component
            
        Returns:
            Edited dictionary
        """
        # Generate key if not provided
        if key is None:
            key = f"json_editor_{uuid.uuid4().hex[:8]}"
        
        st.subheader(title)
        
        # Convert dictionary to JSON string
        json_str = json.dumps(data, indent=2)
        
        # Create text area for editing
        edited_str = st.text_area(
            "Edit JSON data below:",
            value=json_str,
            height=height,
            key=key
        )
        
        # Try to parse edited JSON
        try:
            edited_data = json.loads(edited_str)
            st.success("Valid JSON")
            return edited_data
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {str(e)}")
            return data  # Return original data if invalid
    
    @staticmethod
    def download_dataframe(
        df: pd.DataFrame,
        filename: str = "data.csv",
        file_format: str = "csv",
        key: Optional[str] = None
    ) -> None:
        """Create a download button for a DataFrame.
        
        Args:
            df: DataFrame to download
            filename: Download filename
            file_format: File format (csv, excel, json)
            key: Unique key for the component
        """
        # Generate key if not provided
        if key is None:
            key = f"download_{uuid.uuid4().hex[:8]}"
        
        # Prepare data in specified format
        if file_format == "csv":
            file_data = df.to_csv(index=False)
            mime_type = "text/csv"
            filename = filename if filename.endswith(".csv") else f"{filename}.csv"
        elif file_format == "excel":
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            file_data = buffer.getvalue()
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = filename if filename.endswith(".xlsx") else f"{filename}.xlsx"
        elif file_format == "json":
            file_data = df.to_json(orient="records", indent=2)
            mime_type = "application/json"
            filename = filename if filename.endswith(".json") else f"{filename}.json"
        else:
            st.error(f"Unsupported file format: {file_format}")
            return
        
        # Create download button
        st.download_button(
            label=f"Download {file_format.upper()}",
            data=file_data,
            file_name=filename,
            mime=mime_type,
            key=key
        )
    
    @staticmethod
    def collapsible_container(
        title: str,
        expanded: bool = False,
        key: Optional[str] = None
    ) -> Any:
        """Create a collapsible container.
        
        Args:
            title: Container title
            expanded: Whether the container is expanded by default
            key: Unique key for the component
            
        Returns:
            Expander object
        """
        # Generate key if not provided
        if key is None:
            key = f"container_{uuid.uuid4().hex[:8]}"
        
        return st.expander(title, expanded=expanded)
    
    @staticmethod
    def copyable_code(
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        show_copy_button: bool = True,
        key: Optional[str] = None
    ) -> None:
        """Display code with a copy button.
        
        Args:
            code: Code to display
            language: Programming language
            title: Optional title
            show_copy_button: Whether to show copy button
            key: Unique key for the component
        """
        # Generate key if not provided
        if key is None:
            key = f"code_{uuid.uuid4().hex[:8]}"
        
        if title:
            st.markdown(f"**{title}**")
        
        # Display code
        st.code(code, language=language)
        
        # Add copy button
        if show_copy_button:
            if st.button("Copy to clipboard", key=key):
                # Use JavaScript to copy to clipboard
                st.markdown(
                    f"""
                    <script>
                    navigator.clipboard.writeText(`{code}`);
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                st.success("Copied to clipboard!")
    
    @staticmethod
    def action_button(
        label: str,
        icon: Optional[str] = None,
        help_text: Optional[str] = None,
        type: str = "primary",
        key: Optional[str] = None
    ) -> bool:
        """Create a styled action button.
        
        Args:
            label: Button label
            icon: Optional icon
            help_text: Optional help text
            type: Button type (primary, secondary, success, danger)
            key: Unique key for the component
            
        Returns:
            True if button was clicked, False otherwise
        """
        # Generate key if not provided
        if key is None:
            key = f"button_{uuid.uuid4().hex[:8]}"
        
        # Button styles
        button_styles = {
            "primary": {"bg": "#4CAF50", "fg": "white"},
            "secondary": {"bg": "#9E9E9E", "fg": "white"},
            "success": {"bg": "#28a745", "fg": "white"},
            "danger": {"bg": "#dc3545", "fg": "white"}
        }
        
        # Get style for this button type
        style = button_styles.get(type, button_styles["primary"])
        
        # Create button label with icon if provided
        button_label = f":{icon}: {label}" if icon else label
        
        # Display help text if provided
        if help_text:
            st.caption(help_text)
        
        # Add custom styling
        st.markdown(
            f"""
            <style>
            div[data-testid="stButton"] button[kind="secondary"] {{
                background-color: {style["bg"]};
                color: {style["fg"]};
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Create button
        return st.button(button_label, key=key, type="secondary")
    
    @staticmethod
    def result_display(
        title: str,
        description: Optional[str] = None,
        value: Any = None,
        type: str = "text",
        icon: Optional[str] = None,
        key: Optional[str] = None
    ) -> None:
        """Display a result value with formatting.
        
        Args:
            title: Title for the result
            description: Optional description
            value: The value to display
            type: Type of value (text, number, json, dataframe, code)
            icon: Optional icon
            key: Unique key for the component
        """
        # Generate key if not provided
        if key is None:
            key = f"result_{uuid.uuid4().hex[:8]}"
        
        # Create container with styling
        st.markdown(
            f"""
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">
                <h3 style="margin-top: 0;">{f":{icon}: " if icon else ""}{title}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display description if provided
        if description:
            st.markdown(description)
        
        # Display value based on type
        if value is not None:
            if type == "text":
                st.markdown(str(value))
            
            elif type == "number":
                st.metric(label="", value=value)
            
            elif type == "json":
                if isinstance(value, dict) or isinstance(value, list):
                    st.json(value)
                else:
                    st.write(value)
            
            elif type == "dataframe":
                if isinstance(value, pd.DataFrame):
                    st.dataframe(value, use_container_width=True)
                else:
                    st.warning("Value is not a DataFrame")
                    st.write(value)
            
            elif type == "code":
                language = "python"
                if isinstance(value, str):
                    UIComponents.copyable_code(value, language=language, key=f"{key}_code")
                else:
                    UIComponents.copyable_code(str(value), language=language, key=f"{key}_code")
            
            elif type == "image":
                st.image(value, use_column_width=True)
            
            else:
                st.write(value)

# Helper functions
def show_data_summary(data: pd.DataFrame) -> None:
    """Show a summary of DataFrame data.
    
    Args:
        data: DataFrame to summarize
    """
    st.subheader("Data Summary")
    
    # Show general information
    st.markdown(f"**Shape:** {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overview", "Statistics", "Structure"])
    
    with tab1:
        # Overview tab
        st.markdown("### Data Sample")
        st.dataframe(data.head(), use_container_width=True)
        
        # Column types
        st.markdown("### Column Data Types")
        dtypes_df = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes.astype(str),
            'Non-Null Count': data.count(),
            'Null Count': data.isna().sum(),
            'Null %': (data.isna().sum() / len(data) * 100).round(2)
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
    
    with tab2:
        # Statistics tab
        st.markdown("### Numerical Statistics")
        st.dataframe(data.describe().T, use_container_width=True)
        
        # Categorical statistics for non-numeric columns
        cat_cols = data.select_dtypes(exclude='number').columns
        if not cat_cols.empty:
            st.markdown("### Categorical Statistics")
            cat_stats = pd.DataFrame({
                'Column': cat_cols,
                'Unique Values': [data[col].nunique() for col in cat_cols],
                'Most Common': [data[col].value_counts().index[0] if not data[col].value_counts().empty else None for col in cat_cols],
                'Frequency': [data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0 for col in cat_cols],
                'Frequency %': [(data[col].value_counts().iloc[0] / len(data) * 100).round(2) if not data[col].value_counts().empty else 0 for col in cat_cols]
            })
            st.dataframe(cat_stats, use_container_width=True, hide_index=True)
    
    with tab3:
        # Structure tab
        st.markdown("### Memory Usage")
        memory_usage = data.memory_usage(deep=True)
        memory_total = memory_usage.sum()
        
        # Convert to MB for display
        memory_mb = memory_usage / (1024 ** 2)
        memory_total_mb = memory_total / (1024 ** 2)
        
        st.markdown(f"**Total Memory Usage:** {memory_total_mb:.2f} MB")
        
        # Memory usage by column
        memory_df = pd.DataFrame({
            'Column': list(memory_usage.index),
            'Memory (MB)': memory_mb.values,
            'Percentage': ((memory_usage / memory_total) * 100).values.round(2)
        })
        st.dataframe(memory_df, use_container_width=True, hide_index=True)
        
        # Create pie chart of memory usage
        fig = px.pie(
            memory_df,
            values='Memory (MB)',
            names='Column',
            title='Memory Usage by Column'
        )
        st.plotly_chart(fig, use_container_width=True)

def create_filterable_dataframe(
    data: pd.DataFrame,
    title: str = "Filterable Data",
    max_height: int = 500
) -> pd.DataFrame:
    """Create a filterable DataFrame display.
    
    Args:
        data: DataFrame to display
        title: Component title
        max_height: Maximum height for the dataframe
        
    Returns:
        Filtered DataFrame
    """
    st.subheader(title)
    
    # Create expandable filter section
    with st.expander("Filters", expanded=False):
        # Create filter for each column
        filters = {}
        
        for col in data.columns:
            col_type = data[col].dtype
            
            # Numeric column
            if pd.api.types.is_numeric_dtype(col_type):
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                
                # Use slider for filtering numeric column
                filter_range = st.slider(
                    f"Filter {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=(max_val - min_val) / 100 if max_val > min_val else 0.1
                )
                
                filters[col] = filter_range
            
            # Categorical column
            elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
                # Get unique values
                unique_values = data[col].unique()
                
                # If too many unique values, use text input instead
                if len(unique_values) > 10:
                    filter_text = st.text_input(f"Filter {col} (comma-separated values)")
                    
                    if filter_text:
                        filter_values = [v.strip() for v in filter_text.split(",")]
                        filters[col] = filter_values
                else:
                    filter_values = st.multiselect(
                        f"Filter {col}",
                        options=unique_values,
                        default=list(unique_values)
                    )
                    
                    filters[col] = filter_values
            
            # Date column
            elif pd.api.types.is_datetime64_dtype(col_type):
                min_date = data[col].min().date()
                max_date = data[col].max().date()
                
                # Use date input for filtering date column
                filter_start = st.date_input(f"Filter {col} (start)", value=min_date)
                filter_end = st.date_input(f"Filter {col} (end)", value=max_date)
                
                filters[col] = (filter_start, filter_end)
    
    # Apply filters to create filtered dataframe
    filtered_data = data.copy()
    
    for col, filter_val in filters.items():
        col_type = data[col].dtype
        
        # Numeric filter
        if pd.api.types.is_numeric_dtype(col_type):
            min_val, max_val = filter_val
            filtered_data = filtered_data[(filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)]
        
        # Categorical filter
        elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
            if filter_val and len(filter_val) < len(data[col].unique()):
                filtered_data = filtered_data[filtered_data[col].isin(filter_val)]
        
        # Date filter
        elif pd.api.types.is_datetime64_dtype(col_type):
            start_date, end_date = filter_val
            filtered_data = filtered_data[
                (filtered_data[col].dt.date >= start_date) & 
                (filtered_data[col].dt.date <= end_date)
            ]
    
    # Show filter summary
    if len(filtered_data) < len(data):
        st.success(f"Showing {len(filtered_data)} of {len(data)} rows after filtering")
    
    # Display filtered data
    st.dataframe(filtered_data, use_container_width=True, height=max_height)
    
    # Add download button for filtered data
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Filtered Data", key="download_filtered_data"):
            UIComponents.download_dataframe(filtered_data, "filtered_data.csv")
    
    with col2:
        if st.button("Copy to Clipboard", key="copy_filtered_data"):
            # Generate CSV string
            csv_str = filtered_data.to_csv(index=False)
            
            # Use JavaScript to copy to clipboard
            st.markdown(
                f"""
                <script>
                navigator.clipboard.writeText(`{csv_str}`);
                </script>
                """,
                unsafe_allow_html=True
            )
            st.success("Copied to clipboard!")
    
    # Return filtered dataframe for further use
    return filtered_data