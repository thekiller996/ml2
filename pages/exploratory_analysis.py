"""
Exploratory data analysis page for the ML Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.session import update_session_state, get_session_state
from ui.common import (
    show_header, 
    show_info, 
    create_tabs, 
    create_expander, 
    plot_correlation_matrix,
    display_dataframe
)
from data.explorer import (
    get_dataframe_info, 
    get_numeric_summary, 
    get_categorical_summary,
    analyze_missing_values,
    analyze_outliers,
    analyze_correlations
)
from plugins.plugin_manager import PluginManager

def render():
    """
    Render the exploratory data analysis page.
    """
    show_header(
        "Exploratory Data Analysis",
        "Explore and understand your dataset."
    )
    
    # Get DataFrame from session state
    df = get_session_state('df')
    
    if df is None:
        show_info("No data available. Please upload a dataset first.")
        return
    
    # Create tabs for different analyses
    tabs = create_tabs([
        "Overview", 
        "Statistics", 
        "Visualization", 
        "Missing Values", 
        "Correlations",
        "Target Analysis"
    ])
    
    # Overview tab
    with tabs[0]:
        render_overview_tab(df)
    
    # Statistics tab
    with tabs[1]:
        render_statistics_tab(df)
    
    # Visualization tab
    with tabs[2]:
        render_visualization_tab(df)
    
    # Missing values tab
    with tabs[3]:
        render_missing_values_tab(df)
    
    # Correlations tab
    with tabs[4]:
        render_correlations_tab(df)
    
    # Target analysis tab
    with tabs[5]:
        render_target_analysis_tab(df)
    
    # Let plugins add their own exploratory analysis tabs
    plugin_manager = PluginManager()
    plugin_manager.execute_hook('render_exploratory_analysis_tabs', df=df)

def render_overview_tab(df):
    """
    Render the overview tab with general dataset information.
    """
    st.subheader("Dataset Overview")
    
    # Display basic info
    rows, cols = df.shape
    st.write(f"**Dimensions:** {rows} rows × {cols} columns")
    
    # Display memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    st.write(f"**Memory Usage:** {memory_usage / 1e6:.2f} MB")
    
    # Display column types
    dtypes = df.dtypes.value_counts()
    st.write("**Column Data Types:**")
    for dtype, count in dtypes.items():
        st.write(f"- {dtype}: {count} columns")
    
    # Display DataFrame
    st.subheader("Data Preview")
    display_dataframe(df)
    
    # Column information
    with create_expander("Column Information", expanded=False):
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Set target column
    st.subheader("Target Variable")
    
    # Use numeric columns for classification/regression and all columns for other tasks
    ml_task = get_session_state('ml_task')
    
    if ml_task in ["Classification", "Regression"]:
        target_options = df.columns.tolist()
    else:
        target_options = ["None"] + df.columns.tolist()
    
    current_target = get_session_state('target_column')
    if current_target not in target_options and current_target is not None:
        target_options.append(current_target)
    
    target_column = st.selectbox(
        "Select Target Column",
        options=target_options,
        index=target_options.index(current_target) if current_target in target_options else 0,
        help="Choose the column you want to predict"
    )
    
    if target_column != "None":
        update_session_state('target_column', target_column)
    else:
        update_session_state('target_column', None)
    
    # Set ID column
    st.subheader("ID Column")
    
    id_options = ["None"] + df.columns.tolist()
    current_id = get_session_state('id_column')
    if current_id not in id_options and current_id is not None:
        id_options.append(current_id)
    
    id_column = st.selectbox(
        "Select ID Column",
        options=id_options,
        index=id_options.index(current_id) if current_id in id_options else 0,
        help="Choose the column that uniquely identifies each row (if any)"
    )
    
    if id_column != "None":
        update_session_state('id_column', id_column)
    else:
        update_session_state('id_column', None)

def render_statistics_tab(df):
    """
    Render the statistics tab with descriptive statistics.
    """
    st.subheader("Descriptive Statistics")
    
    # Create subtabs for numeric and categorical
    stat_tabs = create_tabs(["Numeric Columns", "Categorical Columns"])
    
    # Numeric statistics
    with stat_tabs[0]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.write(f"**Number of numeric columns:** {len(numeric_cols)}")
            
            # Column selector
            selected_numeric = st.multiselect(
                "Select Numeric Columns",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                help="Choose columns to display statistics for"
            )
            
            if selected_numeric:
                # Calculate statistics
                num_stats = get_numeric_summary(df, selected_numeric)
                st.dataframe(num_stats, use_container_width=True)
                
                # Show histograms
                if st.checkbox("Show Histograms", value=True):
                    cols = st.columns(min(3, len(selected_numeric)))
                    for i, col_name in enumerate(selected_numeric):
                        with cols[i % 3]:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.histplot(df[col_name].dropna(), kde=True, ax=ax)
                            ax.set_title(f"Distribution of {col_name}")
                            st.pyplot(fig)
            else:
                st.info("Please select at least one numeric column.")
        else:
            st.info("No numeric columns found in the dataset.")
    
    # Categorical statistics
    with stat_tabs[1]:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            st.write(f"**Number of categorical columns:** {len(cat_cols)}")
            
            # Column selector
            selected_cat = st.multiselect(
                "Select Categorical Columns",
                options=cat_cols,
                default=cat_cols[:min(5, len(cat_cols))],
                help="Choose columns to display statistics for"
            )
            
            if selected_cat:
                # Calculate statistics
                cat_stats = get_categorical_summary(df, selected_cat)
                
                for col in selected_cat:
                    with create_expander(f"Statistics for {col}", expanded=True):
                        # Display basic stats
                        st.write(f"**Unique Values:** {df[col].nunique()}")
                        st.write(f"**Missing Values:** {df[col].isnull().sum()} ({df[col].isnull().mean()*100:.2f}%)")
                        
                        # Display value counts
                        if col in cat_stats:
                            st.write("**Value Counts:**")
                            st.dataframe(cat_stats[col]['frequency'], use_container_width=True)
                            
                            # Show bar chart
                            if st.checkbox(f"Show Bar Chart for {col}", value=True):
                                fig, ax = plt.subplots(figsize=(10, min(8, cat_stats[col]['frequency'].shape[0] * 0.4)))
                                cat_stats[col]['frequency'][:20].set_index(col).plot.barh(y='count', ax=ax)
                                ax.set_title(f"Value Counts for {col}")
                                ax.set_xlabel("Count")
                                st.pyplot(fig)
            else:
                st.info("Please select at least one categorical column.")
        else:
            st.info("No categorical columns found in the dataset.")

def render_visualization_tab(df):
    """
    Render the visualization tab with various plots.
    """
    st.subheader("Data Visualization")
    
    # Create subtabs for different visualization types
    viz_tabs = create_tabs([
        "Distribution", 
        "Scatter Plots", 
        "Box Plots", 
        "Pair Plots",
        "Custom Plot"
    ])
    
    # Distribution plots
    with viz_tabs[0]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col_name = st.selectbox(
                "Select Column for Distribution Plot",
                options=numeric_cols,
                index=0
            )
            
            plot_type = st.radio(
                "Plot Type",
                options=["Histogram", "KDE", "ECDF"],
                horizontal=True
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == "Histogram":
                sns.histplot(df[col_name].dropna(), kde=True, ax=ax)
            elif plot_type == "KDE":
                sns.kdeplot(df[col_name].dropna(), fill=True, ax=ax)
            else:  # ECDF
                sns.ecdfplot(df[col_name].dropna(), ax=ax)
                
            ax.set_title(f"{plot_type} of {col_name}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for distribution plots.")
    
    # Scatter plots
    with viz_tabs[1]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1 = st.selectbox(
                "X-Axis Column",
                options=numeric_cols,
                index=0
            )
            
            col2 = st.selectbox(
                "Y-Axis Column",
                options=numeric_cols,
                index=min(1, len(numeric_cols) - 1)
            )
            
            hue_col = st.selectbox(
                "Color By (Optional)",
                options=["None"] + df.columns.tolist(),
                index=0
            )
            
            hue = None if hue_col == "None" else df[hue_col]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df[col1], y=df[col2], hue=hue, ax=ax)
            ax.set_title(f"Scatter Plot: {col1} vs {col2}")
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numeric columns for scatter plots.")
    
    # Box plots
    with viz_tabs[2]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            numeric_col = st.selectbox(
                "Numeric Column",
                options=numeric_cols,
                index=0,
                key="boxplot_numeric"
            )
            
            cat_col = None
            if cat_cols:
                cat_col = st.selectbox(
                    "Categorical Column (Optional)",
                    options=["None"] + cat_cols,
                    index=0
                )
            
            orient = st.radio(
                "Orientation",
                options=["Vertical", "Horizontal"],
                horizontal=True
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if cat_col and cat_col != "None":
                # Limit to top categories if there are too many
                top_cats = df[cat_col].value_counts().head(10).index.tolist()
                plot_data = df[df[cat_col].isin(top_cats)]
                
                if orient == "Vertical":
                    sns.boxplot(x=cat_col, y=numeric_col, data=plot_data, ax=ax)
                else:
                    sns.boxplot(x=numeric_col, y=cat_col, data=plot_data, ax=ax)
            else:
                # Simple boxplot without categories
                if orient == "Vertical":
                    sns.boxplot(y=df[numeric_col], ax=ax)
                else:
                    sns.boxplot(x=df[numeric_col], ax=ax)
            
            ax.set_title(f"Box Plot of {numeric_col}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for box plots.")
    
    # Pair plots
    with viz_tabs[3]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            # Select columns
            selected_cols = st.multiselect(
                "Select Columns for Pair Plot",
                options=numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))],
                help="Choose 2-5 columns for best results"
            )
            
            hue_col = st.selectbox(
                "Color By (Optional)",
                options=["None"] + df.columns.tolist(),
                index=0,
                key="pairplot_hue"
            )
            
            if len(selected_cols) >= 2:
                with st.spinner("Generating pair plot..."):
                    # Limit samples for large datasets
                    sample_size = min(1000, len(df))
                    if len(df) > sample_size:
                        plot_data = df.sample(sample_size, random_state=42)
                    else:
                        plot_data = df
                    
                    # Create pair plot
                    hue = None if hue_col == "None" else hue_col
                    g = sns.pairplot(plot_data[selected_cols] if hue is None else plot_data[[*selected_cols, hue]], 
                                    hue=hue, diag_kind="kde", height=2.5)
                    g.fig.suptitle("Pair Plot", y=1.02)
                    st.pyplot(g.fig)
            else:
                st.info("Please select at least 2 columns for the pair plot.")
        else:
            st.info("Need at least 2 numeric columns for pair plots.")
    
    # Custom plot
    with viz_tabs[4]:
        st.write("Create a custom plot")
        
        plot_type = st.selectbox(
            "Plot Type",
            options=["Bar Chart", "Line Chart", "Violin Plot", "Count Plot", "Heat Map"]
        )
        
        if plot_type == "Bar Chart":
            # Bar chart options
            x_col = st.selectbox("X-Axis Column", options=df.columns.tolist(), key="bar_x")
            y_col = st.selectbox("Y-Axis Column (Optional)", options=["None"] + df.select_dtypes(include=[np.number]).columns.tolist(), key="bar_y")
            
            if y_col == "None":
                # Simple value counts
                value_counts = df[x_col].value_counts().sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts.plot.bar(ax=ax)
                ax.set_title(f"Count of {x_col}")
                ax.set_ylabel("Count")
            else:
                # Aggregate y by x
                agg_func = st.selectbox("Aggregation Function", options=["Mean", "Sum", "Count", "Min", "Max"])
                agg_map = {"Mean": "mean", "Sum": "sum", "Count": "count", "Min": "min", "Max": "max"}
                
                agg_data = df.groupby(x_col)[y_col].agg(agg_map[agg_func]).sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(10, 6))
                agg_data.plot.bar(ax=ax)
                ax.set_title(f"{agg_func} of {y_col} by {x_col}")
                ax.set_ylabel(f"{agg_func} of {y_col}")
            
            st.pyplot(fig)
            
        elif plot_type == "Line Chart":
            # Line chart options
            x_col = st.selectbox("X-Axis Column", options=df.columns.tolist(), key="line_x")
            y_col = st.selectbox("Y-Axis Column", options=df.select_dtypes(include=[np.number]).columns.tolist(), key="line_y")
            
            # Check if x is datetime
            is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])
            if not is_datetime and st.checkbox("Convert X to Datetime", value=False):
                try:
                    x_values = pd.to_datetime(df[x_col])
                    is_datetime = True
                except:
                    st.warning(f"Cannot convert {x_col} to datetime format.")
                    x_values = df[x_col]
            else:
                x_values = df[x_col]
            
            # Group by x if necessary
            if is_datetime:
                # Resample options
                if st.checkbox("Resample Time Series", value=False):
                    freq = st.selectbox("Frequency", options=["Day", "Week", "Month", "Quarter", "Year"])
                    freq_map = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}
                    
                    agg_func = st.selectbox("Aggregation Function", options=["Mean", "Sum", "Count", "Min", "Max"])
                    agg_map = {"Mean": "mean", "Sum": "sum", "Count": "count", "Min": "min", "Max": "max"}
                    
                    # Create time index for resampling
                    ts_data = df[[y_col]].copy()
                    ts_data.index = x_values
                    
                    # Resample and plot
                    resampled = ts_data.resample(freq_map[freq]).agg(agg_map[agg_func])
                    fig, ax = plt.subplots(figsize=(12, 6))
                    resampled.plot(ax=ax)
                    ax.set_title(f"{y_col} ({agg_func}) by {x_col} ({freq})")
                    st.pyplot(fig)
                else:
                    # Simple time series plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(x_values, df[y_col])
                    ax.set_title(f"{y_col} by {x_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)
            else:
                # For non-datetime x, group by x
                agg_func = st.selectbox("Aggregation Function", options=["Mean", "Sum", "Count", "Min", "Max"])
                agg_map = {"Mean": "mean", "Sum": "sum", "Count": "count", "Min": "min", "Max": "max"}
                
                agg_data = df.groupby(x_col)[y_col].agg(agg_map[agg_func]).sort_index()
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(agg_data.index, agg_data.values, marker='o')
                ax.set_title(f"{agg_func} of {y_col} by {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg_func} of {y_col}")
                st.pyplot(fig)
            
        elif plot_type == "Violin Plot":
            # Violin plot options
            x_col = st.selectbox("X-Axis Column (Categorical)", options=df.columns.tolist(), key="violin_x")
            y_col = st.selectbox("Y-Axis Column (Numeric)", options=df.select_dtypes(include=[np.number]).columns.tolist(), key="violin_y")
            
            # Limit to top categories if there are too many
            value_counts = df[x_col].value_counts()
            if len(value_counts) > 10:
                top_cats = value_counts.head(10).index.tolist()
                plot_data = df[df[x_col].isin(top_cats)]
                st.info(f"Limiting to top 10 categories for {x_col}")
            else:
                plot_data = df
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(x=x_col, y=y_col, data=plot_data, ax=ax)
            ax.set_title(f"Distribution of {y_col} by {x_col}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
        elif plot_type == "Count Plot":
            # Count plot options
            x_col = st.selectbox("Column", options=df.columns.tolist(), key="count_x")
            hue_col = st.selectbox("Group By (Optional)", options=["None"] + df.columns.tolist(), key="count_hue")
            
            # Limit to top categories if there are too many
            value_counts = df[x_col].value_counts()
            if len(value_counts) > 15:
                top_cats = value_counts.head(15).index.tolist()
                plot_data = df[df[x_col].isin(top_cats)]
                st.info(f"Limiting to top 15 categories for {x_col}")
            else:
                plot_data = df
            
            fig, ax = plt.subplots(figsize=(12, 6))
            hue = None if hue_col == "None" else hue_col
            sns.countplot(x=x_col, hue=hue, data=plot_data, ax=ax)
            ax.set_title(f"Count of {x_col}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
        elif plot_type == "Heat Map":
            # Heat map options
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.info("Need at least 2 numeric columns for a heat map.")
            else:
                # Select columns for heatmap
                selected_cols = st.multiselect(
                    "Select Columns for Heat Map",
                    options=numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))],
                    help="Choose columns to include in the correlation heat map"
                )
                
                if len(selected_cols) >= 2:
                    # Compute correlation matrix
                    corr_method = st.selectbox(
                        "Correlation Method",
                        options=["Pearson", "Spearman", "Kendall"],
                        index=0
                    )
                    
                    corr_matrix = df[selected_cols].corr(method=corr_method.lower())
                    
                    # Create heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                                cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
                    ax.set_title(f"{corr_method} Correlation Matrix")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Please select at least 2 columns for the heat map.")

def render_missing_values_tab(df):
    """
    Render the missing values analysis tab.
    """
    st.subheader("Missing Values Analysis")
    
    # Calculate missing values
    missing_cols, missing_rows = analyze_missing_values(df)
    
    # Display missing columns
    st.write("**Columns with Missing Values:**")
    if missing_cols[missing_cols['missing_count'] > 0].empty:
        st.write("No missing values found in any column.")
    else:
        st.dataframe(missing_cols[missing_cols['missing_count'] > 0], use_container_width=True)
        
        # Plot missing columns
        if st.checkbox("Show Missing Values Chart", value=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data = missing_cols[missing_cols['missing_count'] > 0].sort_values('missing_percentage', ascending=False)
            
            if not missing_data.empty:
                ax.barh(missing_data['column'], missing_data['missing_percentage'])
                ax.set_xlabel('Missing Percentage')
                ax.set_ylabel('Column')
                ax.set_title('Missing Values by Column')
                st.pyplot(fig)
    
    # Display missing rows
    st.write("**Rows with Missing Values:**")
    st.dataframe(missing_rows, use_container_width=True)
    
    # Visualize missing values pattern
    if st.checkbox("Show Missing Values Pattern", value=True):
        try:
            import missingno as msno
            
            st.write("**Missing Values Pattern:**")
            fig, ax = plt.subplots(figsize=(12, 8))
            msno.matrix(df, ax=ax)
            plt.title('Missing Values Pattern')
            st.pyplot(fig)
        except ImportError:
            st.info("Install 'missingno' package for better missing values visualization.")
            
            # Fallback heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
            plt.title('Missing Values Pattern')
            st.pyplot(fig)

def render_correlations_tab(df):
    """
    Render the correlations analysis tab.
    """
    st.subheader("Correlation Analysis")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
        return
    
    # Correlation method
    corr_method = st.selectbox(
        "Correlation Method",
        options=["Pearson", "Spearman", "Kendall"],
        index=0
    )
    
    # Show correlation matrix
    st.write("**Correlation Matrix:**")
    corr_matrix = df[numeric_cols].corr(method=corr_method.lower())
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1), use_container_width=True)
    
    # Show correlation heatmap
    st.write("**Correlation Heatmap:**")
    plot_correlation_matrix(df[numeric_cols], figsize=(10, 8))
    
    # Show strong correlations
    threshold = st.slider(
        "Correlation Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Show correlations above this threshold"
    )
    
    strong_corr = analyze_correlations(
        df, 
        columns=numeric_cols, 
        method=corr_method.lower(), 
        threshold=threshold
    )
    
    st.write(f"**Strong Correlations (|r| > {threshold}):**")
    if strong_corr.empty:
        st.write(f"No correlations found above threshold {threshold}.")
    else:
        st.dataframe(strong_corr, use_container_width=True)

def render_target_analysis_tab(df):
    """
    Render the target variable analysis tab.
    """
    st.subheader("Target Variable Analysis")
    
    target_column = get_session_state('target_column')
    
    if target_column is None or target_column not in df.columns:
        st.info("Please select a target column in the Overview tab.")
        return
    
    st.write(f"**Analyzing target column:** {target_column}")
    
    # Determine target type
    is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    is_categorical = not is_numeric or df[target_column].nunique() < 10
    
    # Display target distribution
    st.write("**Target Distribution:**")
    
    if is_categorical:
        # Categorical target
        value_counts = df[target_column].value_counts().reset_index()
        value_counts.columns = [target_column, 'Count']
        value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
        
        st.dataframe(value_counts, use_container_width=True)
        
        # Plot distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=target_column, data=df, ax=ax)
        ax.set_title(f"Distribution of {target_column}")
        ax.set_ylabel("Count")
        if len(value_counts) > 10:
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Class balance
        if value_counts.shape[0] > 1:
            st.write("**Class Balance:**")
            imbalance_ratio = value_counts['Count'].max() / value_counts['Count'].min()
            st.write(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 10:
                st.warning("The target variable has a high class imbalance. Consider using techniques like SMOTE, class weights, or resampling.")
    else:
        # Numeric target
        stats = df[target_column].describe()
        st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
        
        # Plot distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[target_column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {target_column}")
        st.pyplot(fig)
        
        # Normality test
        from scipy import stats as scipy_stats
        st.write("**Normality Test (Shapiro-Wilk):**")
        if len(df[target_column].dropna()) > 5000:
            # Shapiro-Wilk is limited to 5000 samples
            sample = df[target_column].dropna().sample(5000, random_state=42)
        else:
            sample = df[target_column].dropna()
            
        stat, p_value = scipy_stats.shapiro(sample)
        st.write(f"p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            st.write("The target variable does not follow a normal distribution.")
        else:
            st.write("The target variable follows a normal distribution.")
    
    # Relationship with features
    st.subheader("Relationship with Features")
    
    # Choose features to analyze
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_column]
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != target_column]
    
    if is_categorical:
        # For categorical target
        if numeric_cols:
            # Select numeric feature to analyze
            selected_numeric = st.selectbox(
                "Select Numeric Feature",
                options=numeric_cols,
                index=0
            )
            
            # Show box plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=target_column, y=selected_numeric, data=df, ax=ax)
            ax.set_title(f"{selected_numeric} by {target_column}")
            if df[target_column].nunique() > 10:
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show violin plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(x=target_column, y=selected_numeric, data=df, ax=ax)
            ax.set_title(f"Distribution of {selected_numeric} by {target_column}")
            if df[target_column].nunique() > 10:
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        if cat_cols:
            # Select categorical feature to analyze
            selected_cat = st.selectbox(
                "Select Categorical Feature",
                options=cat_cols,
                index=0
            )
            
            # Calculate contingency table
            contingency = pd.crosstab(
                index=df[target_column], 
                columns=df[selected_cat],
                normalize='index'
            )
            
            st.write(f"**Contingency Table (% across {selected_cat}):**")
            st.dataframe(contingency.style.background_gradient(cmap='Blues', axis=1), use_container_width=True)
            
            # Plot stacked bar chart
            if contingency.shape[1] <= 15:  # Limit number of categories
                fig, ax = plt.subplots(figsize=(12, 6))
                contingency.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title(f"Relationship between {target_column} and {selected_cat}")
                ax.set_ylabel(f"Proportion of {selected_cat}")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(f"Too many categories in {selected_cat} to plot.")
    else:
        # For numeric target
        if numeric_cols:
            # Select numeric feature to analyze
            selected_numeric = st.selectbox(
                "Select Numeric Feature",
                options=numeric_cols,
                index=0
            )
            
            # Show scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=selected_numeric, y=target_column, data=df, ax=ax)
            ax.set_title(f"{target_column} vs {selected_numeric}")
            
            # Add regression line
            sns.regplot(x=selected_numeric, y=target_column, data=df, scatter=False, ax=ax)
            
            st.pyplot(fig)
            
            # Calculate correlation
            corr = df[[target_column, selected_numeric]].corr().iloc[0, 1]
            st.write(f"Correlation between {target_column} and {selected_numeric}: {corr:.4f}")
        
        if cat_cols:
            # Select categorical feature to analyze
            selected_cat = st.selectbox(
                "Select Categorical Feature",
                options=cat_cols,
                index=0
            )
            
            # Show box plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=selected_cat, y=target_column, data=df, ax=ax)
            ax.set_title(f"{target_column} by {selected_cat}")
            if df[selected_cat].nunique() > 10:
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show violin plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(x=selected_cat, y=target_column, data=df, ax=ax)
            ax.set_title(f"Distribution of {target_column} by {selected_cat}")
            if df[selected_cat].nunique() > 10:
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
