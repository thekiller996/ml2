import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Dict, Any, Optional, Tuple, Callable
import logging
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import io
import base64

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')

class Visualizer:
    """Class for data visualization functions"""
    
    @staticmethod
    def configure_plot_style(
        style: str = 'whitegrid',
        context: str = 'notebook',
        palette: str = 'viridis',
        font_scale: float = 1.0,
        figure_size: Tuple[int, int] = (10, 6)
    ) -> None:
        """Configure global plot style.
        
        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
            context: Context ('notebook', 'paper', 'talk', 'poster')
            palette: Color palette
            font_scale: Font scale factor
            figure_size: Default figure size
        """
        sns.set_style(style)
        sns.set_context(context, font_scale=font_scale)
        sns.set_palette(palette)
        plt.rcParams['figure.figsize'] = figure_size
    
    @staticmethod
    def plot_histogram(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        column: Optional[str] = None,
        bins: int = 30,
        kde: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = 'Frequency',
        figsize: Tuple[int, int] = (10, 6),
        color: str = '#1f77b4',
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot histogram for numerical data.
        
        Args:
            data: Data to plot
            column: Column name if data is DataFrame
            bins: Number of bins
            kde: Whether to plot KDE curve
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            color: Bar color
            ax: Existing axes to plot on
            **kwargs: Additional arguments for histplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Extract data to plot
        if isinstance(data, pd.DataFrame) and column is not None:
            plot_data = data[column]
            if xlabel is None:
                xlabel = column
        elif isinstance(data, pd.Series):
            plot_data = data
            if xlabel is None and data.name is not None:
                xlabel = data.name
        else:
            plot_data = data
        
        # Plot histogram
        sns.histplot(plot_data, bins=bins, kde=kde, color=color, ax=ax, **kwargs)
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add descriptive statistics
        if hasattr(plot_data, 'mean') and hasattr(plot_data, 'std'):
            mean = plot_data.mean()
            std = plot_data.std()
            median = np.median(plot_data)
            stats_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\nMedian: {median:.2f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_boxplot(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        column: Optional[Union[str, List[str]]] = None,
        by: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        vert: bool = True,
        palette: str = 'viridis',
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot boxplot for numerical data.
        
        Args:
            data: Data to plot
            column: Column name(s) if data is DataFrame
            by: Group by column
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            vert: Whether to plot vertical boxplot
            palette: Color palette
            ax: Existing axes to plot on
            **kwargs: Additional arguments for boxplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Prepare data for plotting
        if isinstance(data, pd.DataFrame):
            if by is not None:
                # Group by categorical variable
                if column is None:
                    raise ValueError("column must be specified when using 'by'")
                
                if isinstance(column, list):
                    # Multiple columns, show side by side
                    plot_data = data.melt(id_vars=[by], value_vars=column, var_name='variable', value_name='value')
                    sns.boxplot(x=by, y='value', hue='variable', data=plot_data, palette=palette, ax=ax, **kwargs)
                else:
                    # Single column grouped by categorical variable
                    sns.boxplot(x=by, y=column, data=data, palette=palette, ax=ax, **kwargs)
            else:
                if column is not None:
                    if isinstance(column, list):
                        # Multiple columns
                        plot_data = data[column].melt(var_name='variable', value_name='value')
                        x_var = 'variable' if vert else 'value'
                        y_var = 'value' if vert else 'variable'
                        sns.boxplot(x=x_var, y=y_var, data=plot_data, palette=palette, ax=ax, **kwargs)
                    else:
                        # Single column
                        sns.boxplot(x=data[column], ax=ax, vert=vert, **kwargs)
                else:
                    # Use all numeric columns
                    numeric_cols = data.select_dtypes(include=['number']).columns
                    plot_data = data[numeric_cols].melt(var_name='variable', value_name='value')
                    x_var = 'variable' if vert else 'value'
                    y_var = 'value' if vert else 'variable'
                    sns.boxplot(x=x_var, y=y_var, data=plot_data, palette=palette, ax=ax, **kwargs)
        else:
            # Plot array or series
            sns.boxplot(x=data, ax=ax, vert=vert, **kwargs)
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_scatter(
        data: Union[pd.DataFrame, np.ndarray],
        x: Union[str, np.ndarray],
        y: Union[str, np.ndarray],
        hue: Optional[Union[str, np.ndarray]] = None,
        size: Optional[Union[str, np.ndarray]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        palette: str = 'viridis',
        add_reg_line: bool = False,
        alpha: float = 0.7,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot scatter plot.
        
        Args:
            data: Data source
            x: X-axis data (column name or array)
            y: Y-axis data (column name or array)
            hue: Variable for color mapping
            size: Variable for point size mapping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            palette: Color palette
            add_reg_line: Whether to add regression line
            alpha: Point transparency
            ax: Existing axes to plot on
            **kwargs: Additional arguments for scatterplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Create scatter plot
        if isinstance(data, pd.DataFrame) and isinstance(x, str) and isinstance(y, str):
            # Using DataFrame columns
            if add_reg_line:
                sns.regplot(x=x, y=y, data=data, scatter=False, line_kws={'color': 'red'}, ax=ax)
            
            scatter = sns.scatterplot(
                x=x, y=y, hue=hue, size=size,
                data=data, palette=palette, alpha=alpha, ax=ax, **kwargs
            )
            
            if xlabel is None:
                xlabel = x
            if ylabel is None:
                ylabel = y
        else:
            # Using arrays
            if add_reg_line:
                sns.regplot(x=x, y=y, scatter=False, line_kws={'color': 'red'}, ax=ax)
            
            scatter = sns.scatterplot(
                x=x, y=y, hue=hue, size=size,
                palette=palette, alpha=alpha, ax=ax, **kwargs
            )
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Add correlation coefficient if using two variables
        if isinstance(data, pd.DataFrame) and isinstance(x, str) and isinstance(y, str):
            correlation = data[x].corr(data[y])
            ax.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax.transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_correlation_matrix(
        data: pd.DataFrame,
        method: str = 'pearson',
        annot: bool = True,
        cmap: str = 'coolwarm',
        mask_upper: bool = False,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with variables to correlate
            method: Correlation method ('pearson', 'kendall', 'spearman')
            annot: Whether to annotate cells
            cmap: Colormap
            mask_upper: Whether to mask upper triangle
            figsize: Figure size
            title: Plot title
            ax: Existing axes to plot on
            **kwargs: Additional arguments for heatmap
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Compute correlation matrix
        corr = data.corr(method=method)
        
        # Create mask for upper triangle if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(
            corr, mask=mask, cmap=cmap, annot=annot, fmt='.2f',
            linewidths=0.5, ax=ax, vmin=-1, vmax=1, center=0,
            square=True, **kwargs
        )
        
        # Configure plot
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{method.capitalize()} Correlation Matrix')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pair_grid(
        data: pd.DataFrame,
        vars: Optional[List[str]] = None,
        hue: Optional[str] = None,
        palette: str = 'viridis',
        diag_kind: str = 'kde',
        height: float = 2.5,
        title: Optional[str] = None,
        **kwargs
    ) -> sns.PairGrid:
        """Plot pairwise relationships in dataset.
        
        Args:
            data: DataFrame with variables
            vars: Variables to plot
            hue: Variable for color mapping
            palette: Color palette
            diag_kind: Kind of plot for diagonal ('hist', 'kde')
            height: Height of each subplot
            title: Plot title
            **kwargs: Additional arguments for PairGrid
            
        Returns:
            Seaborn PairGrid object
        """
        # Select only numeric columns if vars not specified
        if vars is None:
            numeric_cols = data.select_dtypes(include=['number']).columns
            # Limit to reasonable number of columns
            if len(numeric_cols) > 10:
                logger.warning(f"Too many numeric columns ({len(numeric_cols)}), using first 10")
                vars = numeric_cols[:10]
            else:
                vars = numeric_cols
        
        # Create PairGrid
        grid = sns.PairGrid(data, vars=vars, hue=hue, palette=palette, height=height, **kwargs)
        
        # Map different plots to upper, diagonal, and lower
        grid = grid.map_upper(sns.scatterplot, alpha=0.7)
        grid = grid.map_lower(sns.kdeplot, levels=5, fill=True, alpha=0.5)
        
        if diag_kind == 'hist':
            grid = grid.map_diag(sns.histplot, kde=True)
        else:
            grid = grid.map_diag(sns.kdeplot, fill=True)
        
        # Add legend
        if hue is not None:
            grid.add_legend()
        
        # Add title if specified
        if title:
            plt.suptitle(title, y=1.02, fontsize=16)
        
        plt.tight_layout()
        return grid
    
    @staticmethod
    def plot_count(
        data: pd.DataFrame,
        column: str,
        hue: Optional[str] = None,
        order: Optional[List[Any]] = None,
        palette: str = 'viridis',
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = 'Count',
        figsize: Tuple[int, int] = (10, 6),
        horizontal: bool = False,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot count of categorical variable values.
        
        Args:
            data: DataFrame with data
            column: Column to count
            hue: Variable for color mapping
            order: Order of categories
            palette: Color palette
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            horizontal: Whether to plot horizontally
            ax: Existing axes to plot on
            **kwargs: Additional arguments for countplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Determine x and y based on orientation
        x_var = None if horizontal else column
        y_var = column if horizontal else None
        
        # Create count plot
        sns.countplot(
            x=x_var, y=y_var, hue=hue, data=data,
            order=order, palette=palette, ax=ax, **kwargs
        )
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add count labels
        if horizontal:
            for p in ax.patches:
                width = p.get_width()
                ax.text(width + 1, p.get_y() + p.get_height()/2, 
                       f'{int(width)}', ha='left', va='center')
        else:
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2, height + 0.1,
                       f'{int(height)}', ha='center')
        
        # Rotate x labels if many categories and not horizontal
        if not horizontal and len(data[column].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_bar(
        data: Union[pd.DataFrame, pd.Series],
        x: Optional[str] = None,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        palette: str = 'viridis',
        horizontal: bool = False,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot bar chart.
        
        Args:
            data: DataFrame with data
            x: X-axis variable
            y: Y-axis variable
            hue: Variable for color mapping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            palette: Color palette
            horizontal: Whether to plot horizontally
            ax: Existing axes to plot on
            **kwargs: Additional arguments for barplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Handle Series input
        if isinstance(data, pd.Series):
            data = data.reset_index()
            x = 'index' if x is None else x
            y = data.name if y is None else y
        
        # Swap x and y if horizontal
        if horizontal and x is not None and y is not None:
            x, y = y, x
        
        # Create bar plot
        sns.barplot(
            x=x, y=y, hue=hue, data=data,
            palette=palette, ax=ax, **kwargs
        )
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Rotate x labels if many categories and not horizontal
        if not horizontal and x is not None and isinstance(data, pd.DataFrame):
            categories = data[x].nunique()
            if categories > 5:
                plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_line(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        x: Optional[Union[str, np.ndarray]] = None,
        y: Optional[Union[str, List[str], np.ndarray]] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        palette: str = 'viridis',
        markers: bool = False,
        dashes: bool = False,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot line chart.
        
        Args:
            data: Data source
            x: X-axis data
            y: Y-axis data (column name, list of columns, or array)
            hue: Variable for color mapping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            palette: Color palette
            markers: Whether to show markers
            dashes: Whether to use different dash patterns
            ax: Existing axes to plot on
            **kwargs: Additional arguments for lineplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Handle different input types
        if isinstance(data, pd.Series):
            # Series input - use index as x
            if x is None:
                x = data.index
            if y is None:
                y = data.values
            sns.lineplot(x=x, y=y, palette=palette, markers=markers, dashes=dashes, ax=ax, **kwargs)
            
            if xlabel is None:
                xlabel = data.index.name or 'Index'
            if ylabel is None:
                ylabel = data.name
        
        elif isinstance(data, pd.DataFrame):
            if x is not None and y is not None:
                if isinstance(y, list):
                    # Multiple columns for y
                    plot_data = data.melt(id_vars=[x], value_vars=y, var_name='variable', value_name='value')
                    sns.lineplot(
                        x=x, y='value', hue='variable', data=plot_data,
                        palette=palette, markers=markers, dashes=dashes, ax=ax, **kwargs
                    )
                    
                    if ylabel is None:
                        ylabel = 'Value'
                else:
                    # Single column for y
                    sns.lineplot(
                        x=x, y=y, hue=hue, data=data,
                        palette=palette, markers=markers, dashes=dashes, ax=ax, **kwargs
                    )
                    
                    if xlabel is None:
                        xlabel = x
                    if ylabel is None:
                        ylabel = y
            else:
                # Use DataFrame with default index
                sns.lineplot(data=data, palette=palette, markers=markers, dashes=dashes, ax=ax, **kwargs)
        
        else:
            # NumPy array
            if x is None:
                x = np.arange(len(y))
            sns.lineplot(x=x, y=y, palette=palette, markers=markers, dashes=dashes, ax=ax, **kwargs)
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Add legend if multiple lines
        if isinstance(y, list) and len(y) > 1:
            ax.legend(title=None)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_time_series(
        data: Union[pd.DataFrame, pd.Series],
        date_col: Optional[str] = None,
        value_cols: Optional[Union[str, List[str]]] = None,
        resample: Optional[str] = None,
        agg_func: str = 'mean',
        title: Optional[str] = None,
        xlabel: str = 'Date',
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        palette: str = 'viridis',
        markers: bool = False,
        add_trend: bool = False,
        trend_window: int = 30,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot time series data.
        
        Args:
            data: Time series data
            date_col: Date column name
            value_cols: Value column(s)
            resample: Resample frequency (e.g., 'D', 'W', 'M')
            agg_func: Aggregation function for resampling
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            palette: Color palette
            markers: Whether to show markers
            add_trend: Whether to add trend line
            trend_window: Window size for trend line
            ax: Existing axes to plot on
            **kwargs: Additional arguments for lineplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Handle Series input
        if isinstance(data, pd.Series):
            series_data = data.copy()
            if not isinstance(series_data.index, pd.DatetimeIndex):
                raise ValueError("Series index must be DatetimeIndex for time series plot")
            
            # Resample if requested
            if resample:
                series_data = series_data.resample(resample).agg(agg_func)
            
            # Plot series
            sns.lineplot(x=series_data.index, y=series_data.values, ax=ax, markers=markers, **kwargs)
            
            # Add trend line if requested
            if add_trend and len(series_data) > trend_window:
                trend = series_data.rolling(window=trend_window).mean()
                ax.plot(trend.index, trend.values, 'r--', linewidth=2, label=f'Trend ({trend_window} periods)')
                ax.legend()
            
            if ylabel is None:
                ylabel = series_data.name
        
        # Handle DataFrame input
        else:
            df_data = data.copy()
            
            # Setup date column
            if date_col is None:
                if isinstance(df_data.index, pd.DatetimeIndex):
                    # Use index as date
                    date_values = df_data.index
                else:
                    # Try to find date column
                    date_cols = [col for col in df_data.columns if 'date' in col.lower() or 'time' in col.lower()]
                    if date_cols:
                        date_col = date_cols[0]
                        date_values = pd.to_datetime(df_data[date_col])
                    else:
                        raise ValueError("No date column found. Please specify 'date_col'")
            else:
                # Convert to datetime
                df_data[date_col] = pd.to_datetime(df_data[date_col])
                date_values = df_data[date_col]
            
            # Determine value columns
            if value_cols is None:
                # Use all numeric columns except date
                value_cols = [col for col in df_data.select_dtypes(include=['number']).columns 
                             if col != date_col]
                if not value_cols:
                    raise ValueError("No numeric columns found for time series plot")
            elif isinstance(value_cols, str):
                value_cols = [value_cols]
            
            # Resample if requested
            if resample:
                # Set date as index if it's not already
                if date_col is not None:
                    df_data = df_data.set_index(date_col)
                
                # Resample data
                df_data = df_data[value_cols].resample(resample).agg(agg_func)
                date_values = df_data.index
                
                # Plot resampled data
                for col in value_cols:
                    sns.lineplot(x=date_values, y=df_data[col], label=col, markers=markers, ax=ax, **kwargs)
                    
                    # Add trend line if requested
                    if add_trend and len(df_data) > trend_window:
                        trend = df_data[col].rolling(window=trend_window).mean()
                        ax.plot(date_values, trend.values, '--', linewidth=1.5, 
                               label=f'{col} Trend ({trend_window})')
            else:
                # Plot without resampling
                for col in value_cols:
                    if date_col is not None:
                        sns.lineplot(x=date_values, y=df_data[col], label=col, markers=markers, ax=ax, **kwargs)
                    else:
                        sns.lineplot(data=df_data, x=df_data.index, y=col, label=col, markers=markers, ax=ax, **kwargs)
                    
                    # Add trend line if requested
                    if add_trend:
                        if date_col is not None:
                            # Need to create temporary DataFrame with date index
                            temp_df = pd.DataFrame({col: df_data[col].values}, index=date_values)
                        else:
                            temp_df = df_data[[col]]
                        
                        if len(temp_df) > trend_window:
                            trend = temp_df[col].rolling(window=trend_window).mean()
                            ax.plot(trend.index, trend.values, '--', linewidth=1.5, 
                                   label=f'{col} Trend ({trend_window})')
            
            if ylabel is None:
                if len(value_cols) == 1:
                    ylabel = value_cols[0]
                else:
                    ylabel = 'Value'
        
        # Configure plot
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Format x-axis for better readability of dates
        fig.autofmt_xdate()
        
        # Add legend if multiple lines
        if isinstance(value_cols, list) and len(value_cols) > 1:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_3d_scatter(
        data: Union[pd.DataFrame, np.ndarray],
        x: Union[str, np.ndarray],
        y: Union[str, np.ndarray],
        z: Union[str, np.ndarray],
        color: Optional[Union[str, np.ndarray]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        alpha: float = 0.7,
        elev: int = 30,
        azim: int = 30,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot 3D scatter plot.
        
        Args:
            data: Data source
            x: X-axis data
            y: Y-axis data
            z: Z-axis data
            color: Values for color mapping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            zlabel: Z-axis label
            figsize: Figure size
            cmap: Colormap
            alpha: Point transparency
            elev: Elevation angle
            azim: Azimuth angle
            ax: Existing axes to plot on
            **kwargs: Additional arguments for scatter
            
        Returns:
            Figure object
        """
        # Create figure and 3D axes if needed
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Extract data from DataFrame if needed
        if isinstance(data, pd.DataFrame):
            if isinstance(x, str):
                x_data = data[x]
                if xlabel is None:
                    xlabel = x
            else:
                x_data = x
            
            if isinstance(y, str):
                y_data = data[y]
                if ylabel is None:
                    ylabel = y
            else:
                y_data = y
            
            if isinstance(z, str):
                z_data = data[z]
                if zlabel is None:
                    zlabel = z
            else:
                z_data = z
            
            if color is not None and isinstance(color, str):
                color_data = data[color]
                color_label = color
            else:
                color_data = color
                color_label = 'Value'
        else:
            x_data, y_data, z_data = x, y, z
            color_data = color
            color_label = 'Value'
        
        # Create 3D scatter plot
        if color_data is not None:
            scatter = ax.scatter(
                x_data, y_data, z_data,
                c=color_data, cmap=cmap, alpha=alpha, **kwargs
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(color_label)
        else:
            ax.scatter(
                x_data, y_data, z_data,
                alpha=alpha, **kwargs
            )
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if zlabel:
            ax.set_zlabel(zlabel)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_heatmap(
        data: Union[pd.DataFrame, np.ndarray],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cmap: str = 'viridis',
        annot: bool = True,
        fmt: str = '.2f',
        linewidths: float = 0.5,
        figsize: Tuple[int, int] = (10, 8),
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot heatmap.
        
        Args:
            data: Data to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            cmap: Colormap
            annot: Whether to annotate cells
            fmt: Number format for annotations
            linewidths: Width of lines between cells
            figsize: Figure size
            ax: Existing axes to plot on
            **kwargs: Additional arguments for heatmap
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Create heatmap
        sns.heatmap(
            data, annot=annot, cmap=cmap, fmt=fmt,
            linewidths=linewidths, ax=ax, **kwargs
        )
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_distribution_grid(
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        kind: str = 'kde',
        figsize: Tuple[int, int] = (12, 10),
        n_cols: int = 3,
        height: float = 4,
        title: Optional[str] = None,
        palette: str = 'viridis',
        **kwargs
    ) -> plt.Figure:
        """Plot grid of distributions.
        
        Args:
            data: DataFrame with data
            columns: Columns to plot
            hue: Variable for color mapping
            kind: Kind of plot ('hist', 'kde')
            figsize: Figure size
            n_cols: Number of columns in grid
            height: Height of each subplot
            title: Plot title
            palette: Color palette
            **kwargs: Additional arguments for distribution plots
            
        Returns:
            Figure object
        """
        # Select columns to plot
        if columns is None:
            columns = data.select_dtypes(include=['number']).columns.tolist()
        
        # Calculate grid dimensions
        n_plots = len(columns)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Create distribution plots
        for i, col in enumerate(columns):
            if i < len(axes):
                if kind == 'hist':
                    sns.histplot(data=data, x=col, hue=hue, kde=True, palette=palette, ax=axes[i], **kwargs)
                else:
                    sns.kdeplot(data=data, x=col, hue=hue, fill=True, palette=palette, ax=axes[i], **kwargs)
                
                # Add stats
                if hue is None:
                    mean = data[col].mean()
                    median = data[col].median()
                    std = data[col].std()
                    axes[i].axvline(mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean:.2f}')
                    axes[i].axvline(median, color='green', linestyle='-.', alpha=0.7, label=f'Median: {median:.2f}')
                    axes[i].legend(fontsize='small')
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # Add title
        if title:
            plt.suptitle(title, fontsize=16, y=1.02)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        labels: Optional[List[str]] = None,
        normalize: Optional[str] = None,
        title: Optional[str] = None,
        cmap: str = 'Blues',
        figsize: Tuple[int, int] = (8, 6),
        fmt: str = 'd',
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            normalize: Normalization method ('true', 'pred', 'all', None)
            title: Plot title
            cmap: Colormap
            figsize: Figure size
            fmt: Number format for annotations
            ax: Existing axes to plot on
            **kwargs: Additional arguments for heatmap
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            if normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
                if title is None:
                    title = 'Normalized Confusion Matrix (by true label)'
            elif normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
                fmt = '.2f'
                if title is None:
                    title = 'Normalized Confusion Matrix (by prediction)'
            elif normalize == 'all':
                cm = cm.astype('float') / cm.sum()
                fmt = '.2f'
                if title is None:
                    title = 'Normalized Confusion Matrix (by all)'
        elif title is None:
            title = 'Confusion Matrix'
        
        # Plot confusion matrix
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=labels, yticklabels=labels,
            ax=ax, **kwargs
        )
        
        # Configure plot
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.5, -0.1, f'Accuracy: {accuracy:.4f}', transform=ax.transAxes,
               ha='center', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curve(
        y_true: Union[np.ndarray, List],
        y_pred_proba: Union[np.ndarray, List],
        title: str = 'Receiver Operating Characteristic (ROC) Curve',
        figsize: Tuple[int, int] = (8, 6),
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            figsize: Figure size
            ax: Existing axes to plot on
            **kwargs: Additional arguments for plot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})', **kwargs)
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Configure plot
        ax.set_title(title)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(
        y_true: Union[np.ndarray, List],
        y_pred_proba: Union[np.ndarray, List],
        title: str = 'Precision-Recall Curve',
        figsize: Tuple[int, int] = (8, 6),
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            figsize: Figure size
            ax: Existing axes to plot on
            **kwargs: Additional arguments for plot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Plot precision-recall curve
        ax.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})', **kwargs)
        
        # Configure plot
        ax.set_title(title)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(
        importance: Union[np.ndarray, List],
        feature_names: Union[List[str], np.ndarray],
        title: str = 'Feature Importance',
        figsize: Tuple[int, int] = (10, 8),
        color: str = '#1f77b4',
        orientation: str = 'vertical',
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot feature importance.
        
        Args:
            importance: Feature importance values
            feature_names: Feature names
            title: Plot title
            figsize: Figure size
            color: Bar color
            orientation: Bar orientation ('vertical' or 'horizontal')
            ax: Existing axes to plot on
            **kwargs: Additional arguments for barplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Create DataFrame with importance and feature names
        df = pd.DataFrame({'importance': importance, 'feature': feature_names})
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False)
        
        # Plot based on orientation
        if orientation == 'horizontal':
            df = df.sort_values('importance')
            sns.barplot(x='importance', y='feature', data=df, color=color, ax=ax, **kwargs)
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
        else:
            sns.barplot(x='feature', y='importance', data=df, color=color, ax=ax, **kwargs)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=45, ha='right')
        
        # Configure plot
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_residuals(
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        title: str = 'Residual Plot',
        figsize: Tuple[int, int] = (10, 6),
        color: str = '#1f77b4',
        add_line: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot residuals for regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            figsize: Figure size
            color: Point color
            add_line: Whether to add horizontal line at y=0
            ax: Existing axes to plot on
            **kwargs: Additional arguments for scatter
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Plot residuals
        ax.scatter(y_pred, residuals, color=color, alpha=0.7, **kwargs)
        
        # Add horizontal line at y=0
        if add_line:
            ax.axhline(y=0, color='r', linestyle='--')
        
        # Configure plot
        ax.set_title(title)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        stats_text = f"Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_learning_curve(
        train_sizes: Union[np.ndarray, List],
        train_scores: Union[np.ndarray, List],
        test_scores: Union[np.ndarray, List],
        title: str = 'Learning Curve',
        figsize: Tuple[int, int] = (10, 6),
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot learning curve.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            test_scores: Test/validation scores
            title: Plot title
            figsize: Figure size
            ax: Existing axes to plot on
            **kwargs: Additional arguments for plot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Calculate means and standard deviations
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score', **kwargs)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='r')
        
        ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score', **kwargs)
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                       test_scores_mean + test_scores_std, alpha=0.1, color='g')
        
        # Configure plot
        ax.set_title(title)
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_scatter(
        X: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, List],
        x_col: Union[int, str] = 0,
        y_col: Union[int, str] = 1,
        title: str = 'Cluster Scatter Plot',
        figsize: Tuple[int, int] = (10, 6),
        cmap: str = 'viridis',
        add_centers: bool = False,
        centers: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot scatter plot of clusters.
        
        Args:
            X: Data points
            labels: Cluster labels
            x_col: Column/feature for x-axis
            y_col: Column/feature for y-axis
            title: Plot title
            figsize: Figure size
            cmap: Colormap
            add_centers: Whether to add cluster centers
            centers: Cluster centers (required if add_centers=True)
            ax: Existing axes to plot on
            **kwargs: Additional arguments for scatter
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Extract data points
        if isinstance(X, pd.DataFrame):
            x_data = X[x_col]
            y_data = X[y_col]
            x_label = x_col
            y_label = y_col
        else:
            x_data = X[:, x_col]
            y_data = X[:, y_col]
            x_label = f'Feature {x_col}'
            y_label = f'Feature {y_col}'
        
        # Create discrete colormap for clusters
        n_clusters = len(np.unique(labels))
        cmap_obj = plt.cm.get_cmap(cmap, n_clusters)
        
        # Plot clusters
        scatter = ax.scatter(x_data, y_data, c=labels, cmap=cmap_obj, **kwargs)
        
        # Add cluster centers if requested
        if add_centers and centers is not None:
            if isinstance(x_col, str) and isinstance(y_col, str):
                # Need to find indices for these column names
                if isinstance(X, pd.DataFrame):
                    col_indices = [list(X.columns).index(col) for col in [x_col, y_col]]
                    center_x = centers[:, col_indices[0]]
                    center_y = centers[:, col_indices[1]]
                else:
                    raise ValueError("Cannot determine center coordinates for string column names with non-DataFrame X")
            else:
                center_x = centers[:, x_col]
                center_y = centers[:, y_col]
            
            ax.scatter(center_x, center_y, s=200, c='red', marker='X', 
                      edgecolors='black', label='Cluster Centers')
            ax.legend()
        
        # Configure plot
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_3d_surface(
        X: Union[np.ndarray, List],
        Y: Union[np.ndarray, List],
        Z: Union[np.ndarray, List],
        title: str = '3D Surface Plot',
        xlabel: str = 'X',
        ylabel: str = 'Y',
        zlabel: str = 'Z',
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        elev: int = 30,
        azim: int = 30,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot 3D surface.
        
        Args:
            X: X coordinates (2D grid)
            Y: Y coordinates (2D grid)
            Z: Z values (2D grid)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            zlabel: Z-axis label
            figsize: Figure size
            cmap: Colormap
            elev: Elevation angle
            azim: Azimuth angle
            ax: Existing axes to plot on
            **kwargs: Additional arguments for plot_surface
            
        Returns:
            Figure object
        """
        # Create figure and 3D axes if needed
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, **kwargs)
        
        # Configure plot
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_qq(
        data: Union[np.ndarray, pd.Series, List],
        title: str = 'Normal Q-Q Plot',
        figsize: Tuple[int, int] = (8, 6),
        color: str = '#1f77b4',
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot Q-Q (quantile-quantile) plot.
        
        Args:
            data: Data to check for normality
            title: Plot title
            figsize: Figure size
            color: Point color
            ax: Existing axes to plot on
            **kwargs: Additional arguments for probplot
            
        Returns:
            Figure object
        """
        import scipy.stats as stats
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Extract data
        if isinstance(data, pd.Series):
            data_array = data.values
        else:
            data_array = np.array(data)
        
        # Create Q-Q plot
        (osm, osr), (slope, intercept, r) = stats.probplot(data_array, plot=ax, **kwargs)
        
        # Change color of points
        ax.get_lines()[0].set_markerfacecolor(color)
        ax.get_lines()[0].set_markeredgecolor(color)
        
        # Configure plot
        ax.set_title(title)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        
        # Add R² value to the plot
        ax.text(0.05, 0.95, f'R² = {r**2:.4f}', transform=ax.transAxes,
               fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_violin(
        data: Union[pd.DataFrame, np.ndarray],
        x: Optional[Union[str, np.ndarray]] = None,
        y: Optional[Union[str, np.ndarray]] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        palette: str = 'viridis',
        split: bool = False,
        inner: str = 'box',
        orient: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot violin plot.
        
        Args:
            data: Data for plotting
            x: X-axis variable
            y: Y-axis variable
            hue: Variable for color mapping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            palette: Color palette
            split: Whether to split violins when hue is used
            inner: Inner representation ('box', 'quartile', 'point', 'stick', None)
            orient: Orientation ('v', 'h')
            ax: Existing axes to plot on
            **kwargs: Additional arguments for violinplot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Create violin plot
        sns.violinplot(
            x=x, y=y, hue=hue, data=data,
            palette=palette, split=split, inner=inner,
            orient=orient, ax=ax, **kwargs
        )
        
        # Configure plot
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Rotate x-axis labels if many categories
        if x is not None and isinstance(data, pd.DataFrame) and data[x].nunique() > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pca_variance(
        explained_variance_ratio: Union[np.ndarray, List],
        title: str = 'PCA Explained Variance',
        figsize: Tuple[int, int] = (10, 6),
        color: str = '#1f77b4',
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot PCA explained variance.
        
        Args:
            explained_variance_ratio: Explained variance ratio for each component
            title: Plot title
            figsize: Figure size
            color: Bar color
            ax: Existing axes to plot on
            **kwargs: Additional arguments for bar plot
            
        Returns:
            Figure object
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Create component indices
        components = np.arange(1, len(explained_variance_ratio) + 1)
        
        # Plot individual explained variance
        ax.bar(components, explained_variance_ratio, color=color, **kwargs)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_xticks(components)
        
        # Plot cumulative explained variance
        cumulative = np.cumsum(explained_variance_ratio)
        ax2 = ax.twinx()
        ax2.plot(components, cumulative, 'r-', marker='o', markersize=6)
        ax2.set_ylabel('Cumulative Explained Variance', color='r')
        ax2.tick_params(axis='y', colors='r')
        ax2.set_ylim([0, 1.05])
        
        # Add threshold lines
        for threshold in [0.8, 0.9, 0.95]:
            # Find first component that exceeds threshold
            try:
                component_idx = np.where(cumulative >= threshold)[0][0]
                component_num = component_idx + 1
                ax2.axhline(y=threshold, color='k', linestyle='--', alpha=0.3)
                ax2.text(
                    components[-1], threshold, f'{threshold:.0%}',
                    ha='right', va='bottom', color='k', alpha=0.5
                )
                ax2.text(
                    component_num, threshold,
                    f'  n={component_num}', ha='left', va='bottom', color='k', alpha=0.7
                )
            except IndexError:
                pass
        
        # Configure plot
        ax.set_title(title)
        
        plt.tight_layout()
        return fig