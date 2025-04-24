"""
Sample plugin for the ML Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
from plugins.plugin_base import PluginBase, hook

# Plugin metadata
PLUGIN_NAME = "Sample Plugin"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "A sample plugin demonstrating the plugin system capabilities."
PLUGIN_AUTHOR = "ML Platform Team"

class SamplePlugin(PluginBase):
    """
    Sample plugin implementation.
    """
    
    @property
    def name(self) -> str:
        return PLUGIN_NAME
    
    @property
    def version(self) -> str:
        return PLUGIN_VERSION
    
    @property
    def description(self) -> str:
        return PLUGIN_DESCRIPTION
    
    @property
    def author(self) -> str:
        return PLUGIN_AUTHOR
    
    @hook('on_app_start')
    def on_app_start(self):
        """
        Called when the application starts.
        """
        print(f"Sample plugin initialized: {self.name} v{self.version}")
    
    @hook('render_custom_page')
    def render_custom_page(self, page_name: str) -> bool:
        """
        Render a custom page.
        
        Args:
            page_name: Name of the page to render
        
        Returns:
            True if the page was rendered, False otherwise
        """
        if page_name == "Sample Plugin Page":
            st.title("Sample Plugin Page")
            st.write("This is a custom page provided by the Sample Plugin.")
            
            st.subheader("Plugin Features")
            st.write("""
            This plugin demonstrates:
            - Adding a custom page
            - Extending preprocessing functionality
            - Providing a sample dataset
            """)
            
            # Show some interactive elements
            if st.button("Click Me!"):
                st.success("Button clicked!")
            
            number = st.slider("Select a number", 0, 100, 50)
            st.write(f"You selected: {number}")
            
            return True
        
        return False
    
    @hook('get_custom_pages')
    def get_custom_pages(self):
        """
        Get list of custom pages provided by this plugin.
        
        Returns:
            List of custom page names
        """
        return ["Sample Plugin Page"]
    
    @hook('get_sample_datasets')
    def get_sample_datasets(self):
        """
        Get list of sample datasets provided by this plugin.
        
        Returns:
            List of sample dataset names
        """
        return ["sample_random_data"]
    
    @hook('load_sample_dataset')
    def load_sample_dataset(self, dataset_name: str):
        """
        Load a sample dataset.
        
        Args:
            dataset_name: Name of the dataset to load
        
        Returns:
            DataFrame with the loaded dataset
        """
        if dataset_name == "sample_random_data":
            # Generate random data
            np.random.seed(42)
            n_samples = 1000
            
            df = pd.DataFrame({
                'id': range(1, n_samples + 1),
                'numeric1': np.random.normal(0, 1, n_samples),
                'numeric2': np.random.normal(5, 2, n_samples),
                'numeric3': np.random.exponential(2, n_samples),
                'category1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
                'category2': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                'date': pd.date_range(start='2020-01-01', periods=n_samples),
                'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            })
            
            # Add some missing values
            mask = np.random.random(n_samples) < 0.05
            df.loc[mask, 'numeric1'] = np.nan
            
            mask = np.random.random(n_samples) < 0.05
            df.loc[mask, 'category1'] = np.nan
            
            return df
            
        return None
    
    @hook('get_preprocessors')
    def get_preprocessors(self):
        """
        Get preprocessors provided by this plugin.
        
        Returns:
            List of preprocessor descriptors
        """
        return [
            {
                'name': 'Random Noise Addition',
                'description': 'Add random noise to numeric columns for data augmentation or testing.',
                'plugin': self.name
            },
            {
                'name': 'Text Length Feature',
                'description': 'Create features from the length of text in string columns.',
                'plugin': self.name
            }
        ]
    
    @hook('render_preprocessor_ui')
    def render_preprocessor_ui(self, preprocessor_name: str, df: pd.DataFrame):
        """
        Render UI for a preprocessor.
        
        Args:
            preprocessor_name: Name of the preprocessor
            df: DataFrame to process
        
        Returns:
            (DataFrame, metadata) tuple if processing is applied, None otherwise
        """
        if preprocessor_name == 'Random Noise Addition':
            st.write("Add random noise to numeric columns.")
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns found in the dataset.")
                return None
            
            # Column selection
            selected_columns = st.multiselect(
                "Select Columns",
                options=numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if not selected_columns:
                st.info("Please select at least one column.")
                return None
            
            # Noise parameters
            noise_type = st.selectbox(
                "Noise Distribution",
                options=["Gaussian", "Uniform"],
                index=0
            )
            
            noise_scale = st.slider(
                "Noise Scale",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Standard deviation for Gaussian or range for Uniform"
            )
            
            # Apply button
            if st.button("Add Noise", type="primary"):
                result_df = df.copy()
                
                # Generate and add noise
                for col in selected_columns:
                    if noise_type == "Gaussian":
                        noise = np.random.normal(0, noise_scale, len(df))
                    else:  # Uniform
                        noise = np.random.uniform(-noise_scale, noise_scale, len(df))
                    
                    # Scale noise by the column's standard deviation
                    if st.session_state.get('scale_by_std', False):
                        col_std = df[col].std()
                        noise *= col_std
                    
                    # Add noise to the column
                    result_df[col] = df[col] + noise
                
                # Create metadata
                metadata = {
                    'params': {
                        'columns': selected_columns,
                        'noise_type': noise_type,
                        'noise_scale': noise_scale
                    },
                    'results': {
                        'columns_modified': len(selected_columns)
                    }
                }
                
                return result_df, metadata
        
        elif preprocessor_name == 'Text Length Feature':
            st.write("Create features from text length in string columns.")
            
            # Get string columns
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            if not text_cols:
                st.warning("No text columns found in the dataset.")
                return None
            
            # Column selection
            selected_columns = st.multiselect(
                "Select Text Columns",
                options=text_cols,
                default=text_cols[:min(3, len(text_cols))]
            )
            
            if not selected_columns:
                st.info("Please select at least one column.")
                return None
            
            # Feature options
            create_char_length = st.checkbox("Character Length", value=True)
            create_word_count = st.checkbox("Word Count", value=True)
            create_avg_word_length = st.checkbox("Average Word Length", value=False)
            
            if not any([create_char_length, create_word_count, create_avg_word_length]):
                st.info("Please select at least one feature type.")
                return None
            
            # Apply button
            if st.button("Create Features", type="primary"):
                result_df = df.copy()
                created_features = []
                
                # Process each column
                for col in selected_columns:
                    # Ensure column is string type
                    text_series = df[col].astype(str)
                    
                    if create_char_length:
                        feat_name = f"{col}_char_length"
                        result_df[feat_name] = text_series.str.len()
                        created_features.append(feat_name)
                    
                    if create_word_count:
                        feat_name = f"{col}_word_count"
                        result_df[feat_name] = text_series.str.split().str.len()
                        created_features.append(feat_name)
                    
                    if create_avg_word_length:
                        feat_name = f"{col}_avg_word_length"
                        
                        # Calculate average word length
                        def avg_word_len(text):
                            words = text.split()
                            if not words:
                                return 0
                            return sum(len(word) for word in words) / len(words)
                        
                        result_df[feat_name] = text_series.apply(avg_word_len)
                        created_features.append(feat_name)
                
                # Create metadata
                metadata = {
                    'params': {
                        'columns': selected_columns,
                        'char_length': create_char_length,
                        'word_count': create_word_count,
                        'avg_word_length': create_avg_word_length
                    },
                    'results': {
                        'created_features': created_features
                    }
                }
                
                return result_df, metadata
        
        return None
    
    @hook('extend_ml_tasks')
    def extend_ml_tasks(self):
        """
        Add additional ML task types.
        
        Returns:
            List of task names to add
        """
        return ["Anomaly Detection", "Topic Modeling"]

# Create plugin instance
plugin_instance = SamplePlugin()

def register_plugin(manager):
    """
    Register the plugin with the plugin manager.
    
    Args:
        manager: Plugin manager instance
    """
    plugin_instance.register_all_hooks(manager)
