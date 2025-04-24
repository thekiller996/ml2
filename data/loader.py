import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
import os
import glob
import json
import xml.etree.ElementTree as ET
import sqlite3
import pymongo
import boto3
import sqlalchemy
import pyodbc
import h5py
import logging
import yaml
from io import StringIO, BytesIO
from urllib.parse import urlparse
import requests
import zipfile
import tarfile
import pickle
import joblib
import re
import csv
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading data from various sources"""
    
    @staticmethod
    def load_csv(
        filepath: str,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        header: Union[int, None] = 0,
        index_col: Optional[Union[int, str]] = None,
        parse_dates: Optional[List[Union[int, str]]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            delimiter: Field delimiter
            encoding: File encoding
            header: Row number to use as header (0-indexed)
            index_col: Column to use as index
            parse_dates: List of columns to parse as dates
            dtype: Dictionary mapping columns to data types
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(
                filepath,
                delimiter=delimiter,
                encoding=encoding,
                header=header,
                index_col=index_col,
                parse_dates=parse_dates,
                dtype=dtype,
                **kwargs
            )
            logger.info(f"Successfully loaded CSV from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_excel(
        filepath: str,
        sheet_name: Union[str, int, List[Union[str, int]], None] = 0,
        header: Union[int, None] = 0,
        index_col: Optional[Union[int, str]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load data from Excel file.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name(s) to load
            header: Row number to use as header (0-indexed)
            index_col: Column to use as index
            dtype: Dictionary mapping columns to data types
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            Loaded DataFrame or dict of DataFrames
        """
        try:
            df = pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                header=header,
                index_col=index_col,
                dtype=dtype,
                **kwargs
            )
            
            if isinstance(df, dict):
                logger.info(f"Successfully loaded Excel from {filepath} with {len(df)} sheets")
                for sheet, sheet_df in df.items():
                    logger.info(f"  Sheet '{sheet}': {sheet_df.shape[0]} rows and {sheet_df.shape[1]} columns")
            else:
                logger.info(f"Successfully loaded Excel from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
            
            return df
        except Exception as e:
            logger.error(f"Error loading Excel from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_json(
        filepath: str,
        orient: str = 'records',
        encoding: str = 'utf-8',
        lines: bool = False,
        convert_dates: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from JSON file.
        
        Args:
            filepath: Path to JSON file
            orient: JSON string format
            encoding: File encoding
            lines: Whether JSON is in lines format
            convert_dates: Whether to convert date strings to datetime
            **kwargs: Additional arguments for pd.read_json
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Load data using pandas
            df = pd.read_json(
                filepath,
                orient=orient,
                encoding=encoding,
                lines=lines,
                convert_dates=convert_dates,
                **kwargs
            )
            logger.info(f"Successfully loaded JSON from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            # Try manual loading for complex JSON
            try:
                logger.warning(f"Standard JSON loading failed, attempting manual parsing: {str(e)}")
                
                with open(filepath, 'r', encoding=encoding) as file:
                    if lines:
                        # Read JSON Lines (one JSON object per line)
                        data = [json.loads(line) for line in file if line.strip()]
                    else:
                        # Read regular JSON
                        data = json.load(file)
                
                # Convert to DataFrame with normalization for nested structures
                if isinstance(data, dict):
                    # Handle dictionary
                    df = pd.json_normalize(data)
                elif isinstance(data, list):
                    # Handle list of records
                    df = pd.json_normalize(data)
                else:
                    raise ValueError(f"Unexpected JSON structure: {type(data)}")
                
                logger.info(f"Successfully loaded JSON from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
                return df
            
            except Exception as nested_e:
                logger.error(f"Error loading JSON from {filepath}: {str(nested_e)}")
                raise
    
    @staticmethod
    def load_xml(
        filepath: str,
        xpath: str = './*',
        namespaces: Optional[Dict[str, str]] = None,
        flatten_nested: bool = True
    ) -> pd.DataFrame:
        """Load data from XML file.
        
        Args:
            filepath: Path to XML file
            xpath: XPath expression to select elements
            namespaces: Namespace mapping
            flatten_nested: Whether to flatten nested elements
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Parse XML file
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find elements matching xpath
            elements = root.findall(xpath, namespaces)
            
            if not elements:
                logger.warning(f"No elements found matching xpath: {xpath}")
                return pd.DataFrame()
            
            # Function to extract element data
            def extract_element_data(elem, prefix=''):
                data = {}
                
                # Extract attributes
                for key, value in elem.attrib.items():
                    data[f"{prefix}{key}"] = value
                
                # Extract text content if it's a leaf node
                if not list(elem) and elem.text and elem.text.strip():
                    data[f"{prefix}text"] = elem.text.strip()
                
                # Extract child elements
                for child in elem:
                    tag = child.tag
                    if '}' in tag:  # Handle namespaced tags
                        tag = tag.split('}', 1)[1]
                    
                    if flatten_nested:
                        # Flatten with prefixed keys
                        child_data = extract_element_data(child, f"{prefix}{tag}_")
                        data.update(child_data)
                    else:
                        # Just extract child tag text
                        if not list(child) and child.text and child.text.strip():
                            data[f"{prefix}{tag}"] = child.text.strip()
                
                return data
            
            # Extract data from all matching elements
            data_list = [extract_element_data(elem) for elem in elements]
            
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            
            logger.info(f"Successfully loaded XML from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading XML from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_sql(
        query: str,
        connection_string: str,
        params: Optional[Dict[str, Any]] = None,
        engine_type: str = 'sqlite',
        chunksize: Optional[int] = None
    ) -> pd.DataFrame:
        """Load data from SQL database.
        
        Args:
            query: SQL query to execute
            connection_string: Database connection string
            params: Parameters for query
            engine_type: Database engine type (sqlite, postgres, mysql, mssql, oracle)
            chunksize: Number of rows to load at a time
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Setup database connection based on engine type
            if engine_type == 'sqlite':
                conn = sqlite3.connect(connection_string)
            else:
                # Create SQLAlchemy engine
                engine = sqlalchemy.create_engine(connection_string)
                conn = engine.connect()
            
            try:
                # Execute query and load data
                if chunksize:
                    chunks = []
                    for chunk in pd.read_sql(query, conn, params=params, chunksize=chunksize):
                        chunks.append(chunk)
                    df = pd.concat(chunks)
                else:
                    df = pd.read_sql(query, conn, params=params)
                
                logger.info(f"Successfully executed SQL query with {df.shape[0]} rows and {df.shape[1]} columns")
                return df
                
            finally:
                # Close connection
                conn.close()
                
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            raise
    
    @staticmethod
    def load_parquet(
        filepath: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[List[Tuple[str, str, Any]]]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from Parquet file.
        
        Args:
            filepath: Path to Parquet file
            columns: List of columns to load
            filters: Filters to apply when loading
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_parquet(
                filepath,
                columns=columns,
                filters=filters,
                **kwargs
            )
            logger.info(f"Successfully loaded Parquet from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_hdf(
        filepath: str,
        key: str,
        where: Optional[str] = None,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            key: Path to group in HDF5 file
            where: Query string for selecting rows
            columns: List of columns to load
            **kwargs: Additional arguments for pd.read_hdf
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_hdf(
                filepath,
                key=key,
                where=where,
                columns=columns,
                **kwargs
            )
            logger.info(f"Successfully loaded HDF5 from {filepath} (key={key}) with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading HDF5 from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_pickle(
        filepath: str
    ) -> Any:
        """Load data from pickle file.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Loaded object
        """
        try:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
            
            if isinstance(data, pd.DataFrame):
                logger.info(f"Successfully loaded DataFrame from pickle {filepath} with {data.shape[0]} rows and {data.shape[1]} columns")
            else:
                logger.info(f"Successfully loaded object of type {type(data)} from pickle {filepath}")
            
            return data
        except Exception as e:
            logger.error(f"Error loading pickle from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_feather(
        filepath: str,
        columns: Optional[List[str]] = None,
        use_threads: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from Feather file.
        
        Args:
            filepath: Path to Feather file
            columns: List of columns to load
            use_threads: Whether to use multithreading
            **kwargs: Additional arguments for pd.read_feather
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_feather(
                filepath,
                columns=columns,
                use_threads=use_threads,
                **kwargs
            )
            logger.info(f"Successfully loaded Feather from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading Feather from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_from_url(
        url: str,
        file_format: str = 'csv',
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Tuple[str, str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from URL.
        
        Args:
            url: URL to load data from
            file_format: Format of the data ('csv', 'json', 'excel', etc.)
            params: Request parameters
            headers: Request headers
            auth: Authentication (username, password)
            **kwargs: Additional arguments for pd.read_* functions
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Make request
            response = requests.get(url, params=params, headers=headers, auth=auth)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # Load data according to format
            if file_format.lower() == 'csv':
                df = pd.read_csv(StringIO(response.text), **kwargs)
            elif file_format.lower() in ['xls', 'xlsx', 'excel']:
                df = pd.read_excel(BytesIO(response.content), **kwargs)
            elif file_format.lower() == 'json':
                df = pd.read_json(StringIO(response.text), **kwargs)
            elif file_format.lower() == 'parquet':
                df = pd.read_parquet(BytesIO(response.content), **kwargs)
            elif file_format.lower() == 'feather':
                df = pd.read_feather(BytesIO(response.content), **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully loaded {file_format} from {url} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {url}: {str(e)}")
            raise
    
    @staticmethod
    def load_compressed(
        filepath: str,
        file_to_extract: Optional[str] = None,
        compression: Optional[str] = None,
        file_format: str = 'csv',
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load data from compressed file.
        
        Args:
            filepath: Path to compressed file
            file_to_extract: Specific file to extract from archive
            compression: Compression type ('zip', 'tar', 'gzip', etc.)
            file_format: Format of the extracted file(s)
            **kwargs: Additional arguments for pd.read_* functions
            
        Returns:
            Loaded DataFrame or dict of DataFrames
        """
        try:
            # Determine compression type if not specified
            if compression is None:
                if filepath.endswith('.zip'):
                    compression = 'zip'
                elif filepath.endswith(('.tar', '.tar.gz', '.tgz')):
                    compression = 'tar'
                else:
                    raise ValueError(f"Could not determine compression type for {filepath}")
            
            # Open archive
            if compression == 'zip':
                with zipfile.ZipFile(filepath, 'r') as archive:
                    # List of files to extract
                    if file_to_extract:
                        files_to_extract = [file_to_extract]
                    else:
                        files_to_extract = archive.namelist()
                    
                    result = {}
                    for file_path in files_to_extract:
                        # Skip directories
                        if file_path.endswith('/'):
                            continue
                        
                        # Skip unsupported file formats
                        _, ext = os.path.splitext(file_path)
                        if not ext or ext.lower() not in ['.csv', '.json', '.xlsx', '.xls', '.parquet', '.feather']:
                            continue
                        
                        # Extract file data
                        with archive.open(file_path) as file:
                            # Determine file format
                            if ext.lower() == '.csv':
                                df = pd.read_csv(file, **kwargs)
                            elif ext.lower() in ['.xlsx', '.xls']:
                                df = pd.read_excel(BytesIO(file.read()), **kwargs)
                            elif ext.lower() == '.json':
                                df = pd.read_json(file, **kwargs)
                            elif ext.lower() == '.parquet':
                                df = pd.read_parquet(BytesIO(file.read()), **kwargs)
                            elif ext.lower() == '.feather':
                                df = pd.read_feather(BytesIO(file.read()), **kwargs)
                            
                            result[file_path] = df
            
            elif compression == 'tar':
                with tarfile.open(filepath, 'r') as archive:
                    # List of files to extract
                    if file_to_extract:
                        files_to_extract = [file_to_extract]
                    else:
                        files_to_extract = archive.getnames()
                    
                    result = {}
                    for file_path in files_to_extract:
                        # Skip directories
                        if not archive.getmember(file_path).isfile():
                            continue
                        
                        # Skip unsupported file formats
                        _, ext = os.path.splitext(file_path)
                        if not ext or ext.lower() not in ['.csv', '.json', '.xlsx', '.xls', '.parquet', '.feather']:
                            continue
                        
                        # Extract file data
                        file = archive.extractfile(file_path)
                        if file:
                            # Determine file format
                            if ext.lower() == '.csv':
                                df = pd.read_csv(file, **kwargs)
                            elif ext.lower() in ['.xlsx', '.xls']:
                                df = pd.read_excel(BytesIO(file.read()), **kwargs)
                            elif ext.lower() == '.json':
                                df = pd.read_json(file, **kwargs)
                            elif ext.lower() == '.parquet':
                                df = pd.read_parquet(BytesIO(file.read()), **kwargs)
                            elif ext.lower() == '.feather':
                                df = pd.read_feather(BytesIO(file.read()), **kwargs)
                            
                            result[file_path] = df
            
            else:
                raise ValueError(f"Unsupported compression type: {compression}")
            
            # If only extracting one file, return DataFrame instead of dict
            if file_to_extract and file_to_extract in result:
                logger.info(f"Successfully loaded {file_to_extract} from {filepath}")
                return result[file_to_extract]
            
            logger.info(f"Successfully loaded {len(result)} files from {filepath}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading compressed data from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_images(
        directory: str,
        pattern: str = '*',
        recursive: bool = False,
        return_filenames: bool = True,
        convert_mode: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[str]]]:
        """Load images from directory.
        
        Args:
            directory: Directory containing images
            pattern: Glob pattern for image files
            recursive: Whether to search recursively
            return_filenames: Whether to return filenames
            convert_mode: Mode to convert images to (e.g., 'RGB', 'L')
            target_size: Size to resize images to
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            List of images or tuple of (images, filenames)
        """
        try:
            # Get list of image files
            if recursive:
                glob_pattern = os.path.join(directory, '**', pattern)
                image_files = glob.glob(glob_pattern, recursive=True)
            else:
                glob_pattern = os.path.join(directory, pattern)
                image_files = glob.glob(glob_pattern)
            
            # Filter for image files
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
            image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in image_exts]
            
            # Load images
            images = []
            filenames = []
            
            for image_file in image_files:
                try:
                    # Open image
                    with Image.open(image_file) as img:
                        # Convert mode if specified
                        if convert_mode:
                            img = img.convert(convert_mode)
                        
                        # Resize if target size specified
                        if target_size:
                            img = img.resize(target_size)
                        
                        # Convert to numpy array
                        img_array = np.array(img)
                        
                        # Normalize if requested
                        if normalize:
                            img_array = img_array.astype(np.float32) / 255.0
                        
                        images.append(img_array)
                        filenames.append(image_file)
                
                except Exception as e:
                    logger.warning(f"Error loading image {image_file}: {str(e)}")
            
            logger.info(f"Successfully loaded {len(images)} images from {directory}")
            
            if return_filenames:
                return images, filenames
            else:
                return images
                
        except Exception as e:
            logger.error(f"Error loading images from {directory}: {str(e)}")
            raise
    
    @staticmethod
    def load_multiple_files(
        directory: str,
        pattern: str = '*.csv',
        recursive: bool = False,
        file_format: str = 'csv',
        concat: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load multiple files from directory.
        
        Args:
            directory: Directory containing files
            pattern: Glob pattern for files
            recursive: Whether to search recursively
            file_format: Format of files
            concat: Whether to concatenate files into single DataFrame
            **kwargs: Additional arguments for pd.read_* functions
            
        Returns:
            Concatenated DataFrame or dict of DataFrames
        """
        try:
            # Get list of files
            if recursive:
                glob_pattern = os.path.join(directory, '**', pattern)
                files = glob.glob(glob_pattern, recursive=True)
            else:
                glob_pattern = os.path.join(directory, pattern)
                files = glob.glob(glob_pattern)
            
            if not files:
                logger.warning(f"No files found matching pattern {glob_pattern}")
                return pd.DataFrame() if concat else {}
            
            # Load each file
            dfs = {}
            for file_path in files:
                try:
                    if file_format.lower() == 'csv':
                        df = pd.read_csv(file_path, **kwargs)
                    elif file_format.lower() in ['xls', 'xlsx', 'excel']:
                        df = pd.read_excel(file_path, **kwargs)
                    elif file_format.lower() == 'json':
                        df = pd.read_json(file_path, **kwargs)
                    elif file_format.lower() == 'parquet':
                        df = pd.read_parquet(file_path, **kwargs)
                    elif file_format.lower() == 'feather':
                        df = pd.read_feather(file_path, **kwargs)
                    elif file_format.lower() == 'hdf':
                        df = pd.read_hdf(file_path, **kwargs)
                    elif file_format.lower() == 'pickle':
                        with open(file_path, 'rb') as f:
                            df = pickle.load(f)
                    else:
                        raise ValueError(f"Unsupported file format: {file_format}")
                    
                    dfs[file_path] = df
                    
                except Exception as e:
                    logger.warning(f"Error loading file {file_path}: {str(e)}")
            
            if not dfs:
                logger.warning("No files were successfully loaded")
                return pd.DataFrame() if concat else {}
            
            logger.info(f"Successfully loaded {len(dfs)} files from {directory}")
            
            # Concatenate if requested
            if concat:
                all_dfs = list(dfs.values())
                result = pd.concat(all_dfs, ignore_index=True)
                logger.info(f"Concatenated into DataFrame with {result.shape[0]} rows and {result.shape[1]} columns")
                return result
            else:
                return dfs
                
        except Exception as e:
            logger.error(f"Error loading multiple files from {directory}: {str(e)}")
            raise
    
    @staticmethod
    def load_yaml(
        filepath: str,
        flat: bool = False
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Load data from YAML file.
        
        Args:
            filepath: Path to YAML file
            flat: Whether to flatten nested structures into DataFrame
            
        Returns:
            DataFrame or dictionary
        """
        try:
            with open(filepath, 'r') as file:
                data = yaml.safe_load(file)
            
            if flat:
                # Convert to DataFrame with flattening of nested structures
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                else:
                    df = pd.json_normalize([data])
                
                logger.info(f"Successfully loaded YAML from {filepath} into DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
                return df
            else:
                logger.info(f"Successfully loaded YAML from {filepath} as dictionary")
                return data
                
        except Exception as e:
            logger.error(f"Error loading YAML from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_model(
        filepath: str,
        format: str = 'auto'
    ) -> Any:
        """Load machine learning model from file.
        
        Args:
            filepath: Path to model file
            format: Model format ('pickle', 'joblib', 'auto')
            
        Returns:
            Loaded model
        """
        try:
            # Determine format if auto
            if format == 'auto':
                ext = os.path.splitext(filepath)[1].lower()
                if ext in ['.pkl', '.pickle']:
                    format = 'pickle'
                elif ext in ['.joblib', '.jl']:
                    format = 'joblib'
                else:
                    format = 'pickle'  # Default to pickle
            
            # Load model
            if format == 'pickle':
                with open(filepath, 'rb') as file:
                    model = pickle.load(file)
            elif format == 'joblib':
                model = joblib.load(filepath)
            else:
                raise ValueError(f"Unsupported model format: {format}")
            
            logger.info(f"Successfully loaded model from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_from_database(
        connection_string: str,
        query_or_table: str,
        is_query: bool = False,
        params: Optional[Dict[str, Any]] = None,
        engine_type: str = 'sqlite',
        limit: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from database.
        
        Args:
            connection_string: Database connection string
            query_or_table: SQL query or table name
            is_query: Whether query_or_table is a query or table name
            params: Parameters for query
            engine_type: Database engine type
            limit: Maximum number of rows to return
            **kwargs: Additional arguments for pd.read_sql
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Create connection based on engine type
            if engine_type == 'sqlite':
                conn = sqlite3.connect(connection_string)
            else:
                # Create SQLAlchemy engine
                engine = sqlalchemy.create_engine(connection_string)
                conn = engine.connect()
            
            try:
                # Modify query to add limit if specified
                if is_query and limit is not None:
                    query = query_or_table
                    # Add LIMIT clause if not already present
                    if 'LIMIT' not in query.upper():
                        query = f"{query} LIMIT {limit}"
                    ##
                    df = pd.read_sql(query, conn, params=params, **kwargs)
                elif not is_query:
                    # Load from table
                    table = query_or_table
                    if limit is not None:
                        df = pd.read_sql(f"SELECT * FROM {table} LIMIT {limit}", conn, **kwargs)
                    else:
                        df = pd.read_sql_table(table, conn, **kwargs)
                else:
                    # Regular query without limit
                    df = pd.read_sql(query_or_table, conn, params=params, **kwargs)
                
                logger.info(f"Successfully loaded data from database with {df.shape[0]} rows and {df.shape[1]} columns")
                return df
                
            finally:
                # Close connection
                conn.close()
                
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            raise
    
    @staticmethod
    def load_from_mongodb(
        connection_string: str,
        database: str,
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, int]] = None,
        limit: Optional[int] = None,
        flatten: bool = True
    ) -> pd.DataFrame:
        """Load data from MongoDB.
        
        Args:
            connection_string: MongoDB connection string
            database: Database name
            collection: Collection name
            query: Query filter
            projection: Fields to include/exclude
            limit: Maximum number of documents
            flatten: Whether to flatten nested documents
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Create MongoDB client
            client = pymongo.MongoClient(connection_string)
            db = client[database]
            collection = db[collection]
            
            # Query documents
            if query is None:
                query = {}
            
            cursor = collection.find(query, projection)
            
            # Apply limit if specified
            if limit is not None:
                cursor = cursor.limit(limit)
            
            # Convert to list of documents
            documents = list(cursor)
            
            # Close connection
            client.close()
            
            if not documents:
                logger.warning("No documents found matching query")
                return pd.DataFrame()
            
            # Convert to DataFrame
            if flatten:
                df = pd.json_normalize(documents)
            else:
                df = pd.DataFrame(documents)
            
            logger.info(f"Successfully loaded {len(documents)} documents from MongoDB with {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from MongoDB: {str(e)}")
            raise
    
    @staticmethod
    def load_from_s3(
        bucket: str,
        key: str,
        file_format: str = 'csv',
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from AWS S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            file_format: Format of the file
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            region_name: AWS region
            **kwargs: Additional arguments for pd.read_* functions
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Create S3 client
            s3 = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
            
            # Get object from S3
            response = s3.get_object(Bucket=bucket, Key=key)
            
            # Load data based on format
            if file_format.lower() == 'csv':
                df = pd.read_csv(response['Body'], **kwargs)
            elif file_format.lower() in ['xls', 'xlsx', 'excel']:
                df = pd.read_excel(BytesIO(response['Body'].read()), **kwargs)
            elif file_format.lower() == 'json':
                df = pd.read_json(response['Body'], **kwargs)
            elif file_format.lower() == 'parquet':
                df = pd.read_parquet(BytesIO(response['Body'].read()), **kwargs)
            elif file_format.lower() == 'feather':
                df = pd.read_feather(BytesIO(response['Body'].read()), **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully loaded {file_format} from S3 bucket {bucket}/{key} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from S3 bucket {bucket}/{key}: {str(e)}")
            raise
    
    @staticmethod
    def load_from_big_query(
        query: str,
        project_id: str,
        credentials_path: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from Google BigQuery.
        
        Args:
            query: SQL query
            project_id: Google Cloud project ID
            credentials_path: Path to service account credentials
            **kwargs: Additional arguments for pd.read_gbq
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Import pandas_gbq
            import pandas_gbq
            
            # Set credentials if provided
            if credentials_path:
                import google.auth
                from google.oauth2 import service_account
                
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                
                # Use pandas_gbq with credentials
                df = pandas_gbq.read_gbq(
                    query,
                    project_id=project_id,
                    credentials=credentials,
                    **kwargs
                )
            else:
                # Use default credentials
                df = pandas_gbq.read_gbq(
                    query,
                    project_id=project_id,
                    **kwargs
                )
            
            logger.info(f"Successfully loaded data from BigQuery with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from BigQuery: {str(e)}")
            raise
    
    @staticmethod
    def load_text_files(
        directory: str,
        pattern: str = '*.txt',
        encoding: str = 'utf-8',
        recursive: bool = False,
        return_filenames: bool = True
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """Load text files from directory.
        
        Args:
            directory: Directory containing text files
            pattern: Glob pattern for files
            encoding: File encoding
            recursive: Whether to search recursively
            return_filenames: Whether to return filenames
            
        Returns:
            List of text contents or tuple of (contents, filenames)
        """
        try:
            # Get list of files
            if recursive:
                glob_pattern = os.path.join(directory, '**', pattern)
                files = glob.glob(glob_pattern, recursive=True)
            else:
                glob_pattern = os.path.join(directory, pattern)
                files = glob.glob(glob_pattern)
            
            if not files:
                logger.warning(f"No files found matching pattern {glob_pattern}")
                return ([], []) if return_filenames else []
            
            # Load each file
            contents = []
            filenames = []
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    
                    contents.append(content)
                    filenames.append(file_path)
                    
                except Exception as e:
                    logger.warning(f"Error loading file {file_path}: {str(e)}")
            
            logger.info(f"Successfully loaded {len(contents)} text files from {directory}")
            
            if return_filenames:
                return contents, filenames
            else:
                return contents
                
        except Exception as e:
            logger.error(f"Error loading text files from {directory}: {str(e)}")
            raise
    
    @staticmethod
    def infer_file_type(filepath: str) -> str:
        """Infer file type from file extension or contents.
        
        Args:
            filepath: Path to file
            
        Returns:
            Detected file format
        """
        # Check file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        if ext in ['.csv']:
            return 'csv'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        elif ext in ['.json']:
            return 'json'
        elif ext in ['.xml']:
            return 'xml'
        elif ext in ['.parquet']:
            return 'parquet'
        elif ext in ['.h5', '.hdf5', '.hdf']:
            return 'hdf'
        elif ext in ['.pkl', '.pickle']:
            return 'pickle'
        elif ext in ['.feather']:
            return 'feather'
        elif ext in ['.txt']:
            return 'text'
        elif ext in ['.yaml', '.yml']:
            return 'yaml'
        
        # Try to infer from content
        try:
            with open(filepath, 'rb') as f:
                header = f.read(4096)
                
            # Check for CSV
            text_header = header.decode('utf-8', errors='ignore')
            if ',' in text_header and '\n' in text_header:
                # Count commas per line to validate CSV
                lines = text_header.split('\n')
                if len(lines) >= 2:
                    comma_counts = [line.count(',') for line in lines[:2]]
                    if comma_counts[0] > 0 and len(set(comma_counts)) == 1:
                        return 'csv'
            
            # Check for JSON
            if header.strip().startswith(b'{') or header.strip().startswith(b'['):
                return 'json'
            
            # Check for XML
            if header.strip().startswith(b'<?xml') or b'<' in header and b'>' in header:
                return 'xml'
            
            # Check for Excel
            if header.startswith(b'PK\x03\x04'):  # XLSX
                return 'excel'
            if header.startswith(b'\xd0\xcf\x11\xe0'):  # XLS
                return 'excel'
            
            # Default to text
            return 'text'
            
        except Exception:
            # If all else fails
            return 'unknown'
    
    @staticmethod
    def load_auto(
        filepath: str,
        **kwargs
    ) -> pd.DataFrame:
        """Automatically detect file type and load data.
        
        Args:
            filepath: Path to file
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Loaded DataFrame
        """
        # Detect file type
        file_type = DataLoader.infer_file_type(filepath)
        
        logger.info(f"Detected file type: {file_type}")
        
        # Load data based on detected type
        try:
            if file_type == 'csv':
                return DataLoader.load_csv(filepath, **kwargs)
            elif file_type == 'excel':
                return DataLoader.load_excel(filepath, **kwargs)
            elif file_type == 'json':
                return DataLoader.load_json(filepath, **kwargs)
            elif file_type == 'xml':
                return DataLoader.load_xml(filepath, **kwargs)
            elif file_type == 'parquet':
                return DataLoader.load_parquet(filepath, **kwargs)
            elif file_type == 'hdf':
                # Need key for HDF5
                if 'key' not in kwargs:
                    logger.warning("HDF5 format requires 'key' parameter")
                    raise ValueError("HDF5 format requires 'key' parameter")
                return DataLoader.load_hdf(filepath, **kwargs)
            elif file_type == 'pickle':
                return DataLoader.load_pickle(filepath)
            elif file_type == 'feather':
                return DataLoader.load_feather(filepath, **kwargs)
            elif file_type == 'yaml':
                return DataLoader.load_yaml(filepath, flat=True)
            elif file_type == 'text':
                # Load as plain text and convert to DataFrame
                with open(filepath, 'r', encoding=kwargs.get('encoding', 'utf-8')) as f:
                    text = f.read()
                return pd.DataFrame({'text': [text]})
            else:
                raise ValueError(f"Unsupported or undetected file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error auto-loading file {filepath}: {str(e)}")
            raise

# Helper functions
def get_file_info(filepath: str) -> Dict[str, Any]:
    """Get information about a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Dictionary with file information
    """
    info = {}
    
    try:
        # Basic file info
        file_stat = os.stat(filepath)
        info['path'] = os.path.abspath(filepath)
        info['size'] = file_stat.st_size
        info['size_mb'] = file_stat.st_size / (1024 * 1024)
        info['modified'] = pd.to_datetime(file_stat.st_mtime, unit='s')
        info['created'] = pd.to_datetime(file_stat.st_ctime, unit='s')
        
        # File type
        info['extension'] = os.path.splitext(filepath)[1].lower()
        info['type'] = DataLoader.infer_file_type(filepath)
        
        # Sample first few lines
        try:
            with open(filepath, 'rb') as f:
                header = f.read(4096)
            info['sample'] = header.decode('utf-8', errors='ignore')
        except Exception:
            info['sample'] = None
            
        return info
        
    except Exception as e:
        logger.error(f"Error getting file info for {filepath}: {str(e)}")
        return {'path': filepath, 'error': str(e)}

def list_directory_contents(
    directory: str,
    pattern: str = '*',
    recursive: bool = False,
    file_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """List contents of a directory as a DataFrame.
    
    Args:
        directory: Directory to list
        pattern: Glob pattern for files
        recursive: Whether to search recursively
        file_types: Filter for specific file types
        
    Returns:
        DataFrame with directory contents information
    """
    try:
        # Get list of files
        if recursive:
            glob_pattern = os.path.join(directory, '**', pattern)
            files = glob.glob(glob_pattern, recursive=True)
        else:
            glob_pattern = os.path.join(directory, pattern)
            files = glob.glob(glob_pattern)
        
        if not files:
            logger.warning(f"No files found matching pattern {glob_pattern}")
            return pd.DataFrame()
        
        # Filter by file types if specified
        if file_types:
            # Convert to lowercase
            file_types = [ft.lower() for ft in file_types]
            filtered_files = []
            for file_path in files:
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Check extension
                ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                if ext in file_types:
                    filtered_files.append(file_path)
            
            files = filtered_files
        
        # Collect information for each file
        file_info = []
        for file_path in files:
            # Skip directories if only listing files
            if os.path.isdir(file_path):
                continue
                
            try:
                info = {
                    'name': os.path.basename(file_path),
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'type': os.path.splitext(file_path)[1].lower().lstrip('.'),
                    'modified': pd.to_datetime(os.path.getmtime(file_path), unit='s'),
                }
                file_info.append(info)
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(file_info)
        
        # Add size in MB
        if 'size' in df.columns:
            df['size_mb'] = df['size'] / (1024 * 1024)
        
        logger.info(f"Found {len(df)} files in {directory}")
        return df
        
    except Exception as e:
        logger.error(f"Error listing directory contents for {directory}: {str(e)}")
        raise