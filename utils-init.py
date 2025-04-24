"""
Collection of utility functions for file operations, data processing, and other common tasks.
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
import logging
from typing import Dict, List, Any, Union, Optional
import datetime
import uuid
import shutil
import tempfile
import hashlib
import base64
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get information about a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dictionary with file information
    """
    if not os.path.exists(file_path):
        return {'error': f'File not found: {file_path}'}
    
    try:
        file_info = {
            'path': os.path.abspath(file_path),
            'name': os.path.basename(file_path),
            'directory': os.path.dirname(os.path.abspath(file_path)),
            'extension': os.path.splitext(file_path)[1].lower(),
            'size': os.path.getsize(file_path),
            'size_formatted': format_file_size(os.path.getsize(file_path)),
            'created': datetime.datetime.fromtimestamp(os.path.getctime(file_path)),
            'modified': datetime.datetime.fromtimestamp(os.path.getmtime(file_path)),
            'is_directory': os.path.isdir(file_path)
        }
        
        return file_info
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {'error': str(e)}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.2f} MB"
    else:
        return f"{size_bytes/(1024**3):.2f} GB"

def list_directory(directory: str, pattern: Optional[str] = None) -> List[str]:
    """List files in a directory, optionally filtered by pattern.
    
    Args:
        directory: Directory path
        pattern: Optional regex pattern to filter files
    
    Returns:
        List of file paths
    """
    try:
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return []
            
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if os.path.isfile(os.path.join(directory, f))]
        
        if pattern:
            files = [f for f in files if re.search(pattern, os.path.basename(f))]
            
        return files
    except Exception as e:
        logger.error(f"Error listing directory: {str(e)}")
        return []

def ensure_directory_exists(directory: str) -> bool:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        return True
    except Exception as e:
        logger.error(f"Error creating directory: {str(e)}")
        return False

def save_dataframe(df: pd.DataFrame, file_path: str, format: str = 'csv') -> bool:
    """Save DataFrame to file in specified format.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
        format: File format ('csv', 'excel', 'parquet', 'pickle', 'json')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(os.path.abspath(file_path))
        ensure_directory_exists(directory)
        
        # Save based on format
        format = format.lower()
        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'excel':
            df.to_excel(file_path, index=False)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
        elif format == 'pickle':
            df.to_pickle(file_path)
        elif format == 'json':
            df.to_json(file_path, orient='records', lines=False, indent=2)
        else:
            logger.error(f"Unsupported format: {format}")
            return False
            
        logger.info(f"DataFrame saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving DataFrame: {str(e)}")
        return False

def load_dataframe(file_path: str, format: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load DataFrame from file.
    
    Args:
        file_path: Path to the file
        format: File format (if None, infer from extension)
    
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        # Infer format from extension if not specified
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                format = 'csv'
            elif ext in ['.xlsx', '.xls']:
                format = 'excel'
            elif ext == '.parquet':
                format = 'parquet'
            elif ext == '.pkl':
                format = 'pickle'
            elif ext == '.json':
                format = 'json'
            else:
                logger.error(f"Could not infer format from extension: {ext}")
                return None
        
        # Load based on format
        format = format.lower()
        if format == 'csv':
            df = pd.read_csv(file_path)
        elif format == 'excel':
            df = pd.read_excel(file_path)
        elif format == 'parquet':
            df = pd.read_parquet(file_path)
        elif format == 'pickle':
            df = pd.read_pickle(file_path)
        elif format == 'json':
            df = pd.read_json(file_path)
        else:
            logger.error(f"Unsupported format: {format}")
            return None
            
        logger.info(f"DataFrame loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame: {str(e)}")
        return None

def save_model(model: Any, file_path: str) -> bool:
    """Save model to file.
    
    Args:
        model: Model object to save
        file_path: Path to save the model
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(os.path.abspath(file_path))
        ensure_directory_exists(directory)
        
        # Save model
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Model saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

def load_model(file_path: str) -> Any:
    """Load model from file.
    
    Args:
        file_path: Path to the model file
    
    Returns:
        Model object if successful, None otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            return None
            
        # Load model
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def create_backup(file_path: str, backup_dir: Optional[str] = None) -> Optional[str]:
    """Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        backup_dir: Directory to store backup (if None, use original directory)
    
    Returns:
        Path to backup file if successful, None otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        # Determine backup directory
        if backup_dir is None:
            backup_dir = os.path.dirname(os.path.abspath(file_path))
        
        # Ensure backup directory exists
        ensure_directory_exists(backup_dir)
        
        # Create backup filename with timestamp
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{name}_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy file to backup
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        return None

def delete_file(file_path: str, confirm: bool = True) -> bool:
    """Delete a file.
    
    Args:
        file_path: Path to the file to delete
        confirm: Whether to require confirmation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        # Delete file
        os.remove(file_path)
        
        logger.info(f"File deleted: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return False

def compute_hash(file_path: str, algorithm: str = 'sha256') -> Optional[str]:
    """Compute hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
    
    Returns:
        Hash string if successful, None otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        # Choose hash algorithm
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            logger.error(f"Unsupported hash algorithm: {algorithm}")
            return None
            
        # Compute hash
        with open(file_path, 'rb') as f:
            # Read and update hash in chunks
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
                
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash: {str(e)}")
        return None

def generate_temporary_file(prefix: str = 'temp', suffix: str = '.tmp') -> str:
    """Generate a temporary file path.
    
    Args:
        prefix: Prefix for temporary file
        suffix: Suffix for temporary file
    
    Returns:
        Path to temporary file
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        return temp_path
    except Exception as e:
        logger.error(f"Error generating temporary file: {str(e)}")
        return None

def compare_files(file1: str, file2: str) -> Dict[str, Any]:
    """Compare two files.
    
    Args:
        file1: Path to first file
        file2: Path to second file
    
    Returns:
        Dictionary with comparison results
    """
    try:
        # Check if files exist
        if not os.path.exists(file1):
            return {'error': f'File not found: {file1}'}
        if not os.path.exists(file2):
            return {'error': f'File not found: {file2}'}
            
        # Get file info
        info1 = get_file_info(file1)
        info2 = get_file_info(file2)
        
        # Compare sizes
        size_match = info1['size'] == info2['size']
        
        # Compare hashes
        hash1 = compute_hash(file1)
        hash2 = compute_hash(file2)
        hash_match = hash1 == hash2
        
        return {
            'size_match': size_match,
            'hash_match': hash_match,
            'file1_info': info1,
            'file2_info': info2,
            'file1_hash': hash1,
            'file2_hash': hash2,
            'identical': size_match and hash_match
        }
    except Exception as e:
        logger.error(f"Error comparing files: {str(e)}")
        return {'error': str(e)}

def extract_archive(archive_path: str, extract_dir: Optional[str] = None) -> bool:
    """Extract an archive file.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to (if None, use current directory)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(archive_path):
            logger.error(f"Archive not found: {archive_path}")
            return False
            
        # Determine extraction directory
        if extract_dir is None:
            extract_dir = os.path.dirname(os.path.abspath(archive_path))
        
        # Ensure extraction directory exists
        ensure_directory_exists(extract_dir)
        
        # Get archive type
        ext = os.path.splitext(archive_path)[1].lower()
        
        # Extract based on type
        if ext in ['.zip']:
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif ext in ['.tar', '.gz', '.tgz', '.bz2', '.tbz2']:
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            logger.error(f"Unsupported archive type: {ext}")
            return False
            
        logger.info(f"Archive extracted to {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"Error extracting archive: {str(e)}")
        return False

def create_archive(directory: str, archive_path: str, archive_type: str = 'zip') -> bool:
    """Create an archive from a directory.
    
    Args:
        directory: Directory to archive
        archive_path: Path to save the archive
        archive_type: Type of archive ('zip', 'tar', 'gztar', 'bztar')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return False
            
        # Create archive
        base_name = os.path.splitext(archive_path)[0]
        root_dir = os.path.abspath(directory)
        
        shutil.make_archive(base_name, archive_type, root_dir)
        
        logger.info(f"Archive created: {archive_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating archive: {str(e)}")
        return False

def get_mime_type(file_path: str) -> Optional[str]:
    """Get MIME type of a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        MIME type string if successful, None otherwise
    """
    try:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type
    except Exception as e:
        logger.error(f"Error getting MIME type: {str(e)}")
        return None

def encode_file_to_base64(file_path: str) -> Optional[str]:
    """Encode file to base64 string.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Base64 encoded string if successful, None otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        return base64.b64encode(file_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding file: {str(e)}")
        return None

def decode_base64_to_file(base64_string: str, output_path: str) -> bool:
    """Decode base64 string to file.
    
    Args:
        base64_string: Base64 encoded string
        output_path: Path to save the decoded file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(os.path.abspath(output_path))
        ensure_directory_exists(directory)
        
        # Decode base64 string
        file_data = base64.b64decode(base64_string)
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(file_data)
            
        logger.info(f"File saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error decoding base64: {str(e)}")
        return False