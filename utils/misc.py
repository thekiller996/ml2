"""
Miscellaneous utility functions.
"""

import os
import json
import logging
import time
import datetime
import uuid
import hashlib
import base64
import platform
from typing import Dict, List, Any, Optional, Union, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_version_info() -> Dict[str, str]:
    """Get application version information.
    
    Returns:
        Dictionary with version information
    """
    return {
        "version": "2.0.0",
        "build_date": "2023-12-15",
        "release": "stable"
    }

def generate_unique_id() -> str:
    """Generate a unique identifier.
    
    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())

def get_timestamp() -> str:
    """Get a formatted timestamp string.
    
    Returns:
        Formatted timestamp
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_system_info() -> Dict[str, str]:
    """Get information about the system.
    
    Returns:
        Dictionary with system information
    """
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "machine": platform.machine()
    }

def format_file_size(size_bytes: int) -> str:
    """Format file size in a human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def format_duration(seconds: float) -> str:
    """Format a duration in a human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)} min, {int(remaining_seconds)} sec"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{int(hours)} hr, {int(minutes)} min, {int(seconds)} sec"

def is_valid_json(json_str: str) -> bool:
    """Check if a string is valid JSON.
    
    Args:
        json_str: JSON string to validate
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False

def get_file_extension(file_path: str) -> str:
    """Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension
    """
    return os.path.splitext(file_path)[1].lower()

def calculate_hash(data: Union[str, bytes]) -> str:
    """Calculate SHA-256 hash of data.
    
    Args:
        data: Input data (string or bytes)
        
    Returns:
        Hex digest of hash
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()

def encode_base64(data: Union[str, bytes]) -> str:
    """Encode data as base64.
    
    Args:
        data: Input data (string or bytes)
        
    Returns:
        Base64 encoded string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return base64.b64encode(data).decode('utf-8')

def decode_base64(data: str) -> bytes:
    """Decode base64 string to bytes.
    
    Args:
        data: Base64 encoded string
        
    Returns:
        Decoded bytes
    """
    return base64.b64decode(data)

def memoize(func):
    """Memoization decorator for caching function results.
    
    Args:
        func: Function to memoize
        
    Returns:
        Decorated function
    """
    cache = {}
    
    def wrapper(*args, **kwargs):
        # Create a cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        key_hash = calculate_hash(key)
        
        if key_hash not in cache:
            cache[key_hash] = func(*args, **kwargs)
        
        return cache[key_hash]
    
    return wrapper

def timer(func):
    """Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {format_duration(duration)}")
        
        return result
    
    return wrapper

def retry(max_attempts=3, delay=1):
    """Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            last_error = None
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    last_error = e
                    
                    if attempts < max_attempts:
                        logger.warning(f"Attempt {attempts} failed: {str(e)}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed. Last error: {str(e)}")
            
            raise last_error
        
        return wrapper
    
    return decorator

def create_directory_if_not_exists(directory_path: str) -> bool:
    """Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory was created or already exists, False otherwise
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {str(e)}")
        return False

def get_nested_value(data: Dict[str, Any], path: str, default=None) -> Any:
    """Get a nested value from a dictionary using dot notation.
    
    Args:
        data: Dictionary to search in
        path: Path to the value using dot notation (e.g., "user.name")
        default: Default value to return if path not found
        
    Returns:
        Value at the path or default if not found
    """
    keys = path.split('.')
    result = data
    
    try:
        for key in keys:
            if isinstance(result, dict):
                result = result[key]
            else:
                return default
        return result
    except (KeyError, TypeError):
        return default

def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def parse_query_string(query: str) -> Dict[str, str]:
    """Parse a URL query string into a dictionary.
    
    Args:
        query: Query string (e.g., "name=John&age=30")
        
    Returns:
        Dictionary of query parameters
    """
    result = {}
    
    if not query:
        return result
    
    params = query.split('&')
    for param in params:
        if '=' in param:
            key, value = param.split('=', 1)
            result[key] = value
    
    return result

def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return singular or plural form based on count.
    
    Args:
        count: Count to determine form
        singular: Singular form
        plural: Plural form (defaults to singular + 's')
        
    Returns:
        Appropriate form based on count
    """
    if count == 1:
        return singular
    else:
        return plural if plural is not None else singular + 's'

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of list chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def is_valid_email(email: str) -> bool:
    """Check if a string is a valid email address.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid email, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))