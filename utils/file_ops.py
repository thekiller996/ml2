"""
File operation utilities for the ML Platform.
"""

import os
import shutil
import pickle
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO, TextIO
import tempfile
import hashlib

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_file(content: Union[bytes, str], filepath: Union[str, Path], mode: str = 'wb') -> Path:
    """
    Save content to a file.
    
    Args:
        content: Content to save
        filepath: Path to save to
        mode: File mode ('wb' for binary, 'w' for text)
    
    Returns:
        Path object for the saved file
    """
    path = Path(filepath)
    ensure_dir(path.parent)
    
    with open(path, mode) as f:
        f.write(content)
    
    return path

def load_file(filepath: Union[str, Path], mode: str = 'rb') -> Union[bytes, str]:
    """
    Load file content.
    
    Args:
        filepath: Path to load from
        mode: File mode ('rb' for binary, 'r' for text)
    
    Returns:
        File content as bytes or string
    """
    path = Path(filepath)
    
    with open(path, mode) as f:
        return f.read()

def get_file_extension(filepath: Union[str, Path]) -> str:
    """
    Get the file extension from a path.
    
    Args:
        filepath: File path
    
    Returns:
        File extension (lowercase, without dot)
    """
    return Path(filepath).suffix.lower().lstrip('.')

def list_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to list files from
        pattern: Glob pattern to match
        recursive: Whether to search recursively
    
    Returns:
        List of matching file paths
    """
    path = Path(directory)
    
    if recursive:
        return list(path.glob(f"**/{pattern}"))
    else:
        return list(path.glob(pattern))

def delete_file(filepath: Union[str, Path]) -> bool:
    """
    Delete a file.
    
    Args:
        filepath: Path to delete
    
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        path = Path(filepath)
        if path.exists():
            path.unlink()
            return True
        return False
    except:
        return False

def save_pickle(obj: Any, filepath: Union[str, Path]) -> Path:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save to
    
    Returns:
        Path object for the saved file
    """
    path = Path(filepath)
    ensure_dir(path.parent)
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    
    return path

def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load an object from a pickle file.
    
    Args:
        filepath: Path to load from
    
    Returns:
        Loaded object
    """
    path = Path(filepath)
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(obj: Any, filepath: Union[str, Path], indent: int = 4) -> Path:
    """
    Save an object to a JSON file.
    
    Args:
        obj: Object to save
        filepath: Path to save to
        indent: Indentation level
    
    Returns:
        Path object for the saved file
    """
    path = Path(filepath)
    ensure_dir(path.parent)
    
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)
    
    return path

def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load an object from a JSON file.
    
    Args:
        filepath: Path to load from
    
    Returns:
        Loaded object
    """
    path = Path(filepath)
    
    with open(path, 'r') as f:
        return json.load(f)

def save_yaml(obj: Any, filepath: Union[str, Path]) -> Path:
    """
    Save an object to a YAML file.
    
    Args:
        obj: Object to save
        filepath: Path to save to
    
    Returns:
        Path object for the saved file
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML operations. Install it with 'pip install pyyaml'.")
    
    path = Path(filepath)
    ensure_dir(path.parent)
    
    with open(path, 'w') as f:
        yaml.dump(obj, f)
    
    return path

def load_yaml(filepath: Union[str, Path]) -> Any:
    """
    Load an object from a YAML file.
    
    Args:
        filepath: Path to load from
    
    Returns:
        Loaded object
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML operations. Install it with 'pip install pyyaml'.")
    
    path = Path(filepath)
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Copy a file.
    
    Args:
        src: Source path
        dst: Destination path
    
    Returns:
        Path object for the destination file
    """
    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    
    shutil.copy2(src_path, dst_path)
    return dst_path

def move_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Move a file.
    
    Args:
        src: Source path
        dst: Destination path
    
    Returns:
        Path object for the destination file
    """
    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    
    shutil.move(src_path, dst_path)
    return dst_path

def get_file_hash(filepath: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Calculate the hash of a file.
    
    Args:
        filepath: Path to the file
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256', etc.)
    
    Returns:
        Hexadecimal hash string
    """
    hash_algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512
    }
    
    if algorithm not in hash_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hash_func = hash_algorithms[algorithm]()
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def create_temp_file(content: Union[bytes, str] = None, suffix: str = None) -> Path:
    """
    Create a temporary file.
    
    Args:
        content: Optional content to write to the file
        suffix: Optional file extension
    
    Returns:
        Path to the temporary file
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    
    if content is not None:
        mode = 'wb' if isinstance(content, bytes) else 'w'
        with open(path, mode) as f:
            f.write(content)
    
    return Path(path)

def create_temp_dir() -> Path:
    """
    Create a temporary directory.
    
    Returns:
        Path to the temporary directory
    """
    path = tempfile.mkdtemp()
    return Path(path)
