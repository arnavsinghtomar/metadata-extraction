"""
Helper utility functions
"""

import re
import uuid
from typing import Any


def normalize_name(name: str) -> str:
    """
    Normalize string to be used as table or column name.
    
    Args:
        name: Original name string
        
    Returns:
        Normalized name safe for PostgreSQL
    """
    if not name:
        return "unnamed"
    
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    return name if name else "unnamed"


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_dict_get(data: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        data: Dictionary to search
        *keys: Nested keys to traverse
        default: Default value if key not found
        
    Returns:
        Value at the nested key or default
        
    Example:
        safe_dict_get({"a": {"b": {"c": 1}}}, "a", "b", "c")  # Returns 1
        safe_dict_get({"a": {}}, "a", "b", "c", default=0)    # Returns 0
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
