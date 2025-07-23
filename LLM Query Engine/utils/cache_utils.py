"""
Cache utilities for Conversational Query Engine.
This module provides common functions for cache management and manipulation.
"""

from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic, Tuple
import json
import hashlib
import time
import logging
from datetime import datetime
import pickle
import os
import base64

# Initialize logger
logger = logging.getLogger(__name__)

# Type variable for generic cache value
T = TypeVar('T')

def generate_cache_key(
    content: Union[str, Dict, List, Any], 
    prefix: str = "", 
    include_timestamp: bool = False
) -> str:
    """
    Generate a deterministic cache key for an object.
    
    Args:
        content: Content to generate key from
        prefix: Optional prefix for the key
        include_timestamp: Whether to include timestamp in key
        
    Returns:
        Cache key string
    """
    # Convert content to JSON string if it's a dict or list
    if isinstance(content, (dict, list)):
        content_str = json.dumps(content, sort_keys=True)
    else:
        content_str = str(content)
    
    # Add timestamp if requested
    if include_timestamp:
        content_str += str(datetime.now().timestamp())
    
    # Generate MD5 hash
    hash_object = hashlib.md5(content_str.encode())
    hash_str = hash_object.hexdigest()
    
    # Add prefix if provided
    if prefix:
        return f"{prefix}:{hash_str}"
    
    return hash_str

def serialize_object(obj: Any) -> str:
    """
    Serialize any Python object to a string using pickle and base64.
    
    Args:
        obj: Python object to serialize
        
    Returns:
        Base64-encoded pickle string
    """
    try:
        # Use pickle to serialize the object
        pickle_data = pickle.dumps(obj)
        # Encode as base64 string
        b64_data = base64.b64encode(pickle_data).decode('utf-8')
        return b64_data
    except Exception as e:
        logger.error(f"Error serializing object: {e}")
        raise

def deserialize_object(data: str) -> Any:
    """
    Deserialize a string back to a Python object.
    
    Args:
        data: Base64-encoded pickle string
        
    Returns:
        Original Python object
    """
    try:
        # Decode base64 string
        pickle_data = base64.b64decode(data)
        # Deserialize using pickle
        obj = pickle.loads(pickle_data)
        return obj
    except Exception as e:
        logger.error(f"Error deserializing object: {e}")
        raise

class CacheRecord(Generic[T]):
    """
    Class representing a cached record with metadata.
    """
    
    def __init__(
        self, 
        key: str,
        value: T,
        created_at: Optional[float] = None,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a cache record.
        
        Args:
            key: Cache key
            value: Cached value
            created_at: Creation timestamp (defaults to current time)
            ttl: Time-to-live in seconds (None for no expiration)
            metadata: Additional metadata for the cache record
        """
        self.key = key
        self.value = value
        self.created_at = created_at if created_at is not None else time.time()
        self.ttl = ttl
        self.metadata = metadata or {}
        self.last_accessed = self.created_at
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """
        Check if the cache record has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.ttl is None:
            return False
        
        return (time.time() - self.created_at) > self.ttl
    
    def access(self) -> None:
        """
        Update access metadata for the record.
        """
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """
        Get the age of the cache record in seconds.
        
        Returns:
            Age in seconds
        """
        return time.time() - self.created_at
    
    def get_time_since_last_access(self) -> float:
        """
        Get time since last access in seconds.
        
        Returns:
            Time since last access in seconds
        """
        return time.time() - self.last_accessed
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert record to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the record
        """
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "metadata": self.metadata,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheRecord':
        """
        Create a cache record from a dictionary.
        
        Args:
            data: Dictionary representation of a cache record
            
        Returns:
            CacheRecord instance
        """
        record = cls(
            key=data["key"],
            value=data["value"],
            created_at=data["created_at"],
            ttl=data["ttl"],
            metadata=data["metadata"]
        )
        record.last_accessed = data.get("last_accessed", record.created_at)
        record.access_count = data.get("access_count", 0)
        return record

def make_file_path_safe(path: str) -> str:
    """
    Convert a string to a file-system safe path.
    
    Args:
        path: Original string
        
    Returns:
        Safe string for file paths
    """
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', path)

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

def generate_semantic_cache_key(
    query: str,
    embeddings_func: Callable[[str], List[float]],
    cached_embeddings: Dict[str, List[float]],
    threshold: float = 0.92
) -> Optional[str]:
    """
    Generate a cache key based on semantic similarity to existing cached queries.
    
    Args:
        query: User query
        embeddings_func: Function to generate embeddings
        cached_embeddings: Dictionary of cached query embeddings
        threshold: Similarity threshold for considering a match
        
    Returns:
        Matching cache key if found, None otherwise
    """
    from utils.embedding_utils import batch_cosine_similarity
    
    # Generate embedding for the query
    query_embedding = embeddings_func(query)
    
    if not cached_embeddings:
        return None
    
    # Get all cached embeddings
    keys = list(cached_embeddings.keys())
    vectors = list(cached_embeddings.values())
    
    # Calculate similarities
    similarities = batch_cosine_similarity(query_embedding, vectors)
    
    # Find highest similarity
    max_sim = max(similarities)
    max_idx = similarities.index(max_sim)
    
    # Check if similarity exceeds threshold
    if max_sim >= threshold:
        return keys[max_idx]
    
    return None

import re  # Add this import to support make_file_path_safe function
