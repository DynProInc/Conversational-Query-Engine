"""
File cache implementation for multi-level caching system.
This module provides a file-based persistent cache implementation.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, TypeVar, Generic
import time
import logging
import os
import json
import shutil
import threading
import hashlib
from pathlib import Path

from caching.cache_manager import BaseCache, CachePolicy, CacheRecord
from utils.cache_utils import serialize_object, deserialize_object, ensure_directory_exists

# Setup logging
logger = logging.getLogger(__name__)

# Generic type for cache values
T = TypeVar('T')

class FileCache(BaseCache[T]):
    """File-based persistent cache implementation."""
    
    def __init__(
        self,
        name: str = "file",
        cache_dir: str = "cache/file_cache",
        ttl_seconds: int = 86400 * 7,  # 7 days
        max_size: Optional[int] = 1000,
        policy: CachePolicy = CachePolicy.TTL,
        use_index: bool = True
    ):
        """
        Initialize the file cache.
        
        Args:
            name: Name of the cache
            cache_dir: Directory to store cache files
            ttl_seconds: Time to live in seconds
            max_size: Maximum number of files to store
            policy: Cache eviction policy
            use_index: Whether to use an index file for metadata
        """
        super().__init__(name, ttl_seconds, max_size, policy)
        self.cache_dir = cache_dir
        self.use_index = use_index
        self._lock = threading.RLock()
        self._index: Dict[str, Dict[str, Any]] = {}
        
        # Initialize cache directory
        self._init_cache_dir()
    
    def _init_cache_dir(self):
        """Initialize the cache directory and load index if available."""
        # Create cache directory if it doesn't exist
        ensure_directory_exists(self.cache_dir)
        
        # Load index if available
        if self.use_index:
            self._load_index()
    
    def _load_index(self):
        """Load the index file if it exists."""
        index_path = os.path.join(self.cache_dir, "index.json")
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache index: {e}")
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self):
        """Save the index to disk."""
        if not self.use_index:
            return
        
        index_path = os.path.join(self.cache_dir, "index.json")
        
        try:
            with open(index_path, 'w') as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")
    
    def _get_file_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            File path for the cache key
        """
        # Use MD5 hash of the key as filename to avoid invalid characters
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.json")
    
    def get(self, key: str) -> Optional[CacheRecord[T]]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cache record if found and valid, None otherwise
        """
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not os.path.exists(file_path):
                self.stats["misses"] += 1
                return None
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Create cache record
                record = CacheRecord(
                    key=data["key"],
                    value=deserialize_object(data["value"]),
                    metadata=data["metadata"],
                    created_at=data["created_at"]
                )
                
                # Check if expired
                if not self.is_valid(record):
                    os.remove(file_path)
                    
                    # Update index if using index
                    if self.use_index and key in self._index:
                        del self._index[key]
                        self._save_index()
                    
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                    return None
                
                # Update last access time in index if using index
                if self.use_index and key in self._index:
                    self._index[key]["last_accessed"] = time.time()
                    self._save_index()
                
                self.stats["hits"] += 1
                return record
            
            except Exception as e:
                logger.error(f"Error reading from file cache: {e}")
                self.stats["misses"] += 1
                return None
    
    def set(self, record: CacheRecord[T]) -> bool:
        """
        Set a value in the cache.
        
        Args:
            record: Cache record to set
            
        Returns:
            True if set successfully, False otherwise
        """
        with self._lock:
            # Check if we need to evict
            if self.max_size:
                current_size = self.get_size()
                if current_size >= self.max_size:
                    # Only evict if the key doesn't already exist
                    file_path = self._get_file_path(record.key)
                    if not os.path.exists(file_path):
                        self._evict()
            
            file_path = self._get_file_path(record.key)
            
            try:
                # Serialize record for storage
                serialized_data = {
                    "key": record.key,
                    "value": serialize_object(record.value),
                    "metadata": record.metadata,
                    "created_at": record.created_at
                }
                
                # Write to file
                with open(file_path, 'w') as f:
                    json.dump(serialized_data, f)
                
                # Update index if using index
                if self.use_index:
                    self._index[record.key] = {
                        "file_path": file_path,
                        "created_at": record.created_at,
                        "last_accessed": time.time(),
                        "size": os.path.getsize(file_path)
                    }
                    self._save_index()
                
                self.stats["insertions"] += 1
                return True
            
            except Exception as e:
                logger.error(f"Error writing to file cache: {e}")
                return False
    
    def _evict(self):
        """
        Evict an item based on the cache policy.
        """
        if self.use_index and self._index:
            key_to_evict = None
            
            if self.policy == CachePolicy.LRU:
                # Evict least recently used
                key_to_evict = min(self._index.items(), key=lambda x: x[1].get("last_accessed", 0))[0]
                
            elif self.policy == CachePolicy.TTL:
                # Evict oldest entry
                key_to_evict = min(self._index.items(), key=lambda x: x[1].get("created_at", 0))[0]
                
            # Remove the evicted item
            if key_to_evict:
                file_path = self._index[key_to_evict]["file_path"]
                if os.path.exists(file_path):
                    os.remove(file_path)
                del self._index[key_to_evict]
                self._save_index()
                self.stats["evictions"] += 1
                
        else:
            # If not using index, get a list of all files and sort by modification time
            files = []
            for filename in os.listdir(self.cache_dir):
                if filename != "index.json" and filename.endswith(".json"):
                    file_path = os.path.join(self.cache_dir, filename)
                    files.append((file_path, os.path.getmtime(file_path)))
            
            if files:
                # Sort by modification time (oldest first)
                files.sort(key=lambda x: x[1])
                # Remove oldest file
                os.remove(files[0][0])
                self.stats["evictions"] += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not os.path.exists(file_path):
                return False
            
            try:
                # Remove file
                os.remove(file_path)
                
                # Update index if using index
                if self.use_index and key in self._index:
                    del self._index[key]
                    self._save_index()
                
                return True
            
            except Exception as e:
                logger.error(f"Error deleting from file cache: {e}")
                return False
    
    def clear(self) -> bool:
        """
        Clear the cache.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        with self._lock:
            try:
                # Remove all files except index.json
                for filename in os.listdir(self.cache_dir):
                    if filename != "index.json":
                        file_path = os.path.join(self.cache_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                
                # Reset index
                if self.use_index:
                    self._index = {}
                    self._save_index()
                
                return True
            
            except Exception as e:
                logger.error(f"Error clearing file cache: {e}")
                return False
    
    def contains(self, key: str) -> bool:
        """
        Check if the cache contains a key.
        
        Args:
            key: Cache key
            
        Returns:
            True if the cache contains the key, False otherwise
        """
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not os.path.exists(file_path):
                return False
            
            # If using index and the key is in the index, check expiration
            if self.use_index and key in self._index:
                created_at = self._index[key]["created_at"]
                if time.time() - created_at > self.ttl_seconds:
                    # Remove expired entry
                    os.remove(file_path)
                    del self._index[key]
                    self._save_index()
                    self.stats["expirations"] += 1
                    return False
                return True
            
            # Otherwise, read the file to check expiration
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                created_at = data["created_at"]
                
                if time.time() - created_at > self.ttl_seconds:
                    # Remove expired entry
                    os.remove(file_path)
                    
                    # Update index if using index
                    if self.use_index and key in self._index:
                        del self._index[key]
                        self._save_index()
                    
                    self.stats["expirations"] += 1
                    return False
                
                return True
            
            except Exception as e:
                logger.error(f"Error checking file cache: {e}")
                return False
    
    def get_size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Current size of the cache
        """
        if self.use_index:
            return len(self._index)
        
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename != "index.json" and filename.endswith(".json"):
                count += 1
        
        return count
    
    def get_total_size_bytes(self) -> int:
        """
        Get the total size of the cache in bytes.
        
        Returns:
            Total size of the cache in bytes
        """
        total_size = 0
        
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
        
        return total_size
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            removed_count = 0
            now = time.time()
            
            if self.use_index:
                # Use index for faster expiration check
                expired_keys = []
                
                for key, metadata in self._index.items():
                    if now - metadata["created_at"] > self.ttl_seconds:
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    file_path = self._index[key]["file_path"]
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    del self._index[key]
                    removed_count += 1
                
                if removed_count > 0:
                    self._save_index()
            
            else:
                # Check each file in the directory
                for filename in os.listdir(self.cache_dir):
                    if filename != "index.json" and filename.endswith(".json"):
                        file_path = os.path.join(self.cache_dir, filename)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            created_at = data["created_at"]
                            
                            if now - created_at > self.ttl_seconds:
                                # Remove expired entry
                                os.remove(file_path)
                                removed_count += 1
                        
                        except Exception as e:
                            logger.error(f"Error checking expiration for {file_path}: {e}")
                            # Remove invalid files
                            os.remove(file_path)
                            removed_count += 1
            
            self.stats["expirations"] += removed_count
            return removed_count
    
    def rebuild_index(self) -> int:
        """
        Rebuild the index from the cache directory.
        
        Returns:
            Number of entries indexed
        """
        if not self.use_index:
            return 0
        
        with self._lock:
            self._index = {}
            
            for filename in os.listdir(self.cache_dir):
                if filename != "index.json" and filename.endswith(".json"):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        key = data["key"]
                        self._index[key] = {
                            "file_path": file_path,
                            "created_at": data["created_at"],
                            "last_accessed": time.time(),
                            "size": os.path.getsize(file_path)
                        }
                    
                    except Exception as e:
                        logger.error(f"Error rebuilding index for {file_path}: {e}")
            
            self._save_index()
            return len(self._index)
    
    def get_stats_extended(self) -> Dict[str, Any]:
        """
        Get extended statistics for the file cache.
        
        Returns:
            Dictionary with extended cache statistics
        """
        basic_stats = self.get_stats()
        
        # Add file-specific stats
        basic_stats.update({
            "total_size_bytes": self.get_total_size_bytes(),
            "cache_dir": self.cache_dir,
            "using_index": self.use_index
        })
        
        return basic_stats
