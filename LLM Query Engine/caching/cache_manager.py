"""
Cache manager module for multi-level caching system.
This module provides a unified interface for managing different cache layers.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Type, TypeVar, Generic, Callable
import logging
import time
import os
import json
from abc import ABC, abstractmethod
from enum import Enum

from utils.cache_utils import CacheRecord, generate_cache_key, serialize_object, deserialize_object

# Setup logging
logger = logging.getLogger(__name__)

# Generic type for cache values
T = TypeVar('T')

class CacheLayer(Enum):
    """Enum for different cache layers."""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    DATABASE = "database"

class CachePolicy(Enum):
    """Enum for different cache policies."""
    LRU = "lru"       # Least Recently Used
    LFU = "lfu"       # Least Frequently Used
    FIFO = "fifo"     # First In First Out
    TTL = "ttl"       # Time To Live

class BaseCache(ABC, Generic[T]):
    """Abstract base class for cache implementations."""
    
    def __init__(
        self,
        name: str,
        ttl_seconds: int = 3600,
        max_size: Optional[int] = None,
        policy: CachePolicy = CachePolicy.LRU
    ):
        """
        Initialize the base cache.
        
        Args:
            name: Name of the cache
            ttl_seconds: Time to live in seconds
            max_size: Maximum size of the cache
            policy: Cache eviction policy
        """
        self.name = name
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.policy = policy
        self.stats = {
            "hits": 0,
            "misses": 0,
            "insertions": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheRecord[T]]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cache record if found and valid, None otherwise
        """
        pass
    
    @abstractmethod
    def set(self, record: CacheRecord[T]) -> bool:
        """
        Set a value in the cache.
        
        Args:
            record: Cache record to set
            
        Returns:
            True if set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear the cache.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def contains(self, key: str) -> bool:
        """
        Check if the cache contains a key.
        
        Args:
            key: Cache key
            
        Returns:
            True if the cache contains the key, False otherwise
        """
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Current size of the cache
        """
        pass
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            **self.stats,
            "size": self.get_size()
        }
    
    def is_valid(self, record: CacheRecord[T]) -> bool:
        """
        Check if a cache record is valid (not expired).
        
        Args:
            record: Cache record to check
            
        Returns:
            True if valid, False otherwise
        """
        return time.time() - record.created_at < self.ttl_seconds
    
    def reset_stats(self):
        """Reset cache statistics."""
        for key in self.stats:
            self.stats[key] = 0

class CacheManager:
    """
    Multi-level cache manager.
    Manages multiple cache layers and provides a unified interface.
    """
    
    def __init__(self, default_ttl_seconds: int = 3600):
        """
        Initialize the cache manager.
        
        Args:
            default_ttl_seconds: Default TTL for cache entries
        """
        self.default_ttl_seconds = default_ttl_seconds
        self.caches: Dict[str, Dict[str, BaseCache]] = {}
        self.default_caches: Dict[str, BaseCache] = {}
    
    def register_cache(
        self,
        cache: BaseCache,
        namespace: Optional[str] = None,
        set_as_default: bool = False
    ):
        """
        Register a cache with the manager.
        
        Args:
            cache: Cache instance to register
            namespace: Optional namespace for the cache
            set_as_default: Whether to set as default cache for namespace
        """
        namespace = namespace or "default"
        
        if namespace not in self.caches:
            self.caches[namespace] = {}
        
        self.caches[namespace][cache.name] = cache
        
        if set_as_default:
            self.default_caches[namespace] = cache
            logger.info(f"Set {cache.name} as default cache for namespace {namespace}")
    
    def get_cache(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> Optional[BaseCache]:
        """
        Get a cache by name and namespace.
        
        Args:
            name: Name of the cache
            namespace: Namespace of the cache
            
        Returns:
            Cache instance if found, None otherwise
        """
        namespace = namespace or "default"
        
        if namespace not in self.caches:
            return None
        
        if name:
            return self.caches[namespace].get(name)
        
        # Return default cache for namespace
        return self.default_caches.get(namespace)
    
    def get(
        self,
        key: str,
        namespace: Optional[str] = None,
        cache_names: Optional[List[str]] = None
    ) -> Tuple[Optional[Any], str]:
        """
        Get a value from the cache hierarchy.
        
        Args:
            key: Cache key
            namespace: Optional namespace
            cache_names: Optional list of cache names to check
            
        Returns:
            Tuple of (value, cache_name) if found, (None, "") otherwise
        """
        namespace = namespace or "default"
        
        if namespace not in self.caches:
            return None, ""
        
        # If cache names specified, check only those caches
        if cache_names:
            for name in cache_names:
                if name in self.caches[namespace]:
                    cache = self.caches[namespace][name]
                    record = cache.get(key)
                    
                    if record and cache.is_valid(record):
                        return record.value, name
        
        # Otherwise check all caches in namespace
        else:
            for name, cache in self.caches[namespace].items():
                record = cache.get(key)
                
                if record and cache.is_valid(record):
                    # Propagate to higher-level caches
                    self._propagate_to_higher_caches(record, name, namespace)
                    return record.value, name
        
        return None, ""
    
    def _propagate_to_higher_caches(
        self,
        record: CacheRecord,
        found_cache_name: str,
        namespace: str
    ):
        """
        Propagate a cache hit to higher-level caches.
        
        Args:
            record: Cache record to propagate
            found_cache_name: Name of cache where record was found
            namespace: Namespace of the cache
        """
        # Define cache hierarchy (from lowest to highest)
        hierarchy = ["file", "redis", "memory"]
        
        # Find position of found cache in hierarchy
        if found_cache_name not in hierarchy:
            return
        
        found_index = hierarchy.index(found_cache_name)
        
        # Propagate to higher-level caches
        for i in range(found_index + 1, len(hierarchy)):
            higher_cache_name = hierarchy[i]
            
            if higher_cache_name in self.caches[namespace]:
                higher_cache = self.caches[namespace][higher_cache_name]
                higher_cache.set(record)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cache_names: Optional[List[str]] = None
    ) -> bool:
        """
        Set a value in the cache hierarchy.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL override
            namespace: Optional namespace
            metadata: Optional metadata
            cache_names: Optional list of cache names to set in
            
        Returns:
            True if set in at least one cache, False otherwise
        """
        namespace = namespace or "default"
        ttl = ttl_seconds or self.default_ttl_seconds
        
        if namespace not in self.caches:
            return False
        
        record = CacheRecord(
            key=key,
            value=value,
            metadata=metadata or {},
            created_at=time.time()
        )
        
        success = False
        
        # If cache names specified, set only in those caches
        if cache_names:
            for name in cache_names:
                if name in self.caches[namespace]:
                    if self.caches[namespace][name].set(record):
                        success = True
        
        # Otherwise set in all caches in namespace
        else:
            for cache in self.caches[namespace].values():
                if cache.set(record):
                    success = True
        
        return success
    
    def delete(
        self,
        key: str,
        namespace: Optional[str] = None,
        cache_names: Optional[List[str]] = None
    ) -> bool:
        """
        Delete a value from the cache hierarchy.
        
        Args:
            key: Cache key
            namespace: Optional namespace
            cache_names: Optional list of cache names to delete from
            
        Returns:
            True if deleted from at least one cache, False otherwise
        """
        namespace = namespace or "default"
        
        if namespace not in self.caches:
            return False
        
        success = False
        
        # If cache names specified, delete only from those caches
        if cache_names:
            for name in cache_names:
                if name in self.caches[namespace]:
                    if self.caches[namespace][name].delete(key):
                        success = True
        
        # Otherwise delete from all caches in namespace
        else:
            for cache in self.caches[namespace].values():
                if cache.delete(key):
                    success = True
        
        return success
    
    def clear(
        self,
        namespace: Optional[str] = None,
        cache_names: Optional[List[str]] = None
    ) -> bool:
        """
        Clear the cache hierarchy.
        
        Args:
            namespace: Optional namespace
            cache_names: Optional list of cache names to clear
            
        Returns:
            True if cleared at least one cache, False otherwise
        """
        if namespace and namespace not in self.caches:
            return False
        
        namespaces = [namespace] if namespace else list(self.caches.keys())
        success = False
        
        for ns in namespaces:
            # If cache names specified, clear only those caches
            if cache_names:
                for name in cache_names:
                    if name in self.caches[ns]:
                        if self.caches[ns][name].clear():
                            success = True
            
            # Otherwise clear all caches in namespace
            else:
                for cache in self.caches[ns].values():
                    if cache.clear():
                        success = True
        
        return success
    
    def get_stats(
        self,
        namespace: Optional[str] = None,
        cache_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for the cache hierarchy.
        
        Args:
            namespace: Optional namespace
            cache_names: Optional list of cache names to get stats for
            
        Returns:
            Dictionary mapping cache names to statistics
        """
        if namespace and namespace not in self.caches:
            return {}
        
        stats = {}
        namespaces = [namespace] if namespace else list(self.caches.keys())
        
        for ns in namespaces:
            ns_stats = {}
            
            # If cache names specified, get stats only for those caches
            if cache_names:
                for name in cache_names:
                    if name in self.caches[ns]:
                        ns_stats[name] = self.caches[ns][name].get_stats()
            
            # Otherwise get stats for all caches in namespace
            else:
                for name, cache in self.caches[ns].items():
                    ns_stats[name] = cache.get_stats()
            
            if namespace:
                stats = ns_stats
            else:
                stats[ns] = ns_stats
        
        return stats
    
    def reset_stats(
        self,
        namespace: Optional[str] = None,
        cache_names: Optional[List[str]] = None
    ):
        """
        Reset statistics for the cache hierarchy.
        
        Args:
            namespace: Optional namespace
            cache_names: Optional list of cache names to reset stats for
        """
        if namespace and namespace not in self.caches:
            return
        
        namespaces = [namespace] if namespace else list(self.caches.keys())
        
        for ns in namespaces:
            # If cache names specified, reset stats only for those caches
            if cache_names:
                for name in cache_names:
                    if name in self.caches[ns]:
                        self.caches[ns][name].reset_stats()
            
            # Otherwise reset stats for all caches in namespace
            else:
                for cache in self.caches[ns].values():
                    cache.reset_stats()
