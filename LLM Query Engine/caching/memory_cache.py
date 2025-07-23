"""
Memory cache implementation for multi-level caching system.
This module provides an in-memory LRU cache implementation.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, TypeVar, Generic
import time
import logging
import threading
from collections import OrderedDict

from caching.cache_manager import BaseCache, CachePolicy, CacheRecord

# Setup logging
logger = logging.getLogger(__name__)

# Generic type for cache values
T = TypeVar('T')

class MemoryCache(BaseCache[T]):
    """In-memory cache implementation using OrderedDict for LRU functionality."""
    
    def __init__(
        self,
        name: str = "memory",
        ttl_seconds: int = 3600,
        max_size: Optional[int] = 1000,
        policy: CachePolicy = CachePolicy.LRU
    ):
        """
        Initialize the memory cache.
        
        Args:
            name: Name of the cache
            ttl_seconds: Time to live in seconds
            max_size: Maximum size of the cache
            policy: Cache eviction policy
        """
        super().__init__(name, ttl_seconds, max_size, policy)
        self._cache: OrderedDict[str, CacheRecord[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._access_count: Dict[str, int] = {}  # For LFU policy
    
    def get(self, key: str) -> Optional[CacheRecord[T]]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cache record if found and valid, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self.stats["misses"] += 1
                return None
            
            record = self._cache[key]
            
            # Check if expired
            if not self.is_valid(record):
                self._cache.pop(key)
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                return None
            
            # Update based on policy
            if self.policy == CachePolicy.LRU:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
            elif self.policy == CachePolicy.LFU:
                # Increment access count
                self._access_count[key] = self._access_count.get(key, 0) + 1
            
            self.stats["hits"] += 1
            return record
    
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
            if self.max_size and len(self._cache) >= self.max_size and record.key not in self._cache:
                self._evict()
            
            self._cache[record.key] = record
            
            # Reset or initialize access count for LFU
            if self.policy == CachePolicy.LFU:
                self._access_count[record.key] = 0
            
            # Move to end for LRU
            if self.policy == CachePolicy.LRU:
                self._cache.move_to_end(record.key)
            
            self.stats["insertions"] += 1
            return True
    
    def _evict(self):
        """
        Evict an item based on the cache policy.
        """
        if not self._cache:
            return
        
        key_to_evict = None
        
        if self.policy == CachePolicy.LRU:
            # Evict least recently used (first item)
            key_to_evict = next(iter(self._cache))
            
        elif self.policy == CachePolicy.LFU:
            # Evict least frequently used
            if self._access_count:
                key_to_evict = min(self._access_count, key=self._access_count.get)
            else:
                key_to_evict = next(iter(self._cache))
                
        elif self.policy == CachePolicy.FIFO:
            # Evict oldest entry (first item)
            key_to_evict = next(iter(self._cache))
            
        elif self.policy == CachePolicy.TTL:
            # Evict oldest TTL entry
            oldest_time = float('inf')
            for key, record in self._cache.items():
                if record.created_at < oldest_time:
                    oldest_time = record.created_at
                    key_to_evict = key
        
        if key_to_evict:
            self._cache.pop(key_to_evict, None)
            if self.policy == CachePolicy.LFU:
                self._access_count.pop(key_to_evict, None)
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
            if key not in self._cache:
                return False
            
            self._cache.pop(key, None)
            if self.policy == CachePolicy.LFU:
                self._access_count.pop(key, None)
            
            return True
    
    def clear(self) -> bool:
        """
        Clear the cache.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            return True
    
    def contains(self, key: str) -> bool:
        """
        Check if the cache contains a key.
        
        Args:
            key: Cache key
            
        Returns:
            True if the cache contains the key, False otherwise
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            # Check if expired
            record = self._cache[key]
            if not self.is_valid(record):
                self._cache.pop(key)
                self.stats["expirations"] += 1
                return False
            
            return True
    
    def get_size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Current size of the cache
        """
        with self._lock:
            return len(self._cache)
    
    def get_keys(self) -> List[str]:
        """
        Get all keys in the cache.
        
        Returns:
            List of keys
        """
        with self._lock:
            return list(self._cache.keys())
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired_keys = []
            
            for key, record in self._cache.items():
                if now - record.created_at > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._cache.pop(key, None)
                if self.policy == CachePolicy.LFU:
                    self._access_count.pop(key, None)
            
            self.stats["expirations"] += len(expired_keys)
            return len(expired_keys)
    
    def update_ttl(self, ttl_seconds: int):
        """
        Update the TTL for the cache.
        
        Args:
            ttl_seconds: New TTL in seconds
        """
        with self._lock:
            self.ttl_seconds = ttl_seconds
    
    def get_oldest_entry(self) -> Optional[Tuple[str, CacheRecord[T]]]:
        """
        Get the oldest entry in the cache.
        
        Returns:
            Tuple of (key, record) for oldest entry, or None if cache is empty
        """
        with self._lock:
            if not self._cache:
                return None
            
            oldest_time = float('inf')
            oldest_key = None
            oldest_record = None
            
            for key, record in self._cache.items():
                if record.created_at < oldest_time:
                    oldest_time = record.created_at
                    oldest_key = key
                    oldest_record = record
            
            return (oldest_key, oldest_record) if oldest_key else None
