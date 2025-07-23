"""
Redis cache implementation for multi-level caching system.
This module provides a Redis-based cache implementation.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, TypeVar, Generic
import time
import logging
import json
import redis
import os
from datetime import timedelta
import threading
from functools import wraps

from caching.cache_manager import BaseCache, CachePolicy, CacheRecord
from utils.cache_utils import serialize_object, deserialize_object

# Setup logging
logger = logging.getLogger(__name__)

# Generic type for cache values
T = TypeVar('T')

def redis_connection_error_handler(default_return_value=None):
    """Decorator to handle Redis connection errors gracefully."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (redis.ConnectionError, redis.TimeoutError, redis.RedisError) as e:
                logger.error(f"Redis error in {func.__name__}: {e}")
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                self._connection_active = False
                return default_return_value
        return wrapper
    return decorator

class RedisCache(BaseCache[T]):
    """Redis-based cache implementation."""
    
    def __init__(
        self,
        name: str = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_seconds: int = 3600,
        max_size: Optional[int] = None,
        policy: CachePolicy = CachePolicy.TTL,
        prefix: str = "cache:",
        serialization_format: str = "json",
        connect_timeout: int = 5
    ):
        """
        Initialize the Redis cache.
        
        Args:
            name: Name of the cache
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            ttl_seconds: Time to live in seconds
            max_size: Maximum items to store
            policy: Cache eviction policy
            prefix: Key prefix for Redis keys
            serialization_format: Format for serialization (json or pickle)
            connect_timeout: Connection timeout in seconds
        """
        super().__init__(name, ttl_seconds, max_size, policy)
        self.host = host
        self.port = port
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.prefix = prefix
        self.serialization_format = serialization_format
        self.connect_timeout = connect_timeout
        self._lock = threading.RLock()
        self._connection_active = False
        self._client = None
        self.stats["errors"] = 0
        
        # Connect to Redis
        self._connect()
    
    def _connect(self):
        """Connect to Redis server."""
        with self._lock:
            if self._connection_active:
                return
            
            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=self.connect_timeout,
                    decode_responses=False  # We handle decoding ourselves
                )
                
                # Test connection
                self._client.ping()
                self._connection_active = True
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
                
            except (redis.ConnectionError, redis.TimeoutError, redis.RedisError) as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._connection_active = False
                self.stats["errors"] += 1
    
    def _prefixed_key(self, key: str) -> str:
        """
        Get a Redis key with prefix.
        
        Args:
            key: Original cache key
            
        Returns:
            Prefixed key for Redis
        """
        return f"{self.prefix}{key}"
    
    @redis_connection_error_handler(None)
    def get(self, key: str) -> Optional[CacheRecord[T]]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cache record if found and valid, None otherwise
        """
        if not self._connection_active:
            self._connect()
            if not self._connection_active:
                self.stats["misses"] += 1
                return None
        
        with self._lock:
            prefixed_key = self._prefixed_key(key)
            
            # Try to get from Redis
            data = self._client.get(prefixed_key)
            
            if data is None:
                self.stats["misses"] += 1
                return None
            
            try:
                # Deserialize data
                if self.serialization_format == "json":
                    data_dict = json.loads(data)
                else:
                    data_dict = deserialize_object(data)
                
                # Create cache record
                record = CacheRecord(
                    key=data_dict["key"],
                    value=deserialize_object(data_dict["value"]),
                    metadata=data_dict["metadata"],
                    created_at=data_dict["created_at"]
                )
                
                # Redis TTL handles expiration, but let's double check
                if not self.is_valid(record):
                    self._client.delete(prefixed_key)
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                    return None
                
                self.stats["hits"] += 1
                return record
            
            except Exception as e:
                logger.error(f"Error deserializing Redis data: {e}")
                self._client.delete(prefixed_key)
                self.stats["misses"] += 1
                return None
    
    @redis_connection_error_handler(False)
    def set(self, record: CacheRecord[T]) -> bool:
        """
        Set a value in the cache.
        
        Args:
            record: Cache record to set
            
        Returns:
            True if set successfully, False otherwise
        """
        if not self._connection_active:
            self._connect()
            if not self._connection_active:
                return False
        
        with self._lock:
            prefixed_key = self._prefixed_key(record.key)
            
            # Check if we need to evict
            if self.max_size:
                current_size = self.get_size()
                if current_size >= self.max_size:
                    self._evict()
            
            try:
                # Serialize record for storage
                if self.serialization_format == "json":
                    data = json.dumps({
                        "key": record.key,
                        "value": serialize_object(record.value),
                        "metadata": record.metadata,
                        "created_at": record.created_at
                    })
                else:
                    data = serialize_object({
                        "key": record.key,
                        "value": serialize_object(record.value),
                        "metadata": record.metadata,
                        "created_at": record.created_at
                    })
                
                # Calculate TTL based on creation time
                ttl_remaining = max(1, int(self.ttl_seconds - (time.time() - record.created_at)))
                
                # Set in Redis with TTL
                result = self._client.setex(
                    prefixed_key,
                    ttl_remaining,
                    data
                )
                
                if result:
                    self.stats["insertions"] += 1
                
                return result
            
            except Exception as e:
                logger.error(f"Error setting value in Redis: {e}")
                return False
    
    @redis_connection_error_handler(False)
    def _evict(self):
        """
        Evict an item based on the cache policy.
        """
        if not self._connection_active:
            return
        
        with self._lock:
            # For Redis, we typically rely on Redis's own eviction policies
            # But we can implement our own if needed
            
            if self.policy == CachePolicy.LRU:
                # Find the oldest key and remove it
                try:
                    keys = self._client.keys(f"{self.prefix}*")
                    if not keys:
                        return
                    
                    # Get all keys with their creation time
                    key_times = []
                    for key in keys:
                        try:
                            data = self._client.get(key)
                            if data:
                                if self.serialization_format == "json":
                                    data_dict = json.loads(data)
                                else:
                                    data_dict = deserialize_object(data)
                                
                                created_at = data_dict.get("created_at", 0)
                                key_times.append((key, created_at))
                        except Exception:
                            continue
                    
                    # Sort by creation time and remove oldest
                    if key_times:
                        oldest_key = sorted(key_times, key=lambda x: x[1])[0][0]
                        self._client.delete(oldest_key)
                        self.stats["evictions"] += 1
                
                except Exception as e:
                    logger.error(f"Error during Redis eviction: {e}")
    
    @redis_connection_error_handler(False)
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        if not self._connection_active:
            self._connect()
            if not self._connection_active:
                return False
        
        with self._lock:
            prefixed_key = self._prefixed_key(key)
            result = self._client.delete(prefixed_key)
            return result > 0
    
    @redis_connection_error_handler(False)
    def clear(self) -> bool:
        """
        Clear the cache.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        if not self._connection_active:
            self._connect()
            if not self._connection_active:
                return False
        
        with self._lock:
            try:
                # Delete all keys with our prefix
                cursor = 0
                while True:
                    cursor, keys = self._client.scan(
                        cursor=cursor, 
                        match=f"{self.prefix}*", 
                        count=100
                    )
                    
                    if keys:
                        self._client.delete(*keys)
                    
                    if cursor == 0:
                        break
                
                return True
            
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
                return False
    
    @redis_connection_error_handler(False)
    def contains(self, key: str) -> bool:
        """
        Check if the cache contains a key.
        
        Args:
            key: Cache key
            
        Returns:
            True if the cache contains the key, False otherwise
        """
        if not self._connection_active:
            self._connect()
            if not self._connection_active:
                return False
        
        with self._lock:
            prefixed_key = self._prefixed_key(key)
            return self._client.exists(prefixed_key) > 0
    
    @redis_connection_error_handler(0)
    def get_size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Current size of the cache
        """
        if not self._connection_active:
            self._connect()
            if not self._connection_active:
                return 0
        
        with self._lock:
            try:
                # Count keys with our prefix
                count = 0
                cursor = 0
                
                while True:
                    cursor, keys = self._client.scan(
                        cursor=cursor, 
                        match=f"{self.prefix}*", 
                        count=100
                    )
                    
                    count += len(keys)
                    
                    if cursor == 0:
                        break
                
                return count
            
            except Exception as e:
                logger.error(f"Error getting Redis cache size: {e}")
                return 0
    
    @redis_connection_error_handler(False)
    def is_connected(self) -> bool:
        """
        Check if connected to Redis.
        
        Returns:
            True if connected, False otherwise
        """
        if not self._connection_active:
            self._connect()
        
        return self._connection_active
    
    @redis_connection_error_handler([])
    def get_keys(self, pattern: str = "*") -> List[str]:
        """
        Get all keys matching a pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List of matching keys (without prefix)
        """
        if not self._connection_active:
            self._connect()
            if not self._connection_active:
                return []
        
        with self._lock:
            try:
                # Get all keys with our prefix and the pattern
                full_pattern = f"{self.prefix}{pattern}"
                keys = []
                cursor = 0
                
                while True:
                    cursor, batch = self._client.scan(
                        cursor=cursor, 
                        match=full_pattern, 
                        count=100
                    )
                    
                    # Remove prefix from keys
                    prefix_len = len(self.prefix)
                    keys.extend(key[prefix_len:].decode('utf-8') for key in batch)
                    
                    if cursor == 0:
                        break
                
                return keys
            
            except Exception as e:
                logger.error(f"Error getting Redis keys: {e}")
                return []
    
    @redis_connection_error_handler({})
    def get_stats_extended(self) -> Dict[str, Any]:
        """
        Get extended statistics for the Redis cache.
        
        Returns:
            Dictionary with extended cache statistics
        """
        basic_stats = self.get_stats()
        
        if not self._connection_active:
            self._connect()
            if not self._connection_active:
                basic_stats["redis_info"] = {"connected": False}
                return basic_stats
        
        try:
            # Get Redis server info
            info = self._client.info()
            basic_stats["redis_info"] = {
                "connected": True,
                "version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "clients": info.get("connected_clients", 0),
                "uptime": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            basic_stats["redis_info"] = {"connected": False, "error": str(e)}
        
        return basic_stats
