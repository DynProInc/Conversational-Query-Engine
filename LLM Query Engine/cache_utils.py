"""
Cache utilities for LLM Query Engine

This module provides caching functionality to improve performance by avoiding
redundant API calls and expensive operations for identical requests.
"""
import os
import json
import time
import hashlib
import pickle
from typing import Any, Optional, Dict, Union, Callable
import logging
import functools
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Redis, fallback to None if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    logger.info("Redis not available, falling back to local cache")

# Try to import cachetools for local caching
try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    TTLCache = None
    logger.info("cachetools not available, falling back to filesystem cache")

class CacheConfig:
    """Configuration for cache settings"""
    def __init__(self):
        self.CACHE_TYPE = os.getenv('CACHE_TYPE', 'local')  # 'local' or 'redis'
        self.CACHE_TTL = int(os.getenv('CACHE_TTL', 86400))  # 24 hours default
        self.CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', 10000))
        self.CACHE_DIR = os.getenv('CACHE_DIR', './cache/')
        
        # Redis settings
        self.REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
        self.REDIS_DB = int(os.getenv('REDIS_DB', 0))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
        
        # Cache TTLs for different operations (in seconds)
        self.QUERY_GENERATION_TTL = int(os.getenv('QUERY_GENERATION_TTL', 7200))  # 2 hours
        self.SQL_EXECUTION_TTL = int(os.getenv('SQL_EXECUTION_TTL', 3600))  # 1 hour
        self.API_RESPONSE_TTL = int(os.getenv('API_RESPONSE_TTL', 86400))  # 24 hours

class CacheManager:
    """Unified cache manager supporting multiple backends"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._local_cache = None
        self._redis_client = None
        self._initialize_cache()
        
        # Create cache directory if using filesystem cache
        if self.config.CACHE_TYPE == 'local' and not CACHETOOLS_AVAILABLE:
            os.makedirs(self.config.CACHE_DIR, exist_ok=True)
    
    def _initialize_cache(self):
        """Initialize the appropriate cache backend"""
        if self.config.CACHE_TYPE == 'redis' and REDIS_AVAILABLE:
            try:
                self._redis_client = redis.StrictRedis(
                    host=self.config.REDIS_HOST,
                    port=self.config.REDIS_PORT,
                    db=self.config.REDIS_DB,
                    password=self.config.REDIS_PASSWORD,
                    decode_responses=False  # We need binary data for pickle
                )
                # Test connection
                self._redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}. Falling back to local cache.")
                self._initialize_local_cache()
        else:
            self._initialize_local_cache()
    
    def _initialize_local_cache(self):
        """Initialize local cache backend"""
        if CACHETOOLS_AVAILABLE:
            self._local_cache = TTLCache(
                maxsize=self.config.CACHE_MAX_SIZE,
                ttl=self.config.CACHE_TTL
            )
            logger.info("TTLCache (cachetools) initialized successfully")
        else:
            # Fallback to simple filesystem cache
            os.makedirs(self.config.CACHE_DIR, exist_ok=True)
            logger.info("Filesystem cache initialized successfully")
    
    def _generate_key(self, prompt: str, client_id: str = None, context: Dict[str, Any] = None) -> str:
        """Generate a cache key from prompt and context
        
        Args:
            prompt: The prompt or query string
            client_id: Optional client ID for isolation
            context: Optional context dictionary
            
        Returns:
            A string key for the cache
        """
        # Create a key that includes client_id and context
        if context is None:
            context = {}
            
        # Extract model information from context if present
        model_info = ""
        if context.get("model"):
            model_info = f":{context['model']}"
        
        # Sort context keys to ensure consistent key generation
        context_str = json.dumps(context, sort_keys=True)
        
        # Include client_id in the key for isolation
        # Format: cache:client_id:model:hash
        # This format makes it easier to clear by client_id and ensures model-specific caching
        client_part = client_id if client_id else "default"
        key_content = f"prompt:{prompt}:{context_str}"
        key_hash = hashlib.md5(key_content.encode()).hexdigest()
        
        return f"cache:{client_part}{model_info}:{key_hash}"
    
    def get(self, prompt: str, client_id: str = None, context: Dict = None) -> Any:
        """Get cached result
        
        Args:
            prompt: The prompt or query string
            client_id: Client ID for isolation (required for client isolation)
            context: Optional context dictionary with parameters like model
            
        Returns:
            Cached value or None if not found
        """
        if client_id is None:
            client_id = "default"
            
        key = self._generate_key(prompt, client_id, context)
        
        # Try to get from cache
        if self.config.CACHE_TYPE == 'redis' and self._redis_client:
            result = self._get_from_redis(key)
        elif self._local_cache is not None:
            result = self._get_from_memory(key)
        else:
            result = self._get_from_filesystem(key)
            
        # Log cache hit/miss
        if result:
            logger.info(f"Cache HIT for client '{client_id}': {key}")
            # Prepare the cached response by updating token usage and execution time
            result = self.prepare_cached_response(result)
        else:
            logger.info(f"Cache MISS for client '{client_id}': {key}")
            
        return result
        
    def prepare_cached_response(self, cached_result: Any) -> Any:
        """Prepare a cached response by updating token usage and execution time
        
        Args:
            cached_result: The cached result to prepare
            
        Returns:
            The prepared cached result
        """
        if isinstance(cached_result, dict):
            # Set the cached flag and timestamp
            cached_result["cached"] = True
            cached_result["cache_timestamp"] = datetime.datetime.now().isoformat()
            
            # Set token usage to zero for cached responses to accurately reflect no tokens were used
            if "token_usage" in cached_result:
                cached_result["token_usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            
            # Preserve the original execution time if it exists
            # This shows how long the original query took to process
            # We don't modify this value as it's valuable information
            
            # Add a note in the response to indicate it was served from cache
            if "user_hint" in cached_result:
                cached_result["user_hint"] = f"[CACHED RESPONSE] {cached_result['user_hint']}"
            else:
                cached_result["user_hint"] = "[CACHED RESPONSE] Result served from cache."
                
        return cached_result
    
    def set(self, prompt: str, value: Any, client_id: str = None, context: Dict[str, Any] = None, ttl: int = None, success: bool = True, include_results: bool = False, query_results: Any = None) -> None:
        """Set a value in the cache
        
        Args:
            prompt: The prompt or query string
            value: Value to cache
            client_id: Client ID for isolation (required for client isolation)
            context: Optional context dictionary with parameters like model
            ttl: Optional TTL override (seconds)
            success: Whether the query execution was successful
            include_results: Whether to include query results in cache
            query_results: The results of the query execution
        """
        try:
            if client_id is None:
                client_id = "default"
                
            cache_key = self._generate_key(prompt, client_id, context)
            
            # Add metadata to the cached value
            if isinstance(value, dict):
                value["cached"] = True
                value["cache_timestamp"] = time.time()
                value["client_id"] = client_id
                value["success"] = success  # Add success flag
                if context and "model" in context:
                    value["model"] = context["model"]
                # Optionally include query results
                if include_results and query_results is not None:
                    value["query_results"] = query_results
            else:
                # Wrap non-dict values with metadata
                wrapped_value = {
                    "value": value,
                    "cached": True,
                    "cache_timestamp": time.time(),
                    "client_id": client_id,
                    "success": success  # Add success flag
                }
                if context and "model" in context:
                    wrapped_value["model"] = context["model"]
                # Optionally include query results
                if include_results and query_results is not None:
                    wrapped_value["query_results"] = query_results
                value = wrapped_value
            
            # Use specified TTL or default from config
            ttl_value = ttl if ttl is not None else self.config.CACHE_TTL
            
            # Only cache if the query was successful (unless explicitly overridden)
            if success or context.get("cache_failures", False):
                logger.info(f"Set memory cache for client '{client_id}': {cache_key}, TTL: {ttl_value}s, Success: {success}")
                
                if self._redis_client:
                    self._set_to_redis(cache_key, value, ttl_value)
                elif self._local_cache is not None:
                    self._set_to_memory(cache_key, value, ttl_value)
                else:
                    self._set_to_filesystem(cache_key, value, ttl_value)
            else:
                logger.info(f"Skipping cache for failed query for client '{client_id}': {cache_key}")
                
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            # Continue execution even if caching fails
    
    def _set_to_memory(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Set a value in the memory cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        try:
            # For TTLCache, the TTL is handled automatically
            if isinstance(self._local_cache, TTLCache):
                self._local_cache[key] = value
            else:
                # For regular dict, we need to store the timestamp and TTL for manual expiration
                self._local_cache[key] = {
                    "value": value,
                    "timestamp": time.time(),
                    "ttl": ttl
                }
            logger.info(f"Set memory cache: {key}, TTL: {ttl}s")
        except Exception as e:
            logger.error(f"Memory cache set error: {e}")
    
    def _set_to_redis(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Set a value in the Redis cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        try:
            self._redis_client.set(key, pickle.dumps(value), ex=ttl)
            logger.info(f"Set Redis cache: {key}, TTL: {ttl}s")
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
    
    def _set_to_filesystem(self, key: str, value: Any, ttl: int = None, client_id: str = None, context: Dict = None) -> None:
        """Set a value in the filesystem cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (seconds)
            client_id: Client ID for isolation
            context: Optional context dictionary with parameters like model
        """
        try:
            os.makedirs(self.config.CACHE_DIR, exist_ok=True)
            cache_file = os.path.join(self.config.CACHE_DIR, f"{key}.cache")
            
            # Add expiration time and metadata to the stored data
            cache_data = {
                "value": value,
                "expires_at": time.time() + (ttl or self.config.CACHE_TTL),
                "client_id": client_id,
                "model": context.get("model") if context else None,
                "timestamp": time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"Set filesystem cache for client '{client_id}': {key}, TTL: {ttl or self.config.CACHE_TTL}s")
        except Exception as e:
            logger.error(f"Filesystem cache set error: {e}")
    
    def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get from Redis cache with proper TTL handling
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value or None if not found
        """
        try:
            data = self._redis_client.get(key)
            if data:
                # Extract client_id and model from key for logging
                parts = key.split(":")
                client_id = parts[1] if len(parts) > 1 else "unknown"
                model = parts[2] if len(parts) > 2 else "unknown"
                
                # Get TTL for logging
                ttl_remaining = self._redis_client.ttl(key)
                
                logger.info(f"Redis cache hit for client '{client_id}', model '{model}': {key}, TTL remaining: {ttl_remaining}s")
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
        return None
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get from memory cache with TTL handling"""
        try:
            if key in self._local_cache:
                # Check if we're using TTLCache or regular dict
                if isinstance(self._local_cache, TTLCache):
                    try:
                        # TTLCache handles expiration automatically
                        cached_item = self._local_cache[key]
                        logger.debug(f"Memory cache hit: {key}")
                        return cached_item["value"] if isinstance(cached_item, dict) and "value" in cached_item else cached_item
                    except KeyError:
                        # Key might have been expired by TTLCache
                        logger.debug(f"Memory cache expired (TTLCache): {key}")
                        return None
                else:
                    # For regular dict, check expiration manually
                    cached_item = self._local_cache[key]
                    current_time = time.time()
                    timestamp = cached_item.get("timestamp", 0)
                    ttl = cached_item.get("ttl", self.config.CACHE_TTL)
                    
                    # Check if expired
                    if current_time - timestamp < ttl:
                        logger.debug(f"Memory cache hit: {key}, age: {current_time - timestamp:.1f}s, ttl: {ttl}s")
                        return cached_item["value"]
                    else:
                        # Expired, remove from cache
                        logger.debug(f"Memory cache expired: {key}, age: {current_time - timestamp:.1f}s, ttl: {ttl}s")
                        del self._local_cache[key]
                        return None
            return None
        except Exception as e:
            logger.error(f"Memory cache get error: {e}")
            # Try to clean up the key if there was an error
            try:
                if key in self._local_cache:
                    del self._local_cache[key]
            except:
                pass
        return None
    
    def _get_from_filesystem(self, key: str) -> Optional[Any]:
        """Get from filesystem cache with TTL handling
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value or None if not found or expired
        """
        try:
            file_path = os.path.join(self.config.CACHE_DIR, f"{key}.cache")
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if expired
            current_time = time.time()
            expires_at = cache_data.get("expires_at", 0)
            
            if current_time < expires_at:
                # Valid cache hit
                client_id = cache_data.get("client_id", "unknown")
                model = cache_data.get("model", "unknown")
                age = current_time - cache_data.get("timestamp", current_time)
                ttl = expires_at - cache_data.get("timestamp", current_time - 3600)
                
                logger.info(f"Filesystem cache hit for client '{client_id}', model '{model}': {key}, age: {age:.1f}s, ttl: {ttl:.1f}s")
                return cache_data["value"]
            else:
                # Expired, remove cache file
                logger.debug(f"Filesystem cache expired: {key}, removing file")
                os.remove(file_path)
                return None
        except Exception as e:
            logger.error(f"Filesystem cache get error: {e}")
            # If there's an error reading the cache file, try to remove it
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        return None
    
    def clear_cache(self, pattern: str = None, client_id: str = None, model: str = None):
        """Clear cache entries
        
        Args:
            pattern: Optional pattern to match for clearing specific keys
            client_id: Optional client ID to clear only that client's cache
            model: Optional model name to clear only that model's cache
        """
        if client_id:
            logger.info(f"Clearing cache for client: {client_id}" + (f", model: {model}" if model else ""))
        
        if self._redis_client:
            try:
                # Build the key pattern based on parameters
                key_pattern = "cache:"
                if client_id:
                    key_pattern += f"{client_id}"
                    if model:
                        key_pattern += f":{model}"
                    key_pattern += ":*"
                elif model:
                    key_pattern += f"*:{model}:*"
                else:
                    key_pattern += "*"
                
                # Add additional pattern if specified
                if pattern:
                    key_pattern += f"*{pattern}*"
                
                # Get keys matching the pattern
                keys = self._redis_client.keys(key_pattern)
                
                if keys:
                    self._redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} Redis cache entries")
                else:
                    logger.info("No Redis cache entries matched the pattern")
            except Exception as e:
                logger.error(f"Redis clear cache error: {e}")
                
        elif self._local_cache is not None:
            try:
                # Clear specific patterns
                keys_to_remove = []
                for k in list(self._local_cache.keys()):
                    should_remove = False
                    
                    # Check client_id match
                    if client_id:
                        client_match = f":cache:{client_id}" in k or f"cache:{client_id}:" in k
                        if client_match:
                            # If model is also specified, check for model match
                            if model:
                                model_match = f":{model}:" in k
                                should_remove = model_match
                            else:
                                should_remove = True
                    # If only model is specified
                    elif model:
                        model_match = f":{model}:" in k
                        should_remove = model_match
                    # If neither client_id nor model is specified
                    else:
                        should_remove = True
                    
                    # Additional pattern matching if needed
                    if should_remove and pattern:
                        should_remove = pattern in k
                    
                    if should_remove:
                        keys_to_remove.append(k)
                
                # Remove the identified keys
                for k in keys_to_remove:
                    try:
                        del self._local_cache[k]
                    except KeyError:
                        # Key might have expired already
                        pass
                
                logger.info(f"Cleared {len(keys_to_remove)} memory cache entries for client: {client_id if client_id else 'all'}, model: {model if model else 'all'}")
                
                # If no specific filters, clear all cache
                if not client_id and not model and not pattern:
                    count = len(self._local_cache)
                    self._local_cache.clear()
                    logger.info(f"Cleared all {count} memory cache entries")
            except Exception as e:
                logger.error(f"Memory clear cache error: {e}")
        
        # Filesystem cache clearing
        else:
            try:
                if os.path.exists(self.config.CACHE_DIR):
                    count = 0
                    pattern_str = pattern if pattern else ""
                    client_str = client_id if client_id else ""
                    model_str = model if model else ""
                    
                    for file in os.listdir(self.config.CACHE_DIR):
                        if not file.endswith('.cache'):
                            continue
                            
                        should_delete = False
                        
                        # Check for client_id match
                        if client_id and client_id in file:
                            # If model is also specified, check for model match
                            if model:
                                should_delete = model in file
                            else:
                                should_delete = True
                        # If only model is specified
                        elif model and model in file:
                            should_delete = True
                        # If neither client_id nor model is specified
                        elif not client_id and not model:
                            should_delete = True
                        
                        # Additional pattern matching if needed
                        if should_delete and pattern:
                            should_delete = pattern in file
                        
                        if should_delete:
                            try:
                                os.remove(os.path.join(self.config.CACHE_DIR, file))
                                count += 1
                            except Exception as e:
                                logger.error(f"Error deleting cache file {file}: {e}")
                    
                    logger.info(f"Cleared {count} filesystem cache files")
            except Exception as e:
                logger.error(f"Filesystem clear cache error: {e}")
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        stats = {
            "cache_type": self.config.CACHE_TYPE,
            "ttl": self.config.CACHE_TTL,
            "max_size": self.config.CACHE_MAX_SIZE,
        }
        
        try:
            if self._redis_client:
                # Get Redis stats
                keys = self._redis_client.keys("cache:*")
                stats["total_entries"] = len(keys)
                stats["backend"] = "redis"
                stats["redis_info"] = {
                    "host": self.config.REDIS_HOST,
                    "port": self.config.REDIS_PORT,
                    "db": self.config.REDIS_DB
                }
            elif self._local_cache is not None:
                # Get memory cache stats
                stats["total_entries"] = len(self._local_cache)
                stats["backend"] = "memory"
                stats["current_size"] = len(self._local_cache)
            else:
                # Get filesystem cache stats
                if os.path.exists(self.config.CACHE_DIR):
                    cache_files = [f for f in os.listdir(self.config.CACHE_DIR) if f.endswith('.cache')]
                    stats["total_entries"] = len(cache_files)
                    stats["backend"] = "filesystem"
                    stats["cache_dir"] = self.config.CACHE_DIR
                    
                    # Calculate total size
                    total_size = 0
                    for file in cache_files:
                        file_path = os.path.join(self.config.CACHE_DIR, file)
                        total_size += os.path.getsize(file_path)
                    
                    stats["total_size_bytes"] = total_size
                    stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            stats["error"] = str(e)
            
        return stats

# Global cache manager instance
cache_manager = CacheManager()

def cache_decorator(ttl: int = None, client_context: bool = True):
    """Decorator for caching function results"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            # Extract client_id from kwargs if available
            client_id = kwargs.get('client_id', '') if client_context else ''
            
            # Create a context dictionary with function name and args
            context = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': {k: v for k, v in kwargs.items() if k != 'client_id'}
            }
            
            # Use the first argument as the prompt (typically the NL query)
            prompt = args[0] if args else kwargs.get('prompt', '')
            
            # Try to get from cache
            cached_result = cache_manager.get(prompt, client_id, context)
            if cached_result is not None:
                logger.info(f"Cache HIT for {func.__name__} - client: {client_id}")
                return cached_result
            
            # Execute function and cache result
            logger.info(f"Cache MISS for {func.__name__} - client: {client_id}")
            result = func(*args, **kwargs)
            cache_manager.set(prompt, result, client_id, context, ttl=ttl)
            
            return result
        return wrapper
    return decorator
