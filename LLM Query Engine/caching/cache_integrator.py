"""
Cache Integrator module for the Conversational Query Engine.
This module connects all caching components and provides a unified interface.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import time

from caching.cache_manager import CacheManager, CachePolicy, CacheRecord
from caching.memory_cache import MemoryCache
from caching.file_cache import FileCache
from caching.redis_cache import RedisCache
from caching.query_cache import QueryCache, SemanticQueryCache
from rag.embeddings_manager import EmbeddingsManager

# Setup logging
logger = logging.getLogger(__name__)

class CacheIntegrator:
    """
    Class that integrates all caching components and provides a unified interface
    for the application to use.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        enable_redis: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        memory_cache_max_size: int = 10000,
        file_cache_max_size: int = 50000,
        query_cache_ttl: int = 3600 * 24 * 7,  # 7 days
        embeddings_cache_ttl: int = 3600 * 24 * 30,  # 30 days
        document_cache_ttl: int = 3600 * 24 * 30,  # 30 days
        enable_semantic_cache: bool = True,
        semantic_similarity_threshold: float = 0.92,
        default_policy: CachePolicy = CachePolicy.LRU
    ):
        """
        Initialize the cache integrator.
        
        Args:
            cache_dir: Base directory for cache storage
            enable_redis: Whether to enable Redis cache
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            redis_password: Redis password
            embeddings_manager: Embeddings manager for semantic caching
            memory_cache_max_size: Max size of memory cache
            file_cache_max_size: Max size of file cache
            query_cache_ttl: TTL for query cache in seconds
            embeddings_cache_ttl: TTL for embeddings cache in seconds
            document_cache_ttl: TTL for document cache in seconds
            enable_semantic_cache: Whether to enable semantic query cache
            semantic_similarity_threshold: Similarity threshold for semantic caching
            default_policy: Default cache eviction policy
        """
        self.cache_dir = cache_dir
        self.enable_redis = enable_redis
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.embeddings_manager = embeddings_manager
        self.enable_semantic_cache = enable_semantic_cache
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.default_policy = default_policy
        
        # Set up caching directories
        self._setup_cache_dirs()
        
        # Initialize cache manager
        self.cache_manager = self._initialize_cache_manager(
            memory_cache_max_size=memory_cache_max_size,
            file_cache_max_size=file_cache_max_size,
            default_policy=default_policy
        )
        
        # Initialize specialized caches
        self.query_cache = self._initialize_query_cache(query_cache_ttl)
        self.embeddings_cache_ttl = embeddings_cache_ttl
        self.document_cache_ttl = document_cache_ttl
    
    def _setup_cache_dirs(self):
        """Set up cache directories."""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "file_cache"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "query_cache"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "embeddings_cache"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "document_cache"), exist_ok=True)
    
    def _initialize_cache_manager(
        self,
        memory_cache_max_size: int,
        file_cache_max_size: int,
        default_policy: CachePolicy
    ) -> CacheManager:
        """
        Initialize the multi-level cache manager.
        
        Args:
            memory_cache_max_size: Max size for memory cache
            file_cache_max_size: Max size for file cache
            default_policy: Default cache eviction policy
            
        Returns:
            Configured cache manager
        """
        # Create cache manager
        cache_manager = CacheManager()
        
        # Add memory cache
        memory_cache = MemoryCache(
            name="memory",
            ttl_seconds=3600,  # 1 hour default TTL for memory cache
            max_size=memory_cache_max_size,
            policy=default_policy
        )
        cache_manager.register_cache(memory_cache)
        
        # Add file cache
        file_cache = FileCache(
            name="file",
            cache_dir=os.path.join(self.cache_dir, "file_cache"),
            ttl_seconds=86400 * 7,  # 7 days default TTL for file cache
            max_size=file_cache_max_size,
            policy=default_policy,
            use_index=True
        )
        cache_manager.register_cache(file_cache)
        
        # Add Redis cache if enabled
        if self.enable_redis:
            try:
                redis_cache = RedisCache(
                    name="redis",
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    ttl_seconds=86400 * 7,  # 7 days default TTL for Redis cache
                    max_size=None,  # Redis handles its own memory management
                    policy=CachePolicy.TTL,
                    prefix="cqe:"
                )
                
                # Only add Redis if connection is successful
                if redis_cache.is_connected():
                    cache_manager.register_cache(redis_cache)
                    logger.info("Redis cache registered successfully")
                else:
                    logger.warning("Redis connection failed, falling back to memory and file caches only")
            except Exception as e:
                logger.error(f"Error initializing Redis cache: {e}")
                logger.warning("Redis cache initialization failed, falling back to memory and file caches only")
        
        return cache_manager
    
    def _initialize_query_cache(self, ttl_seconds: int) -> Union[QueryCache, SemanticQueryCache]:
        """
        Initialize the query cache.
        
        Args:
            ttl_seconds: TTL for query cache
            
        Returns:
            Query cache instance
        """
        cache_layers = ["memory", "file"]
        if self.enable_redis:
            cache_layers.append("redis")
        
        if self.enable_semantic_cache and self.embeddings_manager:
            # Use semantic query cache if embeddings manager is available
            return SemanticQueryCache(
                cache_manager=self.cache_manager,
                embedding_generator=self.embeddings_manager,
                namespace="query_cache",
                ttl_seconds=ttl_seconds,
                similarity_threshold=self.semantic_similarity_threshold,
                cache_layers=cache_layers
            )
        else:
            # Use regular query cache
            return QueryCache(
                cache_manager=self.cache_manager,
                namespace="query_cache",
                ttl_seconds=ttl_seconds,
                cache_layers=cache_layers
            )
    
    def cache_query_result(
        self,
        query: str,
        response: str,
        model: str,
        client_id: str,
        tokens_used: Dict[str, int],
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache a query result.
        
        Args:
            query: Original query text
            response: Generated response
            model: Model used for generation
            client_id: Client identifier
            tokens_used: Token usage statistics
            context: Optional context text
            metadata: Optional metadata
            
        Returns:
            True if cached successfully, False otherwise
        """
        return self.query_cache.set(
            query=query,
            response=response,
            model=model,
            client_id=client_id,
            tokens_used=tokens_used,
            context=context,
            metadata=metadata
        )
    
    def get_cached_query_result(
        self,
        query: str,
        client_id: str,
        model: str,
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached query result.
        
        Args:
            query: Query text
            client_id: Client identifier
            model: Model name
            context: Optional context text
            
        Returns:
            Dictionary with cached result if found, None otherwise
        """
        cache_entry = self.query_cache.get(
            query=query,
            client_id=client_id,
            model=model,
            context=context
        )
        
        if cache_entry:
            return {
                "response": cache_entry.response,
                "model": cache_entry.model,
                "tokens_used": cache_entry.tokens_used,
                "cached": True,
                "cache_age": time.time() - cache_entry.created_at,
                "metadata": cache_entry.metadata
            }
        
        return None
    
    def cache_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache an embedding.
        
        Args:
            text: Original text
            embedding: Generated embedding
            model: Model used for generation
            metadata: Optional metadata
            
        Returns:
            True if cached successfully, False otherwise
        """
        key = f"embedding:{model}:{hash(text)}"
        
        return self.cache_manager.set(
            key=key,
            value=embedding,
            namespace="embeddings_cache",
            ttl_seconds=self.embeddings_cache_ttl,
            metadata={
                "text": text[:100],  # Store truncated text for reference
                "model": model,
                **(metadata or {})
            },
            cache_names=["memory", "file"]
        )
    
    def get_cached_embedding(
        self,
        text: str,
        model: str
    ) -> Optional[List[float]]:
        """
        Get a cached embedding.
        
        Args:
            text: Original text
            model: Model name
            
        Returns:
            Cached embedding if found, None otherwise
        """
        key = f"embedding:{model}:{hash(text)}"
        
        value, cache_name = self.cache_manager.get(
            key=key,
            namespace="embeddings_cache",
            cache_names=["memory", "file"]
        )
        
        return value
    
    def cache_document(
        self,
        document_id: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "document_cache"
    ) -> bool:
        """
        Cache a document.
        
        Args:
            document_id: Document identifier
            content: Document content
            metadata: Optional metadata
            namespace: Cache namespace
            
        Returns:
            True if cached successfully, False otherwise
        """
        return self.cache_manager.set(
            key=document_id,
            value=content,
            namespace=namespace,
            ttl_seconds=self.document_cache_ttl,
            metadata=metadata,
            cache_names=["memory", "file"]
        )
    
    def get_cached_document(
        self,
        document_id: str,
        namespace: str = "document_cache"
    ) -> Optional[Any]:
        """
        Get a cached document.
        
        Args:
            document_id: Document identifier
            namespace: Cache namespace
            
        Returns:
            Cached document if found, None otherwise
        """
        value, cache_name = self.cache_manager.get(
            key=document_id,
            namespace=namespace,
            cache_names=["memory", "file"]
        )
        
        return value
    
    def cache_generic(
        self,
        key: str,
        value: Any,
        namespace: str,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cache_names: Optional[List[str]] = None
    ) -> bool:
        """
        Cache a generic value.
        
        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl_seconds: Optional TTL override
            metadata: Optional metadata
            cache_names: Optional specific cache layers to use
            
        Returns:
            True if cached successfully, False otherwise
        """
        if ttl_seconds is None:
            ttl_seconds = 3600  # 1 hour default
        
        if cache_names is None:
            cache_names = ["memory", "file"]
            if self.enable_redis:
                cache_names.append("redis")
        
        return self.cache_manager.set(
            key=key,
            value=value,
            namespace=namespace,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
            cache_names=cache_names
        )
    
    def get_generic_cached(
        self,
        key: str,
        namespace: str,
        cache_names: Optional[List[str]] = None
    ) -> Optional[Tuple[Any, str]]:
        """
        Get a generic cached value.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            cache_names: Optional specific cache layers to use
            
        Returns:
            Tuple of (value, cache_name) if found, None otherwise
        """
        if cache_names is None:
            cache_names = ["memory", "file"]
            if self.enable_redis:
                cache_names.append("redis")
        
        return self.cache_manager.get(
            key=key,
            namespace=namespace,
            cache_names=cache_names
        )
    
    def delete_from_cache(
        self,
        key: str,
        namespace: str,
        cache_names: Optional[List[str]] = None
    ) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            cache_names: Optional specific cache layers to use
            
        Returns:
            True if deleted, False otherwise
        """
        if cache_names is None:
            cache_names = ["memory", "file"]
            if self.enable_redis:
                cache_names.append("redis")
        
        return self.cache_manager.delete(
            key=key,
            namespace=namespace,
            cache_names=cache_names
        )
    
    def clear_cache(
        self,
        namespace: Optional[str] = None,
        cache_names: Optional[List[str]] = None
    ) -> bool:
        """
        Clear cache contents.
        
        Args:
            namespace: Optional specific namespace to clear
            cache_names: Optional specific cache layers to clear
            
        Returns:
            True if cleared, False otherwise
        """
        if cache_names is None:
            cache_names = ["memory", "file"]
            if self.enable_redis:
                cache_names.append("redis")
        
        return self.cache_manager.clear(
            namespace=namespace,
            cache_names=cache_names
        )
    
    def get_cache_stats(
        self,
        include_query_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            include_query_cache: Whether to include query cache stats
            
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "cache_manager": self.cache_manager.get_stats(),
            "cache_layers": {}
        }
        
        # Get stats for each cache layer
        for cache_name in self.cache_manager.get_registered_cache_names():
            cache = self.cache_manager.get_cache(cache_name)
            if cache:
                if hasattr(cache, "get_stats_extended"):
                    stats["cache_layers"][cache_name] = cache.get_stats_extended()
                else:
                    stats["cache_layers"][cache_name] = cache.get_stats()
        
        # Add query cache stats if requested
        if include_query_cache:
            stats["query_cache"] = self.query_cache.get_stats()
        
        return stats
    
    def cleanup_expired(self) -> Dict[str, int]:
        """
        Clean up expired cache entries.
        
        Returns:
            Dictionary with cleanup results
        """
        results = {}
        
        # Clean up each cache layer
        for cache_name in self.cache_manager.get_registered_cache_names():
            cache = self.cache_manager.get_cache(cache_name)
            if cache and hasattr(cache, "cleanup_expired"):
                removed = cache.cleanup_expired()
                results[cache_name] = removed
        
        return results
    
    def shutdown(self):
        """Perform cleanup operations before shutdown."""
        # Ensure any pending writes are committed
        for cache_name in self.cache_manager.get_registered_cache_names():
            cache = self.cache_manager.get_cache(cache_name)
            if cache and hasattr(cache, "shutdown"):
                cache.shutdown()
