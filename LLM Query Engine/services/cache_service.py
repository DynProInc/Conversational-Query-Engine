"""
Cache service module for the Conversational Query Engine.
This module initializes and provides caching services to the FastAPI application.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import time
from functools import wraps
from datetime import datetime

from caching.cache_manager import CachePolicy
from caching.cache_integrator import CacheIntegrator
from rag.embeddings_manager import EmbeddingsManager

# Setup logging
logger = logging.getLogger(__name__)

class CacheService:
    """
    Service class that provides caching functionality to the FastAPI application.
    Initializes and manages the cache integrator.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one cache service instance."""
        if cls._instance is None:
            cls._instance = super(CacheService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        cache_dir: str = "cache",
        enable_redis: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600 * 24,  # 1 day
        enable_semantic_cache: bool = True
    ):
        """
        Initialize the cache service.
        
        Args:
            embeddings_manager: Embeddings manager instance
            cache_dir: Base directory for cache files
            enable_redis: Whether to use Redis cache
            redis_host: Redis host
            redis_port: Redis port
            redis_password: Redis password
            default_ttl: Default TTL for cached items
            enable_semantic_cache: Whether to enable semantic query cache
        """
        # Only initialize once (singleton)
        if self._initialized:
            return
        
        self.embeddings_manager = embeddings_manager
        self.cache_dir = cache_dir
        self.enable_redis = enable_redis
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.default_ttl = default_ttl
        self.enable_semantic_cache = enable_semantic_cache
        
        # Read Redis configuration from environment if available
        self._configure_from_env()
        
        # Initialize cache integrator
        self.cache_integrator = self._initialize_cache_integrator()
        
        # Track initialization time for statistics
        self.init_time = datetime.now()
        self.cache_stats = {
            "total_queries": 0,
            "cached_queries": 0,
            "tokens_saved": 0,
            "cache_hits_by_model": {},
            "cache_misses_by_model": {},
            "last_cache_cleanup": None
        }
        
        self._initialized = True
        logger.info("Cache service initialized")
    
    def _configure_from_env(self):
        """Configure cache service from environment variables."""
        self.enable_redis = os.environ.get("USE_REDIS_CACHE", "").lower() in ["true", "1", "yes"]
        
        if self.enable_redis:
            self.redis_host = os.environ.get("REDIS_HOST", self.redis_host)
            self.redis_port = int(os.environ.get("REDIS_PORT", self.redis_port))
            self.redis_password = os.environ.get("REDIS_PASSWORD", self.redis_password)
        
        # Additional cache configuration
        cache_ttl_days = os.environ.get("CACHE_TTL_DAYS")
        if cache_ttl_days:
            self.default_ttl = int(cache_ttl_days) * 3600 * 24
        
        self.enable_semantic_cache = os.environ.get("ENABLE_SEMANTIC_CACHE", "").lower() in ["true", "1", "yes"]
    
    def _initialize_cache_integrator(self) -> CacheIntegrator:
        """
        Initialize the cache integrator.
        
        Returns:
            Configured cache integrator
        """
        return CacheIntegrator(
            cache_dir=self.cache_dir,
            enable_redis=self.enable_redis,
            redis_host=self.redis_host,
            redis_port=self.redis_port,
            redis_password=self.redis_password,
            embeddings_manager=self.embeddings_manager,
            query_cache_ttl=self.default_ttl,
            embeddings_cache_ttl=self.default_ttl * 7,  # 7x longer for embeddings
            document_cache_ttl=self.default_ttl * 30,  # 30x longer for documents
            enable_semantic_cache=self.enable_semantic_cache,
            default_policy=CachePolicy.LRU
        )
    
    def get_query_result(
        self,
        query: str,
        client_id: str,
        model: str,
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached query result.
        
        Args:
            query: Original query
            client_id: Client identifier
            model: Model used
            context: Optional conversation context
        
        Returns:
            Cached result if found, None otherwise
        """
        self.cache_stats["total_queries"] += 1
        result = self.cache_integrator.get_cached_query_result(
            query=query,
            client_id=client_id,
            model=model,
            context=context
        )
        
        if result:
            self.cache_stats["cached_queries"] += 1
            self.cache_stats["tokens_saved"] += sum(result.get("tokens_used", {}).values())
            
            # Track by model
            model_hits = self.cache_stats.get("cache_hits_by_model", {})
            model_hits[model] = model_hits.get(model, 0) + 1
            self.cache_stats["cache_hits_by_model"] = model_hits
            
            logger.info(f"Cache hit for query: {query[:50]}... with model {model}")
        else:
            # Track misses by model
            model_misses = self.cache_stats.get("cache_misses_by_model", {})
            model_misses[model] = model_misses.get(model, 0) + 1
            self.cache_stats["cache_misses_by_model"] = model_misses
        
        return result
    
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
            query: Original query
            response: Generated response
            model: Model used
            client_id: Client identifier
            tokens_used: Token usage statistics
            context: Optional conversation context
            metadata: Additional metadata
            
        Returns:
            True if cached successfully, False otherwise
        """
        return self.cache_integrator.cache_query_result(
            query=query,
            response=response,
            model=model,
            client_id=client_id,
            tokens_used=tokens_used,
            context=context,
            metadata=metadata
        )
    
    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text: Text to get embedding for
            model: Embedding model
            
        Returns:
            Cached embedding if found, None otherwise
        """
        return self.cache_integrator.get_cached_embedding(text=text, model=model)
    
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
            model: Model used
            metadata: Additional metadata
            
        Returns:
            True if cached successfully, False otherwise
        """
        return self.cache_integrator.cache_embedding(
            text=text,
            embedding=embedding,
            model=model,
            metadata=metadata
        )
    
    def get_document(self, document_id: str) -> Optional[Any]:
        """
        Get cached document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Cached document if found, None otherwise
        """
        return self.cache_integrator.get_cached_document(document_id=document_id)
    
    def cache_document(
        self,
        document_id: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache a document.
        
        Args:
            document_id: Document identifier
            content: Document content
            metadata: Additional metadata
            
        Returns:
            True if cached successfully, False otherwise
        """
        return self.cache_integrator.cache_document(
            document_id=document_id,
            content=content,
            metadata=metadata
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Get basic stats
        stats = {
            "service_stats": {
                "uptime_seconds": (datetime.now() - self.init_time).total_seconds(),
                "total_queries": self.cache_stats["total_queries"],
                "cached_queries": self.cache_stats["cached_queries"],
                "tokens_saved": self.cache_stats["tokens_saved"],
                "cache_hit_rate": 0,
                "cache_hits_by_model": self.cache_stats["cache_hits_by_model"],
                "cache_misses_by_model": self.cache_stats["cache_misses_by_model"],
                "last_cache_cleanup": self.cache_stats.get("last_cache_cleanup")
            }
        }
        
        # Calculate hit rate
        if stats["service_stats"]["total_queries"] > 0:
            stats["service_stats"]["cache_hit_rate"] = (
                stats["service_stats"]["cached_queries"] / stats["service_stats"]["total_queries"]
            )
        
        # Get detailed cache stats
        integrator_stats = self.cache_integrator.get_cache_stats(include_query_cache=True)
        stats.update(integrator_stats)
        
        return stats
    
    def cleanup_caches(self) -> Dict[str, Any]:
        """
        Clean up expired cache entries.
        
        Returns:
            Results of cleanup operation
        """
        results = self.cache_integrator.cleanup_expired()
        self.cache_stats["last_cache_cleanup"] = datetime.now()
        return results
    
    def clear_caches(self, namespace: Optional[str] = None) -> bool:
        """
        Clear cache contents.
        
        Args:
            namespace: Optional specific namespace to clear
            
        Returns:
            True if cleared successfully, False otherwise
        """
        return self.cache_integrator.clear_cache(namespace=namespace)
    
    def shutdown(self):
        """Perform cleanup before shutdown."""
        self.cache_integrator.shutdown()
        logger.info("Cache service shut down")

# Decorator for caching query results
def cache_llm_query(ttl_seconds: Optional[int] = None):
    """
    Decorator to cache LLM query results.
    
    Args:
        ttl_seconds: Optional TTL override
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract parameters from args or kwargs
            query = kwargs.get("query")
            if query is None and args:
                # Assume first positional arg might be query
                for arg in args:
                    if isinstance(arg, str):
                        query = arg
                        break
            
            # Extract other parameters
            client_id = kwargs.get("client_id", "default")
            model = kwargs.get("model", "default")
            context = kwargs.get("context")
            
            # Get cache service instance
            cache_service = CacheService()
            
            # Try to get cached result
            if query:
                cached_result = cache_service.get_query_result(
                    query=query,
                    client_id=client_id,
                    model=model,
                    context=context
                )
                
                if cached_result:
                    logger.info(f"Using cached result for query: {query[:50]}...")
                    return cached_result
            
            # If not cached, call original function
            result = await func(*args, **kwargs)
            
            # Cache result if appropriate
            if query and result and isinstance(result, dict):
                response = result.get("response")
                tokens_used = result.get("tokens_used", {})
                
                if response:
                    cache_service.cache_query_result(
                        query=query,
                        response=response,
                        model=model,
                        client_id=client_id,
                        tokens_used=tokens_used,
                        context=context,
                        metadata={"source": "decorator"}
                    )
            
            return result
        
        return wrapper
    
    return decorator

# Create singleton instance for import
cache_service = CacheService()
