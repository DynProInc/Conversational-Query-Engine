"""
Query cache module for multi-level caching system.
This module provides specialized caching for LLM query results.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, TypeVar, Generic
import time
import logging
import hashlib
import json
import os
from datetime import datetime

from caching.cache_manager import CacheManager, CacheRecord
from utils.cache_utils import generate_cache_key, serialize_object, deserialize_object

# Setup logging
logger = logging.getLogger(__name__)

class QueryCacheEntry:
    """Class representing a cached query result."""
    
    def __init__(
        self,
        query: str,
        response: str,
        model: str,
        client_id: str,
        tokens_used: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None
    ):
        """
        Initialize a query cache entry.
        
        Args:
            query: Original query text
            response: Generated response
            model: Model used for generation
            client_id: Client identifier
            tokens_used: Dictionary with token usage stats
            metadata: Optional metadata
            created_at: Creation timestamp
        """
        self.query = query
        self.response = response
        self.model = model
        self.client_id = client_id
        self.tokens_used = tokens_used
        self.metadata = metadata or {}
        self.created_at = created_at or time.time()
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "query": self.query,
            "response": self.response,
            "model": self.model,
            "client_id": self.client_id,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryCacheEntry':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            QueryCacheEntry instance
        """
        entry = cls(
            query=data["query"],
            response=data["response"],
            model=data["model"],
            client_id=data["client_id"],
            tokens_used=data["tokens_used"],
            metadata=data["metadata"],
            created_at=data["created_at"]
        )
        entry.access_count = data.get("access_count", 0)
        entry.last_accessed = data.get("last_accessed", entry.created_at)
        return entry
    
    def update_access_stats(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()

class QueryCache:
    """Class for caching query results across multiple cache layers."""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        namespace: str = "query_cache",
        ttl_seconds: int = 3600 * 24 * 7,  # 7 days
        exact_match_only: bool = False,
        cache_layers: List[str] = ["memory", "redis", "file"]
    ):
        """
        Initialize the query cache.
        
        Args:
            cache_manager: Multi-level cache manager
            namespace: Cache namespace
            ttl_seconds: Time to live in seconds
            exact_match_only: Whether to require exact query match
            cache_layers: List of cache layers to use
        """
        self.cache_manager = cache_manager
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self.exact_match_only = exact_match_only
        self.cache_layers = cache_layers
        
        # Cache hit stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "insertions": 0,
            "tokens_saved": 0
        }
    
    def _generate_query_key(
        self,
        query: str,
        client_id: str,
        model: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: Query text
            client_id: Client identifier
            model: Model name
            context: Optional context text
            
        Returns:
            Cache key
        """
        components = {
            "query": query,
            "client_id": client_id,
            "model": model
        }
        
        if context:
            components["context"] = context
        
        return generate_cache_key(prefix="query", **components)
    
    def get(
        self,
        query: str,
        client_id: str,
        model: str,
        context: Optional[str] = None
    ) -> Optional[QueryCacheEntry]:
        """
        Get a cached query result.
        
        Args:
            query: Query text
            client_id: Client identifier
            model: Model name
            context: Optional context text
            
        Returns:
            QueryCacheEntry if found, None otherwise
        """
        key = self._generate_query_key(query, client_id, model, context)
        
        value, cache_name = self.cache_manager.get(
            key=key,
            namespace=self.namespace,
            cache_names=self.cache_layers
        )
        
        if value is None:
            self.stats["misses"] += 1
            return None
        
        # Deserialize entry
        try:
            if isinstance(value, dict):
                entry = QueryCacheEntry.from_dict(value)
            else:
                entry = value
            
            # Update access stats
            entry.update_access_stats()
            
            # Store updated entry back to cache
            self._store_entry(entry, key)
            
            self.stats["hits"] += 1
            
            # Track token savings
            input_tokens = entry.tokens_used.get("prompt_tokens", 0)
            output_tokens = entry.tokens_used.get("completion_tokens", 0)
            self.stats["tokens_saved"] += input_tokens + output_tokens
            
            return entry
        
        except Exception as e:
            logger.error(f"Error deserializing query cache entry: {e}")
            self.stats["misses"] += 1
            return None
    
    def set(
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
            query: Query text
            response: Generated response
            model: Model used for generation
            client_id: Client identifier
            tokens_used: Dictionary with token usage stats
            context: Optional context text
            metadata: Optional metadata
            
        Returns:
            True if cached successfully, False otherwise
        """
        # Create cache entry
        entry = QueryCacheEntry(
            query=query,
            response=response,
            model=model,
            client_id=client_id,
            tokens_used=tokens_used,
            metadata=metadata or {}
        )
        
        # If context was provided, include it in the metadata
        if context:
            entry.metadata["context"] = context[:500]  # Truncate long contexts
        
        # Generate cache key
        key = self._generate_query_key(query, client_id, model, context)
        
        # Store entry
        result = self._store_entry(entry, key)
        
        if result:
            self.stats["insertions"] += 1
        
        return result
    
    def _store_entry(self, entry: QueryCacheEntry, key: str) -> bool:
        """
        Store a cache entry.
        
        Args:
            entry: Cache entry to store
            key: Cache key
            
        Returns:
            True if stored successfully, False otherwise
        """
        # Convert entry to dictionary for storage
        entry_dict = entry.to_dict()
        
        # Store in cache
        return self.cache_manager.set(
            key=key,
            value=entry_dict,
            ttl_seconds=self.ttl_seconds,
            namespace=self.namespace,
            metadata={"type": "query_cache", "model": entry.model},
            cache_names=self.cache_layers
        )
    
    def delete(
        self,
        query: str,
        client_id: str,
        model: str,
        context: Optional[str] = None
    ) -> bool:
        """
        Delete a cached query result.
        
        Args:
            query: Query text
            client_id: Client identifier
            model: Model name
            context: Optional context text
            
        Returns:
            True if deleted, False otherwise
        """
        key = self._generate_query_key(query, client_id, model, context)
        
        return self.cache_manager.delete(
            key=key,
            namespace=self.namespace,
            cache_names=self.cache_layers
        )
    
    def clear(self) -> bool:
        """
        Clear all cached query results.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        return self.cache_manager.clear(
            namespace=self.namespace,
            cache_names=self.cache_layers
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = 0
        if (self.stats["hits"] + self.stats["misses"]) > 0:
            hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "layer_stats": self.cache_manager.get_stats(namespace=self.namespace, cache_names=self.cache_layers)
        }
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "insertions": 0,
            "tokens_saved": 0
        }
        
        self.cache_manager.reset_stats(namespace=self.namespace, cache_names=self.cache_layers)

class SemanticQueryCache(QueryCache):
    """
    Extended query cache that supports semantic matching of similar queries.
    Requires an embedding generator for semantic similarity search.
    """
    
    def __init__(
        self,
        cache_manager: CacheManager,
        embedding_generator,  # EmbeddingGenerator instance
        namespace: str = "semantic_query_cache",
        ttl_seconds: int = 3600 * 24 * 7,  # 7 days
        similarity_threshold: float = 0.92,
        cache_layers: List[str] = ["memory", "redis", "file"]
    ):
        """
        Initialize the semantic query cache.
        
        Args:
            cache_manager: Multi-level cache manager
            embedding_generator: Generator for query embeddings
            namespace: Cache namespace
            ttl_seconds: Time to live in seconds
            similarity_threshold: Minimum similarity for match
            cache_layers: List of cache layers to use
        """
        super().__init__(
            cache_manager=cache_manager,
            namespace=namespace,
            ttl_seconds=ttl_seconds,
            exact_match_only=False,
            cache_layers=cache_layers
        )
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
        
        # Additional stats
        self.stats["semantic_hits"] = 0
        self.stats["exact_hits"] = 0
    
    def _generate_embedding_key(self, client_id: str, model: str) -> str:
        """
        Generate a key for storing embeddings.
        
        Args:
            client_id: Client identifier
            model: Model name
            
        Returns:
            Embedding key
        """
        return f"embeddings:{client_id}:{model}"
    
    def _store_query_embedding(
        self,
        query: str,
        embedding: List[float],
        query_key: str,
        client_id: str,
        model: str
    ):
        """
        Store a query embedding for later similarity search.
        
        Args:
            query: Query text
            embedding: Query embedding
            query_key: Key for the cached query result
            client_id: Client identifier
            model: Model name
        """
        embedding_key = self._generate_embedding_key(client_id, model)
        
        # Get existing embeddings
        value, _ = self.cache_manager.get(
            key=embedding_key,
            namespace=self.namespace
        )
        
        embeddings = value or {}
        
        # Add new embedding
        embeddings[query_key] = {
            "query": query,
            "embedding": embedding,
            "created_at": time.time()
        }
        
        # Store updated embeddings
        self.cache_manager.set(
            key=embedding_key,
            value=embeddings,
            ttl_seconds=self.ttl_seconds,
            namespace=self.namespace,
            metadata={"type": "embeddings", "model": model, "client_id": client_id}
        )
    
    def _find_similar_query(
        self,
        query: str,
        query_embedding: List[float],
        client_id: str,
        model: str
    ) -> Optional[str]:
        """
        Find a similar query in the cache.
        
        Args:
            query: Query text
            query_embedding: Query embedding
            client_id: Client identifier
            model: Model name
            
        Returns:
            Cache key of similar query if found, None otherwise
        """
        embedding_key = self._generate_embedding_key(client_id, model)
        
        # Get existing embeddings
        value, _ = self.cache_manager.get(
            key=embedding_key,
            namespace=self.namespace
        )
        
        if not value:
            return None
        
        embeddings = value
        max_similarity = -1
        most_similar_key = None
        
        # Find most similar query
        for key, data in embeddings.items():
            cached_embedding = data.get("embedding")
            if not cached_embedding:
                continue
            
            # Calculate cosine similarity
            similarity = self._calculate_similarity(query_embedding, cached_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_key = key
        
        # Check if similarity is above threshold
        if max_similarity >= self.similarity_threshold:
            return most_similar_key
        
        return None
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Use embedding_generator's similarity function if available
        if hasattr(self.embedding_generator, "calculate_similarity"):
            return self.embedding_generator.calculate_similarity(embedding1, embedding2)
        
        # Otherwise use a simple dot product for normalized embeddings
        return sum(a * b for a, b in zip(embedding1, embedding2))
    
    def get(
        self,
        query: str,
        client_id: str,
        model: str,
        context: Optional[str] = None
    ) -> Optional[QueryCacheEntry]:
        """
        Get a cached query result, using semantic matching if needed.
        
        Args:
            query: Query text
            client_id: Client identifier
            model: Model name
            context: Optional context text
            
        Returns:
            QueryCacheEntry if found, None otherwise
        """
        # First try exact match
        key = self._generate_query_key(query, client_id, model, context)
        
        value, cache_name = self.cache_manager.get(
            key=key,
            namespace=self.namespace,
            cache_names=self.cache_layers
        )
        
        if value is not None:
            # Handle exact match
            try:
                if isinstance(value, dict):
                    entry = QueryCacheEntry.from_dict(value)
                else:
                    entry = value
                
                # Update access stats
                entry.update_access_stats()
                
                # Store updated entry back to cache
                self._store_entry(entry, key)
                
                self.stats["hits"] += 1
                self.stats["exact_hits"] += 1
                
                # Track token savings
                input_tokens = entry.tokens_used.get("prompt_tokens", 0)
                output_tokens = entry.tokens_used.get("completion_tokens", 0)
                self.stats["tokens_saved"] += input_tokens + output_tokens
                
                return entry
            
            except Exception as e:
                logger.error(f"Error deserializing query cache entry: {e}")
        
        # If no exact match and semantic matching is allowed, try semantic match
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Find similar query
            similar_key = self._find_similar_query(query, query_embedding, client_id, model)
            
            if similar_key:
                # Get cached result for similar query
                value, cache_name = self.cache_manager.get(
                    key=similar_key,
                    namespace=self.namespace,
                    cache_names=self.cache_layers
                )
                
                if value is not None:
                    if isinstance(value, dict):
                        entry = QueryCacheEntry.from_dict(value)
                    else:
                        entry = value
                    
                    # Update access stats
                    entry.update_access_stats()
                    
                    # Store updated entry back to cache
                    self._store_entry(entry, similar_key)
                    
                    self.stats["hits"] += 1
                    self.stats["semantic_hits"] += 1
                    
                    # Track token savings
                    input_tokens = entry.tokens_used.get("prompt_tokens", 0)
                    output_tokens = entry.tokens_used.get("completion_tokens", 0)
                    self.stats["tokens_saved"] += input_tokens + output_tokens
                    
                    # Add note about semantic match to metadata
                    entry.metadata["semantic_match"] = True
                    entry.metadata["original_query"] = entry.query
                    entry.query = query  # Update with current query
                    
                    return entry
        
        except Exception as e:
            logger.error(f"Error during semantic query matching: {e}")
        
        self.stats["misses"] += 1
        return None
    
    def set(
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
        Cache a query result with embedding for semantic matching.
        
        Args:
            query: Query text
            response: Generated response
            model: Model used for generation
            client_id: Client identifier
            tokens_used: Dictionary with token usage stats
            context: Optional context text
            metadata: Optional metadata
            
        Returns:
            True if cached successfully, False otherwise
        """
        # Call parent implementation
        result = super().set(query, response, model, client_id, tokens_used, context, metadata)
        
        if result:
            try:
                # Generate and store query embedding
                query_embedding = self.embedding_generator.generate_embedding(query)
                key = self._generate_query_key(query, client_id, model, context)
                self._store_query_embedding(query, query_embedding, key, client_id, model)
            except Exception as e:
                logger.error(f"Error storing query embedding: {e}")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get extended cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = super().get_stats()
        
        # Add semantic-specific stats
        stats["semantic_hit_rate"] = 0
        stats["exact_hit_rate"] = 0
        
        if stats["hits"] > 0:
            stats["semantic_hit_rate"] = self.stats["semantic_hits"] / stats["hits"]
            stats["exact_hit_rate"] = self.stats["exact_hits"] / stats["hits"]
        
        return stats
