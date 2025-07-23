"""
Embeddings management module for RAG system.
This module handles generation, caching, and management of embeddings.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import os
import logging
import time
import hashlib
import json
import numpy as np
from pathlib import Path

from utils.embedding_utils import EmbeddingGenerator
from utils.cache_utils import CacheRecord, generate_cache_key, serialize_object, deserialize_object

# Setup logging
logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Class for managing and caching document and query embeddings."""
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        cache_dir: str = "cache/embeddings",
        cache_duration_seconds: int = 86400 * 30,  # 30 days
        batch_size: int = 16,
        use_disk_cache: bool = True,
        use_memory_cache: bool = True
    ):
        """
        Initialize the embeddings manager.
        
        Args:
            embedding_generator: Generator for embeddings
            cache_dir: Directory for disk cache
            cache_duration_seconds: Duration to keep cache entries
            batch_size: Batch size for embedding generation
            use_disk_cache: Whether to use disk cache
            use_memory_cache: Whether to use memory cache
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.cache_dir = cache_dir
        self.cache_duration_seconds = cache_duration_seconds
        self.batch_size = batch_size
        self.use_disk_cache = use_disk_cache
        self.use_memory_cache = use_memory_cache
        
        # Create cache directory
        if self.use_disk_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # In-memory cache
        self.memory_cache: Dict[str, CacheRecord] = {}
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            File path for cache
        """
        # Use a safe filename derived from the cache key
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def _check_memory_cache(self, cache_key: str) -> Optional[CacheRecord]:
        """
        Check if an entry exists in memory cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cache record if found and valid, None otherwise
        """
        if not self.use_memory_cache:
            return None
        
        record = self.memory_cache.get(cache_key)
        
        if record is None:
            return None
        
        # Check if expired
        if time.time() - record.created_at > self.cache_duration_seconds:
            # Remove expired entry
            del self.memory_cache[cache_key]
            return None
        
        return record
    
    def _check_disk_cache(self, cache_key: str) -> Optional[CacheRecord]:
        """
        Check if an entry exists in disk cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cache record if found and valid, None otherwise
        """
        if not self.use_disk_cache:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            record = CacheRecord(
                key=data.get("key"),
                value=deserialize_object(data.get("value")),
                metadata=data.get("metadata", {}),
                created_at=data.get("created_at", 0)
            )
            
            # Check if expired
            if time.time() - record.created_at > self.cache_duration_seconds:
                # Remove expired entry
                os.remove(cache_path)
                return None
            
            return record
        
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            return None
    
    def _save_to_memory_cache(self, record: CacheRecord):
        """
        Save a record to memory cache.
        
        Args:
            record: Cache record to save
        """
        if not self.use_memory_cache:
            return
        
        self.memory_cache[record.key] = record
    
    def _save_to_disk_cache(self, record: CacheRecord):
        """
        Save a record to disk cache.
        
        Args:
            record: Cache record to save
        """
        if not self.use_disk_cache:
            return
        
        cache_path = self._get_cache_path(record.key)
        
        try:
            data = {
                "key": record.key,
                "value": serialize_object(record.value),
                "metadata": record.metadata,
                "created_at": record.created_at
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
    
    def get_cached_embedding(
        self, 
        text: str, 
        model: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Get a cached embedding for text.
        
        Args:
            text: Text to get embedding for
            model: Optional model name
            
        Returns:
            Cached embedding if found, None otherwise
        """
        # Generate cache key
        cache_key = generate_cache_key(
            prefix="embedding",
            text=text,
            model=model or self.embedding_generator.model_name
        )
        
        # Check memory cache
        record = self._check_memory_cache(cache_key)
        
        if record is not None:
            self.cache_hits += 1
            return record.value
        
        # Check disk cache
        record = self._check_disk_cache(cache_key)
        
        if record is not None:
            # Add to memory cache
            self._save_to_memory_cache(record)
            self.cache_hits += 1
            return record.value
        
        self.cache_misses += 1
        return None
    
    def cache_embedding(
        self, 
        text: str, 
        embedding: List[float], 
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Cache an embedding.
        
        Args:
            text: Text for the embedding
            embedding: Embedding to cache
            model: Optional model name
            metadata: Optional metadata
        """
        # Generate cache key
        cache_key = generate_cache_key(
            prefix="embedding",
            text=text,
            model=model or self.embedding_generator.model_name
        )
        
        # Create cache record
        record = CacheRecord(
            key=cache_key,
            value=embedding,
            metadata={
                "text_hash": hashlib.md5(text.encode()).hexdigest(),
                "model": model or self.embedding_generator.model_name,
                "embedding_dim": len(embedding),
                "text_length": len(text),
                **(metadata or {})
            },
            created_at=time.time()
        )
        
        # Save to caches
        self._save_to_memory_cache(record)
        self._save_to_disk_cache(record)
    
    def generate_embedding(
        self, 
        text: str, 
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> List[float]:
        """
        Generate an embedding for text, using cache if available.
        
        Args:
            text: Text to generate embedding for
            model: Optional model name
            use_cache: Whether to use cache
            
        Returns:
            Embedding for text
        """
        if use_cache:
            # Check cache
            cached_embedding = self.get_cached_embedding(text, model)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        embedding = self.embedding_generator.generate_embedding(text, model)
        
        # Cache embedding
        if use_cache:
            self.cache_embedding(text, embedding, model)
        
        return embedding
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts, using cache when available.
        
        Args:
            texts: Texts to generate embeddings for
            model: Optional model name
            use_cache: Whether to use cache
            show_progress: Whether to show progress
            
        Returns:
            List of embeddings for texts
        """
        if not texts:
            return []
        
        embeddings = []
        texts_to_generate = []
        indices_to_generate = []
        
        if use_cache:
            # Check cache for each text
            for i, text in enumerate(texts):
                cached_embedding = self.get_cached_embedding(text, model)
                
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    texts_to_generate.append(text)
                    indices_to_generate.append(i)
            
            # If all embeddings were cached, return them
            if not texts_to_generate:
                return embeddings
            
            # Otherwise, initialize embeddings list with None placeholders
            embeddings = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                if i not in indices_to_generate:
                    # Fill in cached embeddings
                    cached_embedding = self.get_cached_embedding(texts[i], model)
                    embeddings[i] = cached_embedding
        else:
            texts_to_generate = texts
            indices_to_generate = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Generate embeddings in batches
        total_to_generate = len(texts_to_generate)
        
        if show_progress:
            logger.info(f"Generating {total_to_generate} embeddings in batches of {self.batch_size}")
        
        for i in range(0, total_to_generate, self.batch_size):
            batch_texts = texts_to_generate[i:i+self.batch_size]
            batch_indices = indices_to_generate[i:i+self.batch_size]
            
            if show_progress and i % (self.batch_size * 5) == 0:
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_to_generate+self.batch_size-1)//self.batch_size}")
            
            # Generate batch embeddings
            batch_embeddings = self.embedding_generator.generate_embeddings(batch_texts, model)
            
            # Store and cache embeddings
            for j, (text, embedding, orig_idx) in enumerate(zip(batch_texts, batch_embeddings, batch_indices)):
                embeddings[orig_idx] = embedding
                
                if use_cache:
                    self.cache_embedding(text, embedding, model)
        
        return embeddings
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about cache usage.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "memory_cache_size": len(self.memory_cache) if self.use_memory_cache else 0,
            "disk_cache_entries": len(os.listdir(self.cache_dir)) if self.use_disk_cache and os.path.exists(self.cache_dir) else 0
        }
    
    def clear_memory_cache(self):
        """Clear the memory cache."""
        self.memory_cache = {}
    
    def clear_disk_cache(self):
        """Clear the disk cache."""
        if not self.use_disk_cache or not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.clear_memory_cache()
        self.clear_disk_cache()
    
    def prune_expired_cache_entries(self):
        """Prune expired cache entries."""
        # Prune memory cache
        if self.use_memory_cache:
            keys_to_remove = []
            
            for key, record in self.memory_cache.items():
                if time.time() - record.created_at > self.cache_duration_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
        
        # Prune disk cache
        if self.use_disk_cache and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        created_at = data.get("created_at", 0)
                        
                        if time.time() - created_at > self.cache_duration_seconds:
                            os.remove(file_path)
                    
                    except Exception as e:
                        logger.error(f"Error pruning cache entry {filename}: {e}")
                        # If we can't read the file, remove it
                        os.remove(file_path)

class ConfigurableEmbeddingsManager:
    """A configurable embeddings manager that can switch between embedding providers."""
    
    def __init__(
        self,
        default_provider: str = "sentence-transformers",
        cache_dir: str = "cache/embeddings",
        cache_duration_seconds: int = 86400 * 30,
        batch_size: int = 16,
        use_disk_cache: bool = True,
        use_memory_cache: bool = True,
        providers_config: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the configurable embeddings manager.
        
        Args:
            default_provider: Default embedding provider
            cache_dir: Directory for disk cache
            cache_duration_seconds: Duration to keep cache entries
            batch_size: Batch size for embedding generation
            use_disk_cache: Whether to use disk cache
            use_memory_cache: Whether to use memory cache
            providers_config: Configuration for embedding providers
        """
        self.default_provider = default_provider
        self.cache_dir = cache_dir
        
        # Default provider configs
        self.providers_config = {
            "sentence-transformers": {
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu",
                "use_cache": True
            },
            "openai": {
                "model_name": "text-embedding-3-small",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "use_cache": True,
                "max_retries": 3,
                "timeout": 30
            }
        }
        
        # Update with provided config
        if providers_config:
            for provider, config in providers_config.items():
                if provider in self.providers_config:
                    self.providers_config[provider].update(config)
                else:
                    self.providers_config[provider] = config
        
        # Initialize embedding generators
        self.embedding_generators = {}
        
        # Initialize embedding managers
        self.embedding_managers = {}
        
        # Initialize the default provider
        self._init_provider(self.default_provider)
    
    def _init_provider(self, provider: str):
        """
        Initialize an embedding provider.
        
        Args:
            provider: Provider name
        """
        if provider in self.embedding_generators:
            return
        
        try:
            config = self.providers_config.get(provider, {})
            
            if provider == "sentence-transformers":
                model_name = config.get("model_name", "all-MiniLM-L6-v2")
                device = config.get("device", "cpu")
                
                embedding_generator = EmbeddingGenerator(
                    model_name=model_name,
                    provider="sentence-transformers",
                    device=device
                )
                
                provider_dir = os.path.join(self.cache_dir, f"st_{model_name.replace('/', '_')}")
                
            elif provider == "openai":
                model_name = config.get("model_name", "text-embedding-3-small")
                api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
                
                if not api_key:
                    logger.error("OpenAI API key is required for OpenAI embeddings")
                    return
                
                embedding_generator = EmbeddingGenerator(
                    model_name=model_name,
                    provider="openai",
                    api_key=api_key
                )
                
                provider_dir = os.path.join(self.cache_dir, f"openai_{model_name.replace('/', '_')}")
                
            else:
                logger.error(f"Unknown embedding provider: {provider}")
                return
            
            self.embedding_generators[provider] = embedding_generator
            
            # Create embedding manager
            self.embedding_managers[provider] = EmbeddingsManager(
                embedding_generator=embedding_generator,
                cache_dir=provider_dir,
                cache_duration_seconds=config.get("cache_duration_seconds", 86400 * 30),
                batch_size=config.get("batch_size", 16),
                use_disk_cache=config.get("use_disk_cache", True),
                use_memory_cache=config.get("use_memory_cache", True)
            )
            
        except Exception as e:
            logger.error(f"Error initializing embedding provider {provider}: {e}")
    
    def _get_manager(self, provider: Optional[str] = None) -> EmbeddingsManager:
        """
        Get an embedding manager for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Embedding manager for the provider
        """
        provider = provider or self.default_provider
        
        if provider not in self.embedding_managers:
            self._init_provider(provider)
        
        if provider not in self.embedding_managers:
            logger.warning(f"Using default provider {self.default_provider} instead of {provider}")
            provider = self.default_provider
        
        return self.embedding_managers[provider]
    
    def generate_embedding(
        self, 
        text: str, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> List[float]:
        """
        Generate an embedding for text.
        
        Args:
            text: Text to generate embedding for
            provider: Optional embedding provider
            model: Optional model name
            use_cache: Whether to use cache
            
        Returns:
            Embedding for text
        """
        manager = self._get_manager(provider)
        return manager.generate_embedding(text, model, use_cache)
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to generate embeddings for
            provider: Optional embedding provider
            model: Optional model name
            use_cache: Whether to use cache
            show_progress: Whether to show progress
            
        Returns:
            List of embeddings for texts
        """
        manager = self._get_manager(provider)
        return manager.generate_embeddings(texts, model, use_cache, show_progress)
    
    def clear_all_caches(self):
        """Clear all caches for all providers."""
        for manager in self.embedding_managers.values():
            manager.clear_all_caches()
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about cache usage for all providers.
        
        Returns:
            Dictionary mapping providers to cache statistics
        """
        stats = {}
        
        for provider, manager in self.embedding_managers.items():
            stats[provider] = manager.get_cache_stats()
        
        return stats
    
    def get_available_providers(self) -> List[str]:
        """
        Get available embedding providers.
        
        Returns:
            List of available provider names
        """
        return list(self.providers_config.keys())
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider configuration
        """
        return self.providers_config.get(provider, {})
