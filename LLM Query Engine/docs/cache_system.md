# Multi-Level Caching System Documentation

## Overview

The Conversational Query Engine's multi-level caching system provides a comprehensive solution for caching various types of data, including query results, embeddings, and documents. It improves performance, reduces latency, and decreases token usage by reusing previous results when possible.

## Architecture

The system implements a three-tier caching architecture:

1. **In-Memory Cache** (Fastest, Volatile)
   - First level of the cache hierarchy
   - Provides the fastest access times
   - Limited by available RAM
   - Clears on application restart

2. **File-Based Cache** (Medium Speed, Persistent)
   - Second level of the cache hierarchy
   - Provides persistent storage on disk
   - Thread-safe with file locking
   - Survives application restarts

3. **Redis Cache** (Distributed, Shared)
   - Third level of the cache hierarchy
   - Enables caching across multiple instances
   - Supports horizontal scaling
   - Optional dependency

## Key Components

### Cache Implementations

- **Memory Cache**: In-memory caching using Python dictionaries with TTL support
- **File Cache**: JSON-based file storage with indexing and thread-safety
- **Redis Cache**: Distributed caching with Redis backend
- **Query Cache**: Specialized cache for LLM queries with semantic matching

### Management Components

- **Cache Integrator**: Combines multiple cache layers into a unified interface
- **Cache Manager**: Manages cache configurations and operations
- **Cache Service**: Singleton service exposing cache functionality
- **Cache Scheduler**: Manages background maintenance tasks

### Support Systems

- **Cache Monitor**: Tracks cache performance metrics and statistics
- **Cache Analytics**: Analyzes cache usage and generates reports
- **Cache API Routes**: Exposes cache operations through FastAPI endpoints

## Features

### Core Features

- **Multi-level Caching**: Automatically cascade through memory, file, and Redis caches
- **TTL Support**: Configure expiration for all cache entries
- **Size Limits**: Set maximum sizes for each cache layer
- **Eviction Policies**: LRU and TTL-based eviction
- **Thread Safety**: Safe for concurrent access with locking mechanisms
- **Serialization**: Automatic JSON serialization/deserialization

### Advanced Features

- **Semantic Caching**: Match queries by meaning/intent using embeddings
- **Query Result Caching**: Specialized caching for LLM queries
- **Transparent Caching**: Function decorators for automatic caching
- **Background Maintenance**: Scheduled cleanup and optimization
- **Cache Monitoring**: Track hits, misses, and performance metrics
- **Analytics**: Visualization and reporting of cache performance

## Configuration

The caching system is highly configurable via environment variables:

### General Settings

```
# Enable/disable caching
CACHE_ENABLED=true

# Cache cleanup schedule (in hours)
CACHE_CLEANUP_INTERVAL_HOURS=6

# Statistics logging interval (in minutes)
CACHE_STATS_LOG_INTERVAL_MINUTES=60
```

### Memory Cache Settings

```
# Memory cache settings
MEMORY_CACHE_ENABLED=true
MEMORY_CACHE_MAX_SIZE=1000000000  # 1 GB
MEMORY_CACHE_TTL=3600  # 1 hour
```

### File Cache Settings

```
# File cache settings
FILE_CACHE_ENABLED=true
FILE_CACHE_DIR=cache/file_cache
FILE_CACHE_MAX_SIZE=10000000000  # 10 GB
FILE_CACHE_TTL=86400  # 24 hours
```

### Redis Cache Settings

```
# Redis cache settings
REDIS_CACHE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_TTL=604800  # 1 week
```

### Semantic Cache Settings

```
# Semantic cache settings
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.92  # Similarity threshold
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Local model name or "openai"
```

## Usage Examples

### Direct Cache Access

```python
from services.cache_service import CacheService

# Get singleton instance
cache_service = CacheService()

# Store in cache
cache_service.set("my_key", {"data": "value"}, ttl=3600)

# Retrieve from cache
result = cache_service.get("my_key")
```

### Decorator Usage

```python
from services.cache_service import CacheService, cached

cache_service = CacheService()

# Cache function results
@cached(ttl=3600)
def expensive_function(arg1, arg2):
    # ... expensive computation ...
    return result

# Cache with semantic matching
@cached(semantic=True, ttl=3600)
def query_llm(prompt):
    # ... call to LLM API ...
    return response
```

### RAG Integration

```python
from services.rag_service import RAGService

# Get RAG service with integrated caching
rag_service = RAGService()

# Process documents (cached)
doc_id = rag_service.process_document(document)

# Retrieve documents (cached)
results = rag_service.retrieve_documents(query)

# Generate enhanced prompt (cached)
prompt = rag_service.enhance_prompt(query)

# Complete query with RAG (cached)
response = rag_service.query(question)
```

## API Routes

The caching system exposes several API endpoints:

- **GET /cache/stats**: Get cache statistics and metrics
- **POST /cache/clear**: Clear specific cache layers
- **POST /cache/cleanup**: Trigger manual cache cleanup
- **GET /cache/health**: Check cache system health

## Monitoring and Analytics

The caching system includes tools for monitoring and analyzing performance:

- **CacheMonitor**: Real-time tracking of hits, misses, and other metrics
- **CacheAnalytics**: Generate reports and visualizations of cache performance
- **Prometheus Integration**: Export metrics for monitoring systems

## Best Practices

1. **Tune TTL Values**: Set appropriate TTL values based on data volatility
2. **Size Limits**: Configure size limits based on available resources
3. **Semantic Threshold**: Adjust semantic similarity threshold based on desired precision
4. **Monitor Hit Rate**: Regularly check hit rate and adjust settings as needed
5. **Background Cleanup**: Enable scheduled maintenance to prevent stale data

## Troubleshooting

### Common Issues

- **Low Hit Rate**: 
  - Consider enabling semantic caching
  - Check if TTL values are too short
  - Review key generation logic

- **High Memory Usage**: 
  - Reduce memory cache size limit
  - Implement more aggressive TTL policy
  - Use file/Redis caches for larger objects

- **Redis Connection Issues**: 
  - Verify Redis server is running
  - Check connection settings
  - Ensure passwords are correct

- **Performance Degradation**: 
  - Check file cache index integrity
  - Monitor cache cleanup timing
  - Review serialization overhead for large objects

## Implementation Details

### Cache Key Generation

Cache keys are generated using a deterministic approach that considers:
- Function name (if using decorators)
- Arguments and their values
- Query text (normalized for semantic caching)

### Eviction Policies

- **TTL-based**: Entries expire after a configured time-to-live
- **Size-based**: Least recently used (LRU) entries removed when size limits reached

### Thread Safety

- **Memory Cache**: Thread-safe using locks
- **File Cache**: File-level locking with `filelock`
- **Redis Cache**: Inherently thread-safe with connection pooling

### Semantic Matching

Semantic matching uses embedding models to calculate similarity between queries:
1. Convert queries to vector embeddings
2. Calculate cosine similarity between vectors
3. Match if similarity exceeds threshold

## Development

### Adding New Cache Layers

New cache implementations should implement the `BaseCache` interface:

```python
from caching.base_cache import BaseCache

class NewCache(BaseCache):
    def get(self, key):
        # Implementation
        
    def set(self, key, value, ttl=None):
        # Implementation
        
    def delete(self, key):
        # Implementation
        
    # Other required methods
```

### Testing Cache Components

Use the provided test suite to validate cache implementations:

```bash
pytest -xvs test/test_cache.py
```

## Future Enhancements

- **Distributed Cache Synchronization**: Better coordination between distributed instances
- **Adaptive TTL**: Dynamic TTL adjustment based on usage patterns
- **Cache Prewarming**: Proactive caching of frequently accessed items
- **Content-Aware Caching**: Specialized cache strategies for different content types
- **Cache Partitioning**: Separate caches for different data domains
