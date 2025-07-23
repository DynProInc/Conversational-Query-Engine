"""
Cache Integration Module

This module provides integration between the existing API and the new caching system.
It includes adapters and utilities to make the caching system work with the existing API.
"""

import logging
import os
import time
import json
import functools
from typing import Dict, Any, Optional, Callable, Union

from services.cache_service import CacheService
from services.rag_service import RAGService
from monitoring.cache_monitor import cache_monitor
from services.cache_scheduler import cache_scheduler

# Setup logging
logger = logging.getLogger(__name__)

def initialize_cache_system():
    """
    Initialize the caching system for the API server.
    This should be called during application startup.
    """
    try:
        # Initialize cache service
        cache_service = CacheService()
        logger.info("Cache service initialized")
        
        # Initialize cache monitor
        logger.info("Cache monitor initialized")
        
        # Start cache scheduler if enabled
        if os.environ.get("BACKGROUND_CLEANUP_ENABLED", "true").lower() in ["true", "1", "yes"]:
            cache_scheduler.start()
            logger.info("Cache scheduler started")
        
        # Initialize RAG service
        rag_service = RAGService(cache_service=cache_service)
        logger.info("RAG service initialized")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing cache system: {e}")
        return False

def shutdown_cache_system():
    """
    Shutdown the caching system properly.
    This should be called during application shutdown.
    """
    try:
        # Shutdown cache scheduler
        if cache_scheduler.running:
            cache_scheduler.shutdown()
            logger.info("Cache scheduler shutdown")
        
        # Shutdown cache monitor
        cache_monitor.shutdown()
        logger.info("Cache monitor shutdown")
        
        # Shutdown cache service
        cache_service = CacheService()
        cache_service.shutdown()
        logger.info("Cache service shutdown")
        
        return True
    except Exception as e:
        logger.error(f"Error shutting down cache system: {e}")
        return False

def cached_llm_query(ttl: int = 3600, semantic: bool = True):
    """
    Decorator to cache LLM query results.
    This is specifically designed to work with the existing API.
    
    Args:
        ttl: Time-to-live in seconds
        semantic: Whether to use semantic matching
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client_id from args or kwargs
            client_id = None
            request = None
            
            # Find request object and client_id
            for arg in args:
                if hasattr(arg, 'client_id'):
                    request = arg
                    client_id = getattr(arg, 'client_id')
                    break
            
            if not client_id and 'request' in kwargs and hasattr(kwargs['request'], 'client_id'):
                request = kwargs['request']
                client_id = getattr(request, 'client_id')
            
            if not client_id:
                client_id = "default"
            
            # Construct cache key
            if request and hasattr(request, 'prompt'):
                query = getattr(request, 'prompt')
                
                # Create a normalized key from function name and request
                model = getattr(request, 'model', 'default')
                include_charts = getattr(request, 'include_charts', False)
                execute_query = getattr(request, 'execute_query', True)
                limit_rows = getattr(request, 'limit_rows', 100)
                
                # Try to get a cache key
                key_parts = [
                    func.__name__,
                    client_id,
                    model,
                    query,
                    str(include_charts),
                    str(execute_query),
                    str(limit_rows)
                ]
                cache_key = "llm:" + ":".join(key_parts)
                
                try:
                    # Get cache service
                    cache_service = CacheService()
                    
                    # Try to get from cache
                    start_time = time.time()
                    cached_result = None
                    
                    if semantic:
                        cached_result = cache_service.query_cache.get_semantic(query, namespace=f"client:{client_id}:model:{model}")
                    else:
                        cached_result = cache_service.get(cache_key)
                    
                    cache_lookup_time = time.time() - start_time
                    
                    # If we have a cache hit, return it
                    if cached_result is not None:
                        logger.info(f"Cache hit for query: '{query[:30]}...' (client: {client_id}, model: {model})")
                        
                        # Update cache stats
                        cache_monitor.record_cache_hit(
                            is_semantic=semantic,
                            token_count=len(query.split())
                        )
                        
                        # Add metadata about caching to the response
                        if isinstance(cached_result, dict):
                            cached_result.setdefault("_cache_metadata", {})
                            cached_result["_cache_metadata"].update({
                                "cache_hit": True,
                                "cache_lookup_time_ms": round(cache_lookup_time * 1000, 2),
                                "cache_type": "semantic" if semantic else "exact"
                            })
                            
                            # Include original execution time in total if it was recorded
                            if "execution_time_ms" in cached_result:
                                cached_result["_cache_metadata"]["total_time_ms"] = cached_result["execution_time_ms"]
                        
                        return cached_result
                    
                    # If not in cache, execute the function
                    logger.info(f"Cache miss for query: '{query[:30]}...' (client: {client_id}, model: {model})")
                    cache_monitor.record_cache_miss()
                    
                    # Call the original function
                    result = func(*args, **kwargs)
                    
                    # Store in cache
                    if result:
                        # Track token usage if available
                        token_count = 0
                        if isinstance(result, dict) and "token_usage" in result:
                            if isinstance(result["token_usage"], dict):
                                # Sum up all token counts
                                token_count = sum(result["token_usage"].values())
                            elif isinstance(result["token_usage"], int):
                                token_count = result["token_usage"]
                        
                        # Save to cache
                        if semantic:
                            cache_service.query_cache.set_semantic(
                                query, result, 
                                namespace=f"client:{client_id}:model:{model}",
                                ttl=ttl
                            )
                        else:
                            cache_service.set(cache_key, result, ttl=ttl)
                        
                        # Add metadata about caching to the response
                        if isinstance(result, dict):
                            result.setdefault("_cache_metadata", {})
                            result["_cache_metadata"].update({
                                "cache_hit": False,
                                "cache_lookup_time_ms": round(cache_lookup_time * 1000, 2),
                                "cache_stored": True,
                                "cache_type": "semantic" if semantic else "exact"
                            })
                        
                        # Record token savings for future retrievals
                        cache_monitor.record_token_usage(token_count)
                    
                    return result
                except Exception as e:
                    # Log error but continue with the original function call
                    logger.error(f"Error using cache: {e}")
                    cache_monitor.record_cache_error()
                    return func(*args, **kwargs)
            
            # If we couldn't handle caching, just call the original function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

# Convenience function to convert the existing QueryResponse to a format suitable for caching
def adapt_query_response_for_cache(response):
    """
    Adapt the existing QueryResponse to a format suitable for caching.
    
    Args:
        response: The QueryResponse object
        
    Returns:
        Dict suitable for caching
    """
    if isinstance(response, dict):
        return response
    
    # If it's an object with __dict__, convert to dict
    if hasattr(response, '__dict__'):
        return response.__dict__
    
    # Fallback
    return response
