"""
Simple LLM query cache decorator for the API server.
This provides a lightweight caching mechanism without complex dependencies.
"""

import os
import json
import time
import hashlib
import logging
from typing import Any, Dict, Optional, Callable, List
import functools
import datetime
import pickle
from pathlib import Path
from filelock import FileLock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache stats
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "errors": 0
}

def get_cache_stats():
    """Get current cache statistics."""
    hit_rate = 0
    total = _cache_stats["hits"] + _cache_stats["misses"]
    if total > 0:
        hit_rate = _cache_stats["hits"] / total * 100
    
    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "errors": _cache_stats["errors"],
        "hit_rate": hit_rate
    }

def reset_cache_stats():
    """Reset cache statistics."""
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0
    _cache_stats["errors"] = 0

def _get_cache_dir(client_id: Optional[str] = None):
    """Get the cache directory for a client."""
    base_dir = os.path.join("cache", "file_cache")
    if client_id:
        return os.path.join(base_dir, client_id.lower())
    return base_dir

def _create_cache_key(client_id: str, prompt: str, model: str, **kwargs) -> str:
    """Create a cache key from the query parameters."""
    # Create a dictionary of all parameters that affect the result
    key_dict = {
        "client_id": client_id,
        "prompt": prompt,
        "model": model
    }
    
    # Add any other parameters that affect the result
    for k, v in kwargs.items():
        if k in ["include_charts", "execute_query", "limit_rows"]:
            key_dict[k] = v
    
    # Create a stable string representation and hash it
    key_str = json.dumps(key_dict, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    return f"query:{key_hash}"

def cache_llm_query(ttl: int = 3600):
    """
    Decorator to cache LLM query results in a file.
    
    Args:
        ttl: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request = None
            for arg in args:
                if hasattr(arg, 'client_id') and hasattr(arg, 'prompt'):
                    request = arg
                    break
            
            if not request and 'request' in kwargs and hasattr(kwargs['request'], 'client_id'):
                request = kwargs['request']
            
            if not request:
                # If no request object found, just call the original function
                return await func(*args, **kwargs)
            
            # Get cache key parameters
            client_id = getattr(request, 'client_id', 'default')
            prompt = getattr(request, 'prompt', '')
            model = getattr(request, 'model', 'default')
            include_charts = getattr(request, 'include_charts', False)
            execute_query = getattr(request, 'execute_query', True)
            limit_rows = getattr(request, 'limit_rows', 100)
            
            if not prompt:
                # If no prompt, just call the original function
                return await func(*args, **kwargs)
            
            # Create cache key and path
            cache_key = _create_cache_key(
                client_id=client_id, 
                prompt=prompt, 
                model=model,
                include_charts=include_charts,
                execute_query=execute_query,
                limit_rows=limit_rows
            )
            cache_dir = _get_cache_dir(client_id)
            cache_file = os.path.join(cache_dir, f"{cache_key}.pickle")
            lock_file = f"{cache_file}.lock"
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            try:
                # Try to get from cache
                # -----------------------------------------------------------------
                # Before even attempting to read the pickle, compare its mtime with
                # the feedback file mtime for this client. If feedback is newer,
                # we delete the pickle so the rest of the logic goes down the
                # normal cache-miss path. This is cheap and avoids having to load
                # the pickle into memory first.
                # -----------------------------------------------------------------
                fb_newer = False
                try:
                    from services.feedback_service import get_feedback_service
                    fb_file = get_feedback_service()._file_for_client(client_id)
                    if fb_file.exists() and os.path.exists(cache_file):
                        if fb_file.stat().st_mtime > os.path.getmtime(cache_file):
                            logger.info("Pre-read check: feedback newer than cache pickle → deleting %s", cache_file)
                            os.remove(cache_file)
                            fb_newer = True
                except Exception:
                    pass

                if os.path.exists(cache_file):
                    # Use file lock to prevent concurrent access
                    with FileLock(lock_file):
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                            cache_timestamp = cache_data.get('timestamp', 0)
                            stale_due_feedback = False

                            # If we now have feedback recorded AFTER this cache entry was saved,
                            # the result is stale.  In that case we delete the cache file and
                            # treat as a cache-miss so the response is regenerated once with
                            # the new user hint / corrected SQL, after which it will be cached
                            # again.  This gives us the flow: miss → regenerate with feedback →
                            # future hits until the next feedback submission.
                            try:
                                from services.feedback_service import get_feedback_service
                                f_service = get_feedback_service()
                                fb_file = f_service._file_for_client(client_id)
                                if fb_file.exists():
                                    feedback_ts = fb_file.stat().st_mtime
                                    if feedback_ts > cache_timestamp:
                                        logger.info("File-cache stale (client '%s') due to feedback at %s > cache at %s. Deleting …", client_id, feedback_ts, cache_timestamp)
                                        os.remove(cache_file)
                                        stale_due_feedback = True
                            except FileNotFoundError:
                                stale_due_feedback = True
                            except Exception as _e:
                                # Any error in feedback check – ignore and proceed
                                pass

                            # If cache is not stale due to feedback and within TTL, serve it
                            if (not stale_due_feedback) and (time.time() - cache_timestamp < ttl):
                                logger.info(f"Cache hit for prompt: '{prompt[:30]}...'")
                                _cache_stats["hits"] += 1
                                
                                # Add cache metadata to response
                                result = cache_data['result']
                                if isinstance(result, dict):
                                    result.setdefault("_cache_metadata", {})
                                    result["_cache_metadata"].update({
                                        "cache_hit": True,
                                        "cached_at": datetime.datetime.fromtimestamp(cache_data['timestamp']).isoformat(),
                                        "ttl": ttl,
                                        "expires_at": datetime.datetime.fromtimestamp(cache_data['timestamp'] + ttl).isoformat()
                                    })
                                
                                return result

                
                # Cache miss or expired
                _cache_stats["misses"] += 1
                logger.info(f"Cache miss for prompt: '{prompt[:30]}...'")
                
                # Call the original function
                start_time = time.time()
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Store in cache
                if result:
                    # Use file lock to prevent concurrent access
                    with FileLock(lock_file):
                        cache_data = {
                            'result': result,
                            'timestamp': time.time()
                        }
                        with open(cache_file, 'wb') as f:
                            pickle.dump(cache_data, f)
                    
                    # Add cache metadata to response
                    if isinstance(result, dict):
                        result.setdefault("_cache_metadata", {})
                        result["_cache_metadata"].update({
                            "cache_hit": False,
                            "cached_at": datetime.datetime.now().isoformat(),
                            "execution_time_ms": round(elapsed * 1000, 2),
                            "ttl": ttl,
                            "expires_at": datetime.datetime.fromtimestamp(time.time() + ttl).isoformat()
                        })
                
                return result
            except Exception as e:
                # Log error but continue with original function call
                _cache_stats["errors"] += 1
                logger.error(f"Error using cache: {e}")
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def clear_cache(client_id: Optional[str] = None):
    """
    Clear the cache for a client or all clients.
    
    Args:
        client_id: Optional client ID
    """
    if client_id:
        cache_dir = _get_cache_dir(client_id)
        if os.path.exists(cache_dir):
            for file in Path(cache_dir).glob("*.pickle"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting cache file {file}: {e}")
        logger.info(f"Cleared cache for client {client_id}")
    else:
        # Clear all client caches
        base_dir = _get_cache_dir()
        if os.path.exists(base_dir):
            for client_dir in Path(base_dir).iterdir():
                if client_dir.is_dir():
                    for file in client_dir.glob("*.pickle"):
                        try:
                            file.unlink()
                        except Exception as e:
                            logger.error(f"Error deleting cache file {file}: {e}")
        logger.info("Cleared cache for all clients")
    
    # Reset stats
    reset_cache_stats()
