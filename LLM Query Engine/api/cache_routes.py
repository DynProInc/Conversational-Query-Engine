"""
API routes for cache management and monitoring.
This module provides FastAPI routes for cache administration and statistics.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import logging

from services.cache_service import CacheService

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/cache",
    tags=["cache"],
    responses={404: {"description": "Not found"}}
)

# Models for API
class ClearCacheRequest(BaseModel):
    """Request model for clearing cache."""
    namespace: Optional[str] = Field(default=None, description="Namespace of cache to clear. If omitted, clears all caches.")
    reason: Optional[str] = Field(default=None, description="Reason for cache clear (recorded in logs).")

class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    service_stats: Dict[str, Any]
    cache_manager: Dict[str, Any]
    cache_layers: Dict[str, Dict[str, Any]]
    query_cache: Optional[Dict[str, Any]] = None

# Get cache service instance
def get_cache_service():
    """Get cache service instance."""
    return CacheService()

@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Get cache statistics.
    
    Returns:
        Cache statistics
    """
    try:
        stats = cache_service.get_stats()
        return CacheStatsResponse(
            service_stats=stats.get("service_stats", {}),
            cache_manager=stats.get("cache_manager", {}),
            cache_layers=stats.get("cache_layers", {}),
            query_cache=stats.get("query_cache")
        )
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")

@router.post("/clear")
async def clear_cache(
    request: ClearCacheRequest,
    background_tasks: BackgroundTasks,
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Clear cache contents.
    
    Args:
        request: Clear cache request
        background_tasks: Background tasks
        cache_service: Cache service
        
    Returns:
        Clear cache result
    """
    try:
        # Run cache clearing in background
        def clear_cache_task():
            try:
                cache_service.clear_caches(namespace=request.namespace)
                logger.info(f"Cache cleared successfully. Namespace: {request.namespace}, Reason: {request.reason}")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
        
        background_tasks.add_task(clear_cache_task)
        
        return {"message": "Cache clearing scheduled", "namespace": request.namespace}
    except Exception as e:
        logger.error(f"Error scheduling cache clear: {e}")
        raise HTTPException(status_code=500, detail=f"Error scheduling cache clear: {str(e)}")

@router.post("/cleanup")
async def cleanup_cache(
    background_tasks: BackgroundTasks,
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Clean up expired cache entries.
    
    Args:
        background_tasks: Background tasks
        cache_service: Cache service
        
    Returns:
        Cleanup scheduled message
    """
    try:
        # Run cache cleanup in background
        def cleanup_cache_task():
            try:
                results = cache_service.cleanup_caches()
                logger.info(f"Cache cleanup completed: {results}")
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
        
        background_tasks.add_task(cleanup_cache_task)
        
        return {"message": "Cache cleanup scheduled"}
    except Exception as e:
        logger.error(f"Error scheduling cache cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Error scheduling cache cleanup: {str(e)}")

@router.get("/health")
async def cache_health(
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Get cache health status.
    
    Args:
        cache_service: Cache service
        
    Returns:
        Cache health status
    """
    try:
        # Get basic stats
        stats = cache_service.get_stats()
        
        # Extract status of each cache layer
        cache_status = {}
        for name, layer_stats in stats.get("cache_layers", {}).items():
            cache_status[name] = {
                "active": True,
                "size": layer_stats.get("size", 0)
            }
            
            # Check Redis connection if applicable
            if name == "redis":
                redis_info = layer_stats.get("redis_info", {})
                cache_status["redis"]["connected"] = redis_info.get("connected", False)
        
        return {
            "status": "healthy",
            "cache_layers": cache_status,
            "hit_rate": stats.get("service_stats", {}).get("cache_hit_rate", 0)
        }
    except Exception as e:
        logger.error(f"Error getting cache health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
