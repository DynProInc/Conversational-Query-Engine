"""
Main FastAPI application for the Conversational Query Engine.
This module initializes the FastAPI application and includes all API routes.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import API routes
from api.cache_routes import router as cache_router
from api.rag_routes import router as rag_router
# Import other routers as needed

# Import services
from services.cache_service import CacheService
from services.rag_service import RAGService
from services.cache_scheduler import cache_scheduler
from monitoring.cache_monitor import cache_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Conversational Query Engine",
    description="A FastAPI-based API for conversational query processing with RAG capabilities",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include API routers
app.include_router(cache_router)
app.include_router(rag_router)
# Include other routers here

# Initialize services at startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    try:
        logger.info("Initializing services...")
        
        # Initialize cache service
        cache_service = CacheService()
        logger.info("Cache service initialized")
        
        # Initialize RAG service
        rag_service = RAGService(cache_service=cache_service)
        logger.info("RAG service initialized")
        
        # Initialize cache monitor
        # cache_monitor is already initialized as a singleton
        logger.info("Cache monitor initialized")
        
        # Start cache scheduler
        if os.environ.get("BACKGROUND_CLEANUP_ENABLED", "true").lower() in ["true", "1", "yes"]:
            cache_scheduler.start()
            logger.info("Cache scheduler started")
        else:
            logger.info("Background cleanup disabled, cache scheduler not started")
        
        # Perform initial cache cleanup
        await cache_service.cleanup_caches()
        logger.info("Initial cache cleanup completed")
        
        # Log initialization complete
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        # Allow startup to continue even if services fail to initialize
        # Services will be re-initialized when accessed

# Shutdown services on application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on application shutdown."""
    try:
        logger.info("Shutting down services...")
        
        # Shutdown cache scheduler first
        if cache_scheduler.running:
            cache_scheduler.shutdown()
            logger.info("Cache scheduler shut down")
        
        # Shutdown cache monitor
        cache_monitor.shutdown()
        logger.info("Cache monitor shut down")
        
        # Clean up cache service
        cache_service = CacheService()
        await cache_service.shutdown()
        logger.info("Cache service shut down")
        
        # Perform final logging
        logger.info("Final cache statistics: ")
        stats = cache_service.get_stats()
        logger.info(f"Cache hit rate: {stats.get('service_stats', {}).get('cache_hit_rate', 0):.2f}")
        logger.info(f"Tokens saved: {stats.get('service_stats', {}).get('tokens_saved', 0)}")
        
        logger.info("All services shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down services: {e}")

# Root endpoint
@app.get("/", tags=["status"])
async def root():
    """Root endpoint."""
    return {
        "message": "Conversational Query Engine API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health", tags=["status"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check cache service health
        cache_service = CacheService()
        cache_health = await cache_service.get_health_status()
        
        # Check RAG service health
        rag_service = RAGService()
        rag_health = {
            "status": "healthy",
            "service_initialized": hasattr(rag_service, "_initialized") and rag_service._initialized
        }
        
        # Check scheduler status
        scheduler_health = {
            "status": "healthy" if cache_scheduler.running else "stopped",
            "tasks_count": len(cache_scheduler.jobs)
        }
        
        return {
            "status": "healthy",
            "cache_service": cache_health,
            "rag_service": rag_health,
            "scheduler": scheduler_health
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Add error handler for application-wide exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred", "detail": str(exc)}
    )

# Run the app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    # Configure uvicorn logging
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    uvicorn_log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=True,  # Enable auto-reload for development
        log_config=uvicorn_log_config,
    )
