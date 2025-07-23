"""
Cache Scheduler Module

This module provides scheduling capabilities for cache maintenance tasks
such as cleanup, eviction, and monitoring.
"""

import logging
import threading
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple, Awaitable
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import os

from services.cache_service import CacheService
from monitoring.cache_monitor import cache_monitor

# Setup logging
logger = logging.getLogger(__name__)

class CacheScheduler:
    """
    Class for scheduling cache maintenance tasks.
    This class manages periodic tasks related to cache maintenance.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one cache scheduler instance."""
        if cls._instance is None:
            cls._instance = super(CacheScheduler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the cache scheduler."""
        # Only initialize once (singleton)
        if self._initialized:
            return
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        self.running = False
        self.jobs = {}
        self.cache_service = CacheService()
        
        # Configure from environment
        self.cleanup_interval = int(os.environ.get("CACHE_CLEANUP_INTERVAL_HOURS", 6))
        self.stats_interval = int(os.environ.get("CACHE_STATS_LOG_INTERVAL_MINUTES", 60))
        self.background_cleanup = os.environ.get("BACKGROUND_CLEANUP_ENABLED", "true").lower() in ["true", "1", "yes"]
        
        # Initialize scheduled tasks
        if self.background_cleanup:
            self._initialize_tasks()
        
        self._initialized = True
        logger.info("Cache scheduler initialized")
    
    def _initialize_tasks(self):
        """Initialize scheduled tasks."""
        # Add cleanup task
        self.add_task(
            "cache_cleanup",
            self._cleanup_task,
            hours=self.cleanup_interval
        )
        
        # Add stats logging task
        self.add_task(
            "cache_stats",
            self._stats_task,
            minutes=self.stats_interval
        )
        
        # Add index rebuild task (weekly)
        self.add_task(
            "rebuild_file_cache_index",
            self._rebuild_index_task,
            day_of_week="sun",
            hour=2  # 2 AM on Sundays
        )
    
    def add_task(
        self,
        name: str,
        func: Callable,
        seconds: Optional[int] = None,
        minutes: Optional[int] = None,
        hours: Optional[int] = None,
        day_of_week: Optional[str] = None,
        hour: Optional[int] = None
    ) -> str:
        """Add a task to the scheduler.
        
        Args:
            name: Task name
            func: Task function
            seconds: Run interval in seconds
            minutes: Run interval in minutes
            hours: Run interval in hours
            day_of_week: Day of week for cron schedule
            hour: Hour for cron schedule
            
        Returns:
            Task ID
        """
        if name in self.jobs:
            logger.warning(f"Task {name} already exists, replacing")
            self.scheduler.remove_job(self.jobs[name])
        
        if day_of_week is not None:
            # Use cron trigger for weekly tasks
            trigger = CronTrigger(
                day_of_week=day_of_week,
                hour=hour or 0
            )
        else:
            # Use interval trigger for regular tasks
            trigger = IntervalTrigger(
                seconds=seconds or 0,
                minutes=minutes or 0,
                hours=hours or 0
            )
        
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=name
        )
        
        self.jobs[name] = job.id
        logger.info(f"Added task {name} to scheduler")
        
        return job.id
    
    def remove_task(self, name: str) -> bool:
        """Remove a task from the scheduler.
        
        Args:
            name: Task name
            
        Returns:
            Whether the task was removed
        """
        if name in self.jobs:
            self.scheduler.remove_job(self.jobs[name])
            del self.jobs[name]
            logger.info(f"Removed task {name} from scheduler")
            return True
        
        return False
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.scheduler.start()
        self.running = True
        logger.info("Cache scheduler started")
    
    def shutdown(self):
        """Shutdown the scheduler."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        self.scheduler.shutdown()
        self.running = False
        logger.info("Cache scheduler shutdown")
    
    def _cleanup_task(self):
        """Task to clean up expired cache entries."""
        try:
            logger.info("Running cache cleanup task")
            results = self.cache_service.cleanup_caches()
            logger.info(f"Cache cleanup completed: {results}")
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")
    
    def _stats_task(self):
        """Task to log cache statistics."""
        try:
            logger.info("Collecting cache statistics")
            stats = self.cache_service.get_stats()
            
            # Update cache monitor with sizes
            cache_monitor.update_cache_sizes(
                memory_size=stats.get("cache_layers", {}).get("memory", {}).get("size", 0),
                file_size=stats.get("cache_layers", {}).get("file", {}).get("size", 0),
                redis_size=stats.get("cache_layers", {}).get("redis", {}).get("size", 0)
            )
            
            # Log important statistics
            hit_rate = stats.get("service_stats", {}).get("cache_hit_rate", 0)
            token_savings = stats.get("service_stats", {}).get("tokens_saved", 0)
            
            logger.info(f"Cache hit rate: {hit_rate:.2f}, Tokens saved: {token_savings}")
        except Exception as e:
            logger.error(f"Error in cache stats task: {e}")
    
    def _rebuild_index_task(self):
        """Task to rebuild file cache index."""
        try:
            logger.info("Rebuilding file cache index")
            cache_manager = self.cache_service.get_cache_manager()
            
            if hasattr(cache_manager, "file_cache"):
                cache_manager.file_cache.rebuild_index()
                logger.info("File cache index rebuilt successfully")
            else:
                logger.warning("File cache not available, skipping index rebuild")
        except Exception as e:
            logger.error(f"Error in rebuild index task: {e}")

# Create singleton instance for import
cache_scheduler = CacheScheduler()
