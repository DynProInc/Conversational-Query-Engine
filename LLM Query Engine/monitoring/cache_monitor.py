"""
Cache Monitoring Module

This module provides monitoring and analytics for the multi-level caching system.
It tracks cache performance metrics, generates reports, and integrates with 
the monitoring system.
"""

import logging
import time
import json
import os
import threading
import datetime
from typing import Dict, List, Any, Optional, Tuple
import csv
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class CacheMetric:
    """Base class for a cache metric"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.timestamp = time.time()

class CounterMetric(CacheMetric):
    """A metric that counts occurrences"""
    
    def __init__(self, name: str, description: str, initial_value: int = 0):
        super().__init__(name, description)
        self.value = initial_value
    
    def increment(self, amount: int = 1):
        """Increment the counter by the specified amount"""
        self.value += amount
        self.timestamp = time.time()
    
    def reset(self):
        """Reset the counter to zero"""
        self.value = 0
        self.timestamp = time.time()

class GaugeMetric(CacheMetric):
    """A metric that represents a current value"""
    
    def __init__(self, name: str, description: str, initial_value: float = 0.0):
        super().__init__(name, description)
        self.value = initial_value
    
    def set(self, value: float):
        """Set the gauge to the specified value"""
        self.value = value
        self.timestamp = time.time()

class HistogramMetric(CacheMetric):
    """A metric that tracks the distribution of values"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.values = []
        self.count = 0
        self.sum = 0.0
        self._min = None
        self._max = None
    
    def observe(self, value: float):
        """Record a new observation"""
        self.values.append(value)
        self.count += 1
        self.sum += value
        self.timestamp = time.time()
        
        if self._min is None or value < self._min:
            self._min = value
        
        if self._max is None or value > self._max:
            self._max = value
    
    def reset(self):
        """Reset the histogram"""
        self.values = []
        self.count = 0
        self.sum = 0.0
        self._min = None
        self._max = None
        self.timestamp = time.time()
    
    @property
    def min(self) -> Optional[float]:
        """Get the minimum observed value"""
        return self._min
    
    @property
    def max(self) -> Optional[float]:
        """Get the maximum observed value"""
        return self._max
    
    @property
    def avg(self) -> Optional[float]:
        """Get the average observed value"""
        if self.count == 0:
            return None
        return self.sum / self.count
    
    @property
    def percentiles(self) -> Dict[str, float]:
        """Get the percentiles of the observed values"""
        if not self.values:
            return {}
        
        sorted_values = sorted(self.values)
        result = {}
        
        for p in [50, 90, 95, 99]:
            idx = int(len(sorted_values) * p / 100)
            result[f"p{p}"] = sorted_values[idx]
        
        return result

class CacheMonitor:
    """
    Class for monitoring cache performance metrics.
    This class collects and reports cache metrics for analysis and optimization.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one cache monitor instance."""
        if cls._instance is None:
            cls._instance = super(CacheMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        stats_dir: str = "stats/cache",
        log_interval: int = 3600,  # 1 hour
        prometheus_enabled: bool = False
    ):
        """Initialize the cache monitor.
        
        Args:
            stats_dir: Directory to store stats files
            log_interval: Interval in seconds to log metrics
            prometheus_enabled: Whether to expose metrics to Prometheus
        """
        # Only initialize once (singleton)
        if self._initialized:
            return
        
        self.stats_dir = stats_dir
        self.log_interval = log_interval
        self.prometheus_enabled = prometheus_enabled
        self.metrics = {}
        
        # Create metrics
        self._create_metrics()
        
        # Create stats directory
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Start logging thread
        self.should_run = True
        self.logging_thread = threading.Thread(
            target=self._periodic_logging,
            daemon=True
        )
        self.logging_thread.start()
        
        self._initialized = True
        logger.info("Cache monitor initialized")
    
    def _create_metrics(self):
        """Create cache metrics."""
        # Basic cache metrics
        self.metrics["hits"] = CounterMetric("cache_hits", "Number of cache hits")
        self.metrics["misses"] = CounterMetric("cache_misses", "Number of cache misses")
        self.metrics["hit_rate"] = GaugeMetric("cache_hit_rate", "Cache hit rate")
        
        # Query cache metrics
        self.metrics["query_hits"] = CounterMetric("query_cache_hits", "Number of query cache hits")
        self.metrics["semantic_hits"] = CounterMetric("semantic_cache_hits", "Number of semantic cache hits")
        self.metrics["token_savings"] = CounterMetric("token_savings", "Number of tokens saved by caching")
        
        # Size metrics
        self.metrics["memory_size"] = GaugeMetric("memory_cache_size", "Memory cache size in bytes")
        self.metrics["file_size"] = GaugeMetric("file_cache_size", "File cache size in bytes")
        self.metrics["redis_size"] = GaugeMetric("redis_cache_size", "Redis cache size in bytes")
        
        # Operation timing
        self.metrics["get_time"] = HistogramMetric("cache_get_time", "Time to get from cache in milliseconds")
        self.metrics["set_time"] = HistogramMetric("cache_set_time", "Time to set in cache in milliseconds")
        
        # Error metrics
        self.metrics["errors"] = CounterMetric("cache_errors", "Number of cache errors")
    
    def record_hit(self, cache_type: str = "generic"):
        """Record a cache hit.
        
        Args:
            cache_type: Type of cache hit (generic, query, semantic, etc.)
        """
        self.metrics["hits"].increment()
        
        if cache_type == "query":
            self.metrics["query_hits"].increment()
        elif cache_type == "semantic":
            self.metrics["semantic_hits"].increment()
        
        # Update hit rate
        total = self.metrics["hits"].value + self.metrics["misses"].value
        if total > 0:
            hit_rate = self.metrics["hits"].value / total
            self.metrics["hit_rate"].set(hit_rate)
    
    def record_miss(self):
        """Record a cache miss."""
        self.metrics["misses"].increment()
        
        # Update hit rate
        total = self.metrics["hits"].value + self.metrics["misses"].value
        if total > 0:
            hit_rate = self.metrics["hits"].value / total
            self.metrics["hit_rate"].set(hit_rate)
    
    def record_token_savings(self, tokens: int):
        """Record tokens saved by the cache.
        
        Args:
            tokens: Number of tokens saved
        """
        self.metrics["token_savings"].increment(tokens)
    
    def record_operation_time(self, operation: str, time_ms: float):
        """Record time taken for a cache operation.
        
        Args:
            operation: Operation type (get, set)
            time_ms: Time taken in milliseconds
        """
        if operation == "get" and "get_time" in self.metrics:
            self.metrics["get_time"].observe(time_ms)
        elif operation == "set" and "set_time" in self.metrics:
            self.metrics["set_time"].observe(time_ms)
    
    def record_error(self):
        """Record a cache error."""
        self.metrics["errors"].increment()
    
    def update_cache_sizes(
        self,
        memory_size: Optional[int] = None,
        file_size: Optional[int] = None,
        redis_size: Optional[int] = None
    ):
        """Update cache size metrics.
        
        Args:
            memory_size: Memory cache size in bytes
            file_size: File cache size in bytes
            redis_size: Redis cache size in bytes
        """
        if memory_size is not None:
            self.metrics["memory_size"].set(memory_size)
        
        if file_size is not None:
            self.metrics["file_size"].set(file_size)
        
        if redis_size is not None:
            self.metrics["redis_size"].set(redis_size)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics.
        
        Returns:
            Dictionary of metrics
        """
        result = {
            "timestamp": time.time(),
            "metrics": {}
        }
        
        for name, metric in self.metrics.items():
            if isinstance(metric, CounterMetric) or isinstance(metric, GaugeMetric):
                result["metrics"][name] = {
                    "value": metric.value,
                    "timestamp": metric.timestamp
                }
            elif isinstance(metric, HistogramMetric):
                result["metrics"][name] = {
                    "count": metric.count,
                    "sum": metric.sum,
                    "min": metric.min,
                    "max": metric.max,
                    "avg": metric.avg,
                    "percentiles": metric.percentiles,
                    "timestamp": metric.timestamp
                }
        
        return result
    
    def reset_metrics(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()
            elif hasattr(metric, "set"):
                metric.set(0)
    
    def _periodic_logging(self):
        """Periodically log metrics to file."""
        while self.should_run:
            time.sleep(self.log_interval)
            self._log_metrics_to_file()
    
    def _log_metrics_to_file(self):
        """Log metrics to a file."""
        try:
            # Get current metrics
            metrics = self.get_metrics()
            
            # Create timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Write to JSON file
            filename = f"{self.stats_dir}/cache_metrics_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Update CSV file for time series
            csv_file = f"{self.stats_dir}/cache_metrics_timeseries.csv"
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                
                # Write header if file doesn't exist
                if not file_exists:
                    header = ["timestamp", "hits", "misses", "hit_rate", 
                              "query_hits", "semantic_hits", "token_savings",
                              "memory_size", "file_size", "redis_size", "errors"]
                    writer.writerow(header)
                
                # Write data
                row = [
                    metrics["timestamp"],
                    metrics["metrics"]["hits"]["value"],
                    metrics["metrics"]["misses"]["value"],
                    metrics["metrics"]["hit_rate"]["value"],
                    metrics["metrics"]["query_hits"]["value"],
                    metrics["metrics"]["semantic_hits"]["value"],
                    metrics["metrics"]["token_savings"]["value"],
                    metrics["metrics"]["memory_size"]["value"],
                    metrics["metrics"]["file_size"]["value"],
                    metrics["metrics"]["redis_size"]["value"],
                    metrics["metrics"]["errors"]["value"]
                ]
                writer.writerow(row)
            
            logger.info(f"Logged cache metrics to {filename} and updated time series")
        except Exception as e:
            logger.error(f"Error logging cache metrics: {e}")
    
    def shutdown(self):
        """Shutdown the cache monitor."""
        self.should_run = False
        if hasattr(self, "logging_thread") and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=2.0)
        
        # Final log before shutdown
        try:
            self._log_metrics_to_file()
        except Exception as e:
            logger.error(f"Error during final metrics logging: {e}")

# Create singleton instance for import
cache_monitor = CacheMonitor()
