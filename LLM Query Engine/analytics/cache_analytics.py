"""
Cache Analytics Module

This module provides analytics functionality for the caching system.
It analyzes cache metrics and generates insights and visualizations.
"""

import logging
import time
import json
import os
import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from services.cache_service import CacheService
from monitoring.cache_monitor import cache_monitor

# Setup logging
logger = logging.getLogger(__name__)

class CacheAnalytics:
    """
    Class for analyzing cache performance and generating insights.
    """
    
    def __init__(
        self,
        stats_dir: str = "stats/cache",
        report_dir: str = "reports/cache",
        cache_service: Optional[CacheService] = None
    ):
        """Initialize the cache analytics.
        
        Args:
            stats_dir: Directory with cached statistics
            report_dir: Directory to store reports
            cache_service: Cache service instance
        """
        self.stats_dir = stats_dir
        self.report_dir = report_dir
        self.cache_service = cache_service or CacheService()
        
        # Create directories
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize metrics data
        self.metrics_data = None
    
    def load_metrics(self, days: int = 7) -> pd.DataFrame:
        """Load cache metrics from files.
        
        Args:
            days: Number of days to load
            
        Returns:
            DataFrame with metrics
        """
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Find metric files
        csv_file = os.path.join(self.stats_dir, "cache_metrics_timeseries.csv")
        
        if not os.path.exists(csv_file):
            logger.warning(f"Metrics file {csv_file} not found")
            return pd.DataFrame()
        
        # Load metrics from CSV
        try:
            df = pd.read_csv(csv_file)
            
            # Convert timestamp to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Filter by cutoff date
            df = df[df["datetime"] >= cutoff_date]
            
            self.metrics_data = df
            return df
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return pd.DataFrame()
    
    def generate_hit_rate_chart(
        self,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """Generate hit rate chart.
        
        Args:
            output_path: Path to save the chart
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved chart
        """
        if self.metrics_data is None:
            self.load_metrics()
        
        if self.metrics_data.empty:
            logger.warning("No metrics data available")
            return ""
        
        try:
            # Create figure and axis
            plt.figure(figsize=(12, 6))
            
            # Plot hit rate
            plt.plot(
                self.metrics_data["datetime"],
                self.metrics_data["hit_rate"],
                label="Hit Rate",
                linewidth=2
            )
            
            # Add title and labels
            plt.title("Cache Hit Rate Over Time", fontsize=16)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Hit Rate", fontsize=12)
            
            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save or show
            if output_path:
                save_path = output_path
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.report_dir, f"hit_rate_{timestamp}.png")
            
            plt.savefig(save_path)
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"Error generating hit rate chart: {e}")
            return ""
    
    def generate_token_savings_chart(
        self,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """Generate token savings chart.
        
        Args:
            output_path: Path to save the chart
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved chart
        """
        if self.metrics_data is None:
            self.load_metrics()
        
        if self.metrics_data.empty:
            logger.warning("No metrics data available")
            return ""
        
        try:
            # Create figure and axis
            plt.figure(figsize=(12, 6))
            
            # Plot token savings
            plt.plot(
                self.metrics_data["datetime"],
                self.metrics_data["token_savings"],
                label="Token Savings",
                linewidth=2,
                color="green"
            )
            
            # Add title and labels
            plt.title("Cumulative Token Savings Over Time", fontsize=16)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Tokens Saved", fontsize=12)
            
            # Add grid
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save or show
            if output_path:
                save_path = output_path
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.report_dir, f"token_savings_{timestamp}.png")
            
            plt.savefig(save_path)
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"Error generating token savings chart: {e}")
            return ""
    
    def generate_cache_size_chart(
        self,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """Generate cache size chart.
        
        Args:
            output_path: Path to save the chart
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved chart
        """
        if self.metrics_data is None:
            self.load_metrics()
        
        if self.metrics_data.empty:
            logger.warning("No metrics data available")
            return ""
        
        try:
            # Create figure and axis
            plt.figure(figsize=(12, 6))
            
            # Plot cache sizes
            plt.plot(
                self.metrics_data["datetime"],
                self.metrics_data["memory_size"] / (1024 * 1024),  # Convert to MB
                label="Memory Cache",
                linewidth=2
            )
            
            plt.plot(
                self.metrics_data["datetime"],
                self.metrics_data["file_size"] / (1024 * 1024),  # Convert to MB
                label="File Cache",
                linewidth=2
            )
            
            plt.plot(
                self.metrics_data["datetime"],
                self.metrics_data["redis_size"] / (1024 * 1024),  # Convert to MB
                label="Redis Cache",
                linewidth=2
            )
            
            # Add title and labels
            plt.title("Cache Size Over Time", fontsize=16)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Size (MB)", fontsize=12)
            
            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save or show
            if output_path:
                save_path = output_path
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.report_dir, f"cache_size_{timestamp}.png")
            
            plt.savefig(save_path)
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"Error generating cache size chart: {e}")
            return ""
    
    def generate_hit_types_chart(
        self,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """Generate chart of different hit types.
        
        Args:
            output_path: Path to save the chart
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved chart
        """
        if self.metrics_data is None:
            self.load_metrics()
        
        if self.metrics_data.empty:
            logger.warning("No metrics data available")
            return ""
        
        try:
            # Create figure and axis
            plt.figure(figsize=(12, 6))
            
            # Get the latest data point for the pie chart
            latest = self.metrics_data.iloc[-1]
            
            # Calculate hit types
            regular_hits = latest["query_hits"] - latest["semantic_hits"]
            semantic_hits = latest["semantic_hits"]
            misses = latest["misses"]
            
            # Create pie chart data
            labels = ["Exact Hits", "Semantic Hits", "Misses"]
            sizes = [regular_hits, semantic_hits, misses]
            colors = ["#4CAF50", "#2196F3", "#F44336"]
            
            # Plot pie chart
            plt.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                shadow=True,
                explode=(0, 0.1, 0)  # Explode semantic hits slice
            )
            
            # Add title
            plt.title("Cache Hit Types Distribution", fontsize=16)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            plt.axis("equal")
            plt.tight_layout()
            
            # Save or show
            if output_path:
                save_path = output_path
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.report_dir, f"hit_types_{timestamp}.png")
            
            plt.savefig(save_path)
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"Error generating hit types chart: {e}")
            return ""
    
    def generate_cache_report(self) -> Dict[str, Any]:
        """Generate a comprehensive cache performance report.
        
        Returns:
            Dictionary with report data
        """
        # Get current cache stats
        cache_stats = self.cache_service.get_stats()
        
        # Load historical metrics
        if self.metrics_data is None:
            self.load_metrics()
        
        # Initialize report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "current_stats": cache_stats,
            "charts": {},
            "insights": [],
            "recommendations": []
        }
        
        # Generate charts
        hit_rate_chart = self.generate_hit_rate_chart()
        if hit_rate_chart:
            report["charts"]["hit_rate"] = hit_rate_chart
        
        token_savings_chart = self.generate_token_savings_chart()
        if token_savings_chart:
            report["charts"]["token_savings"] = token_savings_chart
        
        cache_size_chart = self.generate_cache_size_chart()
        if cache_size_chart:
            report["charts"]["cache_size"] = cache_size_chart
        
        hit_types_chart = self.generate_hit_types_chart()
        if hit_types_chart:
            report["charts"]["hit_types"] = hit_types_chart
        
        # Generate insights
        if not self.metrics_data.empty:
            # Hit rate trend
            latest_hit_rate = cache_stats.get("service_stats", {}).get("cache_hit_rate", 0)
            report["insights"].append({
                "title": "Hit Rate",
                "value": f"{latest_hit_rate:.2f}",
                "trend": self._calculate_trend(self.metrics_data["hit_rate"])
            })
            
            # Token savings
            latest_token_savings = cache_stats.get("service_stats", {}).get("tokens_saved", 0)
            report["insights"].append({
                "title": "Total Token Savings",
                "value": str(latest_token_savings),
                "impact": self._calculate_cost_savings(latest_token_savings)
            })
            
            # Semantic cache effectiveness
            semantic_hits = cache_stats.get("service_stats", {}).get("semantic_hits", 0)
            total_hits = cache_stats.get("service_stats", {}).get("hits", 0)
            if total_hits > 0:
                semantic_percentage = (semantic_hits / total_hits) * 100
                report["insights"].append({
                    "title": "Semantic Cache Effectiveness",
                    "value": f"{semantic_percentage:.1f}%",
                    "description": "Percentage of cache hits from semantic matching"
                })
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(cache_stats)
        
        # Save report to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.report_dir, f"cache_report_{timestamp}.json")
        
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache report: {e}")
        
        return report
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend from a time series.
        
        Args:
            series: Time series data
            
        Returns:
            Trend description
        """
        if len(series) < 2:
            return "stable"
        
        first = series.iloc[0]
        last = series.iloc[-1]
        
        if first == 0:
            return "improving" if last > 0 else "stable"
        
        percentage_change = ((last - first) / first) * 100
        
        if percentage_change > 10:
            return "strongly improving"
        elif percentage_change > 2:
            return "improving"
        elif percentage_change < -10:
            return "strongly declining"
        elif percentage_change < -2:
            return "declining"
        else:
            return "stable"
    
    def _calculate_cost_savings(self, tokens_saved: int) -> str:
        """Calculate cost savings from tokens saved.
        
        Args:
            tokens_saved: Number of tokens saved
            
        Returns:
            Cost savings description
        """
        # Approximate cost per token (e.g., for GPT-4)
        cost_per_1k_tokens = 0.06  # $0.06 per 1K tokens
        
        # Calculate cost savings
        cost_savings = (tokens_saved / 1000) * cost_per_1k_tokens
        
        return f"${cost_savings:.2f} (approx.)"
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on cache statistics.
        
        Args:
            stats: Cache statistics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check hit rate
        hit_rate = stats.get("service_stats", {}).get("cache_hit_rate", 0)
        if hit_rate < 0.3:
            recommendations.append({
                "title": "Low Cache Hit Rate",
                "description": "The hit rate is below 30%. Consider adjusting TTL values or enabling semantic caching to improve cache utilization."
            })
        
        # Check memory cache utilization
        memory_size = stats.get("cache_layers", {}).get("memory", {}).get("size", 0)
        memory_limit = stats.get("cache_layers", {}).get("memory", {}).get("max_size", float("inf"))
        
        if memory_size > 0.9 * memory_limit and memory_limit < float("inf"):
            recommendations.append({
                "title": "Memory Cache Near Capacity",
                "description": "The memory cache is over 90% full. Consider increasing the maximum size or reducing TTL to prevent evictions."
            })
        
        # Check Redis connection
        redis_info = stats.get("cache_layers", {}).get("redis", {}).get("redis_info", {})
        if redis_info and not redis_info.get("connected", True):
            recommendations.append({
                "title": "Redis Connection Issues",
                "description": "The Redis cache is not connected. Check your Redis configuration and ensure the server is running."
            })
        
        # Check semantic cache threshold
        semantic_hits = stats.get("service_stats", {}).get("semantic_hits", 0)
        total_hits = stats.get("service_stats", {}).get("hits", 0)
        
        if semantic_hits == 0 and total_hits > 100:
            recommendations.append({
                "title": "No Semantic Cache Hits",
                "description": "Consider lowering the semantic similarity threshold to improve semantic cache utilization."
            })
        
        return recommendations

# Create analytics instance
cache_analytics = CacheAnalytics()
