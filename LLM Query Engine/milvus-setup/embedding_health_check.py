#!/usr/bin/env python3
"""
Embedding Health Check Integration
=================================

Integrates embedding status checks with the client health check system.
Provides both standalone status checks and integration with the
existing health check API endpoints.

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import embedding generator for status checks - direct import from current directory
from generate_client_embeddings import ClientEmbeddingGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmbeddingHealth")

class EmbeddingHealthCheck:
    """Embedding health check system integration"""
    
    def __init__(self):
        """Initialize the health check"""
        self.generator = ClientEmbeddingGenerator()
    
    def get_embedding_health_status(self, client_id: str) -> Dict[str, Any]:
        """
        Get health status for a specific client's embeddings
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with health status information
        """
        try:
            # Get embedding status
            status = self.generator.get_client_embedding_status(client_id)
            
            # Format for health check integration
            health_status = {
                "component": "embeddings",
                "status": "healthy" if status["embeddings_exist"] and status["up_to_date"] else "degraded",
                "details": {
                    "exists": status["embeddings_exist"],
                    "up_to_date": status["up_to_date"],
                    "last_updated": status["last_updated"],
                    "record_count": status["record_count"],
                    "model": status["model_name"],
                    "dimensions": status["embedding_dimensions"]
                },
                "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # If embeddings don't exist or are outdated, provide instructions
            if not status["embeddings_exist"]:
                health_status["status"] = "unhealthy"
                health_status["details"]["message"] = f"Embeddings for client {client_id} do not exist"
                health_status["details"]["action"] = f"Run 'python milvus_setup/generate_client_embeddings.py --clients {client_id}'"
            elif not status["up_to_date"]:
                health_status["status"] = "degraded"
                health_status["details"]["message"] = f"Embeddings for client {client_id} are outdated"
                health_status["details"]["action"] = f"Run 'python milvus_setup/generate_client_embeddings.py --clients {client_id}'"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking embedding health for {client_id}: {str(e)}")
            return {
                "component": "embeddings",
                "status": "unhealthy",
                "details": {
                    "error": str(e)
                },
                "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def get_all_clients_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get embedding health status for all clients
        
        Returns:
            Dictionary mapping client_id to health status dictionaries
        """
        result = {}
        clients = self.generator.get_client_directories()
        
        for client_id in clients:
            result[client_id] = self.get_embedding_health_status(client_id)
        
        return result
    
    def get_system_embedding_health(self) -> Dict[str, Any]:
        """
        Get system-wide embedding health status
        
        Returns:
            Dictionary with system-wide health information
        """
        client_health = self.get_all_clients_health()
        
        # Count clients by status
        healthy_count = sum(1 for status in client_health.values() if status["status"] == "healthy")
        degraded_count = sum(1 for status in client_health.values() if status["status"] == "degraded")
        unhealthy_count = sum(1 for status in client_health.values() if status["status"] == "unhealthy")
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "component": "embeddings_system",
            "status": overall_status,
            "details": {
                "total_clients": len(client_health),
                "healthy_clients": healthy_count,
                "degraded_clients": degraded_count,
                "unhealthy_clients": unhealthy_count,
                "clients": client_health
            },
            "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save_status_report(self, output_file: str = "embedding_status.json") -> str:
        """
        Save a status report to a file
        
        Args:
            output_file: File to write the report to
            
        Returns:
            Path to the report file
        """
        status = self.get_system_embedding_health()
        
        with open(output_file, "w") as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Saved embedding status report to {output_file}")
        return output_file

# Integration functions for health check API

def get_client_embedding_health(client_id: str) -> Dict[str, Any]:
    """
    Get embedding health status for a specific client
    For integration with existing /health/client/{client_id} endpoint
    
    Args:
        client_id: Client identifier
        
    Returns:
        Health status dictionary
    """
    health_check = EmbeddingHealthCheck()
    return health_check.get_embedding_health_status(client_id)

def get_system_embedding_health() -> Dict[str, Any]:
    """
    Get system-wide embedding health status
    For integration with existing /health/client endpoint
    
    Returns:
        Health status dictionary
    """
    health_check = EmbeddingHealthCheck()
    return health_check.get_system_embedding_health()

# Standalone execution
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check embedding health status')
    parser.add_argument('--client', help='Specific client to check')
    parser.add_argument('--output', help='Output file to save report to')
    
    args = parser.parse_args()
    
    health_check = EmbeddingHealthCheck()
    
    if args.client:
        # Check specific client
        status = health_check.get_embedding_health_status(args.client)
        print(f"\nEMBEDDING HEALTH FOR CLIENT: {args.client}")
        print("=" * 40)
        print(f"Status: {status['status'].upper()}")
        print(f"Last checked: {status['last_checked']}")
        
        details = status['details']
        print("\nDetails:")
        for key, value in details.items():
            print(f"  {key}: {value}")
        
        # If there's an action to take, highlight it
        if "action" in details:
            print(f"\nRecommended action: {details['action']}")
    else:
        # Check all clients
        status = health_check.get_system_embedding_health()
        print("\nSYSTEM-WIDE EMBEDDING HEALTH")
        print("=" * 40)
        print(f"Overall status: {status['status'].upper()}")
        print(f"Last checked: {status['last_checked']}")
        
        details = status['details']
        print(f"\nTotal clients: {details['total_clients']}")
        print(f"Healthy clients: {details['healthy_clients']}")
        print(f"Degraded clients: {details['degraded_clients']}")
        print(f"Unhealthy clients: {details['unhealthy_clients']}")
        
        print("\nClient Status:")
        for client_id, client_status in details['clients'].items():
            print(f"  {client_id}: {client_status['status'].upper()}")
            if client_status['status'] != "healthy":
                if "message" in client_status['details']:
                    print(f"    - {client_status['details']['message']}")
                if "action" in client_status['details']:
                    print(f"    - Action: {client_status['details']['action']}")
    
    # Save report if requested
    if args.output:
        output_file = args.output
    else:
        output_file = "embedding_status.json"
    
    health_check.save_status_report(output_file)
    print(f"\nFull status report saved to {os.path.abspath(output_file)}")
