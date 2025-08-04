#!/usr/bin/env python3
"""
Client RAG Manager
=================

Manages client-specific RAG configurations and ensures strict client isolation.
Integrates with the existing client management system and health checks.

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import csv
import logging
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClientRAGManager")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import client management utilities
try:
    from client_manager import get_client_context, verify_client
    from token_logger import log_token_usage
except ImportError:
    logger.warning("Could not import client manager modules. Running in standalone mode.")
    
# Define client registry path
CLIENT_REGISTRY_PATH = os.path.join(parent_dir, "config", "clients", "client_registry.csv")

# Import RAG system
from multi_client_rag import MultiClientMilvusRAG

class ClientRAGManager:
    """
    Manages client-specific RAG configurations with strict isolation
    """
    
    def __init__(self):
        """Initialize the RAG manager"""
        self.rag_system = MultiClientMilvusRAG()
        self.initialized_clients = set()
        self.client_registry = self._load_client_registry()
        
    def _load_client_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load client information from registry CSV
        
        Returns:
            Dictionary mapping client_id to client information
        """
        clients = {}
        
        if not os.path.exists(CLIENT_REGISTRY_PATH):
            logger.error(f"Client registry not found at {CLIENT_REGISTRY_PATH}")
            return clients
            
        try:
            with open(CLIENT_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    client_id = row['client_id']
                    clients[client_id] = {
                        'name': row['client_name'],
                        'description': row.get('description', ''),
                        'active': row.get('active', 'true').lower() == 'true',
                        'data_dictionary_path': row['data_dictionary_path']
                    }
            
            logger.info(f"Loaded {len(clients)} clients from registry")
                
        except Exception as e:
            logger.error(f"Error loading client registry: {str(e)}")
            
        return clients
        
    def initialize_client(self, client_id: str) -> bool:
        """
        Initialize RAG for a specific client
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if successful, False otherwise
        """
        # Skip if already initialized
        if client_id in self.initialized_clients:
            return True
        
        # Verify client exists in the system
        try:
            if 'verify_client' in globals():
                if not verify_client(client_id):
                    logger.error(f"Client '{client_id}' not found in client registry")
                    return False
        except Exception as e:
            logger.error(f"Error verifying client '{client_id}': {e}")
            return False
        
        # Get client data dictionary path
        data_dict_path = self._get_client_data_dictionary_path(client_id)
        if not data_dict_path:
            logger.error(f"No data dictionary found for client '{client_id}'")
            return False
        
        # Initialize client in RAG system
        try:
            success = self.rag_system.setup_client(client_id, data_dict_path)
            if success:
                self.initialized_clients.add(client_id)
                logger.info(f"Successfully initialized RAG for client '{client_id}'")
                return True
            else:
                logger.error(f"Failed to initialize RAG for client '{client_id}'")
                return False
        except Exception as e:
            logger.error(f"Error initializing RAG for client '{client_id}': {e}")
            return False
    
    def get_optimized_prompt(self, client_id: str, query: str, top_k: int = 5) -> str:
        """
        Get optimized prompt for client using RAG
        
        Args:
            client_id: Client identifier
            query: User query
            top_k: Number of relevant schema elements to include
            
        Returns:
            Optimized prompt with relevant schema context
        """
        # Ensure client is initialized
        if client_id not in self.initialized_clients:
            if not self.initialize_client(client_id):
                raise ValueError(f"Client '{client_id}' could not be initialized for RAG")
        
        # Get optimized prompt from RAG system
        optimized_prompt = self.rag_system.get_optimized_prompt(client_id, query, top_k)
        
        # Log token savings for reporting
        self.rag_system.log_token_savings(client_id, query)
        
        return optimized_prompt
    
    def get_rag_health_status(self, client_id: str = None) -> Dict[str, Any]:
        """
        Get RAG system health status for a client or all clients
        
        Args:
            client_id: Optional client identifier. If None, returns status for all clients.
            
        Returns:
            Dictionary with health status
        """
        if client_id:
            # Get status for specific client
            status = {
                "client_id": client_id,
                "rag_enabled": client_id in self.initialized_clients,
                "collection_exists": False,
                "last_update": None,
                "record_count": 0
            }
            
            # Get collection details if client is initialized
            if client_id in self.initialized_clients:
                collection_name = self.rag_system._get_collection_name(client_id)
                if collection_name and self.rag_system.registered_clients and client_id in self.rag_system.registered_clients:
                    status["collection_exists"] = True
                    # Get collection stats if available
                    try:
                        collection = self.rag_system._get_collection(client_id)
                        if collection:
                            status["record_count"] = collection.num_entities
                    except:
                        pass
            
            return status
        else:
            # Get status for all clients
            all_clients = list(self.initialized_clients)
            if hasattr(self.rag_system, "registered_clients"):
                all_clients = list(set(all_clients) | set(self.rag_system.registered_clients))
            
            return {
                "rag_enabled": len(self.initialized_clients) > 0,
                "clients": [self.get_rag_health_status(client) for client in all_clients],
                "total_initialized": len(self.initialized_clients)
            }
    
    def _get_client_data_dictionary_path(self, client_id: str) -> Optional[str]:
        """
        Get the path to client's data dictionary from registry
        
        Args:
            client_id: Client identifier
            
        Returns:
            Path to data dictionary, or None if not found
        """
        # First check in client registry
        if client_id in self.client_registry:
            dict_path = self.client_registry[client_id].get('data_dictionary_path')
            if dict_path and os.path.exists(dict_path):
                logger.info(f"Found data dictionary for {client_id} in registry: {dict_path}")
                return dict_path
        
        # Fallback to standard location if not in registry or path doesn't exist
        base_dir = os.path.join(parent_dir, "config", "clients", "data_dictionaries")
        client_dir = os.path.join(base_dir, client_id)
        
        if os.path.isdir(client_dir):
            # Look for CSV files in client directory
            csv_files = [f for f in os.listdir(client_dir) if f.endswith('.csv')]
            if csv_files:
                # Find client-specific dictionary first
                for csv_file in csv_files:
                    if client_id in csv_file:
                        path = os.path.join(client_dir, csv_file)
                        logger.info(f"Found data dictionary for {client_id} in standard location: {path}")
                        return path
                
                # If no client-specific file found, use first CSV
                path = os.path.join(client_dir, csv_files[0])
                logger.info(f"Using generic dictionary for {client_id}: {path}")
                return path
        
        logger.error(f"No data dictionary found for client {client_id}")
        return None


# Singleton instance
_rag_manager = None

def get_rag_manager() -> ClientRAGManager:
    """
    Get or create the RAG manager singleton
    
    Returns:
        ClientRAGManager instance
    """
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = ClientRAGManager()
    return _rag_manager


# Integration with client health check system
def get_rag_health(client_id: str = None) -> Dict[str, Any]:
    """
    Get RAG health status for client health check integration
    
    Args:
        client_id: Optional client identifier
        
    Returns:
        Dictionary with health status
    """
    manager = get_rag_manager()
    return manager.get_rag_health_status(client_id)
