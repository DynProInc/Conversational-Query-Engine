#!/usr/bin/env python3
"""
Multi-Client Embedding Pipeline Runner
=====================================

This script runs the complete embedding pipeline for all clients:
1. Checks current embedding status
2. Generates embeddings for clients that need updates
3. Verifies embedding health after generation
4. Initializes RAG system with new embeddings

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import logging
import argparse
import traceback
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmbeddingPipeline")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary modules
try:
    # First import the client manager for registry access
    sys.path.append(os.path.join(parent_dir, "config"))
    
    # Then import milvus setup modules
    sys.path.append(os.path.join(parent_dir, "milvus-setup"))
    
    from generate_client_embeddings import ClientEmbeddingGenerator, main as generate_embeddings
    from check_embedding_status import check_embedding_status, load_client_registry
    from schema_processor import SchemaProcessor
    
    # Try to import RAG components if needed
    try:
        from client_rag_manager import ClientRAGManager
        from multi_client_rag import MultiClientMilvusRAG
        rag_available = True
    except ImportError:
        logger.warning("RAG components not available - skipping RAG initialization")
        rag_available = False
        
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def check_all_clients_status():
    """Check embedding status for all clients and return summary"""
    logger.info("Checking embedding status for all clients...")
    
    # Load client registry
    clients = load_client_registry()
    
    if not clients:
        logger.error("No clients found in registry")
        return {}
    
    # Check status for each client
    status_summary = {}
    for client_id, client_info in clients.items():
        if not client_info.get('active', True):
            logger.info(f"Skipping inactive client: {client_id}")
            continue
            
        try:
            status = check_embedding_status(client_id)
            status_summary[client_id] = status
        except Exception as e:
            logger.error(f"Error checking status for client {client_id}: {str(e)}")
            status_summary[client_id] = {"error": str(e)}
    
    return status_summary

def generate_embeddings_for_clients(client_ids: Optional[List[str]] = None, force: bool = False):
    """Generate embeddings for specified clients or all if None"""
    logger.info(f"Generating embeddings for clients: {client_ids or 'ALL'}")
    
    try:
        # Call the main function from generate_client_embeddings.py
        generate_embeddings(clients=client_ids, force_update=force)
        logger.info("Embedding generation completed")
        return True
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def initialize_rag_for_clients(client_ids: Optional[List[str]] = None):
    """Initialize RAG system for specified clients"""
    if not rag_available:
        logger.warning("RAG components not available - skipping initialization")
        return False
    
    logger.info(f"Initializing RAG for clients: {client_ids or 'ALL'}")
    
    try:
        # Initialize RAG manager
        rag_manager = ClientRAGManager()
        
        # Load client registry if client_ids is None
        if client_ids is None:
            client_ids = list(rag_manager.client_registry.keys())
        
        # Initialize each client
        for client_id in client_ids:
            if client_id not in rag_manager.client_registry:
                logger.warning(f"Client {client_id} not found in registry - skipping")
                continue
                
            if not rag_manager.client_registry[client_id].get('active', True):
                logger.info(f"Skipping inactive client: {client_id}")
                continue
            
            logger.info(f"Initializing client: {client_id}")
            success = rag_manager.initialize_client(client_id)
            logger.info(f"Client {client_id} initialization: {'SUCCESS' if success else 'FAILED'}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Multi-Client Embedding Pipeline Runner")
    parser.add_argument("--clients", nargs="+", help="Specific client IDs to process (default: all active clients)")
    parser.add_argument("--force", action="store_true", help="Force regeneration of embeddings even if up-to-date")
    parser.add_argument("--status-only", action="store_true", help="Only check status without generating embeddings")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG system initialization")
    args = parser.parse_args()
    
    # First check status of all clients
    logger.info("=== CHECKING CURRENT EMBEDDING STATUS ===")
    status_summary = check_all_clients_status()
    
    # Print status summary
    print("\n=== EMBEDDING STATUS SUMMARY ===")
    for client_id, status in status_summary.items():
        print(f"\nClient: {client_id}")
        if "error" in status:
            print(f"  ERROR: {status['error']}")
            continue
            
        print(f"  Embeddings exist: {status.get('embeddings_exist', False)}")
        if status.get('embeddings_exist', False):
            print(f"  Up to date: {status.get('up_to_date', False)}")
            print(f"  Record count: {status.get('record_count', 0)}")
            print(f"  Last updated: {status.get('last_updated', 'N/A')}")
    
    if args.status_only:
        logger.info("Status check completed. Exiting.")
        return
    
    # Generate embeddings if needed
    needs_update = [
        client_id for client_id, status in status_summary.items()
        if "error" not in status and (
            args.force or 
            not status.get('embeddings_exist', False) or 
            not status.get('up_to_date', False)
        )
    ]
    
    if needs_update or args.force:
        logger.info("=== GENERATING EMBEDDINGS ===")
        clients_to_update = args.clients if args.clients else needs_update
        if not clients_to_update:
            logger.info("No clients need updating.")
        else:
            logger.info(f"Clients to update: {clients_to_update}")
            generate_embeddings_for_clients(clients_to_update, args.force)
    else:
        logger.info("All embeddings are up to date. Skipping generation.")
    
    # Check status again after generation
    if needs_update or args.force:
        logger.info("=== CHECKING UPDATED EMBEDDING STATUS ===")
        status_summary = check_all_clients_status()
    
    # Initialize RAG system if requested
    if not args.skip_rag:
        logger.info("=== INITIALIZING RAG SYSTEM ===")
        initialize_rag_for_clients(args.clients)

if __name__ == "__main__":
    main()
