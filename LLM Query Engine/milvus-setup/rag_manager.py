#!/usr/bin/env python3
"""
RAG Manager - Comprehensive Management Tool for Multi-Client Milvus RAG System
=============================================================================

This script provides a complete solution for:
1. Schema processing with support for both tables and views
2. Collection management (create/drop/rebuild)
3. Embedding generation with error handling
4. RAG initialization for all active clients

Author: DynProInc
Date: 2023-07-22
"""

import os
import sys
import csv
import json
import argparse
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path to handle hyphenated directory name
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import modules using direct paths to avoid issues with hyphenated directories
import importlib.util

def import_from_file(file_path, module_name):
    """Import a module from file path to handle hyphenated directory names"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import necessary modules
schema_processor_module = import_from_file(
    os.path.join(os.path.dirname(__file__), "schema_processor.py"),
    "schema_processor"
)

multi_client_rag_module = import_from_file(
    os.path.join(os.path.dirname(__file__), "multi_client_rag.py"),
    "multi_client_rag"
)

# Import classes directly
SchemaProcessor = schema_processor_module.SchemaProcessor
SchemaRecord = schema_processor_module.SchemaRecord
MultiClientMilvusRAG = multi_client_rag_module.MultiClientMilvusRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "rag_manager.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAGManager")

# Constants
CLIENT_REGISTRY = os.path.join(parent_dir, "config", "clients", "client_registry.csv")
COLUMN_MAPPINGS = os.path.join(parent_dir, "config", "column_mappings.json")

class RAGManager:
    """
    Comprehensive manager for Multi-Client RAG system
    
    Handles schema processing, collection management, embedding generation,
    and RAG initialization for all active clients.
    """
    
    def __init__(self, milvus_host: str = "localhost", milvus_port: str = "19530"):
        """
        Initialize RAG Manager
        
        Args:
            milvus_host: Milvus server hostname
            milvus_port: Milvus server port
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.schema_processor = SchemaProcessor()
        self.rag = MultiClientMilvusRAG(milvus_host, milvus_port)
        logger.info(f"Initialized RAG Manager with Milvus at {milvus_host}:{milvus_port}")
        
    def get_active_clients(self) -> List[Tuple[str, str]]:
        """
        Get list of active clients from registry
        
        Returns:
            List of (client_id, dict_path) tuples for active clients
        """
        clients = []
        
        try:
            with open(CLIENT_REGISTRY, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    client_id = row.get('client_id', '')
                    dict_path = row.get('data_dictionary_path', '')
                    active = row.get('active', 'true').lower() == 'true'
                    
                    if client_id and dict_path and active:
                        clients.append((client_id, dict_path))
                        logger.info(f"Found active client: {client_id}")
        except Exception as e:
            logger.error(f"Error reading client registry: {e}")
            
        return clients
        
    def process_client_dictionary(self, client_id: str, dict_path: str) -> List[SchemaRecord]:
        """
        Process client dictionary CSV into schema records
        
        Args:
            client_id: Client identifier
            dict_path: Path to dictionary CSV file
            
        Returns:
            List of schema records
        """
        try:
            # Check if file exists
            if not os.path.exists(dict_path):
                logger.error(f"Dictionary not found: {dict_path}")
                return []
                
            logger.info(f"Processing dictionary for client {client_id}: {dict_path}")
            
            # Process CSV data - now handles file path directly
            records = self.schema_processor.process_csv_data(client_id, dict_path)
            logger.info(f"Processed {len(records)} schema records for {client_id}")
            
            return records
        except Exception as e:
            logger.error(f"Error processing dictionary for {client_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
            
    def rebuild_client(self, client_id: str, dict_path: str, recreate: bool = True) -> bool:
        """
        Rebuild client collection with schema data
        
        Args:
            client_id: Client identifier
            dict_path: Path to dictionary CSV file
            recreate: If True, recreate collection even if it exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Rebuilding collection for client {client_id}")
            
            # Setup client in RAG system
            success = self.rag.setup_client(client_id, dict_path, recreate=recreate)
            
            if success:
                logger.info(f"Successfully rebuilt collection for {client_id}")
                return True
            else:
                logger.error(f"Failed to rebuild collection for {client_id}")
                return False
        except Exception as e:
            logger.error(f"Error rebuilding collection for {client_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def rebuild_all_clients(self, recreate: bool = True) -> Dict[str, bool]:
        """
        Rebuild collections for all active clients
        
        Args:
            recreate: If True, recreate collections even if they exist
            
        Returns:
            Dictionary with client_id: success mapping
        """
        results = {}
        clients = self.get_active_clients()
        
        if not clients:
            logger.warning("No active clients found in registry")
            return results
            
        logger.info(f"Rebuilding collections for {len(clients)} clients")
        
        for client_id, dict_path in clients:
            success = self.rebuild_client(client_id, dict_path, recreate=recreate)
            results[client_id] = success
            
        # Log summary
        success_count = sum(1 for s in results.values() if s)
        logger.info(f"Rebuild complete: {success_count}/{len(clients)} clients successful")
        
        return results
        
    def drop_all_collections(self) -> bool:
        """
        Drop all collections in Milvus
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from pymilvus import connections, utility
            
            # Ensure we're connected
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            
            # Get all collections
            collections = utility.list_collections()
            
            if not collections:
                logger.info("No collections found to drop")
                return True
                
            logger.info(f"Dropping {len(collections)} collections")
            
            # Drop each collection
            for coll in collections:
                utility.drop_collection(coll)
                logger.info(f"Dropped collection: {coll}")
                
            return True
        except Exception as e:
            logger.error(f"Error dropping collections: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def test_rag_query(self, client_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Test RAG query for a client
        
        Args:
            client_id: Client identifier
            query: Natural language query
            top_k: Number of relevant schema elements to retrieve
            
        Returns:
            List of relevant schema elements
        """
        try:
            # Retrieve relevant schema
            results = self.rag.retrieve_relevant_schema(client_id, query, top_k=top_k)
            
            # Format results based on the structure returned
            formatted = []
            
            # Handle different return types (direct hits or SchemaRecord objects)
            for hit in results:
                if isinstance(hit, dict):
                    # Direct dictionary from Milvus search results
                    entry = {
                        'table_name': hit.get('table_name', ''),
                        'column_name': hit.get('column_name', ''),
                        'data_type': hit.get('data_type', ''),
                        'description': hit.get('description', ''),
                        'score': hit.get('score', None)
                    }
                else:
                    # SchemaRecord object
                    entry = {
                        'table_name': getattr(hit, 'table_name', ''),
                        'column_name': getattr(hit, 'column_name', ''),
                        'data_type': getattr(hit, 'data_type', ''),
                        'description': getattr(hit, 'description', ''),
                        'score': getattr(hit, 'score', None)
                    }
                
                formatted.append(entry)
                
            return formatted
        except Exception as e:
            logger.error(f"Error executing RAG query for {client_id}: {e}")
            import traceback
            error_details = traceback.format_exc()
            logger.error(error_details)
            print(f"\nError details:\n{str(e)}")
            return []
            
    def verify_collections(self) -> Dict[str, Any]:
        """
        Verify collections and entity counts
        
        Returns:
            Dictionary with collection stats
        """
        try:
            from pymilvus import connections, Collection, utility
            
            # Ensure we're connected
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            
            # Get all collections
            collections = utility.list_collections()
            
            if not collections:
                logger.info("No collections found")
                return {'collections': [], 'total_count': 0}
                
            # Get stats for each collection
            stats = []
            total_count = 0
            
            for coll_name in collections:
                try:
                    collection = Collection(coll_name)
                    count = collection.num_entities
                    total_count += count
                    stats.append({
                        'name': coll_name,
                        'count': count
                    })
                except Exception as e:
                    logger.error(f"Error checking collection {coll_name}: {e}")
                    stats.append({
                        'name': coll_name,
                        'error': str(e)
                    })
                    
            return {
                'collections': stats,
                'total_count': total_count
            }
        except Exception as e:
            logger.error(f"Error verifying collections: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='RAG Manager - Comprehensive Management Tool for Multi-Client Milvus RAG System')
    
    # Command options
    parser.add_argument('--rebuild', action='store_true', help='Rebuild all client collections')
    parser.add_argument('--drop', action='store_true', help='Drop all collections')
    parser.add_argument('--verify', action='store_true', help='Verify collections and entity counts')
    parser.add_argument('--test', action='store_true', help='Test RAG query')
    parser.add_argument('--client', type=str, help='Client ID for specific operations')
    parser.add_argument('--query', type=str, help='Query text for test')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results for test query')
    
    args = parser.parse_args()
    
    # Create manager
    manager = RAGManager()
    
    # Handle commands
    if args.drop:
        print("\n" + "="*70)
        print(" DROPPING ALL MILVUS COLLECTIONS")
        print("="*70)
        
        success = manager.drop_all_collections()
        
        if success:
            print("\n✓ Successfully dropped all collections")
        else:
            print("\n✗ Failed to drop collections")
    
    if args.rebuild:
        print("\n" + "="*70)
        print(" REBUILDING CLIENT COLLECTIONS")
        print("="*70)
        
        if args.client:
            # Rebuild specific client
            clients = manager.get_active_clients()
            client_dict = {c[0]: c[1] for c in clients}
            
            if args.client not in client_dict:
                print(f"\n✗ Client {args.client} not found in registry")
                return
                
            print(f"\nRebuilding collection for client: {args.client}")
            success = manager.rebuild_client(args.client, client_dict[args.client])
            
            if success:
                print(f"\n✓ Successfully rebuilt collection for {args.client}")
            else:
                print(f"\n✗ Failed to rebuild collection for {args.client}")
        else:
            # Rebuild all clients
            results = manager.rebuild_all_clients()
            
            # Print summary
            print("\nRebuild Results:")
            for client, success in results.items():
                status = "✓" if success else "✗"
                print(f" {status} {client}")
                
            success_count = sum(1 for s in results.values() if s)
            print(f"\n{success_count}/{len(results)} clients rebuilt successfully")
    
    if args.verify:
        print("\n" + "="*70)
        print(" VERIFYING MILVUS COLLECTIONS")
        print("="*70)
        
        stats = manager.verify_collections()
        
        if 'error' in stats:
            print(f"\n✗ Error: {stats['error']}")
            return
            
        print(f"\nFound {len(stats['collections'])} collections with {stats['total_count']} total entities:")
        
        for coll in stats['collections']:
            if 'error' in coll:
                print(f" ✗ {coll['name']}: ERROR - {coll['error']}")
            else:
                print(f" ✓ {coll['name']}: {coll['count']} entities")
    
    if args.test:
        if not args.client or not args.query:
            print("\n✗ Both --client and --query are required for testing")
            return
            
        print("\n" + "="*70)
        print(f" TESTING RAG QUERY FOR {args.client}")
        print("="*70)
        
        print(f"\nQuery: {args.query}")
        results = manager.test_rag_query(args.client, args.query, top_k=args.top_k)
        
        if not results:
            print("\n✗ No results or error occurred")
            return
            
        print(f"\nFound {len(results)} relevant items:")
        
        for i, r in enumerate(results):
            print(f"\n{i+1}. {r['table_name']}.{r['column_name']} ({r['data_type']})")
            if r['description']:
                print(f"   Description: {r['description']}")
            if r['score']:
                print(f"   Relevance: {r['score']:.4f}")
                
        # Estimate token savings
        print("\nEstimating token savings:")
        token_info = manager.rag.estimate_token_savings(args.client, args.query)
        print(f" • RAG context tokens: {token_info.get('rag_tokens', 'N/A')}")
        print(f" • Full schema tokens: {token_info.get('full_schema_tokens', 'N/A')}")
        print(f" • Token savings: {token_info.get('token_savings', 'N/A')}%")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)
