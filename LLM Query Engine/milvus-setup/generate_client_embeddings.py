#!/usr/bin/env python3
"""
Multi-Client Dictionary Embedding Generator
==========================================

Production script to generate and store embeddings for all client data dictionaries.
Implements robust schema processing for all clients with different dictionary formats.

Features:
- Robust schema processing with support for missing columns
- Client-specific embedding generation with strict isolation
- Embedding update detection based on dictionary modification time
- Integration with client health check system

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import csv
import logging
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ClientEmbeddings")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import schema processor and client management
from schema_processor import SchemaProcessor, SchemaRecord

# Define client registry path
CLIENT_REGISTRY_PATH = os.path.join(parent_dir, "config", "clients", "client_registry.csv")

class ClientEmbeddingGenerator:
    """Generate and store embeddings for multi-client data dictionaries with update detection"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: SentenceTransformer model name to use
        """
        self.model_name = model_name
        logger.info(f"Initializing embedding generator with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model initialized with {self.embedding_dim} dimensions")
        
        self.schema_processor = SchemaProcessor()
        self.clients = self._load_client_registry()
    
    def _load_client_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the client registry from CSV
        
        Returns:
            Dictionary of client information indexed by client_id
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
                    
                    # Store only what we need for embedding generation
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
        
    def get_active_clients(self) -> List[str]:
        """Get list of active clients from registry"""
        active_clients = [client_id for client_id, client_info in self.clients.items() 
                         if client_info.get('active', True)]
        
        if not active_clients:
            logger.warning("No active clients found in registry")
            return []
            
        logger.info(f"Found {len(active_clients)} active clients: {', '.join(active_clients)}")
        return active_clients
    
    def process_client_dictionary(self, client_id: str) -> List[SchemaRecord]:
        """
        Process a client's dictionary CSV file into schema records
        
        Args:
            client_id: Client identifier
            
        Returns:
            List of processed schema records
        """
        # Check if client exists in registry
        if client_id not in self.clients:
            logger.error(f"Client {client_id} not found in registry")
            return []
        
        # Get dictionary path from registry
        dict_path = self.clients[client_id]['data_dictionary_path']
        
        if not dict_path or not os.path.exists(dict_path):
            logger.error(f"Dictionary path not found for client {client_id}: {dict_path}")
            return []
        
        logger.info(f"Processing dictionary for client {client_id}: {dict_path}")
        
        try:
            # Read and process CSV data
            df = pd.read_csv(dict_path)
            logger.info(f"Read {len(df)} rows with columns: {', '.join(df.columns)}")
            
            # Process into schema records
            records = self.schema_processor.process_csv_data(client_id, df)
            logger.info(f"Processed {len(records)} schema records for client {client_id}")
            
            return records
        
        except Exception as e:
            logger.error(f"Error processing dictionary for client {client_id}: {str(e)}")
            return []
    
    def generate_client_embeddings(self, client_id: str) -> Tuple[List[SchemaRecord], np.ndarray]:
        """
        Generate embeddings for a client's schema records
        
        Args:
            client_id: Client identifier
            
        Returns:
            Tuple of (schema records, embeddings array)
        """
        # Get schema records
        records = self.process_client_dictionary(client_id)
        if not records:
            return [], np.array([])
        
        # Extract texts for embedding
        texts = [record.combined_text for record in records]
        
        # Generate embeddings
        logger.info(f"Generating {len(texts)} embeddings for client {client_id}")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return records, embeddings
    
    def process_all_clients(self) -> Dict[str, Tuple[List[SchemaRecord], np.ndarray]]:
        """
        Process all available clients and generate embeddings
        
        Returns:
            Dictionary mapping client_id to (records, embeddings) tuples
        """
        result = {}
        clients = self.get_active_clients()
        
        for client_id in clients:
            logger.info(f"Processing client: {client_id}")
            records, embeddings = self.generate_client_embeddings(client_id)
            
            if len(records) > 0:
                result[client_id] = (records, embeddings)
                
                # Output sample record info for verification
                sample_record = records[0]
                logger.info(f"Sample record for {client_id}: {sample_record.table_name}.{sample_record.column_name}")
                
                # Show which columns are included in the embedding
                components = []
                if sample_record.db_schema:
                    db, schema = sample_record.extract_db_schema()
                    if db: components.append("Database")
                    if schema: components.append("Schema")
                if sample_record.table_name: components.append("Table")
                if sample_record.column_name: components.append("Column")
                if sample_record.data_type: components.append("DataType")
                if sample_record.description: components.append("Description")
                if sample_record.distinct_values: components.append("DistinctValues")
                
                logger.info(f"Embedding includes: {', '.join(components)}")
                logger.info(f"Generated {len(records)} embeddings for client {client_id}")
            else:
                logger.warning(f"No records processed for client {client_id}")
        
        return result
    
    def save_embeddings(self, client_id: str, records: List[SchemaRecord], 
                       embeddings: np.ndarray, output_dir: str = None) -> str:
        """
        Save embeddings and records to disk
        
        Args:
            client_id: Client identifier
            records: List of schema records
            embeddings: NumPy array of embeddings
            output_dir: Directory to save files (defaults to milvus-setup/embeddings)
            
        Returns:
            Path where files were saved
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create client directory
        client_dir = os.path.join(output_dir, client_id)
        os.makedirs(client_dir, exist_ok=True)
        
        # Save embeddings
        embeddings_file = os.path.join(client_dir, f"{client_id}_embeddings.npy")
        np.save(embeddings_file, embeddings)
        
        # Save record data for reference
        record_data = {
            "db_schema": [r.db_schema for r in records],
            "table_name": [r.table_name for r in records],
            "column_name": [r.column_name for r in records],
            "data_type": [r.data_type for r in records],
            "description": [r.description for r in records],
            "combined_text": [r.combined_text for r in records],
        }
        
        # Add distinct_values if any record has them
        if any(r.distinct_values for r in records):
            record_data["distinct_values"] = [r.distinct_values for r in records]
        
        # Save as CSV
        records_df = pd.DataFrame(record_data)
        records_file = os.path.join(client_dir, f"{client_id}_records.csv")
        records_df.to_csv(records_file, index=False)
        
        logger.info(f"Saved {len(embeddings)} embeddings and records for {client_id} to {client_dir}")
        return client_dir

    def should_update_embeddings(self, client_id: str) -> bool:
        """
        Check if embeddings need to be updated for a client
        based on dictionary file modification time
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if embeddings need to be updated, False otherwise
        """
        # Check if client exists in registry
        if client_id not in self.clients:
            logger.error(f"Client {client_id} not found in registry")
            return True  # If client not in registry, assume we need to update
            
        # Get dictionary path from registry
        dict_path = self.clients[client_id]['data_dictionary_path']
        
        if not dict_path or not os.path.exists(dict_path):
            logger.error(f"Dictionary path not found for client {client_id}: {dict_path}")
            return True  # If we can't find the dictionary, assume we need to update
        
        # Path to saved embeddings
        embedding_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "embeddings", client_id
        )
        embedding_path = os.path.join(embedding_dir, f"{client_id}_embeddings.npy")
        
        # If embeddings don't exist, generate them
        if not os.path.exists(embedding_path):
            logger.info(f"No existing embeddings found for {client_id}, update needed")
            return True
        
        try:
            # If dictionary was modified after embeddings were created, update them
            dict_mtime = os.path.getmtime(dict_path)
            emb_mtime = os.path.getmtime(embedding_path)
            
            needs_update = dict_mtime > emb_mtime
            
            if needs_update:
                dict_time = datetime.fromtimestamp(dict_mtime).strftime("%Y-%m-%d %H:%M:%S")
                emb_time = datetime.fromtimestamp(emb_mtime).strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"Dictionary for {client_id} was updated on {dict_time} "  
                           f"after embeddings were generated on {emb_time}, update needed")
            else:
                logger.info(f"Embeddings for {client_id} are up to date")
                
            return needs_update
            
        except Exception as e:
            logger.error(f"Error checking update status for {client_id}: {str(e)}")
            return True  # If there's an error, assume we need to update
    
    def get_client_embedding_status(self, client_id: str) -> Dict[str, Any]:
        """
        Get detailed status of a client's embeddings
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with embedding status information
        """
        # Path to client dictionary
        data_dict_path = os.path.join(
            parent_dir, "config", "clients", "data_dictionaries", client_id
        )
        
        # Path to saved embeddings
        embedding_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "embeddings", client_id
        )
        embedding_path = os.path.join(embedding_dir, f"{client_id}_embeddings.npy")
        records_path = os.path.join(embedding_dir, f"{client_id}_records.csv")
        
        status = {
            "client_id": client_id,
            "embeddings_exist": os.path.exists(embedding_path),
            "records_exist": os.path.exists(records_path),
            "up_to_date": not self.should_update_embeddings(client_id) if os.path.exists(embedding_path) else False,
            "last_updated": None,
            "record_count": 0,
            "embedding_dimensions": self.embedding_dim,
            "model_name": self.model_name
        }
        
        # Add additional info if embeddings exist
        if os.path.exists(embedding_path):
            try:
                embeddings = np.load(embedding_path)
                status["record_count"] = len(embeddings)
                status["last_updated"] = datetime.fromtimestamp(
                    os.path.getmtime(embedding_path)
                ).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.error(f"Error loading embeddings for {client_id}: {str(e)}")
        
        return status

    def get_all_clients_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get embedding status for all available clients
        
        Returns:
            Dictionary mapping client_id to status dictionaries
        """
        result = {}
        clients = self.get_active_clients()
        
        for client_id in clients:
            result[client_id] = self.get_client_embedding_status(client_id)
        
        return result

def main(clients=None, force_update=False):
    """
    Main entry point
    
    Args:
        clients: Optional list of client IDs to process (None for all)
        force_update: If True, regenerate embeddings even if up-to-date
    """
    logger.info("Starting multi-client embedding generation")
    
    generator = ClientEmbeddingGenerator()
    
    # If no clients specified, get all available clients
    if clients is None:
        clients = generator.get_client_directories()
    
    # Track results
    updated_clients = []
    skipped_clients = []
    failed_clients = []
    client_results = {}
    
    # Process each client
    for client_id in clients:
        logger.info(f"Processing client: {client_id}")
        
        # Check if embeddings need updating
        if not force_update and not generator.should_update_embeddings(client_id):
            logger.info(f"Skipping {client_id}: embeddings are up-to-date")
            skipped_clients.append(client_id)
            continue
        
        # Generate and save embeddings
        try:
            records, embeddings = generator.generate_client_embeddings(client_id)
            
            if len(records) > 0:
                client_results[client_id] = (records, embeddings)
                generator.save_embeddings(client_id, records, embeddings)
                updated_clients.append(client_id)
            else:
                logger.warning(f"No records processed for {client_id}")
                failed_clients.append(client_id)
                
        except Exception as e:
            logger.error(f"Error processing {client_id}: {str(e)}")
            failed_clients.append(client_id)
    
    # Output summary
    total_updated = len(updated_clients)
    total_skipped = len(skipped_clients)
    total_failed = len(failed_clients)
    total_records = sum(len(records) for records, _ in client_results.values()) if client_results else 0
    
    logger.info(f"Successfully updated {total_updated} clients with {total_records} total records")
    if skipped_clients:
        logger.info(f"Skipped {total_skipped} clients with up-to-date embeddings: {', '.join(skipped_clients)}")
    if failed_clients:
        logger.warning(f"Failed to process {total_failed} clients: {', '.join(failed_clients)}")
    
    logger.info("Multi-client embedding generation completed successfully")
    return True

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate embeddings for client data dictionaries')
    parser.add_argument('--clients', nargs='*', help='Specific clients to process (default: all)')
    parser.add_argument('--force', action='store_true', help='Force update even if embeddings are up-to-date')
    parser.add_argument('--status', action='store_true', help='Only show embedding status without updating')
    
    args = parser.parse_args()
    
    if args.status:
        # Only show status
        generator = ClientEmbeddingGenerator()
        status = generator.get_all_clients_status()
        
        print("\nCLIENT EMBEDDING STATUS:")
        print("=========================")
        
        for client_id, client_status in status.items():
            print(f"\nClient: {client_id}")
            print(f"  Embeddings exist: {client_status['embeddings_exist']}")
            print(f"  Up to date: {client_status['up_to_date']}")
            print(f"  Last updated: {client_status['last_updated'] or 'Never'}")
            print(f"  Record count: {client_status['record_count']}")
            print(f"  Model: {client_status['model_name']} ({client_status['embedding_dimensions']} dimensions)")
    else:
        # Generate embeddings
        success = main(clients=args.clients, force_update=args.force)
        sys.exit(0 if success else 1)
