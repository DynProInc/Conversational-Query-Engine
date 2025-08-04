#!/usr/bin/env python3
"""
Multi-Client Embedding Integration Guide
=======================================

This script demonstrates and explains the entire embedding flow
for multi-client systems with clear step-by-step execution.

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Define client registry path
CLIENT_REGISTRY_PATH = os.path.join(parent_dir, "config", "clients", "client_registry.csv")

def print_header(title, width=80):
    """Print a formatted header"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")

def print_step(step_num, description):
    """Print a step header"""
    print(f"\n--- STEP {step_num}: {description} ---\n")

def load_clients():
    """Load and validate client registry"""
    print_step(1, "Loading client registry")
    
    if not os.path.exists(CLIENT_REGISTRY_PATH):
        print(f"ERROR: Client registry not found: {CLIENT_REGISTRY_PATH}")
        return {}
    
    clients = {}
    try:
        with open(CLIENT_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                client_id = row['client_id']
                clients[client_id] = {
                    'name': row.get('client_name', ''),
                    'active': row.get('active', 'true').lower() == 'true',
                    'data_dictionary_path': row.get('data_dictionary_path', '')
                }
        
        print(f"Successfully loaded {len(clients)} clients")
        for client_id, info in clients.items():
            status = "ACTIVE" if info['active'] else "INACTIVE"
            print(f"- {client_id}: {info['name']} [{status}]")
            print(f"  Dictionary: {info['data_dictionary_path']}")
            print(f"  Exists: {os.path.exists(info['data_dictionary_path'])}")
        
        return clients
    except Exception as e:
        print(f"ERROR loading client registry: {str(e)}")
        return {}

def process_client_dictionary(client_id, dict_path):
    """Process client dictionary into schema records"""
    print_step(2, f"Processing dictionary for client: {client_id}")
    
    if not os.path.exists(dict_path):
        print(f"ERROR: Dictionary does not exist: {dict_path}")
        return None
    
    try:
        # Import schema processor
        from schema_processor import SchemaProcessor
        
        # Load dictionary as dataframe
        df = pd.read_csv(dict_path)
        print(f"Loaded dictionary with {len(df)} rows")
        
        # Display columns
        print("\nColumns in dictionary:")
        for col in df.columns:
            print(f"- {col}")
        
        # Process with schema processor
        processor = SchemaProcessor()
        records = processor.process_csv_data(client_id, df)
        
        print(f"\nProcessed {len(records)} schema records")
        
        # Show sample record
        if records:
            sample = records[0]
            print("\nSample processed record:")
            print(f"Table: {sample.table_name}")
            print(f"Column: {sample.column_name}")
            print(f"Data Type: {sample.data_type or 'N/A'}")
            print(f"Description: {sample.description or 'N/A'}")
            
            # Show combined text
            print("\nCombined text for embedding:")
            print("-" * 40)
            print(sample.combined_text)
            print("-" * 40)
        
        return records
    except Exception as e:
        import traceback
        print(f"ERROR processing dictionary: {str(e)}")
        print(traceback.format_exc())
        return None

def generate_embeddings(records):
    """Generate embeddings for schema records"""
    print_step(3, "Generating embeddings")
    
    if not records:
        print("No records to embed")
        return None
    
    try:
        # Import embedding model
        from sentence_transformers import SentenceTransformer
        
        # Initialize model
        model_name = "all-MiniLM-L6-v2"
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Prepare texts for embedding
        texts = [record.combined_text for record in records]
        print(f"Embedding {len(texts)} texts")
        
        # Generate embeddings
        embeddings = model.encode(texts)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    except Exception as e:
        import traceback
        print(f"ERROR generating embeddings: {str(e)}")
        print(traceback.format_exc())
        return None

def save_embeddings(client_id, records, embeddings):
    """Save embeddings and records to disk"""
    print_step(4, "Saving embeddings to disk")
    
    if embeddings is None or records is None:
        print("No embeddings or records to save")
        return False
    
    try:
        # Create directory
        embedding_dir = os.path.join(os.path.dirname(__file__), "embeddings", client_id)
        os.makedirs(embedding_dir, exist_ok=True)
        print(f"Using directory: {embedding_dir}")
        
        # Save embeddings
        embeddings_path = os.path.join(embedding_dir, f"{client_id}_embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to: {embeddings_path}")
        
        # Save records
        records_path = os.path.join(embedding_dir, f"{client_id}_records.json")
        
        # Convert records to JSON serializable format
        records_json = []
        for record in records:
            records_json.append({
                'client_id': record.client_id,
                'db_schema': record.db_schema,
                'table_name': record.table_name,
                'column_name': record.column_name,
                'data_type': record.data_type,
                'description': record.description,
                'distinct_values': record.distinct_values,
                'combined_text': record.combined_text
            })
        
        with open(records_path, 'w') as f:
            json.dump(records_json, f)
            
        print(f"Saved records to: {records_path}")
        
        # Save metadata
        meta_path = os.path.join(embedding_dir, f"{client_id}_metadata.json")
        metadata = {
            'client_id': client_id,
            'embedding_model': "all-MiniLM-L6-v2",
            'record_count': len(records),
            'embedding_shape': embeddings.shape,
            'last_updated': datetime.now().isoformat(),
            'dictionary_path': None  # Will be set later
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
            
        print(f"Saved metadata to: {meta_path}")
        return True
        
    except Exception as e:
        import traceback
        print(f"ERROR saving embeddings: {str(e)}")
        print(traceback.format_exc())
        return False

def initialize_rag(client_id):
    """Initialize RAG system with embeddings"""
    print_step(5, "Initializing RAG system")
    
    try:
        # Import RAG components
        from client_rag_manager import ClientRAGManager
        
        # Initialize RAG manager
        rag_manager = ClientRAGManager()
        
        # Check if client is in registry
        if client_id not in rag_manager.client_registry:
            print(f"ERROR: Client {client_id} not found in RAG manager registry")
            return False
        
        # Get dictionary path
        dict_path = rag_manager._get_client_data_dictionary_path(client_id)
        print(f"Using dictionary path: {dict_path}")
        
        # Initialize client
        print(f"Initializing client in RAG system")
        success = rag_manager.initialize_client(client_id)
        
        if success:
            print(f"Successfully initialized client {client_id} in RAG system")
        else:
            print(f"Failed to initialize client {client_id} in RAG system")
        
        # Get health status
        health = rag_manager.get_rag_health_status(client_id)
        print("\nRAG health status:")
        for key, value in health.items():
            print(f"- {key}: {value}")
        
        return success
    except Exception as e:
        import traceback
        print(f"ERROR initializing RAG: {str(e)}")
        print(traceback.format_exc())
        return False

def run_complete_flow(client_id):
    """Run the complete embedding flow for a client"""
    print_header(f"COMPLETE EMBEDDING FLOW FOR CLIENT: {client_id}")
    
    # Step 1: Load clients
    clients = load_clients()
    if not clients or client_id not in clients:
        print(f"ERROR: Client {client_id} not found in registry")
        return
    
    # Step 2: Process dictionary
    dict_path = clients[client_id]['data_dictionary_path']
    records = process_client_dictionary(client_id, dict_path)
    if not records:
        print("Failed to process dictionary - stopping flow")
        return
    
    # Step 3: Generate embeddings
    embeddings = generate_embeddings(records)
    if embeddings is None:
        print("Failed to generate embeddings - stopping flow")
        return
    
    # Step 4: Save embeddings
    saved = save_embeddings(client_id, records, embeddings)
    if not saved:
        print("Failed to save embeddings - stopping flow")
        return
    
    # Step 5: Initialize RAG
    initialize_rag(client_id)
    
    # Complete
    print_header("EMBEDDING FLOW COMPLETED")
    print("The multi-client embedding generation flow has been completed.")
    print("Embeddings are now ready for use in the RAG system.")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Client Embedding Flow Guide")
    parser.add_argument("--client", required=True, help="Client ID to process")
    args = parser.parse_args()
    
    run_complete_flow(args.client)

if __name__ == "__main__":
    main()
