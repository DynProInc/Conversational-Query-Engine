#!/usr/bin/env python3
"""
Simple Milvus Collection Rebuilder
=================================

Clear and rebuild all Milvus collections with the new dynamic column mappings.
Supports both tables and views in database schemas.
"""

import os
import sys
import csv
import pandas as pd
from pymilvus import connections, utility, Collection

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import required modules directly using file paths
import importlib.util

def import_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import necessary modules
schema_processor = import_from_file(
    os.path.join(os.path.dirname(__file__), "schema_processor.py"),
    "schema_processor"
)

multi_client_rag = import_from_file(
    os.path.join(os.path.dirname(__file__), "multi_client_rag.py"),
    "multi_client_rag"
)

# Client registry path
CLIENT_REGISTRY = os.path.join(parent_dir, "config", "clients", "client_registry.csv")

def main():
    """Main function to rebuild collections"""
    print("\n" + "="*70)
    print(" MILVUS COLLECTION REBUILD WITH DYNAMIC COLUMN MAPPINGS")
    print("="*70)
    print("\nThis script will rebuild all collections with the new mappings that")
    print("support both tables and views in your database schemas.")
    
    # Step 1: Connect to Milvus
    print("\n[1/4] Connecting to Milvus...")
    try:
        connections.connect(alias="default", host="localhost", port="19530")
        print("✓ Connected to Milvus server")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False
    
    # Step 2: Drop existing collections
    print("\n[2/4] Dropping existing collections...")
    try:
        collections = utility.list_collections()
        if collections:
            print(f"Found {len(collections)} existing collections:")
            for coll in collections:
                try:
                    utility.drop_collection(coll)
                    print(f"  ✓ Dropped collection: {coll}")
                except Exception as e:
                    print(f"  ✗ Error dropping {coll}: {e}")
        else:
            print("No existing collections to drop")
    except Exception as e:
        print(f"✗ Error listing collections: {e}")
        return False
    
    # Step 3: Load client registry
    print("\n[3/4] Loading client registry...")
    clients = []
    
    try:
        if not os.path.exists(CLIENT_REGISTRY):
            print(f"✗ Client registry not found at: {CLIENT_REGISTRY}")
            return False
            
        with open(CLIENT_REGISTRY, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                client_id = row.get('client_id', '')
                dict_path = row.get('data_dictionary_path', '')
                active = row.get('active', 'true').lower() == 'true'
                
                if client_id and dict_path and active:
                    clients.append((client_id, dict_path))
                    print(f"  ✓ Found active client: {client_id}")
    except Exception as e:
        print(f"✗ Error reading client registry: {e}")
        return False
    
    if not clients:
        print("✗ No active clients found in registry")
        return False
    
    # Step 4: Process clients
    print("\n[4/4] Rebuilding collections...")
    
    # Create RAG instance
    try:
        rag = multi_client_rag.MultiClientMilvusRAG()
        print("✓ Initialized RAG system")
    except Exception as e:
        print(f"✗ Error initializing RAG system: {e}")
        return False
    
    # Process each client
    success_count = 0
    for client_id, dict_path in clients:
        print(f"\nProcessing client: {client_id}")
        
        try:
            if not os.path.exists(dict_path):
                print(f"  ✗ Dictionary not found: {dict_path}")
                continue
                
            print(f"  • Dictionary path: {dict_path}")
            
            # Call setup_client to handle everything
            print(f"  • Setting up client collection with schema...")
            
            # Make sure to flush stdout to show progress immediately
            sys.stdout.flush()
            
            success = rag.setup_client(client_id, dict_path, recreate=True)
            
            if success:
                print(f"  ✓ Successfully set up client {client_id}")
                success_count += 1
                
                # Verify collection
                coll_name = rag._get_collection_name(client_id)
                collection = Collection(coll_name)
                count = collection.num_entities
                print(f"  ✓ Collection has {count} records")
            else:
                print(f"  ✗ Failed to set up client {client_id}")
        except Exception as e:
            import traceback
            print(f"  ✗ Error processing client {client_id}: {str(e)}")
            print(traceback.format_exc())
    
    # Final summary
    print("\n" + "="*70)
    if success_count == len(clients):
        print(f"✓ SUCCESS: All {success_count} clients were successfully rebuilt")
    else:
        print(f"⚠ PARTIAL SUCCESS: {success_count}/{len(clients)} clients rebuilt")
    print("="*70)
    
    return success_count > 0

if __name__ == "__main__":
    successful = main()
    sys.exit(0 if successful else 1)
