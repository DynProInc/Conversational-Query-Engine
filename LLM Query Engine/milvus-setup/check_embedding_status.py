#!/usr/bin/env python3
"""
Simple Embedding Status Checker
==============================

Shows the status of client dictionary embeddings with clear output
compatible with PowerShell.

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import csv
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Define client registry path
CLIENT_REGISTRY_PATH = os.path.join(parent_dir, "config", "clients", "client_registry.csv")

def load_client_registry():
    """Load the client registry from CSV"""
    clients = {}
    
    if not os.path.exists(CLIENT_REGISTRY_PATH):
        print(f"Error: Client registry not found at {CLIENT_REGISTRY_PATH}")
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
        
    except Exception as e:
        print(f"Error loading client registry: {str(e)}")
    
    return clients

def get_active_clients():
    """Get list of active clients from registry"""
    clients = load_client_registry()
    active_clients = [client_id for client_id, client_info in clients.items() 
                     if client_info.get('active', True)]
    
    return active_clients

def check_embedding_status(client_id):
    """Check if embeddings exist and are up-to-date"""
    # Get client data from registry
    clients = load_client_registry()
    
    if client_id not in clients:
        print(f"Error: Client {client_id} not found in registry")
        return {
            "client_id": client_id,
            "embeddings_exist": False,
            "records_exist": False,
            "dict_file": None,
            "record_count": 0,
            "up_to_date": False,
            "last_updated": None,
        }
    
    # Get dictionary path from registry
    dict_file = clients[client_id]['data_dictionary_path']
    
    # Embedding path
    emb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings", client_id)
    emb_file = os.path.join(emb_dir, f"{client_id}_embeddings.npy")
    rec_file = os.path.join(emb_dir, f"{client_id}_records.csv")
    
    # Status collection
    status = {
        "client_id": client_id,
        "embeddings_exist": os.path.exists(emb_file),
        "records_exist": os.path.exists(rec_file),
        "dict_file": dict_file,
        "record_count": 0,
        "up_to_date": False,
        "last_updated": None,
    }
    
    # Check if dictionary is newer than embeddings
    if dict_file and os.path.exists(emb_file):
        dict_time = os.path.getmtime(dict_file)
        emb_time = os.path.getmtime(emb_file)
        status["up_to_date"] = dict_time <= emb_time
        status["last_updated"] = datetime.fromtimestamp(emb_time).strftime("%Y-%m-%d %H:%M:%S")
        
        # Get record count
        try:
            embeddings = np.load(emb_file)
            status["record_count"] = len(embeddings)
        except:
            pass
    
    return status

def main():
    """Main function"""
    print("\n===== CLIENT EMBEDDING STATUS =====\n")
    
    clients = get_active_clients()
    
    if not clients:
        print("No client directories found.")
        return
    
    # Check each client
    healthy = []
    degraded = []
    unhealthy = []
    
    for client_id in clients:
        status = check_embedding_status(client_id)
        
        print(f"Client: {client_id}")
        
        if status["embeddings_exist"] and status["up_to_date"]:
            print(f"  Status: HEALTHY")
            healthy.append(client_id)
        elif status["embeddings_exist"]:
            print(f"  Status: DEGRADED (outdated)")
            degraded.append(client_id)
        else:
            print(f"  Status: UNHEALTHY (missing)")
            unhealthy.append(client_id)
            
        print(f"  Embeddings exist: {status['embeddings_exist']}")
        print(f"  Records exist: {status['records_exist']}")
        
        if status["embeddings_exist"]:
            print(f"  Up to date: {status['up_to_date']}")
            print(f"  Last updated: {status['last_updated']}")
            print(f"  Record count: {status['record_count']}")
            
        print("")
    
    # Summary
    print("\n===== SUMMARY =====")
    print(f"Total clients: {len(clients)}")
    print(f"Healthy clients: {len(healthy)}")
    print(f"Degraded clients: {len(degraded)}")
    print(f"Unhealthy clients: {len(unhealthy)}")
    
    # Integration with client-specific health check system
    print("\n===== INTEGRATION WITH CLIENT HEALTH CHECK =====")
    print("To integrate with your health check system:")
    print("1. Add embedding status checks to your client health endpoints")
    print("2. Include embedding health in overall client health status")
    print("3. Add embedding health to your API response format")
    
    print("\nExample code for health API integration:")
    print("-" * 50)
    print("from milvus-setup.generate_client_embeddings import ClientEmbeddingGenerator\n")
    print("def check_client_health(client_id):")
    print("    # Check LLM API keys, Snowflake connection, etc.")
    print("    # ...")
    print("    # Add embedding health check")
    print("    generator = ClientEmbeddingGenerator()")
    print("    embedding_status = generator.get_client_embedding_status(client_id)")
    print("    health_data['components']['embeddings'] = {")
    print("        'status': 'healthy' if embedding_status['up_to_date'] else 'degraded',")
    print("        'last_updated': embedding_status['last_updated'],")
    print("        'record_count': embedding_status['record_count']")
    print("    }")
    print("    return health_data")
    
    print("\n===== EMBEDDING UPDATE PROCESS =====")
    print("When to update embeddings:")
    print("1. Initial setup: Run once for all clients")
    print("2. Dictionary updates: Only when a client's dictionary changes")
    print("3. NOT on every query: This would be inefficient")
    print("\nCommand to update a specific client's embeddings:")
    print(f"python milvus-setup\\generate_client_embeddings.py --clients CLIENT_ID")
    print("\nCommand to update all outdated client embeddings:")
    print(f"python milvus-setup\\generate_client_embeddings.py")
    

if __name__ == "__main__":
    main()
