#!/usr/bin/env python3
"""
RAG Fix Script - Complete solution to diagnose and fix the Milvus RAG system
"""

import os
import sys
import csv
import json
import logging
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAGFix")

# Paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_REGISTRY = os.path.join(parent_dir, "config", "clients", "client_registry.csv")

def connect_milvus(host="localhost", port="19530"):
    """Connect to Milvus server"""
    try:
        connections.connect("default", host=host, port=port)
        logger.info(f"Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False

def load_clients():
    """Load clients from registry"""
    clients = []
    
    try:
        with open(CLIENT_REGISTRY, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                client_id = row.get('client_id', '')
                active = row.get('active', 'true').lower() == 'true'
                
                if active:
                    clients.append(client_id)
        
        logger.info(f"Loaded {len(clients)} active clients: {clients}")
        return clients
    except Exception as e:
        logger.error(f"Error loading client registry: {e}")
        return []

def inspect_collections():
    """Inspect all collections"""
    collections = utility.list_collections()
    logger.info(f"Found {len(collections)} collections: {collections}")
    
    for coll_name in collections:
        try:
            collection = Collection(coll_name)
            count = collection.num_entities
            
            # Get schema field names
            field_names = [field.name for field in collection.schema.fields]
            
            # Load collection
            try:
                collection.load()
            except Exception as e:
                logger.warning(f"Error loading collection (may already be loaded): {e}")
            
            # Sample one record to check structure
            if count > 0:
                sample = collection.query(expr="", output_fields=field_names, limit=1)
                
                if sample:
                    # Display sample record fields
                    logger.info(f"Sample record fields for {coll_name}:")
                    for field, value in sample[0].items():
                        if isinstance(value, str) and len(value) > 100:
                            logger.info(f"  {field}: {value[:50]}... (truncated)")
                        else:
                            logger.info(f"  {field}: {value}")
            
            logger.info(f"Collection {coll_name}: {count} entities, fields: {field_names}")
            
        except Exception as e:
            logger.error(f"Error inspecting collection {coll_name}: {e}")
    
    return collections

def test_query(client_id, query_text="Show me customer orders", top_k=5):
    """Test RAG query for a client"""
    # Get collection name
    collection_name = f"{client_id}_schema_collection"
    
    if not utility.has_collection(collection_name):
        logger.error(f"Collection {collection_name} not found")
        return None
    
    try:
        # Initialize model
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generate query embedding
        logger.info(f"Generating embedding for query: {query_text}")
        query_embedding = model.encode(query_text)
        
        # Prepare search
        collection = Collection(collection_name)
        
        try:
            collection.load()
        except Exception as e:
            logger.warning(f"Error loading collection (may already be loaded): {e}")
        
        # Search
        logger.info("Executing search...")
        search_params = {
            "metric_type": "IP",  # Inner Product similarity - must match index metric_type
            "params": {"nprobe": 8}  # Fast but accurate for daily searches
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["db_schema", "table_name", "column_name", "data_type", 
                          "description", "distinct_values", "combined_text"]
        )
        
        # Process results
        matches = []
        
        for i, hits in enumerate(results):
            for j, hit in enumerate(hits):
                result = {}
                
                # Extract data based on API version
                if hasattr(hit, 'entity'):
                    # Newer versions with entity attribute
                    entity = hit.entity
                    for field in entity:
                        result[field] = entity[field]
                    result["score"] = hit.score
                else:
                    # Direct access to fields
                    for field in dir(hit):
                        if not field.startswith('_') and not callable(getattr(hit, field)):
                            result[field] = getattr(hit, field)
                
                # Print the raw result for debugging
                logger.info(f"Raw hit {j+1}: {hit}")
                
                # Print what was extracted
                formatted = {
                    'table_name': result.get('table_name', 'Unknown'),
                    'column_name': result.get('column_name', ''),
                    'data_type': result.get('data_type', ''),
                    'description': result.get('description', ''),
                    'db_schema': result.get('db_schema', ''),
                    'score': result.get('score', 0.0)
                }
                matches.append(formatted)
        
        return matches
    except Exception as e:
        logger.error(f"Error testing query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def display_query_results(matches):
    """Display query results with proper formatting"""
    if not matches:
        print("No results found", flush=True)
        return
    
    print(f"\nFound {len(matches)} results:", flush=True)
    
    for i, match in enumerate(matches):
        table = match.get('table_name', 'Unknown')
        column = match.get('column_name', '')
        schema = match.get('db_schema', '')
        data_type = match.get('data_type', '')
        description = match.get('description', '')
        score = match.get('score', 0.0)
        
        score_display = f"{score:.4f}" if isinstance(score, float) else "N/A"
        
        print("\n" + "="*60, flush=True)
        print(f"Match {i+1} (Score: {score_display})", flush=True)
        print("="*60, flush=True)
        
        if schema:
            print(f"Schema: {schema}", flush=True)
        print(f"Table: {table}", flush=True)
        if column:
            print(f"Column: {column}", flush=True)
        if data_type:
            print(f"Data Type: {data_type}", flush=True)
        if description:
            print(f"Description: {description}", flush=True)

def main():
    """Main execution function"""
    # Connect to Milvus
    if not connect_milvus():
        print("Failed to connect to Milvus", flush=True)
        return
    
    # List clients
    print("\nLoading clients...", flush=True)
    clients = load_clients()
    
    if not clients:
        print("No clients found in registry", flush=True)
        return
    
    # Inspect collections
    print("\nInspecting collections...", flush=True)
    collections = inspect_collections()
    
    if not collections:
        print("No collections found", flush=True)
        return
    
    # Test query for first client
    client = clients[0]
    print(f"\nTesting query for client: {client}", flush=True)
    query = "Show me customer orders"
    print(f"Query: {query}", flush=True)
    
    matches = test_query(client, query)
    display_query_results(matches)
    
    # Option to test other clients
    if len(clients) > 1:
        print("\nOther available clients:", flush=True)
        for i, other_client in enumerate(clients[1:], 1):
            print(f"{i}. {other_client}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
