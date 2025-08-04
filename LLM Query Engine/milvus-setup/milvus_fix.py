#!/usr/bin/env python3
"""
Direct Milvus Fix Script - Debug and repair Milvus collections
"""

from pymilvus import connections, Collection, utility
import sys
import json

def main():
    """Execute Milvus inspection and fixes"""
    # Connect to Milvus
    print("\nConnecting to Milvus...", flush=True)
    connections.connect("default", host="localhost", port="19530")
    print("Connected!", flush=True)
    
    # List collections
    collections = utility.list_collections()
    print(f"\nFound collections: {collections}", flush=True)
    
    # Check if collections exist
    if not collections:
        print("No collections found!", flush=True)
        return
    
    # Get collection stats
    for coll_name in collections:
        print(f"\nExamining collection: {coll_name}", flush=True)
        collection = Collection(coll_name)
        count = collection.num_entities
        print(f"Number of entities: {count}", flush=True)
        
        # Get schema
        schema_fields = [f.name for f in collection.schema.fields]
        print(f"Schema fields: {schema_fields}", flush=True)
        
        # Sample records - check what's actually stored
        try:
            collection.load()
            print("Collection loaded", flush=True)
        except Exception as e:
            print(f"Error loading collection (may already be loaded): {e}", flush=True)
        
        # Query data directly
        try:
            print("\nSampling records...", flush=True)
            sample = collection.query(expr="", output_fields=schema_fields, limit=2)
            
            if sample:
                print(f"Got {len(sample)} sample records", flush=True)
                print("\nSample record structure:", flush=True)
                for field in schema_fields:
                    if field in sample[0]:
                        value = sample[0][field]
                        print(f"  {field}: {type(value).__name__}", flush=True)
                        
                        # Print actual value if not too large
                        if isinstance(value, (str, int, float, bool)) and (not isinstance(value, str) or len(value) < 100):
                            print(f"    Value: {value}", flush=True)
                        elif isinstance(value, str):
                            print(f"    Value: {value[:50]}...", flush=True)
            else:
                print("No records found in collection", flush=True)
        except Exception as e:
            print(f"Error querying collection: {e}", flush=True)
    
    # Check collection correctness
    print("\nVerifying schema correctness...", flush=True)
    required_fields = [
        'table_name', 'column_name', 'data_type', 'description', 
        'db_schema', 'combined_text', 'embedding'
    ]
    
    for coll_name in collections:
        collection = Collection(coll_name)
        schema_fields = [f.name for f in collection.schema.fields]
        missing = [f for f in required_fields if f not in schema_fields]
        
        if missing:
            print(f"Collection {coll_name} is missing fields: {missing}", flush=True)
        else:
            print(f"Collection {coll_name} has all required fields", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
