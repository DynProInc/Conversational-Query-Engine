#!/usr/bin/env python3
"""
Milvus Status Checker
====================

Quick utility to check the status of Milvus collections and 
verify the dynamic column mapping implementation.

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import logging
from pymilvus import connections, utility, Collection
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MilvusCheck")

def connect_milvus(host="localhost", port="19530"):
    """Connect to Milvus server"""
    try:
        connections.connect(
            alias="default", 
            host=host, 
            port=port
        )
        logger.info(f"Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False

def list_collections():
    """List all collections in Milvus"""
    try:
        collections = utility.list_collections()
        logger.info(f"Found {len(collections)} collections in Milvus")
        
        for idx, coll in enumerate(collections):
            logger.info(f"  {idx+1}. {coll}")
        
        return collections
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []

def collection_stats(collection_name):
    """Get stats for a collection"""
    try:
        if not utility.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist")
            return {}
        
        collection = Collection(collection_name)
        
        # Load collection to ensure stats are available
        if not collection.is_loaded:
            logger.info(f"Loading collection {collection_name}")
            collection.load()
        
        # Get entity count
        count = collection.num_entities
        logger.info(f"Collection {collection_name} has {count} entities")
        
        # Check if has index
        has_index = collection.has_index()
        logger.info(f"Collection {collection_name} has index: {has_index}")
        
        # Query a sample if available
        if count > 0:
            try:
                logger.info(f"Retrieving first record from {collection_name}")
                collection.create_index("embedding", {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 64}}, index_name="embedding_index")  # Optimized for small dataset (~369 rows)  # Changed from COSINE to IP
                collection.load()
                results = collection.query("id in [0]", output_fields=["db_schema", "table_name", "column_name", "data_type", "description"])
                if results:
                    logger.info(f"Sample record: {results[0]}")
            except Exception as e:
                logger.error(f"Error retrieving sample: {e}")
        
        return {
            "name": collection_name,
            "count": count,
            "has_index": has_index
        }
    except Exception as e:
        logger.error(f"Error checking collection {collection_name}: {e}")
        return {}

def check_milvus_status():
    """Check overall Milvus status"""
    # Connect to Milvus
    if not connect_milvus():
        return False
    
    # List collections
    collections = list_collections()
    if not collections:
        return False
    
    # Check each collection
    for collection_name in collections:
        collection_stats(collection_name)
    
    return True

if __name__ == "__main__":
    check_milvus_status()
