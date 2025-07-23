#!/usr/bin/env python3
"""
Test script for the multi-collection retriever functionality.
"""

import os
import sys
import logging

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.multi_collection_retriever import MultiCollectionRetriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multi_collection_retriever():
    """Test the multi-collection retriever."""
    
    print("=== Testing Multi-Collection Retriever ===\n")
    
    # Initialize the retriever
    vector_store_path = os.path.join(os.getcwd(), "vector_stores")
    retriever = MultiCollectionRetriever(
        vector_store_path=vector_store_path,
        store_type="faiss"
    )
    
    print(f"Vector store path: {vector_store_path}")
    print(f"Store type: faiss")
    
    # Check available collections
    collections = retriever.get_available_collections()
    print(f"\nAvailable collections: {collections}")
    
    # Test collection existence
    test_collections = ["mts_data_dictionary", "default", "test_collection"]
    
    for collection in test_collections:
        exists = retriever.collection_exists(collection)
        print(f"Collection '{collection}' exists: {exists}")
    
    # Test retrieval from existing collections
    if collections:
        print(f"\n=== Testing Retrieval ===")
        for collection in collections[:2]:  # Test first 2 collections
            print(f"\nTesting retrieval from collection: {collection}")
            
            try:
                docs = retriever.retrieve(
                    query="test query",
                    collection_name=collection,
                    k=3
                )
                print(f"Retrieved {len(docs)} documents from {collection}")
                
                if docs:
                    print("Sample document keys:", list(docs[0].keys()))
                
            except Exception as e:
                print(f"Error retrieving from {collection}: {e}")
    else:
        print("\nNo collections found - cannot test retrieval")
    
    # Test retrieval from non-existent collection
    print(f"\n=== Testing Non-existent Collection ===")
    docs = retriever.retrieve(
        query="test query",
        collection_name="nonexistent_collection",
        k=3
    )
    print(f"Retrieved {len(docs)} documents from non-existent collection (should be 0)")

if __name__ == "__main__":
    test_multi_collection_retriever()
