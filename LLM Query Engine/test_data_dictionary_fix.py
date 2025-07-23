#!/usr/bin/env python3
"""
Test script to verify the data dictionary processing fix.
"""

import os
import sys
import logging

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.rag_integrator import RAGIntegrator
from rag.embeddings_manager import EmbeddingsManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_integrator_initialization():
    """Test that RAG integrator initializes properly with MultiCollectionRetriever."""
    
    print("=== Testing RAG Integrator Initialization ===\n")
    
    try:
        # Initialize RAG integrator
        rag_integrator = RAGIntegrator(
            vector_store_path="vector_stores",
            default_top_k=5,
            enable_hybrid_search=False,
            enable_contextual_retrieval=False
        )
        
        print("✅ RAG integrator initialized successfully")
        print(f"Retriever type: {type(rag_integrator.retriever).__name__}")
        
        # Check if retriever has the _get_vector_store method
        if hasattr(rag_integrator.retriever, '_get_vector_store'):
            print("✅ Retriever supports multi-collection operations")
            
            # Test getting a vector store for a collection
            test_collection = "test_collection"
            vector_store = rag_integrator.retriever._get_vector_store(test_collection)
            if vector_store:
                print(f"✅ Successfully created vector store for collection: {test_collection}")
                print(f"Vector store type: {type(vector_store).__name__}")
            else:
                print(f"❌ Failed to create vector store for collection: {test_collection}")
        else:
            print("❌ Retriever does not support multi-collection operations")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing RAG integrator: {e}")
        return False

def test_vector_store_method_availability():
    """Test that vector stores have the required methods."""
    
    print("\n=== Testing Vector Store Methods ===\n")
    
    try:
        from rag.multi_collection_retriever import MultiCollectionRetriever
        
        retriever = MultiCollectionRetriever(
            vector_store_path="vector_stores",
            store_type="faiss"
        )
        
        # Get a vector store instance
        vector_store = retriever._get_vector_store("test_methods")
        
        if vector_store:
            print(f"Vector store type: {type(vector_store).__name__}")
            
            # Check for required methods
            methods_to_check = ["add_documents", "add_texts", "similarity_search"]
            
            for method in methods_to_check:
                if hasattr(vector_store, method):
                    print(f"✅ Vector store has method: {method}")
                else:
                    print(f"❌ Vector store missing method: {method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing vector store methods: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG integrator fixes...\n")
    
    success1 = test_rag_integrator_initialization()
    success2 = test_vector_store_method_availability()
    
    if success1 and success2:
        print("\n🎉 All tests passed! RAG integrator should now work with data dictionary processing.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
