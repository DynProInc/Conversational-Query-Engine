#!/usr/bin/env python3
"""
Test script to check vector store contents and debug RAG issues.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from services.rag_client import RAGService
from rag.multi_collection_retriever import MultiCollectionRetriever

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vector_store_contents():
    """Test vector store contents for debugging."""
    try:
        # Initialize RAG service
        logger.info("Initializing RAG service...")
        rag_service = RAGService()
        
        # Test collection name
        collection_name = "mts_data_dictionary"
        
        # Check if collection exists
        logger.info(f"Checking collection: {collection_name}")
        
        # Try to retrieve some context
        query = "sales people region"
        logger.info(f"Testing retrieval with query: '{query}'")
        
        try:
            context = rag_service.retrieve_context(
                query=query,
                collection_name=collection_name,
                top_k=5,
                filters={"client_id": "mts"}
            )
            
            logger.info(f"Retrieved context: {context}")
            
            if context and len(context) > 0:
                logger.info(f"✅ Successfully retrieved {len(context)} documents")
                for i, doc in enumerate(context[:2]):  # Show first 2 docs
                    logger.info(f"Document {i+1}: {doc.get('text', '')[:200]}...")
            else:
                logger.warning("⚠️ No documents retrieved from vector store")
                
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
        
        # Check vector store files
        vector_store_dir = Path("vector_stores")
        if vector_store_dir.exists():
            logger.info(f"Vector store directory contents:")
            for file in vector_store_dir.iterdir():
                if file.is_file():
                    logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
        else:
            logger.warning("Vector store directory doesn't exist")
            
        # Try to access the retriever directly
        if hasattr(rag_service, 'rag_integrator') and hasattr(rag_service.rag_integrator, 'retriever'):
            retriever = rag_service.rag_integrator.retriever
            if isinstance(retriever, MultiCollectionRetriever):
                logger.info("Testing direct retriever access...")
                try:
                    # Get vector store for collection
                    vector_store = retriever._get_vector_store(collection_name)
                    if vector_store:
                        logger.info(f"✅ Vector store created for collection: {collection_name}")
                        
                        # Try to get some stats if available
                        if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'ntotal'):
                            logger.info(f"FAISS index contains {vector_store.index.ntotal} vectors")
                        
                    else:
                        logger.warning(f"⚠️ Could not create vector store for collection: {collection_name}")
                        
                except Exception as e:
                    logger.error(f"Error accessing vector store directly: {e}")
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_store_contents()
