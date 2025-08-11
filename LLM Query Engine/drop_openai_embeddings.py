"""
Script to drop OpenAI embedding collections from Milvus.
"""

import os
import logging
from pymilvus import connections, utility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_milvus(host="localhost", port="19530"):
    """Connect to Milvus server"""
    try:
        connections.connect(alias="default", host=host, port=port)
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
        return collections
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []

def drop_openai_collections():
    """Drop all OpenAI embedding collections"""
    try:
        collections = list_collections()
        openai_collections = [c for c in collections if "openai_embedding_test" in c]
        
        logger.info(f"Found {len(openai_collections)} OpenAI embedding collections: {openai_collections}")
        
        for collection_name in openai_collections:
            try:
                logger.info(f"Dropping collection: {collection_name}")
                utility.drop_collection(collection_name)
                logger.info(f"Successfully dropped collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to drop collection {collection_name}: {e}")
        
        return True, f"Dropped {len(openai_collections)} OpenAI embedding collections"
    except Exception as e:
        logger.error(f"Error dropping OpenAI collections: {e}")
        return False, str(e)

def main():
    """Main function"""
    # Connect to Milvus
    if not connect_to_milvus():
        return
    
    # Drop OpenAI collections
    success, message = drop_openai_collections()
    if success:
        logger.info(message)
    else:
        logger.error(f"Failed: {message}")

if __name__ == "__main__":
    main()
