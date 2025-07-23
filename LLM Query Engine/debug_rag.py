"""
Debug script for RAG functionality - checks vector store and collection status
"""

import os
import logging
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_vector_store(vector_store_path="vector_stores"):
    """Check if vector store directory exists and has collections"""
    logger.info(f"Checking vector store at path: {vector_store_path}")
    
    # Check if vector store directory exists
    if not os.path.exists(vector_store_path):
        logger.error(f"❌ Vector store directory does not exist: {vector_store_path}")
        return False
    
    # List contents
    contents = os.listdir(vector_store_path)
    logger.info(f"Vector store directory contents: {contents}")
    
    # Check for collections
    collections = []
    for item in contents:
        if os.path.isdir(os.path.join(vector_store_path, item)) and not item.startswith('.'):
            collections.append(item)
    
    if collections:
        logger.info(f"Found collections: {collections}")
        for collection in collections:
            check_collection(vector_store_path, collection)
        return True
    else:
        logger.warning("⚠️ No collections found in vector store")
        return False

def check_collection(vector_store_path, collection_name):
    """Check a specific collection in the vector store"""
    collection_path = os.path.join(vector_store_path, collection_name)
    logger.info(f"Checking collection: {collection_name}")
    
    # Check collection directory contents
    if os.path.isdir(collection_path):
        contents = os.listdir(collection_path)
        logger.info(f"Collection contents: {contents}")
        
        # Check for FAISS index
        if "index.faiss" in contents:
            logger.info("✅ FAISS index found")
        else:
            logger.warning("⚠️ No FAISS index found")
        
        # Check for document store
        if "documents.json" in contents:
            try:
                with open(os.path.join(collection_path, "documents.json"), "r") as f:
                    docs = json.load(f)
                logger.info(f"✅ Document store found with {len(docs)} documents")
                
                # Show sample documents
                if docs:
                    logger.info(f"Sample document: {docs[0]}")
            except Exception as e:
                logger.error(f"❌ Error reading documents.json: {e}")
        else:
            logger.warning("⚠️ No documents.json found")
        
        # Check for metadata
        if "metadata.json" in contents:
            try:
                with open(os.path.join(collection_path, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                logger.info(f"✅ Metadata found: {metadata}")
            except Exception as e:
                logger.error(f"❌ Error reading metadata.json: {e}")
        else:
            logger.warning("⚠️ No metadata.json found")
    else:
        logger.error(f"❌ Collection path is not a directory: {collection_path}")

def check_rag_stats(vector_store_path="vector_stores"):
    """Check RAG statistics file"""
    stats_path = os.path.join(vector_store_path, "rag_stats.json")
    
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            logger.info(f"✅ RAG stats found: {stats}")
            return stats
        except Exception as e:
            logger.error(f"❌ Error reading RAG stats: {e}")
            return None
    else:
        logger.warning(f"⚠️ No RAG stats file found at: {stats_path}")
        return None

def check_data_dictionaries():
    """Check data dictionary files"""
    base_path = os.path.join("config", "clients", "data_dictionaries")
    
    if not os.path.exists(base_path):
        logger.error(f"❌ Data dictionaries directory does not exist: {base_path}")
        return
    
    # List client directories
    clients = []
    for item in os.listdir(base_path):
        client_path = os.path.join(base_path, item)
        if os.path.isdir(client_path):
            clients.append(item)
    
    if not clients:
        logger.warning("⚠️ No client directories found in data dictionaries")
        return
    
    logger.info(f"Found client directories: {clients}")
    
    # Check each client directory for dictionary files
    for client in clients:
        client_path = os.path.join(base_path, client)
        files = os.listdir(client_path)
        csv_files = [f for f in files if f.endswith(".csv")]
        
        if csv_files:
            logger.info(f"✅ Client {client} has dictionary files: {csv_files}")
            
            # Check if we have a matching vector store collection
            collection_name = f"{client}_data_dictionary"
            vector_store_path = "vector_stores"
            collection_path = os.path.join(vector_store_path, collection_name)
            
            if os.path.exists(collection_path):
                logger.info(f"✅ Matching vector store collection exists for client {client}")
            else:
                logger.warning(f"⚠️ No matching vector store collection for client {client}")
        else:
            logger.warning(f"⚠️ No dictionary files found for client {client}")

def main():
    """Main function"""
    logger.info("Starting RAG debugging")
    
    # Check vector store
    check_vector_store()
    
    # Check RAG stats
    check_rag_stats()
    
    # Check data dictionaries
    check_data_dictionaries()
    
    logger.info("RAG debugging complete")

if __name__ == "__main__":
    main()
