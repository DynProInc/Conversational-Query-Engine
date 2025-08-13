#!/usr/bin/env python3
"""
Test script to verify path resolution and Milvus connection in Docker
"""

import os
import sys
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DockerConnectionTest")

def main():
    """Main test function"""
    # Test environment detection
    in_docker = os.path.exists('/.dockerenv') or os.path.exists('/app')
    logger.info(f"Running in Docker: {in_docker}")
    
    # Test current directory
    current_dir = os.getcwd()
    logger.info(f"Current directory: {current_dir}")
    
    # Test environment variables
    milvus_host = os.environ.get("MILVUS_HOST", "not set")
    milvus_port = os.environ.get("MILVUS_PORT", "not set")
    logger.info(f"MILVUS_HOST: {milvus_host}")
    logger.info(f"MILVUS_PORT: {milvus_port}")
    
    # Test path resolution
    try:
        # Add the parent directory to the Python path
        parent_dir = os.path.dirname(os.getcwd())
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Try different import paths
        try:
            from milvus_setup.path_resolver import get_client_dictionary_path
            logger.info("Successfully imported path_resolver from milvus_setup")
        except ImportError:
            # Try relative import
            try:
                from .milvus_setup.path_resolver import get_client_dictionary_path
                logger.info("Successfully imported path_resolver with relative import")
            except ImportError:
                # Try direct import
                import sys
                sys.path.append(os.path.join(os.getcwd(), 'milvus-setup'))
                from path_resolver import get_client_dictionary_path
                logger.info("Successfully imported path_resolver directly")
        
        # Test client dictionary path resolution
        client_id = "mts"
        dict_path = get_client_dictionary_path(client_id)
        logger.info(f"Dictionary path for {client_id}: {dict_path}")
        logger.info(f"Dictionary exists: {os.path.exists(dict_path)}")
    except Exception as e:
        logger.error(f"Error importing path_resolver: {e}")
        
        # Try using the client_manager instead
        try:
            from config.client_manager import ClientManager
            client_manager = ClientManager()
            dict_path = client_manager.get_data_dictionary_path("mts")
            logger.info(f"Dictionary path from client_manager: {dict_path}")
            logger.info(f"Dictionary exists: {os.path.exists(dict_path)}")
        except Exception as e2:
            logger.error(f"Error using client_manager: {e2}")
    
    # Test Milvus connection
    try:
        from pymilvus import connections, utility
        
        # Connect to Milvus
        connections.connect(alias="default", host=milvus_host, port=milvus_port)
        logger.info(f"Successfully connected to Milvus at {milvus_host}:{milvus_port}")
        
        # Check if Milvus is healthy
        if utility.has_collection("test_collection"):
            logger.info("Found test_collection in Milvus")
        else:
            logger.info("No test_collection found in Milvus")
            
        # List all collections
        collections = utility.list_collections()
        logger.info(f"Collections in Milvus: {collections}")
        
    except Exception as e:
        logger.error(f"Error connecting to Milvus: {e}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
