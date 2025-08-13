#!/usr/bin/env python3
"""
Fix for the embedding stats endpoint in Docker environment

This script updates the embedding_api.py file to use environment variables
for Milvus connection in the stats endpoints instead of hardcoded localhost.
"""

import os
import re
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fix_embedding_stats")

def fix_embedding_stats():
    """Fix the embedding stats endpoint to use environment variables for Milvus connection"""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_api_path = os.path.join(current_dir, "embedding_api.py")
    
    logger.info(f"Fixing embedding stats endpoint in {embedding_api_path}")
    
    # Read the file
    with open(embedding_api_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace hardcoded localhost connections with environment variables
    # Pattern 1: In get_stats function
    pattern1 = r'connections\.connect\("default", host="localhost", port="19530"\)'
    replacement1 = 'connections.connect("default", host=os.environ.get("MILVUS_HOST", "localhost"), port=os.environ.get("MILVUS_PORT", "19530"))'
    
    # Pattern 2: In _get_stats function
    pattern2 = r'connections\.connect\("default", host="localhost", port="19530"\)'
    replacement2 = 'connections.connect("default", host=os.environ.get("MILVUS_HOST", "localhost"), port=os.environ.get("MILVUS_PORT", "19530"))'
    
    # Apply replacements
    modified_content = re.sub(pattern1, replacement1, content)
    modified_content = re.sub(pattern2, replacement2, modified_content)
    
    # Write the modified content back to the file
    with open(embedding_api_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    logger.info("Successfully updated embedding_api.py to use environment variables for Milvus connection")
    logger.info("Now the stats endpoint should work in Docker environment")

if __name__ == "__main__":
    fix_embedding_stats()
