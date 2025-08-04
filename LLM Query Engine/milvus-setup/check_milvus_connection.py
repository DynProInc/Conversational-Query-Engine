#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick script to check Milvus connection at localhost:19530
"""

import sys
import time
from pymilvus import connections, utility

def check_milvus_connection():
    """Check if Milvus is accessible at localhost:19530"""
    print("Checking Milvus connection at localhost:19530...")
    
    try:
        # Connect to Milvus
        connections.connect(
            alias="default", 
            host="localhost", 
            port="19530"
        )
        
        # Check server version to verify connection
        version = utility.get_server_version()
        
        print(f"✅ Successfully connected to Milvus!")
        print(f"✅ Milvus server version: {version}")
        
        # Check if any collections exist
        collections = utility.list_collections()
        if collections:
            print(f"✅ Found {len(collections)} collections: {', '.join(collections)}")
        else:
            print("ℹ️ No collections found (this is normal if you haven't created any yet)")
        
        # Disconnect
        connections.disconnect("default")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {str(e)}")
        return False

if __name__ == "__main__":
    success = check_milvus_connection()
    sys.exit(0 if success else 1)
