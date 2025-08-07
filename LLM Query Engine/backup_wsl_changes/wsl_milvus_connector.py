#!/usr/bin/env python3
"""
WSL Milvus Connector

This script helps connect to Milvus running in WSL2 from Windows host applications.
It finds the WSL2 IP address and provides connection utilities for the RAG system.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WSLMilvusConnector")

def get_wsl_ip():
    """Get the IP address of the WSL2 Ubuntu distribution"""
    try:
        # Run the command to get WSL IP address
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "ip", "addr", "show", "eth0"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output to find the IP address
        for line in result.stdout.splitlines():
            if "inet " in line:
                # Extract the IP address
                ip_with_subnet = line.strip().split()[1]
                ip = ip_with_subnet.split('/')[0]
                logger.info(f"Found WSL2 IP address: {ip}")
                return ip
        
        logger.error("Could not find WSL2 IP address")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting WSL2 IP address: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting WSL2 IP address: {e}")
        return None

def check_milvus_connection(host, port="19530"):
    """Check if Milvus is running at the specified host and port"""
    try:
        # Import here to avoid dependency issues if pymilvus is not installed
        from pymilvus import connections
        
        # Try to connect to Milvus
        connections.connect(
            alias="default", 
            host=host, 
            port=port
        )
        
        # Check if connection is established
        if connections.has_connection("default"):
            logger.info(f"Successfully connected to Milvus at {host}:{port}")
            connections.disconnect("default")
            return True
        else:
            logger.error(f"Failed to connect to Milvus at {host}:{port}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Milvus at {host}:{port}: {e}")
        return False

def update_rag_connection_settings(wsl_ip):
    """
    Update the RAG connection settings to use the WSL2 IP address
    
    This function creates a small wrapper module that can be imported
    to override the default Milvus connection settings
    """
    # Create the wsl_connection.py file
    connection_file = Path(__file__).parent / "wsl_connection.py"
    
    with open(connection_file, "w") as f:
        f.write(f"""#!/usr/bin/env python3
\"\"\"
WSL Milvus Connection Settings

This module provides connection settings for Milvus running in WSL2.
Import this module before initializing the RAG manager to use WSL2 Milvus.
\"\"\"

import os
import sys
import importlib.util

# WSL2 Milvus connection settings
WSL_MILVUS_HOST = "{wsl_ip}"
WSL_MILVUS_PORT = "19530"

def get_wsl_milvus_connection():
    \"\"\"Get the WSL2 Milvus connection settings\"\"\"
    return WSL_MILVUS_HOST, WSL_MILVUS_PORT

def patch_rag_manager():
    \"\"\"
    Monkey patch the RAGManager class to use WSL2 Milvus connection
    \"\"\"
    try:
        # Get the path to the fixed_rag.py file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        rag_file_path = os.path.join(current_dir, 'fixed_rag.py')
        
        # Import the RAG manager module directly from file
        spec = importlib.util.spec_from_file_location('fixed_rag', rag_file_path)
        fixed_rag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixed_rag)
        
        # Get the RAGManager class
        RAGManager = fixed_rag.RAGManager
        
        # Save the original __init__ method
        original_init = RAGManager.__init__
        
        # Define the new __init__ method with local variables to avoid NameError
        wsl_host = WSL_MILVUS_HOST
        wsl_port = WSL_MILVUS_PORT
        
        def new_init(self, milvus_host=wsl_host, milvus_port=wsl_port, *args, **kwargs):
            print(f"Using WSL2 Milvus connection: {wsl_host}:{wsl_port}")
            return original_init(self, milvus_host=wsl_host, milvus_port=wsl_port, *args, **kwargs)
        
        # Replace the __init__ method
        RAGManager.__init__ = new_init
        
        print(f"Successfully patched RAGManager to use WSL2 Milvus at {wsl_host}:{wsl_port}")
        return True
    except Exception as e:
        print(f"Error patching RAGManager: {e}")
        return False""")
    
    logger.info(f"Created WSL connection settings file: {connection_file}")
    return connection_file

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="WSL Milvus Connector")
    parser.add_argument("--check", action="store_true", help="Check Milvus connection")
    parser.add_argument("--update", action="store_true", help="Update RAG connection settings")
    args = parser.parse_args()
    
    # Get the WSL2 IP address
    wsl_ip = get_wsl_ip()
    if not wsl_ip:
        print("Could not find WSL2 IP address. Make sure WSL2 is running.")
        return False
    
    # Check Milvus connection if requested
    if args.check:
        if check_milvus_connection(wsl_ip):
            print(f"✅ Milvus is running at {wsl_ip}:19530")
        else:
            print(f"❌ Milvus is not running at {wsl_ip}:19530")
            print("Make sure Docker and Milvus containers are running in WSL2.")
            print("Run the start_docker_wsl.bat script to start Docker and Milvus.")
    
    # Update RAG connection settings if requested
    if args.update:
        connection_file = update_rag_connection_settings(wsl_ip)
        print(f"✅ Created WSL connection settings file: {connection_file}")
        print("\nTo use WSL2 Milvus in your code, add these lines at the beginning:")
        print("from milvus_setup.wsl_connection import patch_rag_manager")
        print("patch_rag_manager()")
    
    # If no arguments provided, just print the WSL2 IP
    if not args.check and not args.update:
        print(f"WSL2 IP address: {wsl_ip}")
        print("\nUse --check to check Milvus connection")
        print("Use --update to update RAG connection settings")
    
    return True

if __name__ == "__main__":
    main()
