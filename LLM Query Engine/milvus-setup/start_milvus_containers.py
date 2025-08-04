#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility script to start Milvus containers for LLM Query Engine
"""

import os
import subprocess
import time
import sys

def start_containers():
    """Start Milvus, etcd, and minio containers"""
    print("Starting Milvus Docker containers...")
    
    # Get the current directory (where docker-compose.yml is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Run docker-compose up -d to start containers in background
        subprocess.run(
            ["docker-compose", "up", "-d"], 
            cwd=current_dir,
            check=True
        )
        print("Docker containers started successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error starting containers: {e}")
        return False
    
    # Wait for containers to be ready
    print("Waiting for containers to be ready...")
    time.sleep(5)  # Give containers a moment to initialize
    
    # Check if milvus container is running
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "Up" in result.stdout:
            print("Milvus container is running.")
            return True
        else:
            print("Milvus container is not running. Check docker logs.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error checking container status: {e}")
        return False

def check_status():
    """Check if the containers are running"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=milvus", "--format", "{{.Names}} - {{.Status}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            print("Milvus-related containers status:")
            print(result.stdout)
        else:
            print("No Milvus containers found running.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error checking status: {e}")

if __name__ == "__main__":
    success = start_containers()
    
    if success:
        print("\nAll Milvus containers started successfully.")
        print("You can now run your RAG API.")
        check_status()
    else:
        print("\nFailed to start containers. Check the error messages above.")
