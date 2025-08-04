#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility module to manage Milvus containers from within application code
"""

import os
import subprocess
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

def start_milvus_containers(wait_time=10):
    """
    Start Milvus Docker containers and wait for them to be ready
    
    Args:
        wait_time (int): Seconds to wait for containers to initialize
        
    Returns:
        bool: True if containers started successfully, False otherwise
    """
    logger.info("Starting Milvus Docker containers...")
    
    # Get the milvus-setup directory (where docker-compose.yml is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # First check if containers are already running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if "Up" in result.stdout:
            logger.info("Milvus container is already running.")
            return True
        
        # Check if containers exist but are stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        # If containers exist but are stopped, start them directly
        if result.stdout and "Exited" in result.stdout:
            logger.info("Found stopped Milvus containers. Starting them...")
            for container in ["milvus-standalone", "milvus-etcd", "milvus-minio"]:
                try:
                    subprocess.run(["docker", "start", container], check=True)
                    logger.info(f"Started container: {container}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error starting container {container}: {e}")
        else:
            # Run docker-compose up -d to create and start containers in background
            logger.info("Starting containers with docker-compose...")
            subprocess.run(
                ["docker-compose", "up", "-d"], 
                cwd=current_dir,
                check=True
            )
        
        # Wait for containers to be ready
        logger.info(f"Waiting {wait_time} seconds for containers to be ready...")
        time.sleep(wait_time)
        
        # Check if milvus container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "Up" in result.stdout:
            logger.info("Milvus container is now running.")
            return True
        else:
            logger.error("Milvus container failed to start properly.")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting Milvus containers: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error starting Milvus containers: {str(e)}")
        return False

def check_milvus_status():
    """
    Check if Milvus containers are running
    
    Returns:
        dict: Status information for each container
    """
    container_status = {}
    
    try:
        for container in ["milvus-standalone", "milvus-etcd", "milvus-minio"]:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={container}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True
            )
            container_status[container] = "Running" if "Up" in result.stdout else "Stopped"
        
        return container_status
    
    except Exception as e:
        logger.error(f"Error checking Milvus container status: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Configure logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Start containers
    if start_milvus_containers():
        logger.info("Milvus containers started successfully")
    else:
        logger.error("Failed to start Milvus containers")
    
    # Show status
    status = check_milvus_status()
    for container, state in status.items():
        logger.info(f"{container}: {state}")
