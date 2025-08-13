#!/usr/bin/env python3
"""
Test script to verify the client dictionary API endpoint
"""

import os
import sys
import logging
import requests
import json
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ClientDictionaryAPITest")

def main():
    """Test the client dictionary API endpoint"""
    # Test environment detection
    in_docker = os.path.exists('/.dockerenv') or os.path.exists('/app')
    logger.info(f"Running in Docker: {in_docker}")
    
    # Test current directory
    current_dir = os.getcwd()
    logger.info(f"Current directory: {current_dir}")
    
    # Test API endpoint
    client_id = "mts"
    api_url = "http://localhost:8002/client/dictionary/mts"
    
    try:
        logger.info(f"Testing API endpoint: {api_url}")
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            logger.info("API request successful!")
            # Get the first few rows of the dictionary
            data = response.json()
            logger.info(f"Dictionary has {len(data)} rows")
            logger.info(f"First 3 rows: {json.dumps(data[:3], indent=2)}")
        else:
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Error making API request: {e}")
    
    # Test direct file access
    try:
        # Get dictionary path
        if in_docker:
            # In Docker, use the fixed path
            dict_path = f"/app/LLM Query Engine/config/clients/data_dictionaries/{client_id}/{client_id}_dictionary.csv"
        else:
            # In local environment, try to use client_manager
            try:
                # Import here to avoid issues if not available
                sys.path.append(os.path.dirname(current_dir))
                from config.client_manager import ClientManager
                client_manager = ClientManager()
                dict_path = client_manager.get_data_dictionary_path(client_id)
            except Exception as e:
                logger.error(f"Error importing client_manager: {e}")
                # Fallback to a default path
                dict_path = os.path.join(current_dir, "config", "clients", "data_dictionaries", 
                                        client_id, f"{client_id}_dictionary.csv")
        
        logger.info(f"Dictionary path: {dict_path}")
        logger.info(f"Dictionary file exists: {os.path.exists(dict_path)}")
        
        # Try multiple encodings to read the file
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        success = False
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"Trying to read file with encoding: {encoding}")
                # Try pandas read_csv first
                df = pd.read_csv(dict_path, encoding=encoding)
                logger.info(f"Successfully read dictionary with encoding {encoding}")
                logger.info(f"Dictionary has {len(df)} rows and {len(df.columns)} columns")
                if len(df) > 0:
                    logger.info(f"Column names: {', '.join(df.columns.tolist())}")
                success = True
                break
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to read with encoding {encoding}: {str(e)}")
            except Exception as e:
                logger.error(f"Error reading dictionary file: {str(e)}")
                break
        
        if not success:
            logger.error("Failed to read dictionary file with any encoding")
    except Exception as e:
        logger.error(f"Error accessing dictionary: {str(e)}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()
