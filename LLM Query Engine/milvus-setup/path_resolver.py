#!/usr/bin/env python3
"""
Path Resolver for Conversational Query Engine

This module provides consistent path resolution between Docker and local environments.
It ensures that file paths work correctly regardless of whether the application is
running in Docker or locally.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PathResolver")

class PathResolver:
    """Resolves paths consistently between Docker and local environments"""
    
    def __init__(self):
        """Initialize the path resolver"""
        # Determine the base directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # In Docker, the app is in /app/LLM Query Engine
        # Locally, it's in LLM Query Engine/
        if os.path.exists("/.dockerenv") or os.path.exists("/app"):
            # Docker environment
            self.is_docker = True
            self.base_dir = "/app/LLM Query Engine"
            logger.info("Running in Docker environment")
        else:
            # Local environment
            self.is_docker = False
            # Go up one level from milvus-setup to LLM Query Engine
            self.base_dir = os.path.dirname(self.script_dir)
            logger.info(f"Running in local environment: {self.base_dir}")
    
    def _check_if_docker(self):
        """Check if running in Docker container"""
        return os.path.exists('/.dockerenv') or os.path.exists('/app')
        
    def get_client_registry_path(self):
        """Get the path to the client registry file"""
        return os.path.join(self.base_dir, "config", "clients", "client_registry.csv")
        
    def get_column_mappings_path(self):
        """Get the path to the column mappings file"""
        return os.path.join(self.base_dir, "config", "column_mappings.json")
    
    def get_client_dictionary_path(self, client_name):
        """Get the path to a client's dictionary CSV file"""
        return os.path.join(
            self.base_dir, 
            "config", 
            "clients", 
            "data_dictionaries", 
            client_name, 
            f"{client_name}_dictionary.csv"
        )
    
    def get_env_file_path(self):
        """Get the path to the .env file"""
        return os.path.join(self.base_dir, ".env")
    
    def resolve_path(self, relative_path):
        """Resolve a path relative to the base directory"""
        return os.path.join(self.base_dir, relative_path)

# Create a singleton instance
path_resolver = PathResolver()

# This function must be available at the module level
def get_path_resolver():
    """Get the singleton PathResolver instance"""
    return path_resolver

# Add direct functions at the module level to match what's expected by other modules
def get_client_dictionary_path(client_name):
    """Get the path to a client's dictionary CSV file (module level function)"""
    resolver = get_path_resolver()
    return resolver.get_client_dictionary_path(client_name)

def get_client_registry_path():
    """Get the path to the client registry file (module level function)"""
    resolver = get_path_resolver()
    return resolver.get_client_registry_path()

def get_column_mappings_path():
    """Get the path to the column mappings file (module level function)"""
    resolver = get_path_resolver()
    return resolver.get_column_mappings_path()

# Make the function available at module level
__all__ = ['get_path_resolver', 'PathResolver', 'get_client_dictionary_path', 'get_client_registry_path', 'get_column_mappings_path']

# For testing
if __name__ == "__main__":
    resolver = get_path_resolver()
    print(f"Base directory: {resolver.base_dir}")
    print(f"Client registry: {resolver.get_client_registry_path()}")
    print(f"Column mappings: {resolver.get_column_mappings_path()}")
    print(f"MTS dictionary: {resolver.get_client_dictionary_path('mts')}")
    print(f"Environment file: {resolver.get_env_file_path()}")
