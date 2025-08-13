"""
Path Resolver for Client Manager
- Provides consistent path resolution between Docker and local environments
- Used by client_manager.py to resolve client data dictionary paths
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientPathResolver:
    """Utility class to resolve client paths consistently between Docker and local environments"""
    
    def __init__(self):
        """Initialize the path resolver"""
        self.is_docker = self._check_if_docker()
        self.logger = logging.getLogger("ClientPathResolver")
        
        if self.is_docker:
            self.logger.info("Running in Docker environment")
            self.base_dir = "/app"
        else:
            self.logger.info("Running in local environment")
            # Get the parent directory of the current file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_dir = os.path.dirname(current_dir)
            
        self.logger.info(f"Base directory: {self.base_dir}")
    
    def _check_if_docker(self):
        """Check if running in Docker container"""
        return os.path.exists('/.dockerenv') or os.path.exists('/app')
    
    def resolve_path(self, relative_path):
        """
        Resolve a path consistently between Docker and local environments
        
        Args:
            relative_path: Path relative to the LLM Query Engine directory
            
        Returns:
            Absolute path that works in the current environment
        """
        # If running in Docker and path contains Windows-style drive letter (C:/, etc.),
        # we need to strip it and use only the relative part
        if self.is_docker and (relative_path.startswith('/') or ':/' in relative_path or ':\\' in relative_path):
            # Handle Windows absolute paths in Docker
            if ':/' in relative_path or ':\\' in relative_path:
                # Extract the part after the drive letter and any leading slashes
                parts = relative_path.replace('\\', '/').split('/')
                # Find the index of 'LLM Query Engine' or 'config' in the path
                try:
                    if 'LLM Query Engine' in parts:
                        idx = parts.index('LLM Query Engine')
                    elif 'config' in parts:
                        idx = parts.index('config')
                    else:
                        # If neither is found, just use the filename
                        idx = len(parts) - 1
                    
                    # Join the parts starting from the identified index
                    clean_path = '/'.join(parts[idx:])
                    self.logger.info(f"Converted Windows path '{relative_path}' to '{clean_path}' in Docker")
                    return os.path.join(self.base_dir, clean_path)
                except (ValueError, IndexError):
                    # Fallback to just using the filename
                    filename = os.path.basename(relative_path)
                    self.logger.warning(f"Could not parse Windows path '{relative_path}', using filename '{filename}'")
                    return os.path.join(self.base_dir, filename)
            
        # If the path is already a proper absolute path for the current environment, return it
        if os.path.isabs(relative_path) and not (self.is_docker and (':/' in relative_path or ':\\' in relative_path)):
            return relative_path
            
        # Otherwise, join it with the base directory
        return os.path.join(self.base_dir, relative_path)

# Create a singleton instance
client_path_resolver = ClientPathResolver()

# Export the resolve_path function for easy access
def resolve_path(relative_path):
    """Resolve a path consistently between Docker and local environments"""
    return client_path_resolver.resolve_path(relative_path)
