"""
Client Manager Module
- Handles client configuration, connections and data dictionaries
- Provides dynamic client context without changing existing API structure
- Supports secure credential storage via .env files
"""

import os
import csv
import pandas as pd
from typing import Dict, Optional, Any, List
from pathlib import Path
import dotenv

# Base directory for client configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CLIENT_REGISTRY_PATH = BASE_DIR / "clients" / "client_registry.csv"
CLIENT_ENV_DIR = BASE_DIR / "clients" / "env"
LLM_API_KEYS_PATH = BASE_DIR / "clients" / "llm_api_keys.csv"

# Ensure env directory exists
os.makedirs(CLIENT_ENV_DIR, exist_ok=True)

class ClientManager:
    """
    Client Manager for handling multiple client configurations
    """
    def __init__(self):
        """Initialize the client manager and load configurations"""
        self.clients = {}
        self.llm_configs = {}
        self._load_client_registry()
        # We no longer load LLM keys directly, they're loaded from env files on demand
        
    def _load_client_registry(self):
        """Load the client registry from CSV"""
        if not CLIENT_REGISTRY_PATH.exists():
            raise FileNotFoundError(f"Client registry not found at {CLIENT_REGISTRY_PATH}")
        
        with open(CLIENT_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                client_id = row['client_id']
                
                # We only store non-sensitive information in the registry
                self.clients[client_id] = {
                    'name': row['client_name'],
                    'description': row['description'],
                    'active': row['active'].lower() == 'true',
                    'data_dictionary_path': row['data_dictionary_path']
                }
    
    def _get_client_env_path(self, client_id: str) -> Path:
        """Get path to client environment file"""
        return CLIENT_ENV_DIR / f"{client_id}.env"
        
    def _load_client_env(self, client_id: str) -> Dict[str, str]:
        """Load client environment variables from .env file"""
        env_path = self._get_client_env_path(client_id)
        
        if not env_path.exists():
            print(f"Warning: Environment file for client '{client_id}' not found at {env_path}")
            return {}
            
        # Load environment variables from file
        env_vars = {}
        dotenv.load_dotenv(env_path)
        
        # Create a prefix for this client's environment variables
        prefix = f"CLIENT_{client_id.upper()}_"
        
        # Extract all environment variables for this client
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix to get the actual variable name
                clean_key = key[len(prefix):]
                env_vars[clean_key] = value
                
        return env_vars
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        Get client information by client ID
        
        Args:
            client_id: The client identifier
            
        Returns:
            Dictionary with client information or None if not found
        """
        return self.clients.get(client_id)
    
    def get_snowflake_connection_params(self, client_id: str) -> Dict[str, str]:
        """
        Get Snowflake connection parameters for a client
        
        Args:
            client_id: The client identifier
            
        Returns:
            Dictionary with connection parameters from environment file
        """
        if client_id not in self.clients:
            raise ValueError(f"Client ID '{client_id}' not found")
        
        # Load client environment variables
        env_vars = self._load_client_env(client_id)
        
        # Try to get Snowflake credentials from environment
        snowflake = {
            'account': env_vars.get('SNOWFLAKE_ACCOUNT', ''),
            'user': env_vars.get('SNOWFLAKE_USER', ''),
            'password': env_vars.get('SNOWFLAKE_PASSWORD', ''),
            'warehouse': env_vars.get('SNOWFLAKE_WAREHOUSE', ''),
            'database': env_vars.get('SNOWFLAKE_DATABASE', ''),
            'schema': env_vars.get('SNOWFLAKE_SCHEMA', '')
        }
        
        # Validate required parameters
        missing = [k for k, v in snowflake.items() if not v and k in ['account', 'user', 'password']]
        if missing:
            raise ValueError(f"Missing required Snowflake credentials for client '{client_id}': {', '.join(missing)}")
            
        return snowflake
    
    def get_data_dictionary_path(self, client_id: str) -> str:
        """
        Get the data dictionary path for a client
        
        Args:
            client_id: The client identifier
            
        Returns:
            Path to the data dictionary
        """
        if client_id not in self.clients:
            raise ValueError(f"Client ID '{client_id}' not found")
        
        return self.clients[client_id]['data_dictionary_path']
    
    def get_llm_config(self, client_id: str, model_provider: str) -> Dict[str, str]:
        """
        Get LLM API key and model for a client and model provider
        
        Args:
            client_id: The client identifier
            model_provider: The model provider (openai, anthropic, gemini)
            
        Returns:
            Dictionary with API key and model from environment file
        """
        if client_id not in self.clients:
            raise ValueError(f"Client ID '{client_id}' not found")
        
        if model_provider.lower() not in ['openai', 'anthropic', 'gemini']:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Load client environment variables
        env_vars = self._load_client_env(client_id)
        
        provider = model_provider.lower()
        api_key_var = f"{provider.upper()}_API_KEY"
        model_var = f"{provider.upper()}_MODEL"
        
        api_key = env_vars.get(api_key_var, '')
        model = env_vars.get(model_var, '')
        
        if not api_key:
            raise ValueError(f"API key for {provider} not found for client '{client_id}'")
            
        return {
            'api_key': api_key,
            'model': model
        }
    
    def list_active_clients(self) -> List[Dict[str, Any]]:
        """
        List all active clients
        
        Returns:
            List of active client information
        """
        return [
            {
                'id': client_id, 
                'name': client['name'], 
                'description': client['description']
            }
            for client_id, client in self.clients.items() 
            if client['active']
        ]
        
    def create_client_env_template(self, client_id: str, output_path: Optional[str] = None) -> str:
        """
        Create a template .env file for a client
        
        Args:
            client_id: The client identifier
            output_path: Optional path to write the template
            
        Returns:
            Content of the template file
        """
        if client_id not in self.clients:
            raise ValueError(f"Client ID '{client_id}' not found")
            
        # Generate template content
        prefix = f"CLIENT_{client_id.upper()}_"
        template = f"""# Environment file for client: {client_id}

# Snowflake credentials
{prefix}SNOWFLAKE_ACCOUNT=
{prefix}SNOWFLAKE_USER=
{prefix}SNOWFLAKE_PASSWORD=
{prefix}SNOWFLAKE_WAREHOUSE=
{prefix}SNOWFLAKE_DATABASE=
{prefix}SNOWFLAKE_SCHEMA=

# OpenAI configuration
{prefix}OPENAI_API_KEY=
{prefix}OPENAI_MODEL=gpt-4-turbo

# Anthropic configuration
{prefix}ANTHROPIC_API_KEY=
{prefix}ANTHROPIC_MODEL=claude-3-opus-20240229

# Gemini configuration
{prefix}GEMINI_API_KEY=
{prefix}GEMINI_MODEL=models/gemini-pro
"""
        
        # Write to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template)
                
        return template


# Create a singleton instance for use throughout the application
client_manager = ClientManager()
