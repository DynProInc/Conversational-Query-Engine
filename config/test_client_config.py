"""
Client Configuration Test Script
- Tests the client configuration system
- Verifies environment loading for multiple clients
"""

import os
from pathlib import Path
import sys
import pprint

# Add parent directory to path to allow importing client_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.client_manager import client_manager

def test_client_config(client_id: str):
    """
    Test client configuration
    
    Args:
        client_id: The client identifier
    """
    print(f"\n{'='*50}")
    print(f"Testing configuration for client: {client_id}")
    print(f"{'='*50}")
    
    try:
        # Get basic client info
        client_info = client_manager.get_client_info(client_id)
        print(f"\nClient info:")
        print(f"  Name: {client_info['name']}")
        print(f"  Description: {client_info['description']}")
        print(f"  Active: {client_info['active']}")
        
        # Check data dictionary path
        dict_path = client_info['data_dictionary_path']
        print(f"  Data Dictionary: {dict_path}")
        
        # Check if dictionary file exists
        dict_file = Path(dict_path)
        if dict_file.exists():
            print(f"  Dictionary file exists: ✓")
        else:
            print(f"  Dictionary file exists: ✗ (File not found)")
        
        # Check env file
        env_path = client_manager._get_client_env_path(client_id)
        if env_path.exists():
            print(f"\nEnvironment file: ✓ ({env_path})")
        else:
            print(f"\nEnvironment file: ✗ (Not found at {env_path})")
            print(f"  Run: python -c \"from config.client_setup import setup_client_environment; setup_client_environment('{client_id}')\"")

        # Try to get Snowflake connection parameters
        print("\nSnowflake connection parameters:")
        try:
            snowflake = client_manager.get_snowflake_connection_params(client_id)
            # Only show non-sensitive parts for security
            safe_params = {
                'account': ('✓ ' + snowflake.get('account')) if snowflake.get('account') else '✗ Not configured',
                'user': '✓ Configured' if snowflake.get('user') else '✗ Not configured',
                'password': '✓ Configured' if snowflake.get('password') else '✗ Not configured',
                'warehouse': ('✓ ' + snowflake.get('warehouse')) if snowflake.get('warehouse') else '✗ Not configured',
                'database': ('✓ ' + snowflake.get('database')) if snowflake.get('database') else '✗ Not configured',
                'schema': ('✓ ' + snowflake.get('schema')) if snowflake.get('schema') else '✗ Not configured'
            }
            for key, value in safe_params.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {str(e)}")
            
        # Try to get LLM configurations
        print("\nLLM API Configurations:")
        for provider in ['openai', 'anthropic', 'gemini']:
            print(f"  {provider.capitalize()}:")
            try:
                llm_config = client_manager.get_llm_config(client_id, provider)
                key_status = '✓ Configured' if llm_config.get('api_key') else '✗ Not configured'
                model = llm_config.get('model', '✗ Not configured')
                if model:
                    model = '✓ ' + model
                print(f"    API Key: {key_status}")
                print(f"    Model: {model}")
            except Exception as e:
                print(f"    Status: ✗ Error - {str(e)}")
                
    except Exception as e:
        print(f"Error testing client configuration: {str(e)}")
        
    print(f"\n{'='*50}\n")


def validate_client_environment(client_id):
    """
    Validate and fix client environment setup
    
    This function will check if the client's environment is properly configured
    and provide guidance on how to fix any issues.
    
    Args:
        client_id: The client ID to check
    """
    print(f"\n📋 VALIDATING CLIENT ENVIRONMENT: {client_id}\n")
    
    # Check if client exists in client registry
    try:
        client_info = client_manager.get_client_info(client_id)
        print(f"✅ Client '{client_id}' exists in registry")
    except Exception as e:
        print(f"❌ Client '{client_id}' not found in registry: {e}")
        print(f"   Solution: Create client directory and configuration")
        return
    
    # Check for environment file
    env_path = client_manager._get_client_env_path(client_id)
    if env_path.exists():
        print(f"✅ Environment file exists: {env_path}")
    else:
        print(f"❌ Environment file missing: {env_path}")
        print(f"   Solution: Create a .env file at {env_path} with required credentials")
    
    # Check Snowflake credentials
    try:
        snowflake_params = client_manager.get_snowflake_connection_params(client_id)
        missing = []
        for required in ['account', 'user', 'password']:
            if not snowflake_params.get(required):
                missing.append(required)
        
        if missing:
            print(f"❌ Missing required Snowflake credentials: {', '.join(missing)}")
            print(f"   Solution: Add CLIENT_{client_id.upper()}_SNOWFLAKE_{missing[0].upper()}=value to {env_path}")
        else:
            print(f"✅ All required Snowflake credentials are present")
    except Exception as e:
        print(f"❌ Error checking Snowflake credentials: {e}")
        print(f"   Solution: Check environment variable format in {env_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test client configuration')
    parser.add_argument('client_id', help='Client ID to test')
    parser.add_argument('--validate', action='store_true', help='Run validation checks')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_client_environment(args.client_id)
    else:
        test_client_config(args.client_id)

def main():
    """Run tests for all active clients"""
    # List active clients
    active_clients = client_manager.list_active_clients()
    print(f"Found {len(active_clients)} active clients:")
    for client in active_clients:
        print(f"  - {client['id']}: {client['name']}")
        
    # Test each client
    for client in active_clients:
        test_client_config(client['id'])
        
    print("\n" + "="*50)
    print("Testing complete!")
    print("\nNext steps:")
    print("  1. Edit the client environment files to add actual credentials")
    print("  2. Update your API server to use the client_integration module")
    print("  3. Make API requests with the 'client_id' parameter")
    print("="*50)
        
if __name__ == "__main__":
    main()
