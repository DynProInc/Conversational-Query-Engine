"""
Client Setup Utility
- Helps set up client environment files
- Creates templates for secure credential storage
"""

import os
import argparse
from pathlib import Path
from .client_manager import client_manager, CLIENT_ENV_DIR

def setup_client_environment(client_id: str, secure_output: bool = True):
    """
    Set up a client environment file template
    
    Args:
        client_id: Client ID to set up
        secure_output: Whether to create the file in a secure location
    """
    try:
        # Get client info to validate client exists
        client_info = client_manager.get_client_info(client_id)
        if not client_info:
            print(f"Error: Client '{client_id}' not found in registry")
            return False
            
        # Determine output path
        if secure_output:
            # Create the env directory if it doesn't exist
            os.makedirs(CLIENT_ENV_DIR, exist_ok=True)
            output_path = CLIENT_ENV_DIR / f"{client_id}.env"
        else:
            # Create a template file in the current directory
            output_path = Path(f"{client_id}_template.env")
            
        # Check if file already exists to prevent overwriting
        if output_path.exists():
            print(f"Warning: Environment file already exists at {output_path}")
            response = input("Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return False
                
        # Create the template
        template = client_manager.create_client_env_template(client_id, str(output_path))
        
        print(f"\nEnvironment template created at: {output_path}")
        print("\nImportant: Fill in the credentials in this file and ensure it's not committed to Git!")
        print("The .gitignore file has been updated to exclude .env files.")
        
        return True
    except Exception as e:
        print(f"Error creating environment file: {str(e)}")
        return False

def create_client_registry_template():
    """Create a template for the client registry CSV without sensitive information"""
    output_path = Path("client_registry_template.csv")
    
    template = (
        "client_id,client_name,description,active,data_dictionary_path\n"
        "client1,Client One,Example client 1,true,path/to/dictionary1.csv\n"
        "client2,Client Two,Example client 2,true,path/to/dictionary2.csv\n"
    )
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)
        
    print(f"\nClient registry template created at: {output_path}")
    print("This template excludes sensitive information which should be stored in .env files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client setup utility")
    subparsers = parser.add_subparsers(dest="command")
    
    # Environment file command
    env_parser = subparsers.add_parser("env", help="Create client environment file template")
    env_parser.add_argument("client_id", help="Client ID to set up")
    env_parser.add_argument("--insecure", action="store_true", help="Create template in current directory instead of secure location")
    
    # Registry template command
    registry_parser = subparsers.add_parser("registry", help="Create client registry template")
    
    args = parser.parse_args()
    
    if args.command == "env":
        setup_client_environment(args.client_id, not args.insecure)
    elif args.command == "registry":
        create_client_registry_template()
    else:
        parser.print_help()
