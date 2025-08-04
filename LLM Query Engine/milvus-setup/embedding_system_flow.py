#!/usr/bin/env python3
"""
Multi-Client Embedding System Flow
=================================

This script demonstrates the complete flow of embedding generation
and usage in the multi-client RAG system.

1. Schema processing
2. Embedding generation 
3. Health status checking
4. RAG system integration

Author: DynProInc
Date: 2025-07-22
"""

import os
import sys
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmbeddingFlow")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import all required modules
try:
    # Import directly with module paths to avoid import errors
    sys.path.append(os.path.join(parent_dir, "milvus-setup"))
    from schema_processor import SchemaProcessor, SchemaRecord
    from generate_client_embeddings import ClientEmbeddingGenerator
    from check_embedding_status import check_embedding_status, load_client_registry
    from client_rag_manager import ClientRAGManager
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

def print_section(title):
    """Print a section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")

def demonstrate_client_flow(client_id):
    """Demonstrate the complete flow for a specific client"""
    print_section(f"EMBEDDING FLOW DEMONSTRATION FOR CLIENT: {client_id}")
    
    # Step 1: Check embedding status
    print("Step 1: Checking current embedding status")
    print("-" * 50)
    status = check_embedding_status(client_id)
    print(f"Client: {client_id}")
    print(f"Embeddings exist: {status['embeddings_exist']}")
    print(f"Up to date: {status.get('up_to_date', False)}")
    if status['embeddings_exist']:
        print(f"Record count: {status['record_count']}")
        print(f"Last updated: {status['last_updated']}")
    
    # Step 2: Process schema and generate embeddings
    print("\nStep 2: Processing schema and generating embeddings")
    print("-" * 50)
    generator = ClientEmbeddingGenerator()
    print(f"Loaded client registry with {len(generator.clients)} clients")
    
    # Get schema records for the client
    records = generator.process_client_dictionary(client_id)
    print(f"Processed {len(records)} schema records for client {client_id}")
    
    if records:
        # Display a sample record
        sample = records[0]
        print("\nSample schema record:")
        print(f"Table: {sample.table_name}")
        print(f"Column: {sample.column_name}")
        print(f"Data Type: {sample.data_type}")
        print(f"Description: {sample.description}")
        
        # Generate embeddings (but don't save)
        print("\nGenerating embeddings (preview only)...")
        embeddings = generator.generate_embeddings([r.generate_combined_text() for r in records[:3]])
        print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Step 3: RAG system integration
    print("\nStep 3: RAG System Integration")
    print("-" * 50)
    try:
        rag_manager = ClientRAGManager()
        print(f"Client in registry: {client_id in rag_manager.client_registry}")
        
        # Get dictionary path
        dict_path = rag_manager._get_client_data_dictionary_path(client_id)
        print(f"Dictionary path from registry: {dict_path}")
        print(f"Dictionary exists: {os.path.exists(dict_path) if dict_path else False}")
        
        # Initialize client (optional - may take time)
        print("\nNOTE: Full RAG integration would involve:")
        print("1. Creating Milvus collection for the client")
        print("2. Uploading embeddings to Milvus")
        print("3. Creating vector indices for search")
    except Exception as e:
        print(f"RAG integration preview only - skipping actual initialization: {str(e)}")

def show_system_architecture():
    """Display system architecture information"""
    print_section("MULTI-CLIENT EMBEDDING SYSTEM ARCHITECTURE")
    
    print("""
COMPONENTS AND RESPONSIBILITIES
-------------------------------

1. schema_processor.py
   - Defines SchemaRecord class for storing database schema information
   - Processes CSV data into SchemaRecord objects
   - Handles DB_SCHEMA column parsing and formatting
   - Generates combined text for embedding

2. generate_client_embeddings.py
   - Uses client_registry.csv to identify clients
   - Processes client data dictionaries
   - Generates embeddings using SentenceTransformer
   - Saves embeddings and record CSVs
   - Detects embedding update needs

3. check_embedding_status.py / embedding_health_check.py
   - Check if embeddings exist and are up-to-date
   - Reports on embedding health status
   - Integrates with client health check API

4. client_rag_manager.py
   - Manages client-specific RAG configurations
   - Uses client_registry.csv to resolve client dictionaries
   - Provides optimized prompts with schema context
   - Ensures strict client isolation

5. multi_client_rag.py
   - Implements collection-level multi-tenancy in Milvus
   - Creates client-specific collections
   - Retrieves relevant schema based on user queries
   - Estimates and logs token savings

DATA FLOW
---------
1. Client information sourced from client_registry.csv
2. Schema data loaded from client-specific data dictionaries
3. Schema processed into records with combined text
4. Embeddings generated from combined text
5. Embeddings saved to client-specific directories
6. RAG system loads embeddings into Milvus collections
7. User queries matched against embeddings for context
""")

def main():
    """Main function"""
    # Show system architecture
    show_system_architecture()
    
    # Get client generator to load registry
    generator = ClientEmbeddingGenerator()
    
    # Get list of clients from registry
    clients = list(generator.clients.keys())
    
    if not clients:
        print("No clients found in registry!")
        return
        
    # Demonstrate flow for first client
    demonstrate_client_flow(clients[0])
    
    print_section("EMBEDDING SYSTEM STATUS")
    
    # Show status for all clients
    for client_id in clients:
        status = check_embedding_status(client_id)
        print(f"\nClient: {client_id}")
        print(f"Status: {'HEALTHY' if status['embeddings_exist'] and status.get('up_to_date', False) else 'NEEDS UPDATE'}")
        if status['embeddings_exist']:
            print(f"Record count: {status['record_count']}")
            print(f"Last updated: {status['last_updated']}")

if __name__ == "__main__":
    main()
