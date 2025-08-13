#!/usr/bin/env python3
"""
OpenAI Embedding Test with Milvus

This script demonstrates how to use OpenAI's text-embedding-3-large model
with Milvus vector database for schema-based RAG.

Features:
- Loads environment variables from MTS client config
- Processes schema data from MTS dictionary CSV
- Generates embeddings using OpenAI text-embedding-3-large model
- Stores and retrieves embeddings from Milvus
- Optimized with IP metric type and HNSW index (M=16, efConstruction=200)
"""

import os
import sys
import csv
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Fix import paths for modules in parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OpenAIEmbedding")

# Connect to Milvus
from pymilvus import connections, Collection, utility
from pymilvus import FieldSchema, CollectionSchema, DataType

# Schema record class for processing CSV data
class SchemaRecord:
    """Schema record for database tables and columns"""
    
    def __init__(self, db_schema="", table_name="", column_name="", 
                 data_type="", description="", table_definition="", distinct_values=""):
        self.db_schema = db_schema
        self.table_name = table_name
        self.column_name = column_name
        self.data_type = data_type
        self.description = description
        self.table_definition = table_definition
        self.distinct_values = distinct_values
    
    def generate_combined_text(self):
        """Generate combined text for embedding"""
        parts = []
        
        if self.db_schema:
            parts.append(f"Database: {self.db_schema}")
        
        if self.table_name:
            parts.append(f"Table: {self.table_name}")
        
        if self.column_name:
            parts.append(f"Column: {self.column_name}")
        
        if self.data_type:
            parts.append(f"Type: {self.data_type}")
        
        if self.description:
            parts.append(f"Description: {self.description}")
        
        if self.table_definition:
            parts.append(f"Table Definition: {self.table_definition}")
        
        if self.distinct_values:
            parts.append(f"Values: {self.distinct_values}")
        
        return " | ".join(parts)

class OpenAIEmbeddingTest:
    """Test implementation for OpenAI embeddings with Milvus"""
    
    def __init__(self, client_id="mts", milvus_host=None, milvus_port=None, 
                 model_name="text-embedding-3-large"):
        """Initialize the OpenAI embedding test"""
        self.client_id = client_id
        self.model_name = model_name
        
        # Load environment variables from client config
        self._load_env_variables()
        
        # Get Milvus connection parameters from environment variables or use defaults
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = milvus_port or os.getenv("MILVUS_PORT", "19530")
        
        logger.info(f"Using Milvus connection: {self.milvus_host}:{self.milvus_port}")
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        logger.info(f"Initialized OpenAI Embedding Test for client {client_id} using model {model_name}")
    
    def _load_env_variables(self):
        """Load environment variables from client config"""
        # Construct path to client env file
        env_file = os.path.join(
            parent_dir, 
            "LLM Query Engine", 
            "config", 
            "clients", 
            "env", 
            f"{self.client_id}.env"
        )
        
        # Load environment variables
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        else:
            logger.warning(f"Environment file not found: {env_file}")
    
    def _connect_to_milvus(self):
        """Connect to Milvus server"""
        try:
            # Check if connection already exists
            try:
                if utility.has_connection("default"):
                    # Check if the connection parameters match
                    conn_params = connections.get_connection_addr("default")
                    if conn_params.get('host') == self.milvus_host and str(conn_params.get('port')) == str(self.milvus_port):
                        logger.info(f"Reusing existing Milvus connection to {self.milvus_host}:{self.milvus_port}")
                        return
                    else:
                        # Close existing connection if parameters don't match
                        connections.disconnect("default")
                        logger.info(f"Closed existing Milvus connection with different parameters")
            except Exception as check_err:
                logger.warning(f"Error checking existing connection: {check_err}")
                # Try to disconnect just in case
                try:
                    connections.disconnect("default")
                except:
                    pass
            
            # Create new connection
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            # Import OpenAI library
            from openai import OpenAI
            
            # Get API key from environment variables - try multiple formats
            api_key = os.getenv(f"CLIENT_{self.client_id.upper()}_OPENAI_API_KEY")  # CLIENT_MTS_OPENAI_API_KEY
            if not api_key:
                api_key = os.getenv(f"{self.client_id.upper()}_OPENAI_API_KEY")  # MTS_OPENAI_API_KEY
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")  # Fallback to global API key
            
            if not api_key:
                raise ValueError(f"OpenAI API key not found for client {self.client_id}")
            
            # Initialize OpenAI client
            self.client = OpenAI(api_key=api_key)
            logger.info("Initialized OpenAI client successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def get_collection_name(self):
        """Get collection name for the client"""
        return f"{self.client_id}_openai_embedding_test"
    
    def process_csv_data(self, csv_path):
        """Process CSV data from file path"""
        try:
            logger.info(f"Processing CSV data from {csv_path}")
            
            records = []
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            success = False
            
            for encoding in encodings:
                try:
                    # Read CSV file with current encoding
                    with open(csv_path, 'r', encoding=encoding) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Create schema record - match all fields from rag_embedding.py
                            record = SchemaRecord(
                                db_schema=row.get('DB_SCHEMA', ''),
                                table_name=row.get('TABLE_NAME', ''),
                                column_name=row.get('COLUMN_NAME', ''),
                                data_type=row.get('DATA_TYPE', ''),
                                description=row.get('DESCRIPTION', ''),
                                table_definition=row.get('TABLE_DEFINITION', ''),
                                distinct_values=row.get('DISTINCT_VALUES', '')
                            )
                            records.append(record)
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to read CSV with encoding {encoding}: {e}")
            
            if not success:
                raise Exception("Failed to read CSV with any encoding")
            
            logger.info(f"Processed {len(records)} records from CSV")
            return records
        except Exception as e:
            logger.error(f"Error processing CSV data: {e}")
            raise
    
    def create_collection(self, schema_data):
        """Create Milvus collection for embeddings"""
        try:
            collection_name = self.get_collection_name()
            logger.info(f"Creating collection: {collection_name}")
            
            # Check if collection exists and drop if necessary
            if utility.has_collection(collection_name):
                logger.info(f"Dropping existing collection: {collection_name}")
                utility.drop_collection(collection_name)
                # Verify collection was dropped
                if utility.has_collection(collection_name):
                    logger.error(f"Failed to drop collection: {collection_name}")
                    raise Exception(f"Failed to drop collection: {collection_name}")
                logger.info(f"Successfully dropped collection: {collection_name}")
            
            # Determine vector dimension based on model
            if self.model_name == "text-embedding-3-small":
                dim = 1536
            else:  # text-embedding-3-large
                dim = 3072
            
            # Prepare schema - match the structure in rag_embedding.py
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="db_schema", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="column_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="distinct_values", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="combined_text", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072)  # OpenAI large is 3072
            ]
            schema = CollectionSchema(fields, f"OpenAI embedding test collection for {self.client_id}")
            collection = Collection(collection_name, schema)
            
            # Create index with HNSW parameters
            index_params = {
                "metric_type": "IP",  # Inner Product similarity
                "index_type": "HNSW",
                "params": {
                    "M": 16,  # Number of edges per node (higher = better recall but more memory)
                    "efConstruction": 200  # Higher values give better index quality but slower build
                }
            }
            
            # Create the index
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info("Collection and index created successfully")
            
            return collection
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def generate_embedding(self, text):
        """Generate embedding using OpenAI embedding model"""
        try:
            # Use the embeddings.create method to generate embedding
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            
            # Extract embedding values
            embedding = response.data[0].embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def build_embeddings(self, csv_path):
        """Build embeddings from CSV data and store in Milvus"""
        try:
            # Process CSV data
            schema_data = self.process_csv_data(csv_path)
            
            # Create collection
            collection = self.create_collection(schema_data)
            
            # Generate embeddings and insert records
            logger.info(f"Generating embeddings for {len(schema_data)} records")
            
            data = []
            batch_size = 20
            
            for i, record in enumerate(schema_data):
                if i % 10 == 0:
                    logger.info(f"Processing record {i+1}/{len(schema_data)}")
                
                # Generate combined text for embedding
                combined_text = record.generate_combined_text()
                
                # Generate embedding
                embedding = self.generate_embedding(combined_text)
                
                # Prepare record for insertion with proper field limits (match rag_embedding.py)
                data.append({
                    "db_schema": record.db_schema[:100] if record.db_schema else "",
                    "table_name": record.table_name[:100] if record.table_name else "",
                    "column_name": record.column_name[:100] if record.column_name else "",
                    "data_type": record.data_type[:50] if record.data_type else "",
                    "description": record.description[:2000] if record.description else "",
                    "distinct_values": record.distinct_values[:1000] if record.distinct_values else "",
                    "combined_text": combined_text[:5000],
                    "embedding": embedding
                })
                
                # Insert in batches
                if len(data) >= batch_size or i == len(schema_data) - 1:
                    collection.insert(data)
                    logger.info(f"Inserted batch of {len(data)} records")
                    data = []
            
            # Flush and load collection
            collection.flush()
            collection.load()
            
            count = collection.num_entities
            logger.info(f"Collection has {count} entities")
            
            return True, f"Successfully built embeddings for {count} records"
        except Exception as e:
            logger.error(f"Error building embeddings: {e}")
            return False, str(e)
    
    def query(self, query_text, top_k=5):
        """Query embeddings using OpenAI embedding model"""
        try:
            collection_name = self.get_collection_name()
            
            # Check if collection exists
            if not utility.has_collection(collection_name):
                return False, f"Collection {collection_name} not found", []
            
            # Load collection
            collection = Collection(collection_name)
            collection.load()
            
            # Generate embedding for query
            query_embedding = self.generate_embedding(query_text)
            
            # Search parameters for HNSW
            search_params = {
                "metric_type": "IP",
                "params": {"ef": 64}  # Query-time recall quality parameter for HNSW
            }
            
            # Search for similar records - match fields from rag_embedding.py
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["db_schema", "table_name", "column_name", "data_type", 
                              "description", "distinct_values", "combined_text"]
            )
            
            # Process results
            matches = []
            for hits in results:
                for hit in hits:
                    match = {
                        "db_schema": hit.entity.get("db_schema"),
                        "table_name": hit.entity.get("table_name"),
                        "column_name": hit.entity.get("column_name"),
                        "data_type": hit.entity.get("data_type"),
                        "description": hit.entity.get("description"),
                        "distinct_values": hit.entity.get("distinct_values"),
                        "combined_text": hit.entity.get("combined_text"),
                        "score": hit.score
                    }
                    matches.append(match)
                    
                    # Print match details for better visibility
                    print(f"\n--- Match {len(matches)} (Score: {hit.score:.4f}) ---")
                    print(f"Schema: {hit.entity.get('db_schema')}")
                    print(f"Table: {hit.entity.get('table_name')}")
                    print(f"Column: {hit.entity.get('column_name')}")
                    print(f"Data Type: {hit.entity.get('data_type')}")
                    print(f"Description: {hit.entity.get('description')[:100]}..." if len(hit.entity.get('description', '')) > 100 else f"Description: {hit.entity.get('description')}")
                    print("----------------------------")
            
            return True, f"Found {len(matches)} relevant matches", matches
        except Exception as e:
            logger.error(f"Error querying embeddings: {e}")
            return False, str(e), []
def main():
    """Main entry point"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="OpenAI Embedding Test with Milvus")
    parser.add_argument("--client", default="mts", help="Client ID")
    parser.add_argument("--csv", help="Path to CSV file")
    parser.add_argument("--query", help="Query text")
    parser.add_argument("--build", action="store_true", help="Build embeddings")
    parser.add_argument("--model", default="text-embedding-3-large", 
                        choices=["text-embedding-3-small", "text-embedding-3-large"],
                        help="OpenAI embedding model to use")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--output", help="Output file for query results")
    
    args = parser.parse_args()
    
    # Initialize test
    test = OpenAIEmbeddingTest(client_id=args.client, model_name=args.model)
    
    # Build embeddings if requested
    if args.build:
        success, message = test.build_embeddings(args.csv)
        print(f"Result: {message}")
    elif args.query:
        success, message, results = test.query(args.query, args.top_k)
        print(f"Result: {message}")
        
        # Print results in a readable format
        if results:
            print("\nTop matches for your query:")
            print("==========================")
            for i, match in enumerate(results):
                print(f"\n[Match {i+1}] Score: {match.get('score', 0):.4f}")
                print(f"Schema: {match.get('db_schema', '')}")
                print(f"Table: {match.get('table_name', '')}")
                print(f"Column: {match.get('column_name', '')}")
                print(f"Data Type: {match.get('data_type', '')}")
                desc = match.get('description', '')
                if len(desc) > 100:
                    desc = desc[:97] + '...'
                print(f"Description: {desc}")
                print("-" * 50)
        
        # Save results to output file if specified
        if args.output and results:
            try:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output}")
            except Exception as e:
                print(f"Error saving results to file: {e}")
    else:
        print("Please specify either --build with --csv or --query")

if __name__ == "__main__":
    main()
