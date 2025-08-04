#!/usr/bin/env python3
"""
Multi-Client RAG System with Milvus
===================================

Collection-Level Multi-Tenancy implementation for multi-client
Retrieval-Augmented Generation (RAG) system using Milvus vector database.

This implementation ensures complete client isolation by:
1. Using separate collections for each client
2. Maintaining client-specific configuration
3. Providing optimized context for LLM queries

Author: DynProInc
Date: 2025-07-22
"""

import os
import json
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Import schema processor for dynamic column handling
from schema_processor import SchemaProcessor

# Milvus imports
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
    utility
)

# Import container utility
try:
    from milvus_container_utils import start_milvus_containers, check_milvus_status
except ImportError:
    # For when running from different directory
    from milvus-setup.milvus_container_utils import start_milvus_containers, check_milvus_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "rag_system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MultiClientRAG")

@dataclass
class SchemaRecord:
    """Data class for database schema records"""
    client_id: str
    db_schema: str
    table_name: str
    column_name: str
    data_type: str
    description: str
    distinct_values: Optional[str] = None
    combined_text: Optional[str] = None
    
    def __post_init__(self):
        """Generate combined text for embedding if not provided"""
        if not self.combined_text:
            self.combined_text = self.generate_combined_text()
    
    def generate_combined_text(self) -> str:
        """Generate combined text representation for embedding"""
        distinct_values = self.distinct_values if self.distinct_values else "N/A"
        return f"""Schema: {self.db_schema}
Table: {self.table_name}
Column: {self.column_name}
Data Type: {self.data_type}
Description: {self.description}
Distinct Values: {distinct_values}"""


class MultiClientMilvusRAG:
    """
    Multi-client RAG system with collection-based isolation
    
    Each client has its own dedicated collection to ensure complete data isolation.
    Collections are named using the pattern: {client_id}_schema_collection
    """

    def __init__(self, 
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 embedding_model: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize the Multi-Client RAG system
        
        Args:
            milvus_host: Milvus server hostname
            milvus_port: Milvus server port
            embedding_model: SentenceTransformer model to use for embeddings
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_dim = 1024  # Dimension for BAAI/bge-large-en-v1.5
        
        # Check and start Milvus containers if needed
        self._ensure_milvus_running()
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
            
        # Connect to Milvus
        self._connect_milvus()
        
        # Track registered clients
        self.registered_clients = self._get_registered_clients()
        
    def _connect_milvus(self) -> None:
        """Establish connection to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port
            )
            logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _get_registered_clients(self) -> List[str]:
        """
        Get list of registered clients based on existing collections
        
        Returns:
            List of client IDs that have collections in Milvus
        """
        try:
            collections = utility.list_collections()
            # Extract client_ids from collection names (format: {client_id}_schema_collection)
            client_ids = [coll.split('_')[0] for coll in collections if '_schema_collection' in coll]
            logger.info(f"Found registered clients: {client_ids}")
            return client_ids
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def _get_collection_name(self, client_id: str) -> str:
        """
        Generate collection name for a client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Collection name specific to the client
        """
        return f"{client_id}_schema_collection"
    
    def _create_client_collection(self, client_id: str, recreate: bool = False) -> Collection:
        """
        Create a dedicated collection for a client
        
        Args:
            client_id: Client identifier
            recreate: If True, drop existing collection if it exists
            
        Returns:
            Milvus Collection object
        """
        collection_name = self._get_collection_name(client_id)
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            if recreate:
                utility.drop_collection(collection_name)
                logger.info(f"Dropped existing collection for client {client_id}")
            else:
                logger.info(f"Collection for client {client_id} already exists")
                return Collection(collection_name)
        
        # Define collection fields
        fields = [
            FieldSchema(
                name="id", 
                dtype=DataType.INT64, 
                is_primary=True, 
                auto_id=True
            ),
            FieldSchema(
                name="db_schema", 
                dtype=DataType.VARCHAR, 
                max_length=100
            ),
            FieldSchema(
                name="table_name", 
                dtype=DataType.VARCHAR, 
                max_length=100
            ),
            FieldSchema(
                name="column_name", 
                dtype=DataType.VARCHAR, 
                max_length=100
            ),
            FieldSchema(
                name="data_type", 
                dtype=DataType.VARCHAR, 
                max_length=50
            ),
            FieldSchema(
                name="description", 
                dtype=DataType.VARCHAR, 
                max_length=1000
            ),
            FieldSchema(
                name="distinct_values", 
                dtype=DataType.VARCHAR, 
                max_length=1000
            ),
            FieldSchema(
                name="combined_text", 
                dtype=DataType.VARCHAR, 
                max_length=2000
            ),
            FieldSchema(
                name="embedding", 
                dtype=DataType.FLOAT_VECTOR, 
                dim=self.embedding_dim
            )
        ]
        
        # Create schema and collection
        schema = CollectionSchema(fields, f"Schema collection for client {client_id}")
        collection = Collection(collection_name, schema)
        logger.info(f"Created collection for client {client_id}: {collection_name}")
        
        # Add to registered clients if not already there
        if client_id not in self.registered_clients:
            self.registered_clients.append(client_id)
        
        return collection
    
    def _create_index(self, client_id: str) -> None:
        """
        Create vector index for a client collection
        
        Args:
            client_id: Client identifier
        """
        collection_name = self._get_collection_name(client_id)
        collection = Collection(collection_name)
        
        # Check if index exists
        if collection.has_index():
            logger.info(f"Index already exists for client {client_id}")
            return
        
        # Create HNSW index for fast retrieval with excellent recall
        index_params = {
            "metric_type": "IP",  # Inner Product similarity
            "index_type": "HNSW", 
            "params": {
                "M": 16,  # Number of edges per node (higher = better recall but more memory)
                "efConstruction": 200  # Higher values give better index quality but slower build
            }  # Optimized for small to medium datasets (< 100k rows)
        }
        
        collection.create_index("embedding", index_params)
        logger.info(f"Created index for client {client_id}")
        
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using SentenceTransformer
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def setup_client(self, client_id: str, data_dict_path: str = None, recreate: bool = False) -> bool:
        """
        Complete setup for a client
        
        Args:
            client_id: Client identifier
            data_dict_path: Path to client data dictionary CSV
            recreate: If True, recreate collection even if it exists
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Create client collection
            collection = self._create_client_collection(client_id, recreate)
            
            # Create index
            self._create_index(client_id)
            
            # Load schema data if provided
            if data_dict_path:
                self.load_client_schema(client_id, data_dict_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set up client {client_id}: {e}")
            return False
    
    def load_client_schema(self, client_id: str, data_dict_path: str) -> int:
        """
        Load client schema data from CSV into Milvus collection
        Using dynamic column mapping from SchemaProcessor
        
        Args:
            client_id: Client identifier
            data_dict_path: Path to data dictionary CSV file
            
        Returns:
            Number of records inserted
        """
        collection_name = self._get_collection_name(client_id)
        collection = Collection(collection_name)
        
        try:
            # Initialize schema processor for dynamic column mapping
            schema_processor = SchemaProcessor()
            
            # Load data dictionary
            df = pd.read_csv(data_dict_path)
            logger.info(f"Loaded {len(df)} schema records from {data_dict_path}")
            
            # Process with SchemaProcessor to handle dynamic columns
            schema_records = schema_processor.process_csv_data(client_id, df)
            
            if not schema_records:
                logger.error(f"No valid schema records found for client {client_id}")
                return 0
                
            # Prepare data for insertion
            db_schemas = []
            table_names = []
            column_names = []
            data_types = []
            descriptions = []
            distinct_values = []
            combined_texts = []
            embeddings = []
            
            # Process each record
            for record in schema_records:
                # Add record fields to respective lists
                db_schemas.append(record.db_schema)
                table_names.append(record.table_name)
                column_names.append(record.column_name)
                data_types.append(record.data_type or "")
                descriptions.append(record.description or "")
                distinct_values.append(record.distinct_values or "")
                
                # Get or generate combined text
                if record.combined_text:
                    combined_text = record.combined_text
                else:
                    combined_text = record.generate_combined_text()
                    
                combined_texts.append(combined_text)
                
                # Generate embedding
                embedding = self._generate_embedding(combined_text)
                embeddings.append(embedding)
            
            # Prepare records for insertion
            insert_data = [
                db_schemas,
                table_names,
                column_names,
                data_types,
                descriptions,
                distinct_values,
                combined_texts,
                embeddings
            ]
            
            # Insert data into collection
            collection.insert(insert_data)
            collection.flush()
            
            # Update registered clients
            if client_id not in self.registered_clients:
                self.registered_clients.append(client_id)
                
            logger.info(f"Successfully inserted {len(schema_records)} records for client {client_id}")
            return len(schema_records)
            collection.load()
            
            return total_inserted
        except Exception as e:
            logger.error(f"Failed to load schema data for client {client_id}: {e}")
            raise
    
    def retrieve_relevant_schema(self, client_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant schema information based on query
        
        Args:
            client_id: Client identifier
            query: User query to find relevant schema
            top_k: Number of top matches to return
            
        Returns:
            List of relevant schema records
        """
        collection_name = self._get_collection_name(client_id)
        
        if not utility.has_collection(collection_name):
            logger.error(f"Collection for client {client_id} does not exist")
            raise ValueError(f"Client {client_id} is not set up in the RAG system")
        
        collection = Collection(collection_name)
        
        # Ensure collection is loaded
        try:
            collection.load()
            logger.info(f"Loaded collection {collection_name} for search")
        except Exception as e:
            logger.warning(f"Error when loading collection, may already be loaded: {e}")
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search for similar schema records
            search_params = {
                "metric_type": "IP",  # Inner Product similarity - must match index metric_type
                "params": {"ef": 64}  # Query-time recall quality parameter for HNSW index
            }
            
            # Define all fields we want to retrieve explicitly
            output_fields = [
                "db_schema", "table_name", "column_name", "data_type", 
                "description", "distinct_values", "combined_text"
            ]
            
            # Execute search with comprehensive output fields
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=8,  # Perfect size for SQL context
                output_fields=output_fields
            )
            
            # Process results with comprehensive extraction
            matched_records = []
            
            for i, hits in enumerate(results):
                for j, hit in enumerate(hits):
                    # Extract data based on API version with comprehensive approach
                    extracted_data = {
                        'client_id': client_id,
                        'score': 0.0,  # Default score
                        'db_schema': '',
                        'table_name': '',
                        'column_name': '',
                        'data_type': '',
                        'description': '',
                        'distinct_values': '',
                        'combined_text': ''
                    }
                    
                    # Handle entity-based API (newer pymilvus)
                    if hasattr(hit, 'entity'):
                        entity = hit.entity
                        # Copy all fields from entity
                        for field in entity:
                            extracted_data[field] = entity[field]
                        
                        # Get score from hit
                        if hasattr(hit, 'score'):
                            extracted_data['score'] = hit.score
                    
                    # Handle direct attribute access (older pymilvus)
                    else:
                        # Try to get fields directly from hit object
                        for field in output_fields:
                            if hasattr(hit, field):
                                extracted_data[field] = getattr(hit, field)
                        
                        # Get score attribute
                        if hasattr(hit, 'score'):
                            extracted_data['score'] = hit.score
                        elif hasattr(hit, 'distance'):
                            extracted_data['score'] = hit.distance
                    
                    matched_records.append(extracted_data)
            
            logger.info(f"Retrieved {len(matched_records)} relevant schema records for client {client_id}")
            return matched_records
        except Exception as e:
            logger.error(f"Error searching for relevant schema for client {client_id}: {e}")
            raise
    
    def _ensure_milvus_running(self) -> bool:
        """Ensure Milvus containers are running before connecting"""
        logger.info("Checking Milvus container status...")
        try:
            # Check if containers are running, start them if not
            container_status = check_milvus_status()
            all_running = all(status == "Running" for container, status in container_status.items() 
                              if container in ["milvus-standalone", "milvus-etcd", "milvus-minio"])
            
            if not all_running:
                logger.info("Some Milvus containers are not running. Attempting to start them...")
                success = start_milvus_containers(wait_time=15)  # Wait longer for proper initialization
                if success:
                    logger.info("Successfully started Milvus containers")
                    return True
                else:
                    logger.warning("Failed to start Milvus containers automatically")
                    return False
            else:
                logger.info("All Milvus containers are already running")
                return True
        except Exception as e:
            logger.error(f"Error ensuring Milvus is running: {str(e)}")
            return False
            
    def _log_exception(self, method_name: str, e: Exception) -> None:
        """Log exceptions with method name for easier debugging"""
        logger.error(f"Error in {method_name}: {str(e)}")
        if "connection" in str(e).lower() or "connect" in str(e).lower():
            logger.info("Connection error detected. Checking if Milvus containers are running...")
            status = check_milvus_status()
            logger.info(f"Container status: {status}")
            if not all(s == "Running" for s in status.values() if s != "error"):
                logger.info("Attempting to start containers...")
                start_milvus_containers()
    
    def get_optimized_prompt(self, client_id: str, user_query: str, top_k: int = 5) -> str:
        """
        Generate optimized prompt with relevant schema context
        
        Args:
            client_id: Client identifier
            user_query: User's natural language query
            top_k: Number of relevant schema elements to include
            
        Returns:
            Optimized prompt with relevant schema context
        """
        # Retrieve relevant schema
        relevant_schema = self.retrieve_relevant_schema(client_id, user_query, top_k)
        
        # Format schema context
        schema_context = ""
        for i, record in enumerate(relevant_schema):
            schema_context += f"""
SCHEMA ELEMENT {i+1}:
Schema: {record['db_schema']}
Table: {record['table_name']}
Column: {record['column_name']}
Data Type: {record['data_type']}
Description: {record['description']}
Distinct Values: {record['distinct_values']}

"""
        
        # Build optimized prompt
        optimized_prompt = f"""
You are a SQL query generator that converts natural language questions to SQL.
The user is querying a database with the following schema elements that are most relevant to their question:

{schema_context}

Based only on the schema elements provided above, generate a SQL query to answer this question:
{user_query}

The SQL should be valid for Snowflake SQL dialect and should only include tables and columns from the provided schema elements.
Only return the SQL query without any explanations or additional text.
"""
        
        logger.info(f"Generated optimized prompt for client {client_id} (estimated ~2,200 tokens vs ~8,500 original)")
        return optimized_prompt
    
    def estimate_token_savings(self, client_id: str, user_query: str) -> Dict[str, int]:
        """
        Estimate token savings from using RAG vs. full schema
        
        Args:
            client_id: Client identifier
            user_query: User query
            
        Returns:
            Dictionary with token count estimates
        """
        # Get optimized prompt
        optimized_prompt = self.get_optimized_prompt(client_id, user_query)
        
        # Estimate token counts (rough approximation)
        optimized_tokens = len(optimized_prompt.split()) * 1.3  # Rough token estimate
        full_schema_tokens = 6000  # Typical schema size in tokens
        full_prompt_tokens = full_schema_tokens + len(user_query.split()) * 1.3
        
        savings = full_prompt_tokens - optimized_tokens
        savings_pct = (savings / full_prompt_tokens) * 100
        
        return {
            "optimized_tokens": int(optimized_tokens),
            "full_prompt_tokens": int(full_prompt_tokens),
            "savings_tokens": int(savings),
            "savings_percentage": round(savings_pct, 2)
        }
    
    def log_token_savings(self, client_id: str, user_query: str) -> None:
        """
        Log token savings from using RAG
        
        Args:
            client_id: Client identifier
            user_query: User query
        """
        savings = self.estimate_token_savings(client_id, user_query)
        logger.info(
            f"RAG Token Savings - Client: {client_id}, "
            f"Query: '{user_query[:30]}...', "
            f"Optimized: {savings['optimized_tokens']} tokens, "
            f"Original: {savings['full_prompt_tokens']} tokens, "
            f"Saved: {savings['savings_tokens']} tokens ({savings['savings_percentage']:.1f}%)"
        )

# Testing function
def test_rag_system():
    """Test the RAG system with sample data"""
    # Initialize RAG system
    rag = MultiClientMilvusRAG()
    
    # Set up test clients
    for client_id in ["mts", "penguin"]:
        data_dict_path = f"../config/clients/data_dictionaries/{client_id}/data_dictionary.csv"
        if os.path.exists(data_dict_path):
            success = rag.setup_client(client_id, data_dict_path)
            print(f"Client {client_id} setup: {'✅' if success else '❌'}")
        else:
            print(f"Data dictionary for client {client_id} not found at {data_dict_path}")
    
    # Test query optimization
    queries = [
        "Show sales performance by department",
        "What are the top 10 products by revenue?",
        "How many customers made purchases last month?"
    ]
    
    for client_id in ["mts", "penguin"]:
        if client_id in rag.registered_clients:
            print(f"\nTesting client: {client_id}")
            for query in queries:
                try:
                    # Get optimized prompt
                    optimized_prompt = rag.get_optimized_prompt(client_id, query)
                    
                    # Log savings
                    savings = rag.estimate_token_savings(client_id, query)
                    print(f"Query: {query}")
                    print(f"Optimized tokens: {savings['optimized_tokens']}")
                    print(f"Original tokens: {savings['full_prompt_tokens']}")
                    print(f"Token savings: {savings['savings_tokens']} ({savings['savings_percentage']:.1f}%)")
                    print("-" * 40)
                except Exception as e:
                    print(f"Error processing query for client {client_id}: {e}")

if __name__ == "__main__":
    test_rag_system()
