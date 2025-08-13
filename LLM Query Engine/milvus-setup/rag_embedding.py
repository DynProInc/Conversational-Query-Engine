#!/usr/bin/env python3
"""
Fixed RAG Manager - Complete solution for multi-client Milvus RAG

This script provides a consolidated, final solution for:
1. Schema processing with proper column mapping
2. Milvus collection management (create, drop, rebuild)
3. Embedding generation with proper error handling
4. Client-specific RAG initialization and query
5. Enhanced with Qwen3-Reranker for improved relevance

Features:
- No hardcoded client names or paths
- Proper error handling with no silent fallbacks
- Dynamic client dictionary loading
- Client-isolated collections with separate embeddings
- Comprehensive console output for operations
- FIXED: Proper entity data extraction from Milvus search results
- NEW: Optional reranking with Qwen3-Reranker-0.6B
"""

import os
import sys
import csv
import json
import logging
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from datetime import datetime

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinalRAG")

# Fix import paths for modules in hyphenated directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import path resolver directly
try:
    # Try direct import first
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import path_resolver
    logger.info("Path resolver imported successfully")
except ImportError as e:
    logger.warning(f"Path resolver not available, falling back to default paths: {e}")
    path_resolver = None

# Logger already configured above
# No need to reconfigure

# Import required modules directly using importlib to handle hyphenated directory names
import importlib.util
from pymilvus import connections, Collection, utility

def import_module(file_path, module_name):
    """Import a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import necessary modules
schema_processor = import_module(
    os.path.join(os.path.dirname(__file__), "schema_processor.py"),
    "schema_processor"
)

# Import classes directly
SchemaProcessor = schema_processor.SchemaProcessor

# Import reranker if available
try:
    reranker_module = import_module(
        os.path.join(os.path.dirname(__file__), "rag_reranker.py"),
        "rag_reranker"
    )
    get_reranker = reranker_module.get_reranker
    logger.info("Reranker module loaded successfully")
    RERANKER_AVAILABLE = True
except Exception as e:
    logger.warning(f"Reranker module not available: {e}")
    RERANKER_AVAILABLE = False

# Use path resolver for consistent paths if available
if path_resolver is not None:
    try:
        CLIENT_REGISTRY = path_resolver.get_client_registry_path()
        COLUMN_MAPPINGS = path_resolver.get_column_mappings_path()
        logger.info(f"Using client registry at: {CLIENT_REGISTRY}")
        logger.info(f"Using column mappings at: {COLUMN_MAPPINGS}")
    except Exception as e:
        logger.warning(f"Error using path resolver: {e}, falling back to default paths")
        # Constants (fallback)
        CLIENT_REGISTRY = os.path.join(parent_dir, "config", "clients", "client_registry.csv")
        COLUMN_MAPPINGS = os.path.join(parent_dir, "config", "column_mappings.json")
else:
    # Constants (fallback)
    CLIENT_REGISTRY = os.path.join(parent_dir, "config", "clients", "client_registry.csv")
    COLUMN_MAPPINGS = os.path.join(parent_dir, "config", "column_mappings.json")
    logger.warning(f"Path resolver not available, using default paths:\nClient Registry: {CLIENT_REGISTRY}\nColumn Mappings: {COLUMN_MAPPINGS}")

class RAGManager:
    """Complete RAG solution for multi-client systems"""
    
    def __init__(self, milvus_host="localhost", milvus_port="19530", enable_reranking=True):
        """Initialize the RAG complete solution"""
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.schema_processor = SchemaProcessor()
        self.enable_reranking = enable_reranking
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        # Load embedding model
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        logger.info("Loaded embedding model: BAAI/bge-large-en-v1.5")
        
        logger.info(f"Initialized RAG Manager with Milvus at {milvus_host}:{milvus_port}")
    
    def _connect_to_milvus(self):
        """Connect to Milvus server"""
        try:
            # Get host and port from environment variables if available
            milvus_host = os.environ.get("MILVUS_HOST", self.milvus_host)
            milvus_port = os.environ.get("MILVUS_PORT", self.milvus_port)
            
            logger.info(f"Attempting to connect to Milvus at {milvus_host}:{milvus_port}")
            connections.connect(alias="default", host=milvus_host, port=milvus_port)
            logger.info(f"Successfully connected to Milvus at {milvus_host}:{milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def get_clients(self):
        """Get all clients from registry"""
        clients = []
        
        try:
            if not os.path.exists(CLIENT_REGISTRY):
                logger.error(f"Client registry not found: {CLIENT_REGISTRY}")
                return []
                
            with open(CLIENT_REGISTRY, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    client_id = row.get('client_id', '')
                    dict_path = row.get('data_dictionary_path', '')
                    active = row.get('active', 'true').lower() == 'true'
                    
                    clients.append({
                        'client_id': client_id,
                        'dict_path': dict_path,
                        'active': active
                    })
                    
            return clients
        except Exception as e:
            logger.error(f"Error reading client registry: {e}")
            return []
    
    def get_active_clients(self):
        """Get active clients"""
        clients = self.get_clients()
        active_clients = []
        
        for c in clients:
            if c['active']:
                client_id = c['client_id']
                # Try to use path resolver for consistent paths
                try:
                    if path_resolver is not None:
                        dict_path = path_resolver.get_client_dictionary_path(client_id)
                        logger.info(f"Using resolved path for {client_id}: {dict_path}")
                    else:
                        dict_path = c['dict_path']
                except Exception as e:
                    logger.warning(f"Could not resolve path for {client_id}, using original: {e}")
                    dict_path = c['dict_path']
                    
                active_clients.append((client_id, dict_path))
                
        return active_clients
    
    def _get_collection_name(self, client_id):
        """Get collection name for a client"""
        return f"{client_id}_schema_collection"
    
    def drop_collections(self, client_id=None):
        """
        Drop collections for a specific client or all collections
        
        Args:
            client_id: Optional client ID to drop collection for
        
        Returns:
            Tuple of (success, message)
        """
        try:
            collections = utility.list_collections()
            
            if not collections:
                return True, "No collections found in Milvus"
            
            dropped = 0
            
            # Drop collections
            for coll_name in collections:
                # Only drop the specific client's collection if specified
                if client_id and not coll_name.startswith(f"{client_id}_"):
                    continue
                    
                utility.drop_collection(coll_name)
                logger.info(f"Dropped collection: {coll_name}")
                dropped += 1
                
            return True, f"Successfully dropped {dropped} collections"
        except Exception as e:
            error_msg = f"Failed to drop collections: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def process_schema_data(self, client_id, dict_path):
        """Process schema data for a client"""
        try:
            # Use path resolver if available to get consistent path
            try:
                if path_resolver is not None:
                    resolved_path = path_resolver.get_client_dictionary_path(client_id)
                    logger.info(f"Path resolver found, using resolved path: {resolved_path}")
                    dict_path = resolved_path
            except Exception as e:
                logger.warning(f"Could not use path resolver for dictionary path: {e}")
                
            # Process schema CSV - ensure we pass the actual path, not just client_id
            logger.info(f"Loading CSV from path: {dict_path}")
            records = self.schema_processor.process_csv_data(client_id, dict_path)
            
            if not records:
                error_msg = f"Failed to process schema data for client {client_id}"
                logger.error(error_msg)
                return False, error_msg, None
            
            # Schema processor returns a list of records directly
            schema_data = records
            total_rows = len(records)
            return True, f"Processed {total_rows} schema records", schema_data
        except Exception as e:
            error_msg = f"Error processing schema data: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return False, error_msg, None
    
    def create_collection(self, client_id, schema_data):
        """Create Milvus collection for a client"""
        try:
            print(f"\nCreating collection for client: {client_id}", flush=True)
            print(f"Number of schema records: {len(schema_data)}", flush=True)
            
            collection_name = self._get_collection_name(client_id)
            print(f"Collection name: {collection_name}", flush=True)
            
            # Check if collection exists and drop if necessary
            if utility.has_collection(collection_name):
                print(f"Dropping existing collection: {collection_name}", flush=True)
                utility.drop_collection(collection_name)
                logger.info(f"Dropped existing collection: {collection_name}")
            
            # Prepare schema
            from pymilvus import FieldSchema, CollectionSchema, DataType
            print("Creating collection schema...", flush=True)
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="db_schema", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="column_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="distinct_values", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="combined_text", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]
            
            print("Creating schema object...", flush=True)
            schema = CollectionSchema(fields, "Schema collection for " + client_id)
            print("Creating collection...", flush=True)
            collection = Collection(collection_name, schema)
            
            # Determine appropriate index based on dataset size
            print("Determining optimal index type based on dataset size...", flush=True)
            dataset_size = len(schema_data)
            
            # Dynamic index selection based on size
            if dataset_size < 100000:  # Less than 100k rows
                print(f"Dataset size: {dataset_size} rows - Using HNSW index", flush=True)
                index_params = {
                    "metric_type": "IP",
                    "index_type": "HNSW",
                    "params": {
                        "M": 16,  # Number of edges per node (higher = better recall but more memory)
                        "efConstruction": 200  # Higher values give better index quality but slower build
                    }
                }
                logger.info(f"Using HNSW index for collection with {dataset_size} rows")
            else:  # 100k rows or more
                print(f"Dataset size: {dataset_size} rows - Using IVF_FLAT index", flush=True)
                # For larger datasets, IVF_FLAT with optimized nlist values based on dataset size
                # Following recommended values:
                # For ~10K vectors: nlist = 100
                # For ~1M vectors: nlist = 4096
                if dataset_size < 50000:  # Less than 50K
                    nlist = 100
                elif dataset_size < 500000:  # Between 50K and 500K
                    nlist = 1024
                else:  # 500K or more
                    nlist = 4096
                index_params = {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": nlist}
                }
                logger.info(f"Using IVF_FLAT index with nlist={nlist} for collection with {dataset_size} rows")
            
            # Create the index with the determined parameters
            print("Creating index...", flush=True)
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            print("Collection and index created successfully", flush=True)
            logger.info(f"Created collection: {collection_name}")
            
            # Generate embeddings and insert records
            print("\nPreparing data for insertion...", flush=True)
            data = []
            
            for i, record in enumerate(schema_data):
                if i % 20 == 0:  # Print progress every 20 records
                    print(f"Processing record {i+1}/{len(schema_data)}", flush=True)
                
                # Generate combined text for embedding
                combined_text = record.generate_combined_text()
                
                # Generate embedding
                embedding = self.model.encode(combined_text).tolist()
                
                # Prepare record for insertion with proper field limits
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
            
            # Insert data in batches
            print(f"\nInserting {len(data)} records in batches...", flush=True)
            batch_size = 50
            count = 0
            
            try:
                from tqdm import tqdm
                for i in range(0, len(data), batch_size):
                    print(f"Inserting batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}", flush=True)
                    batch = data[i:i+batch_size]
                    collection.insert(batch)
                    count += len(batch)
                print(f"\nInserted {count} records into collection", flush=True)
            except Exception as e:
                print(f"Error during batch insertion: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                raise
            
            # Load collection
            print("Loading collection...", flush=True)
            collection.flush()
            collection.load()
            print("Collection loaded successfully", flush=True)
            
            count = collection.num_entities
            print(f"Collection has {count} entities", flush=True)
            
            return True, f"Collection created with {count} entities"
        except Exception as e:
            error_msg = f"Error creating collection: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return False, error_msg
    
    def rebuild_client(self, client_id, dict_path=None):
        """Rebuild collection for a specific client"""
        try:
            logger.info(f"Rebuilding collection for client {client_id}")
            
            # If dict_path not provided, look it up from client registry
            if dict_path is None:
                clients = self.get_active_clients()
                client_dict = {c_id: path for c_id, path in clients}
                
                if client_id not in client_dict:
                    error_msg = f"Client {client_id} not found in registry"
                    logger.error(error_msg)
                    return False, error_msg
                    
                dict_path = client_dict[client_id]
                logger.info(f"Using dictionary path from registry: {dict_path}")
            
            if not os.path.exists(dict_path):
                error_msg = f"Dictionary not found: {dict_path}"
                logger.error(error_msg)
                return False, error_msg
            
            # Process schema data
            success, message, schema_data = self.process_schema_data(client_id, dict_path)
            
            if not success:
                return False, message
            
            # Create collection
            return self.create_collection(client_id, schema_data)
            
        except Exception as e:
            error_msg = f"Error rebuilding collection: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return False, error_msg
    
    def rebuild_all(self):
        """Rebuild all active client collections"""
        results = {}
        clients = self.get_active_clients()
        
        for client_id, dict_path in clients:
            print(f"Rebuilding collection for client: {client_id}", flush=True)
            success, message = self.rebuild_client(client_id, dict_path)
            
            results[client_id] = {
                'success': success,
                'message': message
            }
            
            # Print progress
            if success:
                print(f"✓ {client_id}: {message}", flush=True)
            else:
                print(f"✗ {client_id}: {message}", flush=True)
            
        return results
    
    def query(self, client_id, query_text, top_k=5):
        """Execute a RAG query for a client"""
        try:
            print(f"Executing query for client {client_id}: '{query_text}'", flush=True)
            
            # Check if client exists
            collection_name = self._get_collection_name(client_id)
            if not utility.has_collection(collection_name):
                return False, f"Client collection {collection_name} not found", []
            
            # Load collection
            collection = Collection(collection_name)
            try:
                collection.load()
            except Exception as e:
                logger.warning(f"Error loading collection (may already be loaded): {e}")
            
            # Generate embedding for query
            query_embedding = self.model.encode(query_text)
            
            # Search for similar records
            # Get collection info to determine which index type is being used
            collection_info = collection.describe()
            index_info = collection.index().params
            
            # Configure search parameters based on the index type
            if index_info["index_type"] == "HNSW":
                search_params = {
                    "metric_type": "IP",  # Inner Product similarity - must match index metric_type
                    "params": {"ef": 64}  # Query-time recall quality parameter for HNSW index
                }
                logger.info("Using HNSW search parameters with ef=64")
            else:  # IVF_FLAT
                # Determine appropriate nprobe value based on nlist in the index
                nlist = index_info["params"]["nlist"]
                # Recommended values: For 10K vectors with nlist=100, use nprobe=10
                #                     For 1M vectors with nlist=4096, use nprobe=10-100
                if nlist <= 100:
                    nprobe = 8  # Using optimal value from memory
                elif nlist <= 1024:
                    nprobe = 16
                else:
                    nprobe = 32
                
                search_params = {
                    "metric_type": "IP", 
                    "params": {"nprobe": nprobe}
                }
                logger.info(f"Using IVF_FLAT search parameters with nprobe={nprobe}")
            
            # Retrieve more results than needed if reranker is available
            search_limit = top_k * 3 if RERANKER_AVAILABLE else top_k
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=search_limit,
                output_fields=["db_schema", "table_name", "column_name", "data_type", 
                              "description", "distinct_values", "combined_text"]
            )
            
            # Process results with proper entity data extraction
            matches = []
            
            if results and len(results) > 0 and len(results[0]) > 0:
                for hit in results[0]:
                    # Create a match dictionary with default empty values
                    match = {
                        'table_name': '',
                        'column_name': '',
                        'db_schema': '',
                        'data_type': '',
                        'description': '',
                        'distinct_values': '',
                        'combined_text': '',
                        'score': 0.0
                    }
                    
                    # Proper entity data extraction
                    try:
                        # Get the score first
                        match['score'] = float(hit.score) if hasattr(hit, 'score') else 0.0
                        
                        # Extract entity data
                        entity_dict = hit.entity
                        
                        # Handle different entity formats
                        if isinstance(entity_dict, dict):
                            # Direct dictionary access
                            for field in match.keys():
                                if field != 'score' and field in entity_dict:
                                    value = entity_dict[field]
                                    match[field] = str(value) if value is not None else ''
                        else:
                            # Handle entity as object with get method or attributes
                            for field in match.keys():
                                if field != 'score':
                                    try:
                                        if hasattr(entity_dict, 'get'):
                                            value = entity_dict.get(field, '')
                                        elif hasattr(entity_dict, field):
                                            value = getattr(entity_dict, field)
                                        else:
                                            value = ''
                                        match[field] = str(value) if value is not None else ''
                                    except Exception as e:
                                        logger.debug(f"Error extracting field {field}: {e}")
                                        match[field] = ''
                        
                        # Add match to results list
                        matches.append(match)
                            
                    except Exception as e:
                        logger.error(f"Error processing hit: {e}")
                        logger.error(f"Hit type: {type(hit)}")
                        logger.error(f"Hit dir: {dir(hit)}")
                        continue
            
            # Apply reranking if available AND enabled
            if RERANKER_AVAILABLE and self.enable_reranking and len(matches) > 0:
                try:
                    logger.info(f"Applying Qwen3 reranking to {len(matches)} matches")
                    reranker = get_reranker(self.enable_reranking)
                    reranked_matches = reranker.rerank(query_text, matches, enable_reranking=self.enable_reranking)
                    
                    # Limit to original top_k after reranking
                    matches = reranked_matches[:top_k]
                    logger.info(f"Reranking complete, returning top {top_k} matches")
                except Exception as e:
                    logger.error(f"Error during reranking, falling back to original matches: {e}")
                    # Fallback to original top_k matches
                    matches = matches[:top_k]
            else:
                # Limit to top_k if no reranking was applied
                matches = matches[:top_k]
                if not self.enable_reranking:
                    logger.info(f"Reranking disabled, returning top {top_k} matches based on vector similarity")
                    
            return True, f"Found {len(matches)} relevant matches", matches
            
        except Exception as e:
            error_msg = f"Error executing query: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return False, error_msg, []
    
    def enhanced_query(self, client_id, query_text, top_k=10):
        """Enhanced RAG query that provides table-level context for SQL generation"""
        try:
            print(f"Executing enhanced query for client {client_id}: '{query_text}'", flush=True)
            
            # Check if client exists
            collection_name = self._get_collection_name(client_id)
            if not utility.has_collection(collection_name):
                return False, f"Client collection {collection_name} not found", [], {}
            
            # Load collection
            collection = Collection(collection_name)
            try:
                collection.load()
            except Exception as e:
                logger.warning(f"Error loading collection (may already be loaded): {e}")
            
            # Generate embedding for query
            query_embedding = self.model.encode(query_text)
            
            # Search for similar records
            # Get collection info to determine which index type is being used
            collection_info = collection.describe()
            index_info = collection.index().params
            
            # Configure search parameters based on the index type
            if index_info["index_type"] == "HNSW":
                search_params = {
                    "metric_type": "IP",  # Inner Product similarity - must match index metric_type
                    "params": {"ef": 64}  # Query-time recall quality parameter for HNSW index
                }
                logger.info("Using HNSW search parameters with ef=64")
            else:  # IVF_FLAT
                # Determine appropriate nprobe value based on nlist in the index
                nlist = index_info["params"]["nlist"]
                # Recommended values: For 10K vectors with nlist=100, use nprobe=10
                #                     For 1M vectors with nlist=4096, use nprobe=10-100
                if nlist <= 100:
                    nprobe = 8  # Using optimal value from memory
                elif nlist <= 1024:
                    nprobe = 16
                else:
                    nprobe = 32
                
                search_params = {
                    "metric_type": "IP", 
                    "params": {"nprobe": nprobe}
                }
                logger.info(f"Using IVF_FLAT search parameters with nprobe={nprobe}")
            
            # Retrieve more results than needed if reranker is available
            search_limit = top_k * 3 if RERANKER_AVAILABLE else top_k
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=search_limit,
                output_fields=["db_schema", "table_name", "column_name", "data_type", 
                              "description", "distinct_values", "combined_text"]
            )
            
            # FIXED: Process results with proper entity data extraction
            matches = []
            table_context = {}
            
            if results and len(results) > 0 and len(results[0]) > 0:
                for hit in results[0]:
                    # Create a match dictionary with default empty values
                    match = {
                        'table_name': '',
                        'column_name': '',
                        'db_schema': '',
                        'data_type': '',
                        'description': '',
                        'distinct_values': '',
                        'combined_text': '',
                        'score': 0.0
                    }
                    
                    # FIXED: Proper entity data extraction
                    try:
                        # Get the score first
                        match['score'] = float(hit.score) if hasattr(hit, 'score') else 0.0
                        
                        # Extract entity data - this is the key fix
                        entity_dict = hit.entity
                        
                        # Handle different entity formats
                        if isinstance(entity_dict, dict):
                            # Direct dictionary access
                            for field in match.keys():
                                if field != 'score' and field in entity_dict:
                                    value = entity_dict[field]
                                    match[field] = str(value) if value is not None else ''
                        else:
                            # Handle entity as object with get method or attributes
                            for field in match.keys():
                                if field != 'score':
                                    try:
                                        if hasattr(entity_dict, 'get'):
                                            value = entity_dict.get(field, '')
                                        elif hasattr(entity_dict, field):
                                            value = getattr(entity_dict, field)
                                        else:
                                            value = ''
                                        match[field] = str(value) if value is not None else ''
                                    except Exception as e:
                                        logger.debug(f"Error extracting field {field}: {e}")
                                        match[field] = ''
                        
                        # Add match to results list
                        matches.append(match)
                        
                        # Build table context for SQL generation
                        if match['db_schema'] and match['table_name']:
                            table_key = f"{match['db_schema']}.{match['table_name']}"
                            if table_key not in table_context:
                                table_context[table_key] = {
                                    'columns': [],
                                    'total_relevance': 0,
                                    'column_count': 0
                                }
                            
                            table_context[table_key]['columns'].append({
                                'name': match['column_name'],
                                'type': match['data_type'],
                                'description': match['description'],
                                'distinct_values': match['distinct_values'],
                                'relevance': match['score']
                            })
                            table_context[table_key]['total_relevance'] += match['score']
                            table_context[table_key]['column_count'] += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing hit: {e}")
                        logger.error(f"Hit type: {type(hit)}")
                        logger.error(f"Hit dir: {dir(hit)}")
                        continue
            
            # Apply reranking if available AND enabled
            if RERANKER_AVAILABLE and self.enable_reranking and len(matches) > 0:
                try:
                    logger.info(f"Applying Qwen3 reranking to {len(matches)} matches")
                    reranker = get_reranker(self.enable_reranking)
                    reranked_matches = reranker.rerank(query_text, matches, enable_reranking=self.enable_reranking)
                    
                    # Limit to original top_k after reranking
                    matches = reranked_matches[:top_k]
                    logger.info(f"Reranking complete, returning top {top_k} matches")
                except Exception as e:
                    logger.error(f"Error during reranking, falling back to original matches: {e}")
                    # Fallback to original top_k matches
                    matches = matches[:top_k]
            else:
                # Limit to top_k if no reranking was applied
                matches = matches[:top_k]
                if not self.enable_reranking:
                    logger.info(f"Reranking disabled, returning top {top_k} matches based on vector similarity")
            
            # Generate SQL context
            sql_context = self._generate_sql_context(query_text, table_context)
                    
            return True, f"Found {len(matches)} relevant matches", matches, sql_context
            
        except Exception as e:
            error_msg = f"Error executing enhanced query: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return False, error_msg, [], {}
    
    def _generate_sql_context(self, query_text, table_context):
        """Generate SQL context for query generation"""
        sql_context = {
            'query_intent': self._analyze_query_intent(query_text),
            'recommended_tables': [],
            'column_suggestions': {}
        }
        
        # Sort tables by relevance
        sorted_tables = sorted(
            table_context.items(), 
            key=lambda x: x[1]['total_relevance'], 
            reverse=True
        )
        
        # Recommend top tables
        for table_name, context in sorted_tables[:3]:
            table_info = {
                'full_name': table_name,
                'relevance_score': context['total_relevance'],
                'column_count': context['column_count'],
                'columns': sorted(
                    context['columns'], 
                    key=lambda x: x['relevance'], 
                    reverse=True
                )
            }
            sql_context['recommended_tables'].append(table_info)
            
            # Categorize columns by likely SQL usage
            sql_context['column_suggestions'][table_name] = self._categorize_columns(
                context['columns'], query_text
            )
        
        return sql_context
    
    def _analyze_query_intent(self, query_text):
        """Analyze query intent for SQL generation"""
        query_lower = query_text.lower()
        
        intent = {
            'operation': 'SELECT',  # Default
            'aggregation': None,
            'filtering': [],
            'sorting': None,
            'grouping': None,
            'limit': None
        }
        
        # Detect aggregations
        if any(word in query_lower for word in ['top', 'best', 'highest', 'most', 'sum', 'total', 'count']):
            intent['aggregation'] = 'SUM'
            intent['sorting'] = 'DESC'
            intent['grouping'] = True
        
        # Detect filtering
        if any(word in query_lower for word in ['2024', '2023', 'year', 'month']):
            intent['filtering'].append('date/year')
        
        # Detect limits
        if 'top' in query_lower:
            # Extract number after 'top'
            import re
            match = re.search(r'top\s+(\d+)', query_lower)
            if match:
                intent['limit'] = int(match.group(1))
        
        return intent
    
    def _categorize_columns(self, columns, query_text):
        """Categorize columns by SQL usage pattern"""
        categories = {
            'select_columns': [],      # Columns to SELECT
            'aggregate_columns': [],   # Columns to SUM/COUNT
            'filter_columns': [],      # Columns for WHERE
            'group_columns': [],       # Columns for GROUP BY
            'order_columns': []        # Columns for ORDER BY
        }
        
        query_lower = query_text.lower()
        
        for col in columns:
            col_name = col['name'].lower()
            col_desc = col['description'].lower() if col['description'] else ''
            
            # Detect quantity/amount columns for aggregation
            if any(word in col_name for word in ['quantity', 'amount', 'total', 'sum', 'count']):
                categories['aggregate_columns'].append(col)
            
            # Detect category/name columns for grouping
            if any(word in col_name for word in ['name', 'category', 'class', 'type']):
                categories['select_columns'].append(col)
                categories['group_columns'].append(col)
            
            # Detect date/year columns for filtering
            if any(word in col_name for word in ['year', 'date', 'time', 'month']):
                categories['filter_columns'].append(col)
            
            # Detect ID columns
            if 'id' in col_name:
                categories['filter_columns'].append(col)
        
        return categories
        
    def get_collection_stats(self):
        """Get statistics for all collections"""
        try:
            collections = utility.list_collections()
            stats = []
            
            for coll_name in collections:
                try:
                    collection = Collection(coll_name)
                    count = collection.num_entities
                    stats.append({
                        'name': coll_name,
                        'count': count,
                        'client_id': coll_name.split('_')[0] if '_' in coll_name else 'unknown'
                    })
                except Exception as e:
                    stats.append({
                        'name': coll_name,
                        'error': str(e),
                        'client_id': coll_name.split('_')[0] if '_' in coll_name else 'unknown'
                    })
            
            return True, stats
        except Exception as e:
            error_msg = f"Error getting collection stats: {e}"
            logger.error(error_msg)
            return False, error_msg

def print_header(title):
    """Print a formatted header"""
    header = "\n" + "="*70 + f"\n {title}\n" + "="*70
    print(header, flush=True)

def print_result(success, message, indent=0):
    """Print a formatted result"""
    prefix = " " * indent
    symbol = "✓" if success else "✗"
    print(f"{prefix}{symbol} {message}", flush=True)

def display_matches(matches):
    """Display query matches with proper formatting"""
    if not matches:
        print("No matches found", flush=True)
        return
    
    print(f"\nFound {len(matches)} relevant schema elements:\n", flush=True)
    
    for i, match in enumerate(matches):
        # Extract fields with fallbacks
        table = match.get('table_name', 'Unknown')
        column = match.get('column_name', '')
        schema = match.get('db_schema', '')
        data_type = match.get('data_type', '')
        description = match.get('description', '')
        distinct_values = match.get('distinct_values', '')
        score = match.get('score', 0.0)
        
        # Format display
        score_display = f"{score:.4f}" if isinstance(score, float) else "N/A"
        
        # Format full identifier
        if schema and table and column:
            identifier = f"{schema}.{table}.{column}"
        elif table and column:
            identifier = f"{table}.{column}"
        elif table:
            identifier = table
        else:
            identifier = "Unknown"
        
        # Print formatted result
        print(f"{i+1}. {identifier}", flush=True)
        if data_type:
            print(f"   Type: {data_type}", flush=True)
        if description:
            print(f"   Description: {description}", flush=True)
        if distinct_values:
            print(f"   Values: {distinct_values[:100]}{'...' if len(distinct_values) > 100 else ''}", flush=True)
        print(f"   Relevance: {score_display}", flush=True)
        print()

def main():
    """Main entry point"""
    # Create log directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Fixed RAG Manager - Complete solution for multi-client Milvus RAG",
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Drop command
    drop_parser = subparsers.add_parser("drop", help="Drop collections")
    drop_parser.add_argument("--client", "-c", required=False, help="Specific client to drop collection for")
    drop_parser.add_argument("--all", "-a", action="store_true", help="Drop all client collections")
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild collections")
    rebuild_parser.add_argument("--client", "-c", required=False, help="Specific client to rebuild")
    rebuild_parser.add_argument("--all", "-a", action="store_true", help="Rebuild all active clients")
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query schema collection')
    query_parser.add_argument("--client", "-c", required=True, help="Client ID for the collection")
    query_parser.add_argument("--query", "-q", required=True, help="Query text")
    query_parser.add_argument("--top", "-k", type=int, default=10, help="Number of results to return")

    # Enhanced query command
    enhanced_parser = subparsers.add_parser('enhanced', help='Enhanced query with SQL context generation')
    enhanced_parser.add_argument("--client", "-c", required=True, help="Client ID for the collection")
    enhanced_parser.add_argument("--query", "-q", required=True, help="Query text")
    enhanced_parser.add_argument("--top", "-k", type=int, default=10, help="Number of results to return")

    # Debug query command
    debug_parser = subparsers.add_parser('debug', help='Debug query results with detailed entity inspection')
    debug_parser.add_argument("--client", "-c", required=True, help="Client ID for the collection")
    debug_parser.add_argument("--query", "-q", required=True, help="Query text")
    debug_parser.add_argument("--top", "-k", type=int, default=5, help="Number of results to return")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show collection statistics")
    
    # Helper function to display SQL context
    def display_sql_context(sql_context):
        """Display SQL context in a user-friendly format"""
        print("\n===== SQL CONTEXT GENERATION =====\n", flush=True)
        
        # Display query intent
        print("Query Intent Analysis:", flush=True)
        intent = sql_context['query_intent']
        print(f"  - Operation: {intent['operation']}", flush=True)
        if intent['aggregation']:
            print(f"  - Aggregation: {intent['aggregation']}", flush=True)
        if intent['sorting']:
            print(f"  - Sorting: {intent['sorting']}", flush=True)
        if intent['grouping']:
            print(f"  - Grouping: {'Yes' if intent['grouping'] else 'No'}", flush=True)
        if intent['filtering']:
            print(f"  - Filtering: {', '.join(intent['filtering'])}", flush=True)
        if intent['limit'] is not None:
            print(f"  - Limit: {intent['limit']}", flush=True)
        
        # Display recommended tables
        print("\nRecommended Tables:", flush=True)
        for i, table in enumerate(sql_context['recommended_tables'], 1):
            print(f"  {i}. {table['full_name']} (Relevance: {table['relevance_score']:.4f}, Columns: {table['column_count']})", flush=True)
            
            # Display column suggestions for this table
            table_name = table['full_name']
            if table_name in sql_context['column_suggestions']:
                suggestions = sql_context['column_suggestions'][table_name]
                
                if suggestions['select_columns']:
                    print(f"     SELECT columns:", flush=True)
                    for col in suggestions['select_columns'][:3]:  # Top 3
                        print(f"       - {col['name']} ({col['type']})", flush=True)
                
                if suggestions['aggregate_columns']:
                    print(f"     Aggregate columns:", flush=True)
                    for col in suggestions['aggregate_columns'][:3]:  # Top 3
                        print(f"       - {col['name']} ({col['type']})", flush=True)
                
                if suggestions['filter_columns']:
                    print(f"     Filter columns:", flush=True)
                    for col in suggestions['filter_columns'][:3]:  # Top 3
                        print(f"       - {col['name']} ({col['type']})", flush=True)
        
        print("\n==================================\n", flush=True)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set debug logging if needed
    if args.command == "debug":
        logging.getLogger("rag_embedding").setLevel(logging.DEBUG)
        print("Debug logging enabled", flush=True)
    
    # Create RAG manager
    rag_manager = RAGManager()
    
    # Handle commands
    if args.command == "drop":
        if not args.client and not args.all:
            print("Error: Either --client or --all must be specified", flush=True)
            return
            
        if args.all:
            print("Dropping all client collections...", flush=True)
            success, message = rag_manager.drop_collections(None)  # None means all collections
            print(message, flush=True)
        else:
            client_id = args.client
            print(f"Dropping collections for client: {client_id}", flush=True)
            success, message = rag_manager.drop_collections(client_id)
            print(message, flush=True)

    elif args.command == "rebuild":
        if not args.client and not args.all:
            print("Error: Either --client or --all must be specified", flush=True)
            return
            
        # Get active clients
        clients = rag_manager.get_active_clients()
        client_dict = {c_id: path for c_id, path in clients}
        
        if args.all:
            print(f"\nStarting rebuild for ALL active clients ({len(client_dict)} found)", flush=True)
            
            # Track results
            successful = []
            failed = []
            
            # Process each client
            for client_id, dict_path in client_dict.items():
                print(f"\n===== Rebuilding client: {client_id} =====", flush=True)
                print(f"Using dictionary path: {dict_path}", flush=True)
                
                if not os.path.exists(dict_path):
                    print(f"Error: Dictionary file not found at {dict_path}", flush=True)
                    failed.append((client_id, "Dictionary file not found"))
                    continue
                
                # Run rebuild
                success, message = rag_manager.rebuild_client(client_id, dict_path)
                print(message, flush=True)
                
                if success:
                    successful.append(client_id)
                else:
                    failed.append((client_id, message))
            
            # Summary
            print(f"\n===== REBUILD SUMMARY =====", flush=True)
            print(f"Successfully rebuilt {len(successful)} clients: {', '.join(successful)}", flush=True)
            if failed:
                print(f"Failed to rebuild {len(failed)} clients:", flush=True)
                for client_id, error in failed:
                    print(f"  - {client_id}: {error}", flush=True)
        else:
            client_id = args.client
            print(f"\nStarting rebuild for client: {client_id}", flush=True)
            
            if client_id not in client_dict:
                print(f"Error: Client {client_id} not found in registry", flush=True)
                return
                
            dict_path = client_dict[client_id]
            print(f"Using dictionary path: {dict_path}", flush=True)
            
            if not os.path.exists(dict_path):
                print(f"Error: Dictionary file not found at {dict_path}", flush=True)
                return
            
            # Run rebuild
            success, message = rag_manager.rebuild_client(client_id, dict_path)
            print(message, flush=True)

    elif args.command == "query":
        client_id = args.client
        query_text = args.query
        top_k = args.top
        
        print(f"Executing query for client {client_id}: '{query_text}'", flush=True)
        success, message, results = rag_manager.query(client_id, query_text, top_k)
        
        print(message, flush=True)
        if success and results:
            print(f"Top {len(results)} results:", flush=True)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Table: {result['table_name']} | Column: {result['column_name']} | Score: {result['score']:.4f}", flush=True)
                if result['description']:
                    print(f"   Description: {result['description']}", flush=True)
                    
    elif args.command == "enhanced":
        client_id = args.client
        query_text = args.query
        top_k = args.top
        
        print(f"Executing enhanced query for client {client_id}: '{query_text}'", flush=True)
        success, message, results, sql_context = rag_manager.enhanced_query(client_id, query_text, top_k)
        
        print(message, flush=True)
        if success and results:
            # Display top results
            print(f"Top {min(5, len(results))} results:", flush=True)
            for i, result in enumerate(results[:5], 1):  # Show only top 5 for readability
                print(f"\n{i}. Table: {result['table_name']} | Column: {result['column_name']} | Score: {result['score']:.4f}", flush=True)
                if result['description']:
                    print(f"   Description: {result['description']}", flush=True)
            
            # Display SQL context if available
            if sql_context:
                display_sql_context(sql_context)
                
    elif args.command == "debug":
        client_id = args.client
        query_text = args.query
        top_k = args.top
        
        print(f"[DEBUG] Executing query for client {client_id}: '{query_text}'", flush=True)
        success, message, results = rag_manager.query(client_id, query_text, top_k)
        
        print(message, flush=True)
        if success and results:
            print("\nDetailed Debug Results:", flush=True)
            for i, result in enumerate(results, 1):
                print(f"\nResult #{i}:", flush=True)
                for key, value in result.items():
                    print(f"   {key}: {value}", flush=True)
                    
    elif args.command == "stats":
        print("Collecting statistics for all collections...", flush=True)
        success, stats = rag_manager.get_collection_stats()
        
        if success:
            print("Collection statistics:", flush=True)
            for stat in stats:
                if 'error' in stat:
                    print(f"  {stat['name']} (Client: {stat['client_id']}): ERROR - {stat['error']}", flush=True)
                else:
                    print(f"  {stat['name']} (Client: {stat['client_id']}): {stat['count']} entities", flush=True)
            
            if not stats:
                print("No collections found")
            else:
                # Calculate total entities and per-client counts
                total_count = sum(item.get('count', 0) for item in stats if 'count' in item)
                client_counts = {}
                for item in stats:
                    client_id = item.get('client_id', 'unknown')
                    count = item.get('count', 0)
                    client_counts[client_id] = client_counts.get(client_id, 0) + count
                
                # Print summary
                print(f"\nTotal entities across all collections: {total_count}", flush=True)
                print("Entities per client:", flush=True)
                for client_id, count in sorted(client_counts.items()):
                    print(f"  {client_id}: {count} entities", flush=True)
        else:
            print(f"Error: {stats}", flush=True)
    else:
        # No command or unknown command
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
