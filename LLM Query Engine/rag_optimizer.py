"""
RAG Optimizer for LLM Query Engine

This module provides functions for optimizing LLM token usage and enabling query reuse
through Retrieval-Augmented Generation (RAG) techniques.

It uses sentence-transformers for generating embeddings and ChromaDB for the vector database,
both of which are open-source alternatives to proprietary solutions.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import hashlib
from datetime import datetime
from pathlib import Path

# Import ChromaDB for vector storage
import chromadb
from chromadb.utils import embedding_functions

# Flag to track if we're using fallback mode (without sentence-transformers)
USE_FALLBACK_MODE = False

# Try to import sentence-transformers for open-source embeddings
try:
    from sentence_transformers import SentenceTransformer
    print("Using sentence-transformers for embeddings")
except ImportError as e:
    print(f"Warning: Could not import sentence-transformers: {str(e)}")
    print("Falling back to chromadb default embedding function")
    USE_FALLBACK_MODE = True

# Initialize directories for RAG data storage
RAG_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG_Data")
QUERY_CACHE_DIR = os.path.join(RAG_DATA_DIR, "query_cache")
SCHEMA_VECTORS_DIR = os.path.join(RAG_DATA_DIR, "schema_vectors")

# Create directories if they don't exist
for dir_path in [RAG_DATA_DIR, QUERY_CACHE_DIR, SCHEMA_VECTORS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=RAG_DATA_DIR)

# Initialize the embedding function based on availability
if not USE_FALLBACK_MODE:
    # Using sentence-transformers with all-MiniLM-L6-v2, a lightweight but effective model
    try:
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("Successfully initialized sentence-transformers model")
        # Custom embedding function using our model
        def embedding_function(texts):
            return EMBEDDING_MODEL.encode(texts).tolist()
    except Exception as e:
        print(f"Error initializing sentence-transformers: {str(e)}")
        print("Falling back to chromadb default embedding function")
        USE_FALLBACK_MODE = True

# If fallback mode, use ChromaDB's default embedding function
if USE_FALLBACK_MODE:
    print("Using ChromaDB's default embedding function")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # Wrapper to match our expected function signature
    def embedding_function(texts):
        return default_ef(texts)

# Create or get collections
def get_schema_collection():
    """Get or create the schema collection for storing table/column embeddings"""
    try:
        if USE_FALLBACK_MODE:
            return chroma_client.get_collection(
                name="schema_embeddings",
                embedding_function=default_ef
            )
        else:
            return chroma_client.get_collection("schema_embeddings")
    except:
        if USE_FALLBACK_MODE:
            return chroma_client.create_collection(
                name="schema_embeddings",
                embedding_function=default_ef,
                metadata={"description": "Embeddings of database schema elements"}
            )
        else:
            return chroma_client.create_collection(
                name="schema_embeddings",
                metadata={"description": "Embeddings of database schema elements"}
            )

def get_query_collection():
    """Get or create the query collection for storing previous queries"""
    try:
        if USE_FALLBACK_MODE:
            return chroma_client.get_collection(
                name="query_history",
                embedding_function=default_ef
            )
        else:
            return chroma_client.get_collection("query_history")
    except:
        if USE_FALLBACK_MODE:
            return chroma_client.create_collection(
                name="query_history",
                embedding_function=default_ef,
                metadata={"description": "Embeddings of previous queries"}
            )
        else:
            return chroma_client.create_collection(
                name="query_history",
                metadata={"description": "Embeddings of previous queries"}
            )

def generate_schema_embeddings(data_dictionary_path: str, force_refresh: bool = False) -> bool:
    """
    Generate and store embeddings for all tables and columns in the data dictionary.
    
    Args:
        data_dictionary_path: Path to the data dictionary CSV/Excel file
        force_refresh: Whether to force regeneration of all embeddings
        
    Returns:
        bool: Success status
    """
    # Check if embeddings already exist and we're not forcing a refresh
    embedding_marker = os.path.join(SCHEMA_VECTORS_DIR, "embeddings_complete.txt")
    if os.path.exists(embedding_marker) and not force_refresh:
        print("Schema embeddings already exist. Use force_refresh=True to regenerate.")
        return True
        
    try:
        # Load data dictionary
        if data_dictionary_path.endswith('.xlsx') or data_dictionary_path.endswith('.xls'):
            df = pd.read_excel(data_dictionary_path)
        elif data_dictionary_path.endswith('.csv'):
            df = pd.read_csv(data_d
            ictionary_path)
        else:
            raise ValueError("Unsupported file format. Please provide an Excel or CSV file.")
        
        # Get schema collection
        schema_collection = get_schema_collection()
        
        # Clear existing embeddings if forcing refresh
        if force_refresh:
            schema_collection.delete(where={})
        
        # Prepare data for embedding
        documents = []
        metadatas = []
        ids = []
        
        # Track tables we've already seen to avoid duplicates
        seen_tables = set()
        row_index = 0
        
        # Process each table and column
        for _, row in df.iterrows():
            row_index += 1  # Unique index for each row
            table_name = row["TABLE_NAME"]
            column_name = row["COLUMN_NAME"]
            description = row["DESCRIPTION"]
            data_type = row.get("DATA_TYPE", "")
            
            # Prepare schema elements with richer context for better retrieval
            # Table context - only add each table once
            if table_name not in seen_tables:
                table_text = f"Table: {table_name} - A table that contains {description}"
                # Make table ID unique by adding a hash
                table_hash = hashlib.md5(table_name.encode()).hexdigest()[:8]
                table_id = f"table_{table_name}_{table_hash}"
                seen_tables.add(table_name)
                
                # Add to lists for batch processing
                documents.append(table_text)
                metadatas.append({"type": "table", "table_name": table_name, "description": description})
                ids.append(table_id)
            
            # Column context - always unique since columns are table-specific
            column_text = f"Column: {column_name} in table {table_name} - {description} (Type: {data_type})"
            column_id = f"column_{table_name}_{column_name}_{row_index}"  # Add row index for uniqueness
            
            # Add column to lists for batch processing
            documents.append(column_text)
            metadatas.append({
                "type": "column", 
                "table_name": table_name, 
                "column_name": column_name, 
                "description": description, 
                "data_type": data_type
            })
            ids.append(column_id)
        
        # Add documents to collection (ChromaDB will handle the embedding)
        if documents:
            schema_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Mark embeddings as complete
            with open(embedding_marker, "w") as f:
                f.write(f"Embeddings generated at {datetime.now().isoformat()}")
                
            print(f"Successfully generated embeddings for {len(documents)} schema elements")
            return True
        else:
            print("No schema elements found to embed")
            return False
            
    except Exception as e:
        print(f"Error generating schema embeddings: {str(e)}")
        return False

def find_relevant_schema(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Find the most relevant tables and columns for a given query.
    
    Args:
        query: Natural language query
        top_k: Number of relevant schema elements to return
        
    Returns:
        Dictionary with relevant tables and columns
    """
    try:
        # Get schema collection
        schema_collection = get_schema_collection()
        
        # Generate query embedding using sentence-transformers
        if not USE_FALLBACK_MODE:
            query_embedding = EMBEDDING_MODEL.encode(query)
            query_embeddings = [query_embedding.tolist()]
        else:
            query_embeddings = None  # ChromaDB will handle this internally
        
        # Query the schema collection
        if not USE_FALLBACK_MODE:
            results = schema_collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = schema_collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        
        # Extract and organize results
        relevant_tables = set()
        relevant_columns = {}
        
        for i, metadata in enumerate(results["metadatas"][0]):
            if metadata["type"] == "table":
                relevant_tables.add(metadata["table_name"])
            elif metadata["type"] == "column":
                table_name = metadata["table_name"]
                if table_name not in relevant_columns:
                    relevant_columns[table_name] = []
                
                relevant_columns[table_name].append({
                    "name": metadata["column_name"],
                    "description": metadata["description"],
                    "data_type": metadata["data_type"],
                    "relevance_score": 1.0 - results["distances"][0][i]  # Convert distance to similarity score
                })
        
        # Format into a usable result
        tables_info = []
        for table in relevant_tables:
            columns = relevant_columns.get(table, [])
            # Sort columns by relevance score
            columns = sorted(columns, key=lambda x: x["relevance_score"], reverse=True)
            
            tables_info.append({
                "name": table,
                "columns": columns
            })
        
        return {
            "query": query,
            "relevant_tables": tables_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error finding relevant schema: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "relevant_tables": []
        }

def format_reduced_data_dictionary(relevant_schema: Dict[str, Any], full_data_dict_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Format reduced data dictionary based on relevant schema for LLM prompt.
    
    Args:
        relevant_schema: Output from find_relevant_schema
        full_data_dict_df: Full data dictionary DataFrame
        
    Returns:
        List of formatted table dictionaries (like the format_data_dictionary function)
    """
    tables = []
    
    # Extract table names from relevant schema
    relevant_table_names = [table["name"] for table in relevant_schema["relevant_tables"]]
    
    # Filter data dictionary for relevant tables
    relevant_df = full_data_dict_df[full_data_dict_df["TABLE_NAME"].isin(relevant_table_names)]
    
    # Add DB_SCHEMA if available
    schema_aware = 'DB_SCHEMA' in relevant_df.columns
    
    # Group by table name (and schema if available)
    if schema_aware:
        # Create a compound key for grouping that includes schema
        relevant_df['FULL_TABLE_NAME'] = relevant_df['DB_SCHEMA'] + '.' + relevant_df['TABLE_NAME']
        groupby_col = 'FULL_TABLE_NAME'
    else:
        groupby_col = 'TABLE_NAME'
    
    for full_table_name, group in relevant_df.groupby(groupby_col):
        columns = []
        for _, row in group.iterrows():
            column_info = {
                'name': row['COLUMN_NAME'],
                # Use a default type if not available
                'type': row.get('DATA_TYPE', 'VARCHAR'),
                'description': row['DESCRIPTION'],
                # Use the column name as business name if not provided
                'business_name': row.get('BUSINESS_NAME', row['COLUMN_NAME']),
                # Handle optional fields with safe defaults
                'is_primary_key': bool(row.get('IS_PRIMARY_KEY', False)),
                'is_foreign_key': bool(row.get('IS_FOREIGN_KEY', False)),
                'foreign_key_table': row.get('FOREIGN_KEY_TABLE', ''),
                'foreign_key_column': row.get('FOREIGN_KEY_COLUMN', '')
            }
            columns.append(column_info)
        
        # Use the first row's TABLE_NAME and DB_SCHEMA
        table_name = group['TABLE_NAME'].iloc[0]
        db_schema = group['DB_SCHEMA'].iloc[0] if schema_aware else ''
        
        table_info = {
            'name': full_table_name,
            'schema': db_schema if schema_aware else '',
            'simple_name': table_name,
            'description': group.get('TABLE_DESCRIPTION', '').iloc[0] if 'TABLE_DESCRIPTION' in group.columns else '',
            'columns': columns
        }
        tables.append(table_info)
    
    return tables

def store_query_result(query: str, sql: str, execution_successful: bool = True, execution_result: Optional[pd.DataFrame] = None) -> str:
    """
    Store a query, its SQL and results in the vector database for future reuse.
    
    Args:
        query: Original natural language query
        sql: Generated SQL query
        execution_successful: Whether the SQL executed successfully
        execution_result: DataFrame with execution results (optional)
        
    Returns:
        ID of the stored query
    """
    try:
        # Generate a unique ID for this query
        query_hash = hashlib.md5(query.encode()).hexdigest()
        query_id = f"query_{query_hash}_{int(time.time())}"
        
        # Generate query embedding
        if not USE_FALLBACK_MODE:
            query_embedding = EMBEDDING_MODEL.encode(query)
            query_embeddings = [query_embedding.tolist()]
        else:
            query_embeddings = None  # ChromaDB will handle this internally
        
        # Create metadata
        metadata = {
            "original_query": query,
            "sql": sql,
            "execution_successful": execution_successful,
            "timestamp": datetime.now().isoformat(),
            "has_results": execution_result is not None and len(execution_result) > 0
        }
        
        # If there are results, store them as a parquet file
        results_path = None
        if execution_successful and execution_result is not None and len(execution_result) > 0:
            results_path = os.path.join(QUERY_CACHE_DIR, f"{query_id}_results.parquet")
            execution_result.to_parquet(results_path)
            metadata["results_path"] = results_path
        
        # Store in the query collection
        query_collection = get_query_collection()
        
        if not USE_FALLBACK_MODE:
            query_collection.add(
                documents=[query],
                embeddings=query_embeddings,
                metadatas=[metadata],
                ids=[query_id]
            )
        else:
            query_collection.add(
                documents=[query],
                metadatas=[metadata],
                ids=[query_id]
            )
        
        return query_id
        
    except Exception as e:
        print(f"Error storing query result: {str(e)}")
        return ""

def find_similar_query(query: str, threshold: float = 0.85) -> Dict[str, Any]:
    """
    Find a similar query in the query history.
    
    Args:
        query: Natural language query to find similar matches for
        threshold: Similarity threshold (0-1), higher means more similar
        
    Returns:
        Dict with original query, SQL, and similarity score if found, None otherwise
    """
    try:
        print(f"Searching for query similar to: '{query}'")
        print(f"Using similarity threshold: {threshold}")
        
        # Get the query collection
        query_collection = get_query_collection()
        
        # Search for similar queries based on mode
        if not USE_FALLBACK_MODE:
            # Generate embedding for the query using sentence-transformers
            query_embedding = EMBEDDING_MODEL.encode(query)
            
            # Search for similar queries
            results = query_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=1,  # We only need the most similar
                include=["documents", "metadatas", "distances"]
            )
        else:
            # In fallback mode, use text search directly
            results = query_collection.query(
                query_texts=[query],
                n_results=1,  # We only need the most similar
                include=["documents", "metadatas", "distances"]
            )
            
        print(f"Search complete. Found {len(results.get('ids', [[]])[0])} results.")
        
        # Check if any results meet the similarity threshold
        for i, distance in enumerate(results["distances"][0]):
            # Convert distance to similarity score (1 - distance)
            similarity = 1.0 - distance
            
            # Check if similarity meets threshold
            if similarity >= threshold:
                metadata = results["metadatas"][0][i]
                original_query = metadata["original_query"]
                sql = metadata["sql"]
                execution_successful = metadata["execution_successful"]
                
                # If there are stored results, load them
                results_df = None
                if "results_path" in metadata and os.path.exists(metadata["results_path"]):
                    try:
                        results_df = pd.read_parquet(metadata["results_path"])
                    except Exception as e:
                        print(f"Error loading cached results: {str(e)}")
                
                return {
                    "found_similar_query": True,
                    "original_query": original_query,
                    "sql": sql,
                    "similarity_score": similarity,
                    "execution_successful": execution_successful,
                    "cached_results": results_df,
                    "timestamp": metadata["timestamp"]
                }
        
        # No similar queries found
        return None
        
    except Exception as e:
        print(f"Error finding similar query: {str(e)}")
        return None

def optimize_input_prompt(query: str, data_dictionary_path: str) -> Dict[str, Any]:
    """
    Optimize the input prompt using RAG by retrieving only relevant schema information.
    
    This is the second requested function: optimize input prompt using RAG.
    
    Args:
        query: Natural language query
        data_dictionary_path: Path to the data dictionary CSV/Excel file
        
    Returns:
        Dictionary with optimized schema info and formatted prompt
    """
    try:
        # Ensure schema embeddings are generated
        if not os.path.exists(os.path.join(SCHEMA_VECTORS_DIR, "embeddings_complete.txt")):
            generate_schema_embeddings(data_dictionary_path)
        
        # Load full data dictionary for reference
        if data_dictionary_path.endswith('.xlsx') or data_dictionary_path.endswith('.xls'):
            full_dict_df = pd.read_excel(data_dictionary_path)
        elif data_dictionary_path.endswith('.csv'):
            full_dict_df = pd.read_csv(data_dictionary_path)
        else:
            raise ValueError("Unsupported file format. Please provide an Excel or CSV file.")
        
        # Find relevant schema for this query
        relevant_schema = find_relevant_schema(query, top_k=15)  # Get top 15 schema elements
        
        # Format the reduced data dictionary in the same format expected by the LLM
        reduced_tables = format_reduced_data_dictionary(relevant_schema, full_dict_df)
        
        # Format tables info into string (similar to the existing generate_sql_prompt function)
        tables_context = ""
        for table in reduced_tables:
            # Include schema in table name if available
            if table['schema']:
                tables_context += f"Table: {table['name']} (Schema: {table['schema']})\n"
            else:
                tables_context += f"Table: {table['name']}\n"
                
            if table['description']:
                tables_context += f"Description: {table['description']}\n"
                
            tables_context += "Columns:\n"
            
            for col in table['columns']:
                pk_indicator = " (PRIMARY KEY)" if col.get('is_primary_key') else ""
                fk_info = ""
                if col.get('is_foreign_key'):
                    fk_info = f" (FOREIGN KEY to {col.get('foreign_key_table')}.{col.get('foreign_key_column')})"
                
                tables_context += f"  - {col['name']} ({col['type']}){pk_indicator}{fk_info}: {col['description']}"
                
                # Only add business name if it's different from column name
                if col['business_name'] != col['name']:
                    tables_context += f" [Business Name: {col['business_name']}]"
                    
                tables_context += "\n"
            
            tables_context += "\n"
        
        # Create optimized prompt template (similar to generate_sql_prompt)
        optimized_prompt = f"""You are an expert SQL query generator for Snowflake database.

Your task is to convert natural language questions into valid SQL queries that can run on Snowflake.
Use the following data dictionary to understand the database schema:

{tables_context}

When generating SQL:
1. Use proper Snowflake SQL syntax with fully qualified table names including schema (e.g., SCHEMA.TABLE_NAME)
2. Include appropriate JOINs based on the data relationships or column name similarities
3. Format the SQL code clearly with proper indentation and aliases
4. Only use tables and columns that exist in the provided schema
5. Add helpful SQL comments to explain complex parts of the query
6. Return ONLY the SQL code without any other text or explanations
7. Limit results to 100 rows unless specified otherwise

Generate a SQL query for: {query}
"""
        
        # Calculate token savings statistics
        full_tables_count = len(full_dict_df["TABLE_NAME"].unique())
        reduced_tables_count = len(reduced_tables)
        full_columns_count = len(full_dict_df)
        reduced_columns_count = sum(len(table['columns']) for table in reduced_tables)
        
        return {
            "query": query,
            "optimized_prompt": optimized_prompt,
            "original_tables_count": full_tables_count,
            "reduced_tables_count": reduced_tables_count,
            "original_columns_count": full_columns_count,
            "reduced_columns_count": reduced_columns_count,
            "token_reduction_percent": round((1 - (reduced_columns_count / full_columns_count)) * 100, 2),
            "tables": reduced_tables
        }
        
    except Exception as e:
        print(f"Error optimizing input prompt: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "optimized_prompt": None
        }

# Example of how to use the functions
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Optimizer for LLM Query Engine")
    parser.add_argument("--query", "-q", required=True, help="Natural language query")
    parser.add_argument("--data-dict", "-d", required=True, help="Path to data dictionary file")
    parser.add_argument("--generate-embeddings", "-g", action="store_true", help="Generate schema embeddings")
    parser.add_argument("--force-refresh", "-f", action="store_true", help="Force refresh of schema embeddings")
    parser.add_argument("--find-similar", "-s", action="store_true", help="Find similar previous query")
    
    args = parser.parse_args()
    
    # Check if we need to generate embeddings
    if args.generate_embeddings:
        print(f"Generating schema embeddings from {args.data_dict}...")
        generate_schema_embeddings(args.data_dict, force_refresh=args.force_refresh)
    
    # Check for similar query
    if args.find_similar:
        print(f"Finding similar query to: '{args.query}'")
        similar = find_similar_query(args.query)
        if similar:
            print(f"Found similar query: '{similar['original_query']}'")
            print(f"Similarity score: {similar['similarity_score']:.2f}")
            print(f"SQL: {similar['sql']}")
        else:
            print("No similar query found.")
    
    # Always optimize the prompt
    print(f"Optimizing prompt for query: '{args.query}'")
    result = optimize_input_prompt(args.query, args.data_dict)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Reduced from {result['original_tables_count']} to {result['reduced_tables_count']} tables")
        print(f"Reduced from {result['original_columns_count']} to {result['reduced_columns_count']} columns")
        print(f"Token reduction: {result['token_reduction_percent']}%")
        print("\nOptimized prompt:")
        print(result['optimized_prompt'][:500] + "...[truncated]")
