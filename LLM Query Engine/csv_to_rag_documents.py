#!/usr/bin/env python3
"""
Utility script to convert CSV data dictionary to RAG documents format.
This script reads a CSV data dictionary and generates the JSON payload for /rag/documents endpoint.
"""

import pandas as pd
import json
import sys
import os
from typing import List, Dict, Any

def csv_to_rag_documents(
    csv_file_path: str,
    collection_name: str,
    client_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    db_schema_col: str = "DB_SCHEMA",
    table_col: str = "TABLE_NAME", 
    column_col: str = "COLUMN_NAME",
    description_col: str = "DESCRIPTION"
) -> Dict[str, Any]:
    """
    Convert CSV data dictionary to RAG documents format.
    
    Args:
        csv_file_path: Path to the CSV file
        collection_name: Name for the collection (e.g., "mts_data_dictionary")
        client_id: Client identifier
        chunk_size: Chunk size for processing
        chunk_overlap: Chunk overlap for processing
        db_schema_col: Column name for database schema
        table_col: Column name for table names
        column_col: Column name for column names
        description_col: Column name for descriptions
        
    Returns:
        Dictionary ready for /rag/documents API call
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Successfully read CSV file: {csv_file_path}")
        print(f"📊 Total rows: {len(df)}")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return None
    
    # Check if required columns exist
    required_cols = [table_col, column_col, description_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"📋 Available columns: {list(df.columns)}")
        return None
    
    # Create documents list
    documents = []
    
    for index, row in df.iterrows():
        # Extract values
        db_schema = row.get(db_schema_col, "")
        table_name = row.get(table_col, "")
        column_name = row.get(column_col, "")
        description = row.get(description_col, "")
        
        # Create text content
        text_parts = []
        if db_schema:
            text_parts.append(f"DB_SCHEMA: {db_schema}")
        text_parts.append(f"TABLE_NAME: {table_name}")
        text_parts.append(f"COLUMN_NAME: {column_name}")
        text_parts.append(f"DESCRIPTION: {description}")
        
        text = ", ".join(text_parts)
        
        # Create metadata
        metadata = {
            "table_name": table_name,
            "column_name": column_name,
            "row_index": index
        }
        if db_schema:
            metadata["db_schema"] = db_schema
        
        # Create document ID
        doc_id = f"{table_name.lower()}_{column_name.lower()}".replace(" ", "_")
        
        # Create document
        document = {
            "text": text,
            "metadata": metadata,
            "id": doc_id
        }
        
        documents.append(document)
    
    # Create the complete payload
    payload = {
        "documents": documents,
        "collection_name": collection_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "metadata": {
            "client_id": client_id,
            "source_file": os.path.basename(csv_file_path),
            "total_tables": len(df[table_col].unique()),
            "total_columns": len(documents)
        }
    }
    
    print(f"✅ Created {len(documents)} documents")
    print(f"📊 Tables covered: {len(df[table_col].unique())}")
    print(f"📋 Unique tables: {sorted(df[table_col].unique())}")
    
    return payload

def save_payload_to_file(payload: Dict[str, Any], output_file: str):
    """Save the payload to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✅ Payload saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving payload: {e}")

def print_curl_command(payload: Dict[str, Any], base_url: str = "http://localhost:8001"):
    """Print the curl command to call the API."""
    
    print(f"\n🚀 **CURL Command to call /rag/documents API:**")
    print(f"```bash")
    print(f"curl -X 'POST' \\")
    print(f"  '{base_url}/rag/documents' \\")
    print(f"  -H 'accept: application/json' \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(payload, separators=(',', ':'))}'")
    print(f"```")

def main():
    """Main function to demonstrate usage."""
    
    print("=== CSV to RAG Documents Converter ===\n")
    
    # Configuration for MTS
    mts_config = {
        "csv_file_path": "config/clients/data_dictionaries/mts/mts_dictionary.csv",
        "collection_name": "mts_data_dictionary",
        "client_id": "mts",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    # Configuration for Penguin
    penguin_config = {
        "csv_file_path": "config/clients/data_dictionaries/penguin/penguin_dictionary.csv", 
        "collection_name": "penguin_data_dictionary",
        "client_id": "penguin",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    # Process MTS
    print("🔄 Processing MTS data dictionary...")
    mts_payload = csv_to_rag_documents(**mts_config)
    
    if mts_payload:
        # Save to file
        save_payload_to_file(mts_payload, "mts_rag_documents_payload.json")
        
        # Print curl command
        print_curl_command(mts_payload)
        
        print(f"\n📋 **Sample documents (first 2):**")
        for i, doc in enumerate(mts_payload["documents"][:2]):
            print(f"Document {i+1}:")
            print(f"  ID: {doc['id']}")
            print(f"  Text: {doc['text'][:100]}...")
            print(f"  Metadata: {doc['metadata']}")
            print()
    
    print("\n" + "="*60 + "\n")
    
    # Process Penguin
    print("🔄 Processing Penguin data dictionary...")
    penguin_payload = csv_to_rag_documents(**penguin_config)
    
    if penguin_payload:
        # Save to file
        save_payload_to_file(penguin_payload, "penguin_rag_documents_payload.json")
        
        # Print curl command
        print_curl_command(penguin_payload)

if __name__ == "__main__":
    main()
