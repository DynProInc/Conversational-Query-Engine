#!/usr/bin/env python3
"""
Script to upload generated JSON payloads to the RAG documents endpoint.
"""

import requests
import json
import os
from pathlib import Path

def upload_documents_to_rag(json_file_path: str, api_base_url: str = "http://localhost:8001"):
    """
    Upload documents from JSON file to RAG endpoint.
    
    Args:
        json_file_path: Path to the JSON file containing documents
        api_base_url: Base URL of the API server
    """
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ JSON file not found: {json_file_path}")
        return False
    
    # Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        print(f"✅ Loaded JSON payload with {len(payload.get('documents', []))} documents")
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return False
    
    # Upload to RAG endpoint
    try:
        url = f"{api_base_url}/rag/documents"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        print(f"🚀 Uploading to {url}...")
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Collection: {result.get('collection_name')}")
            print(f"   Documents processed: {result.get('documents_processed')}")
            print(f"   Total chunks: {result.get('total_chunks')}")
            print(f"   Processing time: {result.get('processing_time_seconds', 0):.2f}s")
            return True
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading to RAG: {e}")
        return False

def main():
    """Main function to upload all JSON files in the current directory."""
    
    # Look for JSON files in current directory
    current_dir = Path(".")
    json_files = list(current_dir.glob("*_rag_documents.json"))
    
    if not json_files:
        print("❌ No *_rag_documents.json files found in current directory")
        print("   Please run csv_to_rag_documents.py first to generate JSON files")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to upload:")
    for file in json_files:
        print(f"  - {file}")
    
    # Upload each file
    success_count = 0
    for json_file in json_files:
        print(f"\n{'='*50}")
        print(f"Processing: {json_file}")
        print(f"{'='*50}")
        
        if upload_documents_to_rag(str(json_file)):
            success_count += 1
        
    print(f"\n{'='*50}")
    print(f"Summary: {success_count}/{len(json_files)} files uploaded successfully")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
