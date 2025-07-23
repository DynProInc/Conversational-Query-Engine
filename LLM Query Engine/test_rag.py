"""
RAG Testing Script - Verify RAG integration functionality
"""

import os
import json
import requests
import time

# Configuration
BASE_URL = "http://127.0.0.1:8000"  # Adjust if using a different port
CLIENT_ID = "mts"  # Update to your client ID

def print_separator(message):
    """Print a separator line with message"""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n")

def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))
    
def test_rag_stats():
    """Test RAG stats endpoint"""
    print_separator("Testing RAG Stats Endpoint")
    
    # Get RAG stats
    response = requests.get(f"{BASE_URL}/rag/stats")
    
    # Check response
    if response.status_code == 200:
        print("✅ RAG stats endpoint returned 200 OK")
        stats = response.json()
        print_json(stats)
        
        if stats.get("total_queries", 0) > 0:
            print("✅ RAG stats show some queries have been processed")
        else:
            print("⚠️ RAG stats show no queries processed yet")
    else:
        print(f"❌ RAG stats endpoint returned {response.status_code}")
        print(response.text)

def upload_data_dictionary():
    """Upload a test data dictionary for RAG"""
    print_separator("Uploading Test Data Dictionary")
    
    # Check if test data dictionary exists
    data_dict_path = os.path.join("config", "clients", "data_dictionaries", CLIENT_ID, f"{CLIENT_ID}_dictionary.csv")
    
    if not os.path.exists(data_dict_path):
        print(f"❌ Data dictionary not found at {data_dict_path}")
        return False
    
    # Upload the data dictionary
    print(f"Uploading data dictionary from {data_dict_path}")
    
    with open(data_dict_path, 'rb') as f:
        files = {'file': f}
        data = {'client_id': CLIENT_ID}
        response = requests.post(f"{BASE_URL}/rag/data-dictionary", files=files, data=data)
    
    # Check response
    if response.status_code == 200:
        print("✅ Data dictionary uploaded successfully")
        print_json(response.json())
        return True
    else:
        print(f"❌ Data dictionary upload failed with status {response.status_code}")
        print(response.text)
        return False

def test_rag_collections():
    """Test RAG collections endpoint"""
    print_separator("Testing RAG Collections Endpoint")
    
    # Get RAG collections
    response = requests.get(f"{BASE_URL}/rag/collections")
    
    # Check response
    if response.status_code == 200:
        print("✅ RAG collections endpoint returned 200 OK")
        collections = response.json()
        print_json(collections)
        
        # Check if our collection exists
        collection_name = f"{CLIENT_ID}_data_dictionary"
        found = False
        for collection in collections.get("collections", []):
            if collection == collection_name:
                found = True
                print(f"✅ Found collection: {collection_name}")
                break
        
        if not found:
            print(f"⚠️ Collection {collection_name} not found. This may cause RAG to fail.")
            return False
        return True
    else:
        print(f"❌ RAG collections endpoint returned {response.status_code}")
        print(response.text)
        return False

def test_unified_query_with_rag():
    """Test unified query endpoint with RAG enabled"""
    print_separator("Testing Unified Query with RAG Enabled")
    
    # Prepare query with RAG enabled
    collection_name = f"{CLIENT_ID}_data_dictionary"
    payload = {
        "prompt": "explain the daily sales data structure",
        "client_id": CLIENT_ID,
        "use_rag": True,
        "rag_collection": collection_name,
        "rag_top_k": 3,
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    # Send request
    print("Sending query with RAG enabled:")
    print_json(payload)
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/query/unified", json=payload)
    duration = time.time() - start_time
    
    # Check response
    if response.status_code == 200:
        print(f"✅ Query completed in {duration:.2f} seconds")
        result = response.json()
        
        # Check for RAG indicators in response
        print("\nQuery response:")
        print(f"SQL: {result.get('sql', 'No SQL generated')}")
        
        # Check if there's an enhanced prompt indicator
        if result.get("rag_enhanced", False) or "context" in str(result.get("prompt_used", "")):
            print("\n✅ Query appears to use RAG enhancement")
        else:
            print("\n⚠️ No clear indication that RAG was used. Check server logs.")
        
        return True
    else:
        print(f"❌ Query failed with status {response.status_code}")
        print(response.text)
        return False

def compare_with_standard_query():
    """Compare RAG vs standard query results"""
    print_separator("Comparing RAG vs Standard Query")
    
    # Define test query
    test_query = "what are the top sales regions?"
    
    # Run with RAG disabled
    print("Running standard query without RAG...")
    standard_payload = {
        "prompt": test_query,
        "client_id": CLIENT_ID,
        "use_rag": False
    }
    
    standard_start = time.time()
    standard_response = requests.post(f"{BASE_URL}/query/unified", json=standard_payload)
    standard_duration = time.time() - standard_start
    
    print(f"Standard query completed in {standard_duration:.2f} seconds")
    
    if standard_response.status_code != 200:
        print("❌ Standard query failed")
        return
        
    # Run with RAG enabled
    print("\nRunning same query with RAG enabled...")
    rag_payload = {
        "prompt": test_query,
        "client_id": CLIENT_ID,
        "use_rag": True,
        "rag_collection": f"{CLIENT_ID}_data_dictionary",
        "rag_top_k": 3
    }
    
    rag_start = time.time()
    rag_response = requests.post(f"{BASE_URL}/query/unified", json=rag_payload)
    rag_duration = time.time() - rag_start
    
    print(f"RAG query completed in {rag_duration:.2f} seconds")
    
    if rag_response.status_code != 200:
        print("❌ RAG query failed")
        return
        
    # Compare results
    standard_result = standard_response.json()
    rag_result = rag_response.json()
    
    print("\nStandard SQL:")
    print(standard_result.get("sql", "No SQL"))
    
    print("\nRAG SQL:")
    print(rag_result.get("sql", "No SQL"))
    
    # Check RAG stats again to see if counters increased
    print("\nChecking if RAG stats counters increased...")
    stats_response = requests.get(f"{BASE_URL}/rag/stats")
    if stats_response.status_code == 200:
        print_json(stats_response.json())
    
def main():
    """Run all tests"""
    print_separator("RAG Integration Test Suite")
    
    # Step 1: Check RAG Stats before testing
    test_rag_stats()
    
    # Step 2: Upload data dictionary if needed
    upload_data_dictionary()
    
    # Step 3: Verify collections
    collections_exist = test_rag_collections()
    
    if not collections_exist:
        print("\n⚠️ Warning: Vector store collections not found. RAG may not work properly.")
    
    # Step 4: Test unified query with RAG
    test_unified_query_with_rag()
    
    # Step 5: Compare standard vs RAG queries
    compare_with_standard_query()
    
    # Step 6: Check RAG stats again
    print("\nFinal RAG stats:")
    test_rag_stats()
    
    print_separator("Test Suite Complete")
    
if __name__ == "__main__":
    main()
