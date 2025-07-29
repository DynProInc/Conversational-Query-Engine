#!/usr/bin/env python3
"""
Test script for query history and saved queries routes
"""
import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_query_history_routes():
    """Test all query history routes"""
    print("Testing Query History Routes...")
    
    # Test user ID
    user_id = "test_user_123"
    
    # Test 1: Get query history (should be empty initially)
    print("\n1. Testing GET /api/query_history")
    response = requests.get(f"{BASE_URL}/api/query_history", params={"user_id": user_id})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Delete non-existent query (should return 404)
    print("\n2. Testing DELETE /api/query_history/999")
    response = requests.delete(f"{BASE_URL}/api/query_history/999", params={"user_id": user_id})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_saved_queries_routes():
    """Test all saved queries routes"""
    print("\n\nTesting Saved Queries Routes...")
    
    # Test user ID
    user_id = "test_user_123"
    
    # Test 1: Get saved queries (should be empty initially)
    print("\n1. Testing GET /api/saved_queries")
    response = requests.get(f"{BASE_URL}/api/saved_queries", params={"user_id": user_id})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Save a new query
    print("\n2. Testing POST /api/saved_queries")
    new_query = {
        "user_id": user_id,
        "prompt": "Show me all customers",
        "name": "Customer Query",
        "description": "A simple query to get all customers",
        "sql_query": "SELECT * FROM customers",
        "database": "test_db",
        "tags": ["customers", "simple"]
    }
    response = requests.post(f"{BASE_URL}/api/saved_queries", json=new_query)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 201:
        saved_query = response.json()
        query_id = saved_query['id']
        
        # Test 3: Update the saved query
        print(f"\n3. Testing PUT /api/saved_queries/{query_id}")
        update_data = {
            "name": "Updated Customer Query",
            "description": "Updated description",
            "tags": ["customers", "updated", "test"]
        }
        response = requests.put(f"{BASE_URL}/api/saved_queries/{query_id}", 
                              json=update_data, 
                              params={"user_id": user_id})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test 4: Get saved queries again (should show the new query)
        print("\n4. Testing GET /api/saved_queries (after save)")
        response = requests.get(f"{BASE_URL}/api/saved_queries", params={"user_id": user_id})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test 5: Delete the saved query
        print(f"\n5. Testing DELETE /api/saved_queries/{query_id}")
        response = requests.delete(f"{BASE_URL}/api/saved_queries/{query_id}", 
                                 params={"user_id": user_id})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test 6: Get saved queries again (should be empty)
        print("\n6. Testing GET /api/saved_queries (after delete)")
        response = requests.get(f"{BASE_URL}/api/saved_queries", params={"user_id": user_id})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

def test_utility_functions():
    """Test the utility functions for saving queries"""
    print("\n\nTesting Utility Functions...")
    
    try:
        from utils.query_storage import save_query_to_history, save_query_to_saved_queries
        
        # Test saving to history
        print("\n1. Testing save_query_to_history")
        history_id = save_query_to_history(
            user_id="test_user_456",
            prompt="Test prompt for history",
            sql_query="SELECT * FROM test_table",
            model="claude-3-5-sonnet",
            query_executed=True,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003
        )
        print(f"Saved to history with ID: {history_id}")
        
        # Test saving to saved queries
        print("\n2. Testing save_query_to_saved_queries")
        saved_id = save_query_to_saved_queries(
            user_id="test_user_456",
            prompt="Test prompt for saved queries",
            name="Test Saved Query",
            description="A test query for saved queries",
            sql_query="SELECT * FROM saved_test_table",
            tags=["test", "saved"],
            model="claude-3-5-sonnet",
            query_executed=True,
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            input_cost=0.002,
            output_cost=0.004,
            total_cost=0.006
        )
        print(f"Saved to saved queries with ID: {saved_id}")
        
    except Exception as e:
        print(f"Error testing utility functions: {e}")

if __name__ == "__main__":
    print("Starting Query History Routes Test...")
    
    try:
        test_query_history_routes()
        test_saved_queries_routes()
        test_utility_functions()
        print("\n\nAll tests completed!")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"Error during testing: {e}") 