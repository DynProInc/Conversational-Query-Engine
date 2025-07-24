"""
Test script for fixed admin endpoints

This script tests the admin endpoints after the fixes to ensure they work correctly.
"""
import requests
import json

# Base URL for the API
BASE_URL = "https://conversational-query-engine-pp4v.onrender.com"

def test_fixed_endpoints():
    """Test the fixed admin endpoints"""
    
    print("Testing Fixed Admin Endpoints...")
    print("=" * 50)
    
    # First, get a fresh token
    print("\n1. Getting fresh JWT token...")
    login_data = {
        "email": "arman.khan@dynpro.com",
        "password": "Password123!"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data["access_token"]
            print(f"✅ Token received: {access_token[:20]}...")
        else:
            print(f"❌ Login failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Test 2: Fixed users endpoint
    print("\n2. Testing GET /api/admin/users (FIXED)")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/users", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            users = response.json()
            print(f"✅ Users endpoint working! Found {len(users)} users:")
            for user in users:
                print(f"  - {user['name']} ({user['email']}) - {user['role']}")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 3: Fixed analytics endpoint
    print("\n3. Testing GET /api/admin/analytics (FIXED)")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/analytics", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            analytics = response.json()
            print(f"✅ Analytics endpoint working!")
            print(f"  - User Activity: {len(analytics.get('userActivity', []))} days")
            print(f"  - Query Types: {len(analytics.get('queryTypes', []))} types")
            print(f"  - Recent Users: {len(analytics.get('recentUsers', []))} users")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 4: Update user endpoint
    print("\n4. Testing PUT /api/admin/users/1")
    try:
        update_data = {
            "role": "moderator",
            "is_active": True
        }
        response = requests.put(
            f"{BASE_URL}/api/admin/users/1", 
            headers=headers,
            json=update_data
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            user = response.json()
            print(f"✅ User updated: {user['name']} -> {user['role']}")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print("Fixed Endpoints Testing Complete!")

if __name__ == "__main__":
    test_fixed_endpoints() 