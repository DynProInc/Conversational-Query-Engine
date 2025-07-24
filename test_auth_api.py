"""
Test script for Authentication API endpoints

This script tests the new authentication system and admin endpoints with proper JWT tokens.
"""
import requests
import json

# Base URL for the API
BASE_URL = "https://conversational-query-engine-pp4v.onrender.com"

def test_authentication():
    """Test the authentication system"""
    
    print("Testing Authentication API...")
    print("=" * 50)
    
    # Test 1: Login with valid credentials
    print("\n1. Testing POST /api/auth/login with valid credentials")
    login_data = {
        "email": "arman.khan@dynpro.com",
        "password": "Password123!"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data["access_token"]
            print(f"✅ Login successful! Token received: {access_token[:20]}...")
            return access_token
        else:
            print(f"❌ Login failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_admin_endpoints_with_auth(token):
    """Test admin endpoints with proper authentication"""
    
    if not token:
        print("❌ No token available, skipping admin tests")
        return
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print("\n2. Testing admin endpoints with JWT authentication")
    print("=" * 50)
    
    # Test admin users endpoint
    print("\n2.1 Testing GET /api/admin/users")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/users", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            users = response.json()
            print(f"✅ Users found: {len(users)}")
            for user in users:
                print(f"  - {user['name']} ({user['role']})")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test admin stats endpoint
    print("\n2.2 Testing GET /api/admin/stats")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/stats", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ System Statistics:")
            print(f"  - Total Users: {stats['total_users']}")
            print(f"  - Active Users: {stats['active_users']}")
            print(f"  - Total Queries: {stats['total_queries']}")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test admin analytics endpoint
    print("\n2.3 Testing GET /api/admin/analytics")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/analytics", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            analytics = response.json()
            print(f"✅ Analytics retrieved successfully")
            print(f"  - Total Users: {analytics['total_users']}")
            print(f"  - Total Queries: {analytics['total_queries']}")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_user_profile(token):
    """Test user profile endpoint"""
    
    if not token:
        print("❌ No token available, skipping profile test")
        return
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print("\n3. Testing GET /api/auth/me (user profile)")
    try:
        response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            user = response.json()
            print(f"✅ User profile retrieved:")
            print(f"  - Name: {user['name']}")
            print(f"  - Email: {user['email']}")
            print(f"  - Role: {user['role']}")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_invalid_credentials():
    """Test login with invalid credentials"""
    
    print("\n4. Testing POST /api/auth/login with invalid credentials")
    login_data = {
        "email": "invalid@example.com",
        "password": "wrongpassword"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 401:
            print("✅ Correctly rejected invalid credentials")
        else:
            print(f"❌ Unexpected response: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_unauthorized_access():
    """Test admin endpoints without authentication"""
    
    print("\n5. Testing admin endpoints without authentication")
    
    # Test without token
    try:
        response = requests.get(f"{BASE_URL}/api/admin/users")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✅ Correctly rejected unauthorized access")
        else:
            print(f"❌ Unexpected response: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    # Test authentication flow
    token = test_authentication()
    
    # Test admin endpoints with proper auth
    test_admin_endpoints_with_auth(token)
    
    # Test user profile
    test_user_profile(token)
    
    # Test invalid credentials
    test_invalid_credentials()
    
    # Test unauthorized access
    test_unauthorized_access()
    
    print("\n" + "=" * 50)
    print("Authentication API Testing Complete!") 