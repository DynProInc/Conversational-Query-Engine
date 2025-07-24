"""
Test script for Google OAuth integration

This script tests the Google OAuth authentication system.
"""
import requests
import json

# Base URL for the API
BASE_URL = "https://conversational-query-engine-pp4v.onrender.com"

def test_google_oauth_endpoints():
    """Test Google OAuth endpoints"""
    
    print("Testing Google OAuth Integration...")
    print("=" * 50)
    
    # Test 1: Google OAuth login URL
    print("\n1. Testing GET /api/auth/google/login")
    try:
        response = requests.get(f"{BASE_URL}/api/auth/google/login")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Google OAuth URL generated: {data.get('auth_url', '')[:50]}...")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Test admin endpoints without Google token (should fail)
    print("\n2. Testing admin endpoints without Google token (should fail)")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/users")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✅ Correctly rejected request without Google token")
        else:
            print(f"❌ Unexpected response: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 3: Test with invalid Google token (should fail)
    print("\n3. Testing admin endpoints with invalid Google token (should fail)")
    headers = {
        "Authorization": "Bearer invalid_google_token_123",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(f"{BASE_URL}/api/admin/users", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✅ Correctly rejected invalid Google token")
        else:
            print(f"❌ Unexpected response: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print("Google OAuth Testing Complete!")
    print("\n📋 Next Steps:")
    print("1. Deploy the backend changes to Render")
    print("2. Test with a real Google token from your frontend")
    print("3. Verify admin endpoints work with Google authentication")

def test_google_token_verification():
    """Test Google token verification (requires real token)"""
    
    print("\n🔐 Google Token Verification Test")
    print("=" * 50)
    print("This test requires a real Google ID token from your frontend.")
    print("To test:")
    print("1. Login with Google in your frontend")
    print("2. Get the ID token from localStorage")
    print("3. Use it in the test below")
    
    # Example of how to test with a real token
    print("\nExample test with real token:")
    print("""
    # Get token from frontend localStorage
    token = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjEyMzQ1Njc4OTAiLCJ0eXAiOiJKV1QifQ..."
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test admin endpoints
    response = requests.get(f"{BASE_URL}/api/admin/users", headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        users = response.json()
        print(f"Users: {users}")
    """)

if __name__ == "__main__":
    test_google_oauth_endpoints()
    test_google_token_verification() 