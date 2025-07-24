"""
Test script for Admin API endpoints

This script tests the admin API endpoints to ensure they are working correctly.
"""
import requests
import json

# Base URL for the API
BASE_URL = "https://conversational-query-engine-pp4v.onrender.com"

# Admin token for testing (in production, this would be a real JWT token)
ADMIN_TOKEN = "admin_test_token_123"

def test_admin_endpoints():
    """Test all admin endpoints"""
    
    headers = {
        "Authorization": f"Bearer {ADMIN_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("Testing Admin API Endpoints...")
    print("=" * 50)
    
    # Test 1: Get all users
    print("\n1. Testing GET /api/admin/users")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/users", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            users = response.json()
            print(f"Users found: {len(users)}")
            for user in users:
                print(f"  - {user['username']} ({user['role']})")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test 2: Update user role
    print("\n2. Testing PUT /api/admin/users/user1")
    try:
        update_data = {
            "role": "moderator",
            "is_active": True
        }
        response = requests.put(
            f"{BASE_URL}/api/admin/users/user1", 
            headers=headers,
            json=update_data
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            user = response.json()
            print(f"Updated user: {user['username']} -> {user['role']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test 3: Get system statistics
    print("\n3. Testing GET /api/admin/stats")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/stats", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            stats = response.json()
            print(f"System Statistics:")
            print(f"  - Total Users: {stats['total_users']}")
            print(f"  - Active Users: {stats['active_users']}")
            print(f"  - Total Queries: {stats['total_queries']}")
            print(f"  - Success Rate: {stats['success_rate']:.1f}%")
            print(f"  - System Uptime: {stats['system_uptime_hours']} hours")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test 4: Get user analytics
    print("\n4. Testing GET /api/admin/analytics")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/analytics", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            analytics = response.json()
            print(f"Analytics Summary:")
            print(f"  - Total Users: {analytics['total_users']}")
            print(f"  - Total Queries: {analytics['total_queries']}")
            print(f"  - Queries by Model: {analytics['queries_by_model']}")
            print(f"  - User Analytics: {len(analytics['user_analytics'])} users")
            
            # Show first user analytics
            if analytics['user_analytics']:
                first_user = analytics['user_analytics'][0]
                print(f"  - Sample User: {first_user['username']} ({first_user['total_queries']} queries)")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test 5: Admin health check
    print("\n5. Testing GET /api/admin/health")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/health")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            health = response.json()
            print(f"Health Status: {health['status']}")
            print(f"Admin Routes Available: {health['admin_routes_available']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    print("\n" + "=" * 50)
    print("Admin API Testing Complete!")

if __name__ == "__main__":
    test_admin_endpoints() 