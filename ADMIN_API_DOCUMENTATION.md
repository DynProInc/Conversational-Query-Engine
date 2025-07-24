# Admin API Documentation

This document describes the admin API endpoints for the LLM Query Engine.

## Base URL
```
https://conversational-query-engine-pp4v.onrender.com
```

## Authentication
All admin endpoints require authentication using Bearer token authentication. The token must start with `admin_` for demo purposes.

**Header:**
```
Authorization: Bearer admin_your_token_here
```

## Endpoints

### 1. Get All Users
**GET** `/api/admin/users`

Retrieves a list of all users in the system.

**Response:**
```json
[
  {
    "id": "admin",
    "username": "admin",
    "email": "admin@example.com",
    "role": "admin",
    "created_at": "2024-01-01T00:00:00Z",
    "last_login": "2024-01-01T00:00:00Z",
    "is_active": true
  }
]
```

### 2. Update User Role
**PUT** `/api/admin/users/{user_id}`

Updates a user's role and active status.

**Path Parameters:**
- `user_id` (string): The ID of the user to update

**Request Body:**
```json
{
  "role": "moderator",
  "is_active": true
}
```

**Valid Roles:**
- `admin`
- `user`
- `moderator`

**Response:**
```json
{
  "id": "user1",
  "username": "user1",
  "email": "user1@example.com",
  "role": "moderator",
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-01T00:00:00Z",
  "is_active": true
}
```

### 3. Get System Statistics
**GET** `/api/admin/stats`

Retrieves system-wide statistics and performance metrics.

**Response:**
```json
{
  "total_users": 3,
  "active_users": 3,
  "total_queries": 4,
  "successful_queries": 3,
  "failed_queries": 1,
  "success_rate": 75.0,
  "average_response_time_ms": 1250.0,
  "system_uptime_hours": 24.5,
  "memory_usage_mb": 512.0,
  "cpu_usage_percent": 15.5
}
```

### 4. Get User Analytics
**GET** `/api/admin/analytics`

Retrieves detailed analytics about user activity and system performance.

**Response:**
```json
{
  "total_users": 3,
  "total_queries": 4,
  "queries_by_model": {
    "gpt-4": 3,
    "claude": 1
  },
  "queries_by_day": {
    "2024-01-15": 4
  },
  "user_analytics": [
    {
      "user_id": "user1",
      "username": "user1",
      "total_queries": 2,
      "successful_queries": 1,
      "failed_queries": 1,
      "success_rate": 50.0,
      "last_activity": "2024-01-15T14:20:00Z",
      "favorite_model": "gpt-4",
      "average_response_time_ms": 1250.0
    }
  ],
  "system_performance": {
    "uptime_hours": 24.5,
    "memory_usage_mb": 512.0,
    "cpu_usage_percent": 15.5,
    "disk_usage_percent": 45.2,
    "active_connections": 12,
    "requests_per_minute": 8.5
  }
}
```

### 5. Admin Health Check
**GET** `/api/admin/health`

Checks the health status of admin routes (no authentication required).

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "admin_routes_available": true
}
```

## Admin Dashboard UI

A web-based admin dashboard is available at:
```
https://conversational-query-engine-pp4v.onrender.com/admin
```

The dashboard provides a user-friendly interface to:
- View system statistics
- Manage users
- View analytics
- Test API connections

## Testing

### Using curl

```bash
# Get all users
curl -H "Authorization: Bearer admin_test_token_123" \
     https://conversational-query-engine-pp4v.onrender.com/api/admin/users

# Update user role
curl -X PUT \
     -H "Authorization: Bearer admin_test_token_123" \
     -H "Content-Type: application/json" \
     -d '{"role": "moderator", "is_active": true}' \
     https://conversational-query-engine-pp4v.onrender.com/api/admin/users/user1

# Get system statistics
curl -H "Authorization: Bearer admin_test_token_123" \
     https://conversational-query-engine-pp4v.onrender.com/api/admin/stats

# Get analytics
curl -H "Authorization: Bearer admin_test_token_123" \
     https://conversational-query-engine-pp4v.onrender.com/api/admin/analytics
```

### Using Python

```python
import requests

BASE_URL = "https://conversational-query-engine-pp4v.onrender.com"
HEADERS = {
    "Authorization": "Bearer admin_test_token_123",
    "Content-Type": "application/json"
}

# Get all users
response = requests.get(f"{BASE_URL}/api/admin/users", headers=HEADERS)
users = response.json()

# Update user role
update_data = {"role": "moderator", "is_active": True}
response = requests.put(
    f"{BASE_URL}/api/admin/users/user1", 
    headers=HEADERS, 
    json=update_data
)

# Get system statistics
response = requests.get(f"{BASE_URL}/api/admin/stats", headers=HEADERS)
stats = response.json()

# Get analytics
response = requests.get(f"{BASE_URL}/api/admin/analytics", headers=HEADERS)
analytics = response.json()
```

## Error Responses

All endpoints return standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Invalid or missing authentication token
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "detail": "Error message description"
}
```

## Notes

- This is a demo implementation with in-memory storage
- In production, implement proper JWT token validation
- User data and analytics are currently mock data
- Consider implementing database persistence for production use
- Add rate limiting and additional security measures for production deployment 