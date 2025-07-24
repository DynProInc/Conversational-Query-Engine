# Authentication System Documentation

This document describes the JWT-based authentication system implemented for the LLM Query Engine.

## 🔐 Overview

The authentication system uses JWT (JSON Web Tokens) with bcrypt password hashing to provide secure user authentication and authorization.

## 📋 Features

- **JWT Token Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt password hashing for security
- **Role-Based Access Control**: Admin and user roles
- **Token Expiration**: 30-minute token expiration
- **User Profile Management**: Get current user information

## 🚀 Quick Start

### 1. Login to Get Token

```bash
curl -X POST "https://conversational-query-engine-pp4v.onrender.com/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "arman.khan@dynpro.com",
    "password": "Password123!"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. Use Token for Protected Endpoints

```bash
curl -X GET "https://conversational-query-engine-pp4v.onrender.com/api/admin/users" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 👥 Default Users

The system comes with these default users:

| Email | Password | Role | Name |
|-------|----------|------|------|
| `arman.khan@dynpro.com` | `Password123!` | admin | Arman Khan |
| `admin@example.com` | `admin123!` | admin | System Admin |
| `user1@example.com` | `user123!` | user | Test User 1 |
| `user2@example.com` | `user123!` | user | Test User 2 |

## 🔗 API Endpoints

### Authentication Endpoints

#### POST `/api/auth/login`
Login with email and password to get JWT token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### GET `/api/auth/me`
Get current user profile (requires authentication).

**Headers:**
```
Authorization: Bearer YOUR_TOKEN_HERE
```

**Response:**
```json
{
  "id": "1",
  "email": "user@example.com",
  "name": "User Name",
  "role": "admin",
  "is_active": true
}
```

### Protected Admin Endpoints

All admin endpoints require authentication and admin role:

#### GET `/api/admin/users`
Get all users in the system.

#### PUT `/api/admin/users/{user_id}`
Update user role and status.

#### GET `/api/admin/stats`
Get system statistics.

#### GET `/api/admin/analytics`
Get user analytics and system performance.

## 🛡️ Security Features

### Password Security
- **Bcrypt Hashing**: Passwords are hashed using bcrypt
- **Salt**: Each password has a unique salt
- **Cost Factor**: Configurable bcrypt cost factor

### JWT Security
- **Algorithm**: HS256 (HMAC with SHA-256)
- **Expiration**: 30 minutes by default
- **Secret Key**: Configurable via environment variable

### Access Control
- **Role-Based**: Different permissions for admin and user roles
- **Token Validation**: All protected endpoints validate JWT tokens
- **Admin Only**: Admin endpoints require admin role

## 🔧 Configuration

### Environment Variables

Set these environment variables in your deployment:

```bash
JWT_SECRET_KEY=your-secure-secret-key-here
```

### Token Configuration

You can modify these settings in `utils/auth.py`:

```python
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
```

## 🧪 Testing

### Test Script

Run the authentication test script:

```bash
python3 test_auth_api.py
```

### Manual Testing

1. **Login Test:**
```bash
curl -X POST "https://conversational-query-engine-pp4v.onrender.com/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "arman.khan@dynpro.com", "password": "Password123!"}'
```

2. **Admin Endpoint Test:**
```bash
# First get token from login
TOKEN="your_token_here"

curl -X GET "https://conversational-query-engine-pp4v.onrender.com/api/admin/users" \
  -H "Authorization: Bearer $TOKEN"
```

3. **User Profile Test:**
```bash
curl -X GET "https://conversational-query-engine-pp4v.onrender.com/api/auth/me" \
  -H "Authorization: Bearer $TOKEN"
```

## 🎯 Admin Dashboard

Access the web-based admin dashboard at:
```
https://conversational-query-engine-pp4v.onrender.com/admin
```

Features:
- **Login Form**: Email/password authentication
- **User Management**: View and manage users
- **System Statistics**: Real-time system metrics
- **Analytics**: User activity and performance data

## 🔄 Migration from Old System

The old admin system used simple token validation. The new system:

1. **Replaces** simple token validation with JWT authentication
2. **Adds** proper user management with roles
3. **Improves** security with password hashing
4. **Provides** user profile management

### Breaking Changes

- Old admin tokens (starting with `admin_`) no longer work
- All admin endpoints now require proper JWT authentication
- User management is now role-based

## 🚨 Security Best Practices

1. **Change Default Passwords**: Update default user passwords in production
2. **Secure Secret Key**: Use a strong, unique JWT secret key
3. **HTTPS Only**: Always use HTTPS in production
4. **Token Storage**: Store tokens securely on the client side
5. **Token Expiration**: Monitor token expiration and refresh as needed

## 🔍 Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check if token is valid and not expired
   - Verify token format: `Bearer YOUR_TOKEN`

2. **403 Forbidden**
   - User doesn't have admin role
   - Check user permissions

3. **Login Failed**
   - Verify email and password
   - Check if user exists and is active

### Debug Mode

Enable debug logging by setting environment variable:
```bash
DEBUG=true
```

## 📚 Additional Resources

- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT.io](https://jwt.io/) - JWT token decoder
- [Bcrypt Documentation](https://passlib.readthedocs.io/en/stable/lib/passlib.hash.bcrypt.html) 