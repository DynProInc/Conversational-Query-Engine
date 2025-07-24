# Google OAuth Integration Documentation

This document describes the Google OAuth authentication system implemented for the LLM Query Engine.

## 🔐 Overview

The system now supports Google OAuth authentication, allowing users to sign in with their Google accounts. This provides a more secure and user-friendly authentication experience.

## 🚀 Features

- **Google OAuth Login**: Users can sign in with their Google accounts
- **Automatic User Creation**: New users are automatically created when they first sign in
- **Role-Based Access Control**: Admin and user roles are supported
- **Token Verification**: Google ID tokens are verified on the backend
- **Seamless Integration**: Works with existing admin endpoints

## 🔧 Configuration

### Environment Variables

Set these environment variables in your deployment:

```bash
GOOGLE_CLIENT_ID=332727032541-ero3io8bdeejg1843b18teldibqe7sr8.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
GOOGLE_REDIRECT_URI=https://conversational-query-engine-pp4v.onrender.com/api/auth/google/callback
```

### Google Cloud Console Setup

1. **Create OAuth 2.0 Credentials**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to APIs & Services > Credentials
   - Create OAuth 2.0 Client ID
   - Set authorized redirect URIs

2. **Enable Required APIs**:
   - Google+ API
   - Google Identity API

## 🔗 API Endpoints

### Google OAuth Endpoints

#### GET `/api/auth/google/login`
Get Google OAuth authorization URL.

**Response:**
```json
{
  "auth_url": "https://accounts.google.com/o/oauth2/auth?..."
}
```

#### GET `/api/auth/google/callback`
Handle Google OAuth callback (for server-side flow).

**Query Parameters:**
- `code`: Authorization code from Google

**Response:**
- Redirects to frontend with token

#### POST `/api/auth/google/token`
Login with Google ID token (for client-side flow).

**Request Body:**
```json
{
  "id_token": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjEyMzQ1Njc4OTAiLCJ0eXAiOiJKV1QifQ..."
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Protected Admin Endpoints

All admin endpoints now require Google OAuth authentication:

#### GET `/api/admin/users`
Get all users (admin only).

**Headers:**
```
Authorization: Bearer GOOGLE_ID_TOKEN
```

**Response:**
```json
{
  "users": [
    {
      "id": "1",
      "name": "Arman Khan",
      "email": "arman.khan@dynpro.com",
      "role": "admin",
      "status": "active",
      "lastActive": "2024-01-15T10:30:00Z",
      "queries": 12,
      "dateCreated": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### GET `/api/admin/stats`
Get system statistics (admin only).

**Response:**
```json
{
  "totalUsers": 4,
  "userGrowth": 12,
  "totalQueries": 24567,
  "queryGrowth": 18,
  "avgResponseTime": "0.8s",
  "responseTimeImprovement": 5
}
```

#### GET `/api/admin/analytics`
Get user analytics (admin only).

**Response:**
```json
{
  "userActivity": [
    {"name": "Mon", "queries": 65, "users": 24},
    {"name": "Tue", "queries": 59, "users": 22}
  ],
  "queryTypes": [
    {"name": "Text Queries", "value": 65},
    {"name": "Voice Queries", "value": 35}
  ],
  "recentUsers": [
    {"id": "1", "name": "Arman Khan", "email": "arman.khan@dynpro.com", "lastActive": "10 minutes ago", "queries": 12}
  ]
}
```

#### PUT `/api/admin/users/{user_id}/role`
Update user role (admin only).

**Request Body:**
```json
{
  "role": "moderator"
}
```

**Response:**
```json
{
  "message": "Role updated successfully"
}
```

## 🛡️ Security Features

### Token Verification
- **Google ID Token Verification**: All tokens are verified using Google's public keys
- **Client ID Validation**: Tokens are validated against your specific client ID
- **Expiration Checking**: Token expiration is automatically checked

### User Management
- **Automatic User Creation**: New users are created automatically on first login
- **Role Assignment**: New users get 'user' role by default
- **Admin Role**: Only existing admin users can access admin endpoints

### Access Control
- **Admin-Only Endpoints**: Admin endpoints require admin role
- **Token-Based Authentication**: All requests require valid Google token
- **Role Validation**: User roles are checked on each request

## 🔄 Migration from JWT

### Changes Made:
1. **Replaced JWT authentication** with Google OAuth
2. **Updated admin endpoints** to use Google token verification
3. **Modified response formats** to match frontend expectations
4. **Added automatic user creation** for new Google users

### Breaking Changes:
- Old JWT tokens no longer work
- Admin endpoints now require Google authentication
- Response formats have been updated

## 🧪 Testing

### Test Script
Run the Google OAuth test script:

```bash
python3 test_google_oauth.py
```

### Manual Testing

1. **Test OAuth URL Generation:**
```bash
curl -X GET "https://conversational-query-engine-pp4v.onrender.com/api/auth/google/login"
```

2. **Test Admin Endpoints (should fail without token):**
```bash
curl -X GET "https://conversational-query-engine-pp4v.onrender.com/api/admin/users"
```

3. **Test with Google Token (from frontend):**
```bash
# Get token from frontend localStorage
TOKEN="your_google_id_token_here"

curl -X GET "https://conversational-query-engine-pp4v.onrender.com/api/admin/users" \
  -H "Authorization: Bearer $TOKEN"
```

## 🎯 Frontend Integration

### Getting Google ID Token
```javascript
// After Google sign-in with Firebase
const user = firebase.auth().currentUser;
const idToken = await user.getIdToken();

// Store token
localStorage.setItem('googleToken', idToken);
```

### Using Token for API Calls
```javascript
const token = localStorage.getItem('googleToken');

const response = await fetch('/api/admin/users', {
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }
});
```

## 🚨 Security Best Practices

1. **Secure Client Secret**: Never expose client secret in frontend code
2. **HTTPS Only**: Always use HTTPS in production
3. **Token Storage**: Store tokens securely (localStorage for web apps)
4. **Token Expiration**: Handle token expiration gracefully
5. **Role Validation**: Always validate user roles on backend

## 🔍 Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check if Google token is valid and not expired
   - Verify token format: `Bearer YOUR_TOKEN`
   - Ensure user has admin role

2. **403 Forbidden**
   - User doesn't have admin role
   - Check user permissions in database

3. **Google OAuth Errors**
   - Verify client ID and secret are correct
   - Check redirect URIs in Google Cloud Console
   - Ensure required APIs are enabled

### Debug Mode

Enable debug logging by setting environment variable:
```bash
DEBUG=true
```

## 📚 Additional Resources

- [Google OAuth 2.0 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [Google Identity Platform](https://developers.google.com/identity)
- [Firebase Authentication](https://firebase.google.com/docs/auth)
- [Google Cloud Console](https://console.cloud.google.com/) 