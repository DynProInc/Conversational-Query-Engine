from google.oauth2 import id_token
from google.auth.transport import requests
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import uuid
from datetime import datetime

# YOUR CLIENT ID
GOOGLE_CLIENT_ID = "332727032541-ero3io8bdeejg1843b18teldibqe7sr8.apps.googleusercontent.com"

# Security scheme
security = HTTPBearer()

# Mock user database (replace with your actual database)
users_db = {
    "arman.khan@dynpro.com": {
        "id": "1",
        "email": "arman.khan@dynpro.com",
        "name": "Arman Khan",
        "role": "admin",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-15T10:30:00Z"
    },
    "dhruvanc@gmail.com": {
        "id": "2",
        "email": "dhruvanc@gmail.com",
        "name": "Dhruva Navin Chander",
        "role": "admin",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-15T10:30:00Z"
    },
    "admin@example.com": {
        "id": "admin",
        "email": "admin@example.com",
        "name": "System Admin",
        "role": "admin",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-15T10:30:00Z"
    },
    "user1@example.com": {
        "id": "user1",
        "email": "user1@example.com",
        "name": "Test User 1",
        "role": "user",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-15T10:30:00Z"
    },
    "user2@example.com": {
        "id": "user2",
        "email": "user2@example.com",
        "name": "Test User 2",
        "role": "user",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-15T10:30:00Z"
    }
}

async def verify_google_token(token: str):
    """Verify Google OAuth token and return user info"""
    try:
        # Verify the token
        idinfo = id_token.verify_oauth2_token(
            token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )
        
        # Extract user info
        user_email = idinfo['email']
        user_name = idinfo.get('name', '')
        
        # Check if user exists in your database
        user = users_db.get(user_email)
        if not user:
            # Create new user
            user = create_user_from_google(user_email, user_name)
        else:
            # Update last login
            user["last_login"] = datetime.utcnow().isoformat()
        
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google token: {str(e)}"
        )

def create_user_from_google(email: str, name: str):
    """Create new user from Google OAuth"""
    user = {
        "id": str(uuid.uuid4()),
        "email": email,
        "name": name,
        "role": "user",  # Default role
        "is_active": True,
        "created_at": datetime.utcnow().isoformat(),
        "last_login": datetime.utcnow().isoformat()
    }
    
    # Save to database
    users_db[email] = user
    return user

async def get_current_user_google(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from Google OAuth token"""
    try:
        # Extract token from Authorization header
        token = credentials.credentials
        
        # Verify Google token
        user = await verify_google_token(token)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_admin_user_google(current_user = Depends(get_current_user_google)):
    """Check if current user is admin"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user 