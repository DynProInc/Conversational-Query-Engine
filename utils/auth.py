from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

# Mock user database (replace with your actual database)
users_db = {
    "arman.khan@dynpro.com": {
        "id": "1",
        "email": "arman.khan@dynpro.com",
        "name": "Arman Khan",
        "hashed_password": pwd_context.hash("Password123!"),
        "role": "admin",
        "is_active": True
    },
    "admin@example.com": {
        "id": "admin",
        "email": "admin@example.com",
        "name": "System Admin",
        "hashed_password": pwd_context.hash("admin123!"),
        "role": "admin",
        "is_active": True
    },
    "user1@example.com": {
        "id": "user1",
        "email": "user1@example.com",
        "name": "Test User 1",
        "hashed_password": pwd_context.hash("user123!"),
        "role": "user",
        "is_active": True
    },
    "user2@example.com": {
        "id": "user2",
        "email": "user2@example.com",
        "name": "Test User 2",
        "hashed_password": pwd_context.hash("user123!"),
        "role": "user",
        "is_active": True
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(email: str, password: str):
    user = users_db.get(email)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = users_db.get(email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_admin_user(current_user = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user 