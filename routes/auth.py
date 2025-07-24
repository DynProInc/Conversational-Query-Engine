from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from datetime import timedelta
from models.auth import UserLogin, GoogleLogin, Token, User
from utils.auth import (
    authenticate_user, 
    get_or_create_google_user,
    create_access_token, 
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from config.google_oauth import google_oauth_config

router = APIRouter(prefix="/api/auth", tags=["authentication"])

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    user = authenticate_user(user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=User)
async def read_users_me(current_user = Depends(get_current_user)):
    return current_user

@router.get("/google/login")
async def google_login():
    """Get Google OAuth authorization URL"""
    try:
        auth_url = google_oauth_config.get_authorization_url()
        return {"auth_url": auth_url}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate Google OAuth URL: {str(e)}"
        )

@router.get("/google/callback")
async def google_callback(code: str):
    """Handle Google OAuth callback"""
    try:
        # Exchange authorization code for user info
        google_user_info = google_oauth_config.exchange_code_for_token(code)
        
        # Get or create user
        user = get_or_create_google_user(google_user_info)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]}, expires_delta=access_token_expires
        )
        
        # Redirect to frontend with token
        frontend_url = "https://conversational-query-engine-pp4v.onrender.com/admin"
        return RedirectResponse(
            url=f"{frontend_url}?token={access_token}&user={user['name']}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google OAuth callback failed: {str(e)}"
        )

@router.post("/google/token", response_model=Token)
async def google_token_login(google_login: GoogleLogin):
    """Login with Google ID token"""
    try:
        # Verify Google ID token
        google_user_info = google_oauth_config.verify_id_token(google_login.id_token)
        
        # Get or create user
        user = get_or_create_google_user(google_user_info)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Google token login failed: {str(e)}"
        ) 