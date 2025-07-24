"""
Google OAuth Configuration and Utilities

This module handles Google OAuth authentication for the LLM Query Engine.
"""
import os
import json
from typing import Optional, Dict, Any
from google.oauth2 import id_token
from google.auth.transport import requests
from google_auth_oauthlib.flow import Flow
from fastapi import HTTPException, status

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://conversational-query-engine-pp4v.onrender.com/api/auth/google/callback")

# OAuth scopes
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile"
]

class GoogleOAuthConfig:
    """Google OAuth configuration and utilities"""
    
    def __init__(self):
        self.client_id = GOOGLE_CLIENT_ID
        self.client_secret = GOOGLE_CLIENT_SECRET
        self.redirect_uri = GOOGLE_REDIRECT_URI
        self.scopes = SCOPES
    
    def get_authorization_url(self) -> str:
        """Get Google OAuth authorization URL"""
        if not self.client_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth not configured"
            )
        
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri]
                }
            },
            scopes=self.scopes
        )
        
        flow.redirect_uri = self.redirect_uri
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        return authorization_url
    
    def verify_id_token(self, id_token_string: str) -> Dict[str, Any]:
        """Verify Google ID token and return user info"""
        try:
            # Verify the token
            idinfo = id_token.verify_oauth2_token(
                id_token_string, 
                requests.Request(), 
                self.client_id
            )
            
            # ID token is valid. Get the user's Google Account ID and profile info
            user_id = idinfo['sub']
            user_email = idinfo['email']
            user_name = idinfo.get('name', '')
            user_picture = idinfo.get('picture', '')
            
            return {
                'user_id': user_id,
                'email': user_email,
                'name': user_name,
                'picture': user_picture,
                'email_verified': idinfo.get('email_verified', False)
            }
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid ID token: {str(e)}"
            )
    
    def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token and user info"""
        if not self.client_id or not self.client_secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth not configured"
            )
        
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri]
                }
            },
            scopes=self.scopes
        )
        
        flow.redirect_uri = self.redirect_uri
        
        try:
            # Exchange authorization code for tokens
            flow.fetch_token(code=authorization_code)
            
            # Get user info from Google
            session = flow.authorized_session()
            user_info = session.get('https://www.googleapis.com/oauth2/v2/userinfo').json()
            
            return {
                'user_id': user_info['id'],
                'email': user_info['email'],
                'name': user_info.get('name', ''),
                'picture': user_info.get('picture', ''),
                'email_verified': user_info.get('verified_email', False)
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to exchange code for token: {str(e)}"
            )

# Global OAuth config instance
google_oauth_config = GoogleOAuthConfig() 