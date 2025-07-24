from pydantic import BaseModel
from typing import Optional

class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: str
    password: str
    name: str

class GoogleLogin(BaseModel):
    id_token: str

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    id: str
    email: str
    name: str
    role: str
    is_active: bool = True
    picture: Optional[str] = None
    auth_provider: str = "email"  # "email" or "google" 