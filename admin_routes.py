"""
Admin API Routes for LLM SQL Query Engine

This module provides admin endpoints for user management, system statistics, and analytics.
"""
import os
import json
import time
import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from utils.google_auth import get_current_admin_user_google, users_db

# Initialize router
router = APIRouter(prefix="/api/admin", tags=["admin"])

# Mock analytics data (in production, this would come from actual usage logs)
analytics_data = {
    "queries": [
        {"user_id": "user1", "timestamp": "2024-01-15T10:30:00Z", "model": "gpt-4", "success": True},
        {"user_id": "user2", "timestamp": "2024-01-15T11:15:00Z", "model": "claude", "success": True},
        {"user_id": "user1", "timestamp": "2024-01-15T14:20:00Z", "model": "gpt-4", "success": False},
        {"user_id": "admin", "timestamp": "2024-01-15T16:45:00Z", "model": "gpt-4", "success": True},
    ],
    "logins": [
        {"user_id": "user1", "timestamp": "2024-01-15T09:00:00Z"},
        {"user_id": "user2", "timestamp": "2024-01-15T10:00:00Z"},
        {"user_id": "admin", "timestamp": "2024-01-15T08:00:00Z"},
    ]
}

# Pydantic models
class User(BaseModel):
    id: str
    name: str
    email: str
    role: str
    is_active: bool
    created_at: Optional[str] = None
    last_login: Optional[str] = None

class UserUpdate(BaseModel):
    role: str
    is_active: Optional[bool] = None

class SystemStats(BaseModel):
    total_users: int
    active_users: int
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    average_response_time_ms: float
    system_uptime_hours: float
    memory_usage_mb: float
    cpu_usage_percent: float

class UserAnalytics(BaseModel):
    user_id: str
    username: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    last_activity: str
    favorite_model: str
    average_response_time_ms: float

class AnalyticsResponse(BaseModel):
    userActivity: List[Dict[str, Any]]
    queryTypes: List[Dict[str, Any]]
    recentUsers: List[Dict[str, Any]]

# Admin routes
@router.get("/users")
async def get_all_users(current_user = Depends(get_current_admin_user_google)):
    """Get all users in the system (admin only)"""
    # Transform users_db to match the expected format
    users = []
    for email, user_data in users_db.items():
        user_response = {
            "id": user_data["id"],
            "name": user_data["name"],
            "email": user_data["email"],
            "role": user_data["role"],
            "status": "active" if user_data["is_active"] else "inactive",
            "lastActive": user_data["last_login"],
            "queries": 12,  # Mock query count
            "dateCreated": user_data["created_at"]
        }
        users.append(user_response)
    return {"users": users}

@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str, 
    role: str, 
    current_user = Depends(get_current_admin_user_google)
):
    """Update user role (admin only)"""
    # Find user by ID in users_db
    user_found = None
    for email, user_data in users_db.items():
        if user_data["id"] == user_id:
            user_found = user_data
            break
    
    if not user_found:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate role
    valid_roles = ["admin", "user", "moderator"]
    if role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
    
    # Update user role
    user_found["role"] = role
    
    return {"message": "Role updated successfully"}

@router.get("/stats")
async def get_system_statistics(current_user = Depends(get_current_admin_user_google)):
    """Get system statistics (admin only)"""
    return {
        "totalUsers": len(users_db),
        "userGrowth": 12,
        "totalQueries": 24567,
        "queryGrowth": 18,
        "avgResponseTime": "0.8s",
        "responseTimeImprovement": 5
    }

@router.get("/analytics")
async def get_user_analytics(current_user = Depends(get_current_admin_user_google)):
    """Get user analytics (admin only)"""
    return {
        "userActivity": [
            {"name": "Mon", "queries": 65, "users": 24},
            {"name": "Tue", "queries": 59, "users": 22},
            {"name": "Wed", "queries": 80, "users": 28},
            {"name": "Thu", "queries": 81, "users": 29},
            {"name": "Fri", "queries": 56, "users": 20},
            {"name": "Sat", "queries": 55, "users": 18},
            {"name": "Sun", "queries": 40, "users": 15}
        ],
        "queryTypes": [
            {"name": "Text Queries", "value": 65},
            {"name": "Voice Queries", "value": 35}
        ],
        "recentUsers": [
            {"id": "1", "name": "Arman Khan", "email": "arman.khan@dynpro.com", "lastActive": "10 minutes ago", "queries": 12},
            {"id": "admin", "name": "System Admin", "email": "admin@example.com", "lastActive": "10 minutes ago", "queries": 12},
            {"id": "user1", "name": "Test User 1", "email": "user1@example.com", "lastActive": "10 minutes ago", "queries": 12},
            {"id": "user2", "name": "Test User 2", "email": "user2@example.com", "lastActive": "10 minutes ago", "queries": 12}
        ]
    }

# Health check for admin routes
@router.get("/health")
async def admin_health_check():
    """Health check for admin routes"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "admin_routes_available": True
    } 