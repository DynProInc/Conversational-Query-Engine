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
from utils.auth import get_current_admin_user, users_db

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
@router.get("/users", response_model=List[User])
async def get_all_users(current_user = Depends(get_current_admin_user)):
    """Get all users in the system"""
    # Transform users_db to match the User model
    users = []
    for email, user_data in users_db.items():
        # Remove hashed_password from the response
        user_response = {
            "id": user_data["id"],
            "name": user_data["name"],
            "email": user_data["email"],
            "role": user_data["role"],
            "is_active": user_data["is_active"],
            "created_at": "2024-01-01T00:00:00Z",  # Mock creation date
            "last_login": "2024-01-15T10:30:00Z"   # Mock last login
        }
        users.append(user_response)
    return users

@router.put("/users/{user_id}", response_model=User)
async def update_user_role(
    user_id: str, 
    user_update: UserUpdate, 
    current_user = Depends(get_current_admin_user)
):
    """Update user role and status"""
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
    if user_update.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
    
    # Update user
    user_found["role"] = user_update.role
    if user_update.is_active is not None:
        user_found["is_active"] = user_update.is_active
    
    # Return user in the expected format
    return {
        "id": user_found["id"],
        "name": user_found["name"],
        "email": user_found["email"],
        "role": user_found["role"],
        "is_active": user_found["is_active"],
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-15T10:30:00Z"
    }

@router.get("/stats", response_model=SystemStats)
async def get_system_statistics(current_user = Depends(get_current_admin_user)):
    """Get system statistics"""
    # Calculate stats from analytics data
    total_queries = len(analytics_data["queries"])
    successful_queries = len([q for q in analytics_data["queries"] if q["success"]])
    failed_queries = total_queries - successful_queries
    success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
    
    # Mock system metrics
    system_uptime_hours = 24.5  # Mock uptime
    memory_usage_mb = 512.0     # Mock memory usage
    cpu_usage_percent = 15.5    # Mock CPU usage
    average_response_time_ms = 1250.0  # Mock response time
    
    return SystemStats(
        total_users=len(users_db),
        active_users=len([u for u in users_db.values() if u["is_active"]]),
        total_queries=total_queries,
        successful_queries=successful_queries,
        failed_queries=failed_queries,
        success_rate=success_rate,
        average_response_time_ms=average_response_time_ms,
        system_uptime_hours=system_uptime_hours,
        memory_usage_mb=memory_usage_mb,
        cpu_usage_percent=cpu_usage_percent
    )

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_user_analytics(current_user = Depends(get_current_admin_user)):
    """Get user analytics and system performance data"""
    
    # Mock user activity data (daily queries and users)
    user_activity = [
        {"name": "Mon", "queries": 65, "users": 24},
        {"name": "Tue", "queries": 59, "users": 22},
        {"name": "Wed", "queries": 80, "users": 28},
        {"name": "Thu", "queries": 81, "users": 29},
        {"name": "Fri", "queries": 56, "users": 20},
        {"name": "Sat", "queries": 55, "users": 18},
        {"name": "Sun", "queries": 40, "users": 15}
    ]
    
    # Mock query types data
    query_types = [
        {"name": "Text Queries", "value": 65},
        {"name": "Voice Queries", "value": 35}
    ]
    
    # Mock recent users data
    recent_users = []
    for email, user_data in users_db.items():
        recent_users.append({
            "id": user_data["id"],
            "name": user_data["name"],
            "email": user_data["email"],
            "lastActive": "10 minutes ago",
            "queries": 12
        })
    
    return AnalyticsResponse(
        userActivity=user_activity,
        queryTypes=query_types,
        recentUsers=recent_users
    )

# Health check for admin routes
@router.get("/health")
async def admin_health_check():
    """Health check for admin routes"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "admin_routes_available": True
    } 