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
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Initialize router
router = APIRouter(prefix="/api/admin", tags=["admin"])

# Security
security = HTTPBearer()

# In-memory storage (in production, this would be a database)
users_db = {
    "admin": {
        "id": "admin",
        "username": "admin",
        "email": "admin@example.com",
        "role": "admin",
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-01T00:00:00Z",
        "is_active": True
    },
    "user1": {
        "id": "user1",
        "username": "user1",
        "email": "user1@example.com",
        "role": "user",
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-01T00:00:00Z",
        "is_active": True
    },
    "user2": {
        "id": "user2",
        "username": "user2",
        "email": "user2@example.com",
        "role": "user",
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-01T00:00:00Z",
        "is_active": True
    }
}

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
    username: str
    email: str
    role: str
    created_at: str
    last_login: str
    is_active: bool

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
    total_users: int
    total_queries: int
    queries_by_model: Dict[str, int]
    queries_by_day: Dict[str, int]
    user_analytics: List[UserAnalytics]
    system_performance: Dict[str, Any]

# Authentication dependency
async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin token - in production, implement proper JWT validation"""
    token = credentials.credentials
    # For demo purposes, accept any token that starts with 'admin_'
    if not token.startswith('admin_'):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return token

# Admin routes
@router.get("/users", response_model=List[User])
async def get_all_users(token: str = Depends(verify_admin_token)):
    """Get all users in the system"""
    return list(users_db.values())

@router.put("/users/{user_id}", response_model=User)
async def update_user_role(
    user_id: str, 
    user_update: UserUpdate, 
    token: str = Depends(verify_admin_token)
):
    """Update user role and status"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate role
    valid_roles = ["admin", "user", "moderator"]
    if user_update.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
    
    # Update user
    users_db[user_id]["role"] = user_update.role
    if user_update.is_active is not None:
        users_db[user_id]["is_active"] = user_update.is_active
    
    return users_db[user_id]

@router.get("/stats", response_model=SystemStats)
async def get_system_statistics(token: str = Depends(verify_admin_token)):
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
async def get_user_analytics(token: str = Depends(verify_admin_token)):
    """Get user analytics and system performance data"""
    
    # Calculate queries by model
    model_counts = Counter()
    for query in analytics_data["queries"]:
        model_counts[query["model"]] += 1
    
    # Calculate queries by day
    day_counts = Counter()
    for query in analytics_data["queries"]:
        date = query["timestamp"].split("T")[0]
        day_counts[date] += 1
    
    # Calculate user analytics
    user_analytics = []
    for user_id, user_data in users_db.items():
        user_queries = [q for q in analytics_data["queries"] if q["user_id"] == user_id]
        
        if user_queries:
            total_queries = len(user_queries)
            successful_queries = len([q for q in user_queries if q["success"]])
            failed_queries = total_queries - successful_queries
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            
            # Find favorite model
            user_models = [q["model"] for q in user_queries]
            favorite_model = max(set(user_models), key=user_models.count) if user_models else "None"
            
            # Last activity
            last_activity = max(q["timestamp"] for q in user_queries)
            
            user_analytics.append(UserAnalytics(
                user_id=user_id,
                username=user_data["username"],
                total_queries=total_queries,
                successful_queries=successful_queries,
                failed_queries=failed_queries,
                success_rate=success_rate,
                last_activity=last_activity,
                favorite_model=favorite_model,
                average_response_time_ms=1250.0  # Mock value
            ))
        else:
            # User with no queries
            user_analytics.append(UserAnalytics(
                user_id=user_id,
                username=user_data["username"],
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                success_rate=0.0,
                last_activity="Never",
                favorite_model="None",
                average_response_time_ms=0.0
            ))
    
    # System performance data
    system_performance = {
        "uptime_hours": 24.5,
        "memory_usage_mb": 512.0,
        "cpu_usage_percent": 15.5,
        "disk_usage_percent": 45.2,
        "active_connections": 12,
        "requests_per_minute": 8.5
    }
    
    return AnalyticsResponse(
        total_users=len(users_db),
        total_queries=len(analytics_data["queries"]),
        queries_by_model=dict(model_counts),
        queries_by_day=dict(day_counts),
        user_analytics=user_analytics,
        system_performance=system_performance
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