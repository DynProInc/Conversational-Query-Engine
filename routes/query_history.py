from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from models.database import get_db, QueryHistory, SavedQuery
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["query_history"])

# Pydantic models for request/response
class QueryHistoryResponse(BaseModel):
    id: int
    user_id: str
    prompt: str
    sql_query: Optional[str]
    database: str
    timestamp: str
    model: str
    query_executed: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

class SavedQueryResponse(BaseModel):
    id: int
    user_id: str
    prompt: str
    name: str
    description: Optional[str]
    sql_query: Optional[str]
    database: str
    timestamp: str
    tags: Optional[List[str]]
    model: str
    query_executed: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

class SaveQueryRequest(BaseModel):
    user_id: str
    prompt: str
    name: str
    description: Optional[str] = ""
    sql_query: Optional[str] = None
    database: str = "default"
    tags: Optional[List[str]] = []

class UpdateSavedQueryRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

# Query History Routes
@router.get("/query_history", response_model=List[QueryHistoryResponse])
async def get_query_history(user_id: str, db: Session = Depends(get_db)):
    """Get query history for a specific user"""
    queries = db.query(QueryHistory).filter_by(user_id=user_id).order_by(QueryHistory.timestamp.desc()).all()
    return [{
        'id': q.id,
        'user_id': q.user_id,
        'prompt': q.prompt,
        'sql_query': q.sql_query,
        'database': q.database,
        'timestamp': q.timestamp.isoformat(),
        'model': q.model,
        'query_executed': '1' if q.query_executed else '0',
        'prompt_tokens': q.prompt_tokens,
        'completion_tokens': q.completion_tokens,
        'total_tokens': q.total_tokens,
        'input_cost': float(q.input_cost),
        'output_cost': float(q.output_cost),
        'total_cost': float(q.total_cost)
    } for q in queries]

@router.delete("/query_history/{query_id}")
async def delete_query_history(query_id: int, user_id: str, db: Session = Depends(get_db)):
    """Delete a specific query from history"""
    query = db.query(QueryHistory).filter_by(id=query_id, user_id=user_id).first()
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    db.delete(query)
    db.commit()
    return {"success": True}

# Saved Queries Routes
@router.get("/saved_queries", response_model=List[SavedQueryResponse])
async def get_saved_queries(user_id: str, db: Session = Depends(get_db)):
    """Get saved queries for a specific user"""
    queries = db.query(SavedQuery).filter_by(user_id=user_id).order_by(SavedQuery.timestamp.desc()).all()
    return [{
        'id': q.id,
        'user_id': q.user_id,
        'prompt': q.prompt,
        'name': q.name,
        'description': q.description,
        'sql_query': q.sql_query,
        'database': q.database,
        'timestamp': q.timestamp.isoformat(),
        'tags': q.tags,
        'model': q.model,
        'query_executed': '1' if q.query_executed else '0',
        'prompt_tokens': q.prompt_tokens,
        'completion_tokens': q.completion_tokens,
        'total_tokens': q.total_tokens,
        'input_cost': float(q.input_cost),
        'output_cost': float(q.output_cost),
        'total_cost': float(q.total_cost)
    } for q in queries]

@router.post("/saved_queries", response_model=SavedQueryResponse, status_code=201)
async def save_query(request: SaveQueryRequest, db: Session = Depends(get_db)):
    """Save a new query"""
    new_query = SavedQuery(
        user_id=request.user_id,
        prompt=request.prompt,
        name=request.name,
        description=request.description,
        sql_query=request.sql_query,
        database=request.database,
        tags=request.tags
    )
    
    db.add(new_query)
    db.commit()
    db.refresh(new_query)
    
    return {
        'id': new_query.id,
        'user_id': new_query.user_id,
        'prompt': new_query.prompt,
        'name': new_query.name,
        'description': new_query.description,
        'sql_query': new_query.sql_query,
        'database': new_query.database,
        'timestamp': new_query.timestamp.isoformat(),
        'tags': new_query.tags,
        'model': new_query.model,
        'query_executed': '1' if new_query.query_executed else '0',
        'prompt_tokens': new_query.prompt_tokens,
        'completion_tokens': new_query.completion_tokens,
        'total_tokens': new_query.total_tokens,
        'input_cost': float(new_query.input_cost),
        'output_cost': float(new_query.output_cost),
        'total_cost': float(new_query.total_cost)
    }

@router.put("/saved_queries/{query_id}")
async def update_saved_query(query_id: int, request: UpdateSavedQueryRequest, user_id: str, db: Session = Depends(get_db)):
    """Update a saved query"""
    query = db.query(SavedQuery).filter_by(id=query_id, user_id=user_id).first()
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    if request.name is not None:
        query.name = request.name
    if request.description is not None:
        query.description = request.description
    if request.tags is not None:
        query.tags = request.tags
    
    db.commit()
    return {"success": True}

@router.delete("/saved_queries/{query_id}")
async def delete_saved_query(query_id: int, user_id: str, db: Session = Depends(get_db)):
    """Delete a saved query"""
    query = db.query(SavedQuery).filter_by(id=query_id, user_id=user_id).first()
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    
    db.delete(query)
    db.commit()
    return {"success": True} 