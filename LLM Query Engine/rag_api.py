#!/usr/bin/env python3
"""
RAG API - Direct FastAPI integration with rag_embedding.py

This module provides FastAPI endpoints that directly call the RAG embedding functionality
from rag_embedding.py, exposing the same commands as API endpoints:
- /rag/rebuild: Rebuild collections for a client or all clients
- /rag/drop: Drop collections for a client or all clients
- /rag/query: Query a collection
- /rag/enhanced: Enhanced query with SQL context
- /rag/stats: Get collection statistics
"""

import os
import sys
import logging
import importlib.util
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_api")

# Create router
router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    responses={404: {"description": "Not found"}},
)

# Initialize RAG Manager
rag_manager = None
error_message = None

def init_rag_manager():
    """Initialize RAG Manager from rag_embedding.py"""
    global rag_manager, error_message
    
    try:
        # Get the absolute path to the current file and milvus-setup directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        milvus_setup_dir = os.path.join(current_dir, "milvus-setup")
        
        # Import the module directly using importlib
        spec = importlib.util.spec_from_file_location(
            "rag_embedding", 
            os.path.join(milvus_setup_dir, "rag_embedding.py")
        )
        rag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_module)
        
        # Create RAGManager instance
        rag_manager = rag_module.RAGManager()
        logger.info("RAG Manager initialized successfully")
        return True
    except Exception as e:
        error_message = str(e)
        logger.error(f"Failed to initialize RAG Manager: {error_message}")
        return False

# Initialize RAG Manager on module load
init_success = init_rag_manager()

# Define Pydantic models for request/response validation
class RebuildRequest(BaseModel):
    client_id: Optional[str] = Field(None, description="Client ID to rebuild collection for")
    all_clients: bool = Field(False, description="Rebuild all client collections")

class DropRequest(BaseModel):
    client_id: Optional[str] = Field(None, description="Client ID to drop collection for")
    all_clients: bool = Field(False, description="Drop all client collections")

class QueryRequest(BaseModel):
    client_id: str = Field(..., description="Client ID for the collection")
    query: str = Field(..., description="Query text")
    top_k: int = Field(5, description="Number of results to return")

class QueryResult(BaseModel):
    table_name: str
    column_name: str
    description: str
    score: float

class QueryResponse(BaseModel):
    success: bool
    message: str
    results: List[QueryResult] = []

class EnhancedQueryResponse(BaseModel):
    success: bool
    message: str
    results: List[QueryResult] = []
    sql_context: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    success: bool
    message: str
    collections: Optional[List[Dict[str, Any]]] = None

# Helper function to get RAG Manager
def get_rag_manager():
    """Get RAG Manager instance or raise exception if not initialized"""
    if not rag_manager:
        raise HTTPException(
            status_code=500, 
            detail=f"RAG system not available: {error_message or 'Unknown error'}"
        )
    return rag_manager

@router.post("/rebuild", response_model=StatusResponse)
async def rebuild_collection(request: RebuildRequest):
    """Rebuild collection for a client or all clients"""
    manager = get_rag_manager()
    
    if not request.client_id and not request.all_clients:
        raise HTTPException(
            status_code=400, 
            detail="Either client_id or all_clients must be specified"
        )
    
    if request.all_clients:
        # Rebuild all client collections
        successful = []
        failed = []
        
        # Get all active clients
        clients = manager.get_active_clients()
        client_dict = {c_id: path for c_id, path in clients}
        
        # Process each client
        for client_id, dict_path in client_dict.items():
            if not os.path.exists(dict_path):
                failed.append((client_id, "Dictionary file not found"))
                continue
            
            # Run rebuild
            success, message = manager.rebuild_client(client_id, dict_path)
            
            if success:
                successful.append(client_id)
            else:
                failed.append((client_id, message))
        
        # Create summary message
        message = f"Successfully rebuilt {len(successful)} clients"
        if successful:
            message += f": {', '.join(successful)}"
        
        if failed:
            message += f". Failed to rebuild {len(failed)} clients: "
            message += ", ".join([f"{c_id} ({err})" for c_id, err in failed])
        
        return {
            "success": len(successful) > 0,
            "message": message
        }
    else:
        # Rebuild specific client
        client_id = request.client_id
        
        # Get client dictionary path
        clients = manager.get_active_clients()
        client_dict = {c_id: path for c_id, path in clients}
        
        if client_id not in client_dict:
            raise HTTPException(
                status_code=404, 
                detail=f"Client {client_id} not found in registry"
            )
        
        dict_path = client_dict[client_id]
        
        if not os.path.exists(dict_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Dictionary file not found at {dict_path}"
            )
        
        # Run rebuild
        success, message = manager.rebuild_client(client_id, dict_path)
        
        return {
            "success": success,
            "message": message
        }

@router.post("/drop", response_model=StatusResponse)
async def drop_collection(request: DropRequest):
    """Drop collection for a client or all clients"""
    manager = get_rag_manager()
    
    if not request.client_id and not request.all_clients:
        raise HTTPException(
            status_code=400, 
            detail="Either client_id or all_clients must be specified"
        )
    
    if request.all_clients:
        # Drop all collections
        success, message = manager.drop_collections()
    else:
        # Drop specific client collection
        success, message = manager.drop_collections(request.client_id)
    
    return {
        "success": success,
        "message": message
    }

@router.post("/query", response_model=QueryResponse)
async def query_collection(request: QueryRequest):
    """Query a collection"""
    manager = get_rag_manager()
    
    success, message, results = manager.query(
        request.client_id, 
        request.query, 
        request.top_k
    )
    
    return {
        "success": success,
        "message": message,
        "results": results if success and results else []
    }

@router.post("/enhanced", response_model=EnhancedQueryResponse)
async def enhanced_query(request: QueryRequest):
    """Enhanced query with SQL context"""
    manager = get_rag_manager()
    
    success, message, results, sql_context = manager.enhanced_query(
        request.client_id, 
        request.query, 
        request.top_k
    )
    
    return {
        "success": success,
        "message": message,
        "results": results if success and results else [],
        "sql_context": sql_context if success and sql_context else None
    }

@router.get("/stats", response_model=StatusResponse)
async def get_stats():
    """Get collection statistics"""
    manager = get_rag_manager()
    
    try:
        # The get_collection_stats method returns (success, stats) where stats can be either a list or an error message
        success, result = manager.get_collection_stats()
        
        if success:
            # If successful, result is a list of collection stats
            return {
                "success": success,
                "message": f"Found {len(result)} collections",
                "collections": result
            }
        else:
            # If not successful, result is an error message
            return {
                "success": success,
                "message": result,
                "collections": []
            }
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        return {
            "success": False,
            "message": f"Error getting collection stats: {str(e)}",
            "collections": []
        }

# For standalone testing
if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    
    # Create a FastAPI app
    app = FastAPI(
        title="RAG API Test Server",
        description="Test server for RAG API endpoints",
        version="1.0.0"
    )
    
    # Include the router
    app.include_router(router)
    
    # Print the available routes
    print("\nAvailable routes:")
    for route in app.routes:
        print(f"  {route.path} [{', '.join(route.methods)}]")
    
    # Run the server
    print("\nStarting RAG API test server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
