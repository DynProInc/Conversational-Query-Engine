"""
Embedding API - FastAPI integration with both Gemini and OpenAI embedding tests

This module provides FastAPI endpoints for working with embeddings:
- /embeddings/build: Build embeddings for a client using either Gemini or OpenAI
- /embeddings/drop: Drop embedding collections for a client
- /embeddings/query: Query embeddings for a client
- /embeddings/stats: Get embedding collection statistics
"""

import os
import sys
import json
import logging
import importlib.util
from typing import Dict, List, Optional, Any, Union, Literal
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding_api")

# Create router
router = APIRouter(
    prefix="/embeddings",
    tags=["embeddings"],
    responses={404: {"description": "Not found"}}
)

# Initialize embedding test classes
gemini_test_module = None
openai_test_module = None
error_message = None

def load_embedding_modules():
    """Load the embedding test modules"""
    global gemini_test_module, openai_test_module, error_message
    
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load Gemini embedding test module
        gemini_path = os.path.join(current_dir, "gemini_embedding_test.py")
        spec = importlib.util.spec_from_file_location("gemini_embedding_test", gemini_path)
        gemini_test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gemini_test_module)
        
        # Load OpenAI embedding test module
        openai_path = os.path.join(current_dir, "openai_embedding_test.py")
        spec = importlib.util.spec_from_file_location("openai_embedding_test", openai_path)
        openai_test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(openai_test_module)
        
        logger.info("Embedding test modules loaded successfully")
        return True
    except Exception as e:
        error_message = str(e)
        logger.error(f"Failed to load embedding test modules: {error_message}")
        return False

# Load the modules on startup
init_success = load_embedding_modules()

# Define Pydantic models for request/response validation
class BuildRequest(BaseModel):
    client_id: str = Field(..., description="Client ID to build embeddings for")
    csv_path: Optional[str] = Field(None, description="Path to CSV file containing schema data (optional, will use client registry if not provided)")
    model_type: Literal["gemini", "openai"] = Field("gemini", description="Embedding model type")
    model_name: Optional[str] = Field(None, description="Specific model name (e.g., text-embedding-3-large)")

class DropRequest(BaseModel):
    client_id: str = Field(..., description="Client ID to drop collections for")
    model_type: Literal["gemini", "openai", "all"] = Field(..., description="Embedding model type to drop")

class QueryRequest(BaseModel):
    client_id: str = Field(..., description="Client ID for the collection")
    query_text: str = Field(..., description="Query text")
    model_type: Literal["gemini", "openai"] = Field(..., description="Embedding model type")
    top_k: int = Field(5, description="Number of results to return")
    model_name: Optional[str] = Field(None, description="Specific model name")

class StatsRequest(BaseModel):
    client_id: str = Field(..., description="Client ID")
    model_type: Literal["gemini", "openai", "all"] = Field(..., description="Embedding model type")

class EmbeddingMatch(BaseModel):
    db_schema: str
    table_name: str
    column_name: str
    data_type: str
    description: str
    distinct_values: Optional[str] = None
    score: float

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

# Helper functions to get embedding test instances
def get_gemini_test(client_id: str):
    """Get a Gemini embedding test instance"""
    if not gemini_test_module:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini embedding test module not available: {error_message or 'Unknown error'}"
        )
    return gemini_test_module.GeminiEmbeddingTest(client_id=client_id)

def get_openai_test(client_id: str, model_name: Optional[str] = None):
    """Get an OpenAI embedding test instance"""
    if not openai_test_module:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI embedding test module not available: {error_message or 'Unknown error'}"
        )
    return openai_test_module.OpenAIEmbeddingTest(client_id=client_id, model_name=model_name)

def get_client_dictionary_path(client_id: str) -> Optional[str]:
    """Get the path to the client's data dictionary using the client manager."""
    try:
        # First try to use the client manager
        try:
            # Import here to avoid circular imports
            from config.client_manager import ClientManager
            client_manager = ClientManager()
            dict_path = client_manager.get_data_dictionary_path(client_id)
            logger.info(f"Using client manager path for {client_id}: {dict_path}")
            return dict_path
        except Exception as e:
            logger.warning(f"Could not use client manager: {e}, falling back to direct registry access")
        
        # Check if we're running in Docker
        in_docker = os.path.exists('/.dockerenv') or os.path.exists('/app')
        
        if in_docker:
            # In Docker, use the fixed path structure
            docker_path = f"/app/LLM Query Engine/config/clients/data_dictionaries/{client_id}/{client_id}_dictionary.csv"
            if os.path.exists(docker_path):
                logger.info(f"Using Docker path for {client_id}: {docker_path}")
                return docker_path
        
        # Fall back to direct registry access
        current_dir = os.path.dirname(os.path.abspath(__file__))
        registry_path = os.path.join(current_dir, "config", "clients", "client_registry.csv")

        if not os.path.exists(registry_path):
            logger.error(f"Client registry not found at {registry_path}")
            return None

        with open(registry_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if row.get('client_id') == client_id and row.get('active').lower() == 'true':
                    return row.get('data_dictionary_path')
        
        logger.warning(f"Client '{client_id}' not found or is not active in the registry.")
        return None
    except Exception as e:
        logger.error(f"Error getting client dictionary path: {e}")
        return None

@router.post("/build", response_model=StandardResponse)
async def post_build_embeddings(request: BuildRequest):
    """Build embeddings for a client using specified model type via POST"""
    csv_path = get_client_dictionary_path(request.client_id)
    if not csv_path:
        return StandardResponse(
            success=False,
            message=f"Could not find data dictionary path for client '{request.client_id}' in registry."
        )
    return await _build_embeddings(request.client_id, csv_path, request.model_type, request.model_name)

@router.get("/build", response_model=StandardResponse)
async def get_build_embeddings(client_id: str, model_type: str = "gemini", model_name: Optional[str] = None):
    """Build embeddings via GET with query parameters"""
    csv_path = get_client_dictionary_path(client_id)
    if not csv_path:
        return StandardResponse(
            success=False,
            message=f"Could not find data dictionary path for client '{client_id}' in registry."
        )

    if model_type not in ["gemini", "openai"]:
        return StandardResponse(
            success=False,
            message=f"Invalid model_type: {model_type}. Must be one of: gemini, openai"
        )
    return await _build_embeddings(client_id, csv_path, model_type, model_name)

async def _build_embeddings(client_id: str, csv_path: str, model_type: str, model_name: Optional[str] = None):
    """Internal method to build embeddings"""
    try:
        # Validate CSV path
        if not os.path.exists(csv_path):
            return StandardResponse(
                success=False,
                message=f"CSV file not found at {csv_path}"
            )
        
        if model_type == "gemini":
            # Use Gemini embedding model
            test = get_gemini_test(client_id)
        else:
            # Use OpenAI embedding model
            test = get_openai_test(client_id, model_name)
        
        # Build embeddings
        success, message = test.build_embeddings(csv_path)
        
        return StandardResponse(
            success=success,
            message=message
        )
    except Exception as e:
        logger.error(f"Error building embeddings: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Error building embeddings: {str(e)}"
        )

@router.post("/drop", response_model=StandardResponse)
async def post_drop_embeddings(request: DropRequest):
    """Drop embedding collections for a client via POST"""
    return await _drop_embeddings(request.client_id, request.model_type)

@router.get("/drop", response_model=StandardResponse)
async def get_drop_embeddings(client_id: str, model_type: str = "all"):
    """Drop embedding collections via GET with query parameters"""
    if model_type not in ["gemini", "openai", "all"]:
        return StandardResponse(
            success=False,
            message=f"Invalid model_type: {model_type}. Must be one of: gemini, openai, all"
        )
    return await _drop_embeddings(client_id, model_type)

async def _drop_embeddings(client_id: str, model_type: str):
    """Internal method to drop embedding collections"""
    try:
        from pymilvus import connections, utility
        
        # Connect to Milvus
        connections.connect("default", host=os.environ.get("MILVUS_HOST", "localhost"), port=os.environ.get("MILVUS_PORT", "19530"))
        
        collections_dropped = []
        
        if model_type in ["gemini", "all"]:
            # Drop Gemini collection
            gemini_collection = f"{client_id}_gemini_embedding_test"
            if utility.has_collection(gemini_collection):
                utility.drop_collection(gemini_collection)
                collections_dropped.append(gemini_collection)
        
        if model_type in ["openai", "all"]:
            # Drop OpenAI collection
            openai_collection = f"{client_id}_openai_embedding_test"
            if utility.has_collection(openai_collection):
                utility.drop_collection(openai_collection)
                collections_dropped.append(openai_collection)
        
        if collections_dropped:
            return StandardResponse(
                success=True,
                message=f"Successfully dropped collections: {', '.join(collections_dropped)}",
                data={"collections_dropped": collections_dropped}
            )
        else:
            return StandardResponse(
                success=True,
                message=f"No collections found to drop for client {client_id} with model type {model_type}",
                data={"collections_dropped": []}
            )
    except Exception as e:
        logger.error(f"Error dropping collections: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Error dropping collections: {str(e)}"
        )

@router.post("/query", response_model=StandardResponse)
async def post_query_embeddings(request: QueryRequest):
    """Query embeddings for a client via POST"""
    return await _query_embeddings(request.client_id, request.query_text, request.model_type, request.top_k, request.model_name)

@router.get("/query", response_model=StandardResponse)
async def get_query_embeddings(client_id: str, query: str, model_type: str = "gemini", top_k: int = 5, model_name: Optional[str] = None):
    """Query embeddings via GET with query parameters"""
    if model_type not in ["gemini", "openai"]:
        return StandardResponse(
            success=False,
            message=f"Invalid model_type: {model_type}. Must be one of: gemini, openai"
        )
    return await _query_embeddings(client_id, query, model_type, top_k, model_name)

async def _query_embeddings(client_id: str, query_text: str, model_type: str, top_k: int = 5, model_name: Optional[str] = None):
    """Internal method to query embeddings"""
    try:
        if model_type == "gemini":
            # Use Gemini embedding model
            test = get_gemini_test(client_id)
        else:
            # Use OpenAI embedding model
            test = get_openai_test(client_id, model_name)
        
        # Query embeddings
        success, message, matches = test.query(query_text, top_k)
        
        # Format matches as proper response objects
        formatted_matches = []
        if success and matches:
            for match in matches:
                formatted_matches.append({
                    "db_schema": match.get("db_schema", ""),
                    "table_name": match.get("table_name", ""),
                    "column_name": match.get("column_name", ""),
                    "data_type": match.get("data_type", ""),
                    "description": match.get("description", ""),
                    "distinct_values": match.get("distinct_values", ""),
                    "score": match.get("score", 0.0)
                })
        
        return StandardResponse(
            success=success,
            message=message,
            data={"matches": formatted_matches}
        )
    except Exception as e:
        logger.error(f"Error querying embeddings: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Error querying embeddings: {str(e)}"
        )

@router.post("/stats", response_model=StandardResponse)
async def post_stats(request: StatsRequest):
    """Get stats for embedding collections via POST"""
    return await _get_stats(request.client_id, request.model_type)

@router.get("/stats", response_model=StandardResponse)
async def get_stats():
    """Get stats for all embedding collections"""
    # Get stats for all collections using 'all' as the model_type
    try:
        from pymilvus import connections, utility, Collection
        
        # Connect to Milvus
        connections.connect("default", host=os.environ.get("MILVUS_HOST", "localhost"), port=os.environ.get("MILVUS_PORT", "19530"))
        
        all_collections = utility.list_collections()
        embedding_collections = [c for c in all_collections if "embedding_test" in c]
        
        if not embedding_collections:
            return StandardResponse(
                success=True,
                message="No embedding collections found",
                data={"collections": []}
            )
        
        stats = []
        for collection_name in embedding_collections:
            parts = collection_name.split("_")
            if len(parts) >= 3:
                client_id = parts[0]
                model_type = parts[1]  # Will be gemini or openai
                
                # Get collection stats
                collection = Collection(collection_name)
                collection_stats = {
                    "client_id": client_id,
                    "model_type": model_type,
                    "collection_name": collection_name,
                    "entity_count": collection.num_entities
                }
                stats.append(collection_stats)
        
        return StandardResponse(
            success=True,
            message=f"Found {len(stats)} embedding collections",
            data={"collections": stats}
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            message=f"Error retrieving embedding stats: {str(e)}"
        )

async def _get_stats(client_id: str, model_type: str):
    """Get stats for embedding collections"""
    try:
        from pymilvus import connections, utility, Collection
        
        # Connect to Milvus
        connections.connect("default", host=os.environ.get("MILVUS_HOST", "localhost"), port=os.environ.get("MILVUS_PORT", "19530"))
        
        stats = {}
        
        if model_type in ["gemini", "all"]:
            # Get Gemini collection stats
            gemini_collection = f"{client_id}_gemini_embedding_test"
            if utility.has_collection(gemini_collection):
                collection = Collection(gemini_collection)
                stats["gemini"] = {
                    "collection_name": gemini_collection,
                    "entity_count": collection.num_entities,
                    "schema": str(collection.schema)
                }
            else:
                stats["gemini"] = {"exists": False}
        
        if model_type in ["openai", "all"]:
            # Get OpenAI collection stats
            openai_collection = f"{client_id}_openai_embedding_test"
            if utility.has_collection(openai_collection):
                collection = Collection(openai_collection)
                stats["openai"] = {
                    "collection_name": openai_collection,
                    "entity_count": collection.num_entities,
                    "schema": str(collection.schema)
                }
            else:
                stats["openai"] = {"exists": False}
        
        return StandardResponse(
            success=True,
            message=f"Retrieved stats for client {request.client_id}",
            data=stats
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Error getting stats: {str(e)}"
        )

# Allow direct testing of the API if run as a script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("embedding_api:router", host="0.0.0.0", port=8000, reload=True)
