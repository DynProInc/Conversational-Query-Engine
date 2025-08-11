"""
Saved Queries Manager - Allows users to save specific queries with tags for later use
"""
import os
import csv
import json
import uuid
import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import httpx

# Constants
SAVED_QUERIES_FILE = os.path.join(os.path.dirname(__file__), "saved_queries.csv")

# Define the router without a prefix
router = APIRouter(
    tags=["saved_queries"],
    responses={404: {"description": "Not found"}},
)

# Define the columns for the saved queries CSV file
SAVED_QUERIES_COLUMNS = [
    "id", "execution_id", "timestamp", "user_id", "client_id", "model", "prompt", "sql_query", 
    "prompt_tokens", "completion_tokens", "total_tokens", 
    "input_cost", "output_cost", "total_cost", "query_executed",
    "tags", "folder", "notes"
]

class SaveQueryRequest(BaseModel):
    """Model for saving a query"""
    query_id: Optional[str] = None  # Optional ID from token_usage.csv
    user_id: str
    client_id: str = "mts"  # Client ID used for execution (defaults to "mts")
    model: str
    prompt: Optional[str] = None
    query_text: Optional[str] = None  # For backward compatibility
    sql_query: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: Optional[float] = 0.0
    output_cost: Optional[float] = 0.0
    total_cost: Optional[float] = 0.0
    query_executed: Optional[bool] = True
    tags: Optional[List[str]] = []
    folder: Optional[str] = None
    notes: Optional[str] = None
    execution_id: Optional[str] = None  # Link to token_usage.csv execution
    
    @validator('prompt', 'query_text')
    def validate_prompt_or_query_text(cls, v, values):
        # Ensure at least one of prompt or query_text is provided
        if v is None and 'prompt' in values and 'query_text' in values and values.get('prompt') is None and values.get('query_text') is None:
            raise ValueError('Either prompt or query_text must be provided')
        return v
    
    @validator('prompt', pre=True)
    def set_prompt_from_query_text(cls, v, values):
        # If prompt is not provided but query_text is, use query_text as prompt
        if v is None and 'query_text' in values and values['query_text'] is not None:
            return values['query_text']
        return v or ''
    
    @validator('query_text', pre=True)
    def set_query_text_from_prompt(cls, v, values):
        # If query_text is not provided but prompt is, use prompt as query_text
        if v is None and 'prompt' in values and values['prompt'] is not None:
            return values['prompt']
        return v or ''
    
    @validator('tags', pre=True)
    def ensure_tags(cls, v):
        # Ensure tags is always a list
        if v is None:
            return []
        return v

class SavedQuery(BaseModel):
    """Model for a saved query"""
    id: str
    timestamp: str
    user_id: str
    model: Optional[str] = None
    prompt: str  # Same as query_text
    sql_query: Optional[str] = None
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    input_cost: Optional[float] = 0.0
    output_cost: Optional[float] = 0.0
    total_cost: Optional[float] = 0.0
    query_executed: Optional[bool] = False
    tags: List[str]
    folder: Optional[str] = None
    notes: Optional[str] = None

def _ensure_saved_queries_file_exists():
    """Create saved queries file with headers if it doesn't exist"""
    if not os.path.exists(SAVED_QUERIES_FILE):
        os.makedirs(os.path.dirname(SAVED_QUERIES_FILE), exist_ok=True)
        with open(SAVED_QUERIES_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(SAVED_QUERIES_COLUMNS)

def _read_saved_queries():
    """Read all saved queries from the CSV file"""
    _ensure_saved_queries_file_exists()
    
    queries = []
    try:
        with open(SAVED_QUERIES_FILE, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert string representations to appropriate types
                try:
                    # Convert numeric fields
                    for field in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                        if field in row and row[field]:
                            row[field] = int(row[field])
                        else:
                            row[field] = 0
                    
                    # Convert cost fields
                    for field in ['input_cost', 'output_cost', 'total_cost']:
                        if field in row and row[field]:
                            row[field] = float(row[field])
                        else:
                            row[field] = 0.0
                    
                    # Convert boolean fields
                    if 'query_executed' in row:
                        row['query_executed'] = row['query_executed'] == '1'
                    
                    # Convert tags from comma-separated string to list
                    if 'tags' in row and row['tags']:
                        # Handle complex nested tag formats by first trying to normalize
                        tags_str = row['tags']
                        # Remove any nested list representations and quotes
                        tags_str = tags_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                        # Split by comma and clean
                        row['tags'] = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                    else:
                        row['tags'] = []
                except Exception as e:
                    print(f"Error converting row values: {str(e)}")
                
                queries.append(row)
    except Exception as e:
        print(f"Error reading saved queries: {str(e)}")
        return []
    
    return queries

def _write_saved_queries(queries):
    """Write all saved queries to the CSV file"""
    with open(SAVED_QUERIES_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=SAVED_QUERIES_COLUMNS)
        writer.writeheader()
        for query in queries:
            writer.writerow(query)

@router.post("/saved_queries")
@router.post("/saved_queries/save")
async def save_query(request: SaveQueryRequest) -> JSONResponse:
    """
    Save a query to the saved_queries.csv file
    """
    _ensure_saved_queries_file_exists()
    
    # Generate a unique ID with a cleaner format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a simpler query ID format
    if not request.execution_id:
        # Generate a UUID if no execution_id provided
        query_id = f"sq_{str(uuid.uuid4())[:8]}"
    else:
        # Use the execution_id as a reference
        query_id = f"sq_{request.execution_id[:8]}"
    
    # Use either prompt or query_text, with prompt taking precedence
    query_text = request.prompt if request.prompt else request.query_text or ""
    
    # Prepare the query data
    query_data = {
        'id': query_id,
        'execution_id': request.execution_id or "",
        'timestamp': timestamp,
        'user_id': request.user_id,
        'client_id': request.client_id,  # Store the client_id used for execution
        'model': request.model or '',
        'prompt': query_text,
        'sql_query': request.sql_query or '',
        'prompt_tokens': request.prompt_tokens or 0,
        'completion_tokens': request.completion_tokens or 0,
        'total_tokens': request.total_tokens or 0,
        'input_cost': request.input_cost or 0.0,
        'output_cost': request.output_cost or 0.0,
        'total_cost': request.total_cost or 0.0,
        'query_executed': '1' if request.query_executed else '0',
        # Store tags as a simple comma-separated string without list brackets
        'tags': ','.join([tag.strip() for tag in request.tags or []]),
        'folder': request.folder or '',
        'notes': request.notes or ''
    }
    
    try:
        # Read existing queries
        queries = _read_saved_queries()
        
        # Add new query
        queries.append(query_data)
        
        # Write all queries back to CSV
        _write_saved_queries(queries)
        
        return JSONResponse(content={
            "id": query_id, 
            "execution_id": request.execution_id,
            "message": "Query saved successfully",
            "sql_query": request.sql_query,
            "prompt_tokens": request.prompt_tokens,
            "completion_tokens": request.completion_tokens,
            "total_tokens": request.total_tokens,
            "input_cost": request.input_cost,
            "output_cost": request.output_cost,
            "total_cost": request.total_cost,
            "query_executed": request.query_executed,
            "tags": request.tags,
            "folder": request.folder,
            "notes": request.notes
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving query: {str(e)}")

@router.get("/saved_queries")
async def get_saved_queries(
    user_id: Optional[str] = None,
    tag: Optional[str] = None,
    folder: Optional[str] = None,
    client_id: Optional[str] = None
) -> JSONResponse:
    """
    Get saved queries with optional filtering by user_id, tag, or folder
    """
    queries = _read_saved_queries()
    
    # Apply filters
    if user_id:
        queries = [q for q in queries if q.get('user_id') == user_id]
    
    if client_id:
        queries = [q for q in queries if q.get('client_id') == client_id]
    
    if tag:
        queries = [q for q in queries if tag in q.get('tags', [])]
    
    if folder:
        queries = [q for q in queries if q.get('folder') == folder]
    
    return JSONResponse(content={'queries': queries})

@router.get("/saved_queries/tags")
async def get_all_tags(user_id: Optional[str] = None) -> JSONResponse:
    """
    Get all unique tags used in saved queries
    
    Args:
        user_id: Optional user ID to filter tags by
        
    Returns:
        List of unique tags
    """
    queries = _read_saved_queries()
    
    # Filter by user_id if provided
    if user_id:
        queries = [q for q in queries if q.get('user_id') == user_id]
    
    # Extract all tags
    all_tags = set()
    for query in queries:
        tags = query.get('tags', [])
        if tags:
            all_tags.update(tags)
    
    return JSONResponse(content=sorted(list(all_tags)))

@router.get("/saved_queries/folders")
async def get_all_folders(user_id: Optional[str] = None) -> JSONResponse:
    """
    Get all unique folders used in saved queries
    
    Args:
        user_id: Optional user ID to filter folders by
        
    Returns:
        List of unique folders
    """
    queries = _read_saved_queries()
    
    # Filter by user_id if provided
    if user_id:
        queries = [q for q in queries if q.get('user_id') == user_id]
    
    # Extract all folders
    all_folders = set()
    for query in queries:
        folder = query.get('folder')
        if folder:
            all_folders.add(folder)
    
    return JSONResponse(content=sorted(list(all_folders)))

@router.get("/saved_queries/{query_id}")
async def get_saved_query(query_id: str) -> JSONResponse:
    """
    Get a specific saved query by ID
    """
    queries = _read_saved_queries()
    query = next((q for q in queries if q.get('id') == query_id), None)
    
    if not query:
        raise HTTPException(status_code=404, detail=f"Query with ID {query_id} not found")
    
    return JSONResponse(content=query)

@router.delete("/saved_queries/{query_id}")
async def delete_saved_query(query_id: str) -> JSONResponse:
    """
    Delete a saved query by ID
    """
    queries = _read_saved_queries()
    
    # Find the query to delete
    query_to_delete = next((q for q in queries if q.get('id') == query_id), None)
    
    if not query_to_delete:
        raise HTTPException(status_code=404, detail=f"Query with ID {query_id} not found")
    
    # Remove the query from the list
    queries = [q for q in queries if q.get('id') != query_id]
    
    # Write the updated list back to the CSV
    with open(SAVED_QUERIES_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=SAVED_QUERIES_COLUMNS)
        writer.writeheader()
        for query in queries:
            writer.writerow(query)
    
    return JSONResponse(content={"message": f"Query with ID {query_id} deleted successfully"})

# Removed duplicate endpoints

@router.get("/saved_queries/execution_data/{query_id}")
async def get_query_for_execution(query_id: str, client_id: Optional[str] = None) -> JSONResponse:
    """
    Get the saved query data in a format ready for direct execution via the unified query endpoint
    
    Args:
        query_id: ID of the saved query to get execution data for
        client_id: Optional client ID to override the one in the saved query
        
    Returns:
        JSON data ready to be sent to the unified query endpoint
    """
    # Find the saved query
    queries = _read_saved_queries()
    query = next((q for q in queries if q.get('id') == query_id), None)
    
    if not query:
        raise HTTPException(status_code=404, detail=f"Query with ID {query_id} not found")
    
    # Use the client_id from the saved query if one is not explicitly provided
    # This ensures we use the same client context as when the query was originally executed
    execution_client_id = client_id or query.get('client_id', "mts")
    
    # Prepare the request data for the unified query endpoint
    request_data = {
        "client_id": execution_client_id,
        "prompt": query.get('prompt', query.get('query_text', "")),
        "model": query.get('model', 'openai'),
        "limit_rows": 100,
        "execute_query": True,
        "include_charts": True
    }
    
    # If there's a SQL query saved, use it as an edited query
    if query.get('sql_query'):
        request_data["edited_query"] = query.get('sql_query')
    
    # Return the request data that can be sent directly to the unified query endpoint
    return JSONResponse(content={
        "execution_data": request_data,
        "unified_endpoint": "/query/unified",
        "message": "Send this execution_data as JSON to the unified_endpoint to execute this saved query"
    })

# For standalone testing
if __name__ == "__main__":
    import uvicorn
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8003)
