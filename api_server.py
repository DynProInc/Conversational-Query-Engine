"""
FastAPI Server for LLM SQL Query Engine

This API takes natural language questions, converts them to SQL using OpenAI or Claude,
executes them against Snowflake, and returns the results.
"""
import os
import sys
import time
import json
import pandas as pd
import datetime
from typing import Dict, List, Optional, Union, Any

import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our existing functionality
from nlq_to_snowflake import nlq_to_snowflake

# Import Claude functionality
from claude_query_generator import natural_language_to_sql_claude
from nlq_to_snowflake_claude import nlq_to_snowflake_claude

# Import multi-client support
from config.client_integration import with_client_context
from config.client_manager import client_manager

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LLM SQL Query Engine API",
    description="Convert natural language to SQL and execute in Snowflake using OpenAI or Claude",
    version="1.1.0"
)

# Add CORS middleware to allow requests from API clients
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import and include the prompt query history router
from prompt_query_history_route import router as prompt_query_history_router
app.include_router(prompt_query_history_router)



# Import and include the execute query router
from execute_query_route import router as execute_query_router
app.include_router(execute_query_router)

# Import and include the admin router
from admin_routes import router as admin_router
app.include_router(admin_router)

# Define request and response models
class QueryRequest(BaseModel):
    client_id: Optional[str] = "mts"  # Default to 'mts' client for backward compatibility
    prompt: str
    limit_rows: int = 100
    data_dictionary_path: Optional[str] = None
    execute_query: bool = True
    model: Optional[str] = None  # For specifying a specific model
    include_charts: bool = False  # Whether to include chart recommendations
    edited_query: Optional[str] = None  # For user-edited SQL queries

class QueryResponse(BaseModel):
    prompt: str
    query: str
    query_output: List[Dict[str, Any]]
    model: str  # Indicate which model generated the SQL
    token_usage: Optional[Dict[str, int]] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    user_hint: Optional[str] = None  # NEW: user-friendly hint for errors
    chart_recommendations: Optional[List[Dict[str, Any]]] = None  # NEW: chart recommendations
    chart_error: Optional[str] = None  # NEW: error message when charts cannot be generated
    edited: bool = False  # Flag indicating if the SQL was edited by the user


# Route for the chart viewer UI
@app.get("/charts", response_class=HTMLResponse)
async def chart_viewer():
    return FileResponse("static/chart_viewer.html")

# Route for the admin dashboard UI
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    return FileResponse("static/admin_dashboard.html")
    





@app.post("/query", response_model=QueryResponse)
@with_client_context  # Apply client context switching
async def generate_sql_query(request: QueryRequest, data_dictionary_path: Optional[str] = None):
    """Generate SQL from natural language using Claude (default endpoint)"""
    # Redirect to Claude endpoint
    return await generate_sql_query_claude(request, data_dictionary_path)

@app.post("/query/claude", response_model=QueryResponse)
@with_client_context  # Apply client context switching
async def generate_sql_query_claude(request: QueryRequest, data_dictionary_path: Optional[str] = None):
    """Generate SQL from natural language using Claude and optionally execute against Snowflake"""
    try:
        # Use our modular Claude implementation
        claude_model = request.model if request.model else os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        
        # Use the client-specific dictionary path from the context if available
        dict_path = data_dictionary_path if data_dictionary_path else request.data_dictionary_path
        print(f"Claude endpoint using dictionary path: {dict_path}")
            
        result = nlq_to_snowflake_claude(
            prompt=request.prompt,
            data_dictionary_path=dict_path,
            execute_query=request.execute_query,
            limit_rows=request.limit_rows,
            model=claude_model,
            include_charts=request.include_charts
        )
        
        # Check for errors
        if not result.get("success", False) and "error" in result:
            return QueryResponse(
                prompt=request.prompt,
                query=result.get("sql", ""),
                query_output=[],
                model=claude_model,  # Return actual Claude model name
                success=False,
                error_message=result.get("error") or result.get("error_execution", "Unknown error"),
                execution_time_ms=result.get("execution_time_ms"),
                # Always preserve chart recommendations even if the query execution failed
                chart_recommendations=result.get("chart_recommendations", []),
                chart_error=result.get("chart_error")
            )
        
        # Convert DataFrame results to a list of dictionaries for JSON serialization
        query_output = []
        if "results" in result and isinstance(result["results"], pd.DataFrame):
            # Convert DataFrame to list of dictionaries
            query_output = result["results"].to_dict(orient="records")
        
        # Extract token usage if available
        token_usage = None
        if all(k in result for k in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            token_usage = {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"]
            }
        
        # Return the response
        # Generate a helpful hint for successful queries too
        from error_hint_utils import get_success_hint
        model_name = result.get("model", request.model or "claude")
        success_hint = get_success_hint(result.get("sql", ""), model=model_name)
        
        # Check if Claude reported an error during SQL execution
        has_execution_error = result.get("error_execution") or result.get("error_message")
        success = False if has_execution_error else result.get("success", True)
        
        # Get raw error message
        raw_error_message = result.get("error_message") or result.get("error_execution")
        
        # Clean up the error message format for consistency across providers
        if raw_error_message:
            import re
            
            # Function to simplify error messages to a standard format
            def simplify_error(error_text):
                # First, detect the type of error
                
                # Case 1: Database does not exist errors
                if "Database" in error_text and "does not exist" in error_text:
                    # Extract the database name
                    db_match = re.search(r"Database '([^']+)'.*does not exist", error_text)
                    if db_match:
                        db_name = db_match.group(1)
                        return f"SQL compilation error: database '{db_name}' does not exist"
                
                # Case 2: Invalid identifier errors
                elif "invalid identifier" in error_text.lower():
                    # Try to extract just the identifier portion
                    id_match = re.search(r"invalid identifier\s+'?([^'\n]+)'?", error_text, re.IGNORECASE)
                    if id_match:
                        identifier = id_match.group(1)
                        # Remove table aliases (e.g., T2.COLUMN_NAME → COLUMN_NAME)
                        if '.' in identifier:
                            identifier = identifier.split('.')[-1]
                        return f"SQL compilation error: invalid identifier '{identifier}'"
                
                # Case 3: Other SQL compilation errors
                elif "SQL compilation error:" in error_text:
                    # Get just the part after "SQL compilation error:"
                    comp_match = re.search(r"SQL compilation error:(.*?)(?:\n|$)", error_text)
                    if comp_match:
                        return f"SQL compilation error:{comp_match.group(1).strip()}"
                
                # Default: Just clean up formatting without changing content
                # Remove error codes and line positioning info
                clean_error = re.sub(r'^\d+\s+\(\w+\):\s+', '', error_text)
                clean_error = re.sub(r'error line \d+ at position \d+\s+', '', clean_error)
                # Replace newlines with spaces and clean up extra whitespace
                clean_error = re.sub(r'\s+', ' ', clean_error.replace('\n', ' ')).strip()
                return clean_error
            
            # Apply the simplification
            error_message = simplify_error(raw_error_message)
        else:
            error_message = raw_error_message
        
        # If there's an error, update the user hint to reflect it
        if not success and error_message:
            from error_hint_utils import get_user_friendly_hint, identify_error_type
            error_type = identify_error_type(error_message)
            user_hint = get_user_friendly_hint(error_message, model=claude_model)
        else:
            user_hint = success_hint
            
        return QueryResponse(
            prompt=request.prompt,
            query=result.get("sql", ""),
            query_output=query_output,
            model=claude_model,  # Return actual Claude model name
            token_usage=token_usage,
            success=success,  # Use the actual success status from Claude
            error_message=error_message,  # Include error message if present
            execution_time_ms=result.get("execution_time_ms"),
            user_hint=user_hint,  # Use error-specific hint if there was an error
            chart_recommendations=result.get("chart_recommendations", []),
            chart_error=result.get("chart_error")
        )
        
    except Exception as e:
        # Handle any unexpected errors
        print(f"DEBUG: Unexpected error in Claude endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Generate user hint with the appropriate model
        from error_hint_utils import get_user_friendly_hint, clean_error_message, identify_error_type
        error_str = str(e)
        error_type = identify_error_type(error_str)
        clean_error = clean_error_message(error_str, error_type)
        model_name = request.model or "claude"
        user_hint = get_user_friendly_hint(error_str, model=model_name)
        
        return QueryResponse(
            prompt=request.prompt,
            query="",
            query_output=[],
            model="claude",
            token_usage=None,
            success=False,
            error_message=clean_error,
            user_hint=user_hint,
            chart_recommendations=None,
            chart_error="Cannot generate charts: error in SQL generation"
        )


async def health_check():
    """
    Health check endpoint: checks OpenAI, Claude, Gemini APIs and Snowflake connectivity.
    Returns generic error messages rather than specific error details.
    """
    from health_check_utils import check_claude_health, check_snowflake_health
    
    # Check Claude API and database
    claude_ok, claude_msg = check_claude_health()
    snowflake_ok, snowflake_msg = check_snowflake_health()

    # Determine overall system status
    all_ok = claude_ok and snowflake_ok
    status = "healthy" if all_ok else "degraded"
    
    # Include status for all components
    details = {
        "claude": {"ok": claude_ok, "msg": claude_msg},
        "snowflake": {"ok": snowflake_ok, "msg": snowflake_msg},
    }
    
    # Return complete health status
    return {
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
        "models": ["claude"],
        "details": details
    }




@app.get("/health/simple")
async def simple_health_check():
    """
    Simple health check endpoint that just confirms the server is running.
    Useful for basic system monitoring without requiring API keys.
    """
    return {
        "status": "healthy",
        "message": "Server is running",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.1.0"
    }


@app.get("/health/client")
async def all_clients_health_check():
    """
    Check health status for all configured clients
    
    Returns:
        Dictionary of client health statuses
    """
    from config.client_manager import client_manager
    
    try:
        # Get all active clients
        clients = client_manager.list_active_clients()
        client_ids = [client["id"] for client in clients]
        
        # Check health for each client
        results = {}
        for client_id in client_ids:
            try:
                # Call the client-specific health check function
                result = await client_health_check(client_id)
                results[client_id] = result
            except Exception as e:
                # If there's an error checking a specific client, record it but continue
                results[client_id] = {
                    "client_id": client_id,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        # Determine overall system status
        all_healthy = all(results[client_id]["status"] == "healthy" for client_id in results)
        some_healthy = any(results[client_id]["status"] == "healthy" for client_id in results)
        
        if all_healthy:
            overall_status = "healthy"
        elif some_healthy:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
            
        # Return consolidated results
        return {
            "status": overall_status,
            "timestamp": datetime.datetime.now().isoformat(),
            "client_count": len(results),
            "clients": results
        }
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Failed to check health for all clients: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/health/client/{client_id}")
async def client_health_check(client_id: str):
    """
    Client-specific health check: verifies that a specific client's API keys and
    configuration are working properly.
    
    Args:
        client_id: The client identifier to check health for
        
    Returns:
        Health status with client-specific configuration details
    """
    from config.client_manager import client_manager
    
    try:
        # Verify client exists
        if not client_manager.get_client_info(client_id):
            raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found")
        
        # Get client-specific API keys for Claude
        try:
            claude_config = client_manager.get_llm_config(client_id, 'anthropic')
            claude_ok, claude_msg = check_claude_health(api_key=claude_config['api_key'], model=claude_config.get('model'))
        except Exception as e:
            claude_ok, claude_msg = False, f"Claude API error: {str(e)}"
            
        # Check Snowflake with client-specific credentials
        try:
            snowflake_params = client_manager.get_snowflake_connection_params(client_id)
            snowflake_ok, snowflake_msg = check_snowflake_health(connection_params=snowflake_params)
        except Exception as e:
            snowflake_ok, snowflake_msg = False, f"Snowflake error: {str(e)}"
        
        # Determine overall client status
        all_ok = claude_ok and snowflake_ok
        status = "healthy" if all_ok else "degraded"
        
        # Include status for all components
        details = {
            "claude": {"ok": claude_ok, "msg": claude_msg},
            "snowflake": {"ok": snowflake_ok, "msg": snowflake_msg},
        }
        
        # Return client-specific health status
        return {
            "client_id": client_id,
            "status": status,
            "models": ["claude"],
            "details": details,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except HTTPException as http_e:
        # Re-raise HTTP exceptions
        raise http_e
    except Exception as e:
        # Handle any unexpected errors
        error_msg = f"Failed to check client health: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/models")
async def available_models():
    """
    List available language models for SQL generation
    """
    try:
        # Get all models and their providers
        models = {
            "claude": [os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        }
        
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/clients")
async def list_clients():
    """
    List all active clients available in the system.
    Returns basic info about each client.
    """
    try:
        clients = client_manager.list_active_clients()
        return {
            "clients": clients,
            "success": True,
            "count": len(clients)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error listing clients: {str(e)}", "success": False}
        )


@app.get("/clients/{client_id}")
async def get_client_info(client_id: str):
    """
    Get detailed information about a specific client.
    """
    try:
        info = client_manager.get_client_info(client_id)
        # Exclude sensitive information
        if "data_dictionary_path" in info:
            info["data_dictionary_exists"] = os.path.exists(info["data_dictionary_path"])
        
        return {
            "client": info,
            "success": True
        }
    except Exception as e:
        return JSONResponse(
            status_code=404, 
            content={"error": f"Client not found: {str(e)}", "success": False}
        )


@app.post("/query/compare", response_model=QueryResponse)
@with_client_context  # Apply client context switching
async def compare_models(request: QueryRequest, data_dictionary_path: Optional[str] = None):
    """
    Compare models endpoint - redirects to unified endpoint with compare model
    """
    # Set the model to "compare" to trigger comparison logic
    request.model = "compare"
    
    # Call the unified endpoint
    return await unified_query_endpoint(request, data_dictionary_path)


@app.post("/query/unified", response_model=QueryResponse)
@with_client_context  # Apply client context switching
async def unified_query_endpoint(
    request: QueryRequest,
    data_dictionary_path: Optional[str] = None
):
    """
    Unified query endpoint that routes to the appropriate model based on user selection.

    Parameters:
        request: The query request with model parameter that specifies which model to use.
        Model can be a specific model name or a simple provider name like "openai", "claude", "gemini".
        
        If request.edited_query is provided, the edited SQL will be executed directly
        (when execute_query=True) or simply returned (when execute_query=False).
        
        If request.execute_query is False, only the SQL will be generated without execution.
        
    Returns:
        QueryResponse: The query response from the selected model
    """
    try:
        # Print client information for debugging
        print(f"\nUnified API: Request for client_id = {request.client_id}")
        
        # Check if this is an edited query that needs execution
        if request.edited_query:
            print(f"Unified API: Processing edited query from client {request.client_id}")
            
            # If execute_query is False, just return the edited SQL without execution
            if not request.execute_query:
                print("Unified API: Returning edited SQL without execution")
                return QueryResponse(
                    prompt=request.prompt,
                    query=request.edited_query,
                    query_output=[],
                    model=request.model or "edited",
                    success=True,
                    edited=True,
                    chart_recommendations=[]
                )
            
            # Otherwise, execute the edited query using our execute-query endpoint
            from execute_query_route import ExecuteQueryRequest, execute_query
            
            # Prepare the execution request
            execute_request = ExecuteQueryRequest(
                client_id=request.client_id,
                query=request.edited_query,
                limit_rows=request.limit_rows,
                original_prompt=request.prompt,
                include_charts=request.include_charts,
                model=request.model,  # Pass the model used for the original query
                original_sql=request.query if hasattr(request, 'query') else None,  # Pass the original SQL
                original_chart_recommendations=request.chart_recommendations if hasattr(request, 'chart_recommendations') else None  # Pass original chart recommendations
            )
            
            # Execute the query
            execution_result = await execute_query(execute_request, data_dictionary_path)
            
            # Convert execution result to QueryResponse format
            return QueryResponse(
                prompt=request.prompt,
                query=execution_result.query,
                query_output=execution_result.query_output,
                model=request.model or "edited",
                success=execution_result.success,
                error_message=execution_result.error_message,
                execution_time_ms=execution_result.execution_time_ms,
                user_hint=None,
                chart_recommendations=execution_result.chart_recommendations,
                chart_error=execution_result.chart_error,
                edited=True
            )
            
        # Extract the model from the request
        model = request.model.lower() if request.model else ""
        client_id = request.client_id if hasattr(request, 'client_id') else None
        
        print(f"Unified API: Routing request with model = {model}, client_id = {client_id}")
        
        # Support Claude and compare
        if model == "claude" or model == "anthropic":
            # For exact Claude model aliases only - check key first
            print("Unified API: Routing to Claude endpoint")
            
            # When client is specified, verify the API key strictly before proceeding
            if client_id:
                from config.client_manager import client_manager
                try:
                    # This will use client-specific .env file to look for CLIENT_{CLIENT_ID}_ANTHROPIC_API_KEY
                    claude_config = client_manager.get_llm_config(client_id, 'anthropic')
                    
                    # Double check key is not empty
                    if not claude_config or not claude_config.get('api_key') or not claude_config.get('api_key').strip():
                        error_msg = f"Client '{client_id}' anthropic API key not configured"
                        print(f"❌ Error: {error_msg}")
                        raise HTTPException(status_code=400, detail=error_msg)
                except ValueError as e:
                    error_msg = f"Client '{client_id}' anthropic API key not configured"
                    print(f"Error: {error_msg}")
                    raise HTTPException(status_code=400, detail=error_msg)
            
            # Get the model from client configuration
            if client_id:
                from config.client_manager import client_manager
                claude_config = client_manager.get_llm_config(client_id, 'anthropic')
                request.model = claude_config.get('model', "claude-3-5-sonnet-20241022")
            else:
                request.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            print(f"Using Claude model: {request.model}")
                
            # Call Claude endpoint with data_dictionary_path from client context
            response = await generate_sql_query_claude(request, data_dictionary_path=data_dictionary_path)
            
            # If execute_query is False, ensure we're not executing the query
            if not request.execute_query and response.query_output:
                response.query_output = []
                
            return response
        
        elif model == "compare":
            # Handle compare model - return a simple response indicating comparison is not supported
            print("Unified API: Compare model requested, but only Claude is supported")
            return QueryResponse(
                prompt=request.prompt,
                query="",
                query_output=[],
                model="compare",
                success=False,
                error_message="Model comparison is not supported. Only Claude is available. Please use 'claude' or leave empty for default.",
                user_hint="Try using 'claude' as the model instead of 'compare'."
            )
        elif model == "":
            # Handle empty model parameter - default to Claude but verify key exists first
            print("Unified API: No model specified, defaulting to Claude")
            client_id = request.client_id if hasattr(request, 'client_id') else None
            
            # When client is specified, verify the API key strictly before proceeding
            if client_id:
                # Use client_manager which loads the correct client-specific .env file
                from config.client_manager import client_manager
                
                try:
                    # This will raise an error if API key doesn't exist or is empty
                    # Looking specifically for the CLIENT_{CLIENT_ID}_ANTHROPIC_API_KEY in the client's .env file
                    claude_config = client_manager.get_llm_config(client_id, 'anthropic')
                    
                    # Double check that API key is not empty (extra validation)
                    if not claude_config or not claude_config.get('api_key') or not claude_config.get('api_key').strip():
                        error_msg = f"Client '{client_id}' anthropic API key not configured"
                        print(f"❌ Error: {error_msg}")
                        raise HTTPException(status_code=400, detail=error_msg)
                        
                except ValueError as e:
                    # Key doesn't exist or has another issue
                    error_msg = f"Client '{client_id}' anthropic API key not configured"
                    print(f"Error: {error_msg}")
                    raise HTTPException(status_code=400, detail=error_msg)
            
            # Only proceed if key validation passes (client has valid API key or no client specified)
            if client_id:
                from config.client_manager import client_manager
                claude_config = client_manager.get_llm_config(client_id, 'anthropic')
                request.model = claude_config.get('model', "claude-3-5-sonnet-20241022")
            else:
                request.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            print(f"Using Claude model: {request.model}")
            response = await generate_sql_query_claude(request, data_dictionary_path=data_dictionary_path)
            
            # If execute_query is False, ensure we're not executing the query
            if not request.execute_query and response.query_output:
                response.query_output = []
                
            return response
            
        else:
            # Invalid model name - return error with clear, concise guidance
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Only Claude is supported. Please use 'claude' or leave empty for default."
            )
                
    except Exception as e:
        print(f"❌ Error in unified query endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "error_message": f"Server error: {str(e)}",
                "prompt": request.prompt if hasattr(request, 'prompt') else "Unknown"
            }
        )



# Main entry point to run the server directly
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM SQL Query Engine API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API server on (default: 8000)")
    args = parser.parse_args()

    import uvicorn
    print(f"Starting API server on port {args.port}")
    uvicorn.run("api_server:app", host="0.0.0.0", port=args.port, reload=True)


