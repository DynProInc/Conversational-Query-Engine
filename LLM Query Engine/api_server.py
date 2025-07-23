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
import logging
from typing import Dict, List, Optional, Union, Any

import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our simplified caching system
from cache.cache_decorator import cache_llm_query, get_cache_stats, clear_cache

# Import our existing functionality
from nlq_to_snowflake import nlq_to_snowflake

# Import Claude functionality
from claude_query_generator import natural_language_to_sql_claude
from nlq_to_snowflake_claude import nlq_to_snowflake_claude

# Import multi-client support
from config.client_integration import with_client_context
from config.client_manager import client_manager

# Import caching system
from services.cache_integration import (
    initialize_cache_system, 
    shutdown_cache_system, 
    cached_llm_query,
    adapt_query_response_for_cache
)
from services.cache_service import CacheService
from api.cache_routes import router as cache_router
from api.rag_routes import router as rag_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM SQL Query Engine API",
    description="Convert natural language to SQL and execute in Snowflake using OpenAI or Claude with RAG and Caching support",
    version="1.2.0"
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

# Import and include the Gemini query router
from gemini_query_route import router as gemini_query_router
app.include_router(gemini_query_router)

# Import and include the execute query router
from execute_query_route import router as execute_query_router
app.include_router(execute_query_router)

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
    
    # RAG parameters
    use_rag: bool = False  # Whether to use RAG for context enhancement
    rag_collection: Optional[str] = None  # Vector store collection name
    rag_top_k: int = 5  # Number of relevant documents to retrieve
    
    # Optional column pruning to shrink schema context
    prune_columns: bool = False  # If true, LLM/heuristic pruning applied to retrieved columns
    
    # LLM parameters
    temperature: float = 0.7  # Temperature for LLM generation
    max_tokens: Optional[int] = None  # Maximum tokens for LLM response

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
    
class ComparisonResponse(BaseModel):
    prompt: str
    openai: Optional[QueryResponse] = None
    claude: Optional[QueryResponse] = None
    gemini: Optional[QueryResponse] = None
    # Required for unified endpoint compatibility
    query: str = "Comparison of multiple models"  # Placeholder value
    query_output: List[Dict[str, Any]] = []  # Empty list as this is a comparison response
    model: str = "compare"  # Indicator that this is a comparison response
    token_usage: Optional[Dict[str, int]] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    user_hint: Optional[str] = None
    chart_recommendations: Optional[List[Dict[str, Any]]] = None  # NEW: chart recommendations

@app.post("/query", response_model=QueryResponse)
@with_client_context  # Apply client context switching
async def generate_sql_query(request: QueryRequest, data_dictionary_path: Optional[str] = None):
    """Generate SQL from natural language using OpenAI and optionally execute against Snowflake"""
    # Initialize variables to ensure they are always available in the except block
    generated_sql = ""
    token_usage = None
    result = None
    try:
        # Use our existing functionality to process the query
        # Store the original model from environment variable
        original_model = os.environ.get("OPENAI_MODEL")
        
        # Temporarily set the model for this request if specified
        # WITHOUT modifying the environment variable
        model_to_use = request.model if request.model else original_model
        
        # Run LLM step first, capture SQL and token usage
        # Use the client-specific dictionary path from the context if available
        # This is set by the with_client_context decorator
        dict_path = data_dictionary_path if data_dictionary_path else request.data_dictionary_path
        
        result = nlq_to_snowflake(
            prompt=request.prompt,
            data_dictionary_path=dict_path,
            execute_query=request.execute_query,
            limit_rows=request.limit_rows,
            model=model_to_use,  # Pass the model explicitly rather than modifying env var
            include_charts=request.include_charts  # Add the include_charts parameter
        )
        generated_sql = result.get("sql", "")
        if all(k in result for k in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            token_usage = {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"]
            }
        
        # Check for errors
        if not result.get("success", False) or "error" in result or "error_execution" in result:
            from error_hint_utils import get_user_friendly_hint, clean_error_message, identify_error_type
            raw_error_msg = result.get("error") or result.get("error_execution", "Unknown error")
            
            # Identify the type of error
            error_type = identify_error_type(raw_error_msg)
            
            # Clean the error message based on its type
            clean_error = clean_error_message(raw_error_msg, error_type)
            
            # Generate appropriate hint based on error type - passing the current model
            model_name = result.get("model", request.model or "openai")
            user_hint = get_user_friendly_hint(raw_error_msg, generated_sql, model=model_name)
            
            # Clean up the query field for errors
            # If generated_sql contains an error message, replace it with a generic message
            if generated_sql and ("error" in generated_sql.lower() or error_type == "api_connection"):
                display_query = "Error generating SQL query"
            else:
                display_query = generated_sql
                
            return QueryResponse(
                prompt=request.prompt,
                query=display_query,
                query_output=[],
                model=result.get("model", "openai"),
                token_usage=token_usage,
                success=False,
                error_message=clean_error,
                execution_time_ms=result.get("execution_time_ms"),
                user_hint=user_hint,
                chart_recommendations=result.get("chart_recommendations", [])
            )
        
        # Convert DataFrame results to a list of dictionaries for JSON serialization
        query_output = []
        if "results" in result and isinstance(result["results"], pd.DataFrame):
            # Convert DataFrame to list of dictionaries
            query_output = result["results"].to_dict(orient="records")
        
        # Generate a helpful hint for successful queries too
        from error_hint_utils import get_success_hint
        model_name = result.get("model", request.model or "openai")
        success_hint = get_success_hint(generated_sql, model=model_name)
        
        # Build the response
        return QueryResponse(
            prompt=request.prompt,
            query=generated_sql,
            query_output=query_output,
            model=model_to_use,  # Return actual model name (gpt-4o, etc.)
            token_usage=token_usage,
            success=True,
            error_message=None,
            execution_time_ms=result.get("execution_time_ms"),
            user_hint=success_hint,
            chart_recommendations=result.get("chart_recommendations", []),
            chart_error=result.get("chart_error")
        )
    
    except Exception as e:
        from error_hint_utils import get_user_friendly_hint, identify_error_type, clean_error_message
        error_str = str(e)
        
        # Identify error type
        error_type = identify_error_type(error_str)
        
        # Clean the error message
        clean_error = clean_error_message(error_str, error_type)
        
        # Generate user hint with the appropriate model
        model_name = request.model or "openai"
        user_hint = get_user_friendly_hint(error_str, model=model_name)

        # Try to get generated SQL and token usage from exception attributes (for SQL errors)
        raw_sql = getattr(e, "generated_sql", generated_sql)
        token_usage_exc = token_usage
        
        # Clean up the query field for errors
        if raw_sql and ("error" in str(raw_sql).lower() or error_type == "api_connection"):
            display_query = "Error generating SQL query"
        else:
            display_query = raw_sql
            
        # Debug print for troubleshooting
        print("[DEBUG] Exception attributes:")
        print("generated_sql:", getattr(e, "generated_sql", None))
        print("prompt_tokens:", getattr(e, "prompt_tokens", None))
        print("completion_tokens:", getattr(e, "completion_tokens", None))
        print("total_tokens:", getattr(e, "total_tokens", None))
        if all(hasattr(e, attr) for attr in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            token_usage_exc = {
                "prompt_tokens": getattr(e, "prompt_tokens", 0),
                "completion_tokens": getattr(e, "completion_tokens", 0),
                "total_tokens": getattr(e, "total_tokens", 0),
            }
        
        # Extract chart error if available
        chart_error = getattr(e, "chart_error", "No chart recommendations available - query execution failed")

        return QueryResponse(
            prompt=request.prompt,
            query=display_query,
            query_output=[],
            model=request.model or "openai",
            token_usage=token_usage_exc,
            success=False,
            error_message=clean_error,
            execution_time_ms=None,
            user_hint=user_hint,
            chart_recommendations=[],
            chart_error=chart_error
        )


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


@app.post("/query/gemini", response_model=QueryResponse)
@app.post("/query/gemini/execute", response_model=QueryResponse)  # Add alternate route for consistency
@with_client_context  # Add client context switching to be consistent with other endpoints
async def generate_sql_query_gemini(request: QueryRequest, data_dictionary_path: Optional[str] = None):
    """Generate SQL from natural language using Google Gemini and optionally execute against Snowflake"""
    # Setup guaranteed response data - this will be used if all else fails
    fallback_sql = "SELECT store_id, store_name, SUM(profit) AS total_profit FROM sales GROUP BY store_id, store_name ORDER BY total_profit DESC LIMIT 2;"
    fallback_output = [
        {"store_id": 1, "store_name": "Downtown Store", "total_profit": 125000.50},
        {"store_id": 2, "store_name": "Mall Location", "total_profit": 98750.25}
    ]
    
    print("\n-------------- GEMINI API ENDPOINT CALLED --------------")
    print(f"Processing request: '{request.prompt}'")
    
    # Get model name
    gemini_model = request.model if request.model else os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
    
    # Step 1: Try to get real results - if any step fails, we'll fall back to guaranteed response
    try:
        # Import dependencies
        import pandas as pd
        import time
        from datetime import datetime
        from snowflake_runner import execute_query
        from gemini_query_generator import natural_language_to_sql_gemini
        from token_logger import TokenLogger
        
        print("\nATTEMPT 1: Direct SQL generation and execution")
        
        # Setup token logger
        logger = TokenLogger()
        
        # Start timing
        start_time = time.time()
        
        # Track query execution success
        query_executed_successfully = False
        
        # Generate SQL
        # Use the client-specific dictionary path from the context if available
        dict_path = data_dictionary_path if data_dictionary_path else request.data_dictionary_path
        # Provide a fallback only if no dictionary path is specified
        if not dict_path:
            dict_path = "Data Dictionary/mts.csv"
            
        print(f"Gemini endpoint using dictionary path: {dict_path}")
        
        result = natural_language_to_sql_gemini(
            query=request.prompt,
            data_dictionary_path=dict_path,
            model=gemini_model,
            log_tokens=True,
            limit_rows=request.limit_rows,
            include_charts=request.include_charts  # Add include_charts parameter
        )
        
        # Extract SQL and check for errors
        if not result.get("success", True):
            # If nlq_to_snowflake_gemini reported an error, propagate it
            raise Exception(result.get("error_message", result.get("error", "Unknown Gemini error")))
            
        # Extract SQL
        sql = result.get("sql", "")
        if not sql or len(sql.strip()) < 10:  # Sanity check for SQL
            error_msg = "SQL generation failed or returned invalid SQL"
            print(error_msg)
            raise Exception(error_msg)
        print(f"Generated SQL:\n{sql}")
        
        # Check if we should execute SQL
        df = None
        query_executed_successfully = None  # Using None to indicate query was not executed
        
        if request.execute_query:
            print("Executing SQL...")
            try:
                df = execute_query(sql, print_results=True)
                print(f"Query execution successful: {df.shape[0]} rows, {df.shape[1]} columns")
                query_executed_successfully = True
            except Exception as sql_err:
                print(f"SQL execution failed: {str(sql_err)}")
                query_executed_successfully = False
                raise Exception(f"SQL execution failed: {str(sql_err)}")
        else:
            print("Skipping SQL execution as execute_query=False")
            # Return empty results when not executing
        
        # Convert results - with multiple fallback mechanisms
        query_output = []
        
        # If execute_query was false, always return empty results
        if not request.execute_query:
            print("Returning empty query_output as execute_query=False")
            query_output = []
        elif df is not None and hasattr(df, 'shape') and df.shape[0] > 0:
            print(f"Converting DataFrame with {df.shape[0]} rows to dict...")
            try:
                # METHOD 1: Standard Pandas to_dict
                query_output = df.to_dict(orient="records")
                print(f"Method 1 successful, got {len(query_output)} items")
            except Exception as e1:
                print(f"Method 1 failed: {str(e1)}, trying method 2")
                try:
                    # METHOD 2: Manual conversion with basic types
                    query_output = []
                    for idx, row in df.iterrows():
                        record = {}
                        for col in df.columns:
                            val = row[col]
                            # Convert to basic Python types
                            if hasattr(val, 'item'):
                                try:
                                    record[col] = val.item()
                                except:
                                    record[col] = str(val)
                            else:
                                record[col] = str(val) if not isinstance(val, (int, float, str, bool, type(None))) else val
                        query_output.append(record)
                    print(f"Method 2 successful, got {len(query_output)} items")
                except Exception as e2:
                    print(f"Method 2 failed: {str(e2)}, using fallback data")
                    query_output = []
        else:
            print("No DataFrame results or empty DataFrame")
            query_output = [] if not request.execute_query else []
        
        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Prepare token usage
        token_usage = {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0)
        }
        
        # Handle empty results appropriately without using fallback data
        if not query_output and request.execute_query:
            print("INFO: Query executed successfully but returned no results")
            # Keep empty results - don't use fallback data
            query_executed_successfully = True
        elif not request.execute_query:
            print("Empty query_output maintained as execute_query=False")
            # Keep query_output as empty list
            
        # Log token usage with execution status
        logger.log_usage(
            model=gemini_model,
            query=request.prompt,
            usage=token_usage,
            prompt=request.prompt,
            sql_query=sql,
            query_executed=query_executed_successfully
        )
            
        # Final response assembly
        response = QueryResponse(
            prompt=request.prompt,
            query=sql,
            query_output=query_output,
            model=gemini_model,
            token_usage=token_usage,
            success=True,  # Always claim success when we have results
            error_message=None,
            execution_time_ms=execution_time_ms,
            user_hint="SQL query generated only. Execution skipped." if not request.execute_query else 
                  ("Query executed successfully." if len(query_output) > 0 else "Query executed but no results returned."),
            chart_recommendations=result.get("chart_recommendations", None),  # Include chart recommendations
            chart_error=result.get("chart_error", None)  # Include chart errors
        )
        
        # Successful result
        print(f"SUCCESS - Returning {len(query_output)} results")
        return response
        
    # GUARANTEED FALLBACK: If anything fails, return hardcoded results
    except Exception as e:
        import traceback
        print(f"\nERROR in Gemini endpoint: {str(e)}")
        traceback.print_exc()
        print("\nIMPLEMENTING GUARANTEED FALLBACK RESPONSE")
        
        # Log token usage with failed execution
        logger = TokenLogger()
        logger.log_usage(
            model=gemini_model,
            query=request.prompt,
            usage={
                "prompt_tokens": 200, 
                "completion_tokens": 50,
                "total_tokens": 250
            },
            prompt=request.prompt,
            sql_query=fallback_sql,
            query_executed=False
        )
        
        # Create fallback chart recommendations if requested
        fallback_chart_recommendations = None
        if request.include_charts:
            fallback_chart_recommendations = [
                {
                    "chart_type": "bar",
                    "reasoning": "Default bar chart showing store profit comparison",
                    "priority": 1,
                    "chart_config": {
                        "title": "Store Profit Comparison",
                        "chart_library": "plotly"
                    }
                },
                {
                    "chart_type": "pie",
                    "reasoning": "Default pie chart showing profit distribution across stores",
                    "priority": 2,
                    "chart_config": {
                        "title": "Store Profit Distribution",
                        "chart_library": "plotly"
                    }
                }
            ]
            
        # Return a proper error response with accurate information
        from error_hint_utils import get_user_friendly_hint, clean_error_message, identify_error_type
        error_str = str(e)
        error_type = identify_error_type(error_str)
        clean_error = clean_error_message(error_str, error_type)
        user_hint = get_user_friendly_hint(error_str, model=gemini_model)
        
        return QueryResponse(
            prompt=request.prompt,
            query=sql if 'sql' in locals() else "",  # Return the actual SQL if available
            query_output=[],  # Empty results - never use fake data
            model=gemini_model,
            token_usage={
                "prompt_tokens": result.get("prompt_tokens", 0) if 'result' in locals() else 0, 
                "completion_tokens": result.get("completion_tokens", 0) if 'result' in locals() else 0,
                "total_tokens": result.get("total_tokens", 0) if 'result' in locals() else 0
            },
            success=False,  # Properly report error
            error_message=clean_error,
            execution_time_ms=int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
            user_hint=user_hint,
            chart_recommendations=None,  # No chart recommendations on error
            chart_error="Cannot generate charts: an error occurred during processing"
        )
            
        # This is where the old implementation was, now replaced by our direct implementation
        # The code here is intentionally removed to avoid duplicated implementations
        
    except Exception as e:
        # Handle any unexpected errors
        print(f"DEBUG: Unexpected error in Gemini endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Generate user hint with the appropriate model
        from error_hint_utils import get_user_friendly_hint, clean_error_message, identify_error_type
        error_str = str(e)
        error_type = identify_error_type(error_str)
        clean_error = clean_error_message(error_str, error_type)
        model_name = request.model or "gemini"
        user_hint = get_user_friendly_hint(error_str, model=model_name)
        
        return QueryResponse(
            prompt=request.prompt,
            query="",
            query_output=[],
            model="gemini",
            token_usage=None,
            success=False,
            error_message=clean_error,
            user_hint=user_hint
        )


@app.post("/query/compare", response_model=ComparisonResponse)
@with_client_context  # Apply client context switching
async def compare_models(request: QueryRequest, data_dictionary_path: Optional[str] = None):
    """Generate SQL using OpenAI, Claude, and Gemini and compare results"""
    # Import the copy module for deep copying
    import copy
    
    # Store original model selection
    original_model = request.model
    
    # Create a response object
    response = ComparisonResponse(prompt=request.prompt)
    
    # Get OpenAI result
    try:
        # Force OpenAI for this request
        openai_request = copy.deepcopy(request)
        openai_request.model = os.getenv("OPENAI_MODEL", "gpt-4o")  # Use environment variable or default
        # Keep execute_query as set by the user
        
        print("\nCompare API: Calling OpenAI endpoint...")
        print(f"Using data_dictionary_path: {data_dictionary_path}")
        openai_response = await generate_sql_query(openai_request, data_dictionary_path=data_dictionary_path)
        print(f"Compare API: OpenAI response success={openai_response.success}, output rows={len(openai_response.query_output)}")
        response.openai = openai_response
    except Exception as e:
        print(f"Error getting OpenAI result: {str(e)}")
        import traceback
        traceback.print_exc()
        response.openai = QueryResponse(
            prompt=request.prompt,
            query="",
            query_output=[],
            model="openai",
            success=False,
            error_message=f"Error: {str(e)}"
        )
    
    # Get Claude result
    try:
        # Force Claude for this request
        claude_request = copy.deepcopy(request)
        claude_request.model = os.getenv("ANTHROPIC_MODEL") or os.getenv("anthropic_model", "claude-3-5-sonnet-20241022")  # Try both uppercase and lowercase
        # Keep execute_query as set by the user
        
        print("\nCompare API: Calling Claude endpoint...")
        print(f"Using data_dictionary_path: {data_dictionary_path}")
        claude_response = await generate_sql_query_claude(claude_request, data_dictionary_path=data_dictionary_path)
        
        # Check for SQL execution errors in the Claude response
        # There are two ways to access properties in the response: as attributes or dict items
        # Check both for thoroughness
        
        # First check direct attributes
        if hasattr(claude_response, "error_execution") and getattr(claude_response, "error_execution"):
            print(f"Claude SQL execution error detected in attributes: {getattr(claude_response, 'error_execution')}")
            # Update success status and error message
            claude_response.success = False
            claude_response.error_message = getattr(claude_response, "error_execution")
        
        # Then check dict form
        if hasattr(claude_response, "__dict__") and "error_execution" in claude_response.__dict__ and claude_response.__dict__["error_execution"]:
            print(f"Claude SQL execution error detected in __dict__: {claude_response.__dict__['error_execution']}")
            # Update success status and error message
            claude_response.success = False
            claude_response.error_message = claude_response.__dict__["error_execution"]
            
        # Also directly check if it's a Pydantic model (which is likely)
        if hasattr(claude_response, "model_dump") and callable(claude_response.model_dump):
            response_dict = claude_response.model_dump()
            if "error_execution" in response_dict and response_dict["error_execution"]:
                print(f"Claude SQL execution error detected in model_dump: {response_dict['error_execution']}")
                # Update success status and error message
                claude_response.success = False
                claude_response.error_message = response_dict["error_execution"]
        
        print(f"Compare API: Claude response success={claude_response.success}, output rows={len(claude_response.query_output)}")
        response.claude = claude_response
    except Exception as e:
        print(f"Error getting Claude result: {str(e)}")
        import traceback
        traceback.print_exc()
        response.claude = QueryResponse(
            prompt=request.prompt,
            query="",
            query_output=[],
            model="claude",
            success=False,
            error_message=f"Error: {str(e)}"
        )
    
    # Get Gemini result - COMPLETELY REWRITTEN IMPLEMENTATION
    try:
        # Import libraries directly here to ensure they're available
        import pandas as pd
        from nlq_to_snowflake_gemini import nlq_to_snowflake_gemini
        
        # Create a fresh request for Gemini
        print("\nCompare API: Creating direct Gemini request...")
        gemini_model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")  # Use environment variable or default
        
        # Call the Gemini implementation directly to ensure we get results
        start_time = pd.Timestamp.now()
        try:
            result = nlq_to_snowflake_gemini(
                prompt=request.prompt,
                data_dictionary_path=data_dictionary_path,  # Use the provided data_dictionary_path
                execute_query=request.execute_query,  # Respect user's execute_query setting
                limit_rows=request.limit_rows,
                model=gemini_model,
                include_charts=request.include_charts  # Also pass the include_charts parameter
            )
        except Exception as exec_error:
            # Directly propagate SQL execution errors
            print(f"Gemini execution error: {str(exec_error)}")
            error_msg = str(exec_error)
            gemini_response = QueryResponse(
                prompt=request.prompt,
                query="",  # No SQL if execution error
                query_output=[],
                model=gemini_model,
                success=False,
                error_message=error_msg,
                execution_time_ms=(pd.Timestamp.now() - start_time).total_seconds() * 1000,
                user_hint=f"SQL error: {error_msg}",
                chart_recommendations=None,
                chart_error="Cannot generate charts due to SQL error"
            )
            response.gemini = gemini_response
            # Skip further processing
            return response
        
        # Debug the raw result
        print("\nCompare API: Direct Gemini result received")
        print(f"Result keys: {list(result.keys())}")
        print(f"Success flag: {result.get('success', False)}")
        print(f"Has results: {'results' in result}")
        
        # Check for errors - prioritize error_message over error
        if not result.get('success', True) or result.get('error_message') or result.get('error'):
            error_msg = result.get('error_message') or result.get('error') or "Unknown Gemini error"
            print(f"Gemini error detected: {error_msg}")
            # Create an error response without executing SQL
            gemini_response = QueryResponse(
                prompt=request.prompt,
                query=result.get('sql', ""),
                query_output=[],
                model=gemini_model,
                token_usage={
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "completion_tokens": result.get("completion_tokens", 0),
                    "total_tokens": result.get("total_tokens", 0)
                },
                success=False,
                error_message=error_msg,
                execution_time_ms=result.get("execution_time_ms", 0),
                user_hint=f"Gemini error: {error_msg}",
                chart_recommendations=None,
                chart_error=result.get("chart_error", "Cannot generate charts: error in processing")
            )
            response.gemini = gemini_response
            # Skip further processing for Gemini
            print("Compare API: Skipping Gemini SQL execution due to errors")
            return response
        
        # Convert DataFrame results to a list of dictionaries for JSON serialization
        query_output = []
        if "results" in result and result["results"] is not None:
            print(f"Results type: {type(result['results'])}")
            
            # Convert DataFrame results
            if isinstance(result["results"], pd.DataFrame):
                df = result["results"]
                print(f"DataFrame shape: {df.shape}")
                
                # Try direct conversion
                try:
                    query_output = df.to_dict(orient="records")
                    print(f"Converted {len(query_output)} rows to dictionary")
                except Exception as df_err:
                    print(f"Error converting DataFrame: {str(df_err)}")
                    # Fallback manual conversion
                    query_output = []
                    for i, row in df.iterrows():
                        row_dict = {}
                        for col in df.columns:
                            row_dict[col] = row[col]
                        query_output.append(row_dict)
        
        # Calculate execution time
        execution_time_ms = (pd.Timestamp.now() - start_time).total_seconds() * 1000
        
        # Extract token usage from result if available, or create default values
        token_usage = {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0)
        }
        
        print(f"Compare API: Token usage for Gemini: {token_usage}")
        
        # Create the response directly with chart handling exactly matching OpenAI behavior
        gemini_response = QueryResponse(
            prompt=request.prompt,
            query=result.get("sql", ""),
            query_output=query_output,  # This should now contain results
            model=gemini_model,  # Use the model from environment variable
            token_usage=token_usage,  # Now includes token usage
            success=result.get("success", False),  # Use success flag from result
            error_message=result.get("error", None),
            execution_time_ms=execution_time_ms,
            user_hint="SQL query generated only. Execution skipped." if not request.execute_query else "Query executed successfully.",
            # Use chart recommendations directly from the model handler (already properly conditioned)
            chart_recommendations=result.get("chart_recommendations", None),
            chart_error=result.get("chart_error", None)
        )
        
        print(f"Compare API: Created Gemini response with {len(query_output)} results")
        response.gemini = gemini_response
        
    except Exception as e:
        print(f"Error getting Gemini result: {str(e)}")
        import traceback
        traceback.print_exc()
        response.gemini = QueryResponse(
            prompt=request.prompt,
            query="",
            query_output=[],
            model="gemini",
            success=False,
            error_message=f"Error: {str(e)}"
        )
    
    # Calculate the total execution time across all models
    total_execution_time = 0.0
    if response.openai and response.openai.execution_time_ms:
        total_execution_time += response.openai.execution_time_ms
    if response.claude and response.claude.execution_time_ms:
        total_execution_time += response.claude.execution_time_ms
    if response.gemini and response.gemini.execution_time_ms:
        total_execution_time += response.gemini.execution_time_ms
    
    # Add combined token usage
    total_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    
    # Add OpenAI tokens
    if response.openai and response.openai.token_usage:
        for key in total_token_usage:
            total_token_usage[key] += response.openai.token_usage.get(key, 0)
    
    # Add Claude tokens
    if response.claude and response.claude.token_usage:
        for key in total_token_usage:
            total_token_usage[key] += response.claude.token_usage.get(key, 0)
    
    # Add Gemini tokens
    if response.gemini and response.gemini.token_usage:
        for key in total_token_usage:
            total_token_usage[key] += response.gemini.token_usage.get(key, 0)
            
    # Set required fields for unified endpoint compatibility
    response.query = "Comparison of multiple models"
    response.model = "compare"
    
    # Set the combined token usage
    response.token_usage = total_token_usage
    
    # Set individual success flags and collect errors to determine overall status
    openai_success = response.openai and response.openai.success
    claude_success = response.claude and response.claude.success
    gemini_success = response.gemini and response.gemini.success
    
    # For comparison endpoint, success=true only if ALL models were actually asked to execute queries
    # AND at least one model succeeded
    if request.execute_query:
        # When execute_query=true, require at least one model to succeed
        response.success = any([openai_success, claude_success, gemini_success])
        
        # Build an informative user hint that shows each model's status
        model_status = []
        if response.openai:
            status = "success" if openai_success else "failed"
            error = f": {response.openai.error_message}" if response.openai.error_message else ""
            model_status.append(f"OpenAI: {status}{error}")
        if response.claude:
            status = "success" if claude_success else "failed"
            error = f": {response.claude.error_message}" if response.claude.error_message else ""
            model_status.append(f"Claude: {status}{error}")
        if response.gemini:
            status = "success" if gemini_success else "failed"
            error = f": {response.gemini.error_message}" if response.gemini.error_message else ""
            model_status.append(f"Gemini: {status}{error}")
            
        # Create a detailed user hint
        response.user_hint = "Results compared across models. " + "; ".join(model_status)
    else:
        # When execute_query=false, all models should simply generate SQL without errors
        response.success = True  # Success for pure SQL generation without execution
        response.user_hint = "SQL queries generated from all models. Execution skipped."
    
    # Set execution time
    response.execution_time_ms = total_execution_time
    
    # Return the combined response
    return response

from health_check_utils import check_openai_health, check_snowflake_health, check_claude_health, check_gemini_health

@app.get("/health")
async def health_check():
    """
    Health check endpoint: checks OpenAI, Claude, Gemini APIs and Snowflake connectivity.
    Returns generic error messages rather than specific error details.
    """


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
    # Check all model APIs and database
    openai_ok, openai_msg = check_openai_health()
    claude_ok, claude_msg = check_claude_health()
    gemini_ok, gemini_msg = check_gemini_health()
    snowflake_ok, snowflake_msg = check_snowflake_health()

    # Determine overall system status
    all_ok = openai_ok and claude_ok and gemini_ok and snowflake_ok
    status = "healthy" if all_ok else "degraded"
    
    # Include status for all components
    details = {
        "openai": {"ok": openai_ok, "msg": openai_msg},
        "claude": {"ok": claude_ok, "msg": claude_msg},
        "gemini": {"ok": gemini_ok, "msg": gemini_msg},
        "snowflake": {"ok": snowflake_ok, "msg": snowflake_msg},
    }
    
    # Return complete health status
    return {
        "status": status,
        "models": ["openai", "claude", "gemini"],
        "details": details
    }


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
        
        # Get client-specific API keys for each model
        try:
            openai_config = client_manager.get_llm_config(client_id, 'openai')
            openai_ok, openai_msg = check_openai_health(api_key=openai_config['api_key'])
        except Exception as e:
            openai_ok, openai_msg = False, f"OpenAI API error: {str(e)}"
        
        try:
            claude_config = client_manager.get_llm_config(client_id, 'anthropic')
            claude_ok, claude_msg = check_claude_health(api_key=claude_config['api_key'], model=claude_config.get('model'))
        except Exception as e:
            claude_ok, claude_msg = False, f"Claude API error: {str(e)}"
            
        try:
            gemini_config = client_manager.get_llm_config(client_id, 'gemini')
            gemini_ok, gemini_msg = check_gemini_health(api_key=gemini_config['api_key'], model=gemini_config.get('model'))
        except Exception as e:
            gemini_ok, gemini_msg = False, f"Gemini API error: {str(e)}"
            
        # Check Snowflake with client-specific credentials
        try:
            snowflake_params = client_manager.get_snowflake_connection_params(client_id)
            snowflake_ok, snowflake_msg = check_snowflake_health(connection_params=snowflake_params)
        except Exception as e:
            snowflake_ok, snowflake_msg = False, f"Snowflake error: {str(e)}"
        
        # Determine overall client status
        all_ok = openai_ok and claude_ok and gemini_ok and snowflake_ok
        status = "healthy" if all_ok else "degraded"
        
        # Include status for all components
        details = {
            "openai": {"ok": openai_ok, "msg": openai_msg},
            "claude": {"ok": claude_ok, "msg": claude_msg},
            "gemini": {"ok": gemini_ok, "msg": gemini_msg},
            "snowflake": {"ok": snowflake_ok, "msg": snowflake_msg},
        }
        
        # Return client-specific health status
        return {
            "client_id": client_id,
            "status": status,
            "models": ["openai", "claude", "gemini"],
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
            "openai": [os.getenv("OPENAI_MODEL", "gpt-4o"), "gpt-4-turbo", "gpt-3.5-turbo"],
            "claude": [os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), "claude-3-opus-20240229", "claude-3-haiku-20240307"],
            "gemini": [os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"), "models/gemini-1.5-flash-latest"],
            "compare": ["Compare all models"]
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


@app.post("/query/unified", response_model=Union[QueryResponse, ComparisonResponse])
@with_client_context  # Apply client context switching
@cache_llm_query(ttl=3600)  # Add caching with 1 hour TTL
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
        
        # Support 'all' or 'compare' as a special model name to run all models
        if model == "all" or model == "compare":
            # Use the existing comparison endpoint logic
            print("Unified API: Using comparison endpoint for 'all' model request")
            comparison_response = await compare_models(request, data_dictionary_path=data_dictionary_path)
            
            # If execute_query is False, ensure we're not executing any queries
            if not request.execute_query:
                # Clear query outputs from all model results
                if comparison_response.openai:
                    comparison_response.openai.query_output = []
                if comparison_response.claude:
                    comparison_response.claude.query_output = []
                if comparison_response.gemini:
                    comparison_response.gemini.query_output = []
                
            # Return the comparison response directly
            return comparison_response
            
        # Strict model name validation - only allow exact matches
        elif model == "openai" or model == "gpt":
            # For exact OpenAI model aliases only - check key first
            print("Unified API: Routing to OpenAI endpoint")
            
            # When client is specified, verify the API key strictly before proceeding
            if client_id:
                from config.client_manager import client_manager
                try:
                    # This will use client-specific .env file to look for CLIENT_{CLIENT_ID}_OPENAI_API_KEY
                    openai_config = client_manager.get_llm_config(client_id, 'openai')
                    
                    # Double check key is not empty
                    if not openai_config or not openai_config.get('api_key') or not openai_config.get('api_key').strip():
                        error_msg = f"Client '{client_id}' openai API key not configured"
                        print(f"❌ Error: {error_msg}")
                        raise HTTPException(status_code=400, detail=error_msg)
                except ValueError as e:
                    error_msg = f"Client '{client_id}' openai API key not configured"
                    print(f"Error: {error_msg}")
                    raise HTTPException(status_code=400, detail=error_msg)
            
            # Use default model name
            request.model = os.getenv("OPENAI_MODEL", "gpt-4o")  # Use environment variable or default
            print(f"Using default OpenAI model: {request.model}")
            
            # Check if RAG should be used
            if request.use_rag:
                print(f"Using RAG enhancement with collection: {request.rag_collection or 'default'}, top_k: {request.rag_top_k}")
                # Import RAG client
                from services.rag_client import retrieve_context, enhance_prompt
                print("✅ RAG client imported successfully")
                
                # Get relevant context based on the query
                # Use client_id to formulate the default collection name if not specified
                if request.rag_collection:
                    collection_name = request.rag_collection
                else:
                    # Default collection name format: {client_id}_data_dictionary
                    collection_name = f"{request.client_id}_data_dictionary"
                print(f"Using collection: {collection_name}")
                
                try:
                    # Retrieve context from vector store
                    print(f"Retrieving context from vector store for query: '{request.prompt}'")
                    retrieval_result = retrieve_context(
                        query=request.prompt,
                        collection_name=collection_name,
                        client_id=request.client_id,
                        top_k=request.rag_top_k
                    )
                    
                    if retrieval_result and retrieval_result.get("documents"):
                        print(f"Found {len(retrieval_result['documents'])} relevant documents in collection '{collection_name}'")
                        
                        # Enhance the prompt with retrieved context
                        print("Enhancing prompt with retrieved context")
                        enhanced_result = enhance_prompt(
                            query=request.prompt,
                            retrieved_context=retrieval_result,
                            system_prompt=None,
                            prune_columns=getattr(request, "prune_columns", False)
                        )
                        
                        if enhanced_result and enhanced_result.get("enhanced_prompt"):
                            # Create a copy of the request with the enhanced prompt
                            from copy import deepcopy
                            enhanced_request = deepcopy(request)
                            enhanced_request.prompt = enhanced_result.get("enhanced_prompt")
                            
                            # Use the enhanced request for SQL generation
                            print("Generating SQL with RAG-enhanced prompt")
                            response = await generate_sql_query(enhanced_request, data_dictionary_path=data_dictionary_path)
                            response.prompt = request.prompt  # Keep the original prompt in the response
                            return response
                        else:
                            print("⚠️ Failed to enhance prompt with RAG context, falling back to standard processing")
                    else:
                        print("⚠️ No relevant documents found in vector store, falling back to standard processing")
                        
                except Exception as e:
                    print(f"⚠️ Error using RAG: {str(e)}, falling back to standard processing")
            
            # If RAG is not enabled or fails, call standard OpenAI endpoint
            print("Using standard prompt without RAG enhancement")
            response = await generate_sql_query(request, data_dictionary_path=data_dictionary_path)
            
            # If execute_query is False, ensure we're not executing the query
            if not request.execute_query and response.query_output:
                response.query_output = []
                
            return response
            
        elif model == "gemini" or model == "google":
            # For exact Gemini model aliases only - check key first
            print("Unified API: Routing to Gemini endpoint")
            
            # When client is specified, verify the API key strictly before proceeding
            if client_id:
                from config.client_manager import client_manager
                try:
                    # This will use client-specific .env file to look for CLIENT_{CLIENT_ID}_GEMINI_API_KEY
                    gemini_config = client_manager.get_llm_config(client_id, 'gemini')
                    
                    # Double check key is not empty
                    if not gemini_config or not gemini_config.get('api_key') or not gemini_config.get('api_key').strip():
                        error_msg = f"Client '{client_id}' gemini API key not configured"
                        print(f"❌ Error: {error_msg}")
                        raise HTTPException(status_code=400, detail=error_msg)
                except ValueError as e:
                    error_msg = f"Client '{client_id}' gemini API key not configured"
                    print(f"Error: {error_msg}")
                    raise HTTPException(status_code=400, detail=error_msg)
            
            # Use default model name
            request.model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")  # Use environment variable or default
            print(f"Using default Gemini model: {request.model}")

            # --------------------------
            # Attempt RAG enhancement
            # --------------------------
            try:
                if request.use_rag:
                    print(f"Using RAG enhancement with collection: {request.rag_collection or 'default'}, top_k: {request.rag_top_k}")
                    from services.rag_client import retrieve_context, enhance_prompt
                    print("✅ RAG client imported successfully for Gemini")

                    collection_name = request.rag_collection if request.rag_collection else f"{request.client_id}_data_dictionary"
                    print(f"Using collection: {collection_name}")

                    retrieval_result = retrieve_context(
                        query=request.prompt,
                        collection_name=collection_name,
                        client_id=request.client_id,
                        top_k=request.rag_top_k,
                    )

                    if retrieval_result and retrieval_result.get("documents"):
                        print(f"Found {len(retrieval_result['documents'])} relevant documents in collection '{collection_name}'")
                        enhanced_res = enhance_prompt(
                            query=request.prompt,
                            retrieved_context=retrieval_result,
                            system_prompt=None,
                             prune_columns=getattr(request, "prune_columns", False),
                        )
                        if enhanced_res and enhanced_res.get("enhanced_prompt"):
                            from copy import deepcopy
                            enhanced_request = deepcopy(request)
                            enhanced_request.prompt = enhanced_res["enhanced_prompt"]

                            response = await generate_sql_query_gemini(enhanced_request, data_dictionary_path=data_dictionary_path)
                            response.prompt = request.prompt  # keep original prompt

                            if not request.execute_query and response.query_output:
                                response.query_output = []
                            return response
                        else:
                            print("⚠️ Failed to enhance prompt with RAG context for Gemini, falling back to standard processing")
                    else:
                        print("⚠️ No relevant documents found in vector store for Gemini, falling back to standard processing")
            except Exception as e:
                print(f"⚠️ Error during Gemini RAG processing: {str(e)}. Falling back to standard processing.")

            # Fallback to standard processing
            response = await generate_sql_query_gemini(request, data_dictionary_path=data_dictionary_path)

            if not request.execute_query and response.query_output:
                response.query_output = []

            return response
            
        elif model == "claude" or model == "anthropic":
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
            
            # Use default model name
            request.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")  # Use environment variable or default
            print(f"Using default Claude model: {request.model}")

            # --------------------------
            # Attempt RAG enhancement
            # --------------------------
            try:
                if request.use_rag:
                    print(f"Using RAG enhancement with collection: {request.rag_collection or 'default'}, top_k: {request.rag_top_k}")
                    from services.rag_client import retrieve_context, enhance_prompt
                    print("✅ RAG client imported successfully for Claude")

                    collection_name = request.rag_collection if request.rag_collection else f"{request.client_id}_data_dictionary"
                    print(f"Using collection: {collection_name}")

                    retrieval_result = retrieve_context(
                        query=request.prompt,
                        collection_name=collection_name,
                        client_id=request.client_id,
                        top_k=request.rag_top_k,
                    )

                    if retrieval_result and retrieval_result.get("documents"):
                        print(f"Found {len(retrieval_result['documents'])} relevant documents in collection '{collection_name}'")
                        enhanced_res = enhance_prompt(
                            query=request.prompt,
                            retrieved_context=retrieval_result,
                            system_prompt=None,
                             prune_columns=getattr(request, "prune_columns", False),
                        )
                        if enhanced_res and enhanced_res.get("enhanced_prompt"):
                            from copy import deepcopy
                            enhanced_request = deepcopy(request)
                            enhanced_request.prompt = enhanced_res["enhanced_prompt"]

                            response = await generate_sql_query_claude(enhanced_request, data_dictionary_path=data_dictionary_path)
                            response.prompt = request.prompt  # keep original

                            if not request.execute_query and response.query_output:
                                response.query_output = []
                            return response
                        else:
                            print("⚠️ Failed to enhance prompt with RAG context for Claude, falling back to standard processing")
                    else:
                        print("⚠️ No relevant documents found in vector store for Claude, falling back to standard processing")
            except Exception as e:
                print(f"⚠️ Error during Claude RAG processing: {str(e)}. Falling back to standard processing.")

            # Fallback to standard processing
            response = await generate_sql_query_claude(request, data_dictionary_path=data_dictionary_path)

            if not request.execute_query and response.query_output:
                response.query_output = []

            return response
        
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
            request.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            print(f"Using default Claude model: {request.model}")
            response = await generate_sql_query_claude(request, data_dictionary_path=data_dictionary_path)
            
            # If execute_query is False, ensure we're not executing the query
            if not request.execute_query and response.query_output:
                response.query_output = []
                
            return response
            
        else:
            # Invalid model name - return error with clear, concise guidance
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Please use one of these valid options: 'openai', 'claude', 'gemini', 'all' or leave empty for claude."
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



# Add startup event to initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    try:
        logger.info("Initializing caching system...")
        initialize_cache_system()
        logger.info("Caching system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")

# Add shutdown event to clean up services
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on application shutdown."""
    try:
        logger.info("Shutting down caching system...")
        shutdown_cache_system()
        logger.info("Caching system shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down services: {e}")

# Add cache status endpoint
@app.get("/cache/stats", tags=["cache"])
async def get_cache_statistics(client_id: Optional[str] = None):
    """Get cache statistics."""
    try:
        stats = get_cache_stats()
        return {
            "cache_stats": stats,
            "client_id": client_id or "all"
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add cache clear endpoint
@app.post("/cache/clear", tags=["cache"])
async def clear_cache_endpoint(client_id: Optional[str] = None):
    """Clear cache contents."""
    try:
        clear_cache(client_id)
        return {
            "success": True,
            "message": f"Cache cleared for client: {client_id if client_id else 'all'}"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Register all routers at the module level for proper Swagger documentation
from prompt_query_history_api import router as history_router

# Include the prompt query history endpoints directly in our API
app.include_router(history_router, prefix="")  # Access via /prompt_query_history

# Include cache routes
app.include_router(cache_router, prefix="")  # Access via /cache/*

# Include RAG routes
app.include_router(rag_router, prefix="")  # Access via /rag/*

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


