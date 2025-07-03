"""
FastAPI Server for LLM SQL Query Engine

This API takes natural language questions, converts them to SQL using OpenAI or Claude,
executes them against Snowflake, and returns the results.
"""
import os
import pandas as pd
from typing import Dict, Any, List, Optional, Literal
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our existing functionality
from nlq_to_snowflake import nlq_to_snowflake

# Import Claude functionality
from claude_query_generator import natural_language_to_sql_claude
from nlq_to_snowflake_claude import nlq_to_snowflake_claude

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LLM SQL Query Engine API",
    description="Convert natural language to SQL and execute in Snowflake using OpenAI or Claude",
    version="1.1.0"
)

# Import and include the prompt query history router
from prompt_query_history_route import router as prompt_query_history_router
app.include_router(prompt_query_history_router)

# Import and include the Gemini query router
from gemini_query_route import router as gemini_query_router
app.include_router(gemini_query_router)

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str
    limit_rows: int = 100
    data_dictionary_path: Optional[str] = None
    execute_query: bool = True
    model: Optional[str] = None  # For specifying a specific model

class QueryResponse(BaseModel):
    prompt: str
    query: str
    query_output: List[Dict[str, Any]]
    model: str  # Indicate which model generated the SQL
    token_usage: Optional[Dict[str, int]] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
class ComparisonResponse(BaseModel):
    prompt: str
    openai: Optional[QueryResponse] = None
    claude: Optional[QueryResponse] = None

@app.post("/query", response_model=QueryResponse)
async def generate_sql_query(request: QueryRequest):
    """Generate SQL from natural language using OpenAI and optionally execute against Snowflake"""
    try:
        # Use our existing functionality to process the query
        # Set the model in environment variable if specified
        if request.model:
            os.environ["OPENAI_MODEL"] = request.model
            
        result = nlq_to_snowflake(
            question=request.prompt,
            data_dictionary_path=request.data_dictionary_path,
            execute=request.execute_query,
            limit_rows=request.limit_rows
        )
        
        # Check for errors
        if not result.get("success", False) and "error" in result:
            return QueryResponse(
                prompt=request.prompt,
                query=result.get("sql", ""),
                query_output=[],
                model=result.get("model", "openai"),
                success=False,
                error_message=result.get("error") or result.get("error_execution", "Unknown error"),
                execution_time_ms=result.get("execution_time_ms")
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
        
        # Build the response
        return QueryResponse(
            prompt=request.prompt,
            query=result.get("sql", ""),
            query_output=query_output,
            model=result.get("model", "openai"),
            token_usage=token_usage,
            success=True,
            error_message=None,
            execution_time_ms=result.get("execution_time_ms")
        )
    
    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query with OpenAI: {str(e)}"
        )


@app.post("/query/claude", response_model=QueryResponse)
async def generate_sql_query_claude(request: QueryRequest):
    """Generate SQL from natural language using Claude and optionally execute against Snowflake"""
    try:
        # Use our modular Claude implementation
        claude_model = request.model if request.model else "claude-3-5-sonnet-20241022"
            
        result = nlq_to_snowflake_claude(
            question=request.prompt,
            data_dictionary_path=request.data_dictionary_path,
            execute=request.execute_query,
            limit_rows=request.limit_rows,
            model=claude_model
        )
        
        # Check for errors
        if not result.get("success", False) and "error" in result:
            return QueryResponse(
                prompt=request.prompt,
                query=result.get("sql", ""),
                query_output=[],
                model=result.get("model", "claude"),
                success=False,
                error_message=result.get("error") or result.get("error_execution", "Unknown error"),
                execution_time_ms=result.get("execution_time_ms")
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
        return QueryResponse(
            prompt=request.prompt,
            query=result.get("sql", ""),
            query_output=query_output,
            model=result.get("model", "claude"),
            token_usage=token_usage,
            success=True,
            error_message=None,
            execution_time_ms=result.get("execution_time_ms")
        )
        
    except Exception as e:
        # Handle any unexpected errors
        print(f"DEBUG: Unexpected error in Claude endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query with Claude: {str(e)}"
        )


@app.post("/query/compare", response_model=ComparisonResponse)
async def compare_models(request: QueryRequest):
    """Generate SQL using both OpenAI and Claude and compare results"""
    try:
        # Create a copy of the request for each model
        openai_request = QueryRequest(**request.dict())
        claude_request = QueryRequest(**request.dict())
        
        # Get results from both models (don't propagate exceptions)
        openai_response = None
        claude_response = None
        
        try:
            openai_response = await generate_sql_query(openai_request)
        except Exception as e:
            openai_response = QueryResponse(
                prompt=request.prompt,
                query="Error generating SQL",
                query_output=[],
                model="openai",
                success=False,
                error_message=str(e)
            )
            
        try:
            claude_response = await generate_sql_query_claude(claude_request)
        except Exception as e:
            claude_response = QueryResponse(
                prompt=request.prompt,
                query="Error generating SQL",
                query_output=[],
                model="claude",
                success=False,
                error_message=str(e)
            )
        
        # Return the comparison
        return ComparisonResponse(
            prompt=request.prompt,
            openai=openai_response,
            claude=claude_response
        )
    
    except Exception as e:
        # This should only happen if there's a fundamental issue
        raise HTTPException(
            status_code=500,
            detail=f"Error comparing model results: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy", "models": ["openai", "claude"]}


@app.get("/models")
async def available_models():
    """
    List available language models for SQL generation
    """
    # Get available models from environment or use defaults
    openai_models = os.environ.get("OPENAI_MODELS", "gpt-4o,gpt-4-turbo,gpt-3.5-turbo").split(",")
    claude_models = os.environ.get("CLAUDE_MODELS", "claude-3-5-sonnet,claude-3-opus,claude-3-haiku").split(",")
    
    return {
        "openai": openai_models,
        "claude": claude_models,
        "preferred": {
            "openai": openai_models[0],
            "claude": claude_models[0],
        }
    }

# Main entry point to run the server directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
