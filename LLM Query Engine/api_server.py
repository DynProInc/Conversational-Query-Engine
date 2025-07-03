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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LLM SQL Query Engine API",
    description="Convert natural language to SQL and execute in Snowflake using OpenAI or Claude",
    version="1.1.0"
)

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
        print("DEBUG: Claude endpoint called with prompt:", request.prompt)
        # First generate SQL using Claude
        claude_model = request.model if request.model else "claude-3-5-sonnet-20241022"  # Use the requested model or default
        print(f"DEBUG: Using Claude model: {claude_model}")
        
        try:
            claude_result = natural_language_to_sql_claude(
                query=request.prompt,
                data_dictionary_path=request.data_dictionary_path,
                model=claude_model
            )
            print(f"DEBUG: Full Claude result: {claude_result}")
            print(f"DEBUG: Claude result keys: {claude_result.keys() if claude_result else 'None'}")
        except Exception as e:
            print(f"DEBUG: Error in natural_language_to_sql_claude: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Check for errors in Claude SQL generation
        if "error" in claude_result:
            return QueryResponse(
                prompt=request.prompt,
                query=claude_result.get("sql", ""),
                query_output=[],
                model=claude_result.get("model", "claude"),
                success=False,
                error_message=claude_result.get("error", "Unknown Claude error"),
                execution_time_ms=claude_result.get("execution_time_ms")
            )
            
        # If query execution is requested, execute the Claude-generated SQL
        query_output = []
        if request.execute_query:
            from snowflake_runner import execute_query
            
            try:
                # Add LIMIT if not already present
                if "sql" not in claude_result:
                    print("DEBUG: 'sql' key not found in Claude result")
                    claude_result["sql"] = "SELECT 'No SQL query was generated' AS message"
                    raise ValueError("SQL query key is missing in Claude result")
                    
                sql = claude_result.get("sql")
                print(f"DEBUG: SQL from Claude: {sql}")
                print(f"DEBUG: SQL type: {type(sql)}")

                # HARD GUARD: Ensure SQL is a string before any further processing
                if sql is None or not isinstance(sql, str):
                    print(f"DEBUG: SQL is None or not a string ({type(sql)}), forcing fallback SQL.")
                    sql = "SELECT 'No valid SQL query was generated (non-string SQL)' AS message"
                    claude_result["sql"] = sql
                
                print("DEBUG: About to check if SQL is None")
                
                # First, ensure SQL is not None before any processing
                if sql is None:
                    print("DEBUG: SQL is None, using fallback SQL")
                    sql = "SELECT 'No SQL could be generated' AS message"
                    claude_result["sql"] = sql
                # SQL should be a string at this point
                elif not isinstance(sql, str):
                    print(f"DEBUG: Converting SQL from {type(sql)} to string")
                    sql = str(sql)
                    claude_result["sql"] = sql
                
                print("DEBUG: About to check for LIMIT clause")
                print(f"DEBUG: request.limit_rows = {request.limit_rows}")
                print(f"DEBUG: SQL before LIMIT check: '{sql}', type: {type(sql)}")
                
                # HARD GUARD: Ensure SQL is a string
                if not isinstance(sql, str):
                    print(f"DEBUG: SQL is not a string ({type(sql)}), forcing fallback SQL.")
                    sql = "SELECT 'No valid SQL query was generated (non-string SQL)' AS message"
                    claude_result["sql"] = sql
                
                # Safely check and add LIMIT clause if needed
                if sql and request.limit_rows > 0:
                    if "LIMIT" not in sql.upper():
                        try:
                            # Use regular strip instead of rstrip to be safer
                            sql = sql.strip().rstrip(';') + f" LIMIT {request.limit_rows}"
                            claude_result["sql"] = sql
                            print(f"DEBUG: SQL after adding LIMIT: '{sql}'")
                        except Exception as e:
                            print(f"DEBUG: Error adding LIMIT clause: {str(e)}")
                else:
                    print("DEBUG: Not adding LIMIT clause - SQL is empty or limit_rows=0")
                
                print("DEBUG: About to execute query")
                    
                # Execute the query
                try:
                    print(f"DEBUG: Final SQL before execution: '{sql}'")
                    df = execute_query(sql)
                    print("DEBUG: Query executed successfully")
                except Exception as e:
                    print(f"DEBUG: Error executing query: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Handle SQL execution error but don't crash
                    return QueryResponse(
                        prompt=request.prompt,
                        query=sql,
                        query_output=[],
                        model=claude_result["model"],
                        success=False,
                        error_message=f"Error executing SQL: {str(e)}",
                        execution_time_ms=claude_result.get("execution_time_ms"),
                        token_usage={
                            "prompt_tokens": claude_result.get("prompt_tokens", 0),
                            "completion_tokens": claude_result.get("completion_tokens", 0),
                            "total_tokens": claude_result.get("total_tokens", 0)
                        }
                    )
                query_output = df.to_dict(orient="records")
                claude_result["success"] = True
            except Exception as exec_error:
                claude_result["error_execution"] = str(exec_error)
                claude_result["success"] = False
                return QueryResponse(
                    prompt=request.prompt,
                    query=claude_result["sql"],
                    query_output=[],
                    model=claude_result["model"],
                    success=False,
                    error_message=f"SQL generated successfully but execution failed: {str(exec_error)}",
                    execution_time_ms=claude_result.get("execution_time_ms"),
                    token_usage={
                        "prompt_tokens": claude_result.get("prompt_tokens", 0),
                        "completion_tokens": claude_result.get("completion_tokens", 0),
                        "total_tokens": claude_result.get("total_tokens", 0)
                    }
                )
        
        # Extract token usage
        token_usage = None
        if all(k in claude_result for k in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            token_usage = {
                "prompt_tokens": claude_result["prompt_tokens"],
                "completion_tokens": claude_result["completion_tokens"],
                "total_tokens": claude_result["total_tokens"]
            }
        
        # Build the response
        return QueryResponse(
            prompt=request.prompt,
            query=claude_result["sql"],
            query_output=query_output,
            model=claude_result["model"],
            token_usage=token_usage,
            success=True,
            error_message=None,
            execution_time_ms=claude_result.get("execution_time_ms")
        )
    
    except Exception as e:
        # Handle any unexpected errors
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
