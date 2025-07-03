import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from gemini_query_generator import natural_language_to_sql_gemini

router = APIRouter()

class GeminiQueryRequest(BaseModel):
    prompt: str
    data_dictionary_path: Optional[str] = None
    model: Optional[str] = None  # Gemini model name (default is 1.5-flash)
    log_tokens: bool = True

class GeminiQueryResponse(BaseModel):
    prompt: str
    sql: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None
    query: Optional[str] = None
    data_dictionary: Optional[str] = None

from fastapi import HTTPException
import pandas as pd
from nlq_to_snowflake_gemini import nlq_to_snowflake_gemini

@router.post("/query/gemini", response_model=GeminiQueryResponse)
def generate_sql_query_gemini(request: GeminiQueryRequest):
    """
    Generate SQL from natural language using Google Gemini and return as JSON.
    """
    model = request.model or "models/gemini-1.5-flash-latest"
    api_key = os.environ.get("GEMINI_API_KEY")
    result = natural_language_to_sql_gemini(
        query=request.prompt,
        data_dictionary_path=request.data_dictionary_path,
        api_key=api_key,
        model=model,
        log_tokens=request.log_tokens
    )
    return GeminiQueryResponse(
        prompt=request.prompt,
        sql=result.get("sql", ""),
        model=result.get("model", model),
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
        total_tokens=result.get("total_tokens", 0),
        execution_time_ms=result.get("execution_time_ms"),
        error=result.get("error"),
        timestamp=result.get("timestamp"),
        query=result.get("query"),
        data_dictionary=result.get("data_dictionary")
    )

# New endpoint for Gemini SQL generation + Snowflake execution, matching OpenAI/Claude structure
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    prompt: str
    limit_rows: int = 100
    data_dictionary_path: Optional[str] = None
    execute_query: bool = True
    model: Optional[str] = None

class QueryResponse(BaseModel):
    prompt: str
    query: str
    query_output: List[Dict[str, Any]]
    model: str
    token_usage: Optional[Dict[str, int]] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None

@router.post("/query/gemini/execute", response_model=QueryResponse)
def generate_sql_query_gemini_execute(request: QueryRequest):
    """
    Generate SQL from natural language using Gemini, execute in Snowflake, and return results in OpenAI/Claude format.
    Robust error handling and response structure matching api_server.py.
    """
    try:
        result = nlq_to_snowflake_gemini(
            question=request.prompt,
            data_dictionary_path=request.data_dictionary_path,
            execute=request.execute_query,
            limit_rows=request.limit_rows,
            model=request.model or "models/gemini-1.5-flash-latest"
        )
        # Convert DataFrame results to list of dicts
        query_output = []
        if "results" in result and isinstance(result["results"], pd.DataFrame):
            query_output = result["results"].to_dict(orient="records")
        # Extract token usage if available
        token_usage = None
        if all(k in result for k in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            token_usage = {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"]
            }
        # Check for errors
        if not result.get("success", False) and (result.get("error") or result.get("error_execution")):
            return QueryResponse(
                prompt=request.prompt,
                query=result.get("sql", ""),
                query_output=[],
                model=result.get("model", "gemini"),
                token_usage=token_usage,
                success=False,
                error_message=result.get("error") or result.get("error_execution", "Unknown error"),
                execution_time_ms=result.get("execution_time_ms")
            )
        return QueryResponse(
            prompt=request.prompt,
            query=result.get("sql", ""),
            query_output=query_output,
            model=result.get("model", "gemini"),
            token_usage=token_usage,
            success=True,
            error_message=None,
            execution_time_ms=result.get("execution_time_ms")
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query with Gemini: {str(e)}"
        )
