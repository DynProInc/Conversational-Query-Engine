"""
Natural Language to Snowflake Query Runner

This script combines the LLM query generator with Snowflake execution:
1. Takes a natural language query
2. Uses OpenAI to convert it to SQL
3. Executes the SQL against your Snowflake account
4. Returns the results
"""
import os
import time
import pandas as pd
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import our local modules
from llm_query_generator import natural_language_to_sql
from snowflake_runner import execute_query as run_snowflake_query
from token_logger import TokenLogger

# Load environment variables
load_dotenv()


def nlq_to_snowflake(prompt: str, model: str = None, data_dictionary_path: str = None, 
                     limit_rows: int = 100, execute_query: bool = True, include_charts: bool = False,
                     client_id: str = None, use_rag: bool = False, top_k: int = 10, 
                     enable_reranking: bool = True) -> Dict[str, Any]:
    """
    End-to-end pipeline to convert natural language to SQL and execute in Snowflake
    
    Args:
        prompt: Natural language question to convert to SQL
        model: Specific OpenAI model to use, overrides OPENAI_MODEL environment variable
        data_dictionary_path: Path to data dictionary CSV/Excel file
        limit_rows: Maximum number of rows to return (adds LIMIT if not in query)
        execute_query: Whether to execute the SQL in Snowflake (True) or just return SQL (False)
        include_charts: Whether to include chart recommendations from LLM
        client_id: Client ID for RAG context retrieval
        use_rag: Whether to use RAG for context retrieval
        top_k: Number of top results to return from RAG (default: 10)
        enable_reranking: Whether to enable reranking of RAG results (default: True)
        
    Returns:
        Dictionary with SQL, results (if executed), and metadata
    """
    # Start timing the execution
    start_time = time.time()
    
    # Ensure dictionary path is provided
    if not data_dictionary_path:
        # Raise an error instead of silently falling back to MTS dictionary
        raise ValueError("No dictionary path provided. Cannot proceed without a data dictionary.")
    
    # Use provided model, or get from environment, or use default
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    
    print(f"Using model: {model}")
    print(f"Converting: '{prompt}' to SQL...")
    
    try:
        # Generate SQL from natural language using OpenAI
        sql_result = natural_language_to_sql(
            query=prompt,
            data_dictionary_path=data_dictionary_path,
            model=model,
            limit_rows=limit_rows,
            include_charts=include_charts,
            client_id=client_id,  # Add client_id parameter for RAG
            use_rag=use_rag,      # Add use_rag parameter
            top_k=top_k,          # Add top_k parameter for RAG
            enable_reranking=enable_reranking  # Add enable_reranking parameter
        )
        
        # Extract the SQL and clean it
        sql = sql_result["sql"]

        # Clean up markdown formatting if present
        if sql.startswith("```"):
            # Remove markdown code blocks if present
            lines = sql.split("\n")
            # Remove opening ```sql or ``` line
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing ``` line if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            # Join back into a cleaned SQL string
            sql = "\n".join(lines)

        print("\nGenerated SQL:")
        print(sql)

        # Check if the SQL is actually a JSON object with a 'sql' field
        import json
        try:
            # If the SQL starts with a curly brace, it might be JSON
            if sql.strip().startswith('{'):
                print("Detected JSON-formatted query, attempting to parse")
                json_data = json.loads(sql)
                if 'sql' in json_data:
                    print("Extracted SQL from JSON structure")
                    # Extract the SQL and chart recommendations
                    actual_sql = json_data.get('sql', '')
                    chart_recs = json_data.get('chart_recommendations', [])
                    
                    # Update our variables
                    sql = actual_sql
                    sql_result['chart_recommendations'] = chart_recs
        except json.JSONDecodeError:
            # Not valid JSON, continue with the original SQL
            print("Not valid JSON, continuing with original SQL")
            pass
            
        # Always store cleaned SQL and token usage in result dict before execution
        response = {"sql": sql}
        # Defensive: propagate token usage if present
        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            if k not in sql_result:
                sql_result[k] = 0
        response["prompt_tokens"] = sql_result["prompt_tokens"]
        response["completion_tokens"] = sql_result["completion_tokens"]
        response["total_tokens"] = sql_result["total_tokens"]

        # Add LIMIT if not present and execute flag is True
        if execute_query and limit_rows > 0 and "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip('; \n')} LIMIT {limit_rows}"
            print(f"\nAdded limit clause: LIMIT {limit_rows}")
            response["sql_with_limit"] = sql

        # Log token usage for both executed and non-executed queries
        query_executed_value = None  # Default for non-executed queries
        
        # Execute in Snowflake if requested
        if execute_query:
            print("\nExecuting in Snowflake...")
            query_executed_successfully = False
            try:
                df = run_snowflake_query(sql, print_results=True)
                response["results"] = df
                response["success"] = True
                response["row_count"] = len(df)
                query_executed_successfully = True
                query_executed_value = 1  # Success
            except Exception as e:
                print(f"\nError executing SQL in Snowflake: {str(e)}")
                response["success"] = False
                response["error_execution"] = str(e)
                query_executed_value = 0  # Failed execution
        
        # Always log token usage with appropriate query_executed status
        try:
            logger = TokenLogger()
            logger.log_usage(
                model=model,
                query=prompt,  # Original natural language query
                usage={
                    "prompt_tokens": sql_result.get("prompt_tokens", 0),
                    "completion_tokens": sql_result.get("completion_tokens", 0),
                    "total_tokens": sql_result.get("total_tokens", 0)
                },
                # Only log the actual user query, not the system prompt with data dictionary
                prompt=prompt,  # Just the user's natural language question
                sql_query=sql,
                query_executed=query_executed_value  # 1=success, 0=failed, None=not executed
            )
        except Exception as log_err:
            print(f"Error logging token usage: {str(log_err)}")

        # Calculate execution time and add to response
        execution_time_ms = (time.time() - start_time) * 1000
        response["execution_time_ms"] = execution_time_ms
        print(f"Total execution time: {execution_time_ms:.2f} ms")
        
        # Always include chart-related fields in the response with appropriate values
        # This ensures consistency in response structure regardless of parameter combinations
        if include_charts:
            results = response.get("results")
            if response.get("success") and isinstance(results, pd.DataFrame) and not results.empty:
                # Add chart recommendations from LLM result or empty lists if not available
                response["chart_recommendations"] = sql_result.get("chart_recommendations", [])
                response["chart_error"] = sql_result.get("chart_error", None)
            else:
                # Add empty chart recommendations and error message if execution failed
                response["chart_recommendations"] = []
                response["chart_error"] = "Cannot generate charts: query execution failed or returned no results"
        else:
            # Set chart fields to null when charts aren't requested
            response["chart_recommendations"] = None
            response["chart_error"] = None
                            
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Calculate execution time even for errors
        execution_time_ms = (time.time() - start_time) * 1000
        error_response = {
            "error": str(e),
            "success": False,
            "execution_time_ms": execution_time_ms
        }
        
        # Always include chart-related fields for consistent response structure
        if include_charts:
            error_response["chart_recommendations"] = []
            error_response["chart_error"] = f"Cannot generate charts: {str(e)}"
        else:
            error_response["chart_recommendations"] = None
            error_response["chart_error"] = None
            
        return error_response


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert natural language to SQL and execute in Snowflake")
    parser.add_argument("prompt", help="Natural language question to convert to SQL")
    parser.add_argument("--data-dictionary", "-d", dest="data_dictionary_path", help="Path to data dictionary CSV/Excel file")
    parser.add_argument("--model", "-m", help="Specific OpenAI model to use")
    parser.add_argument("--limit-rows", "-l", type=int, default=100, help="Maximum number of rows to return")
    parser.add_argument("--execute-query", "-e", type=lambda x: x.lower() in ["true", "yes", "1", "t"], 
                        default=True, help="Whether to execute the SQL in Snowflake (True/False)")
    parser.add_argument("--include-charts", "-c", type=lambda x: x.lower() in ["true", "yes", "1", "t"], 
                        default=False, help="Whether to include chart recommendations (True/False)")
    
    args = parser.parse_args()
    
    # Debug prints
    print("=== Argument Values ===")
    print(f"Prompt: {args.prompt}")
    print(f"Model: {args.model}")
    print(f"Data Dictionary: {args.data_dictionary_path}")
    print(f"Limit Rows: {args.limit_rows}")
    print(f"Execute Query: {args.execute_query} (type: {type(args.execute_query)})")
    print(f"Include Charts: {args.include_charts} (type: {type(args.include_charts)})")
    print("=======================")
    
    # Run the pipeline
    result = nlq_to_snowflake(
        prompt=args.prompt,
        model=args.model,
        data_dictionary_path=args.data_dictionary_path,
        limit_rows=args.limit_rows,
        execute_query=args.execute_query,
        include_charts=args.include_charts
    )
    
    # Print the structure of the result
    print("\n=== Result Structure ===")
    print(f"Keys: {list(result.keys())}")
    print("======================")
    
    # Display SQL query
    if "sql" in result:
        print("\n=== Generated SQL ===")
        print(result.get('sql'))
        print("====================")
    
    # Display token usage for transparency
    if "prompt_tokens" in result:
        print("\n=== Token Usage ===")
        print(f"  Model: {result.get('model')}")
        print(f"  Input tokens: {result.get('prompt_tokens')}")
        print(f"  Output tokens: {result.get('completion_tokens')}")
        print(f"  Total tokens: {result.get('total_tokens')}")
        print("===================")
        
    # Display chart-related information
    print("\n=== Chart Information ===")
    print(f"  include_charts parameter: {args.include_charts}")
    print(f"  chart_recommendations: {type(result.get('chart_recommendations'))}")
    print(f"  chart_error: {result.get('chart_error')}")
    print("=========================")
