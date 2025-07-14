"""
Natural Language to Snowflake Query Runner for Gemini
This module handles the Gemini-specific implementation of:
1. Takes a natural language query
2. Uses Gemini to convert it to SQL
3. Executes the SQL against your Snowflake account
4. Returns the results
"""
import os
import pandas as pd
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import our local modules
from gemini_query_generator import natural_language_to_sql_gemini
from snowflake_runner import execute_query as run_snowflake_query
from token_logger import TokenLogger

# Load environment variables
load_dotenv()

def nlq_to_snowflake_gemini(prompt: str,
                     data_dictionary_path: Optional[str] = None,
                     execute_query: bool = True,
                     limit_rows: int = 100,
                     model: str = None,
                     include_charts: bool = False) -> Dict[str, Any]:
    """
    End-to-end function to convert natural language to SQL using Gemini, then optionally execute in Snowflake.
    Handles chart recommendations if include_charts is True and execute_query is True.
    """
    # Use environment variable if model not specified
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")
    # Ensure dictionary path is provided
    if not data_dictionary_path:
        # Raise an error instead of silently falling back to MTS dictionary
        raise ValueError("No dictionary path provided. Cannot proceed with Gemini.")
    print(f"Using model: {model}")
    print(f"Converting: '{prompt}' to SQL using Gemini...")
    start_time = pd.Timestamp.now()
    try:
        # Step 1: Use Gemini to generate SQL
        gemini_result = natural_language_to_sql_gemini(
            query=prompt,
            data_dictionary_path=data_dictionary_path,
            model=model,
            log_tokens=True,
            limit_rows=limit_rows,
            include_charts=include_charts
        )
        sql = gemini_result.get("sql", "")
        if not isinstance(sql, str):
            sql = str(sql) if sql is not None else "SELECT 'Non-string SQL' AS message"
            gemini_result["sql"] = sql
        if not sql.strip():
            sql = "SELECT 'Empty SQL query' AS message"
            gemini_result["sql"] = sql
        print("\nGenerated SQL:")
        print(sql)
        # Handle LIMIT clause based on user preference and priority rules
        if execute_query:
            sql = sql.strip()
            if sql.endswith(';'):
                sql = sql[:-1]
                
            # Check if a LIMIT clause already exists (user prompt limit takes first priority)
            import re
            limit_pattern = re.compile(r'LIMIT\s+\d+', re.IGNORECASE)
            limit_match = limit_pattern.search(sql)
            
            if limit_match:
                # A LIMIT already exists in the query - respect it as it's from the LLM following the user's prompt
                # Extract the limit value for logging
                limit_value = int(re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE).group(1))
                print(f"\nPreserving user-specified limit: LIMIT {limit_value}")
            elif limit_rows > 0:
                # No LIMIT clause found, add one using the limit_rows parameter (second priority)
                sql = f"{sql} LIMIT {limit_rows}"
                print(f"\nAdded limit clause: LIMIT {limit_rows}")
            # If no limit in query and limit_rows is 0, don't add any limit
                
            gemini_result["sql"] = sql
        # Log token usage for both executed and non-executed queries
        query_executed_value = None  # Default for non-executed queries

        # Execute query in Snowflake if requested
        if execute_query:
            print("\nExecuting in Snowflake...")
            query_executed_successfully = False
            try:
                df = run_snowflake_query(sql, print_results=True)
                gemini_result["results"] = df
                gemini_result["success"] = True
                gemini_result["row_count"] = len(df)
                query_executed_successfully = True
                query_executed_value = 1  # Success
            except Exception as e:
                print(f"\nError executing SQL in Snowflake: {str(e)}")
                gemini_result["error_execution"] = str(e)
                gemini_result["success"] = False
                query_executed_value = 0  # Failed execution
        else:
            # Not executing query (execute_query=False)
            print("\nSkipping execution (execute_query=False)")
            gemini_result["success"] = True  # No execution requested, so no execution error
            
        # Always log token usage with appropriate query_executed status
        try:
            logger = TokenLogger()
            logger.log_usage(
                model=model,
                query=prompt,
                usage={
                    "prompt_tokens": gemini_result.get("prompt_tokens", 0),
                    "completion_tokens": gemini_result.get("completion_tokens", 0),
                    "total_tokens": gemini_result.get("total_tokens", 0)
                },
                prompt=prompt,
                sql_query=sql,
                query_executed=query_executed_value  # 1=success, 0=failed, None=not executed
            )
        except Exception as log_err:
            print(f"Error logging token usage: {str(log_err)}")
            
        gemini_result["execution_time_ms"] = (pd.Timestamp.now() - start_time).total_seconds() * 1000
        
        # Handle chart recommendations based on execute_query parameter
        # Only include chart recommendations when the query was actually executed
        if include_charts:
            if execute_query and gemini_result.get("success") and isinstance(gemini_result.get("results"), pd.DataFrame) and not gemini_result.get("results").empty:
                # Add chart recommendations when include_charts=true AND query was executed successfully
                gemini_result["chart_recommendations"] = gemini_result.get("chart_recommendations", [])
                gemini_result["chart_error"] = None
            else:
                # No chart recommendations if query wasn't executed or failed
                gemini_result["chart_recommendations"] = None
                gemini_result["chart_error"] = "Cannot generate charts: " + ("query not executed" if not execute_query else "query execution failed or returned no results")
        return gemini_result
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = str(e).replace("'", "''")
        return {
            "sql": "",  # Don't create a fake SQL query with the error message
            "model": model,
            "error": str(e),
            "error_message": str(e),  # Explicitly include error_message
            "success": False,
            "query": prompt,
            "data_dictionary": data_dictionary_path,
            "timestamp": pd.Timestamp.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "chart_recommendations": None,
            "chart_error": f"Cannot generate charts: {error_msg}"
        }

# Command line interface for testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert natural language to SQL with Gemini and run in Snowflake")
    parser.add_argument("question", help="Natural language question to convert to SQL")
    parser.add_argument("--no-execute", action="store_true", help="Don't execute the SQL, just show it")
    parser.add_argument("--model", default="models/gemini-1.5-flash-latest", help="Gemini model to use")
    parser.add_argument("--limit", type=int, default=10, help="Row limit for SQL results")
    args = parser.parse_args()
    result = nlq_to_snowflake_gemini(
        question=args.question,
        execute=not args.no_execute,
        limit_rows=args.limit,
        model=args.model
    )
    if "results" in result and isinstance(result["results"], pd.DataFrame):
        print("\nResults:")
        print(result["results"])
    print("\nFull result:")
    print(result)
