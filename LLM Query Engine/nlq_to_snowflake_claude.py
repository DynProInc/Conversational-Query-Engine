"""
Natural Language to Snowflake Query Runner for Claude
This module handles the Claude-specific implementation of:
1. Takes a natural language query
2. Uses Claude to convert it to SQL
3. Executes the SQL against your Snowflake account
4. Returns the results
"""
import os
import sys
import pandas as pd
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import our local modules
from claude_query_generator import natural_language_to_sql_claude
from snowflake_runner import execute_query

# Load environment variables
load_dotenv()


def nlq_to_snowflake_claude(question: str, 
                     data_dictionary_path: Optional[str] = None,
                     execute: bool = True,
                     limit_rows: int = 100,
                     model: str = "claude-3-5-sonnet-20241022") -> Dict[str, Any]:
    """
    End-to-end pipeline to convert natural language to SQL using Claude and execute in Snowflake
    
    Args:
        question: Natural language question to convert to SQL
        data_dictionary_path: Path to data dictionary CSV/Excel file
        execute: Whether to execute the SQL in Snowflake (True) or just return SQL (False)
        limit_rows: Maximum number of rows to return (adds LIMIT if not in query)
        model: Claude model to use
        
    Returns:
        Dictionary with SQL, results (if executed), and metadata
    """
    # Set default data dictionary path if not provided
    if not data_dictionary_path:
        data_dictionary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "Data Dictionary", "mts.csv")
    
    print(f"Using model: {model}")
    print(f"Converting: '{question}' to SQL using Claude...")
    
    try:
        # Generate SQL using Claude
        claude_result = natural_language_to_sql_claude(
            query=question,
            data_dictionary_path=data_dictionary_path,
            model=model
        )
        
        # Extract the SQL and ensure it's a valid string
        sql = claude_result.get("sql", "")
        if not isinstance(sql, str):
            sql = str(sql) if sql is not None else "SELECT 'Non-string SQL' AS message"
            claude_result["sql"] = sql
        
        if not sql.strip():
            sql = "SELECT 'Empty SQL query' AS message"
            claude_result["sql"] = sql
            
        print("\nGenerated SQL:")
        print(sql)
        
        # Add LIMIT if not present and execute flag is True
        if execute and limit_rows > 0 and "LIMIT" not in sql.upper():
            # Safely add LIMIT clause
            sql = sql.strip()
            if sql.endswith(';'):
                sql = sql[:-1]
            sql = f"{sql} LIMIT {limit_rows}"
            print(f"\nAdded limit clause: LIMIT {limit_rows}")
            claude_result["sql"] = sql  # Update the SQL in the result
        
        # Execute in Snowflake if requested
        if execute:
            print("\nExecuting in Snowflake...")
            try:
                df = execute_query(sql, print_results=True)
                claude_result["results"] = df
                claude_result["success"] = True
                claude_result["row_count"] = len(df)
            except Exception as e:
                print(f"\nError executing SQL in Snowflake: {str(e)}")
                claude_result["error_execution"] = str(e)
                claude_result["success"] = False
        else:
            claude_result["success"] = True  # No execution requested, so no execution error
        
        return claude_result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = str(e).replace("'", "''")
        return {
            "sql": f"SELECT 'Error in SQL generation: {error_msg}' AS error_message",
            "model": model,
            "error": str(e),
            "success": False,
            "query": question,
            "data_dictionary": data_dictionary_path,
            "timestamp": pd.Timestamp.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


# Command line interface for testing
if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert natural language to SQL with Claude and run in Snowflake")
    parser.add_argument("question", help="Natural language question to convert to SQL")
    parser.add_argument("--no-execute", action="store_true", help="Don't execute the SQL, just show it")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Claude model to use")
    parser.add_argument("--limit", type=int, default=10, help="Row limit for SQL results")
    args = parser.parse_args()
    
    result = nlq_to_snowflake_claude(
        question=args.question,
        execute=not args.no_execute,
        limit_rows=args.limit,
        model=args.model
    )
    
    if "results" in result and isinstance(result["results"], pd.DataFrame):
        print("\nResults:")
        print(result["results"])
