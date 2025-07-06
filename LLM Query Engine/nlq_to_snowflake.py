"""
Natural Language to Snowflake Query Runner

This script combines the LLM query generator with Snowflake execution:
1. Takes a natural language query
2. Uses OpenAI to convert it to SQL
3. Executes the SQL against your Snowflake account
4. Returns the results
"""
import os
import sys
import pandas as pd
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import our local modules
from llm_query_generator import natural_language_to_sql
from snowflake_runner import execute_query
from token_logger import TokenLogger

# Load environment variables
load_dotenv()


def nlq_to_snowflake(question: str, 
                     data_dictionary_path: Optional[str] = None,
                     execute: bool = True,
                     limit_rows: int = 100) -> Dict[str, Any]:
    """
    End-to-end pipeline to convert natural language to SQL and execute in Snowflake
    
    Args:
        question: Natural language question to convert to SQL
        data_dictionary_path: Path to data dictionary CSV/Excel file
        execute: Whether to execute the SQL in Snowflake (True) or just return SQL (False)
        limit_rows: Maximum number of rows to return (adds LIMIT if not in query)
        
    Returns:
        Dictionary with SQL, results (if executed), and metadata
    """
    # Set default data dictionary path if not provided
    if not data_dictionary_path:
        data_dictionary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "Data Dictionary", "mts.csv")
    
    # Get OpenAI model from environment or use default
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    
    print(f"Using model: {model}")
    print(f"Converting: '{question}' to SQL...")
    
    try:
        # Generate SQL using OpenAI
        result = natural_language_to_sql(question, data_dictionary_path, model=model, limit_rows=limit_rows)
        
        # Extract the SQL and clean it
        sql = result["sql"]

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

        # Always store cleaned SQL and token usage in result dict before execution
        result["sql"] = sql
        # Defensive: propagate token usage if present
        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            if k not in result:
                result[k] = 0

        # Add LIMIT if not present and execute flag is True
        if execute and limit_rows > 0 and "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip('; \n')} LIMIT {limit_rows}"
            print(f"\nAdded limit clause: LIMIT {limit_rows}")
            result["sql_with_limit"] = sql

        # Log token usage for both executed and non-executed queries
        query_executed_value = None  # Default for non-executed queries
        
        # Execute in Snowflake if requested
        if execute:
            print("\nExecuting in Snowflake...")
            query_executed_successfully = False
            try:
                df = execute_query(sql, print_results=True)
                result["results"] = df
                result["success"] = True
                result["row_count"] = len(df)
                query_executed_successfully = True
                query_executed_value = 1  # Success
            except Exception as e:
                print(f"\nError executing SQL in Snowflake: {str(e)}")
                result["success"] = False
                result["error_execution"] = str(e)
                query_executed_value = 0  # Failed execution
        
        # Always log token usage with appropriate query_executed status
        try:
            logger = TokenLogger()
            logger.log_usage(
                model=model,
                query=question,
                usage={
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "completion_tokens": result.get("completion_tokens", 0),
                    "total_tokens": result.get("total_tokens", 0)
                },
                prompt=question,
                sql_query=sql,
                query_executed=query_executed_value  # 1=success, 0=failed, None=not executed
            )
        except Exception as log_err:
            print(f"Error logging token usage: {str(log_err)}")

        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python nlq_to_snowflake.py \"Your natural language question\"")
        print("Example: python nlq_to_snowflake.py \"Show me the top 5 stores by total sales last month\"")
        sys.exit(1)
    
    # Get question from command line
    question = sys.argv[1]
    
    # Check if data dictionary path is provided as second argument
    data_dictionary_path = None
    if len(sys.argv) > 2:
        data_dictionary_path = sys.argv[2]
    
    # Run the pipeline
    result = nlq_to_snowflake(question, data_dictionary_path)
    
    # Display token usage for transparency
    if "prompt_tokens" in result:
        print("\nToken Usage:")
        print(f"  Input tokens: {result['prompt_tokens']}")
        print(f"  Output tokens: {result['completion_tokens']}")
        print(f"  Total tokens: {result['total_tokens']}")
