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
import json
import pandas as pd
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import our local modules
from claude_query_generator import natural_language_to_sql_claude
from snowflake_runner import execute_query as run_snowflake_query
from token_logger import TokenLogger

# Load environment variables
load_dotenv()


def nlq_to_snowflake_claude(prompt: str, 
                     data_dictionary_path: Optional[str] = None,
                     execute_query: bool = True,
                     limit_rows: int = 100,
                     model: str = "claude-3-5-sonnet-20241022",
                     include_charts: bool = False,
                     claude_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    End-to-end pipeline to convert natural language to SQL using Claude and execute in Snowflake
    
    Args:
        prompt: Natural language question to convert to SQL
        data_dictionary_path: Path to data dictionary CSV/Excel file
        execute_query: Whether to execute the SQL in Snowflake (True) or just return SQL (False)
        limit_rows: Maximum number of rows to return (adds LIMIT if not in query)
        model: Claude model to use
        include_charts: Whether to include chart recommendations from LLM
        
    Returns:
        Dictionary with SQL, results (if executed), and metadata
    """
    # Ensure dictionary path is provided
    if not data_dictionary_path:
        # Raise an error instead of silently falling back to MTS dictionary
        raise ValueError("No dictionary path provided. Cannot proceed with Claude.")
    
    print(f"Using model: {model}")
    print(f"Converting: '{prompt}' to SQL using Claude...")
    
    try:
        # Use pre-generated result if provided, otherwise generate SQL using Claude
        if not claude_result:
            claude_result = natural_language_to_sql_claude(
                query=prompt,
                data_dictionary_path=data_dictionary_path,
                model=model,
                limit_rows=limit_rows,
                include_charts=include_charts
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
        
        # Validate SQL structure and fix common issues
        def validate_and_fix_sql(sql_query):
            # Check for missing spaces between alias and keywords
            import re
            
            # Use a single comprehensive fix function
            def fix_sql_formatting(sql):
                # Fix common SQL formatting issues
                # 1. Add proper spacing around SQL keywords
                # 2. Ensure proper spacing after aliases
                # 3. Remove unwanted line breaks in the middle of statements
                
                # First, normalize whitespace in the SQL
                sql = re.sub(r'\s+', ' ', sql).strip()
                
                # Fix "AS alias" followed immediately by keywords
                keywords = ["FROM", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "JOIN"]
                
                # Pattern for AS followed by an alias then immediately a keyword
                for keyword in keywords:
                    # Pattern: AS alias_nameKEYWORD
                    pattern = r'(AS\s+[A-Za-z0-9_]+)(' + keyword + r')\b'
                    
                    # Find all occurrences in uppercase version for case-insensitive matching
                    sql_upper = sql.upper()
                    for match in re.finditer(pattern, sql_upper):
                        # Get original text as it appears in SQL
                        start, end = match.span()
                        original = sql[start:end]
                        
                        # Extract parts
                        alias_part = original[:-len(keyword)]
                        
                        # Add space between alias and keyword
                        replacement = f"{alias_part} {keyword}"
                        sql = sql[:start] + replacement + sql[end:]
                        print(f"Fixed missing space after alias: {original} → {replacement}")
                        
                        # Need to update SQL_upper after changing SQL
                        sql_upper = sql.upper()
                        
                return sql
                
            # Apply the comprehensive fix
            fixed_sql = fix_sql_formatting(sql_query)
            
            # Fix missing spaces before simple and compound keywords
            # Handle both simple keywords like 'FROM' and compound keywords like 'ORDER BY'
            # First pass - fix compound keywords where the space within the keyword might be missing
            compound_keywords = {"GROUP BY": "GROUPBY", "ORDER BY": "ORDERBY"}
            for correct, incorrect in compound_keywords.items():
                # First fix cases where the compound keyword itself has no space
                fixed_sql = re.sub(r'\b' + incorrect + r'\b', correct, fixed_sql, flags=re.IGNORECASE)
                
                # Then fix cases where a column is followed by the keyword without space
                pattern = r'([A-Za-z0-9_]+)(' + incorrect + r')\b'
                
                # Find all instances in uppercase version
                for match in re.finditer(pattern, fixed_sql.upper()):
                    start, end = match.span()
                    original = fixed_sql[start:end]
                    
                    # Extract the parts
                    column = original[:-len(incorrect)]
                    replacement = f"{column} {correct}"
                    fixed_sql = fixed_sql[:start] + replacement + fixed_sql[end:]
                    print(f"Fixed compound keyword spacing: {original} → {replacement}")
            
            # Second pass - fix spacing between identifiers and regular keywords
            keywords = ["FROM", "WHERE", "GROUP BY", "ORDER BY", "LIMIT", "HAVING", "JOIN"]
            for keyword in keywords:
                # Look for pattern where a word is immediately followed by the keyword without space
                pattern = r'([A-Za-z0-9_]+)(' + keyword.replace(" ", "\\s*") + r')\b'
                
                # Find all instances in uppercase version
                for match in re.finditer(pattern, fixed_sql.upper()):
                    # Extract the original text as it appears in the query
                    start, end = match.span()
                    original = fixed_sql[start:end]
                    
                    # Check if there's already proper spacing
                    if not re.search(r'\s+' + keyword.replace(" ", "\\s+") + r'\b', original, re.IGNORECASE):
                        word = original[:-(len(keyword) + original.upper().find(keyword.upper()) - original.upper().find(keyword.split()[0].upper()))]
                        replacement = f"{word} {keyword}"
                        fixed_sql = fixed_sql[:start] + replacement + fixed_sql[end:]
                        print(f"Fixed missing space: {original} → {replacement}")
                        
                    # We need to re-search after each replacement since string indices change
            
            # Check for incomplete CTE structures
            if "WITH" in fixed_sql.upper() and "SELECT" in fixed_sql.upper():
                # Check if the SQL might be missing a SELECT after a CTE definition
                # This handles cases with missing SELECT * FROM cte_name
                cte_parts = fixed_sql.split(")", 1)
                if len(cte_parts) > 1 and "SELECT" not in cte_parts[1].upper() and "FROM" not in cte_parts[1].upper():
                    # Add the missing SELECT statement
                    cte_name = fixed_sql.upper().split("WITH ", 1)[1].split(" AS", 1)[0].strip()
                    fixed_sql = f"{cte_parts[0]}) SELECT * FROM {cte_name}"
                    print(f"\nFixed incomplete CTE by adding 'SELECT * FROM {cte_name}'")
                    
            # Check for incomplete SQL statements (ending with EOF)
            if fixed_sql.strip().endswith(")"):
                last_cte = fixed_sql.upper().split("WITH ")[-1].split(" AS")[0].strip()
                fixed_sql = f"{fixed_sql.strip()} SELECT * FROM {last_cte}"
                print(f"\nFixed incomplete SQL by adding 'SELECT * FROM {last_cte}'")
                    
            return fixed_sql
            
        # Validate and fix the SQL structure
        sql = validate_and_fix_sql(sql)
        
        # Add LIMIT if not present and execute flag is True
        if execute_query and limit_rows > 0 and "LIMIT" not in sql.upper():
            # Safely add LIMIT clause
            sql = sql.strip()
            if sql.endswith(';'):
                sql = sql[:-1]
            sql = f"{sql} LIMIT {limit_rows}"
            print(f"\nAdded limit clause: LIMIT {limit_rows}")
            claude_result["sql"] = sql  # Update the SQL in the result
        
        # Log token usage for both executed and non-executed queries
        query_executed_value = None  # Default for non-executed queries

        # Execute in Snowflake if requested
        if execute_query:
            print("\nExecuting in Snowflake...")
            query_executed_successfully = False
            try:
                df = run_snowflake_query(sql, print_results=True)
                claude_result["results"] = df
                claude_result["success"] = True
                claude_result["row_count"] = len(df)
                query_executed_successfully = True
                query_executed_value = 1  # Success
            except Exception as e:
                print(f"\nError executing SQL in Snowflake: {str(e)}")
                claude_result["error_execution"] = str(e)
                claude_result["error_message"] = str(e)  # Add error_message for API response
                claude_result["success"] = False
                query_executed_value = 0  # Failed execution
        else:
            print("Query not executed (execute_query=False)")
            claude_result["success"] = True  # No execution requested, so no execution error
            
        # Always log token usage with appropriate query_executed status
        try:
            logger = TokenLogger()
            logger.log_usage(
                model=model,
                query=prompt,
                usage={
                    "prompt_tokens": claude_result.get("prompt_tokens", 0),
                    "completion_tokens": claude_result.get("completion_tokens", 0),
                    "total_tokens": claude_result.get("total_tokens", 0)
                },
                prompt=prompt,
                sql_query=sql,
                query_executed=query_executed_value  # 1=success, 0=failed, None=not executed
            )
        except Exception as log_err:
            print(f"Error logging token usage: {str(log_err)}")
        
        # Handle chart recommendations based on both include_charts and execute_query flags
        # Suppress chart recommendations when execute_query=false, even if include_charts=true
        if include_charts and execute_query:
            # Only include chart recommendations when both flags are true
            if "chart_recommendations" not in claude_result:
                claude_result["chart_recommendations"] = []
            if "chart_error" not in claude_result:
                claude_result["chart_error"] = None
            
            # If we have no chart recommendations and query failed, add an explanatory error
            if not claude_result.get("chart_recommendations") and not claude_result.get("success"):
                claude_result["chart_error"] = "Query execution failed, but any provided chart recommendations are preserved"
        else:
            # Set chart fields to null when charts aren't requested OR when execute_query is false
            claude_result["chart_recommendations"] = None
            claude_result["chart_error"] = None
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
            "query": prompt,
            "data_dictionary": data_dictionary_path,
            "timestamp": pd.Timestamp.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "chart_recommendations": [],
            "chart_error": "Cannot generate charts: query generation failed" if include_charts else None
        }


# Command line interface for testing
if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert natural language to SQL with Claude and run in Snowflake")
    parser.add_argument("prompt", help="Natural language question to convert to SQL")
    parser.add_argument("--no-execute", action="store_true", help="Don't execute the SQL, just show it")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Claude model to use")
    parser.add_argument("--limit", type=int, default=10, help="Row limit for SQL results")
    parser.add_argument("--include-charts", action="store_true", help="Include chart recommendations in the output")
    args = parser.parse_args()
    
    result = nlq_to_snowflake_claude(
        prompt=args.prompt,
        execute_query=not args.no_execute,
        limit_rows=args.limit,
        model=args.model,
        include_charts=args.include_charts
    )
    
    if "results" in result and isinstance(result["results"], pd.DataFrame):
        print("\nResults:")
        print(result["results"])
