"""
Snowflake Query Runner - Execute SQL queries against Snowflake
"""
import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

def execute_query(query: str, print_results: bool = True) -> pd.DataFrame:
    """
    Execute a SQL query against Snowflake and return results as DataFrame
    
    Args:
        query: SQL query to execute
        print_results: Whether to print the results to console
        
    Returns:
        DataFrame with query results
    """
    # Get connection parameters from environment
    user = os.environ.get('SNOWFLAKE_USER')
    password = os.environ.get('SNOWFLAKE_PASSWORD')
    account = os.environ.get('SNOWFLAKE_ACCOUNT')
    warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE')
    database = os.environ.get('SNOWFLAKE_DATABASE', '')  # Optional
    schema = os.environ.get('SNOWFLAKE_SCHEMA', '')      # Optional
    
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database if database else None,
        schema=schema if schema else None
    )
    
    try:
        # Execute query and fetch results
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Get column names
        column_names = [col[0] for col in cursor.description] if cursor.description else []
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=column_names)
        
        # Print results if requested
        if print_results:
            if len(df) > 0:
                print(f"Results ({len(df)} rows):")
                print(df.to_string(index=False))
            else:
                print("Query executed successfully. No results returned.")
        
        return df
    
    finally:
        # Close connection
        conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python snowflake_runner.py \"SQL_QUERY\"")
        print("Example: python snowflake_runner.py \"SELECT * FROM my_table LIMIT 10\"")
        sys.exit(1)
    
    # Get query from command line argument
    query = sys.argv[1]
    
    try:
        print(f"Executing query: {query}")
        execute_query(query)
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        sys.exit(1)
