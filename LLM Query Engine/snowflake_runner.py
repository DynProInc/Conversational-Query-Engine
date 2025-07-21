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

def execute_query(query: str, print_results: bool = True, client_id: str = None) -> pd.DataFrame:
    """
    Execute a SQL query against Snowflake and return results as DataFrame
    
    Args:
        query: SQL query to execute
        print_results: Whether to print the results to console
        
    Returns:
        DataFrame with query results
    """
    # Get connection parameters from client-specific environment if client_id is provided
    # Otherwise fall back to global environment variables
    if client_id:
        try:
            from config.client_manager import client_manager
            # Get client-specific Snowflake configuration using the proper method
            snowflake_params = client_manager.get_snowflake_connection_params(client_id)
            
            # Extract credentials from client connection parameters
            user = snowflake_params.get('user')
            password = snowflake_params.get('password')
            account = snowflake_params.get('account')
            warehouse = snowflake_params.get('warehouse')
            database = snowflake_params.get('database', '')  # Optional
            schema = snowflake_params.get('schema', '')      # Optional
            
            print(f"[Snowflake Runner] Using client-specific credentials for client: {client_id}")
            
            # Verify essential credentials are available
            if not all([user, password, account, warehouse]):
                print(f"[Snowflake Runner] WARNING: Missing required Snowflake credentials for client {client_id}, falling back to global credentials")
                # Fall back to global environment variables
                user = user or os.environ.get('SNOWFLAKE_USER')
                password = password or os.environ.get('SNOWFLAKE_PASSWORD')
                account = account or os.environ.get('SNOWFLAKE_ACCOUNT')
                warehouse = warehouse or os.environ.get('SNOWFLAKE_WAREHOUSE')
                database = database or os.environ.get('SNOWFLAKE_DATABASE', '')  # Optional
                schema = schema or os.environ.get('SNOWFLAKE_SCHEMA', '')      # Optional
                
        except Exception as e:
            print(f"[Snowflake Runner] Error retrieving client-specific credentials: {str(e)}. Using global credentials.")
            # Fall back to global environment variables
            user = os.environ.get('SNOWFLAKE_USER')
            password = os.environ.get('SNOWFLAKE_PASSWORD')
            account = os.environ.get('SNOWFLAKE_ACCOUNT')
            warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE')
            database = os.environ.get('SNOWFLAKE_DATABASE', '')  # Optional
            schema = os.environ.get('SNOWFLAKE_SCHEMA', '')      # Optional
    else:
        # Get connection parameters from global environment
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
        
        # Debug print: Show the SQL to be executed and its repr
        print("[Snowflake Runner] About to execute SQL:")
        print(query)
        print("[Snowflake Runner] repr of SQL:")
        print(repr(query))
        
        # First ensure warehouse is active before executing query
        if warehouse:
            try:
                print(f"[Snowflake Runner] Setting active warehouse to: {warehouse}")
                cursor.execute(f"USE WAREHOUSE {warehouse}")
            except Exception as e:
                print(f"[Snowflake Runner] Error setting warehouse: {str(e)}")
        else:
            print("[Snowflake Runner] WARNING: No warehouse specified, query may fail")
            
        # Execute the actual query
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
