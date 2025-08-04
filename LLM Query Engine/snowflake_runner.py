"""
Snowflake Query Runner - Execute SQL queries against Snowflake
"""
import os
import pandas as pd
import snowflake.connector
import logging
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Configure logging to suppress Snowflake connector INFO messages
logging.getLogger('snowflake.connector').setLevel(logging.WARNING)
logging.getLogger('snowflake.connector.connection').setLevel(logging.WARNING)
logging.getLogger('snowflake.connector.cursor').setLevel(logging.WARNING)

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
            role = snowflake_params.get('role', '')         # Optional - role for the connection
            
            # Client-specific credentials and role information prints removed
            
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
                role = role or os.environ.get('SNOWFLAKE_ROLE', '')           # Optional
                
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
        role = os.environ.get('SNOWFLAKE_ROLE', '')         # Optional
    
    # Connect to Snowflake
    conn_params = {
        "user": user,
        "password": password,
        "account": account,
        "warehouse": warehouse,
        "database": database if database else None,
        "schema": schema if schema else None
    }
    
    # Add role if available
    if role:
        conn_params["role"] = role
        print(f"[Snowflake Runner] Connecting with role: {role}")
        
    conn = snowflake.connector.connect(**conn_params)
    
    try:
        # Execute query and fetch results
        cursor = conn.cursor()
        
        # Verify and print current role
        try:
            cursor.execute("SELECT CURRENT_ROLE()")
            current_role = cursor.fetchone()[0]
            print(f"[Snowflake Runner] Current active role in session: {current_role}")
        except Exception as e:
            print(f"[Snowflake Runner] Could not verify current role: {str(e)}")

        
        # Debug print: Show the SQL to be executed and its repr
        print("[Snowflake Runner] About to execute SQL:")
        print(query)
        
        # Process escape sequences in the SQL query
        # This handles cases where the query contains literal \n instead of actual newlines
        if '\\n' in query:
            processed_query = query.encode().decode('unicode_escape')
            print("[Snowflake Runner] Processed SQL with escape sequences")
        else:
            processed_query = query
        
        # First ensure warehouse is active before executing query
        if warehouse:
            try:
                print(f"[Snowflake Runner] Setting active warehouse to: {warehouse}")
                cursor.execute(f"USE WAREHOUSE {warehouse}")
            except Exception as e:
                print(f"[Snowflake Runner] Error setting warehouse: {str(e)}")
        else:
            print("[Snowflake Runner] WARNING: No warehouse specified, query may fail")
            
        # Execute the query using the processed query with proper newlines
        cursor.execute(processed_query)
        
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
