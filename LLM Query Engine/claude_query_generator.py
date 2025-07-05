"""Claude LLM Query Generator - A module to convert natural language to SQL using Anthropic Claude
"""
import os
import pandas as pd
import datetime
import dotenv
from typing import Dict, List, Any, Optional
from token_logger import TokenLogger

# Load environment variables from .env file
dotenv.load_dotenv()

# Reuse these functions from the OpenAI implementation
from llm_query_generator import load_data_dictionary, format_data_dictionary

# Check if anthropic module is installed, install if not
try:
    import anthropic
except ImportError:
    print("Anthropic module not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "anthropic"])
    import anthropic


def generate_sql_prompt(tables: List[Dict[str, Any]], query: str) -> str:
    """
    Generate system prompt for Claude with table schema information
    
    Args:
        tables: List of formatted table dictionaries
        query: Natural language query
        
    Returns:
        Formatted system prompt for Claude
    """
    # Format tables info into string
    tables_context = ""
    for table in tables:
        # Include schema in table name if available
        if table['schema']:
            tables_context += f"Table: {table['name']} (Schema: {table['schema']})\n"
        else:
            tables_context += f"Table: {table['name']}\n"
            
        if table['description']:
            tables_context += f"Description: {table['description']}\n"
            
        tables_context += "Columns:\n"
        
        for col in table['columns']:
            pk_indicator = " (PRIMARY KEY)" if col.get('is_primary_key') else ""
            fk_info = ""
            if col.get('is_foreign_key'):
                fk_info = f" (FOREIGN KEY to {col.get('foreign_key_table')}.{col.get('foreign_key_column')})"
            
            tables_context += f"  - {col['name']} ({col['type']}){pk_indicator}{fk_info}: {col['description']}"
            
            # Only add business name if it's different from column name
            if col['business_name'] != col['name']:
                tables_context += f" [Business Name: {col['business_name']}]"
                
            tables_context += "\n"
        
        tables_context += "\n"
    
    # Create prompt template for Claude
    prompt = f"""You are an expert SQL query generator for Snowflake database.

Your task is to convert natural language questions into valid SQL queries that can run on Snowflake.
Use the following data dictionary to understand the database schema:

{tables_context}

When generating SQL:
1. Use proper Snowflake SQL syntax with fully qualified table names including schema (e.g., SCHEMA.TABLE_NAME)
2. Include appropriate JOINs based on the data relationships or column name similarities
3. Format the SQL code clearly with proper indentation and aliases
4. Only use tables and columns that exist in the provided schema
5. Add helpful SQL comments to explain complex parts of the query
6. Return ONLY the SQL code without any other text or explanations
7. Limit results to 100 rows unless specified otherwise

Generate a SQL query for: {query}
"""
    return prompt


def generate_sql_query_claude(api_key: str, prompt: str, model: str = "claude-3-5-sonnet-20241022", 
                         query_text: str = "", log_tokens: bool = True) -> Dict[str, Any]:
    """
    Generate SQL query using Anthropic Claude API
    
    Args:
        api_key: Anthropic API key
        prompt: System prompt with schema and query
        model: Claude model to use (default: claude-3-5-sonnet)
        query_text: Original natural language query (for logging)
        log_tokens: Whether to log token usage
        
    Returns:
        Dictionary with SQL query and token usage info
    """
    client = anthropic.Anthropic(api_key=api_key)
    logger = TokenLogger() if log_tokens else None
    
    try:
        
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        
        
        # (DEBUG prints removed)
        # The following blocks are intentionally left empty after debug removal.
        pass

        # Extract SQL query from response - new simplified approach
        # Start with a valid default SQL that will work as a fallback
        sql_query = "SELECT 'No SQL query was successfully extracted' AS error_message"
        
        # Only proceed if we have a valid response with content
        if response and hasattr(response, 'content') and response.content:
            try:
                # Get the text content directly from Claude's response (similar to OpenAI pattern)
                full_text = ""
                
                # Safety check for text attribute
                if hasattr(response.content[0], 'text'):
                    if response.content[0].text is not None:
                        full_text = response.content[0].text
                        if not isinstance(full_text, str):
                            full_text = str(full_text)
                        full_text = full_text.strip()
                
                # Exit early if we have no text to parse
                if not full_text:
                    print("DEBUG: Empty text content from Claude response")
                    return {
                        "sql": sql_query,
                        "model": model,
                        "error": "Empty response from Claude",
                        "prompt_tokens": response.usage.input_tokens if hasattr(response, 'usage') else 0,
                        "completion_tokens": response.usage.output_tokens if hasattr(response, 'usage') else 0,
                        "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, 'usage') else 0
                    }
                
                # Extract SQL from code blocks if present
                if "```" in full_text:
                    import re
                    # First try to find SQL-specific code blocks
                    sql_blocks = re.findall(r'```(?:sql|SQL)?([\s\S]*?)```', full_text)
                    
                    if sql_blocks and sql_blocks[0]:
                        # Successfully found an SQL code block
                        extracted_sql = sql_blocks[0].strip()
                        if extracted_sql:  # Only update if we got something meaningful
                            sql_query = extracted_sql
                    else:
                        # If no SQL-specific blocks, try any code blocks
                        code_blocks = re.findall(r'```([\s\S]*?)```', full_text)
                        if code_blocks and code_blocks[0]:
                            extracted_code = code_blocks[0].strip()
                            if extracted_code:  # Only update if we got something meaningful
                                sql_query = extracted_code
                
                # If we still don't have SQL but the response contains SELECT keywords
                # Use the entire response as the SQL
                elif "SELECT" in full_text.upper():
                    sql_query = full_text
                
                # Final check to ensure sql_query is a string
                if sql_query is None:
                    sql_query = "SELECT 'SQL extraction resulted in None' AS error_message"
                elif not isinstance(sql_query, str):
                    sql_query = str(sql_query)
                
                # Ensure SQL is not empty
                if not sql_query.strip():
                    sql_query = "SELECT 'Empty SQL query extracted' AS error_message"
                    
            except Exception as e:
                print(f"DEBUG: Error extracting SQL from Claude response: {str(e)}")
                sql_query = f"SELECT 'Error extracting SQL: {str(e).replace("'", "''")}' AS error_message"
        else:
            print("DEBUG: Invalid response structure from Claude")
            sql_query = "SELECT 'Invalid response structure from Claude' AS error_message"
    
        # Ensure we return a valid string for SQL even if extraction failed (OpenAI pattern)
        if sql_query is None:
            sql_query = "SELECT 'Error: SQL query extraction resulted in None' AS error_message"
        elif not isinstance(sql_query, str):
            # Try to convert to string if possible
            try:
                sql_query = str(sql_query)
            except:
                sql_query = "SELECT 'Error: Could not convert SQL query to string' AS error_message"
        elif not sql_query.strip():  # Empty string or whitespace
            sql_query = "SELECT 'Error: Empty SQL query extracted' AS error_message"
        
        # Get token usage
        usage_data = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        # NOTE: Token usage is now logged only in nlq_to_snowflake_claude.py after execution
        # to avoid duplicate entries in the token_usage.csv file
            
        return {
            "sql": sql_query,
            "model": model,
            "prompt_tokens": usage_data["prompt_tokens"],
            "completion_tokens": usage_data["completion_tokens"],
            "total_tokens": usage_data["total_tokens"]
        }
        
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        
        error_msg = f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}\nTRACEBACK: {tb_str.replace("'", "''").replace(chr(10), ' ')}' AS error_message"
        return {
            "sql": error_msg,
            "model": model,
            "error": str(e),
            "traceback": tb_str,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


def natural_language_to_sql_claude(query: str, data_dictionary_path: str = None, api_key: Optional[str] = None, 
                               model: str = "claude-3-5-sonnet-20241022", log_tokens: bool = True) -> Dict[str, Any]:
    """
    End-to-end function to convert natural language to SQL using Claude
    
    Args:
        query: Natural language query
        data_dictionary_path: Path to data dictionary Excel/CSV
        api_key: Anthropic API key (will use environment variable if not provided)
        model: Claude model to use
        log_tokens: Whether to log token usage
        
    Returns:
        Dictionary with SQL query, token usage and other metadata
    """
    
    
    # Set default API key from environment if not provided
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    try:
        # Track execution time
        start_time = datetime.datetime.now()
        
        # Load data dictionary
        df = load_data_dictionary(data_dictionary_path)
        
        # Format data dictionary for prompt
        tables = format_data_dictionary(df)
        
        # Generate prompt
        prompt = generate_sql_prompt(tables, query)
        
        # Generate SQL
        result = generate_sql_query_claude(
            api_key=api_key,
            prompt=prompt,
            model=model,
            query_text=query,
            log_tokens=log_tokens
        )
        
        
        
        # Ensure 'sql' key always exists and contains a valid string
        if 'sql' not in result or result['sql'] is None or not isinstance(result['sql'], str):
            
            result['sql'] = "SELECT 'No valid SQL query was generated' AS message"
        
        # Add additional metadata
        result.update({
            "execution_time_ms": (datetime.datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "data_dictionary": data_dictionary_path
        })
        
        
        return result
    
    except Exception as e:
        
        import traceback
        traceback.print_exc()
        
        return {
            "sql": f"SELECT 'Error in SQL generation: {str(e).replace("'", "''")}' AS error_message",
            "model": model,
            "error": str(e),
            "query": query,
            "data_dictionary": data_dictionary_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if we have a query from command line
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "Show me the top 5 stores with highest sales"
        
    # Check if we have a data dictionary path
    data_dict_path = None
    if len(sys.argv) > 2:
        data_dict_path = sys.argv[2]
    else:
        # Use default data dictionary path
        data_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "Data Dictionary", "mts.csv")
    
    # Set the model name
    model = "claude-3-5-sonnet-20241022"
    
    try:
        result = natural_language_to_sql_claude(query, data_dict_path, model=model)
        
        # Match output format with llm_query_generator.py
        print(f"Using model provider: claude")
        print(f"Using specific model: {result['model']}")
        
        print(f"\nQuery: {query}\n")
        print(f"SQL:\n```sql\n{result['sql']}\n```\n")
        
        print("Token Usage:")
        print(f"  Prompt tokens: {result.get('prompt_tokens', 0)}")
        print(f"  Completion tokens: {result.get('completion_tokens', 0)}")
        print(f"  Total tokens: {result.get('total_tokens', 0)}")
        print(f"\nExecution time: {result.get('execution_time_ms', 0):.2f} ms" if result.get('execution_time_ms') is not None else "\nExecution time: Not available")
    except Exception as e:
        print(f"Error: {str(e)}")

